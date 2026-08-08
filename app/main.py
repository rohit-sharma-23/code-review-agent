"""
Main FastAPI application entry point.

Exposes health check endpoints and the GitHub webhook receiver.
"""

import hashlib
import hmac
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import asyncpg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse


from app.agent import review_pr
from app.db import db_manager, get_db_pool

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

app = FastAPI(
    title="AI Code Review Agent API",
    description="Automated AI Code Review System with MCP tools and RAG integration.",
    version="1.0.0",
)


# Signature Verification Helper

def verify_github_signature(payload_bytes: bytes, signature_header: Optional[str], secret_key: Optional[str]) -> bool:
    """
    Verify GitHub X-Hub-Signature-256 using HMAC-SHA256 and constant-time comparison.
    """
    if not secret_key:
        logger.warning("GITHUB_WEBHOOK_SECRET is not configured; skipping signature check in dev mode.")
        return True

    if not signature_header or not signature_header.startswith("sha256="):
        return False

    received_hash = signature_header[len("sha256="):]
    expected_hash = hmac.new(
        secret_key.encode("utf-8"),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_hash, received_hash)


# Health Check & Root Endpoints

@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root status endpoint."""
    return {
        "service": "AI Code Review Agent API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint verifying database pool connectivity.
    """
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            
        return {
            "status": "healthy",
            "database": "connected",
            "db_query": val,
            "environment": os.getenv("LLM_PROVIDER", "gemini")
        }
    except Exception as e:
        logger.error(f"Health check database connection failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )


async def resolve_or_create_repo_id(repo_full_name: str, default_branch: str = "main") -> uuid.UUID:
    """Helper to resolve or register repository ID in repos table."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        pool = await get_db_pool()
        conn = await pool.acquire()
        try:
            row = await conn.fetchrow("SELECT id FROM repos WHERE github_full_name = $1", repo_full_name)
            if row:
                return row["id"]
            repo_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO repos (id, github_full_name, default_branch) VALUES ($1, $2, $3)",
                repo_id,
                repo_full_name,
                default_branch
            )
            return repo_id
        finally:
            await pool.release(conn)

    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow("SELECT id FROM repos WHERE github_full_name = $1", repo_full_name)
        if row:
            return row["id"]

        repo_id = uuid.uuid4()
        await conn.execute(
            "INSERT INTO repos (id, github_full_name, default_branch) VALUES ($1, $2, $3)",
            repo_id,
            repo_full_name,
            default_branch
        )
        logger.info(f"Registered new repo '{repo_full_name}' with ID: {repo_id}")
        return repo_id
    finally:
        await conn.close()




# Webhook Receiver Endpoint

@app.post("/webhook")
async def handle_github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None, alias="X-Hub-Signature-256"),
    x_github_delivery: Optional[str] = Header(None, alias="X-GitHub-Delivery"),
    x_github_event: Optional[str] = Header(None, alias="X-GitHub-Event"),
):
    """
    GitHub Webhook Endpoint.

    Processes 'pull_request' events (actions: 'opened', 'synchronize') and triggers
    asynchronous background AI code reviews.
    """
    delivery_id = x_github_delivery or "unknown-delivery"
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET")

    # 1. Read raw request payload bytes
    payload_bytes = await request.body()

    # 2. HMAC-SHA256 Signature Verification
    if not verify_github_signature(payload_bytes, x_hub_signature_256, webhook_secret):
        logger.warning(
            f"[Delivery {delivery_id}] REJECTED: Invalid or missing X-Hub-Signature-256 header."
        )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "status": "error",
                "error": "Invalid or missing webhook signature.",
                "delivery_id": delivery_id
            }
        )

    # 3. Parse JSON Payload
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        logger.error(f"[Delivery {delivery_id}] Malformed JSON payload: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": f"Malformed JSON body: {str(e)}",
                "delivery_id": delivery_id
            }
        )

    # 4. Filter GitHub Event Type
    event_type = x_github_event or "pull_request"
    if event_type != "pull_request":
        logger.info(f"[Delivery {delivery_id}] Ignored event type: '{event_type}'")
        return {
            "status": "ignored",
            "reason": f"Event '{event_type}' is not supported.",
            "delivery_id": delivery_id
        }

    action = payload.get("action")
    if action not in ["opened", "synchronize"]:
        logger.info(f"[Delivery {delivery_id}] Ignored PR action: '{action}'")
        return {
            "status": "ignored",
            "reason": f"PR action '{action}' is not configured for auto-review.",
            "delivery_id": delivery_id
        }

    # 5. Extract Details
    repo_full_name = payload.get("repository", {}).get("full_name")
    pr_number = payload.get("number")
    pr_head = payload.get("pull_request", {}).get("head", {})
    commit_sha = pr_head.get("sha")

    if not repo_full_name or not pr_number or not commit_sha:
        logger.error(f"[Delivery {delivery_id}] Missing required PR payload fields.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": "Missing repository, number, or head SHA in PR payload.",
                "delivery_id": delivery_id
            }
        )

    logger.info(
        f"[Delivery {delivery_id}] Received PR #{pr_number} webhook for '{repo_full_name}' "
        f"(action: '{action}', commit: '{commit_sha[:7]}'). Signature verified successfully."
    )

    # 6. Resolve repo_id
    default_branch = payload.get("repository", {}).get("default_branch", "main")
    repo_id = await resolve_or_create_repo_id(repo_full_name, default_branch)

    # 7. Launch Background Review Task
    background_tasks.add_task(review_pr, pr_number, repo_id, commit_sha)
    logger.info(f"[Delivery {delivery_id}] Successfully dispatched background review task for PR #{pr_number}.")

    # 8. Return 200 Accepted Response Immediately
    return {
        "status": "accepted",
        "delivery_id": delivery_id,
        "repository": repo_full_name,
        "pr_number": pr_number,
        "action": action,
        "commit_sha": commit_sha,
        "message": "Webhook signature verified. Background AI review task dispatched successfully."
    }

