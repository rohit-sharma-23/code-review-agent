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
from urllib.parse import parse_qs, unquote

import asyncpg

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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

# CORS Configuration
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if frontend_url and frontend_url not in allowed_origins:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Feedback Request Model
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    was_helpful: bool = Field(..., description="Boolean indicating if the comment was helpful")



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

    if not payload_bytes or not payload_bytes.strip():
        logger.warning(f"[Delivery {delivery_id}] REJECTED: Empty request body.")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error": "Payload body is empty.",
                "delivery_id": delivery_id
            }
        )

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

    # 3. Parse JSON Payload (supports application/json and application/x-www-form-urlencoded)
    try:
        payload_text = payload_bytes.decode("utf-8").strip()
        if payload_text.startswith("payload="):
            parsed_form = parse_qs(payload_text)
            raw_json_str = parsed_form.get("payload", [""])[0]
            payload = json.loads(raw_json_str)
        else:
            payload = json.loads(payload_text)
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

    # Handle GitHub Webhook Ping event
    if event_type == "ping":
        zen = payload.get("zen", "No zen provided")
        logger.info(f"[Delivery {delivery_id}] Handled GitHub Ping event. Zen: '{zen}'")
        return {
            "status": "pong",
            "message": "GitHub Webhook Ping received successfully.",
            "zen": zen,
            "delivery_id": delivery_id
        }

    if event_type != "pull_request":
        logger.info(f"[Delivery {delivery_id}] Ignored event type: '{event_type}'")
        return {
            "status": "ignored",
            "reason": f"Event '{event_type}' is not supported.",
            "delivery_id": delivery_id
        }

    action = payload.get("action")
    if action not in ["opened", "synchronize", "reopened"]:
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


# Dashboard API Endpoints

@app.get("/api/reviews")
async def get_reviews(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
) -> Dict[str, Any]:
    """Return paginated list of code reviews, ordered newest first."""
    try:
        db_url = os.getenv("DATABASE_URL")
        conn = await asyncpg.connect(db_url)
        try:
            offset = (page - 1) * page_size
            total_count = await conn.fetchval("SELECT COUNT(*) FROM reviews")
            rows = await conn.fetch(
                """
                SELECT r.id, r.pr_number, r.commit_sha, r.status, r.created_at,
                       rp.github_full_name as repository,
                       COUNT(c.id) as comment_count
                FROM reviews r
                LEFT JOIN repos rp ON r.repo_id = rp.id
                LEFT JOIN comments c ON r.id = c.review_id
                GROUP BY r.id, rp.github_full_name
                ORDER BY r.created_at DESC
                LIMIT $1 OFFSET $2
                """,
                page_size,
                offset
            )
        finally:
            await conn.close()

        items = []
        for row in rows:
            items.append({
                "id": str(row["id"]),
                "repository": row["repository"] or "unknown",
                "pr_number": row["pr_number"],
                "commit_sha": row["commit_sha"],
                "status": row["status"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "completed_at": row["created_at"].isoformat() if row["created_at"] else None,
                "comment_count": row["comment_count"]
            })

        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total_count or 0
        }
    except Exception as e:
        logger.error(f"Error in GET /api/reviews: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Internal server error fetching reviews."}
        )


@app.get("/api/reviews/{review_id}")
async def get_review_detail(review_id: str) -> Dict[str, Any]:
    """Return detailed review info and all generated comments."""
    try:
        review_uuid = uuid.UUID(review_id)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "error": "Invalid review UUID format."}
        )

    try:
        db_url = os.getenv("DATABASE_URL")
        conn = await asyncpg.connect(db_url)
        try:
            r_row = await conn.fetchrow(
                """
                SELECT r.id, r.pr_number, r.commit_sha, r.status, r.created_at,
                       rp.github_full_name as repository
                FROM reviews r
                LEFT JOIN repos rp ON r.repo_id = rp.id
                WHERE r.id = $1
                """,
                review_uuid
            )

            if not r_row:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"status": "error", "error": f"Review '{review_id}' not found."}
                )

            c_rows = await conn.fetch(
                """
                SELECT id, file_path, line_number, comment_type, body, was_helpful
                FROM comments
                WHERE review_id = $1
                ORDER BY file_path ASC, line_number ASC
                """,
                review_uuid
            )
        finally:
            await conn.close()

        comments = []
        for c in c_rows:
            comments.append({
                "id": str(c["id"]),
                "file_path": c["file_path"],
                "line_number": c["line_number"],
                "comment_type": c["comment_type"],
                "body": c["body"],
                "was_helpful": c["was_helpful"]
            })

        return {
            "id": str(r_row["id"]),
            "repository": r_row["repository"] or "unknown",
            "pr_number": r_row["pr_number"],
            "commit_sha": r_row["commit_sha"],
            "status": r_row["status"],
            "created_at": r_row["created_at"].isoformat() if r_row["created_at"] else None,
            "completed_at": r_row["created_at"].isoformat() if r_row["created_at"] else None,
            "comments": comments
        }
    except Exception as e:
        logger.error(f"Error in GET /api/reviews/{review_id}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Internal server error fetching review details."}
        )


@app.post("/api/comments/{comment_id}/feedback")
async def submit_comment_feedback(comment_id: str, payload: FeedbackRequest) -> Dict[str, Any]:
    """Submit helpfulness feedback for a review comment."""
    try:
        c_uuid = uuid.UUID(comment_id)
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "error": "Invalid comment UUID format."}
        )

    try:
        db_url = os.getenv("DATABASE_URL")
        conn = await asyncpg.connect(db_url)
        try:
            existing = await conn.fetchrow("SELECT id FROM comments WHERE id = $1", c_uuid)
            if not existing:
                return JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={"status": "error", "error": f"Comment '{comment_id}' not found."}
                )

            await conn.execute(
                "UPDATE comments SET was_helpful = $1 WHERE id = $2",
                payload.was_helpful,
                c_uuid
            )
        finally:
            await conn.close()

        return {
            "status": "success",
            "id": comment_id,
            "was_helpful": payload.was_helpful,
            "message": "Feedback recorded successfully."
        }
    except Exception as e:
        logger.error(f"Error in POST /api/comments/{comment_id}/feedback: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Internal server error updating comment feedback."}
        )


@app.get("/api/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Return metrics breakdown and helpfulness rate over time."""
    try:
        db_url = os.getenv("DATABASE_URL")
        conn = await asyncpg.connect(db_url)
        try:
            total_reviews = await conn.fetchval("SELECT COUNT(*) FROM reviews") or 0
            total_comments = await conn.fetchval("SELECT COUNT(*) FROM comments") or 0
            helpful_comments = await conn.fetchval("SELECT COUNT(*) FROM comments WHERE was_helpful = true") or 0
            unhelpful_comments = await conn.fetchval("SELECT COUNT(*) FROM comments WHERE was_helpful = false") or 0
            unanswered_comments = await conn.fetchval("SELECT COUNT(*) FROM comments WHERE was_helpful IS NULL") or 0

            history_rows = await conn.fetch(
                """
                SELECT 
                  DATE(r.created_at) as date,
                  COUNT(CASE WHEN c.was_helpful = true THEN 1 END) as helpful,
                  COUNT(CASE WHEN c.was_helpful = false THEN 1 END) as unhelpful
                FROM reviews r
                LEFT JOIN comments c ON r.id = c.review_id
                GROUP BY DATE(r.created_at)
                ORDER BY DATE(r.created_at) ASC
                """
            )
        finally:
            await conn.close()

        denom = helpful_comments + unhelpful_comments
        helpfulness_rate = round((helpful_comments / denom * 100), 1) if denom > 0 else 0.0

        history = []
        for h in history_rows:
            d_str = str(h["date"])
            h_count = h["helpful"] or 0
            u_count = h["unhelpful"] or 0
            d_denom = h_count + u_count
            d_rate = round((h_count / d_denom * 100), 1) if d_denom > 0 else 0.0
            history.append({
                "date": d_str,
                "helpfulness_rate": d_rate,
                "helpful": h_count,
                "unhelpful": u_count
            })

        return {
            "total_reviews": total_reviews,
            "total_comments": total_comments,
            "helpful_comments": helpful_comments,
            "unhelpful_comments": unhelpful_comments,
            "unanswered_comments": unanswered_comments,
            "helpfulness_rate": helpfulness_rate,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error in GET /api/metrics: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": "Internal server error calculating metrics."}
        )



