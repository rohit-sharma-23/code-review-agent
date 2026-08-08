"""
AI Code Review Agent

Orchestrates MCP tools, Gemini LLM review generation, Pydantic validation,
and persistent database record tracking for GitHub Pull Requests.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union

import httpx
import asyncpg
from dotenv import load_dotenv
from pydantic import ValidationError

from app.db import get_db_pool
from app.models import ReviewFinding
from app.mcp_server import get_diff, post_review_comment, search_codebase

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agent")

# Primary LLM model configuration with automatic fallback support
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
MAX_COMMENTS_PER_PR = 8

# Helper Utilities

def parse_diff_hunks(diff_text: str) -> Dict[str, str]:
    """Parse unified diff text into a dictionary mapping file_path to its diff hunk."""
    hunks: Dict[str, str] = {}
    if not diff_text:
        return hunks

    file_blocks = re.split(r'^diff --git ', diff_text, flags=re.MULTILINE)
    for block in file_blocks:
        if not block.strip():
            continue

        header_match = re.search(r'^\+\+\+\s+b/(.+)$', block, flags=re.MULTILINE)
        if not header_match:
            continue

        file_path = header_match.group(1).strip()
        hunks[file_path] = block

    return hunks


async def generate_gemini_review_json(prompt: str, max_retries: int = 2) -> str:
    """Call Gemini generateContent API with fallback model options and retries."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    # Try reliable active models
    model_candidates = ["gemini-flash-latest", DEFAULT_LLM_MODEL, "gemini-2.0-flash-lite"]
    # De-duplicate while preserving order
    models = [m for m in dict.fromkeys(model_candidates) if m != "gemini-2.5-flash"]


    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient() as client:
        last_error = None
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            for attempt in range(max_retries):
                try:
                    resp = await client.post(url, headers=headers, json=body, timeout=45.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            return text
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        logger.warning(
                            f"Gemini API model '{model}' returned status {resp.status_code}. Retry {attempt + 1}/{max_retries}..."
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                    else:
                        logger.warning(f"Model '{model}' returned status {resp.status_code}: {resp.text[:150]}")
                        last_error = f"HTTP {resp.status_code}: {resp.text}"
                        break  # Try next model if 404 or unsupported
                except httpx.RequestError as e:
                    logger.warning(f"Network error calling Gemini model '{model}': {e}")
                    last_error = str(e)
                    await asyncio.sleep(1.0)

    raise RuntimeError(f"Failed to generate review response from Gemini API: {last_error}")


# Core Agent Review Workflow

async def review_pr(
    pr_number: int,
    repo_id: Union[uuid.UUID, str],
    commit_sha: str,
) -> Dict[str, Any]:
    """
    Execute end-to-end AI code review workflow for a Pull Request.

    Args:
        pr_number: The Pull Request number.
        repo_id: The UUID of the repository in the database.
        commit_sha: The head commit SHA of the Pull Request.

    Returns:
        Structured JSON summary of the review operation.
    """
    if isinstance(repo_id, str):
        repo_id = uuid.UUID(repo_id)

    logger.info(f"=== Starting Code Review for PR #{pr_number} (Repo ID: {repo_id}, Commit: {commit_sha[:7]}) ===")
    start_time = time.time()
    pool = await get_db_pool()

    review_id = uuid.uuid4()

    db_url = os.getenv("DATABASE_URL")

    # 1. Idempotency Check & Insert Review Record
    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(
            """
            INSERT INTO reviews (id, repo_id, pr_number, commit_sha, status)
            VALUES ($1, $2, $3, $4, 'in_progress')
            """,
            review_id,
            repo_id,
            pr_number,
            commit_sha,
        )
        logger.info(f"Created new review record with ID: {review_id}")
    except asyncpg.UniqueViolationError:
        logger.info(f"Review for repo '{repo_id}', PR #{pr_number}, commit '{commit_sha[:7]}' already exists. Skipping.")
        return {
            "status": "skipped",
            "message": "Review already exists for this commit SHA.",
            "pr_number": pr_number,
            "commit_sha": commit_sha,
        }
    except Exception as e:
        logger.error(f"Failed to insert review record: {e}")
        return {
            "status": "failed",
            "error": f"Database error creating review record: {str(e)}",
            "pr_number": pr_number,
        }
    finally:
        await conn.close()



    try:
        # 2. Fetch PR Diff via MCP tool
        logger.info(f"Fetching diff for PR #{pr_number}...")
        diff_res = await get_diff(pr_number)

        if diff_res.get("status") == "error":
            error_msg = diff_res.get("error", "Failed to fetch PR diff.")
            logger.error(f"PR diff error: {error_msg}")
            async with pool.acquire() as conn:
                await conn.execute("UPDATE reviews SET status = 'failed' WHERE id = $1", review_id)
            return {
                "status": "failed",
                "error": error_msg,
                "pr_number": pr_number,
            }

        diff_text = diff_res.get("diff", "")
        pr_title = diff_res.get("title", "")
        base_branch = diff_res.get("base_branch", "")

        # 3. Parse File Hunks
        file_hunks = parse_diff_hunks(diff_text)
        logger.info(f"PR #{pr_number} contains changes in {len(file_hunks)} file(s).")

        if not file_hunks:
            logger.info(f"No file changes found in diff for PR #{pr_number}.")
            async with pool.acquire() as conn:
                await conn.execute("UPDATE reviews SET status = 'completed' WHERE id = $1", review_id)
            return {
                "status": "completed",
                "review_id": str(review_id),
                "pr_number": pr_number,
                "generated_comments_count": 0,
                "posted_comments_count": 0,
                "message": "No code changes found to review.",
            }

        # 4. Codebase Context Retrieval (RAG)
        logger.info(f"Searching codebase context concurrently across {len(file_hunks)} file(s)...")

        async def fetch_file_context(file_path: str, hunk: str) -> List[str]:
            query_str = f"File: {file_path}\nDiff Summary:\n{hunk[:500]}"
            search_res = await search_codebase(query=query_str, top_k=3)
            blocks = []
            if search_res.get("status") == "success":
                for r in search_res.get("results", []):
                    blocks.append(
                        f"--- Context from {r['file_path']} ({r['chunk_type']}, similarity: {r['similarity_score']}) ---\n{r['content']}"
                    )
            return blocks

        context_tasks = [fetch_file_context(fp, hunk) for fp, hunk in file_hunks.items()]
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)

        retrieved_context_blocks: List[str] = []
        for res in context_results:
            if isinstance(res, list):
                retrieved_context_blocks.extend(res)

        retrieved_context_str = "\n\n".join(retrieved_context_blocks) if retrieved_context_blocks else "No additional background context found."
        logger.info(f"Retrieved {len(retrieved_context_blocks)} total context snippet(s) from codebase.")


        # 5. Construct Prompt for Gemini LLM
        prompt = f"""
You are an expert senior code reviewer conducting a strict code review on Pull Request #{pr_number}.

PULL REQUEST METADATA:
- PR Number: #{pr_number}
- Title: {pr_title}
- Target Base Branch: {base_branch}
- Head Commit SHA: {commit_sha}

RETRIEVED CODEBASE CONTEXT (RAG):
{retrieved_context_str}

FULL PULL REQUEST UNIFIED DIFF:
{diff_text}

INSTRUCTIONS & RULES:
1. Carefully analyze the added lines in the diff.
2. Identify real bugs, security risks, missing unit tests, or clear style/best practice deviations.
3. Only report actionable issues on ADDED or MODIFIED lines (lines starting with '+' in the diff).
4. Provide precise 1-based line numbers in the NEW version of the modified file.
5. Return your response ONLY as a valid JSON array of objects without markdown formatting wrappers or conversational text.

REQUIRED OUTPUT JSON SCHEMA:
[
  {{
    "file_path": "relative/file/path.py",
    "line": 15,
    "comment_type": "bug_risk",
    "body": "Clear, constructive explanation of the issue and how to fix it."
  }}
]

Allowed values for 'comment_type':
- "bug_risk" (potential bug, security issue, or logic flaw)
- "missing_test" (untested boundary condition or critical logic missing tests)
- "style_deviation" (code style, naming convention, or architectural deviation)

If no significant issues are found, return an empty JSON array `[]`.
"""

        # 6. Call Gemini LLM with Exponential Backoff
        logger.info(f"Submitting prompt to Gemini LLM for PR #{pr_number} review...")
        llm_start = time.time()
        raw_llm_response = await generate_gemini_review_json(prompt)
        llm_latency = time.time() - llm_start
        logger.info(f"Gemini LLM returned response in {llm_latency:.2f} seconds.")

        # Clean JSON markdown formatting if present
        clean_json_str = raw_llm_response.strip()
        if clean_json_str.startswith("```"):
            clean_json_str = re.sub(r"^```(?:json)?\n?", "", clean_json_str, flags=re.IGNORECASE)
            clean_json_str = re.sub(r"\n?```$", "", clean_json_str)
            clean_json_str = clean_json_str.strip()

        # 7. Validate LLM Response using Pydantic
        validated_findings: List[ReviewFinding] = []
        try:
            parsed_data = json.loads(clean_json_str)
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    try:
                        finding = ReviewFinding(**item)
                        validated_findings.append(finding)
                    except ValidationError as ve:
                        logger.warning(f"Discarding malformed finding object {item}: {ve}")
            else:
                logger.warning(f"LLM returned non-list JSON payload: {type(parsed_data)}")
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON response: {e}. Raw response:\n{raw_llm_response}")

        logger.info(f"Validated {len(validated_findings)} review finding(s) from LLM output.")

        # Limit total comments per PR to avoid noisy reviews
        if len(validated_findings) > MAX_COMMENTS_PER_PR:
            logger.info(f"Capping findings to maximum {MAX_COMMENTS_PER_PR} comments per PR.")
            validated_findings = validated_findings[:MAX_COMMENTS_PER_PR]

        # 8. Post Comments via MCP Tool & Store in DB
        posted_comments_count = 0
        for finding in validated_findings:
            logger.info(f"Posting review comment on {finding.file_path}:{finding.line} ({finding.comment_type})...")
            
            post_res = await post_review_comment(
                pr_number=pr_number,
                file_path=finding.file_path,
                line=finding.line,
                body=finding.body,
            )

            # Store comment in database regardless of GitHub PAT write permission
            comment_id = uuid.uuid4()
            c_conn = await asyncpg.connect(db_url)
            try:
                await c_conn.execute(
                    """
                    INSERT INTO comments (id, review_id, file_path, line_number, comment_type, body)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    comment_id,
                    review_id,
                    finding.file_path,
                    finding.line,
                    finding.comment_type,
                    finding.body,
                )
            finally:
                await c_conn.close()

            if post_res.get("status") == "success":
                posted_comments_count += 1
                logger.info(f"  Successfully posted comment #{post_res.get('comment_id')} on GitHub!")
            else:
                logger.warning(f"  GitHub comment post status: {post_res.get('code')} - {post_res.get('error')}")

        # 9. Update Review Status to Completed
        s_conn = await asyncpg.connect(db_url)
        try:
            await s_conn.execute("UPDATE reviews SET status = 'completed' WHERE id = $1", review_id)
        finally:
            await s_conn.close()

        total_latency = time.time() - start_time
        logger.info(f"=== Completed PR #{pr_number} review in {total_latency:.2f}s (LLM: {llm_latency:.2f}s). Posted {posted_comments_count}/{len(validated_findings)} comment(s) ===")

        return {
            "status": "completed",
            "review_id": str(review_id),
            "pr_number": pr_number,
            "commit_sha": commit_sha,
            "generated_comments_count": len(validated_findings),
            "posted_comments_count": posted_comments_count,
            "llm_latency_seconds": round(llm_latency, 2),
            "total_latency_seconds": round(total_latency, 2),
        }

    except Exception as e:
        logger.error(f"Unrecoverable error in review_pr(PR #{pr_number}): {e}", exc_info=True)
        f_conn = await asyncpg.connect(db_url)
        try:
            await f_conn.execute("UPDATE reviews SET status = 'failed' WHERE id = $1", review_id)
        finally:
            await f_conn.close()


        return {
            "status": "failed",
            "error": f"Unrecoverable error during code review: {str(e)}",
            "pr_number": pr_number,
            "commit_sha": commit_sha,
        }

