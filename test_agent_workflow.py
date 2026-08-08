"""
Integration test script for Phase 5 Agent Loop.
Executes review_pr() against PR #1 of rohitsrma/Resume-reviewer.
"""

import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_agent")

from app.db import db_manager, get_db_pool
from app.agent import review_pr


async def run_agent_test():
    logger.info("======================================================")
    logger.info("       STARTING AGENT LOOP INTEGRATION TEST          ")
    logger.info("======================================================")

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Get repository record
        repo = await conn.fetchrow("SELECT id, github_full_name FROM repos WHERE github_full_name = $1", "rohitsrma/Resume-reviewer")
        if not repo:
            logger.error("Repository 'rohitsrma/Resume-reviewer' not found in database. Run ingestion first!")
            return
        
        repo_id = repo["id"]
        logger.info(f"Using Repo ID: {repo_id}")

        pr_number = 1
        commit_sha = "3d9e18f438e163e956e3dc8260bf593fc27f7d29"

        # Clean up existing test review records for PR #1 to test fresh execution
        await conn.execute("DELETE FROM comments WHERE review_id IN (SELECT id FROM reviews WHERE repo_id = $1 AND pr_number = $2)", repo_id, pr_number)
        await conn.execute("DELETE FROM reviews WHERE repo_id = $1 AND pr_number = $2", repo_id, pr_number)


        # ------------------------------------------------------------------
        # Run 1: First time execution of review_pr
        # ------------------------------------------------------------------
        logger.info("\n--- [RUN 1] Executing review_pr for PR #1 ---")
        res1 = await review_pr(pr_number=pr_number, repo_id=repo_id, commit_sha=commit_sha)
        logger.info(f"Run 1 Result: {res1}")
        assert res1.get("status") == "completed", f"Run 1 should complete successfully, got: {res1}"


        # ------------------------------------------------------------------
        # Run 2: Test Idempotency (re-running for exact same commit SHA)
        # ------------------------------------------------------------------
        logger.info("\n--- [RUN 2] Re-running review_pr for exact same commit SHA (Idempotency Test) ---")
        res2 = await review_pr(pr_number=pr_number, repo_id=repo_id, commit_sha=commit_sha)
        logger.info(f"Run 2 Result: {res2}")
        assert res2.get("status") == "skipped", "Run 2 must return status='skipped' due to idempotency constraint."
        logger.info("PASSED [Idempotency Test] (Second run returned status='skipped' without duplicate processing)")

        # ------------------------------------------------------------------
        # Check DB Records
        # ------------------------------------------------------------------
        review_count = await conn.fetchval("SELECT COUNT(*) FROM reviews WHERE repo_id = $1 AND pr_number = $2", repo_id, pr_number)
        comment_count = await conn.fetchval(
            "SELECT COUNT(*) FROM comments c JOIN reviews r ON c.review_id = r.id WHERE r.repo_id = $1 AND r.pr_number = $2",
            repo_id, pr_number
        )
        logger.info(f"\nDatabase State for PR #{pr_number}:")
        logger.info(f"  Reviews in DB: {review_count}")
        logger.info(f"  Comments in DB: {comment_count}")

    logger.info("\n======================================================")
    logger.info("       AGENT LOOP TEST COMPLETED SUCCESSFULLY         ")
    logger.info("======================================================")


async def main():
    try:
        await run_agent_test()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
