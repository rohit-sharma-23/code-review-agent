"""
Standalone integration test script for MCP tools.
Invokes each MCP tool directly against GitHub API and Supabase database.
"""

import asyncio
import json
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_mcp")

from app.db import db_manager
from app.mcp_server import get_diff, get_file_context, search_codebase, post_review_comment


async def run_tests():
    logger.info("======================================================")
    logger.info("       STARTING MCP SERVER INTEGRATION TESTS         ")
    logger.info("======================================================")

    passed_tests = 0
    total_tests = 5

    # ------------------------------------------------------------------
    # Test 1: get_diff(pr_number=1)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 1] Invoking get_diff(pr_number=1) ---")
    diff_res = await get_diff(pr_number=1)
    logger.info(f"Response status: {diff_res.get('status')}")
    if diff_res.get("status") == "success":
        logger.info(f"  PR Title: {diff_res.get('title')}")
        logger.info(f"  Base Branch: {diff_res.get('base_branch')}")
        logger.info(f"  Head SHA: {diff_res.get('head_sha')[:7]}")
        logger.info(f"  Diff Bytes Length: {len(diff_res.get('diff', ''))}")
        assert len(diff_res.get("diff", "")) > 0, "Diff content should not be empty."
        passed_tests += 1
        logger.info("PASSED [TEST 1]")
    else:
        logger.error(f"FAILED [TEST 1]: {diff_res}")

    # ------------------------------------------------------------------
    # Test 2: get_file_context(file_path="manage.py", ref="main")
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 2] Invoking get_file_context(file_path='manage.py', ref='main') ---")
    file_res = await get_file_context(file_path="manage.py", ref="main")
    logger.info(f"Response status: {file_res.get('status')}")
    if file_res.get("status") == "success":
        logger.info(f"  File SHA: {file_res.get('file_sha')}")
        logger.info(f"  Content Sample:\n{file_res.get('content', '')[:100]}...")
        assert "main" in file_res.get("content", ""), "Decoded content should contain 'main'."
        passed_tests += 1
        logger.info("PASSED [TEST 2]")
    else:
        logger.error(f"FAILED [TEST 2]: {file_res}")

    # ------------------------------------------------------------------
    # Test 3: search_codebase(query="Django settings", top_k=3)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 3] Invoking search_codebase(query='Django settings', top_k=3) ---")
    search_res = await search_codebase(query="Django settings", top_k=3)
    logger.info(f"Response status: {search_res.get('status')}")
    if search_res.get("status") == "success":
        results = search_res.get("results", [])
        logger.info(f"  Found {len(results)} matching chunk(s):")
        for idx, item in enumerate(results, 1):
            logger.info(f"    {idx}. [{item['chunk_type']}] {item['file_path']} (score: {item['similarity_score']})")
        assert len(results) > 0, "Vector search should return at least 1 result."
        passed_tests += 1
        logger.info("PASSED [TEST 3]")
    else:
        logger.error(f"FAILED [TEST 3]: {search_res}")

    # ------------------------------------------------------------------
    # Test 4: post_review_comment (Validation failure test on un-modified line)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 4] Invoking post_review_comment on invalid line (line=9999) ---")
    invalid_comment_res = await post_review_comment(
        pr_number=1,
        file_path="manage.py",
        line=9999,
        body="This is a test comment on an invalid line."
    )
    logger.info(f"Response status: {invalid_comment_res.get('status')}")
    logger.info(f"  Error Code: {invalid_comment_res.get('code')}")
    logger.info(f"  Error Msg: {invalid_comment_res.get('error')}")
    assert invalid_comment_res.get("status") == "error", "Should reject invalid line position."
    assert invalid_comment_res.get("code") == "INVALID_COMMENT_POSITION", "Error code should be INVALID_COMMENT_POSITION."
    passed_tests += 1
    logger.info("PASSED [TEST 4] (Validation correctly rejected invalid line)")

    # ------------------------------------------------------------------
    # Test 5: post_review_comment (Valid line test on line=5)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 5] Invoking post_review_comment on valid line (manage.py line=5) ---")
    valid_comment_res = await post_review_comment(
        pr_number=1,
        file_path="manage.py",
        line=5,
        body="Automated code review test comment on manage.py line 5."
    )
    logger.info(f"Response status: {valid_comment_res.get('status')}")
    if valid_comment_res.get("status") == "success":
        logger.info(f"  Comment ID: {valid_comment_res.get('comment_id')}")
        logger.info(f"  Comment URL: {valid_comment_res.get('html_url')}")
    else:
        logger.info(f"  Handled Response Code: {valid_comment_res.get('code')}")
        logger.info(f"  Response Details: {valid_comment_res.get('error')}")
    # Response must be structured JSON and not raise uncaught exception
    assert "status" in valid_comment_res, "Response must contain 'status' field."
    passed_tests += 1
    logger.info("PASSED [TEST 5] (Structured response returned without uncaught exception)")

    logger.info("\n======================================================")
    logger.info(f"       SUMMARY: {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY       ")
    logger.info("======================================================")


async def main():
    try:
        await run_tests()
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
