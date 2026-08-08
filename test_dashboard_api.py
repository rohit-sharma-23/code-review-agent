"""
Integration test script for Phase 7 Dashboard API Endpoints.
"""

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_dashboard_api")

from fastapi.testclient import TestClient
from app.main import app


def run_dashboard_api_tests():
    logger.info("======================================================")
    logger.info("    STARTING DASHBOARD API INTEGRATION TESTS          ")
    logger.info("======================================================")

    client = TestClient(app)
    passed_tests = 0

    # 1. GET /api/reviews
    logger.info("\n--- [TEST 1] GET /api/reviews ---")
    res1 = client.get("/api/reviews?page=1&page_size=10")
    logger.info(f"Status: {res1.status_code}, body: {res1.json()}")
    assert res1.status_code == 200, "GET /api/reviews should return 200"
    data1 = res1.json()
    assert "items" in data1 and "total" in data1, "Response missing items/total"
    passed_tests += 1
    logger.info("PASSED [TEST 1]")

    review_id = None
    if data1["items"]:
        review_id = data1["items"][0]["id"]
        logger.info(f"Target review ID for test 2: {review_id}")

    # 2. GET /api/reviews/{id}
    if review_id:
        logger.info(f"\n--- [TEST 2] GET /api/reviews/{review_id} ---")
        res2 = client.get(f"/api/reviews/{review_id}")
        logger.info(f"Status: {res2.status_code}, body: {res2.json()}")
        assert res2.status_code == 200, "GET /api/reviews/{id} should return 200"
        data2 = res2.json()
        assert "comments" in data2, "Response missing comments array"
        passed_tests += 1
        logger.info("PASSED [TEST 2]")

        # 3. POST /api/comments/{id}/feedback
        if data2["comments"]:
            comment_id = data2["comments"][0]["id"]
            logger.info(f"\n--- [TEST 3] POST /api/comments/{comment_id}/feedback ---")
            res3 = client.post(f"/api/comments/{comment_id}/feedback", json={"was_helpful": True})
            logger.info(f"Status: {res3.status_code}, body: {res3.json()}")
            assert res3.status_code == 200, "Feedback POST should return 200"
            assert res3.json().get("was_helpful") is True, "was_helpful should be True"
            passed_tests += 1
            logger.info("PASSED [TEST 3]")

    # 4. GET /api/metrics
    logger.info("\n--- [TEST 4] GET /api/metrics ---")
    res4 = client.get("/api/metrics")
    logger.info(f"Status: {res4.status_code}, body: {res4.json()}")
    assert res4.status_code == 200, "GET /api/metrics should return 200"
    data4 = res4.json()
    assert "total_reviews" in data4 and "helpfulness_rate" in data4, "Metrics response invalid"
    passed_tests += 1
    logger.info("PASSED [TEST 4]")

    logger.info("\n======================================================")
    logger.info("    DASHBOARD API TESTS PASSED SUCCESSFULLY!          ")
    logger.info("======================================================")


if __name__ == "__main__":
    run_dashboard_api_tests()
