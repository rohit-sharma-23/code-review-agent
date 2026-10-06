"""
Integration test script for Phase 6 Webhook Receiver.
Tests signature verification, event filtering, background dispatching, and health endpoints.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("test_webhook")

from fastapi.testclient import TestClient
from app.main import app
from app.db import db_manager

def create_signature(payload_bytes: bytes, secret: str) -> str:
    hash_val = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={hash_val}"


def run_webhook_tests():
    logger.info("======================================================")
    logger.info("       STARTING WEBHOOK RECEIVER INTEGRATION TESTS    ")
    logger.info("======================================================")

    secret = os.getenv("GITHUB_WEBHOOK_SECRET", "test_secret_key_999")
    os.environ["GITHUB_WEBHOOK_SECRET"] = secret

    client = TestClient(app)
    passed_tests = 0
    total_tests = 6

    # ------------------------------------------------------------------
    # Test 1: GET /health
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 1] GET /health ---")
    res1 = client.get("/health")
    logger.info(f"Health check status: {res1.status_code}, response: {res1.json()}")
    assert res1.status_code == 200, "Health check should return 200."
    assert res1.json().get("status") == "healthy", "Service should be healthy."
    passed_tests += 1
    logger.info("PASSED [TEST 1]")

    # ------------------------------------------------------------------
    # Test 2: POST /webhook (Missing signature -> HTTP 401)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 2] POST /webhook (Missing signature) ---")
    payload2 = json.dumps({"action": "opened", "number": 1}).encode("utf-8")
    res2 = client.post("/webhook", content=payload2, headers={"Content-Type": "application/json"})
    logger.info(f"Response status: {res2.status_code}, body: {res2.json()}")
    assert res2.status_code == 401, "Missing signature should return 401."
    passed_tests += 1
    logger.info("PASSED [TEST 2] (Missing signature correctly rejected)")

    # ------------------------------------------------------------------
    # Test 3: POST /webhook (Invalid signature -> HTTP 401)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 3] POST /webhook (Invalid signature) ---")
    payload3 = json.dumps({"action": "opened", "number": 1}).encode("utf-8")
    res3 = client.post(
        "/webhook",
        content=payload3,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": "sha256=invalid00000000000000000000000000000000000000000000000000000000",
            "X-GitHub-Delivery": "test-delivery-101",
        }
    )
    logger.info(f"Response status: {res3.status_code}, body: {res3.json()}")
    assert res3.status_code == 401, "Invalid signature should return 401."
    passed_tests += 1
    logger.info("PASSED [TEST 3] (Invalid signature correctly rejected)")

    # ------------------------------------------------------------------
    # Test 4: POST /webhook (Valid signature + action="opened" -> HTTP 200 Accepted)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 4] POST /webhook (Valid signature + action='opened') ---")
    valid_payload_dict = {
        "action": "opened",
        "number": 1,
        "repository": {
            "full_name": "rohitsrma/Resume-reviewer",
            "default_branch": "main"
        },
        "pull_request": {
            "head": {
                "sha": "3d9e18f438e163e956e3dc8260bf593fc27f7d29"
            }
        }
    }
    payload4_bytes = json.dumps(valid_payload_dict).encode("utf-8")
    sig4 = create_signature(payload4_bytes, secret)

    res4 = client.post(
        "/webhook",
        content=payload4_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig4,
            "X-GitHub-Delivery": "test-delivery-104",
            "X-GitHub-Event": "pull_request"
        }
    )
    logger.info(f"Response status: {res4.status_code}, body: {res4.json()}")
    assert res4.status_code == 200, "Valid opened PR webhook should return 200."
    assert res4.json().get("status") == "accepted", "Status should be accepted."
    passed_tests += 1
    logger.info("PASSED [TEST 4] (Opened PR webhook accepted)")

    # ------------------------------------------------------------------
    # Test 5: POST /webhook (Valid signature + action="synchronize" -> HTTP 200 Accepted)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 5] POST /webhook (Valid signature + action='synchronize') ---")
    valid_payload_dict["action"] = "synchronize"
    payload5_bytes = json.dumps(valid_payload_dict).encode("utf-8")
    sig5 = create_signature(payload5_bytes, secret)

    res5 = client.post(
        "/webhook",
        content=payload5_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig5,
            "X-GitHub-Delivery": "test-delivery-105",
            "X-GitHub-Event": "pull_request"
        }
    )
    logger.info(f"Response status: {res5.status_code}, body: {res5.json()}")
    assert res5.status_code == 200, "Valid synchronize PR webhook should return 200."
    assert res5.json().get("status") == "accepted", "Status should be accepted."
    passed_tests += 1
    logger.info("PASSED [TEST 5] (Synchronize PR webhook accepted)")

    # ------------------------------------------------------------------
    # Test 6: POST /webhook (Valid signature + action="closed" -> HTTP 200 Ignored)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 6] POST /webhook (Valid signature + action='closed') ---")
    valid_payload_dict["action"] = "closed"
    payload6_bytes = json.dumps(valid_payload_dict).encode("utf-8")
    sig6 = create_signature(payload6_bytes, secret)

    res6 = client.post(
        "/webhook",
        content=payload6_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": sig6,
            "X-GitHub-Delivery": "test-delivery-106",
            "X-GitHub-Event": "pull_request"
        }
    )
    logger.info(f"Response status: {res6.status_code}, body: {res6.json()}")
    assert res6.status_code == 200, "Closed PR webhook should return 200."
    assert res6.json().get("status") == "ignored", "Status should be ignored."
    passed_tests += 1
    logger.info("PASSED [TEST 6] (Closed action correctly ignored)")

    # ------------------------------------------------------------------
    # Test 7: POST /webhook (Form-encoded payload: application/x-www-form-urlencoded)
    # ------------------------------------------------------------------
    logger.info("\n--- [TEST 7] POST /webhook (Form-encoded payload) ---")
    from urllib.parse import quote
    valid_payload_dict["action"] = "reopened"
    raw_json_str = json.dumps(valid_payload_dict)
    form_encoded_str = f"payload={quote(raw_json_str)}"
    payload7_bytes = form_encoded_str.encode("utf-8")
    sig7 = create_signature(payload7_bytes, secret)

    res7 = client.post(
        "/webhook",
        content=payload7_bytes,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Hub-Signature-256": sig7,
            "X-GitHub-Delivery": "test-delivery-107",
            "X-GitHub-Event": "pull_request"
        }
    )
    logger.info(f"Response status: {res7.status_code}, body: {res7.json()}")
    assert res7.status_code == 200, "Form-encoded PR webhook should return 200."
    assert res7.json().get("status") == "accepted", "Status should be accepted."
    passed_tests += 1
    logger.info("PASSED [TEST 7] (Form-encoded payload successfully parsed and accepted)")

    logger.info("\n======================================================")
    logger.info(f"       SUMMARY: {passed_tests}/7 TESTS PASSED SUCCESSFULLY       ")
    logger.info("======================================================")



if __name__ == "__main__":
    run_webhook_tests()
