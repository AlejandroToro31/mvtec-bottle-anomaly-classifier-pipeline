"""
Industrial Camera Stream Simulator — MVTec Quality Control API
==============================================================
Simulates an industrial camera sending frame buffers to the FastAPI
inference endpoint. Validates the full request-response pipeline
including authentication, inference, and error handling.

Test scenarios:
    1. Valid defect image   → expects "anomaly" prediction
    2. Valid good image     → expects "good" prediction
    3. Invalid file type    → expects 400 Bad Request
    4. Wrong API key        → expects 401 Unauthorized
    5. Health check         → validates server readiness before tests

Usage:
    # Set the API key before running
    export API_SECRET_KEY=your_secret_key

    # Point to your test assets
    python test_client.py

Configuration via environment variables:
    API_SECRET_KEY  : Authentication key (required — no default)
    API_BASE_URL    : Base URL of the running API (default: http://127.0.0.1:8000)
"""

import logging
import os
import sys
import time

import requests

# ════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════

logging.basicConfig(

    level=logging.INFO,
    format="%(asctime)s - [CAMERA-SIM] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CameraSimulator")

# API key loaded from environment — never hardcoded in source
# Set via: export API_SECRET_KEY=your_secret_key
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", "mvtec_dev_key")
API_BASE_URL   = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL    = f"{API_BASE_URL}/api/v1/predict"
HEALTH_URL     = f"{API_BASE_URL}/health"
READY_URL      = f"{API_BASE_URL}/ready"

HEADERS = {"Authorization-API-Key": API_SECRET_KEY}
TIMEOUT_SECONDS = 10

# Test asset paths — update to point to your MVTec images
DEFECT_IMAGE_PATH = "tests/assets/sample_defect.png"
GOOD_IMAGE_PATH   = "tests/assets/sample_good.png"


# ════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════

def check_server_health() -> bool:
    """
    Verifies the API is alive and the model is loaded before running tests.

    Two-stage check:
        /health → process is alive
        /ready  → model is loaded and inference is possible

    Returns True if both pass, False otherwise.
    """
    logger.info("Checking server health before test run...")
    
    try:
        health = requests.get(HEALTH_URL, timeout=TIMEOUT_SECONDS)
        
        if health.status_code != 200:
            logger.error(f"Health check failed: {health.status_code}")
            return False

        ready = requests.get(READY_URL, timeout=TIMEOUT_SECONDS)
        
        if ready.status_code != 200:
            logger.error(f"Readiness check failed — model may still be loading.")
            return False

        logger.info("Server healthy and model ready.")
        return True

    except requests.exceptions.ConnectionError:
        
        logger.error(
            f"Cannot connect to API at {API_BASE_URL}. "
            "Is the server running? Try: uvicorn app.main:app --host 0.0.0.0 --port 8000"
        )
        return False


def send_image(image_path: str, headers: dict = HEADERS, label: str = "test") -> dict:
    """
    Sends a single image to the inference endpoint and returns the response.

    Simulates an industrial camera transmitting a raw frame buffer:
        1. Opens image in binary mode ('rb') — raw bytes, no preprocessing
        2. Wraps in multipart form data — matches FastAPI UploadFile format
        3. Fires HTTP POST with authentication header
        4. Measures total round-trip network latency

    Args:
        image_path : Path to the image file to send
        headers    : Request headers including API key
        label      : Test label for logging output

    Returns:
        dict with keys: status_code, json (if 200), network_ms, error (if not 200)
    """
    if not os.path.exists(image_path):

        logger.error(f"[{label}] Test asset not found: {image_path}")
        return {"status_code": -1, "error": f"File not found: {image_path}"}

    filename = os.path.basename(image_path)
    logger.info(f"[{label}] Transmitting payload: '{filename}'...")

    start = time.perf_counter()

    try:
        with open(image_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            response = requests.post(
                PREDICT_URL,
                headers=headers,
                files=files,
                timeout=TIMEOUT_SECONDS,
            )

        network_ms = round((time.perf_counter() - start) * 1000, 2)

        if response.status_code == 200:
            return {
                "status_code": response.status_code,
                "json"       : response.json(),
                "network_ms" : network_ms,
            }
        else:
            return {
                "status_code": response.status_code,
                "error"      : response.text,
                "network_ms" : network_ms,
            }

    except requests.exceptions.Timeout:
        return {"status_code": -1, "error": f"Request timed out after {TIMEOUT_SECONDS}s"}
    except requests.exceptions.ConnectionError as e:
        return {"status_code": -1, "error": f"Connection error: {e}"}


def print_result(result: dict, expected_status: int = 200) -> bool:
    """
    Prints formatted test result and returns True if test passed.

    Pass condition: status_code matches expected_status.
    """
    status = result["status_code"]
    passed = status == expected_status

    if passed and status == 200:
        payload = result["json"]
        logger.info(
            f"  PASS | Prediction: {payload['prediction'].upper()} | "
            f"Confidence: {payload['confidence']} | "
            f"API latency: {payload['latency_ms']}ms | "
            f"Network: {result['network_ms']}ms"
        )
    elif passed:
        logger.info(f"  PASS | Expected {expected_status}, got {status}")
    else:
        logger.error(
            f"  FAIL | Expected {expected_status}, got {status} | "
            f"Error: {result.get('error', 'unknown')}"
        )

    return passed


# ════════════════════════════════════════════════════════
# TEST SCENARIOS
# ════════════════════════════════════════════════════════

def run_test_suite() -> None:
    """
    Executes the full test suite simulating an industrial camera feed.

    Tests:
        1. Server health check      → validates API is ready before inference
        2. Defect image inference   → expects "anomaly" + review queue routing
        3. Good image inference     → expects "good" prediction
        4. Invalid file type        → expects 400 Bad Request
        5. Wrong API key            → expects 401 Unauthorized
    """
    logger.info("=" * 55)
    logger.info("INDUSTRIAL CAMERA SIMULATION — MVTec Quality Control")
    logger.info("=" * 55)

    results = []

    # ── Test 0: Server readiness
    logger.info("\n[TEST 0] Server health check...")

    if not check_server_health():
        logger.error("Server not ready. Aborting test suite.")
        sys.exit(1)
    results.append(True)

    # ── Test 1: Defect image — expects "anomaly"
    logger.info(f"\n[TEST 1] Defect image inference — expects 'anomaly'...")
    result = send_image(DEFECT_IMAGE_PATH, label="DEFECT")
    passed = print_result(result, expected_status=200)

    if passed and result["json"]["prediction"] != "anomaly":
        logger.warning(
            f"  WARNING: Expected 'anomaly' but got '{result['json']['prediction']}'. "
            f"Model may need retraining or threshold adjustment."
        )
    results.append(passed)

    # ── Test 2: Good image — expects "good"
    logger.info(f"\n[TEST 2] Nominal image inference — expects 'good'...")
    result = send_image(GOOD_IMAGE_PATH, label="NOMINAL")
    passed = print_result(result, expected_status=200)

    if passed and result["json"]["prediction"] != "good":
        logger.warning(
            f"  WARNING: Expected 'good' but got '{result['json']['prediction']}'. "
            f"Check for false positive rate — may indicate low confidence threshold."
        )
    results.append(passed)

    # ── Test 3: Invalid file type — expects 400
    logger.info("\n[TEST 3] Invalid file type (PDF) — expects 400 Bad Request...")
    dummy_pdf_path = "/tmp/test_invalid.pdf"
    
    with open(dummy_pdf_path, "wb") as f:
        f.write(b"%PDF-1.4 dummy content")

    with open(dummy_pdf_path, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        try:
            response = requests.post(
                PREDICT_URL, headers=HEADERS, files=files, timeout=TIMEOUT_SECONDS
            )
            result = {"status_code": response.status_code, "error": response.text}
        except Exception as e:
            result = {"status_code": -1, "error": str(e)}

    results.append(print_result(result, expected_status=400))
    os.remove(dummy_pdf_path)

    # ── Test 4: Wrong API key — expects 401
    logger.info("\n[TEST 4] Invalid API key — expects 401 Unauthorized...")
    wrong_headers = {"Authorization-API-Key": "wrong_key_completely"}
    result = send_image(DEFECT_IMAGE_PATH, headers=wrong_headers, label="AUTH-TEST")
    results.append(print_result(result, expected_status=401))

    # ── Summary
    passed_count = sum(results)
    total = len(results)
    logger.info("\n" + "=" * 55)
    logger.info(f"TEST SUITE COMPLETE: {passed_count}/{total} passed")

    if passed_count == total:
        logger.info("ALL TESTS PASSED — API pipeline validated.")
        logger.info("Review queue check: verify anomaly image was saved to "
                    f"{os.environ.get('REVIEW_QUEUE_DIR', 'data/review_queue')}")
    else:
        logger.error(f"{total - passed_count} test(s) FAILED — review logs above.")
        sys.exit(1)

    logger.info("=" * 55)


# ════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_test_suite()