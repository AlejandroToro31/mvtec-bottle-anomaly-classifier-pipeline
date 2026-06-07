"""
Factory Quality Control API — MVTec Binary Anomaly Classifier
=============================================================
Production FastAPI microservice for supervised binary defect detection
on MVTec AD bottle images.

Features:
    - API key authentication for endpoint security
    - Double payload validation (header + actual bytes) — detects spoofed headers
    - Zero disk I/O image decoding via cv2.imdecode
    - asyncio.to_thread for non-blocking inference
    - Automated QA routing — anomaly images saved for active learning loop
    - Model warmup inference on startup

Endpoints:
    GET  /              → API metadata
    GET  /health        → Liveness check
    GET  /ready         → Readiness check (model loaded)
    POST /api/v1/predict → Defect classification inference

Environment Variables:
    MODEL_PATH      : Path to best_model_mvtec.pth (default: models/best_model_mvtec.pth)
    REVIEW_QUEUE_DIR: Path for anomaly image storage (default: data/review_queue)
    API_SECRET_KEY  : Authentication key — REQUIRED, no default

Usage:
    docker run -p 8000:8000 \
      -e API_SECRET_KEY=your_secret_key \
      mvtec-classifier-api:v1
"""

# ── Standard Library
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional

# ── Third Party
import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Header, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms


# ════════════════════════════════════════════════════════
# 1. LOGGING INFRASTRUCTURE
# ════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MVTec-API] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MVTecAPI")


# ════════════════════════════════════════════════════════
# 2. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str       = os.getenv("MODEL_PATH", "models/best_model_mvtec.pth")
REVIEW_QUEUE_DIR: str = os.getenv("REVIEW_QUEUE_DIR", "data/review_queue")
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Class indices must match training ImageFolder.class_to_idx
# ImageFolder assigns alphabetically: {'anomaly': 0, 'good': 1}
# Verified against training script: self.class_names = train_ds.classes
CLASS_NAMES = ["anomaly", "good"]

# Singleton store — model loaded once at startup
ml_state: Dict = {}

# ── API Key Authentication
# No default value — server refuses to boot if key is not configured
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
if not API_SECRET_KEY:
    raise EnvironmentError(
        "API_SECRET_KEY environment variable is not set. "
        "Provide it via: docker run -e API_SECRET_KEY=your_secret_key"
    )

API_KEY_NAME = "Authorization-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Validates the incoming API key against the configured secret.

    Returns the key on success. Raises 401 on mismatch.
    All unauthorized attempts are logged as security warnings.
    """
    if api_key != API_SECRET_KEY:
        logger.warning("SECURITY: Unauthorized access attempt intercepted.")
        raise HTTPException(status_code=401, detail="Invalid API Key. Access Denied.")
    return api_key


# ── Preprocessing Pipeline
# CRITICAL: must match val_transform in training script exactly.
# Any change here must be mirrored in the training script.
# Training val_transform: Resize((224,224)) → ToTensor → Normalize
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ════════════════════════════════════════════════════════
# 3. SERVER LIFESPAN — MODEL SINGLETON PATTERN
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager — controls model and infrastructure lifecycle.

    STARTUP:
        1. Mounts review queue directory for anomaly image storage
        2. Reconstructs ResNet18 architecture (weights=None — custom weights)
        3. Loads trained artifact into device memory
        4. Runs warmup inference to compile CUDA kernels

    SHUTDOWN:
        Clears model state and releases GPU memory.

    Why singleton pattern:
        Loading a model artifact on every request adds seconds of latency.
        One load at boot, millisecond inference for all subsequent requests.
    """
    logger.info("Booting Factory Vision API...")
    logger.info(f"Device: {DEVICE.type.upper()} | Model: {MODEL_PATH}")

    # ── Mount review queue directory for active learning pipeline
    os.makedirs(REVIEW_QUEUE_DIR, exist_ok=True)
    logger.info(f"Review queue mounted at: {REVIEW_QUEUE_DIR}")

    # ── Model reconstruction and artifact injection
    try:
        # weights=None — we load custom fine-tuned weights below, not ImageNet defaults
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval()
        ml_state["model"] = model
        logger.info("Model artifact loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
        raise RuntimeError(
            f"Server boot aborted — artifact not found at: {MODEL_PATH}"
        ) from e

    # ── Warmup inference — compiles CUDA kernels before first real request
    # Without warmup, the first inference call incurs a 3-5x latency spike
    logger.info("Running warmup inference...")
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.inference_mode():
        ml_state["model"](dummy)
    logger.info("Model warmed up. API ready to serve requests.")

    yield  # ── Server live

    # ── Shutdown cleanup
    logger.info("Shutting down. Releasing resources...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Shutdown complete.")


# ════════════════════════════════════════════════════════
# 4. API INSTANTIATION
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="Factory Quality Control API",
    description=(
        "Real-time CV microservice for automated defect detection on MVTec "
        "AD bottle images. Detects surface anomalies and routes detections "
        "to a review queue for active learning pipeline integration."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════
# 5. RESPONSE SCHEMAS
# ════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    """Inference result payload."""
    filename    : str
    prediction  : str    # "good" or "anomaly"
    confidence  : float  # softmax probability of predicted class [0, 1]
    latency_ms  : float  # end-to-end inference latency in milliseconds


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status      : str
    model_path  : str
    device      : str


# ════════════════════════════════════════════════════════
# 6. UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"])
async def root() -> dict:
    """API metadata — entry point for documentation discovery."""
    return {
        "api"    : "Factory Quality Control API",
        "version": "1.1.0",
        "docs"   : "/docs",
        "health" : "/health",
        "ready"  : "/ready",
        "predict": "/api/v1/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """
    Liveness check — confirms the API process is running.

    Used by Docker HEALTHCHECK and Kubernetes liveness probes.
    Returns 200 as long as the server process is alive.
    """
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse, tags=["Utility"])
async def ready() -> ReadyResponse:
    """
    Readiness check — confirms the model is loaded and inference is possible.

    Returns 503 if the model is not yet loaded.

    Liveness vs Readiness distinction:
        /health → Is the process alive? (restart if fails)
        /ready  → Is the model ready?  (stop routing traffic if fails)
    """
    if ml_state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    return ReadyResponse(
        status    ="ready",
        model_path=MODEL_PATH,
        device    =DEVICE.type.upper(),
    )


# ════════════════════════════════════════════════════════
# 7. INFERENCE HELPERS
# ════════════════════════════════════════════════════════

def _run_inference(
    model: nn.Module,
    tensor: torch.Tensor
) -> tuple:
    """
    Synchronous inference function — runs in thread pool via asyncio.to_thread.

    Separated from the async endpoint to enable proper thread offloading.
    PyTorch inference is CPU/GPU-bound — running it directly in async def
    blocks the event loop, preventing other requests from being served.

    Returns:
        (confidence: float, predicted_idx: int)
    """
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
    return float(confidence[0]), int(predicted_idx[0])


def _save_anomaly_image(img_bgr: np.ndarray, filename: str) -> str:
    """
    Persists anomaly image to the review queue directory.

    Part of the active learning loop — flagged images are reviewed by
    human operators, labeled, and fed back into the next training cycle.
    This closes the feedback loop between production inference and model
    improvement.

    Returns the full save path for logging.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_filename = filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(REVIEW_QUEUE_DIR, f"{timestamp}_{safe_filename}")
    cv2.imwrite(save_path, img_bgr)
    return save_path


# ════════════════════════════════════════════════════════
# 8. INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_bottle_quality(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key),
    content_length: int = Header(default=0, alias="Content-Length"),
) -> PredictionResponse:
    """
    Bottle defect classification endpoint.

    Accepts an image upload, classifies it as "good" or "anomaly",
    and routes anomaly detections to a persistent review queue for
    active learning pipeline integration.

    Pipeline:
        1. API key validation          → reject unauthorized requests
        2. Header size check           → early rejection of oversized payloads
        3. Actual bytes size check     → detect spoofed Content-Length headers
        4. MIME type validation        → reject non-image content types
        5. In-memory image decoding    → zero disk I/O via cv2.imdecode
        6. BGR → RGB conversion        → OpenCV reads BGR, PyTorch expects RGB
        7. Thread-offloaded inference  → asyncio.to_thread (non-blocking)
        8. Anomaly routing             → save flagged images to review queue
        9. Structured response         → Pydantic-validated JSON payload

    Args:
        file           : Multipart image upload (JPEG, PNG, WebP)
        api_key        : Authorization-API-Key header value
        content_length : Content-Length header for early size rejection

    Returns:
        PredictionResponse with filename, prediction, confidence, latency.

    Raises:
        401 : Invalid or missing API key
        400 : Invalid content type (non-image upload)
        413 : Payload exceeds 10MB limit
        422 : Valid image type but content cannot be decoded (corrupted)
        500 : Unexpected inference error
        503 : Model not loaded
    """

    # ── Model availability check
    model = ml_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Server may still be initializing."
        )

    # ── Step 1: Header-based size check (early rejection before reading bytes)
    # Content-Length=0 means header absent — actual bytes validated below
    if content_length > MAX_FILE_SIZE_BYTES:
        logger.warning(
            f"SECURITY: Payload header claims {content_length} bytes — "
            f"exceeds {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB limit."
        )
        raise HTTPException(status_code=413, detail="Payload Too Large.")

    # ── Step 2: MIME type validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type: '{file.content_type}'. "
                "Only image/jpeg, image/png, image/webp accepted."
            )
        )

    # ── Step 3: Read bytes + actual size validation (detects spoofed headers)
    image_bytes: bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        logger.warning(
            f"SECURITY: Spoofed Content-Length header detected. "
            f"Actual payload: {len(image_bytes)} bytes."
        )
        raise HTTPException(status_code=413, detail="Payload Too Large.")

    try:
        inference_start = time.perf_counter()

        # ── Step 4: In-memory image decoding (zero disk I/O)
        # np.frombuffer reads raw bytes as 1D array
        # cv2.imdecode decodes into H×W×3 numpy array in BGR format
        np_arr: np.ndarray = np.frombuffer(image_bytes, np.uint8)
        img_bgr: Optional[np.ndarray] = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ── Step 5: Content validation
        # img_bgr is None if content_type was spoofed or file is corrupted
        if img_bgr is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Image decoding failed. File may be corrupted, truncated, "
                    "or content type was incorrectly declared."
                )
            )

        # ── Step 6: BGR → RGB conversion
        # OpenCV reads images in BGR channel order.
        # PyTorch and torchvision transforms expect RGB.
        # Missing this conversion produces silently wrong predictions.
        img_rgb: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil: Image.Image = Image.fromarray(img_rgb)
        img_tensor: torch.Tensor = image_transforms(img_pil).unsqueeze(0).to(DEVICE)

        # ── Step 7: Thread-offloaded inference
        # asyncio.to_thread offloads CPU/GPU-bound inference to a thread pool.
        # Event loop stays free to accept other camera feed requests during compute.
        confidence_score, predicted_idx = await asyncio.to_thread(
            _run_inference, model, img_tensor
        )

        predicted_class = CLASS_NAMES[predicted_idx]
        latency_ms = round((time.perf_counter() - inference_start) * 1000, 2)

        # ── Step 8: Automated QA routing — active learning pipeline
        # Anomaly detections are persisted to the review queue directory.
        # Human operators label these images and feed them back into the
        # next training cycle — closing the production → improvement loop.
        if predicted_class == "anomaly":
            save_path = await asyncio.to_thread(
                _save_anomaly_image,
                img_bgr,
                file.filename or "unknown"
            )
            logger.warning(
                f"ANOMALY DETECTED | File: '{file.filename}' | "
                f"Confidence: {confidence_score:.4f} | "
                f"Routed to review queue: {save_path}"
            )

        logger.info(
            f"Inference complete | File: '{file.filename or 'unknown'}' | "
            f"Prediction: {predicted_class.upper()} | "
            f"Confidence: {confidence_score:.4f} | "
            f"Latency: {latency_ms}ms"
        )

        return PredictionResponse(
            filename   =file.filename or "unknown",
            prediction =predicted_class,
            confidence =round(confidence_score, 4),
            latency_ms =latency_ms,
        )

    except HTTPException:
        # Re-raise intentional HTTP errors — prevents them being caught
        # by the generic Exception handler and returned as 500
        raise

    except Exception as e:
        logger.error(
            f"Unexpected inference error on '{file.filename}': {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference. Check server logs."
        )

    finally:
        # Explicit memory release — prevents tensor accumulation in
        # long-running servers processing high-volume camera feeds
        try:
            del image_bytes, np_arr, img_bgr, img_rgb, img_pil, img_tensor
        except NameError:
            pass  # Variables may not exist if error occurred before assignment