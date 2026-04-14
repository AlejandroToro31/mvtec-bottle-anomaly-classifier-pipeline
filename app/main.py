import io
import os
import time
import logging
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Header
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [-MVTecFACTORY-API] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MVtecAPI")

# --- GLOBAL STATE & I/O INFRASTRUCTURE ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_mvtec.pth")
REVIEW_QUEUE_DIR = os.getenv("REVIEW_QUEUE_DIR", "data/review_queue")
CLASS_NAMES = ['anomaly', 'good']
ml_state = {}

API_KEY_NAME = "Authorization-API-Key"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
VALID_API_KEY = os.getenv("API_SECRET_KEY", "mvtec_dev_key")

def verify_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Validates the incoming cryptographic token to prevent unauthorized inference requests."""
    if api_key_header != VALID_API_KEY:
        logger.warning("SECURITY ALERT: Unauthorized access attempt intercepted.")
        raise HTTPException(status_code=401, detail="Invalid API Key. Access Denied.")
    return api_key_header

# --- THE PREPROCESSING PIPELINE ---
image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- SERVER LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes initial hardware allocation and directory mounting on server boot."""
    logger.info("Booting Factory Vision API...")
    logger.info(f"Allocating model to: {DEVICE.type.upper()}")
    
    # Mount Disk I/O Directory for Automated Labeling Pipeline
    os.makedirs(REVIEW_QUEUE_DIR, exist_ok=True)
    logger.info(f"Disk I/O volume mounted at: {REVIEW_QUEUE_DIR}")
    
    try:
        # Reconstruct Architecture in VRAM
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2) 
        
        # Inject Trained Artifact
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        model.eval() 
        
        ml_state["model"] = model
        logger.info("Neural Network successfully locked in VRAM.")
    except Exception as e:
        logger.critical(f"Failed to load model artifact: {e}")
        raise RuntimeError("Server boot aborted due to missing artifact.")
        
    yield 
    
    logger.info("Shutting down. Clearing VRAM...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Factory Quality Control API",
    description="Real-time Computer Vision microservice for automated defect detection.",
    version="1.1.0",
    lifespan=lifespan
)

# --- RESPONSE SCHEMA ---
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    latency_ms: float

# --- THE INFERENCE ENDPOINT ---
@app.post("/predict/", response_model=PredictionResponse)
async def predict_bottle_quality(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key),
    content_length: int = Header(0, alias="Content-Length")
    ):
    
    # Pre-computation Payload Verification
    MAX_FILE_SIZE_BYTES = 10485760 
    if content_length > MAX_FILE_SIZE_BYTES:
        logger.warning(f"SECURITY ALERT: Payload size {content_length} bytes exceeds hardware limit.")
        raise HTTPException(status_code=413, detail="Payload Too Large.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Image required.")

    try:
        # System Telemetry: Initialize execution timer
        inference_start = time.perf_counter()

        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
             logger.warning("SECURITY ALERT: Spoofed header detected.")
             raise HTTPException(status_code=413, detail="Payload Too Large.")

        # ==========================================
        # UPGRADE 1: OpenCV ETL Bridge
        # Simulating industrial camera raw byte stream decoding.
        # cv2.imdecode parses the byte array directly into a BGR matrix.
        # ==========================================
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            raise UnidentifiedImageError("OpenCV failed to decode byte stream.")

        # Transform BGR (OpenCV standard) to RGB (PyTorch standard)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        img_tensor = image_transforms(img_pil).unsqueeze(0).to(DEVICE)
        model = ml_state["model"] 
        
        # GPU/CPU Forward Pass
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

        # ==========================================
        # UPGRADE 2: Automated QA Routing (Disk I/O)
        # If the forward pass detects an anomaly, write the raw BGR matrix 
        # to the physical disk volume for human review and future training.
        # ==========================================
        if predicted_class == 'anomaly':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(REVIEW_QUEUE_DIR, f"{timestamp}_{file.filename}")
            cv2.imwrite(save_path, img_cv)
            logger.warning(f"ANOMALY DETECTED: Image matrix routed to persistent storage -> {save_path}")

        # System Telemetry: Halt execution timer
        inference_end = time.perf_counter()
        latency_ms = (inference_end - inference_start) * 1000

        logger.info(f"Execution complete: {file.filename} | {predicted_class.upper()} ({confidence_score:.4f}) | Latency: {latency_ms:.2f}ms")

        return PredictionResponse(
            filename=file.filename,
            prediction=predicted_class,
            confidence=round(confidence_score, 4),
            latency_ms=round(latency_ms, 2)
        )

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted image matrix.")
    except Exception as e:
        logger.error(f"Hardware/Execution Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during execution.")