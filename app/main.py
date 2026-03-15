import io
import os
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FACTORY-API] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("FactoryAPI")

# --- GLOBAL STATE ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_mvtec.pth")
CLASS_NAMES = ['anomaly', 'good']
ml_state = {} # UPGRADE: Replaces 'global model' for safer memory management if we ever scale to multi-processing

# --- THE PREPROCESSING PIPELINE ---
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- SERVER LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes exactly once when the server boots up."""
    logger.info("Booting Factory Vision API...")
    logger.info(f"Allocating model to: {DEVICE.type.upper()}")
    
    try:
        # Reconstruct Architecture 
        # UPGRADE: weights=None prevents redundant internet downloads on boot
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
    
    # Cleanup 
    logger.info("Shutting down. Clearing VRAM...")
    ml_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- API INSTANTIATION ---
app = FastAPI(
    title="Factory Quality Control API",
    description="Real-time Computer Vision endpoint for bottle defect detection.",
    version="1.0.0",
    lifespan=lifespan
)

# --- RESPONSE SCHEMA ---
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float

# --- THE ENDPOINT ---
@app.post("/predict/", response_model=PredictionResponse)
async def predict_bottle_quality(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        img_tensor = image_transforms(img).unsqueeze(0).to(DEVICE)
        model = ml_state["model"] # UPGRADE: Fetching model safely from state dictionary
        
        with torch.no_grad():
            logits = model(img_tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()

        logger.info(f"Evaluated '{file.filename}': {predicted_class.upper()} ({confidence_score:.4f})")

        return PredictionResponse(
            filename=file.filename,
            prediction=predicted_class,
            confidence=round(confidence_score, 4)
        )

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Corrupted image file.")
    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")