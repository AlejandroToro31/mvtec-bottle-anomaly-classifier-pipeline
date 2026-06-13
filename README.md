# MVTec Bottle Anomaly Classifier API

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A production-deployed supervised binary classifier for real-time surface defect detection on MVTec AD bottle images. Detects structural anomalies from industrial camera feeds and automatically routes flagged images to a persistent review queue for active learning pipeline integration.

---

## System Architecture

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Classification Engine** | ResNet18, frozen backbone | Fine-tuned on MVTec AD bottle dataset |
| **Web Framework** | FastAPI + Uvicorn | ASGI, async request handling |
| **Inference** | asyncio.to_thread | Non-blocking CPU inference |
| **Image Decoding** | cv2.imdecode | Zero disk I/O — raw bytes decoded in RAM |
| **Authentication** | API Key header | Unauthorized requests rejected with 401 |
| **QA Routing** | cv2.imwrite | Anomaly images persisted for active learning |
| **Container** | python:3.10-slim | Non-root user, layer-cached builds, HEALTHCHECK |

**Dataset Engineering:**
MVTec AD is originally designed for unsupervised anomaly detection. This pipeline restructures it into a supervised binary classification problem by aggregating all defect sub-folders into a unified `anomaly` class — enabling direct comparison between supervised and unsupervised paradigms on identical data.

**Active Learning Loop:**
When the model detects an anomaly, the raw image is automatically saved to a mounted review queue volume. Human operators periodically label these images and feed them back into the next training cycle — closing the production → improvement feedback loop.

---

## Tech Stack

- **Deep Learning:** PyTorch 2.1, Torchvision
- **Web Server:** FastAPI 0.104, Uvicorn (with uvloop + httptools)
- **Computer Vision:** OpenCV (`opencv-python-headless`), NumPy, Pillow
- **DevOps:** Docker, python:3.10-slim base image

---

## API Endpoints

| Method | Endpoint | Auth Required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/` | ❌ | API metadata and endpoint discovery |
| `GET` | `/health` | ❌ | Liveness check — is the process running? |
| `GET` | `/ready` | ❌ | Readiness check — is the model loaded? |
| `POST` | `/api/v1/predict` | ✅ | Defect classification inference |

---

## Project Structure

```
mvtec-bottle-anomaly-classifier-pipeline/
├── app/
│   └── main.py                    # FastAPI inference endpoint
├── models/
│   └── best_model_mvtec.pth       # Model artifact (download separately)
├── tests/
│   ├── test_client.py             # Industrial camera feed simulator
│   └── assets/
│       ├── sample_defect.png      # Test anomaly image
│       └── sample_good.png        # Test nominal image
├── training/
│   └── Anomaly_Detector.ipynb     # ResNet18 training pipeline
├── data/
│   └── review_queue/              # Anomaly images routed here for human review
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/AlejandroToro31/mvtec-bottle-anomaly-classifier-pipeline.git
cd mvtec-bottle-anomaly-classifier-pipeline
```

### 2. Download the Model Artifact

The trained weights are stored externally to keep the repository lightweight.

1. Download `best_model_mvtec.pth` from: [Model Registry (Google Drive)](https://drive.google.com/file/d/1RvJ6Xt1OUKbKwuQTsZ6ynqUdbGNnIXQd/view?usp=sharing)
2. Place it inside the `models/` directory:

```
models/
└── best_model_mvtec.pth
```

### 3. Build the Container

```bash
docker build -t factory-aoi-api:v1 .
```

First build takes ~3-5 minutes. Subsequent code-only changes rebuild in ~2 seconds due to layer caching.

### 4. Run the Container

The API requires an `API_SECRET_KEY` environment variable — the server will refuse to boot without it. Because the microservice writes anomaly images to disk, you must also mount a local volume for the review queue.

```bash
docker run -p 8000:8000 \
  -e API_SECRET_KEY=your_secret_key \
  -v $(pwd)/data/review_queue:/workspace/data/review_queue \
  factory-aoi-api:v1
```

> **Note:** `$(pwd)` is Linux/Mac syntax. Windows users use `%cd%` (CMD) or `${PWD}` (PowerShell).

Verify the API is ready:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

---

## Swagger UI Testing

Navigate to **http://localhost:8000/docs** for the interactive Swagger UI.

**Authentication setup:**
1. Set your `API_SECRET_KEY` when starting the container (see above)
2. Click the **Authorize** button (top right of the Swagger UI)
3. Enter the same key in the `Authorization-API-Key` field
4. Click **Authorize** — all subsequent requests will include the key automatically

**Run inference:**
1. Open `POST /api/v1/predict`
2. Click **Try it out**
3. Upload an MVTec bottle image
4. Click **Execute**

---

## Industrial Camera Simulation — test_client.py

The repository includes `test_client.py` — a test module that simulates an industrial camera transmitting raw frame buffers to the API. It validates the full pipeline including authentication, inference, and anomaly routing.

**Test scenarios covered:**
- Valid defect image → expects `anomaly` + review queue write
- Valid nominal image → expects `good`
- Invalid file type → expects `400 Bad Request`
- Wrong API key → expects `401 Unauthorized`

**Setup — set the API key as environment variable:**
```bash
# Linux / Mac
export API_SECRET_KEY=your_secret_key

# Windows
set API_SECRET_KEY=your_secret_key
```

**Run both terminals simultaneously:**

Terminal 1 — Start the API server:
```bash
# Via Docker (recommended)
docker run -p 8000:8000 \
  -e API_SECRET_KEY=your_secret_key \
  -v $(pwd)/data/review_queue:/workspace/data/review_queue \
  factory-aoi-api:v1

# Or locally via Uvicorn
uvicorn app.main:app --reload
```

Terminal 2 — Run the camera simulator:
```bash
python tests/test_client.py
```

**Expected output:**
```
[TEST 1] Defect image inference — expects 'anomaly'...
  PASS ✓ | Prediction: ANOMALY | Confidence: 0.9341 | API latency: 31.2ms | Network: 45.8ms

[TEST 2] Nominal image inference — expects 'good'...
  PASS ✓ | Prediction: GOOD | Confidence: 0.9712 | API latency: 28.7ms | Network: 42.1ms

[TEST 3] Invalid file type (PDF) — expects 400 Bad Request...
  PASS ✓ | Expected 400, got 400

[TEST 4] Invalid API key — expects 401 Unauthorized...
  PASS ✓ | Expected 401, got 401

TEST SUITE COMPLETE: 5/5 passed
```

After the anomaly test, verify the image was saved:
```bash
ls data/review_queue/
```

---

## Example API Response

```json
{
  "filename": "bottle_defect_cam04.jpg",
  "prediction": "anomaly",
  "confidence": 0.9341,
  "latency_ms": 31.2
}
```

---

## Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `API_SECRET_KEY` | None | ✅ **Yes** | Authentication key — server refuses to boot without it |
| `MODEL_PATH` | `models/best_model_mvtec.pth` | ❌ | Path to trained model artifact |
| `REVIEW_QUEUE_DIR` | `data/review_queue` | ❌ | Directory for anomaly image persistence |

---

## Training Pipeline

To retrain on a custom dataset, open `training/Anomaly_Detector.ipynb` in JupyterLab:

```bash
pip install -r requirements-dev.txt
jupyter lab
```

The notebook handles:
1. MVTec AD dataset download and binary restructuring
2. WeightedRandomSampler for class imbalance
3. ResNet18 fine-tuning with AMP (FP16) training
4. Recall-primary checkpointing (defect recall > accuracy)
5. AUROC evaluation

---

## Edge Case Behavior — Out-of-Distribution Input

This classifier operates strictly within the structural constraints of the MVTec bottle spatial manifold. The model was trained exclusively on dark-field industrial lighting with fixed camera geometry.

Images that fall outside this distribution will produce unreliable predictions:
- Non-bottle objects → likely classified as `anomaly`
- Severe lighting changes → degraded confidence scores
- Non-standard camera angles → feature distribution shift

**This system is designed strictly for fixed-camera, controlled-lighting industrial environments.**

---

## Docker Notes

**Volume mounting is required** — the review queue directory must be mounted to persist anomaly images beyond the container lifecycle. Without the volume mount, images are written inside the container and lost on restart.

**Non-root execution:** The container runs as `api_user` — principle of least privilege.

**Health monitoring:** Docker's native `HEALTHCHECK` polls `/health` every 30 seconds with a 60-second startup grace period for model loading.