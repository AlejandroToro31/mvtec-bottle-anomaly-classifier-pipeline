# Automated Quality Control API (MVTec Anomaly Detection)

An end-to-end Machine Learning pipeline and asynchronous REST API for real-time visual inspection of manufacturing defects. 

## Architecture Overview
This system utilizes a Convolutional Neural Network (ResNet18) fine-tuned on the MVTec AD dataset to detect structural anomalies in glass bottles. The inference engine is wrapped in a highly concurrent FastAPI server and fully containerized via Docker for immutable, cloud-ready deployment.

**Key Engineering Features:**
* **Model:** ResNet18 (PyTorch) adapted for binary classification (Anomaly vs. Good).
* **Inference Backend:** Optimized for CPU batch-size-1 inference to minimize cloud compute costs and cold-start latency.
* **API Framework:** Asynchronous routing via FastAPI and Uvicorn.
* **Infrastructure:** Containerized with Docker, featuring failover configurations for registry network isolation.

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Web Server:** FastAPI, Uvicorn, Pydantic
* **Data Processing:** Pillow (PIL)
* **DevOps/MLOps:** Docker

## Quick Start (Docker Deployment)

The fastest way to run this API is via the pre-configured Docker container. 

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AlejandroToro31/mvtec-bottle-anomaly-classifier-pipeline.git](https://github.com/AlejandroToro31/mvtec-bottle-anomaly-classifier-pipeline.git)
   cd mvtec-bottle-anomaly-classifier-pipeline

2. **Build the immutable image:**
    docker build -t factory-aoi-api:v1 .

3. **Deploy the container (Port Binding 8000):**
    docker run -p 8000:8000 factory-aoi-api:v1

## API Usage & Telemetry
Once the container is actively listening, navigate to the built-in Swagger UI to test the endpoints:
http://localhost:8000/docs

Endpoint: POST /predict/
Accepts multipart/form-data image uploads and returns a strict JSON payload containing the prediction and Softmax confidence score.

Sample Response:

    {
    "filename": "003.png",
    "prediction": "anomaly",
    "confidence": 0.7948
    }

## Edge Case Testing (Domain Shift)
This API has been rigorously tested against Out-of-Distribution (OOD) data. Images lacking the specific spatial manifold and industrial dark-field lighting of the MVTec dataset will trigger feature collapse, defaulting to an anomaly prediction due to Softmax overconfidence. The system is designed strictly for fixed-camera, controlled-lighting environments.