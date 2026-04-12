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
    ```bash
    docker build -t factory-aoi-api:v1 .

3. **Deploy the container (Port Binding 8000):**
    ```bash
    docker run -p 8000:8000 factory-aoi-api:v1

## API Usage & Telemetry
Once the container is actively listening, navigate to the built-in Swagger UI to test the endpoints:
http://localhost:8000/docs

## Live API Testing (DevSecOps Secured)

This inference endpoint is strictly protected by memory-exhaustion payload limiters (10MB max) and an API Key lock to prevent unauthorized VRAM consumption.

To test the model's predictions, please use the following temporary evaluation key:
**`mvtec_dev_key`**

### Option A: The Visual Web Interface (Recommended)
FastAPI provides a built-in interactive testing environment.

1. Run the Docker container and navigate to `http://localhost:8000/docs`.
2. Click the green **Authorize** button in the top right corner.
3. Paste the evaluation key into the `Authorization-API-Key` field and click Authorize.
4. Open the `POST /predict/` dropdown, click **Try it out**, upload any image of an ant or bee, and click **Execute** to see the real-time classification.

### Option B: The Terminal
If you prefer to bypass the UI and test the raw JSON response and header validation directly:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Authorization-API-Key: mvtec_dev_key' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_test_image.jpg'
```

# Research & Development (Local Environment)
To reproduce the training metrics or experiment with the model architecture, you must install the heavier research dependencies.

1. Clone the repository and initialize a virtual environment (Conda recommended).

2. Install the isolated development requirements:

```bash
pip install -r requirements-dev.txt
```

3. Launch the Jupyter environment to access research/Anomaly_Detector.ipynb.

Note: The Jupyter notebook contains an automated pipeline that will automatically download and extract the MVTec dataset if it is not found locally.

## Edge Case Testing (Domain Shift)
This API has been rigorously tested against Out-of-Distribution (OOD) data. Images lacking the specific spatial manifold and industrial dark-field lighting of the MVTec dataset will trigger feature collapse, defaulting to an anomaly prediction due to Softmax overconfidence. The system is designed strictly for fixed-camera, controlled-lighting environments.