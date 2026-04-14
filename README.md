# Industrial Machine Vision Microservice (MVTec Anomaly Detection)

An end-to-end Machine Learning Operations (MLOps) pipeline and asynchronous REST API designed for real-time visual inspection of manufacturing defects. 

## Architecture Overview

This system utilizes a Convolutional Neural Network (ResNet-18) fine-tuned on the MVTec AD dataset to detect structural anomalies. The model is wrapped in a highly concurrent FastAPI server, containerized via Docker, and engineered specifically for physical factory deployment.

**Key Engineering Features:**
* **Industrial ETL Bridge:** Utilizes `opencv-python-headless` to extract and decode raw byte streams (`cv2.imdecode`), simulating the exact data flow of physical industrial cameras.
* **Automated QA Routing (Disk I/O):** Features conditional persistence logic. Anomalous matrices are physically routed and saved to a dedicated mounted volume for human review and future model retraining.
* **Execution Telemetry:** Built-in performance profiling that returns exact millisecond latency metrics for CPU/GPU forward passes.
* **DevSecOps Secured:** Protected by memory-exhaustion payload limiters (10MB max) and cryptographic API Key locks.

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Web Server:** FastAPI, Uvicorn, Pydantic
* **Data Processing:** OpenCV, NumPy
* **DevOps/MLOps:** Docker

## Quick Start (Docker Deployment)

The API is fully containerized for immutable deployment. Because the microservice writes defective images to the disk, you **must** mount a local volume during execution.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AlejandroToro31/mvtec-bottle-anomaly-classifier-pipeline.git](https://github.com/AlejandroToro31/mvtec-bottle-anomaly-classifier-pipeline.git)
   cd mvtec-bottle-anomaly-classifier-pipeline
    ```

2. **Build the immutable image:**
```Bash
docker build -t factory-aoi-api:v1 .
```

3. **Deploy the container (Port Binding & Volume Mounting):**
```Bash
docker run -p 8000:8000 -v $(pwd)/data/review_queue:/app/data/review_queue factory-aoi-api:v1
```

# Hardware Simulation & Integration Testing
This repository includes a test_client.py module designed to simulate an industrial camera streaming raw byte arrays to the FastAPI microservice.

## Execution Protocol:
To verify the end-to-end ETL pipeline and Automated QA routing, open two separate terminal instances:

## Terminal 1 (Boot the Inference Node):
(Either run the Docker container above, or boot locally via Uvicorn)

```Bash
uvicorn app.main:app --reload
```
## Terminal 2 (Execute the Camera Simulator):

```Bash
python tests/test_client.py
```
## Expected Telemetry Output:
The client will transmit the test payload (tests/assets/sample_defect.png). The API will decode the byte stream via OpenCV, execute the PyTorch forward pass, and physically route the anomalous matrix to the data/review_queue volume. Execution latency and network ping will be returned in the terminal.

# Live API Testing (Swagger UI)
Once the container is actively listening, navigate to http://localhost:8000/docs to access the interactive testing environment.

1. To test the endpoint, use the temporary evaluation key: mvtec_dev_key

2. Click the green Authorize button in the top right corner.

3. Paste the evaluation key into the Authorization-API-Key field.

4. Open the POST /predict/ dropdown, click Try it out, upload any sample image of an MVTec bottle, and click Execute.

# Edge Case Testing (Domain Shift)
This API operates strictly within the structural constraints of the MVTec spatial manifold. Images lacking industrial dark-field lighting or presenting severe Out-of-Distribution (OOD) geometry will trigger feature collapse, defaulting to an anomaly prediction. The system is designed strictly for fixed-camera, controlled-lighting environments.