import os
import time
import requests

# --- SYSTEM CONFIGURATION ---
# The local address where your FastAPI microservice is listening
API_URL = "http://127.0.0.1:8000/predict/"

# The exact security token we hardcoded into the API
HEADERS = {
    "Authorization-API-Key": "mvtec_dev_key"
}

# Point this to a real image from your MVTec dataset
# Choose an anomaly image to test the Disk I/O routing!
TEST_IMAGE_PATH = "tests/assets/sample_defect.png" 

def simulate_camera_stream():
    """Simulates an industrial camera sending a frame buffer to the API."""
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"SYSTEM ERROR: Cannot find test image at {TEST_IMAGE_PATH}")
        return

    print(f"INITIALIZING CAMERA MOCK: Preparing to transmit '{TEST_IMAGE_PATH}'...")
    
    # We open the image in 'rb' (Read Binary) mode. 
    # This extracts the raw byte stream just like a physical camera.
    with open(TEST_IMAGE_PATH, "rb") as image_file:
        files = {"file": (os.path.basename(TEST_IMAGE_PATH), image_file, "image/png")}
        
        print("TRANSMITTING PAYLOAD...")
        start_time = time.time()
        
        # Fire the HTTP POST request at the FastAPI microservice
        response = requests.post(API_URL, headers=HEADERS, files=files)
        
        network_latency = (time.time() - start_time) * 1000

    # --- TELEMETRY ANALYSIS ---
    if response.status_code == 200:
        json_payload = response.json()
        print("\n=== INFERENCE SUCCESS ===")
        print(f"Prediction:   {json_payload['prediction'].upper()}")
        print(f"Confidence:   {json_payload['confidence']}")
        print(f"API Latency:  {json_payload['latency_ms']} ms")
        print(f"Network Ping: {network_latency:.2f} ms")
    else:
        print("\n=== INFERENCE FAILED ===")
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Error Details: {response.text}")

if __name__ == "__main__":
    simulate_camera_stream()