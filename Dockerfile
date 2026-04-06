# Base Image
FROM mirror.gcr.io/library/python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific microservice folders
COPY app/ app/
COPY models/ models/

# --- Security upgrade: Non-Root Execution ---
# Create a dummy user named 'api_user' with no admin rights
RUN useradd -m -r api_user 
# Give the dummy user ownership of the workspace folder so it can read the model
RUN chown -R api_user /workspace
# Tell Docker to switch to this dummy user before starting the server
USER api_user

# Expose the API port
EXPOSE 8000

# Execute Uvicorn, pointing it into the app directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]