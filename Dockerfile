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

# Expose the API port
EXPOSE 8000

# Execute Uvicorn, pointing it into the app directory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]