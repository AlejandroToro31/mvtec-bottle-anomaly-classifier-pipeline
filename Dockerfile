# ==============================================================
# MVTec Anomaly Classifier API — Production Dockerfile
# ==============================================================
# Supervised binary defect detection microservice.
# ResNet18 backbone, CPU inference, non-root execution.
#
# Build:
#   docker build -t mvtec-classifier-api:v1 .
#
# Run:
#   docker run -p 8000:8000 \
#     -e API_SECRET_KEY=your_secret_key \
#     mvtec-classifier-api:v1
#
# Run with overrides:
#   docker run -p 8000:8000 \
#     -e API_SECRET_KEY=your_secret_key \
#     -e MODEL_PATH=models/best_model_mvtec.pth \
#     mvtec-classifier-api:v1
# ==============================================================

# ── Base Image
# GCR mirror: faster than Docker Hub, avoids rate limits
# python:3.10-slim: minimal Debian — significantly smaller than full Python image
FROM mirror.gcr.io/library/python:3.10-slim

# ── Image Metadata
LABEL version="1.1.0"
LABEL description="MVTec Anomaly Classifier API — ResNet18 binary defect detection"

# ── Python Runtime Configuration
# PYTHONDONTWRITEBYTECODE=1 : Prevents .pyc bytecode files
# PYTHONUNBUFFERED=1        : Forces real-time stdout/stderr flushing for Docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Application Configuration
# System defaults — override at runtime via docker run -e
# API_SECRET_KEY has no default — must be provided at runtime
ENV MODEL_PATH=models/best_model_mvtec.pth \
    REVIEW_QUEUE_DIR=data/review_queue

# ── OS-Level Dependencies
# curl: required for Docker HEALTHCHECK instruction
# --no-install-recommends: skips suggested packages, minimizes image footprint
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security: Non-Root User
# Running as root inside a container exposes the host if the container is compromised.
# groupadd: explicit group ensures deterministic GID — avoids unpredictable assignments
# -r : system account (no login shell, no cron)
# -m : creates home directory (~/.cache) required by PyTorch weight caching
RUN groupadd -r api_user && useradd -m -r -g api_user api_user

# ── Working Directory
WORKDIR /workspace

# Set workspace ownership immediately — api_user needs write access
# for review queue directory creation and YOLO/PyTorch cache
RUN chown api_user:api_user /workspace

# ── Layer Caching Strategy
# Install dependencies BEFORE copying application code.
# If only main.py changes, Docker reuses the cached pip install layer.
# Full pip install only re-runs when requirements.txt changes.
# Build time: ~3-5 minutes (cold) vs ~2 seconds (cached code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code
# --chown sets file ownership in a single COPY layer.
# Avoids a separate RUN chown command which would add an extra image layer.
COPY --chown=api_user:api_user app/ app/
COPY --chown=api_user:api_user models/ models/

# ── Drop Privileges
# Switch to non-root user for all subsequent operations including CMD.
# All runtime processes run as api_user — principle of least privilege.
USER api_user

# ── Port Declaration
# Documents that the container listens on port 8000.
# Does not publish the port — use -p 8000:8000 at docker run.
EXPOSE 8000

# ── Health Monitoring
# Docker automatically monitors container health using this instruction.
# --start-period=60s: grace period for model loading before checks begin
# Connects directly to the /health liveness endpoint in main.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint
# --host 0.0.0.0 : bind to all interfaces — required for Docker port mapping
#                  (127.0.0.1 would be unreachable from outside the container)
# --workers 4    : 4 independent processes for parallel request handling
#                  Note: each worker loads its own model instance into memory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
