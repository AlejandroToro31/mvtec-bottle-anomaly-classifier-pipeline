# Base Image
FROM mirror.gcr.io/library/python:3.10-slim

# System Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# DevSecOps: Create the non-root system user
# -m ensures the ~/.cache directory exists for PyTorch
RUN useradd -m -r api_user

# Working Directory Setup
# We create the folder and give it to the user immediately
WORKDIR /workspace
RUN chown api_user /workspace

# Layer Caching: Install dependencies as root (to prevent permission errors)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Inject Microservice Code
# We copy and change ownership in a single layer to prevent image bloat
COPY --chown=api_user:api_user app/ app/
COPY --chown=api_user:api_user models/ models/

# 7. Drop Privileges: Switch to the secure non-root user
USER api_user

# 8. Expose the API Port
EXPOSE 8000

# 9. Execute the Inference Engine
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]