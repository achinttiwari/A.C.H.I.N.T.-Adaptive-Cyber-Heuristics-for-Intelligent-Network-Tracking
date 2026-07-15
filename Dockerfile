# =============================================================================
# Dockerfile — Containerised IDS Pipeline (CPU + optional GPU)
# =============================================================================
# AI-Driven Network Traffic Anomaly Detection
#
# Build (CPU):
#   docker build -t ids-pipeline .
#
# Build (GPU — requires NVIDIA Container Toolkit):
#   docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.16.1-gpu \
#       -t ids-pipeline-gpu .
#
# Run (CPU):
#   docker run --rm \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/artifacts:/app/artifacts \
#       ids-pipeline
#
# Run (GPU):
#   docker run --rm --gpus all \
#       -v $(pwd)/data:/app/data \
#       -v $(pwd)/artifacts:/app/artifacts \
#       ids-pipeline-gpu
#
# AWS EC2 (push to ECR then use in SageMaker or run on EC2):
#   aws ecr create-repository --repository-name ids-pipeline
#   docker tag ids-pipeline <account-id>.dkr.ecr.<region>.amazonaws.com/ids-pipeline
#   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/ids-pipeline
# =============================================================================

# Switchable base image — defaults to official TensorFlow CPU image
ARG BASE_IMAGE=tensorflow/tensorflow:2.16.1

FROM ${BASE_IMAGE}

LABEL maintainer="University Cybersecurity Research Group"
LABEL description="AI-Driven Network Traffic Anomaly Detection Pipeline"
LABEL version="1.0.0"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
#   Copy requirements first so Docker layer caching avoids reinstallation
#   when only source files change.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
COPY pipeline/ ./pipeline/
COPY main.py   .

# ---------------------------------------------------------------------------
# Volume mount points (declared but not created here — user mounts at runtime)
#   /app/data/raw/          ← CICIDS2017 CSV files
#   /app/artifacts/         ← Trained models, reports, and plots written here
# ---------------------------------------------------------------------------
VOLUME ["/app/data", "/app/artifacts"]

# ---------------------------------------------------------------------------
# Environment variables (can be overridden with -e at runtime)
# ---------------------------------------------------------------------------
ENV DATA_DIR=/app/data/raw
ENV PROCESSED_DIR=/app/data/processed
ENV ARTIFACT_DIR=/app/artifacts
ENV PYTHONUNBUFFERED=1        
# PYTHONUNBUFFERED=1 ensures Python print() and log output reaches Docker logs
# immediately rather than being buffered — important for monitoring long runs.

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
CMD ["python", "main.py"]
