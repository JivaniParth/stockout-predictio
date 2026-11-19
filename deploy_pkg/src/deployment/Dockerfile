# Multi-stage lightweight and secure Dockerfile for Cloud-Native HSCM Inference Microservice
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies with minimal attack surface
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained model artifacts
COPY src/ /app/src/
COPY results/models/ /app/results/models/
COPY SCMS_Delivery_History_Dataset.csv /app/

# Create unprivileged system user for container execution (Least Privilege principle)
RUN adduser --disabled-password --gecos "" --home /app --no-create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose microservice inference port
EXPOSE 8000

# Healthcheck for Kubernetes / Docker Swarm container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
