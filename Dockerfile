# Multi-Stage Hardened & Non-Root Dockerfile for HSCM Prediction Platform
# Stage 1: Build & Dependencies
FROM python:3.11.9-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /install/requirements.txt
RUN pip install --prefix=/install/deps -r requirements.txt

# Stage 2: Final Minimal Runtime Environment
FROM python:3.11.9-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PATH="/install/deps/bin:$PATH" \
    PYTHONPATH="/install/deps/lib/python3.11/site-packages:/app:$PYTHONPATH"

WORKDIR /app

# Install runtime curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder stage
COPY --from=builder /install/deps /install/deps

# Copy application code, configs, and artifacts
COPY . /app/

# Create dedicated non-root user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -d /app -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /app
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
