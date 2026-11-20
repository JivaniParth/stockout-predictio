"""
Architecture View (Tab 4) for Streamlit HSCM Platform.
Cloud-Native Microservices Architecture, Containerization, and Health Telemetry.
"""

import urllib.request
import json
import streamlit as st
from src.ui.components.primitives import render_alert, render_badge, render_metric_card


def check_fastapi_health() -> dict:
    """Checks liveness of the background FastAPI microservice."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health", headers={"User-Agent": "StreamlitUI"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            if response.status == 200:
                return json.loads(response.read().decode())
    except Exception:
        pass
    return {"status": "UNAVAILABLE", "error": "Microservice not running on port 8000"}


def render_architecture_tab():
    """Renders the Cloud Architecture & Microservice Telemetry View."""
    st.subheader("Cloud-Native Microservices Architecture & Containerization")
    st.markdown(
        "Demonstrating containerized, REST-accessible microservice operationalization "
        "designed for cloud and edge healthcare supply chain management."
    )

    # Live Health Probe Status Card
    st.markdown("#### Live FastAPI Microservice Health Probe (`/health`)")
    health_data = check_fastapi_health()

    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        if health_data.get("status") == "HEALTHY":
            st.success("**Microservice Status: HEALTHY** (Port 8000 Active)")
        else:
            st.warning("**Microservice Status: STANDALONE** (In-Memory Engine Active)")

    with h_col2:
        render_metric_card(
            label="Inference Endpoint",
            value="FastAPI ASGI",
            meta="http://127.0.0.1:8000",
            tooltip="REST API serving real-time predictions",
        )

    with h_col3:
        uptime = health_data.get("uptime_seconds", "N/A")
        uptime_str = f"{uptime:.1f}s" if isinstance(uptime, (int, float)) else str(uptime)
        render_metric_card(
            label="Microservice Uptime",
            value=uptime_str,
            meta="Liveness & Readiness probe",
            tooltip="Service process uptime",
        )

    st.markdown("---")
    c_col1, c_col2 = st.columns([1.2, 1])

    with c_col1:
        st.markdown("#### End-to-End System Pipeline")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["SCMS Pharmaceutical Dataset (N=10,324)"] --> B["HSCM Preprocessor (Regex + Median Impute)"]
                B --> C["Strict Chronological Split (Train <=2013 | Val 2014 | Test 2015)"]
                C --> D["Deep Learning Suite (Hybrid LSTM-Transformer / 1D-CNN)"]
                D --> E["Threshold Calibration on 2014 Validation (τ*)"]
                E --> F["FastAPI Production REST Microservice (:8000)"]
                F --> G["GET /health (Kubernetes Readiness)"]
                F --> H["POST /predict (Real-time Latency Telemetry)"]
                F --> I["POST /predict/batch (Inventory Sync)"]
                F --> J["GET /qos_metrics (Observability Telemetry)"]
                F --> K["Streamlit Production Dashboard UI (:8501)"]
            ```
            """
        )

    with c_col2:
        st.markdown("#### Container Specification (Dockerfile)")
        st.code(
            """
# Multi-stage lightweight Dockerfile for Cloud-Native HSCM Inference Microservice
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \\
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY results/models/ /app/results/models/
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
            """,
            language="dockerfile",
        )

        st.markdown("#### Execution Instructions")
        st.code(
            """
# 1. Run Streamlit UI
streamlit run app_streamlit.py

# 2. Run Cloud FastAPI REST Service
python run_api.py

# 3. Test REST Endpoints
python test_api.py
            """,
            language="bash",
        )
