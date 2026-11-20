"""
Launcher script to run the Cloud-Native HSCM Inference Microservice.
Exposes Swagger interactive UI at: http://localhost:8000/docs
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Cloud-Native HSCM Stockout Prediction Microservice...")
    print("Interactive Swagger UI: http://localhost:8000/docs")
    print("Health Status Probe:   http://localhost:8000/health")
    print("Cloud QoS Metrics:      http://localhost:8000/qos_metrics")
    print("=" * 70)
    uvicorn.run("src.deployment.app:app", host="127.0.0.1", port=8000, reload=False)
