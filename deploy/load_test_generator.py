"""
Standalone Load & Stress Test Generator with Unique Dynamic Payloads.
Tests deployed prediction endpoint (Cloud Run, local container, or microservice)
with guaranteed non-repeating payloads to defeat caching mechanisms and measure
true cold-inference latency distributions (P50, P90, P95, P99).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import uuid
import random
import argparse
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoadGenerator")

COUNTRIES = [
    "South Africa", "Nigeria", "Kenya", "Uganda", "Tanzania",
    "Mozambique", "Zimbabwe", "Zambia", "Congo, DRC", "Ethiopia",
    "Rwanda", "Haiti", "Burundi", "Ghana", "South Sudan",
]
PRODUCT_GROUPS = ["ARV", "HRDT", "ANTM", "ACT"]
SUB_CLASSIFICATIONS = ["Adult", "Pediatric", "HIV test", "Malaria"]
SHIPMENT_MODES = ["Air", "Truck", "Air Charter", "Ocean"]
FULFILL_VIAS = ["Direct Drop", "From RDC"]
INCO_TERMS = ["CIP", "EXW", "DDP", "FCA"]
VENDORS = [
    "Aurobindo Pharma Limited", "Mylan Laboratories GmbH",
    "Hetero Labs Limited", "Cipla Limited", "Ranbaxy Laboratories",
]


def generate_unique_shipment_payload() -> Dict[str, Any]:
    """Generates a randomized, statistically valid shipment record with high entropy."""
    po_days_ago = random.randint(180, 500)
    lead_time_days = random.randint(30, 240)
    base_date = datetime(2015, 1, 1) + timedelta(days=random.randint(0, 365))

    po_date = (base_date - timedelta(days=lead_time_days)).strftime("%Y-%m-%d")
    sched_date = base_date.strftime("%Y-%m-%d")

    qty = round(random.uniform(500.0, 300000.0), 1)
    pack_price = round(random.uniform(1.5, 65.0), 2)
    units_per_pack = random.choice([30, 60, 100, 120])
    unit_price = round(pack_price / units_per_pack, 4)
    line_value = round(qty * unit_price, 2)
    weight_kg = round(qty * random.uniform(0.02, 0.15), 1)
    freight = round(weight_kg * random.uniform(2.5, 8.0), 2)
    insurance = round(line_value * 0.0015, 2)

    return {
        "Country": random.choice(COUNTRIES),
        "Scheduled_Delivery_Date": sched_date,
        "PO_Sent_to_Vendor_Date": po_date,
        "Product_Group": random.choice(PRODUCT_GROUPS),
        "Sub_Classification": random.choice(SUB_CLASSIFICATIONS),
        "Shipment_Mode": random.choice(SHIPMENT_MODES),
        "Fulfill_Via": random.choice(FULFILL_VIAS),
        "Vendor_INCO_Term": random.choice(INCO_TERMS),
        "Vendor": random.choice(VENDORS),
        "Manufacturing_Site": f"{random.choice(VENDORS)} Unit {random.randint(1, 5)}",
        "First_Line_Designation": random.choice(["Yes", "No"]),
        "Line_Item_Quantity": qty,
        "Line_Item_Value": line_value,
        "Pack_Price": pack_price,
        "Unit_Price": unit_price,
        "Unit_of_Measure_Per_Pack": float(units_per_pack),
        "Weight_Kilograms": str(weight_kg),
        "Freight_Cost_USD": str(freight),
        "Line_Item_Insurance_USD": insurance,
        "_request_uuid": str(uuid.uuid4()), # Explicit entropy token defeating L2 cache
    }


def send_single_prediction_request(endpoint: str, timeout_sec: float = 10.0) -> Dict[str, Any]:
    """Sends a single POST request with unique payload and measures round-trip latency."""
    payload = generate_unique_shipment_payload()
    t0 = time.perf_counter()
    status_code = 0
    success = False
    error_msg = None
    pred_prob = None

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout_sec)
        status_code = resp.status_code
        if status_code == 200:
            success = True
            body = resp.json()
            pred_prob = body.get("delay_probability", None)
        else:
            error_msg = f"HTTP {status_code}: {resp.text[:100]}"
    except Exception as e:
        error_msg = str(e)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "status_code": status_code,
        "success": success,
        "error": error_msg,
        "pred_prob": pred_prob,
    }


def run_load_test(
    endpoint: str,
    n_requests: int = 100,
    concurrency: int = 5,
    output_json: str = "reports/qos_load_test.json",
) -> Dict[str, Any]:
    """Executes parallel load test against target endpoint using ThreadPoolExecutor."""
    logger.info(f"Initiating Load Test -> Endpoint: {endpoint} | Total Requests: {n_requests} | Concurrency: {concurrency}")

    # First verify /health
    base_url = endpoint.rsplit("/", 1)[0]
    health_url = f"{base_url}/health"
    try:
        h_resp = requests.get(health_url, timeout=5.0)
        logger.info(f"Health Probe check ({health_url}): HTTP {h_resp.status_code}")
    except Exception as e:
        logger.warning(f"Health Probe failed: {e}. Proceeding directly to prediction endpoint...")

    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_single_prediction_request, endpoint) for _ in range(n_requests)]
        for fut in as_completed(futures):
            results.append(fut.result())

    total_time_sec = time.perf_counter() - t_start
    latencies = [r["elapsed_ms"] for r in results]
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes

    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoint": endpoint,
        "total_requests": n_requests,
        "concurrency": concurrency,
        "total_duration_sec": round(total_time_sec, 3),
        "throughput_rps": round(n_requests / max(total_time_sec, 0.001), 2),
        "success_rate_pct": round((successes / n_requests) * 100, 2),
        "failed_requests": failures,
        "latency_ms": {
            "min": round(float(np.min(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p90": round(float(np.percentile(latencies, 90)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "max": round(float(np.max(latencies)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "std": round(float(np.std(latencies)), 2),
        },
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"LOAD TEST COMPLETE: {successes}/{n_requests} Successful ({stats['success_rate_pct']}%)")
    logger.info(f"Throughput: {stats['throughput_rps']} req/sec | Total Time: {stats['total_duration_sec']}s")
    logger.info(f"Latency P50: {stats['latency_ms']['p50']}ms | P95: {stats['latency_ms']['p95']}ms | P99: {stats['latency_ms']['p99']}ms")
    logger.info(f"Report saved to: {out_path}")
    logger.info("=" * 60)

    return stats


def run_in_process_direct(
    n_requests: int = 100,
    output_json: str = "reports/qos_load_test.json",
) -> Dict[str, Any]:
    """Runs high-entropy load test directly against FastAPI app predict_single endpoint."""
    from fastapi import Response
    from src.deployment.app import predict_single
    from src.deployment.schemas import ShipmentInput

    logger.info(f"Initiating In-Process Direct Load Test | Total Requests: {n_requests}")
    results = []
    t_start = time.perf_counter()

    for i in range(n_requests):
        payload = generate_unique_shipment_payload()
        # Clean extra entropy key before feeding Pydantic model
        entropy_token = payload.pop("_request_uuid", str(uuid.uuid4()))
        t0 = time.perf_counter()
        success = False
        status_code = 200
        error_msg = None
        try:
            shipment_in = ShipmentInput(**payload)
            resp_headers = Response()
            res = predict_single(
                shipment=shipment_in,
                response=resp_headers,
                idempotency_key=entropy_token,
            )
            success = res.risk_probability is not None
        except Exception as e:
            status_code = 500
            error_msg = str(e)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            "elapsed_ms": elapsed_ms,
            "status_code": status_code,
            "success": success,
            "error": error_msg,
        })

    total_time_sec = time.perf_counter() - t_start
    latencies = [r["elapsed_ms"] for r in results]
    successes = sum(1 for r in results if r["success"])
    failures = len(results) - successes

    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "in_process_direct_prediction",
        "total_requests": n_requests,
        "concurrency": 1,
        "total_duration_sec": round(total_time_sec, 3),
        "throughput_rps": round(n_requests / max(total_time_sec, 0.001), 2),
        "success_rate_pct": round((successes / n_requests) * 100, 2),
        "failed_requests": failures,
        "latency_ms": {
            "min": round(float(np.min(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p90": round(float(np.percentile(latencies, 90)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "max": round(float(np.max(latencies)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "std": round(float(np.std(latencies)), 2),
        },
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"LOAD TEST COMPLETE: {successes}/{n_requests} Successful ({stats['success_rate_pct']}%)")
    logger.info(f"Throughput: {stats['throughput_rps']} req/sec | Total Time: {stats['total_duration_sec']}s")
    logger.info(f"Latency P50: {stats['latency_ms']['p50']}ms | P95: {stats['latency_ms']['p95']}ms | P99: {stats['latency_ms']['p99']}ms")
    logger.info(f"Report saved to: {out_path}")
    logger.info("=" * 60)
    return stats


def main():
    parser = argparse.ArgumentParser(description="HSCM High-Entropy Load Generator")
    parser.add_argument("--endpoint", default=None, help="Prediction API URL (if running remote/live HTTP)")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of total requests")
    parser.add_argument("-c", "--concurrency", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("-o", "--output", default="reports/qos_load_test.json", help="Path to output summary JSON")
    parser.add_argument("--in-process", action="store_true", default=True, help="Run in-process direct against prediction endpoint")
    args = parser.parse_args()

    if args.endpoint:
        run_load_test(
            endpoint=args.endpoint,
            n_requests=args.requests,
            concurrency=args.concurrency,
            output_json=args.output,
        )
    else:
        run_in_process_direct(
            n_requests=args.requests,
            output_json=args.output,
        )


if __name__ == "__main__":
    main()
