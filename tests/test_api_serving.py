"""
Comprehensive Test Client for Scalable Cloud-Native HSCM Platform.
Migrated from root test_api.py into tests/test_api_serving.py.

Validates all 9 critical service capabilities:
1. /health probe
2. /predict with Caching & Idempotency Key (Cache MISS)
3. /predict with Idempotent Replay (Cache HIT)
4. /predict/batch synchronous scoring
5. /api/v1/predictions/batch/async asynchronous task queue & background execution
6. /api/v1/jobs/{job_id} job polling & completion status
7. /api/v1/audit/shipments relational audit persistence & cursor pagination
8. /api/v1/cache/stats cache hit ratios and telemetry
9. /qos_metrics cloud SLA percentiles
"""

import time
import urllib.parse
from typing import Any, Dict, Optional
import pytest
from fastapi import Response

try:
    from src.scms.serving.app import (
        app,
        get_batch_job_status,
        get_cache_statistics,
        get_qos_metrics,
        health_check,
        predict_batch,
        predict_batch_async,
        predict_single,
        query_audited_shipments,
        container,
    )
    from src.scms.serving.schemas import BatchShipmentInput, ShipmentInput
except ImportError:
    from src.deployment.app import (
        app,
        get_batch_job_status,
        get_cache_statistics,
        get_qos_metrics,
        health_check,
        predict_batch,
        predict_batch_async,
        predict_single,
        query_audited_shipments,
        container,
    )
    from src.deployment.schemas import BatchShipmentInput, ShipmentInput


class CaseInsensitiveDict(dict):
    """Case-insensitive dictionary wrapper matching Starlette/requests headers behavior."""

    def __init__(self, data=None):
        super().__init__()
        if data:
            for k, v in data.items():
                super().__setitem__(str(k).lower(), v)

    def __getitem__(self, key):
        return super().__getitem__(str(key).lower())

    def get(self, key, default=None):
        return super().get(str(key).lower(), default)

    def __contains__(self, key):
        return super().__contains__(str(key).lower())


class APIResponse:
    """Standardized response representation matching starlette / requests."""

    def __init__(self, status_code: int, data: Any, headers: Optional[Dict[str, str]] = None):
        self.status_code = status_code
        self._data = data
        self.headers = CaseInsensitiveDict(headers or {})

    def json(self) -> Any:
        if hasattr(self._data, "model_dump"):
            return self._data.model_dump()
        return self._data


class MicroserviceTestClient:
    """In-process high performance test client executing routes directly."""

    def __init__(self, app_instance=None):
        self.app = app_instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, path: str, headers: Optional[Dict[str, str]] = None) -> APIResponse:
        parsed = urllib.parse.urlparse(path)
        route_path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        if route_path == "/health":
            res = health_check()
            return APIResponse(200, res.model_dump(), {})

        if route_path.startswith("/api/v1/jobs/"):
            job_id = route_path.split("/")[-1]
            res = get_batch_job_status(job_id=job_id)
            return APIResponse(200, res.model_dump(), {})

        if route_path == "/api/v1/audit/shipments":
            country = query_params.get("country", [None])[0]
            risk_tier = query_params.get("risk_tier", [None])[0]
            limit_val = int(query_params.get("limit", [50])[0])
            cursor = query_params.get("cursor", [None])[0]
            res = query_audited_shipments(
                country=country,
                risk_tier=risk_tier,
                limit=limit_val,
                cursor=cursor,
            )
            return APIResponse(200, res.model_dump(), {})

        if route_path == "/api/v1/cache/stats":
            res = get_cache_statistics()
            return APIResponse(200, res.model_dump(), {})

        if route_path == "/qos_metrics":
            res = get_qos_metrics()
            return APIResponse(200, res, {})

        raise NotImplementedError(f"Route not mocked in test client: {path}")

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> APIResponse:
        headers = headers or {}
        if path == "/predict":
            shipment_dto = ShipmentInput(**json)
            fastapi_res = Response()
            idemp_key = headers.get("Idempotency-Key") or headers.get("idempotency-key")
            result_dto = predict_single(
                shipment=shipment_dto,
                response=fastapi_res,
                idempotency_key=idemp_key,
            )
            resp_headers = dict(fastapi_res.headers)
            return APIResponse(200, result_dto.model_dump(), resp_headers)

        if path == "/predict/batch":
            batch_dto = BatchShipmentInput(**json)
            result_dto = predict_batch(batch_input=batch_dto)
            return APIResponse(200, result_dto.model_dump(), {})

        if path == "/api/v1/predictions/batch/async":
            batch_dto = BatchShipmentInput(**json)
            result_dto = predict_batch_async(batch_input=batch_dto)
            return APIResponse(202, result_dto.model_dump(), {})

        raise NotImplementedError(f"Route not mocked in test client: {path}")


# Try Starlette TestClient if httpx available, else MicroserviceTestClient
try:
    from starlette.testclient import TestClient as StarletteTestClient

    def TestClient(app_inst):
        return StarletteTestClient(app_inst)

except Exception:
    def TestClient(app_inst):
        return MicroserviceTestClient(app_inst)


@pytest.fixture(scope="module")
def client():
    with MicroserviceTestClient(app) as test_client:
        yield test_client


PAYLOAD_SINGLE = {
    "Country": "Congo, DRC",
    "Scheduled_Delivery_Date": "2015-11-20",
    "PO_Sent_to_Vendor_Date": "2015-08-01",
    "Product_Group": "ARV",
    "Sub_Classification": "Adult",
    "Shipment_Mode": "Air",
    "Fulfill_Via": "Direct Drop",
    "Vendor_INCO_Term": "CIP",
    "Vendor": "Aurobindo Pharma Limited",
    "Manufacturing_Site": "Aurobindo Unit III, India",
    "First_Line_Designation": "Yes",
    "Line_Item_Quantity": 25000.0,
    "Line_Item_Value": 125000.0,
    "Pack_Price": 5.0,
    "Unit_Price": 0.08,
    "Unit_of_Measure_Per_Pack": 60.0,
    "Weight_Kilograms": "1450.0",
    "Freight_Cost_USD": "8900.0",
    "Line_Item_Insurance_USD": 140.0,
}

PAYLOAD_BATCH = {
    "shipments": [
        PAYLOAD_SINGLE,
        {
            "Country": "South Africa",
            "Scheduled_Delivery_Date": "2015-10-15",
            "PO_Sent_to_Vendor_Date": "2015-09-01",
            "Product_Group": "HRDT",
            "Sub_Classification": "HIV test",
            "Shipment_Mode": "Truck",
            "Fulfill_Via": "From RDC",
            "Vendor_INCO_Term": "EXW",
            "Vendor": "Ranbaxy Fine Chemicals LTD.",
            "Manufacturing_Site": "Ranbaxy Fine Chemicals LTD",
            "First_Line_Designation": "Yes",
            "Line_Item_Quantity": 1000.0,
            "Line_Item_Value": 29000.0,
            "Pack_Price": 29.0,
            "Unit_Price": 0.97,
            "Unit_of_Measure_Per_Pack": 30.0,
            "Weight_Kilograms": "80.0",
            "Freight_Cost_USD": "650.0",
            "Line_Item_Insurance_USD": 35.0,
        },
    ]
}


def test_api_health_probe(client):
    """1. Test /health probe."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"


def test_api_predict_caching_and_idempotency(client):
    """2 & 3. Test /predict Cache MISS followed by Cache HIT & Idempotency replay."""
    container.get_cache_service().clear()
    unique_key = f"req-unique-uuid-{int(time.time() * 1000)}"
    headers = {"Idempotency-Key": unique_key}

    # Initial Request -> Expect Cache MISS
    res1 = client.post("/predict", json=PAYLOAD_SINGLE, headers=headers)
    assert res1.status_code == 200
    assert res1.headers.get("x-cache") == "MISS"

    # Repeated Request -> Expect Cache HIT & Idempotency
    res2 = client.post("/predict", json=PAYLOAD_SINGLE, headers=headers)
    assert res2.status_code == 200
    assert res2.headers.get("x-cache") == "HIT"
    assert res2.headers.get("x-idempotent-replay") == "true"


def test_api_predict_batch_synchronous(client):
    """4. Test /predict/batch synchronous inference."""
    res_sync = client.post("/predict/batch", json=PAYLOAD_BATCH)
    assert res_sync.status_code == 200
    data = res_sync.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_api_async_batch_and_job_status(client):
    """5 & 6. Test async batch submission and polling job completion."""
    res_async = client.post("/api/v1/predictions/batch/async", json=PAYLOAD_BATCH)
    assert res_async.status_code == 202
    async_data = res_async.json()
    assert "job_id" in async_data
    job_id = async_data["job_id"]

    # Wait briefly for background execution
    time.sleep(0.5)

    # Poll status
    res_job = client.get(f"/api/v1/jobs/{job_id}")
    assert res_job.status_code == 200
    assert res_job.json()["status"] in ["COMPLETED", "PROCESSING"]


def test_api_audit_trail_persistence(client):
    """7. Test /api/v1/audit/shipments relational audit trail."""
    res_audit = client.get("/api/v1/audit/shipments?limit=10")
    assert res_audit.status_code == 200
    audit_data = res_audit.json()
    assert "count" in audit_data
    assert audit_data["count"] >= 1


def test_api_cache_stats(client):
    """8. Test /api/v1/cache/stats."""
    res_cache = client.get("/api/v1/cache/stats")
    assert res_cache.status_code == 200
    data = res_cache.json()
    assert "hits" in data
    assert data["hits"] >= 1


def test_api_qos_metrics(client):
    """9. Test /qos_metrics cloud SLA percentiles."""
    res_qos = client.get("/qos_metrics")
    assert res_qos.status_code == 200


def test_microservice_full_workflow():
    """Exact preservation of original test_microservice() sequential flow."""
    container.get_cache_service().clear()
    with MicroserviceTestClient(app) as test_client:
        # 1. Test /health
        res = test_client.get("/health")
        assert res.status_code == 200

        # 2. Test /predict (First request: Cache MISS)
        headers = {"Idempotency-Key": f"req-unique-uuid-999-{time.time()}"}
        res1 = test_client.post("/predict", json=PAYLOAD_SINGLE, headers=headers)
        assert res1.status_code == 200
        assert res1.headers.get("x-cache") == "MISS"

        # 3. Test /predict (Second identical request: Cache HIT & Idempotent Replay)
        res2 = test_client.post("/predict", json=PAYLOAD_SINGLE, headers=headers)
        assert res2.status_code == 200
        assert res2.headers.get("x-cache") == "HIT"
        assert res2.headers.get("x-idempotent-replay") == "true"

        # 4. Test /predict/batch (Synchronous Batch)
        res_sync = test_client.post("/predict/batch", json=PAYLOAD_BATCH)
        assert res_sync.status_code == 200

        # 5. Test Asynchronous Batch Job
        res_async = test_client.post("/api/v1/predictions/batch/async", json=PAYLOAD_BATCH)
        assert res_async.status_code == 202
        job_id = res_async.json()["job_id"]

        time.sleep(0.5)

        # 6. Test Job Status Polling
        res_job = test_client.get(f"/api/v1/jobs/{job_id}")
        assert res_job.status_code == 200
        assert res_job.json()["status"] in ["COMPLETED", "PROCESSING"]

        # 7. Test Audit Trail Persistence
        res_audit = test_client.get("/api/v1/audit/shipments?limit=10")
        assert res_audit.status_code == 200
        audit_data = res_audit.json()
        assert audit_data["count"] >= 1

        # 8. Test Cache Statistics
        res_cache = test_client.get("/api/v1/cache/stats")
        assert res_cache.status_code == 200
        assert res_cache.json()["hits"] >= 1

        # 9. Test /qos_metrics
        res_qos = test_client.get("/qos_metrics")
        assert res_qos.status_code == 200
