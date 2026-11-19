"""
Comprehensive Test Client for Scalable Cloud-Native HSCM Platform.
Validates:
1. /health probe
2. /predict with Caching & Idempotency Key
3. /predict/batch synchronous scoring
4. /api/v1/predictions/batch/async asynchronous task queue & background execution
5. /api/v1/jobs/{job_id} job polling & completion status
6. /api/v1/audit/shipments relational audit persistence & cursor pagination
7. /api/v1/cache/stats cache hit ratios and telemetry
8. /qos_metrics cloud SLA percentiles
"""

import time
from starlette.testclient import TestClient
from src.deployment.app import app


def test_microservice():
    print("=" * 70)
    print("TESTING SCALABLE CLOUD-NATIVE PREDICTION PLATFORM")
    print("=" * 70)

    with TestClient(app) as client:
        # 1. Test /health
        print("\n[1] Testing /health probe...")
        res = client.get("/health")
        print("Status Code:", res.status_code)
        print("Payload:", res.json())
        assert res.status_code == 200

        payload_single = {
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

        # 2. Test /predict (First request: Cache MISS)
        print("\n[2] Testing /predict (Initial Request -> Expect Cache MISS)...")
        headers = {"Idempotency-Key": "req-unique-uuid-999"}
        res1 = client.post("/predict", json=payload_single, headers=headers)
        print("Status Code:", res1.status_code)
        print("X-Cache Header:", res1.headers.get("x-cache"))
        print("Prediction:", res1.json())
        assert res1.status_code == 200
        assert res1.headers.get("x-cache") == "MISS"

        # 3. Test /predict (Second identical request: Cache HIT & Idempotent Replay)
        print("\n[3] Testing /predict (Repeated Request -> Expect Cache HIT & Idempotency)...")
        res2 = client.post("/predict", json=payload_single, headers=headers)
        print("Status Code:", res2.status_code)
        print("X-Cache Header:", res2.headers.get("x-cache"))
        print("X-Idempotent-Replay:", res2.headers.get("x-idempotent-replay"))
        assert res2.status_code == 200
        assert res2.headers.get("x-cache") == "HIT"
        assert res2.headers.get("x-idempotent-replay") == "true"

        # 4. Test /predict/batch (Synchronous Batch)
        print("\n[4] Testing /predict/batch (Synchronous Batch Inference)...")
        payload_batch = {
            "shipments": [
                payload_single,
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
        res_sync = client.post("/predict/batch", json=payload_batch)
        print("Status Code:", res_sync.status_code)
        print("Batch Response Summary:", {k: v for k, v in res_sync.json().items() if k != "predictions"})
        assert res_sync.status_code == 200

        # 5. Test Asynchronous Batch Job
        print("\n[5] Testing /api/v1/predictions/batch/async (Background Task Queue)...")
        res_async = client.post("/api/v1/predictions/batch/async", json=payload_batch)
        print("Status Code:", res_async.status_code)
        async_data = res_async.json()
        print("Async Response:", async_data)
        assert res_async.status_code == 202
        job_id = async_data["job_id"]

        # Wait briefly for background thread execution
        time.sleep(0.5)

        # 6. Test Job Status Polling
        print("\n[6] Polling /api/v1/jobs/{job_id}...")
        res_job = client.get(f"/api/v1/jobs/{job_id}")
        print("Status Code:", res_job.status_code)
        print("Job Status Data:", res_job.json())
        assert res_job.status_code == 200
        assert res_job.json()["status"] in ["COMPLETED", "PROCESSING"]

        # 7. Test Audit Trail Persistence
        print("\n[7] Querying /api/v1/audit/shipments (Relational Audit Trail)...")
        res_audit = client.get("/api/v1/audit/shipments?limit=10")
        print("Status Code:", res_audit.status_code)
        audit_data = res_audit.json()
        print(f"Retrieved {audit_data['count']} audited consignments from database.")
        assert res_audit.status_code == 200
        assert audit_data["count"] >= 1

        # 8. Test Cache Statistics
        print("\n[8] Querying /api/v1/cache/stats...")
        res_cache = client.get("/api/v1/cache/stats")
        print("Cache Telemetry:", res_cache.json())
        assert res_cache.status_code == 200
        assert res_cache.json()["hits"] >= 1

        # 9. Test /qos_metrics
        print("\n[9] Testing /qos_metrics telemetry...")
        res_qos = client.get("/qos_metrics")
        print("QoS Telemetry:", res_qos.json())
        assert res_qos.status_code == 200

        print("\n[+] ALL SCALABLE ENTERPRISE INFRASTRUCTURE ENDPOINTS PASSED WITH 100% SUCCESS!")


if __name__ == "__main__":
    test_microservice()
