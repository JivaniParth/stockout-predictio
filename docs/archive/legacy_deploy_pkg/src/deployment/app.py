"""
Layer 4: Web Presentation / Delivery Layer (FastAPI Microservice Controller).
Exposes RESTful endpoints for cloud orchestration, single-order prognostication,
high-throughput batch processing, caching, and QoS telemetry.
Incorporates rate limiting, distributed idempotency, tiered caching, audit trails, and async queues.
"""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psutil
import uvicorn

from src.config import DEVICE
from src.container import Container
from src.domain.entities import PredictionPrognosis, ShipmentEntity
from src.deployment.schemas import (
    AsyncBatchJobResponse,
    AuditQueryResponse,
    BatchJobStatusResponse,
    BatchPredictionResult,
    BatchShipmentInput,
    CacheStatsResponse,
    HealthResponse,
    PredictionResult,
    ShipmentInput,
)
from src.deployment.middleware.rate_limiter import RateLimitMiddleware

# Startup telemetry baseline
start_time = time.time()
container: Container = Container.get_instance()


def shipment_input_to_entity(s: ShipmentInput) -> ShipmentEntity:
    """Translates incoming DTO/schema to immutable domain entity."""
    return ShipmentEntity(
        country=s.Country,
        scheduled_delivery_date=s.Scheduled_Delivery_Date,
        po_sent_to_vendor_date=s.PO_Sent_to_Vendor_Date,
        manufacturer_lead_time_days=(
            float(s.Manufacturer_Lead_Time_Days)
            if s.Manufacturer_Lead_Time_Days is not None
            else 90.0
        ),
        product_group=s.Product_Group,
        sub_classification=s.Sub_Classification,
        shipment_mode=s.Shipment_Mode,
        fulfill_via=s.Fulfill_Via,
        vendor_inco_term=s.Vendor_INCO_Term,
        vendor=s.Vendor,
        manufacturing_site=s.Manufacturing_Site,
        first_line_designation=s.First_Line_Designation,
        line_item_quantity=float(s.Line_Item_Quantity),
        line_item_value=float(s.Line_Item_Value),
        pack_price=float(s.Pack_Price),
        unit_price=float(s.Unit_Price),
        unit_of_measure_per_pack=float(s.Unit_of_Measure_Per_Pack),
        weight_kilograms=s.Weight_Kilograms if s.Weight_Kilograms else "350.0",
        freight_cost_usd=s.Freight_Cost_USD if s.Freight_Cost_USD else "3200.0",
        line_item_insurance_usd=(
            float(s.Line_Item_Insurance_USD)
            if s.Line_Item_Insurance_USD is not None
            else 50.0
        ),
    )


def prognosis_to_response(p: PredictionPrognosis) -> PredictionResult:
    """Translates domain prognosis to serializable HTTP response model."""
    tier_val = p.risk_tier.value if hasattr(p.risk_tier, "value") else str(p.risk_tier)
    country_val = (
        p.country_tier.value if hasattr(p.country_tier, "value") else str(p.country_tier)
    )
    return PredictionResult(
        late_delivery_risk=p.late_delivery_risk,
        risk_probability=p.risk_probability,
        risk_tier=tier_val,
        country_tier=country_val,
        qos_latency_ms=p.qos_latency_ms,
    )


def compute_cache_key(model_name: str, shipment: ShipmentInput) -> str:
    """Generates deterministic SHA-256 cache key from input attributes."""
    canonical_repr = (
        f"{model_name}:{shipment.Country}:{shipment.Scheduled_Delivery_Date}:"
        f"{shipment.Manufacturer_Lead_Time_Days}:{shipment.Product_Group}:"
        f"{shipment.Shipment_Mode}:{shipment.Fulfill_Via}:{shipment.Vendor_INCO_Term}:"
        f"{shipment.Line_Item_Quantity}:{shipment.Line_Item_Value}"
    )
    return f"hscm:pred:{hashlib.sha256(canonical_repr.encode()).hexdigest()}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes container artifacts upon service startup."""
    container.get_preprocessor()
    container.get_predictor("HybridLSTMTransformer")
    yield


app = FastAPI(
    title="Cloud-Native Pharmaceutical Stockout Prediction Microservice",
    description="Enterprise Distributed Architecture for medicine stockout inference, QoS profiling, caching, and health equity telemetry.",
    version="3.0.0-production-cloud",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,  # Minimize attack surface in production
)

# 1. Rate Limiting Middleware (Sliding Window per IP / API key)
app.add_middleware(RateLimitMiddleware, requests_per_minute=600)

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# 3. Security Response Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


# 4. Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    incident_id = str(uuid.uuid4())
    print(
        f"[CRITICAL OUTAGE ERROR] [Incident {incident_id}] Path: {request.url.path} | Error: {str(exc)}"
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during model inference. The event has been logged.",
            "incident_id": incident_id,
        },
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness and readiness probe for Kubernetes container orchestration."""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = 0.0

    uptime = time.time() - start_time
    predictor = container.get_predictor("HybridLSTMTransformer")

    return HealthResponse(
        status="HEALTHY",
        service="hscm-stockout-prediction-microservice",
        active_model=predictor.model_name,
        device=str(DEVICE),
        memory_usage_mb=round(mem_mb, 2),
        uptime_seconds=round(uptime, 2),
    )


@app.post("/predict", response_model=PredictionResult)
def predict_single(
    shipment: ShipmentInput,
    response: Response,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    """
    Predicts stockout risk for a single order.
    Features:
    1. Idempotency Check: Returns previously computed payload if Idempotency-Key matches.
    2. Cache-Aside Pattern: Checks L1/L2 cache before invoking forward pass.
    3. Audit Persistence: Commits transaction to relational database.
    """
    idempotency_mgr = container.get_idempotency_manager()
    cache = container.get_cache_service()
    shipment_repo = container.get_shipment_repo()

    # 1. Idempotency Guard
    if idempotency_key:
        cached_idemp = idempotency_mgr.get_existing_response(idempotency_key)
        if cached_idemp:
            response.headers["X-Idempotent-Replay"] = "true"
            response.headers["X-Cache"] = "HIT"
            return PredictionResult(**cached_idemp)

    # 2. Caching Guard (Cache-Aside Pattern)
    cache_key = compute_cache_key("HybridLSTMTransformer", shipment)
    cached_result = cache.get(cache_key)
    if cached_result:
        response.headers["X-Cache"] = "HIT"
        if idempotency_key:
            idempotency_mgr.record_response(idempotency_key, cached_result)
        return PredictionResult(**cached_result)

    # 3. Model Inference via Clean Architecture Use Case
    entity = shipment_input_to_entity(shipment)
    use_case = container.get_predict_shipment_use_case("HybridLSTMTransformer")
    prognosis = use_case.execute(entity)
    res_dto = prognosis_to_response(prognosis)
    res_dict = res_dto.model_dump()

    # 4. Set in Cache (TTL 1 Hour)
    cache.set(cache_key, res_dict, ttl_seconds=3600)
    response.headers["X-Cache"] = "MISS"

    if idempotency_key:
        idempotency_mgr.record_response(idempotency_key, res_dict)

    # 5. Persist to Audit Trail (Asynchronously or Transactionally)
    try:
        shipment_repo.save_audit(
            shipment_data=shipment.model_dump(),
            prediction_result={
                "model_name": "HybridLSTMTransformer",
                "late_delivery_risk": res_dto.late_delivery_risk,
                "risk_probability": res_dto.risk_probability,
                "risk_tier": res_dto.risk_tier,
                "country_tier": res_dto.country_tier,
                "qos_latency_ms": res_dto.qos_latency_ms,
                "cache_hit": False,
            },
            idempotency_key=idempotency_key,
        )
    except Exception as e:
        print(f"[!] Audit logging error: {e}")

    return res_dto


@app.post("/predict/batch", response_model=BatchPredictionResult)
def predict_batch(batch_input: BatchShipmentInput):
    """
    Synchronous high-throughput batch inference endpoint bounded to max 1,000 shipments.
    Delegates to Layer 2 BatchPredictShipmentsUseCase.
    """
    shipments = batch_input.shipments
    if not shipments:
        raise HTTPException(status_code=400, detail="Batch cannot be empty.")

    entities = [shipment_input_to_entity(s) for s in shipments]
    use_case = container.get_batch_predict_use_case("HybridLSTMTransformer")
    prognoses, total_latency_ms, throughput = use_case.execute(entities)

    results = [prognosis_to_response(p) for p in prognoses]

    return BatchPredictionResult(
        total_samples=len(shipments),
        predictions=results,
        total_batch_latency_ms=total_latency_ms,
        throughput_samples_per_sec=throughput,
    )


@app.post(
    "/api/v1/predictions/batch/async",
    response_model=AsyncBatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def predict_batch_async(
    batch_input: BatchShipmentInput,
    model_name: str = "HybridLSTMTransformer",
):
    """
    Asynchronous high-throughput batch scoring endpoint.
    Offloads workload to background task queue workers, returning Job ID immediately.
    Clients poll /api/v1/jobs/{job_id} for completion status.
    """
    job_id = str(uuid.uuid4())
    shipments = batch_input.shipments
    total_records = len(shipments)

    job_repo = container.get_batch_job_repo()
    job_repo.create_job(job_id=job_id, total_records=total_records, model_name=model_name)

    task_queue = container.get_task_queue()

    def _process_batch_worker(j_id: str, items: List[ShipmentInput], m_name: str):
        try:
            job_repo.update_job_progress(j_id, processed_records=0, status="PROCESSING")
            entities = [shipment_input_to_entity(s) for s in items]
            use_case = container.get_batch_predict_use_case(m_name)
            prognoses, total_lat_ms, throughput = use_case.execute(entities)

            at_risk = sum(1 for p in prognoses if p.late_delivery_risk == 1)
            summary = {
                "total_samples": len(entities),
                "at_risk_count": at_risk,
                "risk_rate_percent": round((at_risk / len(entities)) * 100, 2),
                "total_latency_ms": total_lat_ms,
                "throughput_samples_per_sec": throughput,
            }
            job_repo.update_job_progress(
                j_id,
                processed_records=len(entities),
                status="COMPLETED",
                results_summary=summary,
            )
        except Exception as e:
            job_repo.update_job_progress(
                j_id,
                processed_records=0,
                status="FAILED",
                error_message=str(e),
            )

    enqueued = task_queue.enqueue(
        task_id=f"job-{job_id}",
        handler=_process_batch_worker,
        j_id=job_id,
        items=shipments,
        m_name=model_name,
    )

    if not enqueued:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background task worker queue is saturated. Please retry later.",
        )

    return AsyncBatchJobResponse(
        job_id=job_id,
        status="PENDING",
        total_records=total_records,
        check_status_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=BatchJobStatusResponse)
def get_batch_job_status(job_id: str):
    """Retrieves current execution progress and metrics of an asynchronous batch job."""
    job_repo = container.get_batch_job_repo()
    job = job_repo.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {job_id} not found.",
        )
    return BatchJobStatusResponse(**job)


@app.get("/api/v1/audit/shipments", response_model=AuditQueryResponse)
def query_audited_shipments(
    country: Optional[str] = Query(None, description="Filter by destination country"),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier"),
    limit: int = Query(50, ge=1, le=200, description="Page size limit"),
    cursor: Optional[str] = Query(None, description="Pagination cursor (timestamp)"),
):
    """Cursor-paginated historical audit trail of all model inferences."""
    shipment_repo = container.get_shipment_repo()
    data = shipment_repo.query_audits(country=country, risk_tier=risk_tier, limit=limit, cursor=cursor)
    return AuditQueryResponse(**data)


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse)
def get_cache_statistics():
    """Retrieves operational caching telemetry: hit ratio, evictions, memory footprint."""
    cache = container.get_cache_service()
    return CacheStatsResponse(**cache.get_stats())


@app.get("/qos_metrics")
def get_qos_metrics() -> Dict[str, Any]:
    """Cloud QoS telemetry endpoint delegating to Layer 2 GetQoSMetricsUseCase."""
    use_case = container.get_qos_metrics_use_case()
    return use_case.execute()


if __name__ == "__main__":
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=8000, reload=False)
