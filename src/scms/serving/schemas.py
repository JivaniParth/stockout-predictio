"""
Production Hardened Pydantic Schemas for Cloud-Native Pharmaceutical Inference Microservice.
Implements strict boundary validation, non-negative assertions, finite float enforcement,
and payload size limits to mitigate Denial-of-Service (DoS) and integer/buffer overflow exploits.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class ShipmentInput(BaseModel):
    Country: str = Field(
        ...,
        min_length=1,
        max_length=100,
        example="Congo, DRC",
        description="Destination sovereign country name (sanitized, max 100 chars).",
    )
    Scheduled_Delivery_Date: str = Field(
        ...,
        min_length=8,
        max_length=25,
        example="2026-11-20",
        description="Target client arrival date in ISO-8601 or standard calendar format.",
    )
    PO_Sent_to_Vendor_Date: Optional[str] = Field(
        None,
        max_length=25,
        example="2026-08-01",
        description="Contract purchase order transmission date.",
    )
    Manufacturer_Lead_Time_Days: Optional[float] = Field(
        90.0,
        ge=1.0,
        le=730.0,
        example=90.0,
        description="Manufacturer SLA lead time in days (bounded: 1 to 730 days).",
    )
    Product_Group: str = Field(
        ...,
        min_length=1,
        max_length=50,
        example="ARV",
        description="Therapeutic pharmaceutical category.",
    )
    Sub_Classification: str = Field(
        ...,
        min_length=1,
        max_length=100,
        example="Pediatric",
        description="Dosage formulation or screening category.",
    )
    Shipment_Mode: str = Field(
        ...,
        min_length=1,
        max_length=50,
        example="Air",
        description="Logistical freight mode (Air, Truck, Ocean, Air Charter).",
    )
    Fulfill_Via: str = Field(
        ...,
        min_length=1,
        max_length=50,
        example="Direct Drop",
        description="Distribution pipeline channel.",
    )
    Vendor_INCO_Term: str = Field(
        ...,
        min_length=1,
        max_length=20,
        example="CIP",
        description="Commercial incoterms (CIP, EXW, FCA, DDP, CIF, DAP).",
    )
    Vendor: str = Field(
        ...,
        min_length=1,
        max_length=150,
        example="Aurobindo Pharma Limited",
        description="Contracted pharmaceutical supplier name.",
    )
    Manufacturing_Site: str = Field(
        ...,
        min_length=1,
        max_length=150,
        example="Aurobindo Unit III, India",
        description="Certified production manufacturing site location.",
    )
    First_Line_Designation: str = Field(
        ...,
        pattern="^(Yes|No|yes|no)$",
        example="Yes",
        description="Essential frontline WHO medication status.",
    )
    Line_Item_Quantity: float = Field(
        ...,
        ge=0.0,
        le=100_000_000.0,
        example=5000.0,
        description="Total packs ordered. Bounded strictly between 0 and 100M packs.",
    )
    Line_Item_Value: float = Field(
        ...,
        ge=0.0,
        le=1_000_000_000.0,
        example=25000.0,
        description="Invoice monetary value in USD. Bounded strictly between $0 and $1B.",
    )
    Pack_Price: float = Field(
        ...,
        ge=0.0,
        le=1_000_000.0,
        example=5.0,
        description="Procurement price per unit pack in USD.",
    )
    Unit_Price: float = Field(
        ...,
        ge=0.0,
        le=100_000.0,
        example=0.08,
        description="Procurement price per single pill or test in USD.",
    )
    Unit_of_Measure_Per_Pack: float = Field(
        ...,
        ge=1.0,
        le=50_000.0,
        example=60.0,
        description="Item count per pack. Strictly >= 1 to prevent division-by-zero.",
    )
    Weight_Kilograms: Optional[str] = Field(
        "350.0",
        max_length=50,
        example="350.0",
        description="Consignment gross weight in kg (max 50 chars).",
    )
    Freight_Cost_USD: Optional[str] = Field(
        "3200.0",
        max_length=50,
        example="3200.0",
        description="Total transport freight fee in USD (max 50 chars).",
    )
    Line_Item_Insurance_USD: Optional[float] = Field(
        50.0,
        ge=0.0,
        le=10_000_000.0,
        example=50.0,
        description="Transit cargo insurance cost in USD.",
    )

    @field_validator(
        "Line_Item_Quantity",
        "Line_Item_Value",
        "Pack_Price",
        "Unit_Price",
        "Unit_of_Measure_Per_Pack",
        "Line_Item_Insurance_USD",
        "Manufacturer_Lead_Time_Days",
        mode="before",
    )
    @classmethod
    def validate_finite_numbers(cls, v):
        """Rejects NaN, Infinity, and -Infinity to prevent model arithmetic corruption."""
        if v is not None:
            try:
                val = float(v)
                import math
                if math.isnan(val) or math.isinf(val):
                    raise ValueError("Floating-point values must be finite. NaN and Infinity are prohibited.")
                return val
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid numeric input: {e}")
        return v


class BatchShipmentInput(BaseModel):
    shipments: List[ShipmentInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Batch payload bounded between 1 and 1,000 records to prevent CPU/Memory exhaustion DoS.",
    )


class PredictionResult(BaseModel):
    late_delivery_risk: int = Field(..., description="0 = On-Time, 1 = High Risk of Stockout/Delay")
    risk_probability: float = Field(..., description="Calibrated risk probability [0.0 - 1.0]")
    risk_tier: str = Field(..., description="Low, Moderate, High, or Critical Risk")
    country_tier: str = Field(..., description="Geographic Health Equity Tier")
    qos_latency_ms: float = Field(..., description="Inference latency in milliseconds")
    calibrated_probability: Optional[float] = Field(None, description="Calibrated delay probability [0.0 - 1.0]")
    binary_alert: Optional[int] = Field(None, description="Binary classification at calibrated tau* threshold (1 = alert, 0 = normal)")
    risk_band: Optional[str] = Field(None, description="Clinical risk severity band")
    model_version: Optional[str] = Field(None, description="Model release version / git hash")
    tier_threshold_applied: Optional[float] = Field(None, description="Decision threshold applied for recipient country tier")
    top_k_shap_features: Optional[List[dict]] = Field(None, description="Top-k local feature attributions when requested")


class BatchPredictionResult(BaseModel):
    total_samples: int
    predictions: List[PredictionResult]
    total_batch_latency_ms: float
    throughput_samples_per_sec: float


class HealthResponse(BaseModel):
    status: str
    service: str
    active_model: str
    device: str
    memory_usage_mb: float
    uptime_seconds: float


class AsyncBatchJobResponse(BaseModel):
    job_id: str
    status: str
    total_records: int
    check_status_url: str


class BatchJobStatusResponse(BaseModel):
    job_id: str
    tenant_id: str
    model_name: str
    status: str
    total_records: int
    processed_records: int
    throughput_samples_per_sec: float
    results_summary: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class AuditRecord(BaseModel):
    audit_id: str
    model_name: str
    late_delivery_risk: int
    risk_probability: float
    risk_tier: str
    country_tier: str
    qos_latency_ms: float
    created_at: str
    country: Optional[str] = None
    product_group: Optional[str] = None
    scheduled_delivery_date: Optional[str] = None


class AuditQueryResponse(BaseModel):
    items: List[AuditRecord]
    count: int
    next_cursor: Optional[str] = None


class CacheStatsResponse(BaseModel):
    engine: str
    size: Optional[int] = None
    max_capacity: Optional[int] = None
    hits: int
    misses: int
    hit_ratio_percent: float
    used_memory_human: Optional[str] = None
    connected: Optional[bool] = None

