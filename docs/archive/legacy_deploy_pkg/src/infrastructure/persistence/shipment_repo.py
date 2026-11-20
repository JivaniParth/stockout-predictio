"""
Layer 3 Persistence Adapter: Relational Repositories.
Implements IShipmentRepository and IBatchJobRepository against DatabaseEngine.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional
import uuid

from src.application.ports.repository import IShipmentRepository, IBatchJobRepository
from src.infrastructure.persistence.database import DatabaseEngine


class RelationalShipmentRepository(IShipmentRepository):
    """Concrete repository for transactional shipments and prediction audit records."""

    def __init__(self, db_engine: DatabaseEngine):
        self._db = db_engine

    def save_audit(
        self,
        shipment_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        tenant_id: str = "default_tenant",
    ) -> str:
        audit_id = str(uuid.uuid4())
        shipment_id = str(uuid.uuid4())

        # Canonical hash of input payload to detect duplicate payload requests
        serialized_input = json.dumps(shipment_data, sort_keys=True)
        payload_hash = hashlib.sha256(serialized_input.encode()).hexdigest()

        with self._db.session() as conn:
            # 1. Insert Consignment Record
            conn.execute(
                """
                INSERT INTO shipments (
                    id, tenant_id, country, scheduled_delivery_date, po_sent_to_vendor_date,
                    manufacturer_lead_time_days, product_group, sub_classification,
                    shipment_mode, fulfill_via, vendor_inco_term, vendor,
                    manufacturing_site, first_line_designation, line_item_quantity,
                    line_item_value, pack_price, unit_price, unit_of_measure_per_pack,
                    weight_kilograms, freight_cost_usd, line_item_insurance_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shipment_id,
                    tenant_id,
                    shipment_data.get("country", ""),
                    str(shipment_data.get("scheduled_delivery_date", "")),
                    str(shipment_data.get("po_sent_to_vendor_date", "")),
                    float(shipment_data.get("manufacturer_lead_time_days", 90.0)),
                    shipment_data.get("product_group", ""),
                    shipment_data.get("sub_classification", ""),
                    shipment_data.get("shipment_mode", ""),
                    shipment_data.get("fulfill_via", ""),
                    shipment_data.get("vendor_inco_term", ""),
                    shipment_data.get("vendor", ""),
                    shipment_data.get("manufacturing_site", ""),
                    shipment_data.get("first_line_designation", ""),
                    float(shipment_data.get("line_item_quantity", 0.0)),
                    float(shipment_data.get("line_item_value", 0.0)),
                    float(shipment_data.get("pack_price", 0.0)),
                    float(shipment_data.get("unit_price", 0.0)),
                    float(shipment_data.get("unit_of_measure_per_pack", 1.0)),
                    str(shipment_data.get("weight_kilograms", "350.0")),
                    str(shipment_data.get("freight_cost_usd", "3200.0")),
                    float(shipment_data.get("line_item_insurance_usd", 50.0)),
                ),
            )

            # 2. Insert Immutable Prediction Audit Record
            tier_val = prediction_result.get("risk_tier")
            risk_tier_str = tier_val.value if hasattr(tier_val, "value") else str(tier_val)
            country_val = prediction_result.get("country_tier")
            country_tier_str = country_val.value if hasattr(country_val, "value") else str(country_val)

            conn.execute(
                """
                INSERT INTO prediction_audits (
                    audit_id, shipment_id, tenant_id, idempotency_key, model_name,
                    late_delivery_risk, risk_probability, risk_tier, country_tier,
                    qos_latency_ms, cache_hit, input_payload_hash, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    shipment_id,
                    tenant_id,
                    idempotency_key,
                    prediction_result.get("model_name", "HybridLSTMTransformer"),
                    int(prediction_result.get("late_delivery_risk", 0)),
                    float(prediction_result.get("risk_probability", 0.0)),
                    risk_tier_str,
                    country_tier_str,
                    float(prediction_result.get("qos_latency_ms", 0.0)),
                    1 if prediction_result.get("cache_hit") else 0,
                    payload_hash,
                    json.dumps(prediction_result.get("metadata", {})),
                ),
            )

        return audit_id

    def get_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        with self._db.session() as conn:
            cursor = conn.execute(
                """
                SELECT a.*, s.country, s.product_group, s.scheduled_delivery_date
                FROM prediction_audits a
                LEFT JOIN shipments s ON a.shipment_id = s.id
                WHERE a.audit_id = ?
                """,
                (audit_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)

    def query_audits(
        self,
        country: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cursor-based pagination using monotonic timestamp."""
        query = """
            SELECT a.audit_id, a.model_name, a.late_delivery_risk, a.risk_probability,
                   a.risk_tier, a.country_tier, a.qos_latency_ms, a.created_at,
                   s.country, s.product_group, s.scheduled_delivery_date
            FROM prediction_audits a
            LEFT JOIN shipments s ON a.shipment_id = s.id
            WHERE 1=1
        """
        params: List[Any] = []

        if country:
            query += " AND s.country = ?"
            params.append(country)
        if risk_tier:
            query += " AND a.risk_tier = ?"
            params.append(risk_tier)
        if cursor:
            query += " AND a.created_at < ?"
            params.append(cursor)

        query += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(limit + 1)

        with self._db.session() as conn:
            rows = conn.execute(query, params).fetchall()
            results = [dict(r) for r in rows]

            next_cursor = None
            if len(results) > limit:
                results = results[:limit]
                next_cursor = results[-1]["created_at"]

            return {
                "items": results,
                "count": len(results),
                "next_cursor": next_cursor,
            }


class RelationalBatchJobRepository(IBatchJobRepository):
    """Concrete repository for tracking long-running asynchronous batch prediction tasks."""

    def __init__(self, db_engine: DatabaseEngine):
        self._db = db_engine

    def create_job(self, job_id: str, total_records: int, model_name: str) -> Dict[str, Any]:
        with self._db.session() as conn:
            conn.execute(
                """
                INSERT INTO batch_jobs (
                    job_id, total_records, model_name, status, processed_records
                ) VALUES (?, ?, ?, 'PENDING', 0)
                """,
                (job_id, total_records, model_name),
            )
        return {
            "job_id": job_id,
            "status": "PENDING",
            "total_records": total_records,
            "model_name": model_name,
        }

    def update_job_progress(
        self,
        job_id: str,
        processed_records: int,
        status: str,
        error_message: Optional[str] = None,
        results_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary_json = json.dumps(results_summary) if results_summary else None
        throughput = 0.0
        if results_summary and "throughput_samples_per_sec" in results_summary:
            throughput = float(results_summary["throughput_samples_per_sec"])

        with self._db.session() as conn:
            conn.execute(
                """
                UPDATE batch_jobs
                SET processed_records = ?, status = ?, error_message = ?,
                    results_summary = ?, throughput_samples_per_sec = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_id = ?
                """,
                (processed_records, status, error_message, summary_json, throughput, job_id),
            )

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._db.session() as conn:
            row = conn.execute(
                "SELECT * FROM batch_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            if res.get("results_summary"):
                try:
                    res["results_summary"] = json.loads(res["results_summary"])
                except Exception:
                    pass
            return res
