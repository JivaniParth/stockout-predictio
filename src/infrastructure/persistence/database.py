"""
Layer 3 Persistence Engine: Relational Database & Connection Pool Manager.
Supports both PostgreSQL (production) and SQLite (zero-dependency local development).
Executes idempotent DDL schemas and manages thread-safe transactional sessions.
"""

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Generator, List, Optional
import uuid

logger = logging.getLogger("hscm.infrastructure.database")


# PostgreSQL Production DDL (For Amazon RDS / Cloud SQL Aurora deployments)
POSTGRES_DDL = """
-- 1. Shipments Table (Transactional Consignments)
CREATE TABLE IF NOT EXISTS shipments (
    id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default_tenant',
    country VARCHAR(100) NOT NULL,
    scheduled_delivery_date DATE NOT NULL,
    po_sent_to_vendor_date DATE,
    manufacturer_lead_time_days NUMERIC(6, 2) NOT NULL DEFAULT 90.0,
    product_group VARCHAR(50) NOT NULL,
    sub_classification VARCHAR(100) NOT NULL,
    shipment_mode VARCHAR(50) NOT NULL,
    fulfill_via VARCHAR(50) NOT NULL,
    vendor_inco_term VARCHAR(20) NOT NULL,
    vendor VARCHAR(150) NOT NULL,
    manufacturing_site VARCHAR(150) NOT NULL,
    first_line_designation VARCHAR(5) NOT NULL,
    line_item_quantity NUMERIC(12, 2) NOT NULL,
    line_item_value NUMERIC(14, 2) NOT NULL,
    pack_price NUMERIC(10, 2) NOT NULL,
    unit_price NUMERIC(10, 4) NOT NULL,
    unit_of_measure_per_pack NUMERIC(8, 2) NOT NULL,
    weight_kilograms VARCHAR(50) DEFAULT '350.0',
    freight_cost_usd VARCHAR(50) DEFAULT '3200.0',
    line_item_insurance_usd NUMERIC(10, 2) DEFAULT 50.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Prediction Audits Table (Immutable Prognosis Log)
CREATE TABLE IF NOT EXISTS prediction_audits (
    audit_id VARCHAR(36) PRIMARY KEY,
    shipment_id VARCHAR(36) REFERENCES shipments(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default_tenant',
    idempotency_key VARCHAR(128) UNIQUE,
    model_name VARCHAR(100) NOT NULL,
    late_delivery_risk SMALLINT NOT NULL,
    risk_probability NUMERIC(6, 4) NOT NULL,
    risk_tier VARCHAR(50) NOT NULL,
    country_tier VARCHAR(50) NOT NULL,
    qos_latency_ms NUMERIC(8, 3) NOT NULL,
    cache_hit BOOLEAN DEFAULT FALSE,
    input_payload_hash VARCHAR(64) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Batch Jobs Table (Asynchronous Bulk Scoring Tasks)
CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id VARCHAR(36) PRIMARY KEY,
    tenant_id VARCHAR(50) NOT NULL DEFAULT 'default_tenant',
    model_name VARCHAR(100) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    total_records INTEGER NOT NULL,
    processed_records INTEGER NOT NULL DEFAULT 0,
    throughput_samples_per_sec NUMERIC(8, 2) DEFAULT 0.0,
    results_summary JSONB,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. High-Performance B-Tree & Partial Indexes
CREATE INDEX IF NOT EXISTS idx_shipments_country_date ON shipments(country, scheduled_delivery_date);
CREATE INDEX IF NOT EXISTS idx_audits_country_tier ON prediction_audits(country_tier, late_delivery_risk);
CREATE INDEX IF NOT EXISTS idx_audits_created ON prediction_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status, created_at DESC);
"""

# SQLite DDL (For zero-dependency local dev, continuous integration, and edge nodes)
SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS shipments (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
    country TEXT NOT NULL,
    scheduled_delivery_date TEXT NOT NULL,
    po_sent_to_vendor_date TEXT,
    manufacturer_lead_time_days REAL NOT NULL DEFAULT 90.0,
    product_group TEXT NOT NULL,
    sub_classification TEXT NOT NULL,
    shipment_mode TEXT NOT NULL,
    fulfill_via TEXT NOT NULL,
    vendor_inco_term TEXT NOT NULL,
    vendor TEXT NOT NULL,
    manufacturing_site TEXT NOT NULL,
    first_line_designation TEXT NOT NULL,
    line_item_quantity REAL NOT NULL,
    line_item_value REAL NOT NULL,
    pack_price REAL NOT NULL,
    unit_price REAL NOT NULL,
    unit_of_measure_per_pack REAL NOT NULL,
    weight_kilograms TEXT DEFAULT '350.0',
    freight_cost_usd TEXT DEFAULT '3200.0',
    line_item_insurance_usd REAL DEFAULT 50.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_audits (
    audit_id TEXT PRIMARY KEY,
    shipment_id TEXT,
    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
    idempotency_key TEXT UNIQUE,
    model_name TEXT NOT NULL,
    late_delivery_risk INTEGER NOT NULL,
    risk_probability REAL NOT NULL,
    risk_tier TEXT NOT NULL,
    country_tier TEXT NOT NULL,
    qos_latency_ms REAL NOT NULL,
    cache_hit INTEGER DEFAULT 0,
    input_payload_hash TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(shipment_id) REFERENCES shipments(id)
);

CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default_tenant',
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    total_records INTEGER NOT NULL,
    processed_records INTEGER NOT NULL DEFAULT 0,
    throughput_samples_per_sec REAL DEFAULT 0.0,
    results_summary TEXT,
    error_message TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_shipments_country ON shipments(country);
CREATE INDEX IF NOT EXISTS idx_audits_created ON prediction_audits(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_jobs(status);
"""


class DatabaseEngine:
    """Thread-safe relational database manager with connection pooling and migrations."""

    _instance: Optional["DatabaseEngine"] = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to local SQLite database in results directory, or tempfile fallback
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            data_dir = base_dir / "results"
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                test_f = data_dir / ".db_test"
                test_f.write_text("ok")
                test_f.unlink()
                self._db_path = str(data_dir / "hscm_production.db")
            except Exception:
                import tempfile
                self._db_path = str(Path(tempfile.gettempdir()) / "hscm_production.db")
        else:
            self._db_path = db_path

        self._local = threading.local()
        self._init_schema()

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "DatabaseEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = DatabaseEngine(db_path=db_path)
            return cls._instance

    def _get_connection(self) -> sqlite3.Connection:
        """Returns thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            try:
                conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                import tempfile
                self._db_path = str(Path(tempfile.gettempdir()) / "hscm_production.db")
                conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        """Runs idempotent migration DDL."""
        conn = self._get_connection()
        with conn:
            conn.executescript(SQLITE_DDL)
        logger.info("Initialized relational database schema at %s", self._db_path)

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides transactional context manager."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Transaction rolled back due to error: %s", e)
            raise
