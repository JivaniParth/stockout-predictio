"""
Application Layer Port: Repository Interfaces.
Defines contracts for transactional persistence and query operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IShipmentRepository(ABC):
    """Abstract persistence contract for audited shipments and predictions."""

    @abstractmethod
    def save_audit(
        self,
        shipment_data: Dict[str, Any],
        prediction_result: Dict[str, Any],
        idempotency_key: Optional[str] = None,
        tenant_id: str = "default_tenant",
    ) -> str:
        """Persists shipment data and model prediction into immutable audit trail."""
        pass

    @abstractmethod
    def get_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an audit entry by its unique identifier."""
        pass

    @abstractmethod
    def query_audits(
        self,
        country: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries audited shipments with cursor-based pagination."""
        pass


class IBatchJobRepository(ABC):
    """Abstract persistence contract for asynchronous batch jobs."""

    @abstractmethod
    def create_job(self, job_id: str, total_records: int, model_name: str) -> Dict[str, Any]:
        """Registers a new batch job in PENDING state."""
        pass

    @abstractmethod
    def update_job_progress(
        self,
        job_id: str,
        processed_records: int,
        status: str,
        error_message: Optional[str] = None,
        results_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Updates progress and status of a batch job."""
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves status and results of a batch job."""
        pass
