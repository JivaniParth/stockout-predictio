"""
Layer 2 Port: Telemetry Repository Interface.
Decouples application use cases from specific storage, metrics sinks, or Prometheus/DataDog providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ITelemetryRepository(ABC):
    """Abstract contract for persisting and retrieving QoS telemetry data."""

    @abstractmethod
    def record_latency(self, latency_ms: float) -> None:
        """Records a single inference latency sample."""
        pass

    @abstractmethod
    def get_qos_summary(self) -> Dict[str, Any]:
        """Calculates and returns QoS percentiles and buffer health statistics."""
        pass
