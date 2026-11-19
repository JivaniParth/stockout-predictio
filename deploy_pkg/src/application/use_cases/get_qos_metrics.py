"""
Layer 2 Use Case: QoS Telemetry Extraction.
"""

from typing import Dict, Any
from src.application.ports.telemetry import ITelemetryRepository


class GetQoSMetricsUseCase:
    """Retrieves QoS telemetry percentiles and queue statistics."""

    def __init__(self, telemetry_repo: ITelemetryRepository):
        self.telemetry_repo = telemetry_repo

    def execute(self) -> Dict[str, Any]:
        return self.telemetry_repo.get_qos_summary()
