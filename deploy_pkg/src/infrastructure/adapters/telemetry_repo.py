"""
Layer 3 Adapter: In-Memory Bounded QoS Telemetry Repository.
Implements ITelemetryRepository port using an O(1) bounded circular buffer (collections.deque).
Guarantees strictly bounded memory footprint with zero leak in high-throughput cloud environments.
"""

from collections import deque
from typing import Dict, Any
import numpy as np
from src.application.ports.telemetry import ITelemetryRepository


class InMemoryTelemetryRepository(ITelemetryRepository):
    """
    In-memory bounded circular buffer implementation of ITelemetryRepository.
    Caps memory at max_samples to guarantee zero memory fragmentation or unbounded growth.
    """

    def __init__(self, max_samples: int = 1000):
        self._max_samples = max_samples
        self._history: deque = deque(maxlen=max_samples)

    def record_latency(self, latency_ms: float) -> None:
        """Records latency sample in O(1) time."""
        self._history.append(float(latency_ms))

    def get_qos_summary(self) -> Dict[str, Any]:
        """Calculates percentile distribution and queue statistics."""
        if not self._history:
            return {"status": "No inference requests recorded yet."}

        latencies = list(self._history)
        return {
            "Total_Buffer_Samples": len(latencies),
            "Buffer_Capacity": self._max_samples,
            "Latency_P50_ms": round(float(np.percentile(latencies, 50)), 3),
            "Latency_P95_ms": round(float(np.percentile(latencies, 95)), 3),
            "Latency_P99_ms": round(float(np.percentile(latencies, 99)), 3),
            "Latency_Mean_ms": round(float(np.mean(latencies)), 3),
            "Latency_Min_ms": round(float(np.min(latencies)), 3),
            "Latency_Max_ms": round(float(np.max(latencies)), 3),
        }

    def clear(self) -> None:
        """Resets the circular buffer."""
        self._history.clear()
