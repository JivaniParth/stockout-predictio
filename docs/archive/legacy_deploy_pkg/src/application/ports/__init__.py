"""
Application Layer Ports.
Abstract interfaces defining boundaries for outbound adapters.
"""

from src.application.ports.predictor import IPredictor
from src.application.ports.transformer import IFeatureTransformer
from src.application.ports.telemetry import ITelemetryRepository
from src.application.ports.cache import ICacheService
from src.application.ports.repository import IShipmentRepository, IBatchJobRepository
from src.application.ports.queue import ITaskQueue

__all__ = [
    "IPredictor",
    "IFeatureTransformer",
    "ITelemetryRepository",
    "ICacheService",
    "IShipmentRepository",
    "IBatchJobRepository",
    "ITaskQueue",
]
