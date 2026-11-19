"""
Layer 2 Port: Feature Transformer Interface.
Decouples application use cases from tabular encoders, imputer strategies, and scaling engines.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from src.domain.entities import ShipmentEntity


class IFeatureTransformer(ABC):
    """Abstract contract for transforming domain entities into model-ready numerical tensors."""

    @abstractmethod
    def transform_entities(self, entities: List[ShipmentEntity]) -> np.ndarray:
        """Transforms a batch of domain shipment entities into a 2D float array (N, Features)."""
        pass
