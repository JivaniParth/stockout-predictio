"""
Layer 3 Adapter: Scikit-Learn & XGBoost Model Adapter.
Implements IPredictor port for Tree Ensembles and Linear Baselines.
"""

from typing import Any
import numpy as np
from src.application.ports.predictor import IPredictor


class SklearnModelAdapter(IPredictor):
    """Adapter bridging Scikit-learn and XGBoost estimators to IPredictor contract."""

    def __init__(
        self,
        estimator: Any,
        model_name: str,
        optimal_threshold: float = 0.5,
    ):
        self._estimator = estimator
        self._model_name = model_name
        self._optimal_threshold = float(optimal_threshold)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probs = self._estimator.predict_proba(features)
        # Extract positive class probabilities (Column 1)
        return probs[:, 1].flatten()

    @property
    def optimal_threshold(self) -> float:
        return self._optimal_threshold

    @property
    def model_name(self) -> str:
        return self._model_name
