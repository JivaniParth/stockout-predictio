"""
Layer 2 Port: Predictor Interface.
Decouples application use cases from PyTorch, Scikit-learn, XGBoost, or ONNX runtimes.
"""

from abc import ABC, abstractmethod
import numpy as np


class IPredictor(ABC):
    """Abstract contract for all predictive model adapters."""

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Receives normalized feature matrix of shape (N, D).
        Returns positive class probabilities array of shape (N,).
        """
        pass

    @property
    @abstractmethod
    def optimal_threshold(self) -> float:
        """Returns the validation-calibrated decision threshold (tau*)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable canonical identifier of the model."""
        pass
