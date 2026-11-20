"""
Layer 3 Adapter: PyTorch Model Adapter.
Implements IPredictor port for PyTorch Neural Networks (Transformers, CNNs, LSTMs, Res-MLPs).
Encapsulates tensor conversions, no_grad scopes, and execution devices.
"""

from typing import Union
import numpy as np
import torch
from src.application.ports.predictor import IPredictor


class PyTorchModelAdapter(IPredictor):
    """Adapter bridging PyTorch nn.Module architectures to IPredictor contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        optimal_threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self._model = model.to(device)
        self._model.eval()
        self._model_name = model_name
        self._optimal_threshold = float(optimal_threshold)
        self._device = device

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self._device)
            output = self._model(tensor)
            # Ensure 1D numpy array output
            return output.cpu().numpy().flatten()

    @property
    def optimal_threshold(self) -> float:
        return self._optimal_threshold

    @property
    def model_name(self) -> str:
        return self._model_name
