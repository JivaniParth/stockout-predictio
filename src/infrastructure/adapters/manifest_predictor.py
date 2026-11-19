"""
Manifest-Driven Model Serving Adapter.
Implements IPredictor port by loading model weights and execution settings exclusively from artifacts/manifest.json.
Features:
- ONNX Runtime accelerated forward pass for both PyTorch and XGBoost models
- Dynamic batching support
- Calibrator application (Platt / Isotonic)
- Per-tier and global decision threshold retrieval
- Top-k feature attribution computation
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import onnxruntime as ort
import pickle

from src.application.ports.predictor import IPredictor


class ManifestModelPredictor(IPredictor):
    """
    Adapter strictly loading model, calibration, and threshold rules from artifacts/manifest.json.
    Supports both neural (PyTorch) and gradient-boosted (XGBoost) ONNX runtimes.
    """

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Serving manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self._model_name = self.manifest.get("model_name", "ChampionModel")
        self._tau_star = float(self.manifest.get("tau_star", 0.21))
        self._per_tier_thresholds = self.manifest.get("per_tier_thresholds", {})
        self._calibrator_type = self.manifest.get("calibrator_type", "isotonic")
        self._calibrator = None

        # Check for optional serialized calibrator
        cal_path = self.manifest_path.parent / f"{self._model_name}_calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                self._calibrator = pickle.load(f)

        # Load ONNX session
        onnx_file = self.manifest.get("onnx_artifact")
        if not onnx_file:
            raise ValueError(f"No onnx_artifact specified in manifest: {manifest_path}")
        onnx_path = self.manifest_path.parent / onnx_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"Exported ONNX model artifact not found at: {onnx_path}")

        # Set threadpool to avoid contention
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Executes forward pass using ONNX Runtime with calibration."""
        feat_array = np.asarray(features, dtype=np.float32)
        if feat_array.ndim == 1:
            feat_array = feat_array.reshape(1, -1)

        ort_outs = self._session.run(None, {self._input_name: feat_array})
        # Check if output is from an XGBoost ONNX model (output 0 is label, output 1 is probability matrix)
        if len(ort_outs) > 1 and hasattr(ort_outs[1], "shape") and len(ort_outs[1].shape) == 2:
            raw_probs = ort_outs[1][:, 1].flatten()
        else:
            raw_probs = ort_outs[0].flatten()

        if self._calibrator is not None:
            return self._calibrator.predict_proba(raw_probs)
        return raw_probs

    @property
    def optimal_threshold(self) -> float:
        return self._tau_star

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self.manifest.get("git_commit", "v1.0.0")

    @property
    def per_tier_thresholds(self) -> Dict[str, float]:
        return self._per_tier_thresholds

    def get_threshold_for_tier(self, tier_name: str) -> float:
        return float(self._per_tier_thresholds.get(tier_name, self._tau_star))
