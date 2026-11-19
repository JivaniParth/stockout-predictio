"""
Classical Machine Learning Baselines:
1. Logistic Regression (L2-regularized, Class-Balanced)
2. Random Forest Classifier (Multi-core CPU)
3. Extreme Gradient Boosting (XGBoost with GPU acceleration and CPU fallback)
"""

from typing import Dict, Any, Tuple, Optional
import os
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from pathlib import Path
from src.config import MODELS_DIR, RANDOM_SEED
from src.utils.device_manager import probe_xgboost_gpu

logger = logging.getLogger(__name__)


class BaselineSuite:
    def __init__(self, random_state: int = RANDOM_SEED, device: Optional[str] = "auto"):
        self.random_state = random_state
        self.requested_device = str(device).lower() if device else "auto"
        self.models: Dict[str, Any] = {}
        self.active_devices: Dict[str, str] = {}
        self.cpu_cores = os.cpu_count() or 4

    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all classical baselines using training split only."""
        # Calculate positive class weight for imbalanced learning
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos = num_neg / max(num_pos, 1)

        print(f"[-] Training Logistic Regression (class_weight='balanced')...")
        lr = LogisticRegression(
            penalty="l2",
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
        )
        lr.fit(X_train, y_train)
        self.models["LogisticRegression"] = lr
        self.active_devices["LogisticRegression"] = f"CPU ({self.cpu_cores} threads)"

        print(f"[-] Training Random Forest (150 trees, class_weight='balanced')...")
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train)
        self.models["RandomForest"] = rf
        self.active_devices["RandomForest"] = f"CPU ({self.cpu_cores} threads)"

        # Resolve XGBoost Device
        use_gpu = False
        if self.requested_device in ["cuda", "gpu"]:
            use_gpu = True
        elif self.requested_device == "auto":
            is_avail, _ = probe_xgboost_gpu()
            use_gpu = is_avail

        xgb_params: Dict[str, Any] = {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": self.random_state,
            "tree_method": "hist",
        }

        if use_gpu:
            xgb_params["device"] = "cuda"
            active_xgb_device = "cuda"
            print(f"[-] Training XGBoost on GPU (device='cuda', scale_pos_weight={scale_pos:.2f})...")
        else:
            xgb_params["device"] = "cpu"
            xgb_params["n_jobs"] = -1
            active_xgb_device = f"CPU ({self.cpu_cores} threads)"
            print(f"[-] Training XGBoost on CPU (device='cpu', scale_pos_weight={scale_pos:.2f})...")

        xgb_clf = xgb.XGBClassifier(**xgb_params)
        try:
            xgb_clf.fit(X_train, y_train)
        except Exception as e:
            if use_gpu:
                logger.warning(
                    f"[BaselineSuite] XGBoost GPU fit failed ({e}). Falling back to multi-core CPU..."
                )
                active_xgb_device = f"CPU (fallback, {self.cpu_cores} threads)"
                xgb_params["device"] = "cpu"
                xgb_params["n_jobs"] = -1
                xgb_clf = xgb.XGBClassifier(**xgb_params)
                xgb_clf.fit(X_train, y_train)
            else:
                raise e

        self.models["XGBoost"] = xgb_clf
        self.active_devices["XGBoost"] = active_xgb_device

    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Returns positive class probability (Late Delivery Risk)."""
        model = self.models[model_name]
        return model.predict_proba(X)[:, 1]

    def predict(self, model_name: str, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions based on decision threshold."""
        probs = self.predict_proba(model_name, X)
        return (probs >= threshold).astype(int)

    def save_model(self, model_name: str, filepath: Path = None):
        """Persists trained baseline model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.models[model_name], f)

    def load_model(self, model_name: str, filepath: Path = None):
        """Loads baseline model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.pkl"
        with open(filepath, "rb") as f:
            self.models[model_name] = pickle.load(f)
