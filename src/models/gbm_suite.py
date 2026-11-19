"""
High-Performance Gradient-Boosted Decision Tree Suite with GPU Acceleration
and Automatic Multi-Core CPU Fallback.
Features:
- LightGBM with native categorical handling, scale_pos_weight, and GPU probing
- CatBoost with native categorical handling (cat_features), class weighting, and GPU acceleration
- XGBoost with histogram binning, scale_pos_weight, and CUDA acceleration
- Two-tier architecture: proactive hardware probing + reactive fit try-except fallback
- Full active_device and fallback_occurred telemetry tracking
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import copy
import logging
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna

from src.utils.device_manager import (
    probe_xgboost_gpu,
    probe_catboost_gpu,
    probe_lightgbm_gpu,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMWrapper:
    """Wrapper around LightGBM Classifier with categorical support, GPU probing, and CPU fallback."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        cat_indices: Optional[List[int]] = None,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        device: Optional[str] = "auto",
    ):
        self.cat_indices = cat_indices or []
        self.random_state = random_state
        self.requested_device = str(device).lower() if device else "auto"
        self.fallback_occurred = False
        self.user_params = copy.deepcopy(params) if params else {}

        # Proactive device resolution
        use_gpu = False
        if self.requested_device in ["cuda", "gpu"]:
            use_gpu = True
        elif self.requested_device == "auto":
            is_avail, _ = probe_lightgbm_gpu()
            use_gpu = is_avail

        self.default_params: Dict[str, Any] = {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "random_state": random_state,
        }

        if use_gpu:
            self.default_params["device"] = "cuda"
            self.active_device = "cuda"
        else:
            self.default_params["device"] = "cpu"
            self.default_params["n_jobs"] = -1
            self.active_device = "cpu"

        if scale_pos_weight is not None:
            self.default_params["scale_pos_weight"] = float(scale_pos_weight)
        self.default_params.update(self.user_params)

        # Update active_device if user passed device in params
        if "device" in self.user_params:
            dev_val = str(self.user_params["device"]).lower()
            if dev_val in ["cuda", "gpu"]:
                self.active_device = dev_val
            else:
                self.active_device = "cpu"

        self.model = lgb.LGBMClassifier(**self.default_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=15, verbose=False))

        cat_feature = self.cat_indices if len(self.cat_indices) > 0 else "auto"

        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                callbacks=callbacks,
                categorical_feature=cat_feature,
            )
        except Exception as e:
            if self.active_device in ["cuda", "gpu"]:
                logger.warning(
                    f"[LightGBMWrapper] GPU fit failed with error: {e}. "
                    "Gracefully falling back to multi-core CPU parallel processing (device='cpu', n_jobs=-1)..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                cpu_params = copy.deepcopy(self.default_params)
                cpu_params["device"] = "cpu"
                cpu_params["n_jobs"] = -1
                self.model = lgb.LGBMClassifier(**cpu_params)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    callbacks=callbacks,
                    categorical_feature=cat_feature,
                )
            else:
                raise e
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class CatBoostWrapper:
    """Wrapper around CatBoost Classifier with native cat_features, GPU acceleration, and CPU fallback."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        cat_indices: Optional[List[int]] = None,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        device: Optional[str] = "auto",
    ):
        self.cat_indices = cat_indices or []
        self.random_state = random_state
        self.requested_device = str(device).lower() if device else "auto"
        self.fallback_occurred = False
        self.user_params = copy.deepcopy(params) if params else {}

        # Proactive device resolution
        use_gpu = False
        if self.requested_device in ["cuda", "gpu"]:
            use_gpu = True
        elif self.requested_device == "auto":
            is_avail, _ = probe_catboost_gpu()
            use_gpu = is_avail

        import tempfile
        catboost_dir = Path(__file__).resolve().parent.parent.parent / "catboost_info"
        try:
            tmp_sub = catboost_dir / "tmp"
            tmp_sub.mkdir(parents=True, exist_ok=True)
            test_f = tmp_sub / ".test_write"
            test_f.write_text("ok")
            test_f.unlink()
            default_train_dir = str(catboost_dir)
        except Exception:
            default_train_dir = tempfile.gettempdir()

        self.default_params: Dict[str, Any] = {
            "iterations": 300,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "verbose": False,
            "random_seed": random_state,
            "train_dir": default_train_dir,
        }

        if use_gpu:
            self.default_params["task_type"] = "GPU"
            self.active_device = "GPU"
        else:
            self.default_params["task_type"] = "CPU"
            self.default_params["thread_count"] = -1
            self.active_device = "CPU"

        if scale_pos_weight is not None:
            self.default_params["auto_class_weights"] = "Balanced"
        self.default_params.update(self.user_params)

        # Update active_device if user passed task_type in params
        if "task_type" in self.user_params:
            tt = str(self.user_params["task_type"]).upper()
            self.active_device = "GPU" if tt == "GPU" else "CPU"

        self.model = cb.CatBoostClassifier(**self.default_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        eval_set = None
        if len(self.cat_indices) > 0:
            import pandas as pd
            X_tr = pd.DataFrame(X_train)
            for idx in self.cat_indices:
                X_tr[idx] = X_tr[idx].astype(int).astype(str)

            if X_val is not None and y_val is not None:
                X_v = pd.DataFrame(X_val)
                for idx in self.cat_indices:
                    X_v[idx] = X_v[idx].astype(int).astype(str)
                eval_set = (X_v, y_val)
        else:
            X_tr = X_train
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)

        try:
            self.model.fit(
                X_tr,
                y_train,
                cat_features=self.cat_indices if len(self.cat_indices) > 0 else None,
                eval_set=eval_set,
                early_stopping_rounds=15,
                verbose=False,
            )
        except Exception as e:
            if "GPU" in self.active_device:
                logger.warning(
                    f"[CatBoostWrapper] GPU fit failed with error: {e}. "
                    "Gracefully falling back to multi-core CPU parallel processing (task_type='CPU', thread_count=-1)..."
                )
                self.fallback_occurred = True
                self.active_device = "CPU (fallback)"
                cpu_params = copy.deepcopy(self.default_params)
                cpu_params["task_type"] = "CPU"
                cpu_params["thread_count"] = -1
                self.model = cb.CatBoostClassifier(**cpu_params)
                self.model.fit(
                    X_tr,
                    y_train,
                    cat_features=self.cat_indices if len(self.cat_indices) > 0 else None,
                    eval_set=eval_set,
                    early_stopping_rounds=15,
                    verbose=False,
                )
            else:
                raise e
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(self.cat_indices) > 0:
            import pandas as pd
            X_eval = pd.DataFrame(X)
            for idx in self.cat_indices:
                X_eval[idx] = X_eval[idx].astype(int).astype(str)
        else:
            X_eval = X
        return self.model.predict_proba(X_eval)


class XGBoostWrapper:
    """Wrapper around XGBoost Classifier with histogram binning, GPU acceleration, and CPU fallback."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        device: Optional[str] = "auto",
    ):
        self.random_state = random_state
        self.requested_device = str(device).lower() if device else "auto"
        self.fallback_occurred = False
        self.user_params = copy.deepcopy(params) if params else {}

        # Proactive device resolution
        use_gpu = False
        if self.requested_device in ["cuda", "gpu"]:
            use_gpu = True
        elif self.requested_device == "auto":
            is_avail, _ = probe_xgboost_gpu()
            use_gpu = is_avail

        self.default_params: Dict[str, Any] = {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": random_state,
            "tree_method": "hist",
        }

        if use_gpu:
            self.default_params["device"] = "cuda"
            self.active_device = "cuda"
        else:
            self.default_params["device"] = "cpu"
            self.default_params["n_jobs"] = -1
            self.active_device = "cpu"

        if scale_pos_weight is not None:
            self.default_params["scale_pos_weight"] = float(scale_pos_weight)
        self.default_params.update(self.user_params)

        # Update active_device if user passed device in params
        if "device" in self.user_params:
            dev_val = str(self.user_params["device"]).lower()
            if dev_val in ["cuda", "gpu"]:
                self.active_device = "cuda"
            else:
                self.active_device = "cpu"

        self.model = xgb.XGBClassifier(**self.default_params)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        try:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
        except Exception as e:
            if "cuda" in self.active_device:
                logger.warning(
                    f"[XGBoostWrapper] GPU fit failed with error: {e}. "
                    "Gracefully falling back to multi-core CPU parallel processing (device='cpu', n_jobs=-1)..."
                )
                self.fallback_occurred = True
                self.active_device = "cpu (fallback)"
                cpu_params = copy.deepcopy(self.default_params)
                cpu_params["device"] = "cpu"
                cpu_params["n_jobs"] = -1
                self.model = xgb.XGBClassifier(**cpu_params)
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=eval_set,
                    verbose=False,
                )
            else:
                raise e
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
