"""
Classical Machine Learning Baselines:
1. Logistic Regression (L2-regularized, Class-Balanced)
2. Random Forest Classifier
3. Extreme Gradient Boosting (XGBoost with scale_pos_weight for class imbalance)
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
from pathlib import Path
from src.config import MODELS_DIR, RANDOM_SEED


class BaselineSuite:
    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}

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

        print(f"[-] Training XGBoost (scale_pos_weight={scale_pos:.2f})...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train)
        self.models["XGBoost"] = xgb_clf

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
