"""
Two-Stage Delay Prediction Formulation.
Stage 1: Predicts continuous days of delivery delay (Delay_Days) using a regression model.
Stage 2: Feeds the expected delay days as an auxiliary predictive feature into the binary classifier.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc


class TwoStageDelayModel:
    """
    Two-stage model combining continuous delay magnitude prediction with binary classification.
    """

    def __init__(self, cat_indices: Optional[List[int]] = None, random_state: int = 42):
        self.cat_indices = cat_indices or []
        self.random_state = random_state

        # Stage 1: Regression model for delay magnitude
        self.regressor = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )

        # Stage 2: Classifier combining base features + predicted delay magnitude
        self.classifier = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.0,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        delay_days_train: Optional[np.ndarray] = None,
        delay_days_val: Optional[np.ndarray] = None,
    ):
        if delay_days_train is None:
            delay_days_train = np.where(y_train == 1, 14.0, -5.0)
        # 1. Fit Stage 1 Regressor on continuous delay magnitude
        cat_feat = self.cat_indices if len(self.cat_indices) > 0 else "auto"
        self.regressor.fit(X_train, delay_days_train, categorical_feature=cat_feat)

        # 2. Predict expected delay days
        pred_delay_tr = self.regressor.predict(X_train).reshape(-1, 1)

        # Append auxiliary regression feature
        X_tr_augmented = np.hstack([X_train, pred_delay_tr])

        eval_set = None
        callbacks = []
        if X_val is not None and y_val is not None:
            pred_delay_v = self.regressor.predict(X_val).reshape(-1, 1)
            X_v_augmented = np.hstack([X_val, pred_delay_v])
            eval_set = [(X_v_augmented, y_val)]
            callbacks.append(lgb.early_stopping(15, verbose=False))

        # 3. Fit Stage 2 Classifier
        self.classifier.fit(
            X_tr_augmented,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=cat_feat,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred_delay = self.regressor.predict(X).reshape(-1, 1)
        X_augmented = np.hstack([X, pred_delay])
        return self.classifier.predict_proba(X_augmented)
