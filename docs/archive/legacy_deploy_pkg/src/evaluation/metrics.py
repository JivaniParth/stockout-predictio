"""
Comprehensive Machine Learning Evaluation Metrics for Pharmaceutical Stockout Prediction.
Includes standard, imbalanced, and probabilistic calibration metrics.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    confusion_matrix,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes a full suite of classification and calibration metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)

    # ROC-AUC & PR-AUC
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recalls, precisions)

    # Calibration
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision_Late": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_Late": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Specificity": float(specificity),
        "F1_Late": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "Brier_Score": float(brier),
        "Threshold": float(threshold),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }

    return metrics
