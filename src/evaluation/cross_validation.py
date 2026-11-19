"""
Expanding-Window Time-Ordered Cross-Validation and Paired Bootstrap Confirmation.
Features:
- 4 expanding time-ordered folds over 2009-2013 training history
- Zero temporal leakage: every preprocessor and model is fit strictly on past years
- Out-of-Fold (OOF) prediction generation for meta-learning / stacking
- 1,000-sample paired bootstrap confirmation on held-out 2014 validation split
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, f1_score

from src.features.engineered_preprocessor import EngineeredPreprocessor


FOLDS_SPEC = [
    {"name": "Fold 1 (Train <= 2009, Val 2010)", "train_max_year": 2009, "val_year": 2010},
    {"name": "Fold 2 (Train <= 2010, Val 2011)", "train_max_year": 2010, "val_year": 2011},
    {"name": "Fold 3 (Train <= 2011, Val 2012)", "train_max_year": 2011, "val_year": 2012},
    {"name": "Fold 4 (Train <= 2012, Val 2013)", "train_max_year": 2012, "val_year": 2013},
]


def compute_fold_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Precision-Recall Area Under Curve."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    p, r, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(r, p))


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> Tuple[float, float, float]:
    """Finds decision threshold maximizing F_beta on validation predictions."""
    best_tau = 0.50
    best_f = -1.0
    best_rec = 0.0
    thresholds = np.linspace(0.05, 0.85, 81)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        rec = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        if beta != 1.0:
            p = float(np.sum((preds == 1) & (y_true == 1)) / max(np.sum(preds == 1), 1))
            f_b = (1.0 + beta**2) * (p * rec) / max((beta**2 * p + rec), 1e-6)
            score = f_b
        else:
            score = f

        if score > best_f:
            best_f = score
            best_tau = t
            best_rec = rec

    return float(best_tau), float(best_rec), float(best_f)


def evaluate_expanding_window_cv(
    train_df: pd.DataFrame,
    model_factory: Callable[[int, List[int]], Any],
    categorical_mode: str = "native",
    smoothing_m: float = 10.0,
    target_col: str = "Late_Delivery_Risk",
) -> Dict[str, Any]:
    """
    Executes expanding-window cross-validation over historical 2009-2013 data.
    Fits preprocessor and model strictly on each expanding fold.
    Returns mean and std of out-of-time CV PR-AUC and OOF predictions.
    """
    df = train_df.copy()
    if "Scheduled_Year" not in df.columns:
        df["Scheduled_Year"] = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed").dt.year

    fold_metrics = []
    oof_indices = []
    oof_probs = []
    oof_targets = []

    for f in FOLDS_SPEC:
        tr_mask = df["Scheduled_Year"] <= f["train_max_year"]
        val_mask = df["Scheduled_Year"] == f["val_year"]

        fold_train_df = df[tr_mask].copy()
        fold_val_df = df[val_mask].copy()

        # Fit preprocessor strictly on fold training data
        prep = EngineeredPreprocessor(categorical_mode=categorical_mode, smoothing_m=smoothing_m)
        prep.fit(fold_train_df)

        X_tr = prep.transform(fold_train_df)
        y_tr = fold_train_df[target_col].values

        X_val = prep.transform(fold_val_df)
        y_val = fold_val_df[target_col].values

        cat_indices = prep.get_categorical_indices() if categorical_mode == "native" else []

        # Train model
        model = model_factory(X_tr.shape[1], cat_indices)
        if hasattr(model, "fit_with_early_stopping"):
            model.fit_with_early_stopping(X_tr, y_tr, X_val, y_val)
        elif hasattr(model, "fit"):
            model.fit(X_tr, y_tr)

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)
            if probs.ndim == 2:
                probs = probs[:, 1]
        elif hasattr(model, "predict"):
            probs = model.predict(X_val)
        else:
            raise AttributeError("Model has neither predict_proba nor predict method.")

        pr_auc = compute_fold_pr_auc(y_val, probs)
        roc_auc = float(roc_auc_score(y_val, probs)) if len(np.unique(y_val)) > 1 else 0.5
        tau, rec, f1 = find_optimal_threshold(y_val, probs)

        fold_metrics.append({
            "fold": f["name"],
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall": rec,
            "f1": f1,
            "tau_star": tau,
            "train_size": len(fold_train_df),
            "val_size": len(fold_val_df),
            "val_positives": int(y_val.sum()),
        })

        oof_indices.extend(fold_val_df.index.tolist())
        oof_probs.extend(probs.tolist())
        oof_targets.extend(y_val.tolist())

    pr_aucs = [m["pr_auc"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    recalls = [m["recall"] for m in fold_metrics]

    return {
        "fold_metrics": fold_metrics,
        "cv_pr_auc_mean": float(np.mean(pr_aucs)),
        "cv_pr_auc_std": float(np.std(pr_aucs)),
        "cv_f1_mean": float(np.mean(f1s)),
        "cv_recall_mean": float(np.mean(recalls)),
        "oof_df": pd.DataFrame({
            "index": oof_indices,
            "y_true": oof_targets,
            "y_prob": oof_probs,
        }).set_index("index"),
    }


def paired_bootstrap_test_2014(
    y_true: np.ndarray,
    probs_candidate: np.ndarray,
    probs_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes 1,000-sample paired bootstrap on identical 2014 validation rows:
    Delta_PR_AUC = Candidate_PR_AUC - Baseline_PR_AUC.
    """
    np.random.seed(seed)
    N = len(y_true)
    indices = np.arange(N)

    deltas_pr_auc = []
    deltas_f1 = []

    for _ in range(n_bootstraps):
        b_idx = np.random.choice(indices, size=N, replace=True)
        b_true = y_true[b_idx]
        if len(np.unique(b_true)) < 2:
            continue

        b_cand_p = probs_candidate[b_idx]
        b_base_p = probs_baseline[b_idx]

        pr_cand = compute_fold_pr_auc(b_true, b_cand_p)
        pr_base = compute_fold_pr_auc(b_true, b_base_p)
        deltas_pr_auc.append(pr_cand - pr_base)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    mean_delta = float(np.mean(deltas_pr_auc))
    ci_lower = float(np.percentile(deltas_pr_auc, lower_pct))
    ci_upper = float(np.percentile(deltas_pr_auc, upper_pct))

    return {
        "mean_delta_pr_auc": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_str": f"[{ci_lower:+.4f}, {ci_upper:+.4f}]",
        "statistically_superior": bool(ci_lower > 0.0),
    }
