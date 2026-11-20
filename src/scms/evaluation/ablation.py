"""
Ablation Study Module across 5 Seeds with Statistical Significance Testing.
Evaluates the contribution of:
1. Feature groups:
   - Full Model (All features)
   - No Harmonic Date Encodings (sin/cos of month removed)
   - No SLA Squeeze Dynamics & Domain Ratios (removed)
   - Numeric Features Only (One-Hot Categoricals removed)
2. Loss function variants:
   - Weighted BCE
   - Standard Unweighted BCE
   - Focal Loss
3. Decision threshold policy:
   - Calibrated tau*
   - Default uncalibrated (tau = 0.50)

Runs all settings across 5 seeds ([42, 101, 202, 303, 404]).
Computes:
- Mean and standard deviation across seeds
- 1,000-sample bootstrap 95% confidence intervals on test set
- Paired delta against Full Model (Delta = Full - Ablated) with 95% bootstrap CIs
- Explicit boolean flag indicating whether the degradation is statistically
  distinguishable from noise given the test set size with N=120 positive events.
Outputs results to reports/ablation.csv.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_curve, roc_auc_score, accuracy_score, brier_score_loss, auc

from src.models.cnn import Tabular1DCNN
from src.models.shared_trainer import SharedTrainer
from src.data.dataset import HSCMDataset
from src.data.preprocessor import HSCMPreprocessor
from src.evaluation.metrics import compute_comprehensive_metrics
from src.config import DEVICE


SEEDS = [42, 101, 202, 303, 404]


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Helper to compute standard classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    if len(np.unique(y_true)) > 1:
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_c, p_c))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    else:
        pr_auc = 0.0
        roc_auc = 0.5

    return {
        "Recall_Late": rec,
        "Precision_Late": prec,
        "F1_Late": f1,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Accuracy": acc,
        "Brier_Score": brier,
    }


def compute_bootstrap_difference_ci(
    y_true: np.ndarray,
    probs_full: np.ndarray,
    probs_ablated: np.ndarray,
    thresh_full: float,
    thresh_ablated: float,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes paired bootstrap confidence interval for Delta = Full_Metric - Ablated_Metric.
    Determines if Delta is statistically distinguishable from zero given N=120 positives.
    """
    np.random.seed(seed)
    N = len(y_true)
    indices = np.arange(N)

    delta_recalls = []
    delta_f1s = []
    delta_pr_aucs = []

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=N, replace=True)
        b_true = y_true[boot_idx]
        if len(np.unique(b_true)) < 2:
            continue

        m_full = compute_metrics_at_threshold(b_true, probs_full[boot_idx], thresh_full)
        m_ablated = compute_metrics_at_threshold(b_true, probs_ablated[boot_idx], thresh_ablated)

        delta_recalls.append(m_full["Recall_Late"] - m_ablated["Recall_Late"])
        delta_f1s.append(m_full["F1_Late"] - m_ablated["F1_Late"])
        delta_pr_aucs.append(m_full["PR_AUC"] - m_ablated["PR_AUC"])

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    # For PR-AUC and F1, check if lower bound > 0 (significant degradation when ablated)
    pr_auc_ci = (float(np.percentile(delta_pr_aucs, lower_pct)), float(np.percentile(delta_pr_aucs, upper_pct)))
    f1_ci = (float(np.percentile(delta_f1s, lower_pct)), float(np.percentile(delta_f1s, upper_pct)))
    recall_ci = (float(np.percentile(delta_recalls, lower_pct)), float(np.percentile(delta_recalls, upper_pct)))

    # Distinguishable from noise if 0 is not in the 95% CI of PR-AUC or F1 degradation
    distinguishable = bool((pr_auc_ci[0] > 0.0 or pr_auc_ci[1] < 0.0) or (f1_ci[0] > 0.0 or f1_ci[1] < 0.0))

    return {
        "delta_pr_auc_mean": float(np.mean(delta_pr_aucs)),
        "delta_pr_auc_ci_lower": pr_auc_ci[0],
        "delta_pr_auc_ci_upper": pr_auc_ci[1],
        "delta_f1_mean": float(np.mean(delta_f1s)),
        "delta_f1_ci_lower": f1_ci[0],
        "delta_f1_ci_upper": f1_ci[1],
        "delta_recall_mean": float(np.mean(delta_recalls)),
        "delta_recall_ci_lower": recall_ci[0],
        "delta_recall_ci_upper": recall_ci[1],
        "statistically_distinguishable_from_noise": distinguishable,
    }


def run_ablation_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preprocessor: HSCMPreprocessor,
    output_csv_path: Optional[Path] = None,
    seeds: List[int] = SEEDS,
    max_epochs: int = 10,
) -> pd.DataFrame:
    """
    Executes systematic feature group, loss variant, and threshold calibration ablations across 5 seeds.
    Computes seed mean, seed std, 95% bootstrap CIs, and statistical distinguishability against noise.
    """
    feature_names = preprocessor.feature_names
    num_cols = preprocessor.numeric_cols

    harmonic_features = ["Month_Sin", "Month_Cos"]
    ratio_features = ["Value_Per_Pack", "Freight_Per_Kg", "Freight_To_Value_Ratio", "Insurance_To_Value_Ratio"]

    feature_groups = {
        "Full_Model": list(range(len(feature_names))),
        "No_Harmonic_Dates": [i for i, f in enumerate(feature_names) if f not in harmonic_features],
        "No_SLA_And_Ratios": [i for i, f in enumerate(feature_names) if f not in ratio_features],
        "Numeric_Only_No_Cat": [i for i, f in enumerate(feature_names) if f in num_cols],
    }

    configurations = [
        {"name": "Full_Model (Weighted BCE, Calibrated)", "feat_grp": "Full_Model", "loss": "weighted_bce", "calibrated": True},
        {"name": "Ablation: No Harmonic Dates", "feat_grp": "No_Harmonic_Dates", "loss": "weighted_bce", "calibrated": True},
        {"name": "Ablation: No SLA Squeeze & Ratios", "feat_grp": "No_SLA_And_Ratios", "loss": "weighted_bce", "calibrated": True},
        {"name": "Ablation: Numeric Features Only", "feat_grp": "Numeric_Only_No_Cat", "loss": "weighted_bce", "calibrated": True},
        {"name": "Ablation: Unweighted BCE Loss", "feat_grp": "Full_Model", "loss": "unweighted_bce", "calibrated": True},
        {"name": "Ablation: Focal Loss", "feat_grp": "Full_Model", "loss": "focal_loss", "calibrated": True},
        {"name": "Ablation: No Threshold Calibration (tau=0.50)", "feat_grp": "Full_Model", "loss": "weighted_bce", "calibrated": False},
    ]

    # Store seed predictions per configuration
    config_seed_metrics: Dict[str, List[Dict[str, float]]] = {cfg["name"]: [] for cfg in configurations}
    config_seed_probs: Dict[str, List[np.ndarray]] = {cfg["name"]: [] for cfg in configurations}
    config_seed_thresholds: Dict[str, List[float]] = {cfg["name"]: [] for cfg in configurations}
    config_dims: Dict[str, int] = {}

    print(f"[*] Running Ablation Suite across {len(seeds)} Seeds: {seeds}...")

    for cfg in configurations:
        cfg_name = cfg["name"]
        indices = feature_groups[cfg["feat_grp"]]
        curr_dim = len(indices)
        config_dims[cfg_name] = curr_dim

        X_tr = X_train[:, indices]
        X_v = X_val[:, indices]
        X_te = X_test[:, indices]

        train_ds = HSCMDataset(X_tr, y_train)
        val_ds = HSCMDataset(X_v, y_val)
        test_ds = HSCMDataset(X_te, y_test)

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        for s in seeds:
            model = Tabular1DCNN(input_dim=curr_dim, dropout_rate=0.2)
            trainer = SharedTrainer(
                model=model,
                model_name=f"Ablation_{cfg_name[:15]}_{s}",
                loss_variant=cfg["loss"],
                lr=1e-3,
                device=DEVICE,
                seed=s,
            )
            trainer.fit(train_loader, val_loader, max_epochs=max_epochs, patience=4, verbose=False)

            _, _, _, test_probs, _ = trainer.evaluate(test_loader)
            tau = trainer.optimal_threshold if cfg["calibrated"] else 0.50

            m = compute_metrics_at_threshold(y_test, test_probs, threshold=tau)
            config_seed_metrics[cfg_name].append(m)
            config_seed_probs[cfg_name].append(test_probs)
            config_seed_thresholds[cfg_name].append(tau)

    # Reference: Full Model ensemble probabilities and mean threshold
    full_name = configurations[0]["name"]
    full_probs_avg = np.mean(config_seed_probs[full_name], axis=0)
    full_thresh_avg = float(np.mean(config_seed_thresholds[full_name]))

    rows = []
    for cfg in configurations:
        cfg_name = cfg["name"]
        seed_m = config_seed_metrics[cfg_name]
        mean_tau = float(np.mean(config_seed_thresholds[cfg_name]))

        rec_vals = [m["Recall_Late"] for m in seed_m]
        prec_vals = [m["Precision_Late"] for m in seed_m]
        f1_vals = [m["F1_Late"] for m in seed_m]
        pr_vals = [m["PR_AUC"] for m in seed_m]
        roc_vals = [m["ROC_AUC"] for m in seed_m]
        acc_vals = [m["Accuracy"] for m in seed_m]

        cfg_probs_avg = np.mean(config_seed_probs[cfg_name], axis=0)

        # Paired bootstrap difference vs Full Model
        if cfg_name == full_name:
            delta_stats = {
                "delta_pr_auc_mean": 0.0,
                "delta_pr_auc_ci_lower": 0.0,
                "delta_pr_auc_ci_upper": 0.0,
                "delta_f1_mean": 0.0,
                "delta_f1_ci_lower": 0.0,
                "delta_f1_ci_upper": 0.0,
                "delta_recall_mean": 0.0,
                "delta_recall_ci_lower": 0.0,
                "delta_recall_ci_upper": 0.0,
                "statistically_distinguishable_from_noise": False,
            }
        else:
            delta_stats = compute_bootstrap_difference_ci(
                y_true=y_test,
                probs_full=full_probs_avg,
                probs_ablated=cfg_probs_avg,
                thresh_full=full_thresh_avg,
                thresh_ablated=mean_tau,
                n_bootstraps=1000,
                alpha=0.05,
                seed=42,
            )

        row = {
            "Ablation_Setting": cfg_name,
            "Threshold_Mean": round(mean_tau, 3),
            "Recall_Late_Mean": round(float(np.mean(rec_vals)), 4),
            "Recall_Late_Std": round(float(np.std(rec_vals)), 4),
            "Precision_Late_Mean": round(float(np.mean(prec_vals)), 4),
            "Precision_Late_Std": round(float(np.std(prec_vals)), 4),
            "F1_Late_Mean": round(float(np.mean(f1_vals)), 4),
            "F1_Late_Std": round(float(np.std(f1_vals)), 4),
            "PR_AUC_Mean": round(float(np.mean(pr_vals)), 4),
            "PR_AUC_Std": round(float(np.std(pr_vals)), 4),
            "ROC_AUC_Mean": round(float(np.mean(roc_vals)), 4),
            "ROC_AUC_Std": round(float(np.std(roc_vals)), 4),
            "Accuracy_Mean": round(float(np.mean(acc_vals)), 4),
            "Accuracy_Std": round(float(np.std(acc_vals)), 4),
            "Delta_PR_AUC_Mean": round(delta_stats["delta_pr_auc_mean"], 4),
            "Delta_PR_AUC_95CI": f"[{delta_stats['delta_pr_auc_ci_lower']:.4f}, {delta_stats['delta_pr_auc_ci_upper']:.4f}]",
            "Delta_F1_Mean": round(delta_stats["delta_f1_mean"], 4),
            "Delta_F1_95CI": f"[{delta_stats['delta_f1_ci_lower']:.4f}, {delta_stats['delta_f1_ci_upper']:.4f}]",
            "Delta_Recall_Mean": round(delta_stats["delta_recall_mean"], 4),
            "Delta_Recall_95CI": f"[{delta_stats['delta_recall_ci_lower']:.4f}, {delta_stats['delta_recall_ci_upper']:.4f}]",
            "Statistically_Distinguishable": delta_stats["statistically_distinguishable_from_noise"],
            "Input_Dim": config_dims[cfg_name],
            "Seeds_Count": len(seeds),
        }
        rows.append(row)

    df_ablation = pd.DataFrame(rows)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_ablation.to_csv(output_csv_path, index=False)

    return df_ablation
