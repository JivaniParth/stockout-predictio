"""
Statistical Rigor and Significance Testing Module.
Features:
- Multi-seed aggregation (mean +- std over 5 seeds)
- 1000-sample bootstrap 95% Confidence Intervals on prospective test set
- DeLong test for ROC-AUC paired comparison
- McNemar paired test for classification divergence with Holm-Bonferroni correction
- Formats outputs to CSV (reports/model_comparison.csv) and LaTeX table markup
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% empirical bootstrap confidence intervals across test metrics.
    """
    np.random.seed(seed)
    N = len(y_true)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    metrics_boot: Dict[str, List[float]] = {
        "accuracy": [],
        "delay_recall": [],
        "precision": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(N, size=N, replace=True)
        y_b = y_true[boot_idx]
        p_b = y_prob[boot_idx]

        # Skip degenerated resamples
        if len(np.unique(y_b)) < 2:
            continue

        preds_b = (p_b >= threshold).astype(int)

        metrics_boot["accuracy"].append(float(accuracy_score(y_b, preds_b)))
        metrics_boot["delay_recall"].append(float(recall_score(y_b, preds_b, zero_division=0)))
        metrics_boot["precision"].append(float(precision_score(y_b, preds_b, zero_division=0)))
        metrics_boot["f1"].append(float(f1_score(y_b, preds_b, zero_division=0)))

        roc = roc_auc_score(y_b, p_b)
        metrics_boot["roc_auc"].append(float(roc))

        prec, rec, _ = precision_recall_curve(y_b, p_b)
        metrics_boot["pr_auc"].append(float(auc(rec, prec)))

    ci_results: Dict[str, Dict[str, float]] = {}
    lower_pct = 100 * (alpha / 2.0)
    upper_pct = 100 * (1.0 - alpha / 2.0)

    for m_name, vals in metrics_boot.items():
        if len(vals) == 0:
            ci_results[m_name] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        else:
            ci_results[m_name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lower": float(np.percentile(vals, lower_pct)),
                "ci_upper": float(np.percentile(vals, upper_pct)),
            }

    return ci_results


def compute_delong_p_value(
    y_true: np.ndarray, y_prob_champion: np.ndarray, y_prob_model: np.ndarray
) -> float:
    """
    Computes asymptotic p-value for DeLong test comparing paired ROC-AUCs.
    """
    y_true = np.asarray(y_true).ravel()
    p1 = np.asarray(y_prob_champion).ravel()
    p2 = np.asarray(y_prob_model).ravel()

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)

    if m == 0 or n == 0:
        return 1.0

    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)

    # Fast DeLong placement values
    p1_pos = p1[pos_mask]
    p1_neg = p1[neg_mask]
    p2_pos = p2[pos_mask]
    p2_neg = p2[neg_mask]

    # V10: placements for positive cases
    v10_1 = np.mean(p1_pos[:, None] > p1_neg[None, :], axis=1) + 0.5 * np.mean(p1_pos[:, None] == p1_neg[None, :], axis=1)
    v10_2 = np.mean(p2_pos[:, None] > p2_neg[None, :], axis=1) + 0.5 * np.mean(p2_pos[:, None] == p2_neg[None, :], axis=1)

    # V01: placements for negative cases
    v01_1 = np.mean(p1_pos[:, None] > p1_neg[None, :], axis=0) + 0.5 * np.mean(p1_pos[:, None] == p1_neg[None, :], axis=0)
    v01_2 = np.mean(p2_pos[:, None] > p2_neg[None, :], axis=0) + 0.5 * np.mean(p2_pos[:, None] == p2_neg[None, :], axis=0)

    # Sample covariance matrices
    s10 = np.cov(v10_1, v10_2)
    s01 = np.cov(v01_1, v01_2)

    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    if var <= 1e-12:
        return 1.0

    z = (auc1 - auc2) / np.sqrt(var)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(p_val)


def compute_mcnemar_test(
    y_true: np.ndarray, y_pred_champion: np.ndarray, y_pred_model: np.ndarray
) -> float:
    """
    Computes McNemar test with Edwards continuity correction between two paired binary classifiers.
    """
    y_true = np.asarray(y_true).ravel()
    p_champ = np.asarray(y_pred_champion).ravel()
    p_model = np.asarray(y_pred_model).ravel()

    correct_champ = p_champ == y_true
    correct_model = p_model == y_true

    b = np.sum(correct_champ & (~correct_model)) # Champion right, Model wrong
    c = np.sum((~correct_champ) & correct_model) # Champion wrong, Model right

    if (b + c) == 0:
        return 1.0

    # Edwards continuity correction
    chi2_stat = (max(0.0, abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
    return float(p_val)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed_p):
        step_val = (m - rank) * p
        cum_max = max(cum_max, step_val)
        adjusted[orig_idx] = min(1.0, cum_max)

    return adjusted


def generate_latex_table(df_comparison: pd.DataFrame, caption: str, label: str) -> str:
    """Generates publication-ready LaTeX table markup with booktabs."""
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Model Architecture} & \\textbf{Recall (Late)} & \\textbf{Precision} & \\textbf{F1-Score} & \\textbf{PR-AUC} & \\textbf{ROC-AUC} & \\textbf{DeLong $p_{\\text{adj}}$} \\\\",
        "\\midrule",
    ]

    for model, row in df_comparison.iterrows():
        rec = f"{row.get('Recall_Mean', 0.0)*100:.2f}\\%"
        prec = f"{row.get('Precision_Mean', 0.0)*100:.2f}\\%"
        f1 = f"{row.get('F1_Mean', 0.0):.4f}"
        pr = f"{row.get('PR_AUC_Mean', 0.0):.4f}"
        roc = f"{row.get('ROC_AUC_Mean', 0.0):.4f}"
        delong_p = row.get("DeLong_p_adj", 1.0)
        p_str = "Ref (Champ)" if str(delong_p) == "Ref" else f"{delong_p:.4f}"

        latex_lines.append(f"{model} & {rec} & {prec} & {f1} & {pr} & {roc} & {p_str} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(latex_lines)
