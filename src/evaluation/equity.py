"""
Health Equity and Fairness Evaluation across Geographic and Resource Tiers.
Directly addresses RQ4 and mitigates algorithmic bias against resource-constrained healthcare facilities.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score


def evaluate_health_equity(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes stratified performance metrics and fairness disparities across geographic resource tiers.
    """
    df = test_df.copy()
    df["Ground_Truth"] = y_true
    df["Predicted_Prob"] = y_prob
    df["Predicted_Class"] = (y_prob >= threshold).astype(int)

    tiers = ["Tier_1_Low_Risk_Hub", "Tier_2_Moderate_Risk", "Tier_3_High_Risk_Underserved"]
    tier_metrics = {}

    for tier in tiers:
        sub = df[df["Country_Tier"] == tier]
        n_samples = len(sub)
        if n_samples == 0:
            continue

        y_t = sub["Ground_Truth"].values
        y_p = sub["Predicted_Class"].values
        y_prob_sub = sub["Predicted_Prob"].values

        rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
        prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
        roc = float(roc_auc_score(y_t, y_prob_sub)) if len(np.unique(y_t)) > 1 else 0.5
        selection_rate = float(y_p.mean())
        base_rate = float(y_t.mean())

        tier_metrics[tier] = {
            "Samples": n_samples,
            "Base_Late_Rate": round(base_rate, 4),
            "Selection_Rate": round(selection_rate, 4),
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "F1": round(f1, 4),
            "ROC_AUC": round(roc, 4),
        }

    # Disparity calculations between Tier 3 (Underserved) and Tier 1 (Centralized Hub)
    rec_tier3 = tier_metrics.get("Tier_3_High_Risk_Underserved", {}).get("Recall", 0.0)
    rec_tier1 = tier_metrics.get("Tier_1_Low_Risk_Hub", {}).get("Recall", 0.0)
    sel_tier3 = tier_metrics.get("Tier_3_High_Risk_Underserved", {}).get("Selection_Rate", 0.0)
    sel_tier1 = tier_metrics.get("Tier_1_Low_Risk_Hub", {}).get("Selection_Rate", 0.0)

    equal_opportunity_diff = abs(rec_tier3 - rec_tier1)
    stat_parity_diff = sel_tier3 - sel_tier1
    disparate_impact = (sel_tier3 / sel_tier1) if sel_tier1 > 0 else 1.0

    return {
        "Model": model_name,
        "Tier_Metrics": tier_metrics,
        "Equal_Opportunity_Difference": round(equal_opportunity_diff, 4),
        "Statistical_Parity_Difference": round(stat_parity_diff, 4),
        "Disparate_Impact_Ratio": round(disparate_impact, 4),
    }
