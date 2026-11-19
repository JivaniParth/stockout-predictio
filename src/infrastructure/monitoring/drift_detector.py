"""
Model Monitoring and Drift Detection Module.
Features:
- Structured prediction logging (feature hash, score, timestamp, model version) to SQLite and JSONL
- Population Stability Index (PSI) computation for continuous feature and prediction score distributions
- Kolmogorov-Smirnov (KS) two-sample test for non-parametric distribution drift
- Quarterly simulation over 2015+ prospective windows assessing drift and metric decay
- Saves drift audit reports to reports/drift/
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import datetime
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score, f1_score, precision_recall_curve, auc


class PredictionLogger:
    """Logs prediction inference payloads and scores for continuous auditability."""

    def __init__(self, db_path: Path = Path("reports/drift/prediction_monitoring.db"), jsonl_path: Optional[Path] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path or self.db_path.parent / "predictions.jsonl")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_hash TEXT NOT NULL,
                    risk_probability REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    country TEXT,
                    tier TEXT
                )
                """
            )
            conn.commit()

    def log(
        self,
        features: np.ndarray,
        score: float,
        model_version: str,
        country: str = "Unknown",
        tier: str = "Tier_2_Moderate_Risk",
    ):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        feat_bytes = np.asarray(features, dtype=np.float32).tobytes()
        feat_hash = hashlib.sha256(feat_bytes).hexdigest()

        # Insert to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO prediction_audit_log (feature_hash, risk_probability, timestamp, model_version, country, tier)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (feat_hash, float(score), ts, model_version, country, tier),
                )
                conn.commit()
        except Exception as e:
            print(f"[!] Warning: Prediction logging to SQLite failed: {e}")

        # Append to JSONL
        try:
            record = {
                "feature_hash": feat_hash,
                "risk_probability": float(score),
                "timestamp": ts,
                "model_version": model_version,
                "country": country,
                "tier": tier,
            }
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass


def compute_psi(
    baseline: np.ndarray, current: np.ndarray, num_bins: int = 10, eps: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI).
    PSI = sum ( (Current% - Baseline%) * ln(Current% / Baseline%) )
    """
    b = np.asarray(baseline).ravel()
    c = np.asarray(current).ravel()

    if len(b) == 0 or len(c) == 0:
        return 0.0

    # Quantile bin edges from baseline
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.percentile(b, quantiles * 100)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    b_counts, _ = np.histogram(b, bins=bin_edges)
    c_counts, _ = np.histogram(c, bins=bin_edges)

    b_pct = (b_counts + eps) / (np.sum(b_counts) + eps * num_bins)
    c_pct = (c_counts + eps) / (np.sum(c_counts) + eps * num_bins)

    psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(psi)


def compute_ks_drift(
    baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Two-sample Kolmogorov-Smirnov test for distribution divergence.
    """
    b = np.asarray(baseline).ravel()
    c = np.asarray(current).ravel()
    if len(b) == 0 or len(c) == 0:
        return 0.0, 1.0, False

    ks_stat, p_val = stats.ks_2samp(b, c)
    drift_detected = bool(p_val < alpha)
    return float(ks_stat), float(p_val), drift_detected


def evaluate_drift(
    baseline_features: np.ndarray,
    current_features: np.ndarray,
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    feature_names: List[str],
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.05,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Audits feature and prediction score drift between historical and live serving windows.
    """
    # 1. Prediction Score Drift
    score_psi = compute_psi(baseline_scores, current_scores)
    score_ks, score_p, score_drift = compute_ks_drift(baseline_scores, current_scores, alpha=ks_alpha)

    # 2. Per-feature Drift
    feature_drifts = {}
    total_drifted_features = 0

    for idx, name in enumerate(feature_names):
        b_col = baseline_features[:, idx]
        c_col = current_features[:, idx]

        col_psi = compute_psi(b_col, c_col)
        col_ks, col_p, col_drift = compute_ks_drift(b_col, c_col, alpha=ks_alpha)

        if col_drift or col_psi >= psi_threshold:
            total_drifted_features += 1

        feature_drifts[name] = {
            "psi": round(col_psi, 4),
            "ks_statistic": round(col_ks, 4),
            "p_value": round(col_p, 4),
            "drift_detected": col_drift or (col_psi >= psi_threshold),
        }

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "score_drift": {
            "psi": round(score_psi, 4),
            "ks_statistic": round(score_ks, 4),
            "p_value": round(score_p, 4),
            "drift_detected": score_drift or (score_psi >= psi_threshold),
            "status": "ALERT_RETRAIN_REQUIRED" if (score_psi >= psi_threshold) else "STABLE",
        },
        "feature_summary": {
            "total_features": len(feature_names),
            "drifted_features_count": total_drifted_features,
            "drift_percentage": round(100.0 * total_drifted_features / max(len(feature_names), 1), 2),
        },
        "per_feature_drift": feature_drifts,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def simulate_quarterly_drift(
    test_df: pd.DataFrame,
    test_features: np.ndarray,
    test_probs: np.ndarray,
    baseline_scores: np.ndarray,
    threshold: float = 0.21,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Simulates consecutive quarterly performance decay and distribution drift over prospective 2015+ data.
    """
    df = test_df.copy()
    df["_prob"] = test_probs
    df["_pred"] = (test_probs >= threshold).astype(int)

    sched_dates = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce")
    df["Quarter"] = sched_dates.dt.to_period("Q").astype(str)

    quarters = sorted(df["Quarter"].dropna().unique())
    records = []

    for q in quarters:
        sub = df[df["Quarter"] == q]
        if len(sub) == 0:
            continue

        q_probs = sub["_prob"].values
        q_targets = sub["Late_Delivery_Risk"].values
        q_preds = sub["_pred"].values

        psi = compute_psi(baseline_scores, q_probs)
        ks_stat, p_val, drift = compute_ks_drift(baseline_scores, q_probs)

        rec = float(recall_score(q_targets, q_preds, zero_division=0))
        f1 = float(f1_score(q_targets, q_preds, zero_division=0))

        if len(np.unique(q_targets)) > 1:
            p_c, r_c, _ = precision_recall_curve(q_targets, q_probs)
            pr = float(auc(r_c, p_c))
        else:
            pr = 0.0

        records.append({
            "Quarter": q,
            "Sample_Count": len(sub),
            "Delay_Count": int(q_targets.sum()),
            "Delay_Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "PR_AUC": round(pr, 4),
            "Score_PSI": round(psi, 4),
            "KS_Statistic": round(ks_stat, 4),
            "KS_p_value": round(p_val, 4),
            "Drift_Detected": drift,
        })

    res_df = pd.DataFrame(records)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(output_path, index=False)

    return res_df
