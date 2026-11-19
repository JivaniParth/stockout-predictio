"""
Fast integration tests for Master Pipeline modules (< 30 seconds).
Covers:
- Probability Calibration (Platt, Isotonic, 15-bin ECE)
- Fairness Governance (Tier metrics, mitigation tuning, bootstrap CIs)
- Statistical Tests (DeLong, McNemar, Holm-Bonferroni)
- Monitoring & Drift (PSI, KS test, SQLite logging)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import compute_ece, calibrate_and_benchmark, ModelCalibrator
from src.evaluation.fairness_engine import (
    load_tiers_config,
    compute_per_tier_metrics,
    tune_per_tier_thresholds,
)
from src.evaluation.statistical_tests import (
    compute_delong_p_value,
    compute_mcnemar_test,
    holm_bonferroni_correction,
)
from src.infrastructure.monitoring.drift_detector import (
    compute_psi,
    compute_ks_drift,
    PredictionLogger,
)


@pytest.fixture
def synthetic_predictions():
    np.random.seed(42)
    N = 200
    y_true = np.random.binomial(1, 0.25, N)
    # Generate uncalibrated probabilities with slight overconfidence
    probs = np.clip(y_true * 0.7 + np.random.uniform(0.05, 0.45, N), 0.01, 0.99)
    return y_true, probs


def test_ece_computation(synthetic_predictions):
    y_true, probs = synthetic_predictions
    ece = compute_ece(y_true, probs, n_bins=15)
    assert 0.0 <= ece <= 1.0


def test_platt_and_isotonic_calibration(synthetic_predictions):
    y_true, probs = synthetic_predictions
    cal = calibrate_and_benchmark(
        y_val=y_true[:100],
        val_probs=probs[:100],
        y_test=y_true[100:],
        test_probs=probs[100:],
        n_bins=15,
    )
    assert "raw" in cal
    assert "platt" in cal
    assert "isotonic" in cal
    assert 0.1 <= cal["platt"]["tau_star"] <= 0.9
    assert 0.1 <= cal["isotonic"]["tau_star"] <= 0.9
    assert cal["platt"]["test_ece"] >= 0.0
    assert cal["isotonic"]["test_ece"] >= 0.0


def test_fairness_engine_and_mitigation():
    tiers_path = PROJECT_ROOT / "config" / "tiers.yaml"
    country_to_tier, criterion = load_tiers_config(tiers_path)
    assert "South Africa" in country_to_tier
    assert country_to_tier["South Africa"] == "Tier_1_Low_Risk_Hub"
    assert "Burundi" in country_to_tier
    assert country_to_tier["Burundi"] == "Tier_3_High_Risk_Underserved"

    # Synthetic dataframe
    df = pd.DataFrame({
        "Country": ["South Africa"] * 50 + ["Kenya"] * 50 + ["Burundi"] * 50,
        "Late_Delivery_Risk": np.random.binomial(1, 0.2, 150),
    })
    y_true = df["Late_Delivery_Risk"].values
    probs = np.random.uniform(0.1, 0.9, 150)

    # Global threshold
    res = compute_per_tier_metrics(df, y_true, probs, threshold=0.5, country_to_tier=country_to_tier)
    assert "max_min_recall_gap" in res
    assert "equalized_odds_diff" in res
    assert len(res["tiers"]) == 3

    # Per-tier tuning
    tier_th = tune_per_tier_thresholds(df, y_true, probs, country_to_tier=country_to_tier)
    assert "Tier_1_Low_Risk_Hub" in tier_th
    assert "Tier_3_High_Risk_Underserved" in tier_th

    # Mitigated metrics
    mit_res = compute_per_tier_metrics(df, y_true, probs, threshold=tier_th, country_to_tier=country_to_tier)
    assert mit_res["overall_recall"] >= 0.0


def test_delong_and_mcnemar_statistical_tests():
    np.random.seed(42)
    N = 100
    y_true = np.random.binomial(1, 0.3, N)
    p1 = np.clip(y_true * 0.8 + np.random.uniform(0, 0.3, N), 0, 1)
    p2 = np.clip(y_true * 0.7 + np.random.uniform(0, 0.4, N), 0, 1)

    p_delong = compute_delong_p_value(y_true, p1, p2)
    assert 0.0 <= p_delong <= 1.0

    preds1 = (p1 >= 0.5).astype(int)
    preds2 = (p2 >= 0.5).astype(int)
    p_mcnemar = compute_mcnemar_test(y_true, preds1, preds2)
    assert 0.0 <= p_mcnemar <= 1.0

    # Test Holm correction
    raw_p = [0.01, 0.04, 0.03, 0.20]
    adj_p = holm_bonferroni_correction(raw_p)
    assert len(adj_p) == 4
    assert adj_p[0] <= adj_p[2] <= adj_p[1] <= adj_p[3]


def test_drift_detection_and_prediction_logging(tmp_path):
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 500)
    # Similar distribution: low PSI
    current_similar = np.random.normal(0.05, 1, 500)
    psi_low = compute_psi(baseline, current_similar)
    assert psi_low < 0.2

    # Shifted distribution: high PSI
    current_shifted = np.random.normal(2.0, 1.5, 500)
    psi_high = compute_psi(baseline, current_shifted)
    assert psi_high > 0.2

    # KS test
    ks_stat, p_val, drift = compute_ks_drift(baseline, current_shifted)
    assert drift is True
    assert p_val < 0.05

    # Prediction logger test
    db_path = tmp_path / "test_drift.db"
    jsonl_path = tmp_path / "test_drift.jsonl"
    logger = PredictionLogger(db_path=db_path, jsonl_path=jsonl_path)
    logger.log(np.array([1.0, 2.0, 3.0]), score=0.45, model_version="v1.0-test")

    assert db_path.exists()
    assert jsonl_path.exists()


def test_fairness_provenance_audit():
    from src.evaluation.fairness_engine import audit_tier_data_provenance

    prov = audit_tier_data_provenance()
    assert prov["data_split_independence"] is True
    assert prov["derived_from_val_or_test"] is False
    assert "CERTIFIED" in prov["audit_certification"]
    assert "Tier_1_Low_Risk_Hub" in prov["tier_definitions"]
    assert "Tier_3_High_Risk_Underserved" in prov["tier_definitions"]


def test_quarterly_drift_simulation(tmp_path):
    from src.infrastructure.monitoring.drift_detector import simulate_quarterly_drift

    df = pd.DataFrame({
        "Scheduled Delivery Date": ["2015-02-10", "2015-05-15", "2015-08-20", "2015-11-25"] * 25,
        "Late_Delivery_Risk": np.random.binomial(1, 0.2, 100),
    })
    test_probs = np.random.uniform(0.1, 0.8, 100)
    baseline_scores = np.random.uniform(0.1, 0.8, 200)

    out_csv = tmp_path / "quarterly_drift.csv"
    res_df = simulate_quarterly_drift(
        test_df=df,
        test_features=np.zeros((100, 5)),
        test_probs=test_probs,
        baseline_scores=baseline_scores,
        threshold=0.25,
        output_path=out_csv,
    )
    assert out_csv.exists()
    assert len(res_df) > 0
    assert "Quarter" in res_df.columns
    assert "Score_PSI" in res_df.columns
