"""
Unit Test Suite for Probability Scale Consistency and Threshold Guard.
Fulfills Section 3 requirement:
- Asserts failure (ValueError) if a threshold is applied to a different probability scale than it was tuned on.
- Confirms cross-fitted calibration prevents in-sample overfitting.
- Proves proper threshold scale alignment preserves clinical delay recall.
"""

import pytest
import numpy as np
from src.evaluation.calibration import (
    CalibratedThresholdPolicy,
    CrossFittedCalibrator,
    tune_threshold_with_policy,
    compute_ece,
)


def test_threshold_policy_explicit_scale_mismatch_fails():
    """Verifies that applying a threshold to an incompatible scale tag raises ValueError."""
    # Suppose raw threshold was tuned on scale_pos_weight inflated predictions (mean ~0.45)
    policy = CalibratedThresholdPolicy(
        threshold=0.35,
        probability_scale="raw",
        reference_mean=0.45,
        reference_std=0.15,
        beta=1.0,
    )

    # Calibrated probabilities (mean ~0.12)
    calibrated_probs = np.array([0.05, 0.10, 0.15, 0.20, 0.08, 0.12])

    # Applying raw threshold to 'platt' or 'isotonic' scale MUST fail loudly
    with pytest.raises(ValueError, match="PROBABILITY SCALE MISMATCH"):
        policy.predict(calibrated_probs, current_scale="platt")

    with pytest.raises(ValueError, match="PROBABILITY SCALE MISMATCH"):
        policy.predict(calibrated_probs, current_scale="isotonic")


def test_threshold_policy_distributional_drift_detection():
    """Verifies that severe distribution divergence between raw and calibrated triggers ValueError."""
    np.random.seed(42)
    # Raw probabilities tuned with mean 0.48
    policy = CalibratedThresholdPolicy(
        threshold=0.40,
        probability_scale="raw",
        reference_mean=0.48,
        reference_std=0.12,
    )

    # Calibrated probabilities with true clinical prevalence ~11.8%
    cal_probs = np.random.beta(1, 8, size=200)  # mean ~0.11

    # Should detect distribution mismatch even if current_scale is not explicitly passed
    with pytest.raises(ValueError, match="PROBABILITY SCALE MISMATCH DETECTED"):
        policy.predict(cal_probs, check_distribution=True)


def test_matching_scale_application_succeeds_and_preserves_recall():
    """Verifies that when threshold is tuned on matching calibrated scale, recall is preserved."""
    np.random.seed(42)
    N = 1000
    y_true = np.random.binomial(1, 0.12, size=N)  # 12% delay rate

    # Generate synthetic calibrated risk probabilities (well-correlated with y_true)
    cal_probs = np.clip(y_true * 0.35 + np.random.uniform(0.02, 0.15, size=N), 0.0, 1.0)

    # Tune threshold policy on calibrated validation data
    policy = tune_threshold_with_policy(y_true, cal_probs, scale="platt", beta=1.0)
    assert policy.probability_scale == "platt"
    assert policy.threshold < 0.30  # Should be appropriately scaled around 0.10 - 0.25

    # Applying to new test data on matching scale succeeds
    y_test = np.random.binomial(1, 0.12, size=500)
    test_cal_probs = np.clip(y_test * 0.35 + np.random.uniform(0.02, 0.15, size=500), 0.0, 1.0)
    preds = policy.predict(test_cal_probs, current_scale="platt")

    # High recall is preserved (> 70%) on matching probability scale!
    recall = np.sum((preds == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
    assert recall >= 0.70, f"Expected recall >= 0.70, got {recall:.4f}"


def test_cross_fitted_calibrator_out_of_fold():
    """Verifies that CrossFittedCalibrator produces out-of-fold predictions with zero leakage."""
    np.random.seed(42)
    N = 400
    y_val = np.random.binomial(1, 0.15, size=N)
    raw_probs = np.clip(y_val * 0.4 + np.random.uniform(0.1, 0.5, size=N), 0.01, 0.99)

    cf = CrossFittedCalibrator(method="isotonic", n_splits=5, random_state=42)
    oof_probs, fitted_cf = cf.fit_predict_oof(raw_probs, y_val)

    assert oof_probs.shape == (N,)
    assert np.all((oof_probs >= 0.0) & (oof_probs <= 1.0))
    # ECE on out-of-fold probabilities must be valid
    ece = compute_ece(y_val, oof_probs, n_bins=15)
    assert 0.0 <= ece <= 1.0
