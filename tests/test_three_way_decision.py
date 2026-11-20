"""
Unit and Integration Tests for Three-Way Decision (3WD) Policy
and Ghazvinian Multi-Objective ANN Regression.
"""

import pytest
import numpy as np
import torch
from src.domain.three_way_decision_policy import (
    ThreeWayDecisionPolicy,
    DecisionRegion,
    LogisticalAction,
    ConsignmentClassification,
)
from src.models.multi_objective_ann import (
    GhazvinianMultiObjectiveANN,
    evaluate_multi_objective_ann,
)


def test_three_way_decision_threshold_validation():
    """Verify that tau_safe must be strictly less than tau_alert."""
    with pytest.raises(ValueError):
        ThreeWayDecisionPolicy(tau_safe=0.5, tau_alert=0.3)
    with pytest.raises(ValueError):
        ThreeWayDecisionPolicy(tau_safe=0.4, tau_alert=0.4)


def test_three_way_decision_classification_regions():
    """Verify correct partition into Positive, Negative, and Boundary regions."""
    policy = ThreeWayDecisionPolicy(tau_safe=0.10, tau_alert=0.30)

    c1 = policy.classify_single(0.05)
    assert c1.decision_region == DecisionRegion.NEGATIVE_SAFE
    assert c1.is_path2_safe is True
    assert c1.is_path1_alert is False
    assert c1.recommended_action == LogisticalAction.STANDARD_SURFACE_TRANSIT

    c2 = policy.classify_single(0.40)
    assert c2.decision_region == DecisionRegion.POSITIVE_ALERT
    assert c2.is_path1_alert is True
    assert c2.is_path2_safe is False
    assert c2.recommended_action == LogisticalAction.IMMEDIATE_STOCKOUT_INTERVENTION

    c3 = policy.classify_single(0.20)
    assert c3.decision_region == DecisionRegion.BOUNDARY_MONITOR
    assert c3.is_path1_alert is False
    assert c3.is_path2_safe is False
    assert c3.recommended_action == LogisticalAction.ACTIVE_TELEMETRY_TRACKING


def test_three_way_decision_batch_classification():
    """Verify vectorized classification output matches expected region enums."""
    policy = ThreeWayDecisionPolicy(tau_safe=0.10, tau_alert=0.30)
    probs = np.array([0.02, 0.15, 0.35, 0.08, 0.29, 0.85])
    regions = policy.classify_batch(probs)
    assert list(regions) == [
        DecisionRegion.NEGATIVE_SAFE.value,
        DecisionRegion.BOUNDARY_MONITOR.value,
        DecisionRegion.POSITIVE_ALERT.value,
        DecisionRegion.NEGATIVE_SAFE.value,
        DecisionRegion.BOUNDARY_MONITOR.value,
        DecisionRegion.POSITIVE_ALERT.value,
    ]


def test_three_way_decision_calibration_from_validation():
    """Verify calibration algorithm selects valid thresholds from empirical distribution."""
    np.random.seed(42)
    # Synthetic validation set with 20% positives
    y_val = np.random.binomial(1, 0.20, size=500)
    # Give higher probabilities to true positives
    y_prob = np.where(y_val == 1, np.random.beta(5, 2, size=500), np.random.beta(2, 5, size=500))

    policy = ThreeWayDecisionPolicy.calibrate_from_validation(
        y_val_true=y_val,
        y_val_prob=y_prob,
        target_path1_recall=0.85,
        target_path2_npv=0.85,
        target_path2_spec=0.70,
    )
    assert policy.tau_safe < policy.tau_alert
    assert policy.tau_safe > 0.0
    assert policy.tau_alert <= 1.0

    metrics = policy.evaluate(y_val, y_prob)
    assert metrics.path1_delay_recall >= 0.85
    assert metrics.path2_safe_npv >= 0.80


def test_ghazvinian_multi_objective_ann_forward():
    """Verify multi-objective neural network shapes and backpropagation."""
    input_dim = 20
    batch_size = 16
    model = GhazvinianMultiObjectiveANN(input_dim=input_dim, hidden_dims=(32, 16, 8))

    x = torch.randn(batch_size, input_dim)
    pred_freight, pred_time = model(x)

    assert pred_freight.shape == (batch_size,)
    assert pred_time.shape == (batch_size,)

    loss = pred_freight.sum() + pred_time.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
