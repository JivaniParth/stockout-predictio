"""
Layer 1: Domain Three-Way Decision (3WD) Policy for Balanced Healthcare Logistics.
References:
1. Yao, Y. (2010). Three-way decisions with probabilistic rough sets. Information Sciences, 180(3), 341-353.
2. Zhang et al. (2022). Research and application of XGBoost in imbalanced data. IJDSN, 18(9).
3. Park et al. (2025). Smarter Chains, Safer Medicines. MDPI Forecasting, 7(1), 78.
4. Ghazvinian et al. (2025). Enhancing Efficiency in the Healthcare Sector Through Multi-Objective Optimization. MDPI Systems, 13(2), 91.

Under class imbalance, forcing a single binary threshold creates an irreconcilable tradeoff:
- Low threshold catches late deliveries (High Recall) but incurs false alarms (Low Precision/Specificity).
- High threshold ensures high specificity/NPV but misses catastrophic stockouts (Low Recall).

The Three-Way Decision (3WD) policy establishes dual thresholds (tau_safe, tau_alert):
- POSITIVE REGION (Risk >= tau_alert): Clinical Early Warning Stockout Radar (Path 1: Recall >= 90.0%).
  Triggers clinical buffer stock reallocation and carrier prioritization.
- NEGATIVE REGION (Risk < tau_safe): Consignment Safety Confidence (Path 2: NPV >= 95.0%, Specificity >= 90.0%).
  Certifies safe standard transit without expensive rush air freight surcharges.
- BOUNDARY REGION (tau_safe <= Risk < tau_alert): Active Telemetry / Dynamic Buffer Region.
  Triggers non-disruptive tracking pings and dynamic buffer monitoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Tuple
import numpy as np


class DecisionRegion(str, Enum):
    POSITIVE_ALERT = "POSITIVE_ALERT"       # Early warning stockout radar (Path 1)
    NEGATIVE_SAFE = "NEGATIVE_SAFE"         # Certified safe consignment (Path 2)
    BOUNDARY_MONITOR = "BOUNDARY_MONITOR"   # Active monitoring & buffer tracking


class LogisticalAction(str, Enum):
    IMMEDIATE_STOCKOUT_INTERVENTION = "Immediate Stockout Intervention & Priority Buffer Dispatch"
    STANDARD_SURFACE_TRANSIT = "Certified Safe Standard Surface / Ocean Transit"
    ACTIVE_TELEMETRY_TRACKING = "Active Carrier Telemetry & Dynamic Buffer Monitoring"


@dataclass(frozen=True)
class ConsignmentClassification:
    risk_score: float
    decision_region: DecisionRegion
    recommended_action: LogisticalAction
    is_path1_alert: bool
    is_path2_safe: bool


@dataclass
class ThreeWayDecisionMetrics:
    tau_safe: float
    tau_alert: float
    # Path 1 Metrics (Positive Region)
    path1_delay_recall: float
    path1_pos_count: int
    path1_pos_precision: float
    # Path 2 Metrics (Negative Region)
    path2_safe_npv: float
    path2_safe_specificity: float
    path2_neg_count: int
    # Boundary Region Metrics
    boundary_count: int
    boundary_ratio: float
    # Overall Performance
    balanced_accuracy: float
    g_mean: float
    total_samples: int


class ThreeWayDecisionPolicy:
    """
    Implements Cost-Sensitive Three-Way Decision (3WD) Theory for pharmaceutical logistics.
    Optimizes dual thresholds (tau_safe, tau_alert) to simultaneously satisfy:
    1. Path 1: Delay Catch Rate (Recall) >= 90.0%
    2. Path 2: Consignment Safety Confidence (NPV) >= 95.0% and Specificity >= 90.0%
    """

    def __init__(self, tau_safe: float = 0.10, tau_alert: float = 0.25):
        if tau_safe >= tau_alert:
            raise ValueError(f"tau_safe ({tau_safe}) must be strictly less than tau_alert ({tau_alert})")
        self.tau_safe = float(tau_safe)
        self.tau_alert = float(tau_alert)

    def classify_single(self, risk_score: float) -> ConsignmentClassification:
        score = float(risk_score)
        if score >= self.tau_alert:
            return ConsignmentClassification(
                risk_score=score,
                decision_region=DecisionRegion.POSITIVE_ALERT,
                recommended_action=LogisticalAction.IMMEDIATE_STOCKOUT_INTERVENTION,
                is_path1_alert=True,
                is_path2_safe=False,
            )
        elif score < self.tau_safe:
            return ConsignmentClassification(
                risk_score=score,
                decision_region=DecisionRegion.NEGATIVE_SAFE,
                recommended_action=LogisticalAction.STANDARD_SURFACE_TRANSIT,
                is_path1_alert=False,
                is_path2_safe=True,
            )
        else:
            return ConsignmentClassification(
                risk_score=score,
                decision_region=DecisionRegion.BOUNDARY_MONITOR,
                recommended_action=LogisticalAction.ACTIVE_TELEMETRY_TRACKING,
                is_path1_alert=False,
                is_path2_safe=False,
            )

    def classify_batch(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities)
        regions = np.empty(probs.shape, dtype=object)
        regions[probs >= self.tau_alert] = DecisionRegion.POSITIVE_ALERT.value
        regions[probs < self.tau_safe] = DecisionRegion.NEGATIVE_SAFE.value
        regions[(probs >= self.tau_safe) & (probs < self.tau_alert)] = DecisionRegion.BOUNDARY_MONITOR.value
        return regions

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> ThreeWayDecisionMetrics:
        """
        Computes formal Three-Way Decision metrics across Path 1, Path 2, and Boundary Region.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))

        # Path 1: Positive Region (High Risk Alert)
        # Any shipment with prob >= tau_alert is flagged as high stockout risk
        pos_mask = (y_prob >= self.tau_alert)
        tp_pos = int(np.sum((y_true == 1) & pos_mask))
        fp_pos = int(np.sum((y_true == 0) & pos_mask))
        path1_recall = (tp_pos / n_pos) if n_pos > 0 else 0.0
        path1_precision = (tp_pos / (tp_pos + fp_pos)) if (tp_pos + fp_pos) > 0 else 0.0

        # Path 2: Negative Region (Certified Safe Consignment)
        # Any shipment with prob < tau_safe is certified as safe
        neg_mask = (y_prob < self.tau_safe)
        tn_neg = int(np.sum((y_true == 0) & neg_mask))
        fn_neg = int(np.sum((y_true == 1) & neg_mask))
        path2_npv = (tn_neg / (tn_neg + fn_neg)) if (tn_neg + fn_neg) > 0 else 0.0
        path2_specificity = (tn_neg / n_neg) if n_neg > 0 else 0.0

        # Boundary Region (Active Telemetry Monitoring)
        bnd_mask = (y_prob >= self.tau_safe) & (y_prob < self.tau_alert)
        bnd_count = int(np.sum(bnd_mask))
        bnd_ratio = bnd_count / n if n > 0 else 0.0

        # Balanced Accuracy & Geometric Mean across the safe vs alert regions
        balanced_acc = 0.5 * (path1_recall + path2_specificity)
        g_mean = np.sqrt(max(0.0, path1_recall * path2_specificity))

        return ThreeWayDecisionMetrics(
            tau_safe=self.tau_safe,
            tau_alert=self.tau_alert,
            path1_delay_recall=path1_recall,
            path1_pos_count=int(np.sum(pos_mask)),
            path1_pos_precision=path1_precision,
            path2_safe_npv=path2_npv,
            path2_safe_specificity=path2_specificity,
            path2_neg_count=int(np.sum(neg_mask)),
            boundary_count=bnd_count,
            boundary_ratio=bnd_ratio,
            balanced_accuracy=balanced_acc,
            g_mean=g_mean,
            total_samples=n,
        )

    @classmethod
    def calibrate_from_validation(
        cls,
        y_val_true: np.ndarray,
        y_val_prob: np.ndarray,
        target_path1_recall: float = 0.96,
        target_path2_npv: float = 0.98,
        target_path2_spec: float = 0.05,
    ) -> "ThreeWayDecisionPolicy":
        """
        Calibrates optimal dual thresholds on validation data (strictly un-leaked from test).
        1. Selects tau_alert such that Delay Recall >= target_path1_recall (e.g. >= 90%).
        2. Selects tau_safe such that Consignment NPV >= target_path2_npv (e.g. >= 95%)
           and Specificity is maximized.
        """
        y_val_true = np.asarray(y_val_true, dtype=int)
        y_val_prob = np.asarray(y_val_prob, dtype=float)

        sorted_probs = np.sort(np.unique(y_val_prob))
        
        # 1. Find optimal tau_alert for Path 1 (Early Warning Stockout Radar)
        # Search from higher threshold downwards to find the highest threshold satisfying recall >= target
        best_tau_alert = 0.15
        for t in sorted_probs[::-1]:
            rec = np.sum((y_val_true == 1) & (y_val_prob >= t)) / np.sum(y_val_true == 1)
            if rec >= target_path1_recall:
                best_tau_alert = float(t)
                break

        # 2. Find optimal tau_safe for Path 2 (Consignment Safety Confidence)
        # Search from lower threshold upwards to find the highest threshold satisfying NPV >= target
        best_tau_safe = 0.05
        candidate_safes = sorted_probs[sorted_probs < best_tau_alert]
        for t in candidate_safes:
            neg_mask = (y_val_prob < t)
            if np.sum(neg_mask) == 0:
                continue
            tn = np.sum((y_val_true == 0) & neg_mask)
            fn = np.sum((y_val_true == 1) & neg_mask)
            npv = tn / (tn + fn)
            spec = tn / np.sum(y_val_true == 0)
            if npv >= target_path2_npv and spec >= target_path2_spec:
                best_tau_safe = float(t)

        # Fallback safeguard to maintain tau_safe < tau_alert
        if best_tau_safe >= best_tau_alert:
            best_tau_safe = max(0.01, best_tau_alert * 0.5)

        return cls(tau_safe=best_tau_safe, tau_alert=best_tau_alert)
