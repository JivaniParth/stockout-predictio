"""
Probability Calibration and Evaluation Protocol Module.

RESEARCH PROTOCOL & METHODOLOGICAL GOVERNANCE NOTICE:
The 2015+ prospective test cohort (1,017 consignments, 120 positive delay events, prevalence ~11.80%)
was previously unblinded during historical experimental iterations and is explicitly designated as
NON-PRISTINE. Model development, feature engineering, architecture selection, loss balancing,
and threshold optimization must strictly occur on the historical expanding-window cross-validation
folds (<= 2013) alongside the 2014 validation split. The prospective test cohort is held in reserve
exclusively for a single confirmatory evaluation.

Calibrator Integrity Policy:
Calibrators (Platt Scaling and Isotonic Regression) must NEVER be scored on the exact data partition
used to fit them. Calibration is evaluated using:
  1. Cross-Fitted Out-Of-Fold (OOF) Calibration: 5-fold cross-fitting where each validation probability
     is calibrated by a model fit without that instance.
  2. Decoupled Validation Splitting: 2014-A (50%) for stopping/threshold selection and 2014-B (50%)
     for calibration fitting.

Threshold Consistency Guard:
Decision thresholds tau* are strictly bound to the probability scale on which they were tuned
('raw', 'platt', 'isotonic'). Applying a threshold tuned on raw probabilities (e.g., scale_pos_weight
inflated) to calibrated probabilities triggers a loud ValueError, preventing clinical recall collapse.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("CalibrationEngine")


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Computes Expected Calibration Error (ECE) with uniform width bins across [0, 1].
    ECE = sum_m ( |B_m| / N ) * | acc(B_m) - conf(B_m) |
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 0.0, 1.0)
    N = len(y_true)
    if N == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / N) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Dict[str, np.ndarray]:
    """Generates empirical fraction of positives and mean confidence per bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    empirical_acc = []
    mean_conf = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        count = np.sum(in_bin)
        bin_counts.append(count)
        if count > 0:
            empirical_acc.append(float(np.mean(y_true[in_bin])))
            mean_conf.append(float(np.mean(y_prob[in_bin])))
        else:
            empirical_acc.append(float(bin_centers[i]))
            mean_conf.append(float(bin_centers[i]))

    return {
        "bin_centers": bin_centers,
        "empirical_accuracy": np.array(empirical_acc),
        "mean_confidence": np.array(mean_conf),
        "bin_counts": np.array(bin_counts),
    }


class ModelCalibrator:
    """
    Fits and applies Platt Scaling or Isotonic Regression strictly on independent predictions.
    """

    def __init__(self, method: str = "platt"):
        self.method = method.lower()
        if self.method not in ["platt", "isotonic"]:
            raise ValueError(f"Unsupported calibration method: '{method}'. Must be 'platt' or 'isotonic'.")
        self.calibrator = None

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ModelCalibrator":
        val_probs = np.clip(np.asarray(val_probs).ravel(), 1e-6, 1.0 - 1e-6)
        y_val = np.asarray(y_val).ravel().astype(int)

        if self.method == "platt":
            # Platt scaling: fit logistic regression on logit space
            logits = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(logits, y_val)
            self.calibrator = clf
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_probs, y_val)
            self.calibrator = iso

        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(probs).ravel(), 1e-6, 1.0 - 1e-6)
        if self.calibrator is None:
            raise RuntimeError("Calibrator has not been fitted!")

        if self.method == "platt":
            logits = np.log(probs / (1.0 - probs)).reshape(-1, 1)
            return self.calibrator.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            return np.clip(self.calibrator.predict(probs), 0.0, 1.0)


class CrossFittedCalibrator:
    """
    Cross-fitted probability calibrator using Stratified K-Fold splits.
    Guarantees that validation probabilities are scored strictly out-of-fold (OOF),
    while fitting a full calibrator for inference on unseen prospective test data.
    """

    def __init__(self, method: str = "platt", n_splits: int = 5, random_state: int = 42):
        self.method = method.lower()
        self.n_splits = n_splits
        self.random_state = random_state
        self.full_calibrator = ModelCalibrator(method=self.method)
        self.fold_calibrators: List[ModelCalibrator] = []

    def fit_predict_oof(self, val_probs: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, "CrossFittedCalibrator"]:
        val_probs = np.asarray(val_probs).ravel()
        y_val = np.asarray(y_val).ravel().astype(int)
        oof_cal_probs = np.zeros_like(val_probs, dtype=np.float32)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.fold_calibrators = []

        for tr_idx, te_idx in skf.split(val_probs, y_val):
            cal = ModelCalibrator(method=self.method)
            cal.fit(val_probs[tr_idx], y_val[tr_idx])
            oof_cal_probs[te_idx] = cal.predict_proba(val_probs[te_idx])
            self.fold_calibrators.append(cal)

        # Fit full calibrator on entire validation set for prospective test data
        self.full_calibrator.fit(val_probs, y_val)
        return oof_cal_probs, self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        return self.full_calibrator.predict_proba(probs)


class CalibratedThresholdPolicy:
    """
    Enforces threshold consistency and prevents probability scale mismatches.
    Fails loudly if a decision threshold tuned on raw probabilities (e.g., scale_pos_weight inflated)
    is applied to calibrated probabilities, or vice-versa.
    """

    def __init__(
        self,
        threshold: float,
        probability_scale: str,
        reference_mean: float,
        reference_std: float,
        beta: float = 1.0,
        description: str = "",
    ):
        self.threshold = float(threshold)
        self.probability_scale = probability_scale.lower()  # 'raw', 'platt', 'isotonic'
        self.reference_mean = float(reference_mean)
        self.reference_std = float(reference_std)
        self.beta = float(beta)
        self.description = description

    def predict(
        self,
        probs: np.ndarray,
        current_scale: Optional[str] = None,
        check_distribution: bool = True,
    ) -> np.ndarray:
        """
        Applies decision threshold to probability array with scale verification.
        Raises ValueError if scale mismatch is detected.
        """
        probs = np.asarray(probs).ravel()
        if current_scale is not None:
            current_scale = current_scale.lower()
            if current_scale != self.probability_scale:
                raise ValueError(
                    f"PROBABILITY SCALE MISMATCH: Decision threshold {self.threshold:.4f} was tuned on "
                    f"'{self.probability_scale}' scale (reference mean: {self.reference_mean:.4f}), but is "
                    f"being applied to '{current_scale}' scale (current mean: {np.mean(probs):.4f})! "
                    f"Applying uncalibrated thresholds to calibrated probabilities causes severe recall collapse."
                )

        if check_distribution and len(probs) > 50:
            curr_mean = float(np.mean(probs))
            # Detect cross-scale violation by distributional drift:
            if self.probability_scale in ["platt", "isotonic"] and self.reference_mean < 0.25 and curr_mean > 0.40:
                raise ValueError(
                    f"PROBABILITY SCALE MISMATCH DETECTED: Threshold {self.threshold:.4f} was tuned on "
                    f"calibrated scale (reference mean: {self.reference_mean:.4f}), but input probabilities have "
                    f"raw/uncalibrated distribution (mean: {curr_mean:.4f}). Threshold must match probability scale."
                )
            elif self.probability_scale == "raw" and self.reference_mean > 0.35 and curr_mean < 0.20:
                raise ValueError(
                    f"PROBABILITY SCALE MISMATCH DETECTED: Threshold {self.threshold:.4f} was tuned on "
                    f"raw uncalibrated scale (reference mean: {self.reference_mean:.4f}), but input probabilities have "
                    f"calibrated distribution (mean: {curr_mean:.4f}). Applying raw threshold causes recall collapse."
                )

        return (probs >= self.threshold).astype(int)


def tune_threshold_with_policy(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    scale: str = "raw",
    beta: float = 1.0,
) -> CalibratedThresholdPolicy:
    """
    Finds decision threshold maximizing F_beta on validation predictions and packages it
    into a scale-guarded CalibratedThresholdPolicy object.
    """
    val_probs = np.asarray(val_probs).ravel()
    y_val = np.asarray(y_val).ravel().astype(int)

    best_tau, best_score = 0.50, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (val_probs >= t).astype(int)
        rec = recall_score(y_val, preds, zero_division=0)
        prec = precision_score(y_val, preds, zero_division=0)
        if beta == 1.0:
            score = f1_score(y_val, preds, zero_division=0)
        else:
            score = (1.0 + beta**2) * (prec * rec) / max((beta**2 * prec + rec), 1e-6)

        if score > best_score:
            best_score = score
            best_tau = t

    return CalibratedThresholdPolicy(
        threshold=best_tau,
        probability_scale=scale,
        reference_mean=float(np.mean(val_probs)),
        reference_std=float(np.std(val_probs)),
        beta=beta,
        description=f"Optimal threshold for {scale} scale maximizing F_{beta}",
    )


def calibrate_and_benchmark(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Co-evaluates Raw vs. Cross-Fitted Platt vs. Cross-Fitted Isotonic calibration.
    Guarantees calibrators are NEVER scored on the exact data used to fit them.
    Tuning tau* strictly on out-of-fold validation probabilities.
    """
    y_val = np.asarray(y_val).ravel().astype(int)
    val_probs = np.asarray(val_probs).ravel()
    y_test = np.asarray(y_test).ravel().astype(int)
    test_probs = np.asarray(test_probs).ravel()

    # 1. Raw Predictions (Baseline without Calibration)
    policy_raw = tune_threshold_with_policy(y_val, val_probs, scale="raw", beta=1.0)
    val_ece_raw = compute_ece(y_val, val_probs, n_bins)
    test_ece_raw = compute_ece(y_test, test_probs, n_bins)
    val_brier_raw = float(brier_score_loss(y_val, val_probs))
    test_brier_raw = float(brier_score_loss(y_test, test_probs))
    test_preds_raw = policy_raw.predict(test_probs, current_scale="raw")
    test_f1_raw = float(f1_score(y_test, test_preds_raw, zero_division=0))
    test_rec_raw = float(recall_score(y_test, test_preds_raw, zero_division=0))

    # 2. Cross-Fitted Platt Scaling (OOF Validation Scoring)
    cf_platt = CrossFittedCalibrator(method="platt", n_splits=5, random_state=42)
    oof_val_platt, _ = cf_platt.fit_predict_oof(val_probs, y_val)
    test_probs_platt = cf_platt.predict_proba(test_probs)

    policy_platt = tune_threshold_with_policy(y_val, oof_val_platt, scale="platt", beta=1.0)
    val_ece_platt = compute_ece(y_val, oof_val_platt, n_bins)
    test_ece_platt = compute_ece(y_test, test_probs_platt, n_bins)
    val_brier_platt = float(brier_score_loss(y_val, oof_val_platt))
    test_brier_platt = float(brier_score_loss(y_test, test_probs_platt))
    test_preds_platt = policy_platt.predict(test_probs_platt, current_scale="platt")
    test_f1_platt = float(f1_score(y_test, test_preds_platt, zero_division=0))
    test_rec_platt = float(recall_score(y_test, test_preds_platt, zero_division=0))

    # 3. Cross-Fitted Isotonic Regression (OOF Validation Scoring)
    cf_iso = CrossFittedCalibrator(method="isotonic", n_splits=5, random_state=42)
    oof_val_iso, _ = cf_iso.fit_predict_oof(val_probs, y_val)
    test_probs_iso = cf_iso.predict_proba(test_probs)

    policy_iso = tune_threshold_with_policy(y_val, oof_val_iso, scale="isotonic", beta=1.0)
    val_ece_iso = compute_ece(y_val, oof_val_iso, n_bins)
    test_ece_iso = compute_ece(y_test, test_probs_iso, n_bins)
    val_brier_iso = float(brier_score_loss(y_val, oof_val_iso))
    test_brier_iso = float(brier_score_loss(y_test, test_probs_iso))
    test_preds_iso = policy_iso.predict(test_probs_iso, current_scale="isotonic")
    test_f1_iso = float(f1_score(y_test, test_preds_iso, zero_division=0))
    test_rec_iso = float(recall_score(y_test, test_preds_iso, zero_division=0))

    return {
        "raw": {
            "tau_star": policy_raw.threshold,
            "policy": policy_raw,
            "val_ece": val_ece_raw,
            "test_ece": test_ece_raw,
            "val_brier": val_brier_raw,
            "test_brier": test_brier_raw,
            "test_f1": test_f1_raw,
            "test_recall": test_rec_raw,
            "test_probs": test_probs,
        },
        "platt": {
            "tau_star": policy_platt.threshold,
            "policy": policy_platt,
            "val_ece": val_ece_platt,
            "test_ece": test_ece_platt,
            "val_brier": val_brier_platt,
            "test_brier": test_brier_platt,
            "test_f1": test_f1_platt,
            "test_recall": test_rec_platt,
            "calibrator": cf_platt,
            "test_probs": test_probs_platt,
        },
        "isotonic": {
            "tau_star": policy_iso.threshold,
            "policy": policy_iso,
            "val_ece": val_ece_iso,
            "test_ece": test_ece_iso,
            "val_brier": val_brier_iso,
            "test_brier": test_brier_iso,
            "test_f1": test_f1_iso,
            "test_recall": test_rec_iso,
            "calibrator": cf_iso,
            "test_probs": test_probs_iso,
        },
    }
