"""
Probability Calibration Evaluation and Recalibration Module.
Features:
- 15-bin Expected Calibration Error (ECE) computation
- Reliability curve data generation
- Brier score computation
- Platt scaling (logistic) and Isotonic regression fit strictly on the 2014 validation set
- Threshold tau* recomputation before and after calibration
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


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
    Fits and applies Platt Scaling or Isotonic Regression strictly on validation predictions.
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


def calibrate_and_benchmark(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Co-evaluates raw vs. Platt vs. Isotonic calibration.
    Re-tunes tau* on calibrated validation probabilities and reports prospective test results.
    """

    def find_best_tau(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = f1_score(targets, (probs >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return float(best_t), float(best_f1)

    # 1. Raw Calibration Metrics
    val_ece_raw = compute_ece(y_val, val_probs, n_bins)
    test_ece_raw = compute_ece(y_test, test_probs, n_bins)
    val_brier_raw = float(brier_score_loss(y_val, val_probs))
    test_brier_raw = float(brier_score_loss(y_test, test_probs))
    tau_raw, val_f1_raw = find_best_tau(val_probs, y_val)
    test_f1_raw = float(f1_score(y_test, (test_probs >= tau_raw).astype(int), zero_division=0))

    # 2. Platt Scaling (fit on val only)
    platt = ModelCalibrator(method="platt").fit(val_probs, y_val)
    val_probs_platt = platt.predict_proba(val_probs)
    test_probs_platt = platt.predict_proba(test_probs)

    val_ece_platt = compute_ece(y_val, val_probs_platt, n_bins)
    test_ece_platt = compute_ece(y_test, test_probs_platt, n_bins)
    val_brier_platt = float(brier_score_loss(y_val, val_probs_platt))
    test_brier_platt = float(brier_score_loss(y_test, test_probs_platt))
    tau_platt, val_f1_platt = find_best_tau(val_probs_platt, y_val)
    test_f1_platt = float(f1_score(y_test, (test_probs_platt >= tau_platt).astype(int), zero_division=0))

    # 3. Isotonic Regression (fit on val only)
    iso = ModelCalibrator(method="isotonic").fit(val_probs, y_val)
    val_probs_iso = iso.predict_proba(val_probs)
    test_probs_iso = iso.predict_proba(test_probs)

    val_ece_iso = compute_ece(y_val, val_probs_iso, n_bins)
    test_ece_iso = compute_ece(y_test, test_probs_iso, n_bins)
    val_brier_iso = float(brier_score_loss(y_val, val_probs_iso))
    test_brier_iso = float(brier_score_loss(y_test, test_probs_iso))
    tau_iso, val_f1_iso = find_best_tau(val_probs_iso, y_val)
    test_f1_iso = float(f1_score(y_test, (test_probs_iso >= tau_iso).astype(int), zero_division=0))

    return {
        "raw": {
            "tau_star": tau_raw,
            "val_ece": val_ece_raw,
            "test_ece": test_ece_raw,
            "val_brier": val_brier_raw,
            "test_brier": test_brier_raw,
            "test_f1": test_f1_raw,
            "test_probs": test_probs,
        },
        "platt": {
            "tau_star": tau_platt,
            "val_ece": val_ece_platt,
            "test_ece": test_ece_platt,
            "val_brier": val_brier_platt,
            "test_brier": test_brier_platt,
            "test_f1": test_f1_platt,
            "calibrator": platt,
            "test_probs": test_probs_platt,
        },
        "isotonic": {
            "tau_star": tau_iso,
            "val_ece": val_ece_iso,
            "test_ece": test_ece_iso,
            "val_brier": val_brier_iso,
            "test_brier": test_brier_iso,
            "test_f1": test_f1_iso,
            "calibrator": iso,
            "test_probs": test_probs_iso,
        },
    }
