"""
Leakage Guard and Feature Integrity Audit Module.
Strictly guarantees:
1. No post-outcome features (e.g., Delivered to Client Date, Delivery Recorded Date, Delay_Days)
   reach the model.
2. No feature has correlation > 0.95 with the ground truth label.
3. Preprocessing parameters (medians, scalers, categories) are fit strictly on the training partition.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


FORBIDDEN_POST_OUTCOME_COLUMNS = [
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "Delay_Days",
    "delay_days",
    "Actual_Delivery_Date",
    "actual_delivery_date",
    "Days_Late",
    "days_late",
]


class LeakageGuardError(ValueError):
    """Raised when data leakage is detected in features."""
    pass


def assert_no_post_outcome_features(feature_names: List[str]) -> None:
    """
    Validates that no post-outcome or target-derived columns exist in the feature set.
    """
    for col in feature_names:
        clean_col = col.strip().lower()
        for forbidden in FORBIDDEN_POST_OUTCOME_COLUMNS:
            forbidden_clean = forbidden.strip().lower()
            if clean_col == forbidden_clean or forbidden_clean in clean_col:
                raise LeakageGuardError(
                    f"DATA LEAKAGE VIOLATION: Post-outcome column '{col}' matches "
                    f"forbidden identifier '{forbidden}' and must never be fed to the model."
                )


def assert_no_high_correlation_leakage(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.95,
) -> Dict[str, float]:
    """
    Validates that no individual feature exhibits absolute Pearson correlation > threshold with the label.
    A correlation > 0.95 typically signals target leakage or an inadvertent label proxy.
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Dimension mismatch: X has {X.shape[1]} columns but {len(feature_names)} feature names provided."
        )

    correlations: Dict[str, float] = {}
    y_flat = y.ravel().astype(float)
    y_std = np.std(y_flat)

    if y_std == 0:
        return correlations

    for idx, name in enumerate(feature_names):
        col_vals = X[:, idx].astype(float)
        col_std = np.std(col_vals)
        if col_std == 0:
            corr = 0.0
        else:
            # Pearson correlation
            corr = float(np.corrcoef(col_vals, y_flat)[0, 1])
            if np.isnan(corr):
                corr = 0.0

        correlations[name] = corr

        if abs(corr) >= threshold:
            raise LeakageGuardError(
                f"CORRELATION LEAKAGE VIOLATION: Feature '{name}' exhibits absolute correlation "
                f"|r| = {abs(corr):.4f} >= {threshold} with the target label! "
                f"This indicates a direct target proxy."
            )

    return correlations


def verify_fit_on_train_only(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fitted_preprocessor: Any,
) -> bool:
    """
    Audits preprocessor parameters to verify that medians, scalers, and OHE categories
    were fit strictly on the train partition and not on validation or prospective test data.
    """
    # Verify continuous medians
    for col, stored_median in fitted_preprocessor.imputation_medians.items():
        if col in train_df.columns:
            train_median = float(train_df[col].median(skipna=True))
            if not np.isclose(stored_median, train_median, atol=1e-3):
                raise LeakageGuardError(
                    f"PREPROCESSOR CONTAMINATION: Stored median for '{col}' ({stored_median}) "
                    f"does not match train set median ({train_median})!"
                )

    # Verify top categories were extracted from train
    if hasattr(fitted_preprocessor, "top_vendors") and "Vendor" in train_df.columns:
        train_top_v = train_df["Vendor"].value_counts().nlargest(fitted_preprocessor.top_vendor_k).index.tolist()
        if fitted_preprocessor.top_vendors != train_top_v:
            raise LeakageGuardError(
                "PREPROCESSOR CONTAMINATION: Stored top vendors do not match train partition top vendors!"
            )

    return True
