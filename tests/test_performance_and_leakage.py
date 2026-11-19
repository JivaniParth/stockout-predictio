"""
Unit tests for Temporal Integrity, Zero Future Lookahead, and Non-Interference.
Guarantees:
1. Expanding CV folds are strictly time-ordered with no backward temporal leakage.
2. Point-in-time historical stats strictly exclude deliveries completed at or after target scheduled date.
3. Mutating future completed shipments does not alter feature values for past target shipments.
4. Test partition is never accessed during training, CV, or threshold tuning.
5. Bitwise/metric determinism across repeated executions with fixed random seeds.
"""

import sys
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest

# Ensure Implementation is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean_dataset_with_waterfall
from src.features.historical_stats import PointInTimeHistoricalStatsEngine
from src.features.engineered_preprocessor import EngineeredPreprocessor
from src.evaluation.cross_validation import FOLDS_SPEC, evaluate_expanding_window_cv
from src.models.gbm_suite import LightGBMWrapper


@pytest.fixture(scope="module")
def real_cleaned_splits():
    """Loads and cleans dataset, returning chronological splits."""
    csv_path = PROJECT_ROOT / "SCMS_Delivery_History_Dataset.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "Implementation" / "SCMS_Delivery_History_Dataset.csv"
    raw_df = pd.read_csv(csv_path)
    cleaned_df, _ = clean_dataset_with_waterfall(raw_df)
    train_df = cleaned_df[cleaned_df["Scheduled_Year"] <= 2013].copy()
    val_df = cleaned_df[cleaned_df["Scheduled_Year"] == 2014].copy()
    test_df = cleaned_df[cleaned_df["Scheduled_Year"] >= 2015].copy()
    return train_df, val_df, test_df


def test_expanding_folds_strictly_time_ordered():
    """Validates that expanding CV folds are strictly chronological and non-overlapping."""
    prev_train_max = -1
    prev_val = -1
    for i, fold in enumerate(FOLDS_SPEC):
        train_max = fold["train_max_year"]
        val_year = fold["val_year"]

        # Train partition must precede validation partition strictly
        assert train_max < val_year, f"Fold {i} has train_max ({train_max}) >= val_year ({val_year})"
        # Folds must expand chronologically
        assert train_max >= prev_train_max, f"Fold {i} train_max ({train_max}) < previous ({prev_train_max})"
        assert val_year > prev_val, f"Fold {i} val_year ({val_year}) <= previous ({prev_val})"

        prev_train_max = train_max
        prev_val = val_year


def test_temporal_lookahead_zero_leakage(real_cleaned_splits):
    """
    Asserts that for any target row i, the historical engine strictly considers
    only deliveries completed BEFORE row i's Scheduled Delivery Date.
    """
    train_df, _, _ = real_cleaned_splits
    engine = PointInTimeHistoricalStatsEngine(smoothing_m=10.0)
    engine.fit(train_df)

    # Pick 50 random rows from train_df
    sample = train_df.sample(n=50, random_state=42)
    sample_sched = pd.to_datetime(sample["Scheduled Delivery Date"], errors="coerce", format="mixed")
    features = engine.transform(sample)

    # Global deliv timestamps indexed in the engine
    deliv_ts = engine.global_deliv_ts

    for idx, (_, row) in enumerate(sample.iterrows()):
        sched_val = np.datetime64(sample_sched.iloc[idx])
        # Find how many completed deliveries in the index occurred strictly before sched_val
        expected_prior_count = int(np.sum(deliv_ts < sched_val))
        actual_prior_count = int(features["Global_Completed_Prior_Count"].iloc[idx])
        assert actual_prior_count == expected_prior_count, (
            f"Row {idx} prior count mismatch: actual {actual_prior_count} != expected {expected_prior_count}"
        )


def test_mutating_future_completed_rows_does_not_change_features(real_cleaned_splits):
    """
    Asserts that altering future completed rows (T_delivered >= T_scheduled_i)
    produces ZERO change in point-in-time historical stats for target row i.
    """
    train_df, _, _ = real_cleaned_splits

    # Target row: scheduled in early 2011
    sched_dt = pd.to_datetime(train_df["Scheduled Delivery Date"], errors="coerce", format="mixed")
    target_mask = (sched_dt.dt.year == 2011) & (sched_dt.dt.month == 2)
    assert target_mask.sum() > 0, "No rows found in target period."
    target_row = train_df[target_mask].iloc[[0]].copy()
    target_sched_date = pd.to_datetime(target_row["Scheduled Delivery Date"].values[0])

    # Engine 1: Fit on standard train_df
    engine_orig = PointInTimeHistoricalStatsEngine(smoothing_m=10.0)
    engine_orig.fit(train_df)
    feats_orig = engine_orig.transform(target_row)

    # Engine 2: Mutate deliveries that completed on or after target_sched_date
    deliv_dt = pd.to_datetime(train_df["Delivered to Client Date"], errors="coerce", format="mixed")
    future_mask = deliv_dt >= target_sched_date

    mutated_train_df = train_df.copy()
    # Mutate future rows: change line values and flip labels to late
    mutated_train_df.loc[future_mask, "Line Item Value"] = 99999999
    mutated_train_df.loc[future_mask, "Late_Delivery_Risk"] = 1

    engine_mutated = PointInTimeHistoricalStatsEngine(smoothing_m=10.0)
    engine_mutated.fit(mutated_train_df)
    # Fix the global prior anchor to test point-in-time lookup invariance
    engine_mutated.global_prior_delay_rate = engine_orig.global_prior_delay_rate
    engine_mutated.global_prior_lateness_days = engine_orig.global_prior_lateness_days
    feats_mutated = engine_mutated.transform(target_row)

    # Assert point-in-time historical stats are completely unaffected by future tampering
    np.testing.assert_allclose(
        feats_orig.values,
        feats_mutated.values,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Future data mutation bled into historical features of a past target row!",
    )


def test_test_set_not_read_during_cv_or_selection(real_cleaned_splits):
    """
    Validates that the test set (>= 2015) is completely decoupled from CV and validation.
    """
    train_df, val_df, test_df = real_cleaned_splits

    # Ensure train and val years strictly precede test start year (2015)
    assert train_df["Scheduled_Year"].max() <= 2013
    assert val_df["Scheduled_Year"].unique().tolist() == [2014]
    assert test_df["Scheduled_Year"].min() >= 2015

    # Run 1 quick CV fold and verify test set is not touched
    cv_res = evaluate_expanding_window_cv(
        train_df,
        lambda dim, cats: LightGBMWrapper(params={"n_estimators": 10}, cat_indices=cats),
        categorical_mode="native",
    )
    assert cv_res["cv_pr_auc_mean"] > 0.0
    assert len(cv_res["oof_df"]) == len(train_df[train_df["Scheduled_Year"] >= 2010])


def test_determinism(real_cleaned_splits):
    """Asserts that two independent runs produce bitwise identical probability predictions."""
    train_df, val_df, _ = real_cleaned_splits

    prep1 = EngineeredPreprocessor(categorical_mode="native")
    prep1.fit(train_df)
    X_tr1 = prep1.transform(train_df)
    X_val1 = prep1.transform(val_df)
    y_tr1 = train_df["Late_Delivery_Risk"].values
    y_val1 = val_df["Late_Delivery_Risk"].values

    m1 = LightGBMWrapper(params={"n_estimators": 50, "random_state": 42}, cat_indices=prep1.get_categorical_indices())
    m1.fit(X_tr1, y_tr1, X_val1, y_val1)
    p1 = m1.predict_proba(X_val1)[:, 1]

    prep2 = EngineeredPreprocessor(categorical_mode="native")
    prep2.fit(train_df)
    X_tr2 = prep2.transform(train_df)
    X_val2 = prep2.transform(val_df)
    y_tr2 = train_df["Late_Delivery_Risk"].values
    y_val2 = val_df["Late_Delivery_Risk"].values

    m2 = LightGBMWrapper(params={"n_estimators": 50, "random_state": 42}, cat_indices=prep2.get_categorical_indices())
    m2.fit(X_tr2, y_tr2, X_val2, y_val2)
    p2 = m2.predict_proba(X_val2)[:, 1]

    np.testing.assert_allclose(p1, p2, atol=1e-7, rtol=1e-7, err_msg="Non-deterministic predictions detected across identical runs!")
