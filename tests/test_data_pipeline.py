"""
Unit tests for Data Cleaning, Waterfall Accounting, and Leakage Guard.
Must run fast (< 10 seconds) on synthetic fixture and small subset.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

# Ensure Implementation root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean_dataset_with_waterfall
from src.data.leakage_guard import (
    assert_no_post_outcome_features,
    assert_no_high_correlation_leakage,
    verify_fit_on_train_only,
    LeakageGuardError,
)
from src.data.preprocessor import HSCMPreprocessor


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """Creates synthetic raw dataframe simulating SCMS anomalies."""
    data = {
        "Scheduled Delivery Date": [
            "2012-01-10", "2012-03-15", "2013-05-20", "2014-06-10", "2015-08-01",
            "2015-09-12", "invalid-date", "2012-01-10" # one invalid, one duplicate
        ],
        "Delivered to Client Date": [
            "2012-01-15", # Late (1)
            "2012-03-10", # On time (0)
            "2013-05-20", # On time (0)
            "2014-06-25", # Late (1)
            "2015-08-01", # On time (0)
            "2015-08-30", # On time (0)
            "2015-01-01", # Will be dropped due to invalid sched date
            "2012-01-15"  # Duplicate row
        ],
        "PO Sent to Vendor Date": [
            "2011-12-01", "Date Not Captured", "2013-01-01", "Date Not Captured",
            "2015-01-01", "2015-02-01", "2014-01-01", "2011-12-01"
        ],
        "Weight (Kilograms)": [
            "100", "See ASN/DN", "250", "See ASN", "500", "750", "100", "100"
        ],
        "Freight Cost (USD)": [
            "1000", "Freight Included in Process", "See ASN/DN", "2500", "4000", "5000", "1000", "1000"
        ],
        "Country": ["South Africa", "Nigeria", "Congo, DRC", "Kenya", "South Sudan", "Vietnam", "Uganda", "South Africa"],
        "Vendor": ["Vendor A", "Vendor B", "Vendor A", "Vendor C", "Vendor A", "Vendor B", "Vendor A", "Vendor A"],
        "Manufacturing Site": ["Site 1", "Site 2", "Site 1", "Site 3", "Site 1", "Site 2", "Site 1", "Site 1"],
        "Shipment Mode": ["Air", "Air", "Truck", "Ocean", "Air", "Truck", "Air", "Air"],
        "Fulfill Via": ["Direct Drop", "Direct Drop", "From RDC", "Direct Drop", "From RDC", "Direct Drop", "Direct Drop", "Direct Drop"],
        "Vendor INCO Term": ["EXW", "CIP", "DDP", "FCA", "EXW", "CIP", "EXW", "EXW"],
        "Product Group": ["ARV", "HRDT", "ARV", "ARV", "HRDT", "ARV", "ARV", "ARV"],
        "Sub Classification": ["Adult", "HIV test", "Pediatric", "Adult", "HIV test", "Adult", "Adult", "Adult"],
        "First Line Designation": ["Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
        "Line Item Quantity": [1000, 2000, 1500, 3000, 500, 4000, 1000, 1000],
        "Line Item Value": [10000, 25000, 18000, 45000, 6000, 60000, 10000, 10000],
        "Pack Price": [10, 12.5, 12, 15, 12, 15, 10, 10],
        "Unit Price": [0.1, 0.25, 0.2, 0.15, 0.2, 0.15, 0.1, 0.1],
        "Unit of Measure (Per Pack)": [100, 50, 60, 100, 60, 100, 100, 100],
        "Line Item Insurance (USD)": [15, 30, 25, 60, 10, 80, 15, 15],
    }
    return pd.DataFrame(data)


def test_cleaner_waterfall_accounting(synthetic_raw_df, tmp_path):
    waterfall_path = tmp_path / "waterfall.json"
    cleaned_df, waterfall = clean_dataset_with_waterfall(
        synthetic_raw_df,
        waterfall_output_path=waterfall_path,
        target_col="Late_Delivery_Risk",
    )

    assert waterfall["initial_row_count"] == 8
    assert waterfall["duplicates_removed"] == 1
    assert waterfall["date_not_captured_detected"] >= 2
    assert waterfall["see_asn_dn_detected"] >= 2
    assert waterfall["unparseable_target_dates_dropped"] == 1
    assert waterfall["final_cleaned_row_count"] == 6
    assert waterfall_path.exists()

    # Ground truth label verification
    # Row 0: 2012-01-15 > 2012-01-10 -> Late (1)
    # Row 1: 2012-03-10 <= 2012-03-15 -> On time (0)
    assert cleaned_df.iloc[0]["Late_Delivery_Risk"] == 1
    assert cleaned_df.iloc[1]["Late_Delivery_Risk"] == 0


def test_leakage_guard_post_outcome_column_detection():
    safe_features = ["Scheduled_Month", "Weight_Clean", "PO_Lead_Time_Days", "Log_Quantity"]
    assert_no_post_outcome_features(safe_features) # should pass

    leaked_features = ["Scheduled_Month", "Delivered to Client Date", "Log_Quantity"]
    with pytest.raises(LeakageGuardError, match="Post-outcome column"):
        assert_no_post_outcome_features(leaked_features)

    leaked_delay_days = ["Scheduled_Month", "Delay_Days", "Log_Quantity"]
    with pytest.raises(LeakageGuardError, match="Post-outcome column"):
        assert_no_post_outcome_features(leaked_delay_days)


def test_leakage_guard_high_correlation():
    np.random.seed(42)
    N = 100
    y = np.random.binomial(1, 0.3, N)

    # Feature 0: random, low correlation
    f0 = np.random.randn(N)
    # Feature 1: synthetic leakage feature, correlation ~ 0.99 with y
    f1 = y.astype(float) + np.random.normal(0, 0.02, N)

    X_safe = np.column_stack([f0, np.random.randn(N)])
    assert_no_high_correlation_leakage(X_safe, y, ["feat0", "feat1"], threshold=0.95)

    X_leaked = np.column_stack([f0, f1])
    with pytest.raises(LeakageGuardError, match="CORRELATION LEAKAGE VIOLATION"):
        assert_no_high_correlation_leakage(X_leaked, y, ["feat0", "leakage_feat"], threshold=0.95)


def test_fit_on_train_only(synthetic_raw_df):
    cleaned_df, _ = clean_dataset_with_waterfall(synthetic_raw_df)
    train_df = cleaned_df[cleaned_df["Scheduled_Year"] <= 2013].copy()
    val_df = cleaned_df[cleaned_df["Scheduled_Year"] == 2014].copy()
    test_df = cleaned_df[cleaned_df["Scheduled_Year"] >= 2015].copy()

    prep = HSCMPreprocessor()
    engineered_train = prep.clean_and_feature_engineer(train_df)
    prep.fit(engineered_train)

    # Validate that parameters reflect train_df
    assert verify_fit_on_train_only(engineered_train, val_df, test_df, prep) is True
