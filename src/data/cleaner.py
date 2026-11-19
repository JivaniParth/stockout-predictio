"""
Data Cleaning and Waterfall Accounting Module.
Handles anomalies such as "Date Not Captured", "See ASN/DN", unparseable dates,
and duplicate records before the temporal split.
Generates an audit-ready row-count waterfall saved to reports/data_waterfall.json.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import numpy as np
import pandas as pd


def clean_dataset_with_waterfall(
    raw_df: pd.DataFrame,
    waterfall_output_path: Optional[Path] = None,
    target_col: str = "Late_Delivery_Risk",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw SCMS dataframe and records full accounting waterfall.

    Waterfall steps:
    1. Initial record count.
    2. Exact duplicate row detection and removal.
    3. Parsing and accounting of "Date Not Captured" entries.
    4. Parsing and accounting of "See ASN/DN" in Freight & Weight.
    5. Dropping and accounting of rows where target label cannot be computed.
    6. Computation of explicit ground truth label:
       delay = (Delivered to Client Date > Scheduled Delivery Date).
    7. Post-cleaning validation.
    """
    df = raw_df.copy()
    initial_count = len(df)

    waterfall: Dict[str, Any] = {
        "initial_row_count": initial_count,
        "duplicates_removed": 0,
        "date_not_captured_detected": 0,
        "see_asn_dn_detected": 0,
        "unparseable_target_dates_dropped": 0,
        "uncomputable_label_dropped": 0,
        "final_cleaned_row_count": 0,
        "class_distribution": {},
    }

    # Step 1: Duplicate Detection & Removal
    dup_mask = df.duplicated()
    dups_count = int(dup_mask.sum())
    waterfall["duplicates_removed"] = dups_count
    if dups_count > 0:
        df = df[~dup_mask].copy()

    # Step 2: Handle "Date Not Captured"
    for col in df.select_dtypes(include=["object"]).columns:
        mask = df[col].astype(str).str.contains(r"Date Not Captured", case=False, na=False)
        count = int(mask.sum())
        if count > 0:
            waterfall["date_not_captured_detected"] += count
            df.loc[mask, col] = np.nan

    # Step 3: Handle "See ASN/DN" in Weight and Freight
    for col in ["Weight (Kilograms)", "Freight Cost (USD)"]:
        if col in df.columns:
            mask = df[col].astype(str).str.contains(r"See ASN", case=False, na=False)
            count = int(mask.sum())
            if count > 0:
                waterfall["see_asn_dn_detected"] += count
                # The text indicates freight/weight is captured in another document;
                # set to NaN so robust median imputation handles it with explicit missing flag
                df.loc[mask, col] = np.nan

    # Step 4: Parse Scheduled Delivery Date & Delivered to Client Date
    if "Scheduled Delivery Date" not in df.columns or "Delivered to Client Date" not in df.columns:
        raise ValueError("Missing critical delivery date columns in raw dataset.")

    sched_dt = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed")
    deliv_dt = pd.to_datetime(df["Delivered to Client Date"], errors="coerce", format="mixed")

    # Uncomputable label filter
    invalid_mask = sched_dt.isna() | deliv_dt.isna()
    dropped_uncomputable = int(invalid_mask.sum())
    waterfall["unparseable_target_dates_dropped"] = dropped_uncomputable
    waterfall["uncomputable_label_dropped"] = dropped_uncomputable

    if dropped_uncomputable > 0:
        df = df[~invalid_mask].copy()
        sched_dt = sched_dt[~invalid_mask]
        deliv_dt = deliv_dt[~invalid_mask]

    # Step 5: Define ground truth binary label
    # delay = 1 if Delivered to Client Date > Scheduled Delivery Date else 0
    df[target_col] = (deliv_dt > sched_dt).astype(int)
    df["Scheduled_Year"] = sched_dt.dt.year

    waterfall["final_cleaned_row_count"] = len(df)
    waterfall["class_distribution"] = {
        "on_time_0": int((df[target_col] == 0).sum()),
        "late_delay_1": int((df[target_col] == 1).sum()),
        "delay_rate": float(df[target_col].mean()),
    }

    # Save to disk if path requested
    if waterfall_output_path is not None:
        waterfall_output_path = Path(waterfall_output_path)
        waterfall_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(waterfall_output_path, "w", encoding="utf-8") as f:
            json.dump(waterfall, f, indent=2)

    return df, waterfall
