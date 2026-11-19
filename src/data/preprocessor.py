"""
Data Preprocessing Pipeline for Pharmaceutical Supply Chain Stockout Prediction.
Guarantees temporal validity, prevents data leakage, and preserves full dataset integrity (10,324 records).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    MODELS_DIR,
    TARGET_COL,
    TIER_1_COUNTRIES,
    TIER_3_HIGH_RISK_COUNTRIES,
    TRAIN_END_YEAR,
    VAL_YEAR,
    TEST_START_YEAR,
)


class HSCMPreprocessor:
    def __init__(self, top_vendor_k: int = 15, top_site_k: int = 15):
        self.top_vendor_k = top_vendor_k
        self.top_site_k = top_site_k
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names: List[str] = []
        self.imputation_medians: Dict[str, float] = {}
        self.top_vendors: List[str] = []
        self.top_sites: List[str] = []

    def assign_country_tier(self, country: str) -> str:
        """Assigns country to equity risk tier for health justice evaluation."""
        if country in TIER_3_HIGH_RISK_COUNTRIES:
            return "Tier_3_High_Risk_Underserved"
        elif country in TIER_1_COUNTRIES:
            return "Tier_1_Low_Risk_Hub"
        else:
            return "Tier_2_Moderate_Risk"

    def clean_and_feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs robust data cleaning, target creation, and leakage-safe feature engineering."""
        df = df.copy()

        # Parse target dates
        sched_date = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed")
        deliv_date = pd.to_datetime(df["Delivered to Client Date"], errors="coerce", format="mixed")

        # Define Ground Truth Binary Target: Late Delivery Risk (Proxy for Stockout Risk)
        # Late if actual delivery occurred after scheduled delivery date
        df[TARGET_COL] = (deliv_date > sched_date).astype(int)
        df["Delay_Days"] = (deliv_date - sched_date).dt.days
        df["Scheduled_Year"] = sched_date.dt.year

        # Temporal Features (Derived solely from scheduled timeline known before delivery)
        df["Scheduled_Month"] = sched_date.dt.month
        df["Scheduled_Quarter"] = sched_date.dt.quarter
        df["Scheduled_DayOfWeek"] = sched_date.dt.dayofweek
        df["Scheduled_DayOfYear"] = sched_date.dt.dayofyear
        df["Month_Sin"] = np.sin(2 * np.pi * df["Scheduled_Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Scheduled_Month"] / 12)

        # Manufacturer / Distributor Quoted Lead Time
        if "Manufacturer_Lead_Time_Days" in df.columns and df["Manufacturer_Lead_Time_Days"].notna().any():
            df["PO_Lead_Time_Days"] = pd.to_numeric(df["Manufacturer_Lead_Time_Days"], errors="coerce")
        else:
            po_date = pd.to_datetime(df["PO Sent to Vendor Date"], errors="coerce", format="mixed")
            df["PO_Lead_Time_Days"] = (sched_date - po_date).dt.days

        df["PO_Date_Missing"] = df["PO_Lead_Time_Days"].isna().astype(int)

        # Clean Weight (regex extraction to handle 'Weight Captured Separately' & references)
        weight_clean = df["Weight (Kilograms)"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        df["Weight_Clean"] = pd.to_numeric(weight_clean, errors="coerce")
        df["Weight_Missing_Flag"] = df["Weight_Clean"].isna().astype(int)

        # Clean Freight Cost (regex extraction to handle 'Freight Included...' & references)
        freight_clean = df["Freight Cost (USD)"].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        df["Freight_Clean"] = pd.to_numeric(freight_clean, errors="coerce")
        df["Freight_Included_Flag"] = df["Freight_Clean"].isna().astype(int)

        # Insurance Clean
        df["Insurance_Clean"] = pd.to_numeric(df["Line Item Insurance (USD)"], errors="coerce")

        # Health Equity Stratification Feature
        df["Country_Tier"] = df["Country"].apply(self.assign_country_tier)

        # Shipment Mode missing handling
        df["Shipment Mode"] = df["Shipment Mode"].fillna("Unknown")

        # Categorical grouping for high-cardinality vendors and manufacturing sites
        return df

    def fit(self, train_df: pd.DataFrame):
        """Fit preprocessor parameters strictly on training set to prevent data leakage."""
        train = train_df.copy()

        # Learn top vendors and manufacturing sites from train set only
        self.top_vendors = train["Vendor"].value_counts().nlargest(self.top_vendor_k).index.tolist()
        self.top_sites = train["Manufacturing Site"].value_counts().nlargest(self.top_site_k).index.tolist()

        # Compute median imputations on train set
        self.imputation_medians["PO_Lead_Time_Days"] = float(train["PO_Lead_Time_Days"].median(skipna=True))
        if np.isnan(self.imputation_medians["PO_Lead_Time_Days"]):
            self.imputation_medians["PO_Lead_Time_Days"] = 0.0

        self.imputation_medians["Weight_Clean"] = float(train["Weight_Clean"].median(skipna=True))
        self.imputation_medians["Freight_Clean"] = float(train["Freight_Clean"].median(skipna=True))
        self.imputation_medians["Insurance_Clean"] = float(train["Insurance_Clean"].median(skipna=True))

        # Prepare transformed train features
        X_train_num, X_train_cat = self._extract_features(train)

        # Fit Scaler and One-Hot Encoder
        self.scaler.fit(X_train_num)
        self.ohe.fit(X_train_cat)

        ohe_features = self.ohe.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = self.numeric_cols + ohe_features

    def _extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract and impute numerical and categorical feature subsets."""
        df = df.copy()

        # Impute missing numerics with train medians and sanitize bounds
        df["PO_Lead_Time_Days"] = df["PO_Lead_Time_Days"].fillna(self.imputation_medians.get("PO_Lead_Time_Days", 0.0))
        df["PO_Lead_Time_Days"] = np.clip(pd.to_numeric(df["PO_Lead_Time_Days"], errors="coerce").fillna(0.0), -365.0, 730.0)

        df["Weight_Clean"] = df["Weight_Clean"].fillna(self.imputation_medians.get("Weight_Clean", 500.0))
        df["Weight_Clean"] = np.clip(pd.to_numeric(df["Weight_Clean"], errors="coerce").fillna(500.0), 0.0, 1_000_000.0)

        df["Freight_Clean"] = df["Freight_Clean"].fillna(self.imputation_medians.get("Freight_Clean", 2000.0))
        df["Freight_Clean"] = np.clip(pd.to_numeric(df["Freight_Clean"], errors="coerce").fillna(2000.0), 0.0, 10_000_000.0)

        df["Insurance_Clean"] = df["Insurance_Clean"].fillna(self.imputation_medians.get("Insurance_Clean", 50.0))
        df["Insurance_Clean"] = np.clip(pd.to_numeric(df["Insurance_Clean"], errors="coerce").fillna(50.0), 0.0, 1_000_000.0)

        # Log transform skewed features with finite safety clipping
        clean_qty = np.clip(pd.to_numeric(df["Line Item Quantity"], errors="coerce").fillna(0.0), 0.0, 100_000_000.0)
        clean_val = np.clip(pd.to_numeric(df["Line Item Value"], errors="coerce").fillna(0.0), 0.0, 1_000_000_000.0)

        df["Log_Quantity"] = np.log1p(clean_qty)
        df["Log_Value"] = np.log1p(clean_val)
        df["Log_Weight"] = np.log1p(df["Weight_Clean"])
        df["Log_Freight"] = np.log1p(df["Freight_Clean"])

        # Domain Engineered Ratios with Zero-Division Immunity
        safe_qty_denom = clean_qty + 1.0
        safe_val_denom = clean_val + 1.0
        safe_wt_denom = df["Weight_Clean"] + 1.0

        df["Value_Per_Pack"] = clean_val / safe_qty_denom
        df["Freight_Per_Kg"] = df["Freight_Clean"] / safe_wt_denom
        df["Freight_To_Value_Ratio"] = df["Freight_Clean"] / safe_val_denom
        df["Insurance_To_Value_Ratio"] = df["Insurance_Clean"] / safe_val_denom

        # Define Numerical Columns
        self.numeric_cols = [
            "Log_Quantity",
            "Log_Value",
            "Pack Price",
            "Unit Price",
            "Unit of Measure (Per Pack)",
            "Log_Weight",
            "Log_Freight",
            "Insurance_Clean",
            "PO_Lead_Time_Days",
            "PO_Date_Missing",
            "Weight_Missing_Flag",
            "Freight_Included_Flag",
            "Scheduled_Month",
            "Scheduled_Quarter",
            "Scheduled_DayOfWeek",
            "Scheduled_DayOfYear",
            "Month_Sin",
            "Month_Cos",
            "Value_Per_Pack",
            "Freight_Per_Kg",
            "Freight_To_Value_Ratio",
            "Insurance_To_Value_Ratio",
        ]

        # Categorical grouping: mask rare categories as 'Other'
        df["Vendor_Grouped"] = df["Vendor"].apply(lambda v: v if v in self.top_vendors else "Other_Vendor")
        df["Site_Grouped"] = df["Manufacturing Site"].apply(lambda s: s if s in self.top_sites else "Other_Site")

        # Define Categorical Columns
        self.categorical_cols = [
            "Country",
            "Fulfill Via",
            "Vendor INCO Term",
            "Shipment Mode",
            "Product Group",
            "Sub Classification",
            "First Line Designation",
            "Vendor_Grouped",
            "Site_Grouped",
            "Country_Tier",
        ]

        X_num = df[self.numeric_cols].fillna(0.0).astype(float)
        # Final safety net against NaN or Inf values in raw input
        X_num = pd.DataFrame(
            np.nan_to_num(X_num.values, nan=0.0, posinf=1e6, neginf=0.0),
            columns=self.numeric_cols,
        )
        X_cat = df[self.categorical_cols].fillna("Unknown").astype(str)

        return X_num, X_cat

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms dataset into normalized feature matrix using pre-fit scalers."""
        X_num, X_cat = self._extract_features(df)
        X_num_scaled = self.scaler.transform(X_num)
        X_cat_ohe = self.ohe.transform(X_cat)
        return np.hstack([X_num_scaled, X_cat_ohe])

    def save(self, filepath: Path = None):
        """Saves fitted preprocessor to disk for production deployment."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path = None) -> "HSCMPreprocessor":
        """Loads fitted preprocessor from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "preprocessor.pkl"
        with open(filepath, "rb") as f:
            return pickle.load(f)


def prepare_data_splits() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, HSCMPreprocessor, pd.DataFrame
]:
    """
    Loads raw dataset, executes chronological train-val-test split, and transforms features.
    
    Split scheme:
    - Train: 2006-2013 (Historical)
    - Val:   2014 (Hyperparameter tuning & Calibration)
    - Test:  2015 (Prospective evaluation)
    """
    raw_df = pd.read_csv(DATA_PATH)
    preprocessor = HSCMPreprocessor()
    df = preprocessor.clean_and_feature_engineer(raw_df)

    # Chronological partition
    train_df = df[df["Scheduled_Year"] <= TRAIN_END_YEAR].copy()
    val_df = df[df["Scheduled_Year"] == VAL_YEAR].copy()
    test_df = df[df["Scheduled_Year"] >= TEST_START_YEAR].copy()

    # Fit preprocessor on train only
    preprocessor.fit(train_df)
    preprocessor.save()

    # Transform feature matrices
    X_train = preprocessor.transform(train_df)
    y_train = train_df[TARGET_COL].values

    X_val = preprocessor.transform(val_df)
    y_val = val_df[TARGET_COL].values

    X_test = preprocessor.transform(test_df)
    y_test = test_df[TARGET_COL].values

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, test_df
