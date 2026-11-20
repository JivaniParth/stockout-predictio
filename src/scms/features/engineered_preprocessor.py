"""
Leakage-Safe Feature Engineering Preprocessor.
Features:
- Point-in-time historical delay rates and lateness
- Rolling 90d, 180d, 365d volume and delays
- Comprehensive logistical lead times (PO-to-Scheduled, PQ-to-PO, PQ-to-Scheduled)
- Domain financial and density ratios
- Harmonic calendar encodings (EXCLUDING Scheduled_Year as input)
- Dual categorical handling: Native Integer Encoding (for LightGBM/CatBoost/Embeddings) and One-Hot Encoding
- All scalers, medians, and priors fit strictly on training partition
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder

from src.features.historical_stats import PointInTimeHistoricalStatsEngine


CATEGORICAL_COLUMNS = [
    "Country",
    "Managed By",
    "Fulfill Via",
    "Vendor INCO Term",
    "Shipment Mode",
    "Product Group",
    "Sub Classification",
    "Vendor",
    "Manufacturing Site",
    "First Line Designation",
]

NUMERIC_BASE_COLUMNS = [
    "Line Item Quantity",
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Unit of Measure (Per Pack)",
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)",
]


class EngineeredPreprocessor:
    """
    Production-grade, leakage-safe preprocessor for pharmaceutical logistics.
    """

    def __init__(self, categorical_mode: str = "native", smoothing_m: float = 10.0):
        self.categorical_mode = categorical_mode  # "native" or "onehot"
        self.smoothing_m = smoothing_m

        self.hist_engine = PointInTimeHistoricalStatsEngine(smoothing_m=smoothing_m)
        self.scaler = RobustScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.fitted_ = False
        self.numeric_medians_: Dict[str, float] = {}
        self.label_encoders_: Dict[str, Dict[str, int]] = {}
        self.feature_names_: List[str] = []
        self.cat_dims_: Dict[str, int] = {}
        self.numeric_feature_names_: List[str] = []
        self.cat_feature_names_: List[str] = []

    def _extract_lead_times_and_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts pre-outcome logistical lead times, domain ratios, and calendar signals."""
        out = pd.DataFrame(index=df.index)

        sched_dt = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed")
        po_dt = pd.to_datetime(df.get("PO Sent to Vendor Date"), errors="coerce", format="mixed")
        pq_dt = pd.to_datetime(df.get("PQ First Sent to Client Date"), errors="coerce", format="mixed")

        # 1. Lead Times (days)
        po_to_sched = (sched_dt - po_dt).dt.total_seconds() / 86400.0
        out["PO_to_Scheduled_Days"] = po_to_sched
        out["PO_Date_Missing"] = po_dt.isna().astype(np.float32)

        pq_to_po = (po_dt - pq_dt).dt.total_seconds() / 86400.0
        out["PQ_to_PO_Days"] = pq_to_po
        out["PQ_to_Scheduled_Days"] = (sched_dt - pq_dt).dt.total_seconds() / 86400.0
        out["PQ_Date_Missing"] = pq_dt.isna().astype(np.float32)

        # Check for "Pre-PQ Process" indicator
        if "PQ First Sent to Client Date" in df.columns:
            out["Is_Pre_PQ_Process"] = (
                df["PQ First Sent to Client Date"].astype(str).str.contains("Pre-PQ", case=False, na=False)
            ).astype(np.float32)
        else:
            out["Is_Pre_PQ_Process"] = 0.0

        # 2. Base Numerics & Cleaning
        qty = pd.to_numeric(df.get("Line Item Quantity"), errors="coerce").fillna(0.0)
        val = pd.to_numeric(df.get("Line Item Value"), errors="coerce").fillna(0.0)
        pack_price = pd.to_numeric(df.get("Pack Price"), errors="coerce").fillna(0.0)
        unit_price = pd.to_numeric(df.get("Unit Price"), errors="coerce").fillna(0.0)
        uom = pd.to_numeric(df.get("Unit of Measure (Per Pack)"), errors="coerce").fillna(1.0)
        weight = pd.to_numeric(df.get("Weight (Kilograms)"), errors="coerce")
        freight = pd.to_numeric(df.get("Freight Cost (USD)"), errors="coerce")
        ins = pd.to_numeric(df.get("Line Item Insurance (USD)"), errors="coerce").fillna(0.0)

        out["Line_Item_Quantity"] = qty
        out["Line_Item_Value"] = val
        out["Pack_Price"] = pack_price
        out["Unit_Price"] = unit_price
        out["Unit_of_Measure"] = uom
        out["Weight_Kilograms"] = weight
        out["Freight_Cost_USD"] = freight
        out["Line_Item_Insurance_USD"] = ins

        # 3. Financial & Physical Ratios
        out["Value_Per_Unit"] = val / np.maximum(qty, 1.0)
        out["Weight_Per_Unit"] = weight / np.maximum(qty, 1.0)
        out["Freight_Per_Kg"] = freight / np.maximum(weight, 1.0)
        out["Freight_To_Value_Ratio"] = freight / np.maximum(val, 1.0)
        out["Insurance_To_Value_Ratio"] = ins / np.maximum(val, 1.0)
        out["Pack_Price_Ratio"] = pack_price / np.maximum(unit_price * uom, 0.001)

        # 4. Cyclical Calendar Signals (EXCLUDING Scheduled_Year)
        month = sched_dt.dt.month.fillna(6).astype(int)
        dow = sched_dt.dt.dayofweek.fillna(2).astype(int)
        day = sched_dt.dt.day.fillna(15).astype(int)
        quarter = sched_dt.dt.quarter.fillna(2).astype(int)

        out["Day_of_Week"] = dow.astype(np.float32)
        out["Day_of_Month"] = day.astype(np.float32)
        out["Quarter"] = quarter.astype(np.float32)
        out["Month_Sin"] = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
        out["Month_Cos"] = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)

        return out

    def fit(self, train_df: pd.DataFrame) -> "EngineeredPreprocessor":
        """Fits imputer medians, scalers, and categorical vocabularies on training data only."""
        # 1. Fit historical stats engine
        self.hist_engine.fit(train_df)

        # 2. Extract engineered features
        lead_ratios = self._extract_lead_times_and_ratios(train_df)
        hist_feats = self.hist_engine.transform(train_df)

        numeric_df = pd.concat([lead_ratios, hist_feats], axis=1)
        self.numeric_feature_names_ = list(numeric_df.columns)

        # Compute train-only medians
        self.numeric_medians_ = {}
        for col in self.numeric_feature_names_:
            med = float(numeric_df[col].median(skipna=True))
            if np.isnan(med):
                med = 0.0
            self.numeric_medians_[col] = med

        # Impute training numerics and fit scaler
        imputed_numeric = numeric_df.fillna(self.numeric_medians_)
        self.scaler.fit(imputed_numeric)

        # 3. Fit Categorical Vocabularies
        self.cat_feature_names_ = [c for c in CATEGORICAL_COLUMNS if c in train_df.columns]
        self.label_encoders_ = {}
        self.cat_dims_ = {}

        for col in self.cat_feature_names_:
            vals = train_df[col].fillna("Unknown").astype(str).unique()
            vocab = {val: i + 1 for i, val in enumerate(sorted(vals))}
            vocab["__UNK__"] = 0
            self.label_encoders_[col] = vocab
            self.cat_dims_[col] = len(vocab)

        if self.categorical_mode == "onehot":
            clean_cats = train_df[self.cat_feature_names_].fillna("Unknown").astype(str)
            self.ohe.fit(clean_cats)
            ohe_cols = list(self.ohe.get_feature_names_out(self.cat_feature_names_))
            self.feature_names_ = self.numeric_feature_names_ + ohe_cols
        else:
            self.feature_names_ = self.numeric_feature_names_ + [c.replace(" ", "_") for c in self.cat_feature_names_]

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input dataframe into model feature matrix."""
        if not self.fitted_:
            raise RuntimeError("EngineeredPreprocessor must be fit before transform!")

        # 1. Extract numerics & historical stats
        lead_ratios = self._extract_lead_times_and_ratios(df)
        hist_feats = self.hist_engine.transform(df)

        numeric_df = pd.concat([lead_ratios, hist_feats], axis=1)

        # Impute using train-only medians
        for col in self.numeric_feature_names_:
            if col not in numeric_df.columns:
                numeric_df[col] = self.numeric_medians_[col]
            else:
                numeric_df[col] = numeric_df[col].fillna(self.numeric_medians_[col])

        numeric_imputed = numeric_df[self.numeric_feature_names_]
        scaled_numeric = self.scaler.transform(numeric_imputed)

        # 2. Transform Categoricals
        if self.categorical_mode == "onehot":
            clean_cats = df[self.cat_feature_names_].fillna("Unknown").astype(str)
            cat_matrix = self.ohe.transform(clean_cats)
            return np.hstack([scaled_numeric, cat_matrix]).astype(np.float32)
        else:
            # Native integer encoding
            N = len(df)
            cat_cols = []
            for col in self.cat_feature_names_:
                vocab = self.label_encoders_[col]
                col_vals = df[col].fillna("Unknown").astype(str).values if col in df.columns else ["Unknown"] * N
                encoded = np.array([vocab.get(v, vocab["__UNK__"]) for v in col_vals], dtype=np.float32).reshape(-1, 1)
                cat_cols.append(encoded)

            cat_matrix = np.hstack(cat_cols)
            return np.hstack([scaled_numeric, cat_matrix]).astype(np.float32)

    def get_categorical_indices(self) -> List[int]:
        """Returns feature column indices corresponding to categorical features."""
        num_len = len(self.numeric_feature_names_)
        return list(range(num_len, num_len + len(self.cat_feature_names_)))

    @property
    def feature_names(self) -> List[str]:
        return self.feature_names_

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EngineeredPreprocessor":
        with open(filepath, "rb") as f:
            return pickle.load(f)
