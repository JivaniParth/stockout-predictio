"""
Point-in-Time Historical Statistics and Rolling Feature Engineering.

Computes historical performance statistics and rolling operational metrics
with STRICT temporal integrity:
- Every statistic for shipment i is computed strictly using prior completed shipments
  whose delivery outcome was physically known BEFORE row i's Scheduled Delivery Date
  (i.e., Delivered_Date_j < Scheduled_Date_i).
- Completely eliminates temporal future leakage and right-censoring bias.
- Implements empirical Bayes / m-estimate smoothing toward the global training prior.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd


CATEGORICAL_TARGET_GROUPS = [
    "Country",
    "Vendor",
    "Manufacturing Site",
    "Product Group",
    "Shipment Mode",
    "Vendor INCO Term",
]


class PointInTimeHistoricalStatsEngine:
    """
    Leakage-safe historical feature generator.
    Indexes historical completed deliveries from training partition only,
    and performs fast, exact point-in-time lookups using binary search.
    """

    def __init__(self, smoothing_m: float = 10.0, rolling_windows_days: List[int] = [90, 180, 365]):
        self.smoothing_m = smoothing_m
        self.rolling_windows_days = rolling_windows_days
        self.group_cols = CATEGORICAL_TARGET_GROUPS

        self.global_prior_delay_rate = 0.11
        self.global_prior_lateness_days = 2.0
        self.fitted_ = False
        self.history_index: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.global_deliv_ts: np.ndarray = np.array([], dtype="datetime64[ns]")
        self.global_is_late: np.ndarray = np.array([], dtype=int)
        self.global_lateness_days: np.ndarray = np.array([], dtype=float)

    def fit(self, train_df: pd.DataFrame) -> "PointInTimeHistoricalStatsEngine":
        """
        Builds chronological lookup indices from training shipments with valid delivery dates.
        """
        df = train_df.copy()
        sched_dt = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed")
        deliv_dt = pd.to_datetime(df["Delivered to Client Date"], errors="coerce", format="mixed")

        # Keep rows where both scheduled and delivered dates are valid
        valid_mask = sched_dt.notna() & deliv_dt.notna()
        completed = df[valid_mask].copy()

        completed["_sched_ts"] = sched_dt[valid_mask]
        completed["_deliv_ts"] = deliv_dt[valid_mask]
        completed["_is_late"] = (completed["_deliv_ts"] > completed["_sched_ts"]).astype(int)
        completed["_lateness_days"] = np.maximum(
            0, (completed["_deliv_ts"] - completed["_sched_ts"]).dt.total_seconds() / 86400.0
        )
        completed["_line_val"] = pd.to_numeric(completed.get("Line Item Value", 0.0), errors="coerce").fillna(0.0)

        # Sort strictly by delivery timestamp (when the outcome physically became known)
        completed = completed.sort_values("_deliv_ts").reset_index(drop=True)

        self.global_prior_delay_rate = float(completed["_is_late"].mean()) if len(completed) > 0 else 0.11
        self.global_prior_lateness_days = float(completed["_lateness_days"].mean()) if len(completed) > 0 else 2.0

        self.global_deliv_ts = completed["_deliv_ts"].values
        self.global_is_late = np.concatenate([[0], np.cumsum(completed["_is_late"].values)])
        self.global_lateness_days = np.concatenate([[0], np.cumsum(completed["_lateness_days"].values)])

        # Build group-level sorted arrays
        self.history_index = {}
        for col in self.group_cols:
            self.history_index[col] = {}
            if col not in completed.columns:
                continue

            grouped = completed.groupby(col, observed=True)
            for val, grp in grouped:
                grp_deliv = grp["_deliv_ts"].values
                grp_late = grp["_is_late"].values
                grp_days = grp["_lateness_days"].values
                grp_val = grp["_line_val"].values

                self.history_index[col][str(val)] = {
                    "deliv_ts": grp_deliv,
                    "cum_count": np.arange(len(grp_deliv) + 1),
                    "cum_late": np.concatenate([[0], np.cumsum(grp_late)]),
                    "cum_days": np.concatenate([[0], np.cumsum(grp_days)]),
                    "cum_val": np.concatenate([[0.0], np.cumsum(grp_val)]),
                }

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes point-in-time historical delay rates, lateness, and rolling volume
        strictly before each row's Scheduled Delivery Date.
        """
        if not self.fitted_:
            raise RuntimeError("PointInTimeHistoricalStatsEngine must be fit before transform!")

        sched_dates = pd.to_datetime(df["Scheduled Delivery Date"], errors="coerce", format="mixed")
        N = len(df)
        features: Dict[str, np.ndarray] = {}

        # Global point-in-time statistics
        global_idx = np.searchsorted(self.global_deliv_ts, sched_dates.values, side="left")
        global_completed = global_idx
        global_late = self.global_is_late[global_idx]
        features["Global_Completed_Prior_Count"] = global_completed
        features["Global_Prior_Delay_Rate"] = np.where(
            global_completed > 0,
            (global_late + self.smoothing_m * self.global_prior_delay_rate) / (global_completed + self.smoothing_m),
            self.global_prior_delay_rate,
        )

        # Group-level point-in-time statistics
        for col in self.group_cols:
            clean_col = col.replace(" ", "_")
            if col in df.columns:
                col_vals = np.array([str(x) if pd.notna(x) else "Unknown" for x in df[col]], dtype=str)
            else:
                col_vals = np.array(["Unknown"] * N, dtype=str)

            rate_arr = np.full(N, self.global_prior_delay_rate, dtype=np.float32)
            lateness_arr = np.full(N, self.global_prior_lateness_days, dtype=np.float32)
            count_arr = np.zeros(N, dtype=np.float32)

            # Rolling window arrays
            rolling_counts = {w: np.zeros(N, dtype=np.float32) for w in self.rolling_windows_days}
            rolling_lates = {w: np.zeros(N, dtype=np.float32) for w in self.rolling_windows_days}

            grp_dict = self.history_index.get(col, {})
            unique_vals = np.unique(col_vals)

            for val in unique_vals:
                mask = col_vals == val
                if val not in grp_dict:
                    continue

                grp_info = grp_dict[val]
                deliv_ts = grp_info["deliv_ts"]
                if len(deliv_ts) == 0:
                    continue

                q_dates = sched_dates.values[mask]
                idx = np.searchsorted(deliv_ts, q_dates, side="left")

                n_comp = grp_info["cum_count"][idx]
                n_late = grp_info["cum_late"][idx]
                s_days = grp_info["cum_days"][idx]

                smoothed_rate = (n_late + self.smoothing_m * self.global_prior_delay_rate) / (
                    n_comp + self.smoothing_m
                )
                smoothed_days = (s_days + self.smoothing_m * self.global_prior_lateness_days) / (
                    n_comp + self.smoothing_m
                )

                rate_arr[mask] = smoothed_rate
                lateness_arr[mask] = smoothed_days
                count_arr[mask] = n_comp

                # Compute rolling windows: completed in [q_dates - w days, q_dates)
                for w in self.rolling_windows_days:
                    dt_delta = np.timedelta64(w, "D")
                    start_dates = q_dates - dt_delta
                    start_idx = np.searchsorted(deliv_ts, start_dates, side="left")
                    w_comp = n_comp - grp_info["cum_count"][start_idx]
                    w_late = n_late - grp_info["cum_late"][start_idx]
                    rolling_counts[w][mask] = w_comp
                    rolling_lates[w][mask] = w_late

            features[f"{clean_col}_Hist_Delay_Rate"] = rate_arr
            features[f"{clean_col}_Hist_Lateness_Days"] = lateness_arr
            features[f"{clean_col}_Completed_Prior_Count"] = count_arr

            # Add rolling features for Vendor and Country
            if col in ["Vendor", "Country"]:
                for w in self.rolling_windows_days:
                    features[f"{clean_col}_Rolling_{w}d_Volume"] = rolling_counts[w]
                    features[f"{clean_col}_Rolling_{w}d_Delays"] = rolling_lates[w]

        return pd.DataFrame(features, index=df.index)
