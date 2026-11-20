"""
State Management Layer for Production Streamlit Application.
Provides centralized, type-safe Session State initializers, mutators, and accessors
to eliminate state bugs, key collisions, and race conditions.
"""

from typing import Dict, Any, Optional, List
from datetime import date
import streamlit as st


class SessionStateManager:
    """Centralized typed manager for all Streamlit session state keys."""

    # Key constants
    KEY_PREDICTION_HISTORY = "hscm_prediction_history"
    KEY_BATCH_DATA = "hscm_batch_data"
    KEY_ACTIVE_PRESET = "hscm_active_preset"
    KEY_TELEMETRY_LOGS = "hscm_telemetry_logs"
    KEY_CUSTOM_THRESHOLD = "hscm_custom_threshold"
    KEY_LAST_PREDICTION = "hscm_last_prediction"

    @classmethod
    def initialize(cls):
        """Initializes all session state keys with immutable defaults."""
        defaults = {
            cls.KEY_PREDICTION_HISTORY: [],
            cls.KEY_BATCH_DATA: None,
            cls.KEY_ACTIVE_PRESET: "None",
            cls.KEY_TELEMETRY_LOGS: [],
            cls.KEY_CUSTOM_THRESHOLD: 0.5,
            cls.KEY_LAST_PREDICTION: None,
        }
        for key, default_val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_val

    @classmethod
    def get_prediction_history(cls) -> List[Dict[str, Any]]:
        return st.session_state.get(cls.KEY_PREDICTION_HISTORY, [])

    @classmethod
    def record_prediction(cls, prediction_payload: Dict[str, Any]):
        """Records a successful prediction to session telemetry history (FIFO cap 50)."""
        history = st.session_state.get(cls.KEY_PREDICTION_HISTORY, [])
        history.insert(0, prediction_payload)
        st.session_state[cls.KEY_PREDICTION_HISTORY] = history[:50]
        st.session_state[cls.KEY_LAST_PREDICTION] = prediction_payload

    @classmethod
    def get_last_prediction(cls) -> Optional[Dict[str, Any]]:
        return st.session_state.get(cls.KEY_LAST_PREDICTION, None)

    @classmethod
    def set_batch_data(cls, df: Any):
        st.session_state[cls.KEY_BATCH_DATA] = df

    @classmethod
    def get_batch_data(cls) -> Optional[Any]:
        return st.session_state.get(cls.KEY_BATCH_DATA, None)

    @classmethod
    def clear_batch_data(cls):
        st.session_state[cls.KEY_BATCH_DATA] = None

    @classmethod
    def record_telemetry(cls, latency_ms: float, model_name: str, endpoint: str):
        logs = st.session_state.get(cls.KEY_TELEMETRY_LOGS, [])
        logs.append({
            "latency_ms": latency_ms,
            "model_name": model_name,
            "endpoint": endpoint,
            "timestamp": date.today().isoformat(),
        })
        st.session_state[cls.KEY_TELEMETRY_LOGS] = logs[:100]
