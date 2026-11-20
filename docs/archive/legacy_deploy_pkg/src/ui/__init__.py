"""
Production UI Design System Package for Streamlit HSCM Platform.
Exports accessible primitives, domain scorecards, telemetry bars, and views.
"""

from src.ui.theme import inject_custom_theme
from src.ui.state import SessionStateManager
from src.ui.components.primitives import (
    render_header,
    render_metric_card,
    render_badge,
    render_skeleton,
    render_empty_state,
    render_alert,
)
from src.ui.components.telemetry import (
    render_route_corridor_banner,
    render_qos_telemetry_bar,
)
from src.ui.components.scorecards import (
    render_lead_time_tracker,
    render_prediction_scorecard,
)
from src.ui.views.prediction_view import render_prediction_tab
from src.ui.views.batch_view import render_batch_tab
from src.ui.views.benchmarks_view import render_benchmarks_tab
from src.ui.views.architecture_view import render_architecture_tab

__all__ = [
    "inject_custom_theme",
    "SessionStateManager",
    "render_header",
    "render_metric_card",
    "render_badge",
    "render_skeleton",
    "render_empty_state",
    "render_alert",
    "render_route_corridor_banner",
    "render_qos_telemetry_bar",
    "render_lead_time_tracker",
    "render_prediction_scorecard",
    "render_prediction_tab",
    "render_batch_tab",
    "render_benchmarks_tab",
    "render_architecture_tab",
]
