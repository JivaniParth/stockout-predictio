"""
Scalable Multipage Streamlit Application Entry Point.
Uses st.navigation for modern multi-view clinical decision support.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scms.ui.theme import inject_custom_theme

st.set_page_config(
    page_title="HSCM Clinical Decision Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_theme()

# Multipage Navigation Setup via st.navigation
pages_dir = Path(__file__).parent / "pages"

pages = [
    st.Page(pages_dir / "1_triage.py", title="Consignment Triage", icon="🎯", default=True),
    st.Page(pages_dir / "2_model_comparison.py", title="Model Comparison", icon="📊"),
    st.Page(pages_dir / "3_decision_policy.py", title="Decision Policy (3WD)", icon="⚖️"),
    st.Page(pages_dir / "4_fairness.py", title="Health Equity & Fairness", icon="🌐"),
    st.Page(pages_dir / "5_drift_monitoring.py", title="Drift & Monitoring", icon="📡"),
    st.Page(pages_dir / "6_about_limitations.py", title="About & Academic Disclosures", icon="ℹ️"),
]

pg = st.navigation(pages)
pg.run()
