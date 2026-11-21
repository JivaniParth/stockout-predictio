"""
Page 4: Algorithmic Health Equity Governance & Data Provenance Audit.
Visualizes per-tier recall parity, mitigated equity gap, and F1 cost trade-offs.
"""

import yaml
from pathlib import Path
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None
import streamlit as st

st.header("🌐 Algorithmic Health Equity Governance & Fairness Auditing")
st.markdown(
    "Monitoring and mitigating disparate impact across country logistical vulnerability tiers. "
    "Ensures that fragile, landlocked health corridors (Tier 3) do not suffer degraded stockout detection."
)

TIERS_CONFIG_PATH = Path("config/tiers.yaml")

# Load Tier Definitions
if TIERS_CONFIG_PATH.exists():
    with open(TIERS_CONFIG_PATH, "r", encoding="utf-8") as f:
        tiers_cfg = yaml.safe_load(f)

    st.subheader("Vulnerability Tier Definitions (Derived from Pre-2014 Corridors)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Tier 1: Low-Risk Hubs**")
        st.caption("Mature port/highway infrastructure, lower delay volatility.")
        st.write(", ".join(tiers_cfg.get("tiers", {}).get("Tier_1_Low_Risk_Hub", {}).get("countries", [])))
    with col2:
        st.markdown("**Tier 2: Moderate-Risk Transit**")
        st.caption("Regional distribution facilities and intermediate logistics resilience.")
        st.write("Default baseline fallback across transit corridors.")
    with col3:
        st.markdown("**Tier 3: Fragile / Underserved**")
        st.caption("Protracted customs clearance, landlocked corridors, fragile supply networks.")
        st.write(", ".join(tiers_cfg.get("tiers", {}).get("Tier_3_High_Risk_Underserved", {}).get("countries", [])))

st.markdown("---")
st.subheader("Audited Per-Tier Performance & Validation Threshold Mitigation")
st.markdown(
    "Tier-specific threshold tuning balances clinical safety without compromising overall F1 utility."
)

fairness_records = [
    {"Model Architecture": "RandomForest", "Baseline Tier 3 Recall": 77.46, "Mitigated Tier 3 Recall": 83.10, "Baseline Recall Gap": 0.8913, "Mitigated Recall Gap": 0.9348, "F1 Cost": -0.0024},
    {"Model Architecture": "XGBoost", "Baseline Tier 3 Recall": 63.38, "Mitigated Tier 3 Recall": 78.87, "Baseline Recall Gap": 0.7174, "Mitigated Recall Gap": 0.7887, "F1 Cost": -0.0347},
    {"Model Architecture": "BiLSTM", "Baseline Tier 3 Recall": 71.83, "Mitigated Tier 3 Recall": 74.65, "Baseline Recall Gap": 0.8478, "Mitigated Recall Gap": 0.8478, "F1 Cost": -0.0052},
    {"Model Architecture": "Tabular1DCNN", "Baseline Tier 3 Recall": 77.46, "Mitigated Tier 3 Recall": 83.10, "Baseline Recall Gap": 0.5145, "Mitigated Recall Gap": 0.8478, "F1 Cost": +0.0017},
    {"Model Architecture": "HybridLSTMTransformer", "Baseline Tier 3 Recall": 77.46, "Mitigated Tier 3 Recall": 77.46, "Baseline Recall Gap": 0.8913, "Mitigated Recall Gap": 0.8913, "F1 Cost": +0.0008},
    {"Model Architecture": "DeepTabularMLP", "Baseline Tier 3 Recall": 67.61, "Mitigated Tier 3 Recall": 71.83, "Baseline Recall Gap": 0.9130, "Mitigated Recall Gap": 0.9130, "F1 Cost": -0.0027},
    {"Model Architecture": "LogisticRegression", "Baseline Tier 3 Recall": 67.61, "Mitigated Tier 3 Recall": 81.69, "Baseline Recall Gap": 0.8913, "Mitigated Recall Gap": 0.9348, "F1 Cost": -0.0111},
]

df_fairness = pd.DataFrame(fairness_records)
st.dataframe(df_fairness, use_container_width=True)

# Visualizing Tier 3 Recall Improvement
if go is not None:
    fig_tier3 = go.Figure()
    fig_tier3.add_trace(
        go.Bar(
            name="Baseline Tier 3 Recall (%)",
            x=df_fairness["Model Architecture"],
            y=df_fairness["Baseline Tier 3 Recall"],
            marker_color="#94a3b8",
        )
    )
    fig_tier3.add_trace(
        go.Bar(
            name="Mitigated Tier 3 Recall (%)",
            x=df_fairness["Model Architecture"],
            y=df_fairness["Mitigated Tier 3 Recall"],
            marker_color="#22c55e",
        )
    )
    fig_tier3.update_layout(
        barmode="group",
        title="Tier 3 (Fragile / Underserved Corridors) Recall Uplift after Mitigation",
        height=420,
        yaxis_title="Stockout Recall (%)",
    )
    st.plotly_chart(fig_tier3, use_container_width=True)
else:
    st.dataframe(df_fairness[["Model Architecture", "Baseline Tier 3 Recall", "Mitigated Tier 3 Recall", "Equity Disparity Gap (%)"]], use_container_width=True)

# Data Provenance Disclosure
st.markdown(
    "<div class='disclosure-box'><strong>Data Provenance Certification:</strong> "
    "All Tier categorizations are strictly derived from pre-2014 historical data (≤ 2013) and "
    "multilateral logistics corridor reports (UN/WHO/USAID). No post-2013 validation or test target "
    "outcomes influenced tier definitions.</div>",
    unsafe_allow_html=True,
)
