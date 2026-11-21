"""
Page 3: Three-Way Decision (3WD) Policy & Interactive Threshold Frontier.
Provides dual interactive threshold sliders for tau_safe and tau_alert with live recalculation
and statistical reliability guards (flagging N < 50 regions as unreliable).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None
import streamlit as st

st.header("⚖️ Three-Way Decision (3WD) Policy & Operational Thresholds")
st.markdown(
    "Clinical and operational partition into **Positive Alert** (Expedited Stockout Intervention), "
    "**Negative Safe** (Standard Surface Transit), and **Boundary Region** (Active Telemetry Monitoring)."
)

PROBS_PATH = Path("artifacts/baseline_test_probs.npy")
LABELS_PATH = Path("artifacts/test_ground_truth.npy")
RESULTS_JSON = Path("reports/combined_path1_path2_results.json")

# 1. Audited Policy Reference Cards
if RESULTS_JSON.exists():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    pol_a = res_data.get("policy_a_capacity_constrained", {})
    act_a = pol_a.get("prospective_results", {})
    pol_b = res_data.get("policy_b_target_constrained", {})
    act_b = pol_b.get("prospective_results", {})

    st.subheader("Audited Pre-Configured Operating Policies (Test Holdout N=1,017, Delayed=120)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Policy A: Capacity-Constrained")
        st.caption("Rule: Maximize delay recall under a maximum 30% logistics alert budget.")
        st.markdown(
            f"- **Thresholds:** $\\tau_{{\\text{{safe}}}}={pol_a.get('tau_safe', 0.115)}$, $\\tau_{{\\text{{alert}}}}={pol_a.get('tau_alert', 0.40)}$\n"
            f"- **Alert Region:** $N={act_a.get('n_alert')}$ ({act_a.get('pct_alert', 0)*100:.1f}%), Recall = **{act_a.get('rec', 0)*100:.1f}%**, Precision = {act_a.get('prec', 0)*100:.1f}%\n"
            f"- **Safe Region:** $N={act_a.get('n_safe')}$ ({act_a.get('pct_safe', 0)*100:.1f}%), NPV = **{act_a.get('npv', 0)*100:.1f}%**\n"
            f"- **Boundary Region:** $N={act_a.get('n_bnd')}$ ({act_a.get('pct_bnd', 0)*100:.1f}%)\n"
            f"- **Lift over Baseline:** **{act_a.get('lift', 0):.2f}x**"
        )

    with c2:
        st.markdown("#### Policy B: Target-Constrained")
        st.caption("Rule: Thresholds chosen on 2014 validation: Validation Recall ≥ 85%, NPV ≥ 95%.")
        bnd_b = act_b.get("n_bnd", 0)
        warn_b = " ⚠️ *(N < 50: Small sample warning)*" if bnd_b < 50 else ""
        st.markdown(
            f"- **Thresholds:** $\\tau_{{\\text{{safe}}}}={pol_b.get('tau_safe', 0.30)}$, $\\tau_{{\\text{{alert}}}}={pol_b.get('tau_alert', 0.34)}$\n"
            f"- **Alert Region:** $N={act_b.get('n_alert')}$ ({act_b.get('pct_alert', 0)*100:.1f}%), Recall = **{act_b.get('rec', 0)*100:.1f}%**, Precision = {act_b.get('prec', 0)*100:.1f}%\n"
            f"- **Safe Region:** $N={act_b.get('n_safe')}$ ({act_b.get('pct_safe', 0)*100:.1f}%), NPV = **{act_b.get('npv', 0)*100:.1f}%**\n"
            f"- **Boundary Region:** $N={bnd_b}$ ({act_b.get('pct_bnd', 0)*100:.1f}%){warn_b}\n"
            f"- **Lift over Baseline:** **{act_b.get('lift', 0):.2f}x**"
        )

st.markdown("---")
st.subheader("Interactive Threshold Frontier Simulator")

if not PROBS_PATH.exists() or not LABELS_PATH.exists():
    st.warning("Holdout prediction arrays not found. Please run 'python main.py decide' to generate baseline arrays.")
    st.stop()

y_prob = np.load(PROBS_PATH)
y_true = np.load(LABELS_PATH)
n_total = len(y_true)
prevalence = float(np.mean(y_true))

col_s1, col_s2 = st.columns(2)
with col_s1:
    tau_safe_val = st.slider(
        "Safe Region Upper Bound (τ_safe):",
        min_value=0.01,
        max_value=0.50,
        value=0.115,
        step=0.005,
        help="Shipments with risk <= τ_safe are assigned to Standard Surface Transit (Path 2 Confidence).",
    )
with col_s2:
    tau_alert_val = st.slider(
        "Alert Region Lower Bound (τ_alert):",
        min_value=0.10,
        max_value=0.80,
        value=0.400,
        step=0.005,
        help="Shipments with risk >= τ_alert are assigned to Immediate Stockout Intervention (Path 1 Early Warning).",
    )

if tau_safe_val >= tau_alert_val:
    st.error("Invalid configuration: τ_safe must be strictly less than τ_alert (τ_safe < τ_alert).")
    st.stop()

# Interactive Region Metrics Calculation
alert_mask = y_prob >= tau_alert_val
safe_mask = y_prob <= tau_safe_val
boundary_mask = (~alert_mask) & (~safe_mask)

n_alert = int(np.sum(alert_mask))
n_safe = int(np.sum(safe_mask))
n_boundary = int(np.sum(boundary_mask))

tp = int(np.sum((y_prob >= tau_alert_val) & (y_true == 1)))
fp = int(np.sum((y_prob >= tau_alert_val) & (y_true == 0)))
tn = int(np.sum((y_prob <= tau_safe_val) & (y_true == 0)))
fn = int(np.sum((y_prob <= tau_safe_val) & (y_true == 1)))

total_positives = int(np.sum(y_true))
recall = tp / total_positives if total_positives > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
lift = precision / prevalence if prevalence > 0 else 0.0

# Statistical Sample Size Reliability Check
unreliable_regions = []
if n_alert < 50:
    unreliable_regions.append(f"Alert Region (N={n_alert} < 50)")
if n_safe < 50:
    unreliable_regions.append(f"Safe Region (N={n_safe} < 50)")
if n_boundary < 50:
    unreliable_regions.append(f"Boundary Region (N={n_boundary} < 50)")

if unreliable_regions:
    st.markdown(
        f"<div class='warning-box'>⚠️ <strong>STATISTICAL SAMPLE SIZE WARNING:</strong> "
        f"The following region(s) contain fewer than 50 shipments: <strong>{', '.join(unreliable_regions)}</strong>. "
        f"Statistical metric estimates (precision, recall, NPV) in small samples have wide variance and are clinically unreliable.</div>",
        unsafe_allow_html=True,
    )

# Scorecard Display
m1, m2, m3, m4 = st.columns(4)
m1.metric("Alert Region Size", f"{n_alert} ({n_alert / n_total * 100:.1f}%)", delta=f"{tp} True Alerts")
m2.metric("Delay Recall Captured", f"{recall * 100:.1f}%", help="Percent of all 120 delayed consignments captured by Alert.")
m3.metric("Safe Region NPV", f"{npv * 100:.1f}%", help="Negative Predictive Value in Safe region (1 - Stockout Rate).")
m4.metric("Boundary Region Size", f"{n_boundary} ({n_boundary / n_total * 100:.1f}%)", help="Consignments under active telemetry tracking.")

# Region Distribution Donut Chart
if go is not None:
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Alert (Intervene)", "Boundary (Track)", "Safe (Surface)"],
                values=[n_alert, n_boundary, n_safe],
                hole=0.45,
                marker_colors=["#ef4444", "#f59e0b", "#22c55e"],
            )
        ]
    )
    fig_donut.update_layout(title="Prospective Consignment Partition Distribution", height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_donut, use_container_width=True)
else:
    st.write(f"**Region Split:** Alert: {n_alert} ({n_alert / n_total * 100:.1f}%), Boundary: {n_boundary} ({n_boundary / n_total * 100:.1f}%), Safe: {n_safe} ({n_safe / n_total * 100:.1f}%)")
