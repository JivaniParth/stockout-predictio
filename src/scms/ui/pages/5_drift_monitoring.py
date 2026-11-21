"""
Page 5: Quarterly Distributional Drift Monitoring & Production Database Telemetry.
Reads drift metrics from reports/drift/ and queries live relational audit records from results/hscm_production.db.
"""

from pathlib import Path
import sqlite3
import pandas as pd
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None
import streamlit as st

st.header("📡 Distributional Drift Monitoring & Operational Telemetry")
st.markdown(
    "Prospective monitoring of prediction score stability, concept drift (PSI / KS tests), "
    "and real-time SQLite audit trail persistence."
)

DRIFT_CSV = Path("reports/drift/quarterly_drift_simulation.csv")
DB_PATH = Path("results/hscm_production.db")

# 1. Quarterly Drift Tracking
if DRIFT_CSV.exists():
    st.subheader("Quarterly Prospective Score Drift Simulation (2015+ Windows)")
    df_drift = pd.read_csv(DRIFT_CSV)
    st.dataframe(df_drift, use_container_width=True)

    c1, c2 = st.columns(2)
    if px is not None:
        with c1:
            fig_psi = px.bar(
                df_drift,
                x="Quarter",
                y="Score_PSI",
                color="Drift_Detected",
                title="Population Stability Index (PSI) per Quarter",
                color_discrete_map={True: "#ef4444", False: "#22c55e"},
            )
            fig_psi.add_hline(y=0.25, line_dash="dash", line_color="orange", annotation_text="Significant Shift Threshold (0.25)")
            st.plotly_chart(fig_psi, use_container_width=True)

        with c2:
            fig_ks = px.line(
                df_drift,
                x="Quarter",
                y="KS_Statistic",
                markers=True,
                title="Kolmogorov-Smirnov (KS) Statistic over Time",
            )
            st.plotly_chart(fig_ks, use_container_width=True)

st.markdown("---")
# 2. Live Relational Audit Trail
st.subheader("Live Relational Audit Trail (results/hscm_production.db)")

if DB_PATH.exists():
    try:
        conn = sqlite3.connect(DB_PATH)
        df_audits = pd.read_sql_query(
            """
            SELECT id, timestamp, risk_probability, risk_tier, country_tier, binary_alert, qos_latency_ms
            FROM prediction_audits
            ORDER BY timestamp DESC
            LIMIT 100
            """,
            conn,
        )
        conn.close()

        m1, m2, m3 = st.columns(3)
        m1.metric("Audited Consignments in DB", f"{len(df_audits):,}")
        lat_p50 = float(df_audits["qos_latency_ms"].median()) if not df_audits.empty else 0.0
        lat_p95 = float(df_audits["qos_latency_ms"].quantile(0.95)) if not df_audits.empty else 0.0
        m2.metric("Median Inference Latency (P50)", f"{lat_p50:.2f} ms")
        m3.metric("Tail Latency (P95 SLA)", f"{lat_p95:.2f} ms")

        st.dataframe(df_audits.head(20), use_container_width=True)

    except Exception as e:
        st.error(f"Database query error: {e}")
else:
    st.info("Relational database results/hscm_production.db will be populated upon live microservice inference.")
