"""
Benchmarks View (Tab 3) for Streamlit HSCM Platform.
Systematic Review Research Gaps, Quantitative Tables, and Publication Figures.
"""

from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

from src.ui.components.primitives import render_empty_state, render_alert


def render_benchmarks_tab(results_dir: Path, figures_dir: Path):
    """Renders the Systematic Literature Review Gaps & Empirical Benchmarks View."""
    st.subheader("Systematic Literature Review Gaps & Empirical Benchmarks")
    st.markdown(
        "Direct empirical evidence addressing the gaps formulated in the review paper draft. "
        "Evaluated strictly on the **Prospective 2015 Test Set** with out-of-sample temporal validity."
    )

    metrics_csv = results_dir / "benchmark_metrics.csv"
    qos_csv = results_dir / "qos_benchmark.csv"
    eq_csv = results_dir / "health_equity_benchmark.csv"

    # Edge Case: Files not generated yet
    if not metrics_csv.exists() or not qos_csv.exists():
        render_empty_state(
            title="Benchmark Datasets Not Found",
            description="The benchmark metrics have not been generated yet. Please run `python run_experiments.py` to train all models and profile QoS metrics.",
        )
        return

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("#### 1. Predictive Performance Comparison (Prospective 2015 Test)")
        df_m = pd.read_csv(metrics_csv, index_col=0)
        show_m_cols = ["Accuracy", "Precision_Late", "Recall_Late", "F1_Late", "ROC_AUC", "PR_AUC", "Brier_Score"]
        st.dataframe(
            df_m[[c for c in show_m_cols if c in df_m.columns]].style.highlight_max(
                subset=["Accuracy", "Recall_Late", "F1_Late", "ROC_AUC", "PR_AUC"], color="#D1FAE5"
            ),
            use_container_width=True,
        )
        st.caption("All models evaluated strictly on unseen 2015 test shipments without temporal leakage.")

    with g2:
        st.markdown("#### 2. Cloud QoS & Latency Profiling (500 Iterations)")
        df_q = pd.read_csv(qos_csv, index_col=0)
        param_col = "Total_Parameters" if "Total_Parameters" in df_q.columns else "Parameters"
        show_q_cols = [param_col, "Disk_Size_MB", "Latency_P50_ms", "Latency_P95_ms", "Throughput_BS64_samples_per_sec"]
        st.dataframe(
            df_q[[c for c in show_q_cols if c in df_q.columns]].style.highlight_min(
                subset=["Latency_P50_ms", "Latency_P95_ms", "Disk_Size_MB"], color="#D1FAE5"
            ),
            use_container_width=True,
        )
        st.caption("P50 and P95 latency percentiles measured across 500 single-inference trials on CPU.")

    st.markdown("---")
    st.markdown("#### 3. Health Equity & Geographic Tier Disparities (RQ4)")
    if eq_csv.exists():
        df_e = pd.read_csv(eq_csv, index_col=0)
        st.dataframe(df_e, use_container_width=True)
        st.caption("Demonstrating high stockout detection recall across high-vulnerability sub-Saharan African facilities.")

    st.markdown("---")
    st.markdown("#### 4. Interactive Benchmarks & Cloud QoS Pareto Frontier")
    chart_col1, chart_col2 = st.columns(2)

    df_m_reset = df_m.reset_index().rename(columns={"index": "Model"} if "index" in df_m.reset_index().columns else {})
    df_q_reset = df_q.reset_index().rename(columns={"index": "Model"} if "index" in df_q.reset_index().columns else {})
    if "Model" not in df_m_reset.columns:
        df_m_reset["Model"] = df_m.index
    if "Model" not in df_q_reset.columns:
        df_q_reset["Model"] = df_q.index

    with chart_col1:
        st.markdown("##### Latency vs. F1 Trade-off (Interactive Pareto Frontier)")
        try:
            merged = pd.merge(df_m_reset, df_q_reset, on="Model")
            scatter_chart = (
                alt.Chart(merged)
                .mark_circle(size=150)
                .encode(
                    x=alt.X("Latency_P50_ms:Q", title="P50 CPU Inference Latency (ms)", scale=alt.Scale(type="log")),
                    y=alt.Y("F1_Late:Q", title="Minority Stockout F1-Score", scale=alt.Scale(domain=[0.3, 0.45])),
                    color=alt.Color("Model:N", title="Architecture", scale=alt.Scale(scheme="category10")),
                    tooltip=[
                        alt.Tooltip("Model:N", title="Architecture"),
                        alt.Tooltip("Latency_P50_ms:Q", title="Latency P50 (ms)", format=".2f"),
                        alt.Tooltip("F1_Late:Q", title="F1 Late", format=".4f"),
                        alt.Tooltip("Recall_Late:Q", title="Recall Late", format=".2%"),
                        alt.Tooltip("Throughput_BS64_samples_per_sec:Q", title="Throughput (BS64)", format=",.0f"),
                        alt.Tooltip("Disk_Size_MB:Q", title="Disk Size (MB)", format=".2f"),
                    ],
                )
                .properties(height=340)
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)
            st.caption("Interactive Altair Pareto Frontier: Hover circles to inspect model QoS SLAs and predictive trade-offs.")
        except Exception:
            p_qos = figures_dir / "qos_latency_vs_f1_tradeoff.png"
            if p_qos.exists():
                st.image(str(p_qos), use_container_width=True)

    with chart_col2:
        st.markdown("##### Accuracy vs. Minority Recall Comparison")
        try:
            bar_data = df_m_reset[["Model", "Accuracy", "Recall_Late", "F1_Late"]].melt(
                id_vars=["Model"],
                value_vars=["Accuracy", "Recall_Late", "F1_Late"],
                var_name="Metric",
                value_name="Score",
            )
            bar_chart = (
                alt.Chart(bar_data)
                .mark_bar()
                .encode(
                    x=alt.X("Model:N", title="Architecture", axis=alt.Axis(labelAngle=-30)),
                    y=alt.Y("Score:Q", title="Metric Score", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("Metric:N", title="Metric"),
                    xOffset="Metric:N",
                    tooltip=[
                        alt.Tooltip("Model:N"),
                        alt.Tooltip("Metric:N"),
                        alt.Tooltip("Score:Q", format=".2%"),
                    ],
                )
                .properties(height=340)
                .interactive()
            )
            st.altair_chart(bar_chart, use_container_width=True)
            st.caption("Grouped multi-metric comparison demonstrating high minority stockout recall across models.")
        except Exception:
            p_roc = figures_dir / "roc_pr_comparison.png"
            if p_roc.exists():
                st.image(str(p_roc), use_container_width=True)

    with st.expander("View Publication-Ready Static Figures (IEEE / Elsevier Submissions)", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("##### Out-of-Sample ROC & PR Curves")
            p_roc = figures_dir / "roc_pr_comparison.png"
            if p_roc.exists():
                st.image(str(p_roc), use_container_width=True)
            else:
                render_alert("Figure not found. Run experiment script to regenerate.", variant="info")

        with f_col2:
            st.markdown("##### Cloud QoS Pareto Frontier (Static)")
            p_qos = figures_dir / "qos_latency_vs_f1_tradeoff.png"
            if p_qos.exists():
                st.image(str(p_qos), use_container_width=True)
            else:
                render_alert("Figure not found. Run experiment script to regenerate.", variant="info")

        with f_col3:
            st.markdown("##### Health Equity Disparities Across Tiers")
            p_eq = figures_dir / "health_equity_tier_disparities.png"
            if p_eq.exists():
                st.image(str(p_eq), use_container_width=True)
            else:
                render_alert("Figure not found. Run experiment script to regenerate.", variant="info")
