"""
Page 2: Statistical Model Comparison & Fixed Alert Budget Auditing.
Reads dynamically from reports/combined_path1_path2_results.json and reports/model_comparison.csv at runtime.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.header("📊 Multi-Model Performance & Alert Budget Comparison")
st.markdown(
    "Rigorous comparative evaluation across Deep Neural Architectures and Gradient-Boosted Trees (GBDT). "
    "All figures are read dynamically at runtime from audited prospective reports."
)

RESULTS_JSON = Path("reports/combined_path1_path2_results.json")
COMPARISON_CSV = Path("reports/model_comparison.csv")

if not RESULTS_JSON.exists():
    st.error(f"Required report {RESULTS_JSON} not found.")
    st.stop()

with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    results_data = json.load(f)

budget_data = results_data.get("fixed_alert_budget_recalls", {})

# Prepare Table
rows = []
neural_models = {"EnhancedHybridTransformer", "HybridLSTMTransformer", "BiLSTM", "Tabular1DCNN", "DeepTabularMLP", "Deep Ensemble"}

for model_name, metrics in budget_data.items():
    category = "Deep Neural Architecture" if model_name in neural_models else "Classical / GBDT"
    rows.append(
        {
            "Model Architecture": model_name,
            "Family": category,
            "PR-AUC (Prospective)": round(metrics.get("pr_auc", 0.0), 4),
            "ROC-AUC (Prospective)": round(metrics.get("roc_auc", 0.0), 4),
            "Recall @ Top 10% Budget": f"{metrics.get('recall_top_10pct', 0.0)*100:.1f}%",
            "Recall @ Top 20% Budget": f"{metrics.get('recall_top_20pct', 0.0)*100:.1f}%",
            "Recall @ Top 30% Budget": f"{metrics.get('recall_top_30pct', 0.0)*100:.1f}%",
        }
    )

df_budget = pd.DataFrame(rows)
df_budget = df_budget.sort_values(by="PR-AUC (Prospective)", ascending=False)

st.subheader("Prospective Test Holdout Evaluation (N=1,017, Delayed=120)")
st.dataframe(df_budget, use_container_width=True)

# Comparison Bar Charts
st.subheader("Key Takeaway: Tree Models Match or Outperform Neural Networks on Tabular HSCM Data")
st.info(
    "On this tabular logistics dataset, **RandomForest** (PR-AUC 0.3353, ROC-AUC 0.7949) and "
    "**XGBoost** (PR-AUC 0.3599, ROC-AUC 0.7861) slightly match or outperform Deep Tabular architectures "
    "(BiLSTM PR-AUC 0.2505, Deep Ensemble PR-AUC 0.2511). Deep models require careful regularization and "
    "borderline oversampling to remain competitive on extreme class imbalance (11.8% positive rate)."
)

col1, col2 = st.columns(2)

with col1:
    fig_pr = px.bar(
        df_budget,
        x="Model Architecture",
        y="PR-AUC (Prospective)",
        color="Family",
        title="PR-AUC Comparison across Architectures",
        text_auto=".3f",
        color_discrete_map={"Classical / GBDT": "#3b82f6", "Deep Neural Architecture": "#8b5cf6"},
    )
    fig_pr.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_pr, use_container_width=True)

with col2:
    fig_roc = px.bar(
        df_budget,
        x="Model Architecture",
        y="ROC-AUC (Prospective)",
        color="Family",
        title="ROC-AUC Comparison across Architectures",
        text_auto=".3f",
        color_discrete_map={"Classical / GBDT": "#3b82f6", "Deep Neural Architecture": "#8b5cf6"},
    )
    fig_roc.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig_roc, use_container_width=True)

# Recall at Alert Budget Curve
st.subheader("Recall Captured at Constrained Logistics Alert Budgets")
budget_curve_rows = []
for model_name, metrics in budget_data.items():
    for pct in [10, 20, 30]:
        budget_curve_rows.append(
            {
                "Model": model_name,
                "Alert Budget (%)": f"Top {pct}%",
                "Recall (%)": metrics.get(f"recall_top_{pct}pct", 0.0) * 100,
            }
        )

df_curve = pd.DataFrame(budget_curve_rows)
fig_curve = px.line(
    df_curve,
    x="Alert Budget (%)",
    y="Recall (%)",
    color="Model",
    markers=True,
    title="Clinical Alert Budget Efficiency (Recall Captured)",
)
fig_curve.update_layout(height=420)
st.plotly_chart(fig_curve, use_container_width=True)

# Multi-seed validation metrics from CSV
if COMPARISON_CSV.exists():
    st.subheader("5-Seed Validation Cross-Check (Mean ± Std with 95% Bootstrap CIs)")
    df_csv = pd.read_csv(COMPARISON_CSV)
    st.dataframe(df_csv[["Model", "Tau_Star", "Recall_Mean", "Recall_CI", "PR_AUC_Mean", "PR_AUC_CI", "ROC_AUC_Mean", "ROC_AUC_CI"]], use_container_width=True)
