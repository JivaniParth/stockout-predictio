"""
Publication-Quality Visualization Suite for HSCM Deep Learning & Cloud QoS Research.
Generates vector and high-resolution raster plots for paper inclusion.
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.config import FIGURES_DIR

# Set scientific styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def plot_roc_pr_comparison(
    model_predictions: Dict[str, Dict[str, np.ndarray]],
    y_test: np.ndarray,
    save_path: Path = None,
):
    """
    Plots side-by-side ROC and Precision-Recall curves on prospective test set.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "roc_pr_comparison.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Palette
    colors = sns.color_palette("tab10", len(model_predictions))

    # 1. ROC Curves
    for (name, data), color in zip(model_predictions.items(), colors):
        probs = data["probs"]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_val:.3f})", lw=2, color=color)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Guess (0.500)")
    axes[0].set_title("Prospective ROC Curves (2015 Test Split)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])

    # 2. Precision-Recall Curves
    baseline_rate = y_test.mean()
    for (name, data), color in zip(model_predictions.items(), colors):
        probs = data["probs"]
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_val = auc(recall, precision)
        axes[1].plot(recall, precision, label=f"{name} (PR-AUC = {pr_val:.3f})", lw=2, color=color)

    axes[1].axhline(y=baseline_rate, color="k", linestyle="--", lw=1.5, alpha=0.7, label=f"No-Skill Rate ({baseline_rate:.3f})")
    axes[1].set_title("Prospective Precision-Recall Curves (Imbalanced Target)")
    axes[1].set_xlabel("Recall (Stockout Risk Coverage)")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved ROC/PR comparison plot to {save_path}")


def plot_qos_tradeoff(
    qos_data: List[Dict[str, Any]],
    ml_data: Dict[str, Dict[str, Any]],
    save_path: Path = None,
):
    """
    Plots Cloud QoS Latency (P50 ms) vs Predictive F1-Score (Pareto Frontier).
    Directly answers RQ3: Trading inference latency against predictive fidelity.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "qos_latency_vs_f1_tradeoff.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    models = []
    latencies = []
    f1_scores = []
    recalls = []
    sizes = []

    for item in qos_data:
        m = item["Model"]
        if m in ml_data:
            models.append(m)
            lat = item["Latency_P50_ms"]
            f1 = ml_data[m]["F1_Late"]
            rec = ml_data[m]["Recall_Late"]
            param = item["Total_Parameters"]

            latencies.append(lat)
            f1_scores.append(f1)
            recalls.append(rec)
            # bubble size based on model parameter count (or fixed baseline)
            sizes.append(max(80, np.sqrt(param) * 1.5 if param > 0 else 100))

    scatter = ax.scatter(
        latencies,
        f1_scores,
        s=sizes,
        c=recalls,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=1.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Recall on Late Stockout Class", rotation=270, labelpad=15)

    for i, m in enumerate(models):
        ax.annotate(
            m,
            (latencies[i], f1_scores[i]),
            xytext=(7, 5),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Cloud QoS Pareto Frontier: Inference Latency vs. Predictive F1-Score")
    ax.set_xlabel("Median Inference Latency P50 (ms) [Log Scale preferred]")
    ax.set_ylabel("F1-Score on Late Delivery Class (2015 Prospective Test)")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Cloud QoS trade-off plot to {save_path}")


def plot_equity_tiers(equity_results: Dict[str, Dict[str, Any]], save_path: Path = None):
    """
    Plots stratified Recall across geographic health equity tiers.
    Directly answers RQ4: Disparity across low-risk hubs vs high-risk underserved tiers.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "health_equity_tier_disparities.png"

    models = list(equity_results.keys())
    tiers = ["Tier_1_Low_Risk_Hub", "Tier_2_Moderate_Risk", "Tier_3_High_Risk_Underserved"]
    labels = ["Tier 1: Central Hubs", "Tier 2: Regional Nodes", "Tier 3: Underserved/Remote"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for i, (t, lbl) in enumerate(zip(tiers, labels)):
        vals = [equity_results[m]["Tier_Metrics"].get(t, {}).get("Recall", 0.0) for m in models]
        ax.bar(x + (i - 1) * width, vals, width, label=lbl, alpha=0.9, edgecolor="black")

    ax.set_ylabel("Recall (Late Delivery / Stockout Sensitivity)")
    ax.set_title("Health Equity Analysis: Stockout Detection Sensitivity across Geographic Tiers")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Health Equity tier plot to {save_path}")
