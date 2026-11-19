"""
Model Explainability and Attention Visualization Module.
Features:
- Global and local SHAP explanations for XGBoost and Neural Architectures
- Attention-weight heatmaps for Hybrid LSTM-Transformer and BiLSTM Attention Pooling
- Publication-ready figures saved to reports/figures/
"""

from pathlib import Path
from typing import List, Optional, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from src.models.hybrid_transformer import HybridLSTMTransformer
from src.models.lstm import BiLSTMStockoutPredictor


def explain_xgboost_shap(
    xgb_estimator: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Computes and saves TreeExplainer SHAP plots for XGBoost."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(xgb_estimator)
    shap_values = explainer.shap_values(X_explain)

    # 1. Global Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=15,
        plot_type="bar",
        show=False,
    )
    plt.title("XGBoost Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_xgboost_global_bar.png", dpi=200)
    plt.close()

    # 2. Local Waterfall / Bar Plot for high-risk consignment
    high_risk_idx = int(np.argmax(xgb_estimator.predict_proba(X_explain)[:, 1]))
    plt.figure(figsize=(10, 6))
    top_feat_idx = np.argsort(np.abs(shap_values[high_risk_idx]))[::-1][:10]
    top_names = [feature_names[i] for i in top_feat_idx]
    top_vals = [shap_values[high_risk_idx][i] for i in top_feat_idx]

    colors = ["#dc2626" if v > 0 else "#2563eb" for v in top_vals]
    plt.barh(range(len(top_names)), top_vals, color=colors)
    plt.yticks(range(len(top_names)), top_names)
    plt.xlabel("SHAP Impact on Late Delivery Risk")
    plt.title(f"XGBoost Local SHAP Attribution (Sample #{high_risk_idx})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "shap_xgboost_local.png", dpi=200)
    plt.close()


def explain_neural_model_shap(
    model: torch.nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    model_name: str = "Tabular1DCNN",
) -> None:
    """Computes KernelExplainer SHAP values on neural model predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    def predict_fn(x_numpy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(x_numpy, dtype=torch.float32)
            out = model(tensor).cpu().numpy().flatten()
            return out

    # Sample background subset for fast kernel explanation (< 5 sec)
    bg = shap.sample(X_background, min(30, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, bg)
    explain_sub = X_explain[: min(25, len(X_explain))]
    shap_vals = explainer.shap_values(explain_sub, nsamples=40)

    # Global importance plot
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:12]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [mean_abs_shap[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_scores, color="#059669")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Mean |SHAP Value| (Neural Model Attribution)")
    plt.title(f"{model_name} Tabular Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_{model_name}_global.png", dpi=200)
    plt.close()


def visualize_attention_weights(
    model: torch.nn.Module,
    X_sample: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    """
    Extracts and visualizes attentive pooling weights from Transformer or BiLSTM.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(X_sample[:20], dtype=torch.float32)

        if isinstance(model, HybridLSTMTransformer):
            tokens = model.token_proj(tensor).view(tensor.size(0), model.seq_len, model.embed_dim)
            lstm_out, _ = model.lstm(tokens)
            h_recurrent = model.recurrent_norm(tokens + model.lstm_proj(lstm_out))
            h_trans = model.transformer_block(h_recurrent)
            attn_weights = torch.softmax(model.pool_attn(h_trans), dim=1).cpu().numpy().squeeze(-1) # (B, seq_len)
            seq_labels = [f"Token Cluster {i+1}" for i in range(model.seq_len)]

        elif isinstance(model, BiLSTMStockoutPredictor):
            tokens = model.token_proj(tensor).view(tensor.size(0), model.seq_len, model.embed_dim)
            lstm_out, _ = model.lstm(tokens)
            # Pool attention
            weights = torch.softmax(model.attn_pool.attn(lstm_out), dim=1).cpu().numpy().squeeze(-1)
            attn_weights = weights
            seq_labels = [f"Temporal Token {i+1}" for i in range(model.seq_len)]

        else:
            return

    # Average attention across sample batch
    mean_attn = np.mean(attn_weights, axis=0)

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(seq_labels, mean_attn, color="#8b5cf6", edgecolor="#4c1d95", alpha=0.85)
    plt.ylabel("Attention Allocation Weight")
    plt.title(f"{model_name} Sequence Token Attention Weights")
    plt.ylim(0, max(mean_attn) * 1.25)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f"attention_{model_name}.png", dpi=200)
    plt.close()
