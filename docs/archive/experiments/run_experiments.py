"""
Master Research Experiment Pipeline.
Executes end-to-end training, validation, testing, Cloud QoS profiling,
health equity assessment, and publication-ready artifact generation.
Addresses all 4 Research Questions and Literature Review Gaps.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch

from src.config import (
    DEVICE,
    RANDOM_SEED,
    RESULTS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    MAX_EPOCHS,
)
from src.data.preprocessor import prepare_data_splits
from src.data.dataset import get_dataloaders
from src.models.baselines import BaselineSuite
from src.models.dnn import DeepTabularMLP
from src.models.lstm import BiLSTMStockoutPredictor
from src.models.hybrid_transformer import HybridLSTMTransformer
from src.models.cnn import Tabular1DCNN
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.qos_profiler import benchmark_model_qos
from src.evaluation.equity import evaluate_health_equity
from src.utils.visualization import (
    plot_roc_pr_comparison,
    plot_qos_tradeoff,
    plot_equity_tiers,
)

# Set global seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def calibrate_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds decision threshold maximizing validation F1 score."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def run_pipeline():
    print("=" * 80)
    print("STARTING HSCM RESEARCH BENCHMARK: DEEP LEARNING, CLOUD QOS & HEALTH EQUITY")
    print("=" * 80)
    start_total_time = time.time()

    # 1. Prepare Data Splits (Leakage-Safe Chronological Partition)
    print("\n[Step 1] Loading data and preparing temporal splits...")
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, test_df = prepare_data_splits()
    input_dim = X_train.shape[1]
    print(f"  Training Set (<=2013):   {X_train.shape[0]} samples | Late Rate: {y_train.mean():.4f}")
    print(f"  Validation Set (2014):    {X_val.shape[0]} samples | Late Rate: {y_val.mean():.4f}")
    print(f"  Prospective Test (2015):  {X_test.shape[0]} samples | Late Rate: {y_test.mean():.4f}")
    print(f"  Feature Dimensions:       {input_dim} features")

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE
    )

    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    all_qos: list = []
    all_equity: Dict[str, Dict[str, Any]] = {}

    # 2. Train Classical Baselines
    print("\n[Step 2] Training Classical Baselines (LR, RF, XGBoost)...")
    baseline_suite = BaselineSuite(random_state=RANDOM_SEED)
    baseline_suite.fit_all(X_train, y_train)

    for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
        print(f"\n[-] Evaluating Baseline: {name}")
        baseline_suite.save_model(name)

        # Calibrate threshold on 2014 validation set
        val_probs = baseline_suite.predict_proba(name, X_val)
        opt_thresh = calibrate_threshold(y_val, val_probs)
        print(f"  Optimal Threshold from 2014 Validation: {opt_thresh:.2f}")

        # Prospective 2015 Test Predictions
        test_probs = baseline_suite.predict_proba(name, X_test)
        metrics = compute_comprehensive_metrics(y_test, test_probs, threshold=opt_thresh)
        all_metrics[name] = metrics
        all_predictions[name] = {"probs": test_probs, "preds": (test_probs >= opt_thresh).astype(int)}

        # QoS Profiling
        print(f"  Profiling Cloud QoS for {name}...")
        raw_model = baseline_suite.models[name]
        qos = benchmark_model_qos(raw_model, input_dim=input_dim, model_name=name, device=DEVICE)
        all_qos.append(qos)

        # Health Equity Evaluation
        print(f"  Evaluating Health Equity for {name}...")
        equity = evaluate_health_equity(test_df, y_test, test_probs, threshold=opt_thresh, model_name=name)
        all_equity[name] = equity

    # 3. Train Deep Learning Models
    dl_models = {
        "DeepTabularMLP": DeepTabularMLP(input_dim=input_dim, hidden_dim=256, dropout_rate=0.2),
        "BiLSTM": BiLSTMStockoutPredictor(input_dim=input_dim, seq_len=5, embed_dim=64, hidden_dim=64, dropout_rate=0.2),
        "HybridLSTMTransformer": HybridLSTMTransformer(input_dim=input_dim, seq_len=6, embed_dim=64, lstm_hidden_dim=32, num_heads=4),
        "Tabular1DCNN": Tabular1DCNN(input_dim=input_dim, dropout_rate=0.2),
    }

    pos_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    print(f"\n[Step 3] Training Deep Learning Suite (Class weight pos_ratio: {pos_ratio:.2f})...")

    for name, model in dl_models.items():
        print(f"\n" + "-" * 60)
        print(f"[-] Training Deep Learning Model: {name}")
        trainer = ModelTrainer(
            model=model,
            model_name=name,
            pos_weight=min(pos_ratio, 6.0), # balanced penalty
            device=DEVICE,
            lr=1e-3,
        )
        trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS, patience=8)
        opt_thresh = trainer.optimal_threshold

        # Prospective Test Evaluation
        _, roc_test, pr_auc_test, test_probs, _ = trainer.evaluate(test_loader)
        metrics = compute_comprehensive_metrics(y_test, test_probs, threshold=opt_thresh)
        all_metrics[name] = metrics
        all_predictions[name] = {"probs": test_probs, "preds": (test_probs >= opt_thresh).astype(int)}

        # QoS Profiling
        print(f"  Profiling Cloud QoS for {name}...")
        qos = benchmark_model_qos(trainer.model, input_dim=input_dim, model_name=name, device=DEVICE)
        all_qos.append(qos)

        # Health Equity Evaluation
        print(f"  Evaluating Health Equity for {name}...")
        equity = evaluate_health_equity(test_df, y_test, test_probs, threshold=opt_thresh, model_name=name)
        all_equity[name] = equity

    # 4. Save Benchmark Results
    print("\n[Step 4] Saving Benchmark Data...")
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "Model"
    metrics_path = RESULTS_DIR / "benchmark_metrics.csv"
    metrics_df.to_csv(metrics_path)
    print(f"[+] Saved ML Metrics to {metrics_path}")

    qos_df = pd.DataFrame(all_qos).set_index("Model")
    qos_path = RESULTS_DIR / "qos_benchmark.csv"
    qos_df.to_csv(qos_path)
    print(f"[+] Saved Cloud QoS Benchmark to {qos_path}")

    equity_records = []
    for m, eq in all_equity.items():
        t1 = eq["Tier_Metrics"].get("Tier_1_Low_Risk_Hub", {})
        t2 = eq["Tier_Metrics"].get("Tier_2_Moderate_Risk", {})
        t3 = eq["Tier_Metrics"].get("Tier_3_High_Risk_Underserved", {})
        equity_records.append({
            "Model": m,
            "Tier1_Samples": t1.get("Samples", 0),
            "Tier1_Recall": t1.get("Recall", 0.0),
            "Tier2_Samples": t2.get("Samples", 0),
            "Tier2_Recall": t2.get("Recall", 0.0),
            "Tier3_Samples": t3.get("Samples", 0),
            "Tier3_Recall": t3.get("Recall", 0.0),
            "Equal_Opportunity_Diff": eq["Equal_Opportunity_Difference"],
            "Disparate_Impact_Ratio": eq["Disparate_Impact_Ratio"],
            "Statistical_Parity_Diff": eq["Statistical_Parity_Difference"],
        })
    equity_df = pd.DataFrame(equity_records).set_index("Model")
    equity_path = RESULTS_DIR / "health_equity_benchmark.csv"
    equity_df.to_csv(equity_path)
    print(f"[+] Saved Health Equity Benchmark to {equity_path}")

    # Master summary JSON
    summary_data = {
        "ml_metrics": all_metrics,
        "qos_metrics": all_qos,
        "health_equity": all_equity,
    }
    with open(RESULTS_DIR / "master_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 5. Generate Publication-Quality Visualizations
    print("\n[Step 5] Generating Publication Figures...")
    plot_roc_pr_comparison(all_predictions, y_test)
    plot_qos_tradeoff(all_qos, all_metrics)
    plot_equity_tiers(all_equity)

    total_duration = time.time() - start_total_time
    print(f"\n[+] BENCHMARK COMPLETE IN {total_duration:.1f} SECONDS!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
