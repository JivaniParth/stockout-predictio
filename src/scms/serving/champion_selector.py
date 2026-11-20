"""
Champion Selection Engine & MAUT Sensitivity Analysis.

Scoring Rule:
Scoring is conducted EXCLUSIVELY on 2014 validation metrics (val PR-AUC, val delay recall
at calibrated tau*, val equity gap across tiers) combined with measured QoS (P95 latency,
memory footprint). Prospective test metrics (2015+) are reported strictly POST-SELECTION
to prevent decision leakage.

MAUT Multi-Criteria Utility:
- Accuracy Champion: Prioritizes predictive discrimination (PR-AUC) and clinical minority recall.
- Edge Champion: Prioritizes edge deployability (sub-millisecond latency, compact memory) while retaining recall.

Includes MAUT sensitivity analysis varying weights across a multi-dimensional grid to evaluate
stability of champion selection against preference perturbations.
Saved to artifacts/champion.json and reports/champion_sensitivity.json.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import itertools
import json
import numpy as np
import pandas as pd


DEFAULT_ACCURACY_WEIGHTS = {
    "pr_auc": 0.40,
    "delay_recall": 0.35,
    "equity_gap": 0.15,
    "p95_latency": 0.05,
    "memory_mb": 0.05,
}

DEFAULT_EDGE_WEIGHTS = {
    "pr_auc": 0.20,
    "delay_recall": 0.25,
    "p95_latency": 0.25,
    "memory_mb": 0.20,
    "equity_gap": 0.10,
}


def min_max_norm(values: List[float], higher_is_better: bool = True) -> np.ndarray:
    """Performs min-max feature normalization to [0, 1] range."""
    arr = np.array(values, dtype=float)
    min_v, max_v = np.min(arr), np.max(arr)
    if np.isclose(min_v, max_v):
        return np.ones_like(arr) * 0.5
    if higher_is_better:
        return (arr - min_v) / (max_v - min_v)
    else:
        return (max_v - arr) / (max_v - min_v)


def compute_maut_scores(
    model_benchmarks: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Computes linear additive Multi-Attribute Utility Theory (MAUT) scores.
    """
    models = list(model_benchmarks.keys())
    pr_aucs = [model_benchmarks[m].get("PR_AUC", 0.0) for m in models]
    recalls = [model_benchmarks[m].get("Recall_Late", 0.0) for m in models]
    gaps = [model_benchmarks[m].get("Max_Min_Recall_Gap", 0.1) for m in models]
    latencies = [model_benchmarks[m].get("Latency_P95", 1.0) for m in models]
    memories = [model_benchmarks[m].get("Memory_MB", 1.0) for m in models]

    norm_pr_auc = min_max_norm(pr_aucs, higher_is_better=True)
    norm_recall = min_max_norm(recalls, higher_is_better=True)
    norm_equity = min_max_norm(gaps, higher_is_better=False)  # lower gap is better
    norm_lat = min_max_norm(latencies, higher_is_better=False)  # lower latency is better
    norm_mem = min_max_norm(memories, higher_is_better=False)  # lower memory is better

    scores = {}
    for i, m in enumerate(models):
        score = (
            weights.get("pr_auc", 0.0) * norm_pr_auc[i]
            + weights.get("delay_recall", 0.0) * norm_recall[i]
            + weights.get("equity_gap", 0.0) * norm_equity[i]
            + weights.get("p95_latency", 0.0) * norm_lat[i]
            + weights.get("memory_mb", 0.0) * norm_mem[i]
        )
        scores[m] = float(score)
    return scores


def select_champions(
    validation_benchmarks: Dict[str, Dict[str, float]],
    test_benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
    accuracy_weights: Optional[Dict[str, float]] = None,
    edge_weights: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Computes multi-attribute utility scores exclusively on 2014 VALIDATION metrics + QoS.
    Attaches prospective test metrics strictly post-selection.
    """
    acc_w = accuracy_weights or DEFAULT_ACCURACY_WEIGHTS
    edge_w = edge_weights or DEFAULT_EDGE_WEIGHTS

    models = list(validation_benchmarks.keys())
    if len(models) == 0:
        raise ValueError("No models provided for champion selection.")

    acc_scores = compute_maut_scores(validation_benchmarks, acc_w)
    edge_scores = compute_maut_scores(validation_benchmarks, edge_w)

    accuracy_champion = max(acc_scores.keys(), key=lambda k: acc_scores[k])
    edge_champion = max(edge_scores.keys(), key=lambda k: edge_scores[k])

    result = {
        "scoring_rule": "Multi-Attribute Utility Theory (MAUT) on 2014 Validation Split + Measured QoS",
        "validation_only_scoring": True,
        "weights": {
            "accuracy_champion_weights": acc_w,
            "edge_champion_weights": edge_w,
        },
        "accuracy_champion": {
            "model_name": accuracy_champion,
            "validation_score": acc_scores[accuracy_champion],
            "validation_metrics": validation_benchmarks[accuracy_champion],
            "post_selection_test_metrics": (
                test_benchmarks.get(accuracy_champion, {}) if test_benchmarks else "Reported Post-Selection"
            ),
        },
        "edge_champion": {
            "model_name": edge_champion,
            "validation_score": edge_scores[edge_champion],
            "validation_metrics": validation_benchmarks[edge_champion],
            "post_selection_test_metrics": (
                test_benchmarks.get(edge_champion, {}) if test_benchmarks else "Reported Post-Selection"
            ),
        },
        "all_model_validation_scores": {
            m: {
                "accuracy_score": acc_scores[m],
                "edge_score": edge_scores[m],
                "validation_metrics": validation_benchmarks[m],
            }
            for m in models
        },
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def run_maut_sensitivity_analysis(
    validation_benchmarks: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = Path("reports/champion_sensitivity.json"),
) -> Dict[str, Any]:
    """
    Varies MAUT weights across a dense multi-dimensional grid to evaluate
    the sensitivity and stability of the Accuracy and Edge champions.
    """
    models = list(validation_benchmarks.keys())
    if not models:
        return {}

    # Grid search ranges
    # Accuracy Grid: vary PR-AUC weight from 0.20 to 0.60, recall from 0.20 to 0.50
    accuracy_results = []
    edge_results = []

    # Grid steps
    pr_auc_vals = [0.20, 0.30, 0.40, 0.50, 0.60]
    recall_vals = [0.20, 0.30, 0.40, 0.50]
    equity_vals = [0.05, 0.15, 0.25]
    lat_vals = [0.02, 0.05, 0.10, 0.20, 0.30]
    mem_vals = [0.02, 0.05, 0.10, 0.20, 0.30]

    acc_champion_counts: Dict[str, int] = {m: 0 for m in models}
    edge_champion_counts: Dict[str, int] = {m: 0 for m in models}
    total_acc_trials = 0
    total_edge_trials = 0

    # 1. Sweep Accuracy Grid (Quality & equity dominant)
    for p_auc in pr_auc_vals:
        for rec in recall_vals:
            for eq in equity_vals:
                remaining = max(1.0 - (p_auc + rec + eq), 0.0)
                lat = remaining / 2.0
                mem = remaining / 2.0
                total_w = p_auc + rec + eq + lat + mem
                w = {
                    "pr_auc": p_auc / total_w,
                    "delay_recall": rec / total_w,
                    "equity_gap": eq / total_w,
                    "p95_latency": lat / total_w,
                    "memory_mb": mem / total_w,
                }
                scores = compute_maut_scores(validation_benchmarks, w)
                champ = max(scores.keys(), key=lambda k: scores[k])
                acc_champion_counts[champ] += 1
                total_acc_trials += 1
                accuracy_results.append({
                    "weights": {k: round(v, 4) for k, v in w.items()},
                    "champion": champ,
                    "score": round(scores[champ], 4),
                })

    # 2. Sweep Edge Grid (QoS dominant)
    for lat in lat_vals:
        for mem in mem_vals:
            for rec in [0.15, 0.25, 0.35]:
                for p_auc in [0.10, 0.20]:
                    remaining = max(1.0 - (lat + mem + rec + p_auc), 0.0)
                    eq = remaining
                    total_w = lat + mem + rec + p_auc + eq
                    w = {
                        "p95_latency": lat / total_w,
                        "memory_mb": mem / total_w,
                        "delay_recall": rec / total_w,
                        "pr_auc": p_auc / total_w,
                        "equity_gap": eq / total_w,
                    }
                    scores = compute_maut_scores(validation_benchmarks, w)
                    champ = max(scores.keys(), key=lambda k: scores[k])
                    edge_champion_counts[champ] += 1
                    total_edge_trials += 1
                    edge_results.append({
                        "weights": {k: round(v, 4) for k, v in w.items()},
                        "champion": champ,
                        "score": round(scores[champ], 4),
                    })

    acc_frequencies = {
        m: {
            "count": acc_champion_counts[m],
            "selection_rate_pct": round(100.0 * acc_champion_counts[m] / max(total_acc_trials, 1), 2),
        }
        for m in models
    }

    edge_frequencies = {
        m: {
            "count": edge_champion_counts[m],
            "selection_rate_pct": round(100.0 * edge_champion_counts[m] / max(total_edge_trials, 1), 2),
        }
        for m in models
    }

    report = {
        "summary": "MAUT Weight Sensitivity Grid Analysis across Accuracy & Edge Formulations",
        "total_accuracy_grid_evaluations": total_acc_trials,
        "total_edge_grid_evaluations": total_edge_trials,
        "accuracy_champion_selection_frequencies": acc_frequencies,
        "edge_champion_selection_frequencies": edge_frequencies,
        "accuracy_champion_stable": max(acc_frequencies.keys(), key=lambda k: acc_frequencies[k]["count"]),
        "edge_champion_stable": max(edge_frequencies.keys(), key=lambda k: edge_frequencies[k]["count"]),
        "sample_grid_accuracy_trials": accuracy_results[:10],
        "sample_grid_edge_trials": edge_results[:10],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
