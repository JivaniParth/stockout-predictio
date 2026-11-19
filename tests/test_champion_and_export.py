"""
Unit tests for Champion Selection and ONNX/TorchScript Model Export with Parity Assertion.
Runs fast (< 10 seconds).
"""

import sys
from pathlib import Path
import json
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import Tabular1DCNN
from src.deployment.champion_selector import select_champions
from src.deployment.model_exporter import export_model_artifacts


def test_champion_selection_scoring(tmp_path):
    benchmarks = {
        "HybridLSTMTransformer": {
            "PR_AUC": 0.28,
            "Recall_Late": 0.775,
            "Max_Min_Recall_Gap": 0.02,
            "Latency_P95": 0.85,
            "Memory_MB": 12.0,
        },
        "Tabular1DCNN": {
            "PR_AUC": 0.27,
            "Recall_Late": 0.775,
            "Max_Min_Recall_Gap": 0.02,
            "Latency_P95": 0.35, # Low latency
            "Memory_MB": 2.5,   # Low memory
        },
        "RandomForest": {
            "PR_AUC": 0.32,
            "Recall_Late": 0.708,
            "Max_Min_Recall_Gap": 0.06,
            "Latency_P95": 35.0, # High latency
            "Memory_MB": 45.0,
        },
    }

    output_path = tmp_path / "champion.json"
    result = select_champions(benchmarks, output_path=output_path)

    assert output_path.exists()
    assert "accuracy_champion" in result
    assert "edge_champion" in result
    # Tabular1DCNN should win or be top for edge champion due to low latency/memory
    assert result["edge_champion"]["model_name"] == "Tabular1DCNN"


def test_onnx_torchscript_export_and_parity(tmp_path):
    input_dim = 15
    model = Tabular1DCNN(input_dim=input_dim, dropout_rate=0.0)
    sample_input = np.random.randn(20, input_dim).astype(np.float32)

    fake_prep = tmp_path / "fake_prep.pkl"
    fake_prep.write_text("dummy")
    fake_cfg = tmp_path / "fake_cfg.yaml"
    fake_cfg.write_text("dummy")

    manifest = export_model_artifacts(
        model=model,
        sample_input=sample_input,
        artifacts_dir=tmp_path,
        preprocessor_path=fake_prep,
        config_path=fake_cfg,
        model_name="TestCNN",
        tau_star=0.21,
    )

    assert (tmp_path / "TestCNN.onnx").exists()
    assert (tmp_path / "TestCNN.torchscript.pt").exists()
    assert (tmp_path / "manifest.json").exists()
    assert manifest["numerical_parity_max_diff"] < 1e-4


def test_maut_sensitivity_analysis(tmp_path):
    from src.deployment.champion_selector import run_maut_sensitivity_analysis

    benchmarks = {
        "Tabular1DCNN": {
            "PR_AUC": 0.27,
            "Recall_Late": 0.775,
            "Max_Min_Recall_Gap": 0.02,
            "Latency_P95": 0.35,
            "Memory_MB": 2.5,
        },
        "XGBoost": {
            "PR_AUC": 0.32,
            "Recall_Late": 0.780,
            "Max_Min_Recall_Gap": 0.03,
            "Latency_P95": 12.0,
            "Memory_MB": 18.0,
        },
    }
    sens_out = tmp_path / "sensitivity.json"
    res = run_maut_sensitivity_analysis(benchmarks, output_path=sens_out)

    assert sens_out.exists()
    assert res["total_accuracy_grid_evaluations"] > 0
    assert res["total_edge_grid_evaluations"] > 0
    assert "accuracy_champion_selection_frequencies" in res
    assert "edge_champion_selection_frequencies" in res


def test_xgboost_onnx_export_and_manifest(tmp_path):
    import xgboost as xgb
    from src.deployment.model_exporter import export_xgboost_onnx

    X = np.random.randn(25, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=25)
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    res = export_xgboost_onnx(model, X, tmp_path, model_name="TestXGB")
    assert (tmp_path / "TestXGB.onnx").exists()
    assert res["numerical_parity_max_diff"] < 1e-4
