"""
Comprehensive Test Suite for Hardware Acceleration, Device Management,
and Automatic Graceful CPU Fallback across Model Families.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure Implementation is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device_manager import (
    detect_system_gpu,
    get_optimal_device,
    probe_xgboost_gpu,
    probe_catboost_gpu,
    probe_lightgbm_gpu,
    DeviceManager,
)
from src.models.gbm_suite import LightGBMWrapper, CatBoostWrapper, XGBoostWrapper
from src.models.tabular_resnet import TabResNet, TabResNetWrapper
from src.models.baselines import BaselineSuite


@pytest.fixture
def synthetic_data():
    """Generates synthetic tabular continuous and categorical data."""
    np.random.seed(42)
    N = 60
    # 4 continuous features, 2 categorical features (cardinalities 3, 4)
    X_cont = np.random.randn(N, 4).astype(np.float32)
    X_cat1 = np.random.randint(0, 3, size=(N, 1)).astype(np.float32)
    X_cat2 = np.random.randint(0, 4, size=(N, 1)).astype(np.float32)
    X = np.hstack([X_cont, X_cat1, X_cat2])
    y = np.random.binomial(1, 0.3, size=N).astype(int)
    return X, y


def test_system_gpu_detection():
    """Validates that detect_system_gpu returns consistent hardware telemetry."""
    info = detect_system_gpu()
    assert isinstance(info, dict)
    assert "platform" in info
    assert "cpu_count" in info
    assert info["cpu_count"] >= 1
    assert "cuda_available" in info
    assert isinstance(info["cuda_available"], bool)


def test_device_manager_initialization():
    """Validates DeviceManager with auto and forced CPU configurations."""
    dm_auto = DeviceManager(preferred="auto")
    assert dm_auto.torch_device is not None
    assert isinstance(dm_auto.get_torch_device(), torch.device)
    assert isinstance(dm_auto.get_xgboost_params(), dict)
    assert isinstance(dm_auto.get_catboost_params(), dict)
    assert isinstance(dm_auto.get_lightgbm_params(), dict)

    dm_cpu = DeviceManager(preferred="cpu")
    assert dm_cpu.torch_device == torch.device("cpu")
    assert dm_cpu.get_xgboost_params()["device"] == "cpu"
    assert dm_cpu.get_catboost_params()["task_type"] == "CPU"
    assert dm_cpu.get_lightgbm_params()["device"] == "cpu"

    summary = dm_auto.get_summary_table()
    assert "PyTorch" in summary
    assert "XGBoost" in summary
    assert "CatBoost" in summary
    assert "LightGBM" in summary


def test_xgboost_gpu_and_cpu_fallback(synthetic_data):
    """Validates XGBoost execution on probed device and graceful CPU fallback."""
    X, y = synthetic_data
    # 1. Proactive auto execution
    m_auto = XGBoostWrapper(params={"n_estimators": 5, "max_depth": 2}, device="auto")
    m_auto.fit(X, y)
    probs = m_auto.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert m_auto.active_device in ["cuda", "cpu"]

    # 2. Forced CPU execution
    m_cpu = XGBoostWrapper(params={"n_estimators": 5, "max_depth": 2}, device="cpu")
    m_cpu.fit(X, y)
    assert m_cpu.active_device == "cpu"
    assert m_cpu.fallback_occurred is False

    # 3. Simulated GPU failure triggering reactive fallback
    m_fall = XGBoostWrapper(params={"n_estimators": 5, "max_depth": 2}, device="cuda")
    # Simulate an error on GPU fit
    original_fit = m_fall.model.fit

    def failing_fit(*args, **kwargs):
        if m_fall.active_device == "cuda":
            raise RuntimeError("CUDA out of memory simulation")
        return original_fit(*args, **kwargs)

    m_fall.model.fit = failing_fit
    m_fall.fit(X, y)

    assert m_fall.fallback_occurred is True
    assert "cpu" in m_fall.active_device
    p_fall = m_fall.predict_proba(X)
    assert p_fall.shape == (len(X), 2)


def test_catboost_gpu_and_cpu_fallback(synthetic_data):
    """Validates CatBoost execution on probed device and graceful CPU fallback."""
    X, y = synthetic_data
    # 1. Proactive auto execution
    m_auto = CatBoostWrapper(params={"iterations": 5, "depth": 2}, cat_indices=[4, 5], device="auto")
    m_auto.fit(X, y)
    probs = m_auto.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # 2. Forced CPU execution
    m_cpu = CatBoostWrapper(params={"iterations": 5, "depth": 2}, cat_indices=[4, 5], device="cpu")
    m_cpu.fit(X, y)
    assert m_cpu.active_device == "CPU"
    assert m_cpu.fallback_occurred is False

    # 3. Simulated GPU failure triggering reactive fallback
    m_fall = CatBoostWrapper(params={"iterations": 5, "depth": 2}, cat_indices=[4, 5], device="GPU")
    original_fit = m_fall.model.fit

    def failing_fit(*args, **kwargs):
        if "GPU" in m_fall.active_device:
            raise RuntimeError("CatBoost GPU device initialization failed")
        return original_fit(*args, **kwargs)

    m_fall.model.fit = failing_fit
    m_fall.fit(X, y)

    assert m_fall.fallback_occurred is True
    assert "CPU" in m_fall.active_device
    p_fall = m_fall.predict_proba(X)
    assert p_fall.shape == (len(X), 2)


def test_lightgbm_safe_probing_and_fallback(synthetic_data):
    """Validates LightGBM safe probing, execution, and forced GPU fallback."""
    X, y = synthetic_data
    # 1. Proactive auto execution (defaults to CPU safely if wheel lacks GPU learner)
    m_auto = LightGBMWrapper(params={"n_estimators": 5, "num_leaves": 7}, device="auto")
    m_auto.fit(X, y)
    probs = m_auto.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # 2. Forced GPU attempt on prebuilt wheel triggers reactive CPU fallback
    m_gpu = LightGBMWrapper(params={"n_estimators": 5, "num_leaves": 7}, device="cuda")
    m_gpu.fit(X, y)
    # Since standard Windows wheel lacks CUDA tree learner, it should have gracefully fallen back
    assert m_gpu.fallback_occurred is True
    assert "cpu" in m_gpu.active_device
    p_gpu = m_gpu.predict_proba(X)
    assert p_gpu.shape == (len(X), 2)


def test_tabresnet_gpu_and_cpu_fallback(synthetic_data):
    """Validates TabResNet initialization, execution, and fallback to multi-core CPU."""
    X, y = synthetic_data
    # num_continuous = 4, cat_cardinalities = [3, 4]
    m = TabResNetWrapper(
        num_continuous=4,
        cat_cardinalities=[3, 4],
        hidden_dim=32,
        num_blocks=1,
        device="cpu",
        seed=42,
    )
    m.fit(X, y, max_epochs=2, batch_size=16, patience=2)
    probs = m.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Probabilities sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)

    # Simulated CUDA fallback test
    m_sim = TabResNetWrapper(
        num_continuous=4,
        cat_cardinalities=[3, 4],
        hidden_dim=32,
        num_blocks=1,
        device="cpu",
        seed=42,
    )
    # Artificially set device type to cuda to test the try-except handler
    m_sim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if m_sim.device.type != "cuda":
        # Create a mock device with type 'cuda'
        class MockCudaDevice:
            type = "cuda"
            def __str__(self):
                return "cuda:0"
        m_sim.device = MockCudaDevice()

    original_fit_loop = m_sim._fit_loop

    def failing_fit_loop(*args, **kwargs):
        if getattr(m_sim.device, "type", None) == "cuda":
            raise RuntimeError("Synthetic CUDA Out of Memory")
        return original_fit_loop(*args, **kwargs)

    m_sim._fit_loop = failing_fit_loop
    m_sim.fit(X, y, max_epochs=1, batch_size=16)

    assert m_sim.fallback_occurred is True
    assert "cpu" in m_sim.active_device
    p_sim = m_sim.predict_proba(X)
    assert p_sim.shape == (len(X), 2)


def test_baseline_suite_device_handling(synthetic_data):
    """Validates that BaselineSuite initializes and trains with device telemetry."""
    X, y = synthetic_data
    suite = BaselineSuite(random_state=42, device="auto")
    suite.fit_all(X, y)

    assert "LogisticRegression" in suite.models
    assert "RandomForest" in suite.models
    assert "XGBoost" in suite.models
    assert "XGBoost" in suite.active_devices

    p_lr = suite.predict_proba("LogisticRegression", X)
    p_rf = suite.predict_proba("RandomForest", X)
    p_xgb = suite.predict_proba("XGBoost", X)

    assert len(p_lr) == len(X)
    assert len(p_rf) == len(X)
    assert len(p_xgb) == len(X)
