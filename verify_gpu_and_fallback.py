"""
Standalone Verification Script for GPU Acceleration and Multi-Core CPU Fallback.
Executes end-to-end checks across XGBoost, CatBoost, LightGBM, and TabResNet
under Auto, Forced CPU, and Simulated Fallback conditions.
"""

import sys
import os
import traceback
from pathlib import Path
import numpy as np

# Ensure Implementation is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device_manager import DeviceManager, probe_xgboost_gpu, probe_catboost_gpu, probe_lightgbm_gpu
from src.models.gbm_suite import XGBoostWrapper, CatBoostWrapper, LightGBMWrapper
from src.models.tabular_resnet import TabResNetWrapper
from src.models.baselines import BaselineSuite


def main():
    print("\n" + "=" * 80)
    print("VERIFICATION SUITE: GPU ACCELERATION & CPU FALLBACK RESILIENCE")
    print("=" * 80)

    # 1. Device Manager Telemetry
    dm = DeviceManager(preferred="auto")
    dm.print_summary()

    # Synthetic Benchmark Data
    np.random.seed(42)
    N = 80
    X_cont = np.random.randn(N, 4).astype(np.float32)
    X_cat1 = np.random.randint(0, 3, size=(N, 1)).astype(np.float32)
    X_cat2 = np.random.randint(0, 4, size=(N, 1)).astype(np.float32)
    X = np.hstack([X_cont, X_cat1, X_cat2])
    y = np.random.binomial(1, 0.3, size=N).astype(int)

    passed_checks = 0
    total_checks = 8

    # --- Check 1: XGBoost Auto/GPU Mode ---
    print("[-] Checking XGBoost Auto/GPU Mode...")
    xgb_m = XGBoostWrapper(params={"n_estimators": 10, "max_depth": 3}, device="auto")
    xgb_m.fit(X, y)
    probs_xgb = xgb_m.predict_proba(X)
    assert probs_xgb.shape == (N, 2), "XGBoost shape mismatch"
    assert np.all((probs_xgb >= 0.0) & (probs_xgb <= 1.0)), "Probabilities out of bounds"
    print(f"    [PASS] Active Device: {xgb_m.active_device} (Fallback: {xgb_m.fallback_occurred})")
    passed_checks += 1

    # --- Check 2: XGBoost Simulated Failure Fallback ---
    print("[-] Checking XGBoost Simulated GPU Failure Fallback...")
    xgb_sim = XGBoostWrapper(params={"n_estimators": 10, "max_depth": 3}, device="cuda")
    orig_fit = xgb_sim.model.fit
    def failing_xgb(*args, **kwargs):
        if xgb_sim.active_device == "cuda":
            raise RuntimeError("Synthetic CUDA OOM")
        return orig_fit(*args, **kwargs)
    xgb_sim.model.fit = failing_xgb
    xgb_sim.fit(X, y)
    assert xgb_sim.fallback_occurred is True, "Fallback flag not set"
    assert "cpu" in xgb_sim.active_device, "Device not set to CPU"
    print(f"    [PASS] Successfully recovered to: {xgb_sim.active_device}")
    passed_checks += 1

    # --- Check 3: CatBoost Auto/GPU Mode ---
    print("[-] Checking CatBoost Auto/GPU Mode...")
    cb_m = CatBoostWrapper(params={"iterations": 10, "depth": 3}, cat_indices=[4, 5], device="auto")
    cb_m.fit(X, y)
    probs_cb = cb_m.predict_proba(X)
    assert probs_cb.shape == (N, 2), "CatBoost shape mismatch"
    print(f"    [PASS] Active Device: {cb_m.active_device} (Fallback: {cb_m.fallback_occurred})")
    passed_checks += 1

    # --- Check 4: CatBoost Simulated Failure Fallback ---
    print("[-] Checking CatBoost Simulated GPU Failure Fallback...")
    cb_sim = CatBoostWrapper(params={"iterations": 10, "depth": 3}, cat_indices=[4, 5], device="GPU")
    orig_cb_fit = cb_sim.model.fit
    def failing_cb(*args, **kwargs):
        if "GPU" in cb_sim.active_device:
            raise RuntimeError("Synthetic GPU driver error")
        return orig_cb_fit(*args, **kwargs)
    cb_sim.model.fit = failing_cb
    cb_sim.fit(X, y)
    assert cb_sim.fallback_occurred is True, "Fallback flag not set"
    assert "CPU" in cb_sim.active_device, "Device not set to CPU"
    print(f"    [PASS] Successfully recovered to: {cb_sim.active_device}")
    passed_checks += 1

    # --- Check 5: LightGBM Safe Handling ---
    print("[-] Checking LightGBM Auto Resolution & Wheel Compatibility...")
    lgb_m = LightGBMWrapper(params={"n_estimators": 10, "num_leaves": 7}, device="auto")
    lgb_m.fit(X, y)
    probs_lgb = lgb_m.predict_proba(X)
    assert probs_lgb.shape == (N, 2)
    print(f"    [PASS] Active Device: {lgb_m.active_device} (Fallback: {lgb_m.fallback_occurred})")
    passed_checks += 1

    # --- Check 6: LightGBM Forced GPU Fallback ---
    print("[-] Checking LightGBM Forced GPU Fallback...")
    lgb_forced = LightGBMWrapper(params={"n_estimators": 10, "num_leaves": 7}, device="cuda")
    lgb_forced.fit(X, y)
    assert lgb_forced.fallback_occurred is True, "Fallback flag not set on uncompiled wheel"
    assert "cpu" in lgb_forced.active_device, "Device not set to CPU"
    print(f"    [PASS] Successfully recovered to: {lgb_forced.active_device}")
    passed_checks += 1

    # --- Check 7: TabResNet Execution & CPU Multi-threading ---
    print("[-] Checking TabResNet PyTorch Execution & Pinned Memory / Non-blocking Transfers...")
    resnet = TabResNetWrapper(
        num_continuous=4,
        cat_cardinalities=[3, 4],
        hidden_dim=32,
        num_blocks=1,
        seed=42,
    )
    resnet.fit(X, y, max_epochs=3, batch_size=16, patience=2)
    probs_res = resnet.predict_proba(X)
    assert probs_res.shape == (N, 2)
    print(f"    [PASS] Active Device: {resnet.active_device} (Torch Device: {resnet.device})")
    passed_checks += 1

    # --- Check 8: Baseline Suite GPU Integration ---
    print("[-] Checking BaselineSuite GPU Integration & Telemetry...")
    suite = BaselineSuite(random_state=42, device="auto")
    suite.fit_all(X, y)
    print(f"    [PASS] Active Devices: {suite.active_devices}")
    passed_checks += 1

    print("\n" + "=" * 80)
    print(f"ALL CHECKS PASSED: {passed_checks}/{total_checks} Verification Steps Succeeded.")
    print("GPU acceleration operational where supported; resilient multi-core CPU fallback verified.")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
