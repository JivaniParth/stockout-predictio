"""
Centralized Hardware Acceleration and Device Capability Engine.
Detects GPU capabilities for PyTorch, XGBoost, CatBoost, and LightGBM,
providing graceful, automatic fallback to multi-core CPU parallel processing.
"""

import os
import platform
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

logger = logging.getLogger("DeviceManager")

# Module-level cached probe results
_SYS_GPU_INFO: Optional[Dict[str, Any]] = None
_XGB_PROBE: Optional[Tuple[bool, str]] = None
_CB_PROBE: Optional[Tuple[bool, str]] = None
_LGB_PROBE: Optional[Tuple[bool, str]] = None


def detect_system_gpu() -> Dict[str, Any]:
    """
    Detects system GPU hardware and CUDA driver readiness.
    Caches results to avoid repeated hardware queries.
    """
    global _SYS_GPU_INFO
    if _SYS_GPU_INFO is not None:
        return _SYS_GPU_INFO

    cpu_cores = os.cpu_count() or 4
    gpu_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": cpu_cores,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "device_count": 0,
        "device_name": "None",
        "vram_gb": 0.0,
    }

    if torch.cuda.is_available():
        try:
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_info["vram_gb"] = round(total_mem / (1024**3), 2)
        except Exception as e:
            logger.warning(f"Failed querying CUDA device properties: {e}")

    # Fallback to nvidia-smi if torch lacks CUDA binaries
    if not gpu_info["cuda_available"]:
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                if len(parts) >= 2:
                    gpu_info["hardware_detected"] = True
                    gpu_info["hardware_gpu_name"] = parts[0]
                    gpu_info["hardware_vram_mb"] = parts[1]
                    if len(parts) >= 3:
                        gpu_info["hardware_driver_version"] = parts[2]
        except Exception:
            pass

    _SYS_GPU_INFO = gpu_info
    return _SYS_GPU_INFO


def get_optimal_device(prefer_cuda: bool = True) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Robustly resolves optimal PyTorch execution device.
    Actively probes CUDA tensor allocation; falls back to tuned multi-core CPU on failure.
    """
    cpu_cores = os.cpu_count() or 4
    info: Dict[str, Any] = {
        "device_type": "cpu",
        "device_name": f"CPU ({cpu_cores} threads)",
        "cpu_threads": cpu_cores,
    }

    if prefer_cuda and torch.cuda.is_available():
        try:
            # Active probe: test device memory allocation
            probe = torch.zeros(1, device="cuda")
            del probe
            device_name = torch.cuda.get_device_name(0)
            device = torch.device("cuda")
            total_mem = torch.cuda.get_device_properties(0).total_memory
            info.update({
                "device_type": "cuda",
                "device_name": device_name,
                "cuda_device_count": torch.cuda.device_count(),
                "vram_gb": round(total_mem / (1024**3), 2),
            })
            return device, info
        except Exception as e:
            logger.warning(f"CUDA reported available but tensor allocation probe failed: {e}. Falling back to CPU.")
            info["fallback_reason"] = str(e)

    # Multi-core CPU Fallback path
    device = torch.device("cpu")
    try:
        torch.set_num_threads(cpu_cores)
    except Exception:
        pass
    return device, info


def probe_xgboost_gpu() -> Tuple[bool, str]:
    """
    Probes XGBoost histogram GPU execution capability.
    Returns (True, 'cuda') if operational, else (False, error_reason).
    """
    global _XGB_PROBE
    if _XGB_PROBE is not None:
        return _XGB_PROBE

    try:
        import xgboost as xgb
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1])
        clf = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            random_state=42,
        )
        clf.fit(X, y)
        _XGB_PROBE = (True, "cuda")
    except Exception as e:
        _XGB_PROBE = (False, str(e))

    return _XGB_PROBE


def probe_catboost_gpu() -> Tuple[bool, str]:
    """
    Probes CatBoost GPU execution capability.
    Returns (True, 'GPU') if operational, else (False, error_reason).
    """
    global _CB_PROBE
    if _CB_PROBE is not None:
        return _CB_PROBE

    try:
        import catboost as cb
        import tempfile
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1])
        clf = cb.CatBoostClassifier(
            iterations=1,
            depth=1,
            task_type="GPU",
            train_dir=tempfile.gettempdir(),
            verbose=False,
            random_seed=42,
        )
        clf.fit(X, y)
        _CB_PROBE = (True, "GPU")
    except Exception as e:
        _CB_PROBE = (False, str(e))

    return _CB_PROBE


def probe_lightgbm_gpu() -> Tuple[bool, str]:
    """
    Probes LightGBM CUDA/OpenCL GPU execution capability.
    Returns (True, device_str) if operational, else (False, error_reason).
    """
    global _LGB_PROBE
    if _LGB_PROBE is not None:
        return _LGB_PROBE

    try:
        import lightgbm as lgb
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.array([0, 1])
        for dev in ["cuda", "gpu"]:
            try:
                clf = lgb.LGBMClassifier(
                    n_estimators=1,
                    max_depth=1,
                    device=dev,
                    verbose=-1,
                    random_state=42,
                )
                clf.fit(X, y)
                _LGB_PROBE = (True, dev)
                return _LGB_PROBE
            except Exception:
                continue
        _LGB_PROBE = (False, "CUDA/GPU Tree Learner not enabled in LightGBM build")
    except Exception as e:
        _LGB_PROBE = (False, str(e))

    return _LGB_PROBE


class DeviceManager:
    """
    Centralized Device Manager providing capability detection,
    optimal parameter resolution, and hardware telemetry.
    """

    def __init__(self, preferred: str = "auto", require_gpu: bool = False):
        self.preferred = preferred.lower()
        self.require_gpu = require_gpu
        self.sys_gpu = detect_system_gpu()
        self.cpu_cores = os.cpu_count() or 4

        # Tune PyTorch CPU threads upfront
        try:
            torch.set_num_threads(self.cpu_cores)
        except Exception:
            pass

        # Probe frameworks
        if self.preferred in ["cpu", "none"]:
            self.xgb_gpu, self.xgb_msg = False, "Forced CPU mode"
            self.cb_gpu, self.cb_msg = False, "Forced CPU mode"
            self.lgb_gpu, self.lgb_msg = False, "Forced CPU mode"
            self.torch_device = torch.device("cpu")
            self.torch_device_info = {"device_type": "cpu", "device_name": f"CPU ({self.cpu_cores} threads)"}
        else:
            self.torch_device, self.torch_device_info = get_optimal_device(prefer_cuda=True)
            self.xgb_gpu, self.xgb_msg = probe_xgboost_gpu()
            self.cb_gpu, self.cb_msg = probe_catboost_gpu()
            self.lgb_gpu, self.lgb_msg = probe_lightgbm_gpu()

        if self.require_gpu:
            errors = []
            if not torch.cuda.is_available():
                errors.append(f"PyTorch CUDA is unavailable (torch={torch.__version__})")
            if not self.xgb_gpu:
                errors.append(f"XGBoost GPU failed: {self.xgb_msg}")
            if not self.cb_gpu:
                errors.append(f"CatBoost GPU failed: {self.cb_msg}")
            if errors:
                raise RuntimeError("--require-gpu failed loudly: " + " | ".join(errors))

    def get_torch_device(self) -> torch.device:
        """Returns resolved PyTorch torch.device."""
        return self.torch_device

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Returns optimal parameters for XGBoost."""
        if self.preferred != "cpu" and self.xgb_gpu:
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "hist", "device": "cpu", "n_jobs": -1}

    def get_catboost_params(self) -> Dict[str, Any]:
        """Returns optimal parameters for CatBoost."""
        if self.preferred != "cpu" and self.cb_gpu:
            return {"task_type": "GPU"}
        return {"task_type": "CPU", "thread_count": -1}

    def get_lightgbm_params(self) -> Dict[str, Any]:
        """Returns optimal parameters for LightGBM."""
        if self.preferred != "cpu" and self.lgb_gpu:
            return {"device": "cuda"}
        return {"device": "cpu", "n_jobs": -1}

    def get_device_info(self) -> Dict[str, Any]:
        """Returns complete hardware and execution telemetry dictionary."""
        return {
            "platform": self.sys_gpu.get("platform", platform.platform()),
            "cpu_cores": self.cpu_cores,
            "torch_device": str(self.torch_device),
            "torch_device_name": self.torch_device_info.get("device_name", "Unknown"),
            "xgboost_device": "cuda" if (self.preferred != "cpu" and self.xgb_gpu) else f"cpu ({self.cpu_cores} threads)",
            "xgboost_gpu_supported": self.xgb_gpu,
            "catboost_device": "GPU" if (self.preferred != "cpu" and self.cb_gpu) else f"CPU ({self.cpu_cores} threads)",
            "catboost_gpu_supported": self.cb_gpu,
            "lightgbm_device": "cuda" if (self.preferred != "cpu" and self.lgb_gpu) else f"cpu ({self.cpu_cores} threads)",
            "lightgbm_gpu_supported": self.lgb_gpu,
            "lightgbm_status_note": "Multi-core CPU (Prebuilt wheel lacks OpenCL/CUDA tree learner)" if not self.lgb_gpu else "GPU accelerated",
        }

    def get_summary_table(self) -> str:
        """Generates a Markdown table of active devices per framework."""
        info = self.get_device_info()
        lines = [
            "| Framework / Model | Requested | Active Execution Device | Capability Status / Note |",
            "|---|---|---|---|",
            f"| **PyTorch (TabResNet)** | {self.preferred} | `{info['torch_device']}` | {info['torch_device_name']} |",
            f"| **XGBoost** | {self.preferred} | `{info['xgboost_device']}` | {'Operational on GPU' if self.xgb_gpu else self.xgb_msg} |",
            f"| **CatBoost** | {self.preferred} | `{info['catboost_device']}` | {'Operational on GPU' if self.cb_gpu else self.cb_msg} |",
            f"| **LightGBM** | {self.preferred} | `{info['lightgbm_device']}` | {info['lightgbm_status_note']} |",
        ]
        return "\n".join(lines)

    def print_summary(self):
        """Prints a clean summary banner of hardware capabilities to stdout."""
        info = self.get_device_info()
        print("\n" + "=" * 78)
        print("HARDWARE ACCELERATION & DEVICE CAPABILITY ENGINE")
        print("=" * 78)
        print(f"[-] Platform: {info['platform']} | Total CPU Cores: {info['cpu_cores']}")
        hw_name = self.sys_gpu.get("hardware_gpu_name", self.sys_gpu.get("device_name", "None"))
        print(f"[-] Dedicated GPU Hardware: {hw_name}")
        print(f"[-] PyTorch Stack: {info['torch_device']} ({info['torch_device_name']})")
        print(f"[-] XGBoost Stack: {info['xgboost_device']} (GPU Probed: {self.xgb_gpu})")
        print(f"[-] CatBoost Stack: {info['catboost_device']} (GPU Probed: {self.cb_gpu})")
        print(f"[-] LightGBM Stack: {info['lightgbm_device']} ({info['lightgbm_status_note']})")
        print("=" * 78 + "\n")
