"""
Model Export and Integrity Manifest Module.
Features:
- Exports PyTorch architectures to TorchScript and ONNX
- Exports XGBoost classifiers to ONNX via onnxmltools
- Supports dynamic batch dimensions for serving
- Optional dynamic INT8 quantization for PyTorch models
- Asserts strict numerical parity (< 1e-4) against native frameworks
- Generates artifacts/manifest.json with SHA256 checksums, git commit, and calibration parameters
- Serves both Accuracy and Edge pathways
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import hashlib
import json
import subprocess
import numpy as np
import onnxruntime as ort
import torch

try:
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hexadecimal hash of target file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash or returns local-dev fallback."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "UNTRACKED_LOCAL_REVISION"


def export_pytorch_artifacts(
    model: torch.nn.Module,
    sample_input: np.ndarray,
    artifacts_dir: Path,
    model_name: str = "Tabular1DCNN",
    quantize_int8: bool = False,
) -> Dict[str, Any]:
    """Exports PyTorch model to TorchScript and ONNX, verifying parity < 1e-4."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.eval()

    torch_tensor = torch.tensor(sample_input[:10], dtype=torch.float32)

    # 1. Export TorchScript
    torchscript_path = artifacts_dir / f"{model_name}.torchscript.pt"
    traced_model = torch.jit.trace(model, torch_tensor)
    traced_model.save(str(torchscript_path))

    # 2. Export ONNX
    onnx_path = artifacts_dir / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        torch_tensor,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_features"],
        output_names=["risk_probability"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "risk_probability": {0: "batch_size"},
        },
    )

    # 3. Optional INT8 Dynamic Quantization
    quantized_path = None
    if quantize_int8:
        try:
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            quantized_path = artifacts_dir / f"{model_name}_int8.pt"
            torch.save(quantized_model.state_dict(), quantized_path)
        except Exception as e:
            print(f"[!] Warning: INT8 dynamic quantization skipped: {e}")

    # 4. Numerical Parity Assertion
    with torch.no_grad():
        pt_preds = model(torch_tensor).cpu().numpy().flatten()

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"input_features": torch_tensor.numpy().astype(np.float32)}
    onnx_preds = ort_session.run(None, ort_inputs)[0].flatten()

    max_abs_diff = float(np.max(np.abs(pt_preds - onnx_preds)))
    if max_abs_diff >= 1e-4:
        raise AssertionError(
            f"NUMERICAL PARITY VIOLATION (PyTorch vs ONNX): max absolute diff {max_abs_diff:.6e} >= 1e-4!"
        )

    return {
        "onnx_artifact": str(onnx_path.name),
        "torchscript_artifact": str(torchscript_path.name),
        "int8_artifact": str(quantized_path.name) if quantized_path else None,
        "numerical_parity_max_diff": max_abs_diff,
        "onnx_sha256": compute_file_sha256(onnx_path),
        "torchscript_sha256": compute_file_sha256(torchscript_path),
    }


def export_xgboost_onnx(
    xgb_model: Any,
    sample_input: np.ndarray,
    artifacts_dir: Path,
    model_name: str = "XGBoost",
) -> Dict[str, Any]:
    """Exports XGBoost classifier to ONNX, verifying numerical parity < 1e-4."""
    if not ONNXMLTOOLS_AVAILABLE:
        raise ImportError("onnxmltools and skl2onnx are required for XGBoost ONNX export.")

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    input_dim = sample_input.shape[1]
    initial_type = [("input_features", FloatTensorType([None, input_dim]))]
    onnx_model = convert_xgboost(xgb_model, initial_types=initial_type, target_opset=15)

    onnx_path = artifacts_dir / f"{model_name}.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Numerical Parity Check
    py_sample = sample_input[:10].astype(np.float32)
    py_probs = xgb_model.predict_proba(py_sample)[:, 1]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = session.run(None, {"input_features": py_sample})
    onnx_probs = ort_outs[1][:, 1].flatten()

    max_abs_diff = float(np.max(np.abs(py_probs - onnx_probs)))
    if max_abs_diff >= 1e-4:
        raise AssertionError(
            f"NUMERICAL PARITY VIOLATION (XGBoost vs ONNX): max absolute diff {max_abs_diff:.6e} >= 1e-4!"
        )

    return {
        "onnx_artifact": str(onnx_path.name),
        "numerical_parity_max_diff": max_abs_diff,
        "onnx_sha256": compute_file_sha256(onnx_path),
    }


def export_model_artifacts(
    model: Any,
    sample_input: np.ndarray,
    artifacts_dir: Path,
    preprocessor_path: Path,
    config_path: Path,
    model_name: str = "ChampionModel",
    tau_star: float = 0.21,
    calibrator_type: str = "isotonic",
    per_tier_thresholds: Optional[Dict[str, float]] = None,
    training_range: str = "2006-2013",
    validation_range: str = "2014",
    test_range: str = "2015+",
    quantize_int8: bool = False,
    edge_model: Optional[Any] = None,
    edge_model_name: str = "Tabular1DCNN",
) -> Dict[str, Any]:
    """
    Exports champion models (PyTorch or XGBoost), checks parity < 1e-4, and writes artifacts/manifest.json.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Export Primary Model
    if isinstance(model, torch.nn.Module):
        primary_export = export_pytorch_artifacts(
            model=model,
            sample_input=sample_input,
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            quantize_int8=quantize_int8,
        )
    else:
        # XGBoost or sklearn tree model
        primary_export = export_xgboost_onnx(
            xgb_model=model,
            sample_input=sample_input,
            artifacts_dir=artifacts_dir,
            model_name=model_name,
        )

    # Export Edge Model if provided and distinct
    edge_export = None
    if edge_model is not None and edge_model_name != model_name:
        if isinstance(edge_model, torch.nn.Module):
            edge_export = export_pytorch_artifacts(
                model=edge_model,
                sample_input=sample_input,
                artifacts_dir=artifacts_dir,
                model_name=edge_model_name,
                quantize_int8=False,
            )
        else:
            edge_export = export_xgboost_onnx(
                xgb_model=edge_model,
                sample_input=sample_input,
                artifacts_dir=artifacts_dir,
                model_name=edge_model_name,
            )

    active_onnx = primary_export["onnx_artifact"]
    active_sha = primary_export["onnx_sha256"]
    active_ts = primary_export.get("torchscript_artifact")
    active_parity = primary_export["numerical_parity_max_diff"]

    manifest = {
        "model_name": model_name,
        "onnx_artifact": active_onnx,
        "torchscript_artifact": active_ts,
        "int8_artifact": primary_export.get("int8_artifact"),
        "numerical_parity_max_diff": active_parity,
        "model_onnx_sha256": active_sha,
        "model_torchscript_sha256": primary_export.get("torchscript_sha256"),
        "preprocessor_sha256": compute_file_sha256(preprocessor_path),
        "config_sha256": compute_file_sha256(config_path),
        "git_commit": get_git_commit_hash(),
        "tau_star": float(tau_star),
        "calibrator_type": calibrator_type,
        "per_tier_thresholds": per_tier_thresholds or {
            "Tier_1_Low_Risk_Hub": float(tau_star),
            "Tier_2_Moderate_Risk": float(tau_star),
            "Tier_3_High_Risk_Underserved": float(tau_star),
        },
        "training_date_range": training_range,
        "validation_date_range": validation_range,
        "test_date_range": test_range,
        "served_pathways": {
            "accuracy_champion": {
                "model_name": model_name,
                "onnx_artifact": primary_export["onnx_artifact"],
                "numerical_parity_max_diff": primary_export["numerical_parity_max_diff"],
                "sha256": primary_export["onnx_sha256"],
            },
            "edge_champion": (
                {
                    "model_name": edge_model_name,
                    "onnx_artifact": edge_export["onnx_artifact"],
                    "torchscript_artifact": edge_export.get("torchscript_artifact"),
                    "numerical_parity_max_diff": edge_export["numerical_parity_max_diff"],
                    "sha256": edge_export["onnx_sha256"],
                }
                if edge_export
                else {
                    "model_name": model_name,
                    "onnx_artifact": primary_export["onnx_artifact"],
                    "torchscript_artifact": primary_export.get("torchscript_artifact"),
                    "numerical_parity_max_diff": primary_export["numerical_parity_max_diff"],
                    "sha256": primary_export["onnx_sha256"],
                }
            ),
        },
    }

    manifest_path = artifacts_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
