"""
Automated Model Retraining and Manifest Versioning Trigger.
Executes training, calibration, champion selection, and model export to produce
a new timestamped manifest without overwriting previous versions, ensuring rollback capability.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import datetime
import json
import shutil

from src.deployment.champion_selector import select_champions
from src.deployment.model_exporter import export_model_artifacts


def execute_versioned_retrain(
    model: Any,
    sample_input: Any,
    artifacts_dir: Path,
    preprocessor_path: Path,
    config_path: Path,
    model_name: str,
    tau_star: float,
    calibrator_type: str = "isotonic",
    per_tier_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Executes model retraining export and saves a new versioned manifest.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_manifest_filename = f"manifest_v_{timestamp}.json"
    versioned_manifest_path = artifacts_dir / versioned_manifest_filename

    # Export artifacts
    manifest = export_model_artifacts(
        model=model,
        sample_input=sample_input,
        artifacts_dir=artifacts_dir,
        preprocessor_path=preprocessor_path,
        config_path=config_path,
        model_name=f"{model_name}_v_{timestamp}",
        tau_star=tau_star,
        calibrator_type=calibrator_type,
        per_tier_thresholds=per_tier_thresholds,
    )

    # Save timestamped manifest
    with open(versioned_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Copy to latest manifest.json
    latest_manifest_path = artifacts_dir / "manifest.json"
    shutil.copyfile(versioned_manifest_path, latest_manifest_path)

    manifest["manifest_version"] = f"v_{timestamp}"
    manifest["versioned_manifest_path"] = str(versioned_manifest_path)

    return manifest
