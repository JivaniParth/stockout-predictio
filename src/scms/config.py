"""
Configuration module for Cloud-Native DL Healthcare Supply Chain Prediction.
Reads config.yaml as the single source of truth without path hardcoding.
"""

from pathlib import Path
import yaml
import torch

# Base Paths (Implementation root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_yaml_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


CONFIG = load_yaml_config()

DATA_PATH = PROJECT_ROOT / CONFIG.get("data", {}).get("raw_path", "SCMS_Delivery_History_Dataset.csv")
MODELS_DIR = PROJECT_ROOT / CONFIG.get("system", {}).get("models_dir", "models")
REPORTS_DIR = PROJECT_ROOT / CONFIG.get("system", {}).get("reports_dir", "reports")
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = REPORTS_DIR / "figures"
TIERS_PATH = PROJECT_ROOT / "config" / "tiers.yaml"

# Ensure directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
from src.scms.utils.device_manager import get_optimal_device

DEVICE, DEVICE_INFO = get_optimal_device(prefer_cuda=True)
RANDOM_SEED = CONFIG.get("system", {}).get("seed", 42)

# Temporal Split Configuration
TRAIN_END_YEAR = CONFIG.get("data", {}).get("split", {}).get("train_end_year", 2013)
VAL_YEAR = CONFIG.get("data", {}).get("split", {}).get("val_year", 2014)
TEST_START_YEAR = CONFIG.get("data", {}).get("split", {}).get("test_start_year", 2015)
