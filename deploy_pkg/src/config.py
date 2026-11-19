"""
Configuration module for Cloud-Native DL Healthcare Supply Chain Prediction.
Co-evaluates predictive accuracy, Cloud QoS, and Health Equity across temporal splits.
"""

from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "SCMS_Delivery_History_Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

# Temporal Split Configuration
TRAIN_END_YEAR = 2013       # 2006-2013: Historical Training Data
VAL_YEAR = 2014             # 2014: Validation for hyperparameter tuning & threshold calibration
TEST_START_YEAR = 2015      # 2015: Strictly prospective out-of-sample test

# Target Definition
TARGET_COL = "Late_Delivery_Risk"

# Deep Learning Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

# QoS Profiling Parameters
QOS_WARMUP_RUNS = 50
QOS_BENCHMARK_RUNS = 500
QOS_BATCH_SIZES = [1, 16, 64, 128]

# Health Equity / Country Tier Categorization
# Derived from historical delay rates and resource infrastructure gradients in sub-Saharan Africa and Asia
TIER_1_COUNTRIES = [
    "South Africa", "Vietnam", "Ethiopia", "Namibia", "Botswana", 
    "Dominican Republic", "Guatemala"
]
TIER_3_HIGH_RISK_COUNTRIES = [
    "Burundi", "Congo, DRC", "Mozambique", "Zambia", "South Sudan", 
    "Togo", "Senegal", "Cameroon"
]
# All other countries fall into TIER_2 (Moderate Risk / Urban & Regional nodes)
