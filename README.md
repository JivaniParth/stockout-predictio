# Healthcare Supply Chain Stockout Prediction Platform (HSCM)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204050-green.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Clean Architecture](https://img.shields.io/badge/Clean%20Architecture-Hexagonal-blueviolet)](ARCHITECTURE.md)

An enterprise-grade, single-entry-point machine learning and clinical decision support system for forecasting pharmaceutical stockout risk and delivery delays in global healthcare supply chains (PEPFAR / USAID / SCMS).

---

## 1. Quick Start & Installation

### 1.1 Prerequisites
- Python 3.11+ or 3.12+ (tested on Python 3.12.10 and Python 3.14 on Windows 11)
- Optional: NVIDIA CUDA-capable GPU (NVIDIA RTX 4050 / RTX 30-series / A100 / T4) for accelerated training and GBDT GPU histogram trees

### 1.2 Environment Setup
```bash
# Clone the repository and navigate to Implementation directory
cd Implementation

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 1.3 GPU Configuration & Automated CPU Fallback
Hardware acceleration is managed dynamically by `DeviceManager`. If a dedicated CUDA device is detected, PyTorch, XGBoost (`device="cuda"`), and CatBoost (`task_type="GPU"`) run on GPU. If CUDA is unavailable or memory allocation fails, execution safely falls back to multi-core CPU without crash.
```bash
# Enforce GPU execution (aborts loudly if no CUDA device is present)
python main.py train --require-gpu

# Force multi-core CPU execution
python main.py evaluate --device cpu
```

---

## 2. Single Point of Entry: `main.py` CLI

All pipeline workflows, statistical evaluations, serving microservices, and user dashboards are executed via `main.py`.

```bash
python main.py <subcommand> [--config config.yaml] [--device {auto,cuda,cpu}] [--require-gpu]
```

### 2.1 Complete Command Reference Table

| Subcommand | Description | Primary Output Artifact |
| :--- | :--- | :--- |
| `python main.py clean` | Ingests raw data and logs row-count waterfall accounting. | `reports/data_waterfall.json` |
| `python main.py train` | Trains 7 models across 5 random seeds (AMP, AdamW, early stopping). | `models/*_best.pt`, `models/*.pkl` |
| `python main.py calibrate` | Fits Platt & Isotonic calibration on 2014 validation data, computing $\tau^*$. | `reports/calibration_curves.png` |
| `python main.py evaluate` | Computes 1000-sample bootstrap CIs, DeLong & McNemar statistical tests. | `reports/model_comparison.csv` |
| `python main.py fairness` | Evaluates equity tiers (Tier 1/2/3) and applies threshold mitigation. | `reports/fairness_audit.json` |
| `python main.py ablation` | Runs 5-seed feature, loss, and threshold ablation study. | `reports/ablation_study.csv` |
| `python main.py explain` | Generates global/local TreeSHAP and sequence attention heatmaps. | `reports/figures/shap_summary.png` |
| `python main.py decide` | Audits Three-Way Decision Policy (Safe, Boundary, Alert) with bootstrap CIs. | CLI audit log & holdout report |
| `python main.py select` | Selects champion models using validation-only MAUT grid scoring. | `artifacts/champion.json` |
| `python main.py export` | Exports champions to ONNX and TorchScript with numerical parity validation. | `artifacts/manifest.json` |
| `python main.py benchmark` | Runs serving QoS micro-benchmarks across concurrency levels. | `reports/qos_benchmark.json` |
| `python main.py drift` | Simulates quarterly prospective score drift (PSI and KS tests). | `reports/drift/quarterly_drift_simulation.csv` |
| `python main.py report` | Regenerates comprehensive research report markdown document. | `reports/RESULTS.md` |
| `python main.py serve` | Launches production-ready FastAPI REST microservice. | Serving on `http://0.0.0.0:8000` |
| `python main.py ui` | Launches multipage Streamlit decision dashboard. | Dashboard on `http://localhost:8501` |
| `python main.py all` | Runs complete end-to-end pipeline sequentially. | Complete research outputs |

---

## 3. Honest Empirical Results Summary

Evaluated on the prospective holdout partition ($N=1,017$ shipments, $120$ positive delay events, $11.8\%$ prevalence):

### 3.1 Model Comparison: Gradient Boosting vs. Neural Networks

| Model Architecture | Model Family | PR-AUC (Holdout) | ROC-AUC (Holdout) | Recall @ Top 10% Alert Budget | Recall @ Top 20% Alert Budget | Recall @ Top 30% Alert Budget |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Classical / GBDT | **0.3599** | 0.7861 | 28.3% | 55.8% | 73.3% |
| **RandomForest** | Classical / GBDT | 0.3353 | **0.7949** | **30.8%** | **57.5%** | **76.7%** |
| **DeepTabularMLP** | Deep Neural Network | 0.2606 | 0.7694 | 25.8% | 50.8% | 70.8% |
| **Tabular1DCNN** | Deep Neural Network | 0.2790 | 0.7576 | 27.5% | 50.0% | 68.3% |
| **BiLSTM** | Deep Neural Network | 0.2505 | 0.7494 | 24.2% | 44.2% | 66.7% |
| **EnhancedHybridTransformer** | Deep Neural Network | 0.2406 | 0.7423 | 21.7% | 50.0% | 70.0% |
| **Deep Ensemble** | Neural Ensemble | 0.2511 | 0.7598 | 20.8% | 43.3% | 70.0% |
| **LogisticRegression** | Linear Baseline | 0.1806 | 0.6504 | 20.0% | 32.5% | 45.8% |

> **Key Finding:** On tabular pharmaceutical supply chain data, tree-based models (**RandomForest** and **XGBoost**) match or slightly outperform deep sequence architectures in precision-recall efficiency. Deep models require strong inductive bias (Borderline-SMOTE, Focal Loss, AdamW) to remain competitive.

### 3.2 Three-Way Decision (3WD) Policy Results

| Operating Policy | Rule & Constraints | Alert Region ($N$, %) | Safe Region ($N$, %) | Boundary Region ($N$, %) | Delay Recall | Safe Region NPV | Alert Precision | Lift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy A** *(Capacity-Constrained)* | Alert rate capped at $\le 30.0\%$ | $N=298$ (29.3%) | $N=428$ (42.1%) | $N=291$ (28.6%) | **68.3%** [60.5%–77.1%] | **97.2%** [95.6%–98.6%] | 27.5% | **2.33x** |
| **Policy B** *(Target-Constrained)* | Val Recall $\ge 85\%$, Val NPV $\ge 95\%$ | $N=351$ (34.5%) | $N=643$ (63.2%) | $N=23$ (2.3%)\* | **75.8%** [68.3%–83.2%] | **96.1%** [94.6%–97.5%] | 25.9% | **2.20x** |

*\*Note: Policy B's Boundary region contains fewer than 50 shipments ($N=23 < 50$), meaning statistical estimates in that region are unreliable.*

### 3.3 Serving Quality of Service (QoS) & Infrastructure SLA
- **Median Inference Latency (P50):** `11.93 ms` (FastAPI + single-flight cache)
- **Tail Latency (P95 SLA):** `15.82 ms`
- **Memory Footprint:** `< 120 MB RSS`
- **Cache Hit Acceleration:** Latency drops from ~12 ms to **0.35 ms** on repeated queries.

---

## 4. Academic Limitations & Transparency Disclosures

1. **Contaminated Test Set Disclosure:** The prospective holdout partition ($N=1,017$ consignments from 2015+) was observed during earlier research phases. Consequently, this test split is not a pristine holdout. Metrics reported represent confirmatory audit benchmarks.
2. **Constrained Positive Sample Size ($N=120$):** There are exactly 120 positive delay events in the prospective test split. 95% bootstrap confidence intervals for differences between top models often span zero, indicating fine differences are statistically indistinguishable.
3. **Post-Hoc Policy Formulation:** Threshold selection rules were formalized after exploratory analysis of validation distributions; they do not represent pre-registered clinical trial thresholds.
4. **Validation Budget Cap Exceedance:** Policy A achieved a 31.9% alert rate on validation data before settling at 29.3% on prospective test data, slightly exceeding its nominal 30% cap during tuning.
5. **Delivery-Time Regressor Exclusion:** Continuous delivery time prediction yielded negative $R^2$ ($-0.063$) once post-outcome date columns were removed. It is excluded from production operations.

---

## 5. Software Architecture & Verification

The codebase strictly adheres to **Clean Architecture** (see [ARCHITECTURE.md](ARCHITECTURE.md) for Mermaid diagrams and layer definitions). Automated boundary tests enforce that domain decision policies have zero imports from web frameworks or databases.

```bash
# Run complete test suite (46 passed tests in < 30 seconds)
pytest tests/ -v

# Run Clean Architecture boundary gate
pytest tests/test_clean_architecture.py -v
```
