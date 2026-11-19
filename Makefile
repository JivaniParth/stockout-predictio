# Makefile for End-to-End Cloud-Native HSCM Research Pipeline
.PHONY: help clean label split features train calibrate evaluate fairness ablation explain select export benchmark report all test lint

PYTHON ?= python

help:
	@echo "Available Research Pipeline Stages:"
	@echo "  make clean      - Clean raw anomalies and write row-count waterfall"
	@echo "  make label      - Explicitly compute late delivery target label"
	@echo "  make split      - Execute chronological 3-stage forward partition"
	@echo "  make features   - Run feature engineering and leakage audit"
	@echo "  make train      - Train all 7 models across 5 random seeds"
	@echo "  make calibrate  - Fit Platt & Isotonic calibration, recompute tau*"
	@echo "  make evaluate   - Compute bootstrap CIs, DeLong & McNemar tests"
	@echo "  make fairness   - Evaluate equity tiers, tune mitigation thresholds"
	@echo "  make ablation   - Run feature, loss, and threshold ablation study"
	@echo "  make explain    - Generate global/local SHAP & attention visualizations"
	@echo "  make select     - Execute MAUT scoring and name accuracy/edge champions"
	@echo "  make export     - Export champion to ONNX/TorchScript with parity check"
	@echo "  make benchmark  - Run live multi-concurrency serving QoS profiler"
	@echo "  make report     - Auto-generate comprehensive reports/RESULTS.md"
	@echo "  make all        - Run entire end-to-end pipeline sequentially"
	@echo "  make test       - Run complete pytest test suite (< 2 min)"

clean:
	$(PYTHON) master_pipeline.py --stage clean

label:
	$(PYTHON) master_pipeline.py --stage label

split:
	$(PYTHON) master_pipeline.py --stage split

features:
	$(PYTHON) master_pipeline.py --stage features

train:
	$(PYTHON) master_pipeline.py --stage train

calibrate:
	$(PYTHON) master_pipeline.py --stage calibrate

evaluate:
	$(PYTHON) master_pipeline.py --stage evaluate

fairness:
	$(PYTHON) master_pipeline.py --stage fairness

ablation:
	$(PYTHON) master_pipeline.py --stage ablation

explain:
	$(PYTHON) master_pipeline.py --stage explain

select:
	$(PYTHON) master_pipeline.py --stage select

export:
	$(PYTHON) master_pipeline.py --stage export

benchmark:
	$(PYTHON) master_pipeline.py --stage benchmark

report:
	$(PYTHON) master_pipeline.py --stage report

all:
	$(PYTHON) master_pipeline.py --stage all

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m py_compile src/data/*.py
	$(PYTHON) -m py_compile src/models/*.py
	$(PYTHON) -m py_compile src/evaluation/*.py
	$(PYTHON) -m py_compile src/deployment/*.py
