# Makefile for End-to-End Cloud-Native HSCM Research Pipeline
.PHONY: help clean label split features train calibrate evaluate fairness ablation explain decide select export benchmark drift report serve ui all test lint

PYTHON ?= python

help:
	@echo "Available Research Pipeline Stages (Orchestrated via main.py):"
	@echo "  make clean      - Clean raw anomalies and write row-count waterfall"
	@echo "  make train      - Train all 7 models across 5 random seeds"
	@echo "  make calibrate  - Fit Platt & Isotonic calibration, recompute tau*"
	@echo "  make evaluate   - Compute bootstrap CIs, DeLong & McNemar tests"
	@echo "  make fairness   - Evaluate equity tiers, tune mitigation thresholds"
	@echo "  make ablation   - Run feature, loss, and threshold ablation study"
	@echo "  make explain    - Generate global/local SHAP & attention visualizations"
	@echo "  make decide     - Three-Way Decision Policy (Safe, Boundary, Alert) audit"
	@echo "  make select     - Execute MAUT scoring and name accuracy/edge champions"
	@echo "  make export     - Export champion to ONNX/TorchScript with parity check"
	@echo "  make benchmark  - Run live multi-concurrency serving QoS profiler"
	@echo "  make drift      - Run quarterly prospective drift simulation"
	@echo "  make report     - Auto-generate comprehensive reports/RESULTS.md"
	@echo "  make serve      - Launch production-grade FastAPI microservice"
	@echo "  make ui         - Launch Streamlit operational decision dashboard"
	@echo "  make all        - Run entire end-to-end pipeline sequentially"
	@echo "  make test       - Run complete pytest test suite (< 2 min)"

clean:
	$(PYTHON) main.py clean

label:
	$(PYTHON) master_pipeline.py --stage label

split:
	$(PYTHON) master_pipeline.py --stage split

features:
	$(PYTHON) master_pipeline.py --stage features

train:
	$(PYTHON) main.py train

calibrate:
	$(PYTHON) main.py calibrate

evaluate:
	$(PYTHON) main.py evaluate

fairness:
	$(PYTHON) main.py fairness

ablation:
	$(PYTHON) main.py ablation

explain:
	$(PYTHON) main.py explain

decide:
	$(PYTHON) main.py decide

select:
	$(PYTHON) main.py select

export:
	$(PYTHON) main.py export

benchmark:
	$(PYTHON) main.py benchmark

drift:
	$(PYTHON) main.py drift

report:
	$(PYTHON) main.py report

serve:
	$(PYTHON) main.py serve

ui:
	$(PYTHON) main.py ui

all:
	$(PYTHON) main.py all

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m py_compile main.py
	$(PYTHON) -m py_compile src/scms/**/*.py
