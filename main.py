"""
HSCM Single Entry Point CLI Orchestrator.
Unified Command Line Interface for training, calibration, evaluation,
three-way decision policy, serving, monitoring, and dashboard orchestration.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scms.cli")


def setup_device(device_arg: str = "auto", require_gpu: bool = False) -> torch.device:
    """Enforces device allocation rules, GPU requirements, and logs active device."""
    has_gpu = torch.cuda.is_available()

    if require_gpu and not has_gpu:
        logger.error(
            "FATAL: --require-gpu was specified, but no CUDA-capable device was detected on the system."
        )
        sys.exit(1)

    if device_arg == "cuda":
        if not has_gpu:
            if require_gpu:
                logger.error("FATAL: CUDA requested with --require-gpu but unavailable.")
                sys.exit(1)
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            active_device = torch.device("cpu")
        else:
            active_device = torch.device("cuda")
    elif device_arg == "cpu":
        active_device = torch.device("cpu")
    else:  # "auto"
        active_device = torch.device("cuda" if has_gpu else "cpu")

    device_name = (
        torch.cuda.get_device_name(0) if active_device.type == "cuda" else "Host CPU Core Architecture"
    )
    logger.info(f"[Hardware Diagnostics] Active Compute Device: {active_device.type.upper()} ({device_name})")
    if active_device.type == "cuda":
        logger.info(
            f"[Hardware Diagnostics] CUDA Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    return active_device


def get_pipeline(config_path: str, device: torch.device):
    """Instantiates MasterPipeline with configured config and active device."""
    from master_pipeline import MasterPipeline

    pipeline = MasterPipeline(config_path=config_path)
    pipeline.device = device
    return pipeline


def cmd_clean(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_clean()
    logger.info("[+] Stage clean completed successfully.")


def cmd_train(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_clean()
    pipeline.stage_label()
    pipeline.stage_split()
    pipeline.stage_features()
    pipeline.stage_train()
    logger.info("[+] Stage train completed successfully.")


def cmd_calibrate(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_calibrate()
    logger.info("[+] Stage calibrate completed successfully.")


def cmd_evaluate(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_evaluate()
    logger.info("[+] Stage evaluate completed successfully.")


def cmd_fairness(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_fairness()
    logger.info("[+] Stage fairness completed successfully.")


def cmd_ablation(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_ablation()
    logger.info("[+] Stage ablation completed successfully.")


def cmd_explain(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_explain()
    logger.info("[+] Stage explain completed successfully.")


def cmd_decide(args):
    """Executes Three-Way Decision Policy (Safe, Boundary, Alert) audit and evaluation."""
    import json
    setup_device(args.device, args.require_gpu)
    report_path = Path("reports/combined_path1_path2_results.json")

    if not report_path.exists():
        logger.error(f"Required report missing: {report_path}")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pol_a = data.get("policy_a_capacity_constrained", {})
    res_a = pol_a.get("prospective_results", {})
    pol_b = data.get("policy_b_target_constrained", {})
    res_b = pol_b.get("prospective_results", {})

    logger.info("=" * 70)
    logger.info("THREE-WAY DECISION POLICY TEST HOLDOUT AUDIT (N=1,017, Positives=120)")
    logger.info("=" * 70)
    logger.info(f"POLICY A (Capacity-Constrained: Alert cap <= 30%):")
    logger.info(f"  Thresholds: tau_safe={pol_a.get('tau_safe')}, tau_alert={pol_a.get('tau_alert')}")
    logger.info(
        f"  Alert Region:    N={res_a.get('n_alert')} ({res_a.get('pct_alert', 0)*100:.1f}%), "
        f"Recall={res_a.get('rec', 0)*100:.1f}% [95% CI: {res_a.get('ci', {}).get('rec', [0,0])[0]*100:.1f}% - {res_a.get('ci', {}).get('rec', [0,0])[1]*100:.1f}%], "
        f"Precision={res_a.get('prec', 0)*100:.1f}%, Lift={res_a.get('lift', 0):.2f}x"
    )
    logger.info(
        f"  Safe Region:     N={res_a.get('n_safe')} ({res_a.get('pct_safe', 0)*100:.1f}%), "
        f"NPV={res_a.get('npv', 0)*100:.1f}% [95% CI: {res_a.get('ci', {}).get('npv', [0,0])[0]*100:.1f}% - {res_a.get('ci', {}).get('npv', [0,0])[1]*100:.1f}%]"
    )
    bnd_a = res_a.get('n_bnd', 0)
    flag_a = " [WARNING: N < 50, unreliable]" if bnd_a < 50 else ""
    logger.info(
        f"  Boundary Region: N={bnd_a} ({res_a.get('pct_bnd', 0)*100:.1f}%){flag_a}"
    )
    logger.info("-" * 70)
    logger.info(f"POLICY B (Target-Constrained: Validation Recall >= 85%, NPV >= 95%):")
    logger.info(f"  Thresholds: tau_safe={pol_b.get('tau_safe')}, tau_alert={pol_b.get('tau_alert')}")
    logger.info(
        f"  Alert Region:    N={res_b.get('n_alert')} ({res_b.get('pct_alert', 0)*100:.1f}%), "
        f"Recall={res_b.get('rec', 0)*100:.1f}% [95% CI: {res_b.get('ci', {}).get('rec', [0,0])[0]*100:.1f}% - {res_b.get('ci', {}).get('rec', [0,0])[1]*100:.1f}%], "
        f"Precision={res_b.get('prec', 0)*100:.1f}%, Lift={res_b.get('lift', 0):.2f}x"
    )
    logger.info(
        f"  Safe Region:     N={res_b.get('n_safe')} ({res_b.get('pct_safe', 0)*100:.1f}%), "
        f"NPV={res_b.get('npv', 0)*100:.1f}% [95% CI: {res_b.get('ci', {}).get('npv', [0,0])[0]*100:.1f}% - {res_b.get('ci', {}).get('npv', [0,0])[1]*100:.1f}%]"
    )
    bnd_b = res_b.get('n_bnd', 0)
    flag_b = " [WARNING: Region has fewer than 50 shipments (N=23); statistical estimates are unreliable]" if bnd_b < 50 else ""
    logger.info(
        f"  Boundary Region: N={bnd_b} ({res_b.get('pct_bnd', 0)*100:.1f}%){flag_b}"
    )
    logger.info("=" * 70)


def cmd_select(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_select()
    logger.info("[+] Stage select completed successfully.")


def cmd_export(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_export()
    logger.info("[+] Stage export completed successfully.")


def cmd_benchmark(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_benchmark()
    logger.info("[+] Stage benchmark completed successfully.")


def cmd_drift(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_drift()
    logger.info("[+] Stage drift completed successfully.")


def cmd_report(args):
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_report()
    logger.info("[+] Stage report completed successfully.")


def cmd_serve(args):
    """Launches high-performance FastAPI microservice."""
    import uvicorn

    setup_device(args.device, args.require_gpu)
    host = getattr(args, "host", "0.0.0.0")
    port = int(getattr(args, "port", 8000))
    logger.info(f"Starting HSCM FastAPI microservice on http://{host}:{port} ...")
    uvicorn.run("src.scms.serving.app:app", host=host, port=port, reload=False)


def cmd_ui(args):
    """Launches Streamlit decision support dashboard."""
    setup_device(args.device, args.require_gpu)
    port = getattr(args, "port", "8501")
    ui_script = "src/scms/ui/app.py"
    if not Path(ui_script).exists():
        ui_script = "app_streamlit.py"

    logger.info(f"Starting Streamlit dashboard from {ui_script} on port {port} ...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        ui_script,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
    ]
    subprocess.run(cmd)


def cmd_all(args):
    """Executes complete research and engineering pipeline sequentially."""
    device = setup_device(args.device, args.require_gpu)
    pipeline = get_pipeline(args.config, device)
    pipeline.stage_clean()
    pipeline.stage_label()
    pipeline.stage_split()
    pipeline.stage_features()
    pipeline.stage_train()
    pipeline.stage_calibrate()
    pipeline.stage_evaluate()
    pipeline.stage_fairness()
    pipeline.stage_ablation()
    pipeline.stage_explain()
    pipeline.stage_select()
    pipeline.stage_export()
    pipeline.stage_benchmark()
    pipeline.stage_drift()
    pipeline.stage_report()
    cmd_decide(args)
    logger.info("[+] Complete end-to-end pipeline execution succeeded.")


def build_parser() -> argparse.ArgumentParser:
    """Builds comprehensive command-line parser with shared hardware & config arguments."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Scalable Cloud-Native HSCM Stockout Prediction Platform",
    )

    # Shared argument parent parser
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    shared_parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Hardware compute device to utilize: auto, cuda, or cpu (default: auto)",
    )
    shared_parser.add_argument(
        "--require-gpu",
        action="store_true",
        default=False,
        help="Enforces that GPU execution is strictly required; aborts if CUDA is unavailable.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True, help="Subcommand to execute")

    # Pipeline stages
    subparsers.add_parser("clean", parents=[shared_parser], help="Clean dataset and log waterfall accounting").set_defaults(func=cmd_clean)
    subparsers.add_parser("train", parents=[shared_parser], help="Train neural and GBDT models across 5 seeds").set_defaults(func=cmd_train)
    subparsers.add_parser("calibrate", parents=[shared_parser], help="Fit Platt and Isotonic calibration").set_defaults(func=cmd_calibrate)
    subparsers.add_parser("evaluate", parents=[shared_parser], help="Compute bootstrap CIs, DeLong & McNemar tests").set_defaults(func=cmd_evaluate)
    subparsers.add_parser("fairness", parents=[shared_parser], help="Evaluate equity tiers and mitigate disparities").set_defaults(func=cmd_fairness)
    subparsers.add_parser("ablation", parents=[shared_parser], help="Run 5-seed feature, loss, and threshold ablation").set_defaults(func=cmd_ablation)
    subparsers.add_parser("explain", parents=[shared_parser], help="Generate SHAP and attention visualizations").set_defaults(func=cmd_explain)
    subparsers.add_parser("decide", parents=[shared_parser], help="Evaluate Three-Way Decision Policy (Safe, Boundary, Alert)").set_defaults(func=cmd_decide)
    subparsers.add_parser("select", parents=[shared_parser], help="MAUT scoring and champion selection").set_defaults(func=cmd_select)
    subparsers.add_parser("export", parents=[shared_parser], help="Export champion models to ONNX and TorchScript").set_defaults(func=cmd_export)
    subparsers.add_parser("benchmark", parents=[shared_parser], help="Run live multi-concurrency serving QoS profiler").set_defaults(func=cmd_benchmark)
    subparsers.add_parser("drift", parents=[shared_parser], help="Run quarterly prospective drift simulation").set_defaults(func=cmd_drift)
    subparsers.add_parser("report", parents=[shared_parser], help="Auto-generate reports/RESULTS.md").set_defaults(func=cmd_report)
    subparsers.add_parser("all", parents=[shared_parser], help="Run complete pipeline sequentially").set_defaults(func=cmd_all)

    # Serving & UI
    serve_parser = subparsers.add_parser("serve", parents=[shared_parser], help="Launch FastAPI microservice")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host IP (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Binding port number (default: 8000)")
    serve_parser.set_defaults(func=cmd_serve)

    ui_parser = subparsers.add_parser("ui", parents=[shared_parser], help="Launch Streamlit operations dashboard")
    ui_parser.add_argument("--port", type=int, default=8501, help="Binding port number (default: 8501)")
    ui_parser.set_defaults(func=cmd_ui)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
