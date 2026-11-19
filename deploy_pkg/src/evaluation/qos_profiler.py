"""
Cloud QoS Profiler for Benchmarking Real-Time Inference Latency, Throughput, and Memory Footprint.
Directly addresses RQ3 and bridges the gap between ML accuracy and Cloud Quality of Service (QoS).
"""

import time
import os
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    QOS_WARMUP_RUNS,
    QOS_BENCHMARK_RUNS,
    MODELS_DIR,
)


def benchmark_model_qos(
    model: Union[nn.Module, Any],
    input_dim: int,
    model_name: str,
    device: torch.device = DEVICE,
    warmup_runs: int = QOS_WARMUP_RUNS,
    benchmark_runs: int = QOS_BENCHMARK_RUNS,
    batch_sizes: List[int] = [1, 16, 64],
) -> Dict[str, Any]:
    """
    Benchmarks single-sample latency percentiles (P50, P95, P99), batch throughput,
    and computational footprint of an ML/DL model.
    """
    is_torch = isinstance(model, nn.Module)
    sample_single = np.random.randn(1, input_dim).astype(np.float32)

    if is_torch:
        model.to(device)
        model.eval()
        x_single_tensor = torch.tensor(sample_single, device=device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x_single_tensor)

        # Single-sample latency profiling
        latencies_ms = []
        with torch.no_grad():
            for _ in range(benchmark_runs):
                t0 = time.perf_counter()
                _ = model(x_single_tensor)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        # Throughput benchmark across batch sizes
        throughput_dict = {}
        for bs in batch_sizes:
            x_batch = torch.randn(bs, input_dim, device=device)
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x_batch)

            t0 = time.perf_counter()
            runs = max(20, benchmark_runs // 5)
            with torch.no_grad():
                for _ in range(runs):
                    _ = model(x_batch)
            t1 = time.perf_counter()
            total_time = t1 - t0
            throughput = (bs * runs) / total_time
            throughput_dict[f"Throughput_BS{bs}_samples_per_sec"] = float(throughput)

    else:
        # Classical scikit-learn / XGBoost model
        num_params = 0
        trainable_params = 0

        # Warmup
        for _ in range(warmup_runs):
            _ = model.predict_proba(sample_single)

        latencies_ms = []
        for _ in range(benchmark_runs):
            t0 = time.perf_counter()
            _ = model.predict_proba(sample_single)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        throughput_dict = {}
        for bs in batch_sizes:
            x_batch = np.random.randn(bs, input_dim).astype(np.float32)
            t0 = time.perf_counter()
            runs = max(20, benchmark_runs // 5)
            for _ in range(runs):
                _ = model.predict_proba(x_batch)
            t1 = time.perf_counter()
            total_time = t1 - t0
            throughput = (bs * runs) / total_time
            throughput_dict[f"Throughput_BS{bs}_samples_per_sec"] = float(throughput)

    # Disk footprint
    model_file = None
    for ext in [".pt", ".pkl"]:
        candidate = MODELS_DIR / f"{model_name}{ext}"
        if not candidate.exists():
            candidate = MODELS_DIR / f"{model_name}_best{ext}"
        if candidate.exists():
            model_file = candidate
            break

    disk_size_mb = (os.path.getsize(model_file) / (1024 * 1024)) if model_file else 0.0

    qos_results = {
        "Model": model_name,
        "Total_Parameters": int(num_params),
        "Trainable_Parameters": int(trainable_params),
        "Disk_Size_MB": round(disk_size_mb, 3),
        "Latency_Mean_ms": round(float(np.mean(latencies_ms)), 4),
        "Latency_P50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "Latency_P95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "Latency_P99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
    }
    qos_results.update({k: round(v, 2) for k, v in throughput_dict.items()})

    return qos_results
