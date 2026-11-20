"""
Verification and Benchmark Suite for Implementation/src/data/fast_dataloader.py.
Tests all requirements and benchmarks loading throughput on the USAID SCMS Delivery History dataset.
"""

import sys
import os
import time
from pathlib import Path

# Add Implementation to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch

from src.data.fast_dataloader import (
    FastTabularDataset,
    FastMixupCollate,
    FastBatchIterator,
    StratifiedBatchSampler,
    ChronologicalExpandingFoldSplitter,
    create_fast_dataloaders,
    create_chronological_fold_dataloaders,
    benchmark_dataloader_throughput,
)
from src.data.preprocessor import prepare_data_splits


def test_fast_tabular_dataset():
    print("\n--- Test 1: FastTabularDataset Instantiation & Dtypes ---")
    N, D_cont, D_cat = 500, 10, 5
    x_cont_raw = np.random.randn(N, D_cont).astype(np.float64)
    x_cat_raw = np.random.randint(0, 15, size=(N, D_cat)).astype(np.int32)
    y_raw = np.random.randint(0, 2, size=N)

    ds = FastTabularDataset(x_cont=x_cont_raw, x_cat=x_cat_raw, y=y_raw)
    assert ds.x_cont.dtype == torch.float32, f"Expected float32, got {ds.x_cont.dtype}"
    assert ds.x_cat.dtype == torch.long, f"Expected long, got {ds.x_cat.dtype}"
    assert ds.y.dtype == torch.float32, f"Expected float32, got {ds.y.dtype}"
    assert ds.x_cont.shape == (N, D_cont)
    assert ds.x_cat.shape == (N, D_cat)
    assert ds.y.shape == (N, 1)
    assert len(ds) == N

    # Item indexing
    c, cat, y = ds[0]
    assert c.dtype == torch.float32
    assert cat.dtype == torch.long
    assert y.dtype == torch.float32

    # Vectorized slice indexing
    c_slice, cat_slice, y_slice = ds[0:10]
    assert c_slice.shape == (10, D_cont)
    assert cat_slice.shape == (10, D_cat)
    assert y_slice.shape == (10, 1)

    # Combined X instantiation
    X_combined = np.hstack([x_cont_raw, x_cat_raw])
    cat_idx = list(range(D_cont, D_cont + D_cat))
    ds_comb = FastTabularDataset.from_combined(X_combined, y_raw, cat_indices=cat_idx)
    assert ds_comb.num_continuous == D_cont
    assert ds_comb.num_categorical == D_cat
    assert ds_comb.total_features == D_cont + D_cat
    print("[PASS] FastTabularDataset verified: types, shapes, and indexing are correct.")


def test_create_fast_dataloaders():
    print("\n--- Test 2: create_fast_dataloaders & CUDA Pinned Memory Detection ---")
    N_tr, N_val = 1000, 200
    D = 12
    cat_idx = [8, 9, 10, 11]

    X_tr = np.random.randn(N_tr, D)
    y_tr = np.random.randint(0, 2, size=N_tr)
    X_val = np.random.randn(N_val, D)
    y_val = np.random.randint(0, 2, size=N_val)

    # Test automatic pin_memory detection
    cuda_avail = torch.cuda.is_available()
    train_loader, val_loader = create_fast_dataloaders(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        cat_indices=cat_idx,
        batch_size=128,
        pin_memory=True,  # User requests True
    )

    # DataLoader should safely match cuda availability
    assert train_loader.pin_memory == cuda_avail, (
        f"Expected pin_memory == {cuda_avail}, got {train_loader.pin_memory}"
    )
    assert val_loader.pin_memory == cuda_avail, (
        f"Expected val pin_memory == {cuda_avail}, got {val_loader.pin_memory}"
    )

    # Verify iteration
    for bx_c, bx_cat, by in train_loader:
        assert bx_c.dtype == torch.float32
        assert bx_cat.dtype == torch.long
        assert by.dtype == torch.float32
        assert bx_c.shape[1] == 8
        assert bx_cat.shape[1] == 4
        assert by.shape[1] == 1
        break

    for bx_c, bx_cat, by in val_loader:
        assert bx_c.dtype == torch.float32
        assert bx_cat.dtype == torch.long
        assert by.dtype == torch.float32
        break

    print(f"[PASS] create_fast_dataloaders verified: CUDA detection={cuda_avail}, pin_memory={train_loader.pin_memory}.")


def test_mixup_collation():
    print("\n--- Test 3: Tabular Mixup Collation ---")
    N = 256
    x_cont = torch.ones((N, 4), dtype=torch.float32)
    x_cat = torch.zeros((N, 4), dtype=torch.long)
    y = torch.zeros((N, 1), dtype=torch.float32)

    # Set half to 2.0 / 1
    x_cont[N // 2 :] = 5.0
    x_cat[N // 2 :] = 10
    y[N // 2 :] = 1.0

    ds = FastTabularDataset(x_cont=x_cont, x_cat=x_cat, y=y)
    collate = FastMixupCollate(mixup_alpha=0.5, mixup_prob=1.0, categorical_mix_strategy="bernoulli")
    loader = torch.utils.data.DataLoader(ds, batch_size=N, collate_fn=collate, shuffle=False)

    batch_c, batch_cat, batch_y = next(iter(loader))

    # Verify targets are soft values between 0.0 and 1.0
    has_mixed_targets = torch.any((batch_y > 0.0) & (batch_y < 1.0))
    print(f"Mixup targets contain soft values: {has_mixed_targets.item()} (Min: {batch_y.min():.3f}, Max: {batch_y.max():.3f})")
    assert has_mixed_targets, "Mixup should produce continuous/soft target probabilities!"

    # Verify categoricals remain valid discrete integers (tokens)
    assert batch_cat.dtype == torch.long, "Categorical tokens must remain int64/long after Mixup!"
    assert torch.all((batch_cat == 0) | (batch_cat == 10)), "Bernoulli categorical tokens must be from valid indices!"
    print("[PASS] Tabular Mixup verified: continuous features linearly interpolated, categoricals safely masked, soft targets produced.")


def test_stratified_batch_sampler():
    print("\n--- Test 4: StratifiedBatchSampler ---")
    # Imbalanced dataset: 10% positive (similar to SCMS stockout rate)
    N = 1000
    y = np.zeros(N, dtype=np.float32)
    y[:100] = 1.0  # 10% positive
    np.random.shuffle(y)

    sampler = StratifiedBatchSampler(y=y, batch_size=100, shuffle=True)
    batch_indices = next(iter(sampler))

    pos_in_batch = (y[batch_indices] == 1.0).sum()
    print(f"Positive samples in batch of 100: {pos_in_batch} (Target: ~10)")
    assert 8 <= pos_in_batch <= 12, f"Stratified ratio deviated too far: {pos_in_batch}"
    print("[PASS] StratifiedBatchSampler preserves class distribution across batches.")


def test_chronological_expanding_fold():
    print("\n--- Test 5: Chronological Expanding Fold Splitter ---")
    years = np.array([2008] * 100 + [2009] * 100 + [2010] * 100 + [2011] * 100 + [2012] * 100)
    splitter = ChronologicalExpandingFoldSplitter(years=years, min_train_years=3)

    folds = list(splitter.split())
    assert len(folds) == 2, f"Expected 2 folds, got {len(folds)}"
    fold_0 = folds[0]
    print(f"Fold 0: Train years <= {fold_0[2]} (N={len(fold_0[0])}), Val year == {fold_0[3]} (N={len(fold_0[1])})")
    assert fold_0[2] == 2010 and fold_0[3] == 2011
    fold_1 = folds[1]
    print(f"Fold 1: Train years <= {fold_1[2]} (N={len(fold_1[0])}), Val year == {fold_1[3]} (N={len(fold_1[1])})")
    assert fold_1[2] == 2011 and fold_1[3] == 2012
    print("[PASS] Chronological Expanding Fold Splitter verified.")


def benchmark_scms_pipeline():
    print("\n" + "=" * 78)
    print("USAID SCMS DELIVERY HISTORY DATASET THROUGHPUT BENCHMARK")
    print("=" * 78)

    # Load and preprocess SCMS dataset
    print("Loading SCMS Delivery History Dataset...")
    X_tr, y_tr, X_val, y_val, X_te, y_te, preprocessor, test_df = prepare_data_splits()
    num_train = len(X_tr)
    num_val = len(X_val)
    print(f"Loaded SCMS Train: {num_train} rows x {X_tr.shape[1]} cols (Positives: {np.sum(y_tr):,} / {np.mean(y_tr)*100:.2f}%)")
    print(f"Loaded SCMS Val:   {num_val} rows x {X_val.shape[1]} cols (Positives: {np.sum(y_val):,} / {np.mean(y_val)*100:.2f}%)")

    # Benchmarking different configurations
    results = []

    # 1. Standard DataLoader with FastTabularDataset (no mixup)
    tr_loader, _ = create_fast_dataloaders(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        cat_indices=[],
        batch_size=256,
        pin_memory=True,
        mixup_alpha=0.0,
    )
    res_std = benchmark_dataloader_throughput(tr_loader, num_train, num_epochs=10, name="FastTabular DataLoader (Standard)")
    results.append(res_std)

    # 2. FastMixup DataLoader (mixup_alpha=0.2)
    tr_mixup_loader, _ = create_fast_dataloaders(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        cat_indices=[],
        batch_size=256,
        pin_memory=True,
        mixup_alpha=0.2,
        mixup_prob=1.0,
    )
    res_mixup = benchmark_dataloader_throughput(tr_mixup_loader, num_train, num_epochs=10, name="FastMixup DataLoader (alpha=0.2)")
    results.append(res_mixup)

    # 3. Stratified Batch DataLoader
    tr_strat_loader, _ = create_fast_dataloaders(
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        cat_indices=[],
        batch_size=256,
        pin_memory=True,
        use_stratified_sampler=True,
    )
    res_strat = benchmark_dataloader_throughput(tr_strat_loader, num_train, num_epochs=10, name="Stratified Batch DataLoader")
    results.append(res_strat)

    # 4. Ultra-Fast FastBatchIterator (Direct Contiguous Slicing)
    fast_iter = FastBatchIterator(
        x_cont=torch.from_numpy(X_tr.astype(np.float32)),
        x_cat=torch.empty((num_train, 0), dtype=torch.long),
        y=torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1),
        batch_size=256,
        shuffle=True,
        mixup_alpha=0.0,
    )
    res_fast = benchmark_dataloader_throughput(fast_iter, num_train, num_epochs=20, name="FastBatchIterator (Zero-Overhead)")
    results.append(res_fast)

    # 5. FastBatchIterator with Mixup
    fast_iter_mixup = FastBatchIterator(
        x_cont=torch.from_numpy(X_tr.astype(np.float32)),
        x_cat=torch.empty((num_train, 0), dtype=torch.long),
        y=torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1),
        batch_size=256,
        shuffle=True,
        mixup_alpha=0.2,
        mixup_prob=1.0,
    )
    res_fast_mixup = benchmark_dataloader_throughput(fast_iter_mixup, num_train, num_epochs=20, name="FastBatchIterator + Mixup (alpha=0.2)")
    results.append(res_fast_mixup)

    print("\n" + "=" * 90)
    print(f"{'Pipeline Configuration':<42} | {'Throughput (samples/s)':<22} | {'Epoch Time':<12} | {'Batch Time'}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<42} | {r['samples_per_sec']:>18,.0f} samples/s | {r['ms_per_epoch']:>8.2f} ms | {r['us_per_batch']:>8.1f} us")
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    test_fast_tabular_dataset()
    test_create_fast_dataloaders()
    test_mixup_collation()
    test_stratified_batch_sampler()
    test_chronological_expanding_fold()
    benchmark_scms_pipeline()
    print("ALL TESTS AND BENCHMARKS COMPLETED SUCCESSFULLY!")
