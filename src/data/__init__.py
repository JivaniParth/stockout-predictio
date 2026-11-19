"""
Data module for the Pharmaceutical Supply Chain Stockout Prediction project.
"""

from src.data.dataset import HSCMDataset, get_dataloaders
from src.data.preprocessor import HSCMPreprocessor, prepare_data_splits
from src.data.cleaner import clean_dataset_with_waterfall
from src.data.leakage_guard import (
    assert_no_post_outcome_features,
    assert_no_high_correlation_leakage,
    LeakageGuardError,
)
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

__all__ = [
    "HSCMDataset",
    "get_dataloaders",
    "HSCMPreprocessor",
    "prepare_data_splits",
    "clean_dataset_with_waterfall",
    "assert_no_post_outcome_features",
    "assert_no_high_correlation_leakage",
    "LeakageGuardError",
    "FastTabularDataset",
    "FastMixupCollate",
    "FastBatchIterator",
    "StratifiedBatchSampler",
    "ChronologicalExpandingFoldSplitter",
    "create_fast_dataloaders",
    "create_chronological_fold_dataloaders",
    "benchmark_dataloader_throughput",
]
