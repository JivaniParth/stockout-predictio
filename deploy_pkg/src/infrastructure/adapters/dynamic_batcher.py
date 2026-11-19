"""
Infrastructure Adapter: Dynamic Micro-Batching Inference Engine.
Solves the Latency vs. Throughput dilemma in production ML serving.
Aggregates concurrent single prediction requests into micro-batches,
saturating CPU/GPU vector units while bounding latency via condition variables.
"""

from concurrent.futures import Future
import logging
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from src.application.ports.predictor import IPredictor
from src.application.ports.transformer import IFeatureTransformer
from src.domain.entities import PredictionPrognosis, ShipmentEntity
from src.domain.policies import HealthEquityPolicy, RiskClassificationPolicy

logger = logging.getLogger("hscm.infrastructure.dynamic_batcher")


class DynamicBatchingEngine:
    """
    Adaptive Dynamic Micro-Batcher.
    Collects individual requests arriving within a short latency window (max_latency_ms)
    or up to max_batch_size, executing a single vectorized forward pass.
    """

    def __init__(
        self,
        predictor: IPredictor,
        transformer: IFeatureTransformer,
        max_batch_size: int = 32,
        max_latency_ms: float = 8.0,
    ):
        self._predictor = predictor
        self._transformer = transformer
        self._max_batch_size = max_batch_size
        self._max_latency_sec = max_latency_ms / 1000.0

        self._queue: List[Tuple[ShipmentEntity, float, Future]] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = True

        # Statistics
        self._total_batches = 0
        self._total_samples = 0
        self._stats_lock = threading.Lock()

        # Dedicated low-latency worker thread
        self._worker_thread = threading.Thread(
            target=self._batch_processing_loop,
            daemon=True,
            name="dynamic-micro-batcher",
        )
        self._worker_thread.start()

    def predict(
        self, shipment: ShipmentEntity, custom_threshold: Optional[float] = None, timeout: float = 5.0
    ) -> PredictionPrognosis:
        """
        Enqueues single shipment for micro-batch execution and awaits vectorized resolution.
        """
        future: Future = Future()
        thresh = custom_threshold if custom_threshold is not None else self._predictor.optimal_threshold

        with self._condition:
            self._queue.append((shipment, thresh, future))
            # If batch full, signal worker immediately
            if len(self._queue) >= self._max_batch_size:
                self._condition.notify()
            elif len(self._queue) == 1:
                # First item in new window, notify to start timer
                self._condition.notify()

        return future.result(timeout=timeout)

    def _batch_processing_loop(self) -> None:
        """Background thread executing vectorized forward passes."""
        while self._running:
            with self._condition:
                while self._running and len(self._queue) == 0:
                    self._condition.wait()

                if not self._running:
                    break

                # Wait for batch window to fill or timeout to expire
                deadline = time.time() + self._max_latency_sec
                while len(self._queue) < self._max_batch_size:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)

                # Pop current micro-batch
                current_batch = self._queue[: self._max_batch_size]
                self._queue = self._queue[self._max_batch_size :]

            if not current_batch:
                continue

            # Process micro-batch without holding the lock
            self._execute_micro_batch(current_batch)

    def _execute_micro_batch(
        self, batch: List[Tuple[ShipmentEntity, float, Future]]
    ) -> None:
        batch_size = len(batch)
        t0 = time.perf_counter()

        with self._stats_lock:
            self._total_batches += 1
            self._total_samples += batch_size

        try:
            entities = [item[0] for item in batch]
            thresholds = [item[1] for item in batch]
            futures = [item[2] for item in batch]

            # 1. High-speed vectorized tabular feature transformation
            features = self._transformer.transform_entities(entities)

            # 2. Single forward pass for entire micro-batch
            probabilities = self._predictor.predict_proba(features)

            t1 = time.perf_counter()
            total_lat_ms = round((t1 - t0) * 1000.0, 3)
            per_sample_lat_ms = round(total_lat_ms / batch_size, 3)

            # 3. Resolve each future with its domain prognosis
            for entity, prob_val, thresh, fut in zip(entities, probabilities, thresholds, futures):
                prob = float(prob_val)
                flag = 1 if prob >= thresh else 0
                risk_tier = RiskClassificationPolicy.classify(prob)
                country_tier = HealthEquityPolicy.classify_country(entity.country)

                prognosis = PredictionPrognosis(
                    late_delivery_risk=flag,
                    risk_probability=round(prob, 4),
                    risk_tier=risk_tier,
                    country_tier=country_tier,
                    qos_latency_ms=per_sample_lat_ms,
                )
                fut.set_result(prognosis)

        except Exception as exc:
            logger.error("Error executing dynamic micro-batch: %s", exc, exc_info=True)
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)

    def get_stats(self) -> dict:
        with self._stats_lock:
            avg_batch = (
                round(self._total_samples / self._total_batches, 2)
                if self._total_batches > 0
                else 0.0
            )
            return {
                "total_batches_executed": self._total_batches,
                "total_samples_processed": self._total_samples,
                "average_batch_size": avg_batch,
                "max_batch_size": self._max_batch_size,
                "window_ms": self._max_latency_sec * 1000.0,
            }

    def shutdown(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._worker_thread.join(timeout=2.0)
