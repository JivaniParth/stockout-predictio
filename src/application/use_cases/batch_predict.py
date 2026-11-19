"""
Layer 2 Use Case: High-Throughput Batch Shipment Prognostication.
Orchestrates vectorization, batch model execution, throughput profiling,
and health equity stratification across multi-record payloads.
"""

import time
from typing import List, Tuple
from src.domain.entities import ShipmentEntity, PredictionPrognosis
from src.domain.policies import HealthEquityPolicy, RiskClassificationPolicy
from src.application.ports.predictor import IPredictor
from src.application.ports.transformer import IFeatureTransformer


class BatchPredictShipmentsUseCase:
    """Orchestrates batch-level stockout prediction workflow."""

    def __init__(
        self,
        predictor: IPredictor,
        transformer: IFeatureTransformer,
    ):
        self.predictor = predictor
        self.transformer = transformer

    def execute(
        self,
        shipments: List[ShipmentEntity],
        custom_threshold: float = None,
    ) -> Tuple[List[PredictionPrognosis], float, float]:
        """
        Executes batch inference.
        Returns (predictions_list, total_batch_latency_ms, throughput_samples_per_sec).
        """
        if not shipments:
            return [], 0.0, 0.0

        t0 = time.perf_counter()

        # 1. High-speed tabular vectorization
        features = self.transformer.transform_entities(shipments)

        # 2. Vectorized forward pass
        probabilities = self.predictor.predict_proba(features)

        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000.0, 3)
        throughput = (len(shipments) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        threshold = custom_threshold if custom_threshold is not None else self.predictor.optimal_threshold
        per_item_latency_ms = round(total_latency_ms / len(shipments), 3)

        results: List[PredictionPrognosis] = []
        for s, p_val in zip(shipments, probabilities):
            prob = float(p_val)
            flag = 1 if prob >= threshold else 0
            risk_tier = RiskClassificationPolicy.classify(prob)
            country_tier = HealthEquityPolicy.classify_country(s.country)

            results.append(
                PredictionPrognosis(
                    late_delivery_risk=flag,
                    risk_probability=round(prob, 4),
                    risk_tier=risk_tier,
                    country_tier=country_tier,
                    qos_latency_ms=per_item_latency_ms,
                )
            )

        return results, total_latency_ms, round(throughput, 2)
