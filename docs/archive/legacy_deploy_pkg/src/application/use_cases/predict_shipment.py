"""
Layer 2 Use Case: Single Shipment Stockout Prognostication.
Orchestrates domain policies, feature transformation, model execution,
and QoS telemetry recording.
"""

import time
from src.domain.entities import ShipmentEntity, PredictionPrognosis
from src.domain.policies import HealthEquityPolicy, RiskClassificationPolicy
from src.application.ports.predictor import IPredictor
from src.application.ports.transformer import IFeatureTransformer
from src.application.ports.telemetry import ITelemetryRepository


class PredictShipmentUseCase:
    """Orchestrates single-shipment stockout prediction workflow."""

    def __init__(
        self,
        predictor: IPredictor,
        transformer: IFeatureTransformer,
        telemetry_repo: ITelemetryRepository,
    ):
        self.predictor = predictor
        self.transformer = transformer
        self.telemetry_repo = telemetry_repo

    def execute(self, shipment: ShipmentEntity, custom_threshold: float = None) -> PredictionPrognosis:
        t0 = time.perf_counter()

        # 1. Transform domain entity to model-ready feature vector
        features = self.transformer.transform_entities([shipment])

        # 2. Execute Model Inference via Port
        probabilities = self.predictor.predict_proba(features)
        prob = float(probabilities[0])

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 3)

        # 3. Record QoS Telemetry via Port
        self.telemetry_repo.record_latency(latency_ms)

        # 4. Apply Business Rules & Decision Threshold
        threshold = custom_threshold if custom_threshold is not None else self.predictor.optimal_threshold
        risk_flag = 1 if prob >= threshold else 0

        # 5. Apply Domain Policies
        risk_tier = RiskClassificationPolicy.classify(prob)
        country_tier = HealthEquityPolicy.classify_country(shipment.country)

        return PredictionPrognosis(
            late_delivery_risk=risk_flag,
            risk_probability=round(prob, 4),
            risk_tier=risk_tier,
            country_tier=country_tier,
            qos_latency_ms=latency_ms,
        )
