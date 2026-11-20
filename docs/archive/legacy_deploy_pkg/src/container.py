"""
Composition Root & Dependency Injection Container.
Wires together concrete infrastructure adapters with application layer use cases.
Decouples framework, storage, caching, persistence, and ML runtime dependencies from business logic.
"""

import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional
import torch

from src.config import MODELS_DIR, DEVICE
from src.data.preprocessor import HSCMPreprocessor
from src.models.dnn import DeepTabularMLP
from src.models.lstm import BiLSTMStockoutPredictor
from src.models.hybrid_transformer import HybridLSTMTransformer
from src.models.cnn import Tabular1DCNN

from src.application.ports.predictor import IPredictor
from src.application.ports.transformer import IFeatureTransformer
from src.application.ports.telemetry import ITelemetryRepository
from src.application.ports.cache import ICacheService
from src.application.ports.repository import IShipmentRepository, IBatchJobRepository
from src.application.ports.queue import ITaskQueue

from src.application.use_cases.predict_shipment import PredictShipmentUseCase
from src.application.use_cases.batch_predict import BatchPredictShipmentsUseCase
from src.application.use_cases.get_qos_metrics import GetQoSMetricsUseCase

from src.infrastructure.adapters.feature_adapter import HSCMFeatureAdapter
from src.infrastructure.adapters.pytorch_adapter import PyTorchModelAdapter
from src.infrastructure.adapters.sklearn_adapter import SklearnModelAdapter
from src.infrastructure.adapters.telemetry_repo import InMemoryTelemetryRepository
from src.infrastructure.cache.cache_service import InMemoryTTLCache, RedisCacheService
from src.infrastructure.persistence.database import DatabaseEngine
from src.infrastructure.persistence.shipment_repo import (
    RelationalShipmentRepository,
    RelationalBatchJobRepository,
)
from src.infrastructure.queue.task_queue import ConcurrentTaskQueue
from src.deployment.middleware.idempotency import IdempotencyManager


class Container:
    """Centralized Dependency Injection Container for Clean Architecture."""

    _instance: Optional["Container"] = None

    def __init__(self, models_dir: Path = MODELS_DIR):
        self._models_dir = models_dir
        self._preprocessor: Optional[HSCMPreprocessor] = None
        self._transformer: Optional[IFeatureTransformer] = None
        self._telemetry_repo: ITelemetryRepository = InMemoryTelemetryRepository(max_samples=1000)
        self._predictors: Dict[str, IPredictor] = {}

        # Enterprise Infrastructure Services
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self._cache_service: ICacheService = RedisCacheService(redis_url=redis_url)
        else:
            self._cache_service: ICacheService = InMemoryTTLCache(max_size=10000)

        self._db_engine = DatabaseEngine.get_instance()
        self._shipment_repo: IShipmentRepository = RelationalShipmentRepository(self._db_engine)
        self._batch_job_repo: IBatchJobRepository = RelationalBatchJobRepository(self._db_engine)
        self._task_queue: ITaskQueue = ConcurrentTaskQueue.get_instance(max_workers=4)
        self._idempotency_mgr = IdempotencyManager(self._cache_service)

    @classmethod
    def get_instance(cls, models_dir: Path = MODELS_DIR) -> "Container":
        """Singleton accessor for application container."""
        if cls._instance is None:
            cls._instance = Container(models_dir=models_dir)
        return cls._instance

    def get_preprocessor(self) -> HSCMPreprocessor:
        """Retrieves or lazily loads the preprocessor."""
        if self._preprocessor is None:
            prep_path = self._models_dir / "preprocessor.pkl"
            if not prep_path.exists():
                raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
            self._preprocessor = HSCMPreprocessor.load(prep_path)
        return self._preprocessor

    def get_transformer(self) -> IFeatureTransformer:
        """Retrieves or lazily instantiates the feature transformer adapter."""
        if self._transformer is None:
            prep = self.get_preprocessor()
            self._transformer = HSCMFeatureAdapter(prep)
        return self._transformer

    def get_telemetry_repo(self) -> ITelemetryRepository:
        """Retrieves the telemetry repository singleton."""
        return self._telemetry_repo

    def get_cache_service(self) -> ICacheService:
        """Retrieves the distributed or tiered cache service."""
        return self._cache_service

    def get_shipment_repo(self) -> IShipmentRepository:
        """Retrieves the shipment persistence repository."""
        return self._shipment_repo

    def get_batch_job_repo(self) -> IBatchJobRepository:
        """Retrieves the batch job repository."""
        return self._batch_job_repo

    def get_task_queue(self) -> ITaskQueue:
        """Retrieves the background task queue."""
        return self._task_queue

    def get_idempotency_manager(self) -> IdempotencyManager:
        """Retrieves the idempotency guard."""
        return self._idempotency_mgr

    def get_predictor(self, model_name: str = "HybridLSTMTransformer") -> IPredictor:
        """
        Retrieves or instantiates an IPredictor adapter for the requested model.
        Supports seamless swapping of PyTorch and Scikit-Learn estimators.
        """
        if model_name in self._predictors:
            return self._predictors[model_name]

        prep = self.get_preprocessor()
        input_dim = len(prep.feature_names)

        # 1. Classical ML Models (Scikit-Learn / XGBoost)
        if model_name in ["LogisticRegression", "RandomForest", "XGBoost"]:
            path = self._models_dir / f"{model_name}.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Estimator artifact not found for {model_name} at {path}")
            with open(path, "rb") as f:
                estimator = pickle.load(f)
            threshold = 0.48 if model_name == "RandomForest" else 0.40
            adapter = SklearnModelAdapter(
                estimator=estimator,
                model_name=model_name,
                optimal_threshold=threshold,
            )
            self._predictors[model_name] = adapter
            return adapter

        # 2. Deep Learning Models (PyTorch)
        pt_map = {
            "HybridLSTMTransformer": (HybridLSTMTransformer(input_dim=input_dim), 0.21),
            "BiLSTM": (BiLSTMStockoutPredictor(input_dim=input_dim), 0.22),
            "Tabular1DCNN": (Tabular1DCNN(input_dim=input_dim), 0.31),
            "DeepTabularMLP": (DeepTabularMLP(input_dim=input_dim), 0.46),
        }

        if model_name not in pt_map:
            raise ValueError(f"Unsupported model architecture: {model_name}")

        nn_module, default_threshold = pt_map[model_name]
        ckpt_path = self._models_dir / f"{model_name}_best.pt"
        optimal_threshold = default_threshold

        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            except (TypeError, Exception):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
            nn_module.load_state_dict(ckpt["model_state_dict"])
            optimal_threshold = float(ckpt.get("optimal_threshold", default_threshold))

        adapter = PyTorchModelAdapter(
            model=nn_module,
            model_name=model_name,
            optimal_threshold=optimal_threshold,
            device=DEVICE,
        )
        self._predictors[model_name] = adapter
        return adapter

    def get_available_model_names(self) -> List[str]:
        """Returns list of supported models in the registry."""
        return [
            "HybridLSTMTransformer",
            "BiLSTM",
            "Tabular1DCNN",
            "DeepTabularMLP",
            "RandomForest",
            "XGBoost",
            "LogisticRegression",
        ]

    # --- Use Case Factory Methods ---

    def get_predict_shipment_use_case(
        self, model_name: str = "HybridLSTMTransformer"
    ) -> PredictShipmentUseCase:
        """Instantiates use case for single shipment prediction."""
        return PredictShipmentUseCase(
            predictor=self.get_predictor(model_name),
            transformer=self.get_transformer(),
            telemetry_repo=self.get_telemetry_repo(),
        )

    def get_batch_predict_use_case(
        self, model_name: str = "HybridLSTMTransformer"
    ) -> BatchPredictShipmentsUseCase:
        """Instantiates use case for batch prediction."""
        return BatchPredictShipmentsUseCase(
            predictor=self.get_predictor(model_name),
            transformer=self.get_transformer(),
        )

    def get_qos_metrics_use_case(self) -> GetQoSMetricsUseCase:
        """Instantiates use case for QoS telemetry retrieval."""
        return GetQoSMetricsUseCase(telemetry_repo=self.get_telemetry_repo())
