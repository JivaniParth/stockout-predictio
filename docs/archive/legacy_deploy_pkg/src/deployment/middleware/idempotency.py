"""
API Delivery Middleware: Distributed Idempotency Guard.
Prevents duplicate processing of financial or logistical operations upon network retry.
"""

from typing import Optional
import json
from src.application.ports.cache import ICacheService


class IdempotencyManager:
    """Manages idempotency keys and cached API responses."""

    def __init__(self, cache_service: ICacheService, ttl_seconds: int = 86400):
        self._cache = cache_service
        self._ttl = ttl_seconds

    def get_existing_response(self, idempotency_key: str) -> Optional[dict]:
        """Checks if a response was already generated for this idempotency key."""
        cache_key = f"idempotency:{idempotency_key}"
        return self._cache.get(cache_key)

    def record_response(self, idempotency_key: str, response_payload: dict) -> None:
        """Stores the response payload under the idempotency key for 24 hours."""
        cache_key = f"idempotency:{idempotency_key}"
        self._cache.set(cache_key, response_payload, ttl_seconds=self._ttl)
