"""
Application Layer Port: Cache Service Interface.
Decouples application use cases from specific caching technology (Redis, Memcached, In-Memory LRU).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional


class ICacheService(ABC):
    """Abstract contract for distributed and tiered caching."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from cache. Returns None if missing or expired."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Stores an item in cache with a defined Time-To-Live (TTL)."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Invalidates an item from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Checks if a non-expired key exists in the cache."""
        pass

    @abstractmethod
    def get_or_set(
        self,
        key: str,
        producer: Callable[[], Any],
        ttl_seconds: int = 3600,
    ) -> Any:
        """
        Cache-Aside with Thundering Herd / Stampede Protection.
        Atomically gets or computes value via producer function.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Returns cache telemetry: hits, misses, hit ratio, memory usage."""
        pass
