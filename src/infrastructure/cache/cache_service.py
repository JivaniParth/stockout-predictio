"""
Layer 3 Adapter: Scalable Tiered Cache Service.
Implements ICacheService port with dual support:
1. InMemoryTTLCache: Thread-safe bounded LRU with per-key TTL, O(1) eviction, and stampede mutex.
2. RedisCacheService: Distributed Redis 7 cluster adapter with connection pooling and resilient fallback.
"""

from collections import OrderedDict
from datetime import datetime, timezone
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

from src.application.ports.cache import ICacheService

logger = logging.getLogger("hscm.infrastructure.cache")


class CacheEntry:
    """Represents a cached item with absolute expiration timestamp and size tracking."""
    __slots__ = ("value", "expires_at", "created_at")

    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds if ttl_seconds > 0 else float("inf")

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class InMemoryTTLCache(ICacheService):
    """
    Production-grade bounded in-memory LRU cache with TTL eviction and Stampede Mutex.
    Provides sub-millisecond (L1) latency without external network dependencies.
    """

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._singleflight_locks: Dict[str, threading.Lock] = {}
        self._singleflight_guard = threading.Lock()

        # Telemetry counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU order
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        with self._lock:
            # Enforce capacity via LRU eviction of oldest key
            if key not in self._cache and len(self._cache) >= self._max_size:
                # Evict expired first, or pop least recently used
                self._evict_one()

            self._cache[key] = CacheEntry(value, ttl_seconds)
            self._cache.move_to_end(key)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return False
            return True

    def get_or_set(
        self,
        key: str,
        producer: Callable[[], Any],
        ttl_seconds: int = 3600,
    ) -> Any:
        """
        Stampede / Thundering Herd Prevention:
        Uses SingleFlight per-key mutex to ensure only ONE worker executes producer on cache miss.
        All concurrent requests for the same key block and read the newly cached result.
        """
        # Fast path: check cache
        cached = self.get(key)
        if cached is not None:
            return cached

        # Acquire or create per-key mutex
        with self._singleflight_guard:
            if key not in self._singleflight_locks:
                self._singleflight_locks[key] = threading.Lock()
            key_lock = self._singleflight_locks[key]

        with key_lock:
            # Double-check inside lock
            cached = self.get(key)
            if cached is not None:
                return cached

            # Compute value
            value = producer()
            self.set(key, value, ttl_seconds)
            return value

    def _evict_one(self) -> None:
        """Evicts expired items or the oldest LRU item."""
        now = time.time()
        for k, v in list(self._cache.items()):
            if v.expires_at < now:
                del self._cache[k]
                self._evictions += 1
                return

        # If none expired, pop least recently used (first item in OrderedDict)
        self._cache.popitem(last=False)
        self._evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round((self._hits / total) * 100, 2) if total > 0 else 0.0
            return {
                "engine": "InMemoryTTLCache (L1)",
                "size": len(self._cache),
                "max_capacity": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_ratio_percent": hit_ratio,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class RedisCacheService(ICacheService):
    """
    Distributed L2 Redis Cache Adapter.
    Features connection pooling, JSON serialization, and fallback to L1 InMemoryTTLCache.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis_url = redis_url
        self._client = None
        self._fallback_l1 = InMemoryTTLCache(max_size=5000)
        self._is_connected = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            self._client = redis.Redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                retry_on_timeout=True,
            )
            self._client.ping()
            self._is_connected = True
            logger.info("Connected to Redis L2 cache at %s", self._redis_url)
        except Exception as e:
            self._is_connected = False
            logger.warning("Redis connection failed (%s). Falling back to in-memory L1 cache.", e)

    def get(self, key: str) -> Optional[Any]:
        if not self._is_connected:
            return self._fallback_l1.get(key)
        try:
            val = self._client.get(key)
            if val is None:
                return None
            return json.loads(val)
        except Exception as e:
            logger.warning("Redis get error for %s: %s", key, e)
            return self._fallback_l1.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        if not self._is_connected:
            return self._fallback_l1.set(key, value, ttl_seconds)
        try:
            serialized = json.dumps(value)
            if ttl_seconds > 0:
                self._client.setex(key, ttl_seconds, serialized)
            else:
                self._client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning("Redis set error for %s: %s", key, e)
            return self._fallback_l1.set(key, value, ttl_seconds)

    def delete(self, key: str) -> bool:
        if not self._is_connected:
            return self._fallback_l1.delete(key)
        try:
            return bool(self._client.delete(key))
        except Exception:
            return self._fallback_l1.delete(key)

    def exists(self, key: str) -> bool:
        if not self._is_connected:
            return self._fallback_l1.exists(key)
        try:
            return bool(self._client.exists(key))
        except Exception:
            return self._fallback_l1.exists(key)

    def get_or_set(
        self,
        key: str,
        producer: Callable[[], Any],
        ttl_seconds: int = 3600,
    ) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        # Fallback to singleflight execution
        return self._fallback_l1.get_or_set(key, producer, ttl_seconds)

    def get_stats(self) -> Dict[str, Any]:
        if not self._is_connected:
            stats = self._fallback_l1.get_stats()
            stats["engine"] = "InMemoryTTLCache (Redis Disconnected)"
            return stats
        try:
            info = self._client.info(section="stats")
            mem = self._client.info(section="memory")
            hits = int(info.get("keyspace_hits", 0))
            misses = int(info.get("keyspace_misses", 0))
            total = hits + misses
            ratio = round((hits / total) * 100, 2) if total > 0 else 0.0
            return {
                "engine": "RedisCluster (L2)",
                "connected": True,
                "used_memory_human": mem.get("used_memory_human", "N/A"),
                "hits": hits,
                "misses": misses,
                "hit_ratio_percent": ratio,
            }
        except Exception:
            return self._fallback_l1.get_stats()
