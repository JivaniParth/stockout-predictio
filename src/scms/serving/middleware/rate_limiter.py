"""
API Delivery Middleware: Token Bucket / Sliding Window Rate Limiter.
Protects inference microservice endpoints against API abuse, scraping, and DoS.
"""

from collections import defaultdict
import time
import threading
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter per client key."""

    def __init__(self, requests_per_minute: int = 120):
        self._rate_limit = requests_per_minute
        self._window_seconds = 60.0
        self._clients: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Returns (is_allowed, remaining_quota, reset_seconds).
        """
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            # Purge timestamps outside current window
            timestamps = [t for t in self._clients[client_id] if t > window_start]
            remaining = max(0, self._rate_limit - len(timestamps))
            reset_seconds = int(self._window_seconds - (now - timestamps[0])) if timestamps else 60

            if len(timestamps) >= self._rate_limit:
                self._clients[client_id] = timestamps
                return False, 0, reset_seconds

            timestamps.append(now)
            self._clients[client_id] = timestamps
            return True, remaining - 1, reset_seconds


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI ASGI middleware enforcing per-IP rate limits with standard headers."""

    def __init__(self, app, requests_per_minute: int = 300):
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter(requests_per_minute=requests_per_minute)
        self.limit = requests_per_minute

    async def dispatch(self, request: Request, call_next):
        # Exclude healthchecks and static docs
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Identify client by IP or API Key
        client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "127.0.0.1")
        allowed, remaining, reset = self.limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Maximum {self.limit} requests per minute.",
                    "retry_after_seconds": reset,
                },
                headers={
                    "Retry-After": str(reset),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)
        return response
