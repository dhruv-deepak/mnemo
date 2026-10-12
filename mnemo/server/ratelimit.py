"""In-process token-bucket rate limiter, keyed by API key or client IP.

Each key gets a bucket of ``capacity`` tokens that refills continuously at
``capacity / 60`` tokens per second, so short bursts are allowed while the
sustained rate is capped at ``capacity`` requests per minute. For multi-replica
deployments the same algorithm would move to Redis.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(
        self, per_minute: int, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000
    ) -> None:
        self.capacity = float(per_minute)
        self.refill_per_second = per_minute / 60.0
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def acquire(self, key: str) -> float:
        """Take one token. Returns 0 if allowed, else seconds until a token is available."""
        if self.capacity <= 0:
            return 0.0
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    self._evict_full(now)
                bucket = self._buckets[key] = _Bucket(self.capacity, now)
            bucket.tokens = min(
                self.capacity, bucket.tokens + (now - bucket.updated) * self.refill_per_second
            )
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return 0.0
            return (1.0 - bucket.tokens) / self.refill_per_second

    def _evict_full(self, now: float) -> None:
        # Buckets that would be full again carry no state worth keeping.
        full_after = self.capacity / self.refill_per_second
        for key in [k for k, b in self._buckets.items() if now - b.updated >= full_after]:
            del self._buckets[key]
