"""Per-tenant rate limiting via Redis sliding window."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# Route → (limit, window_seconds)
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    "observe": (100, 60),
    "remember": (20, 60),
    "recall": (60, 60),
    "trigger": (2, 3600),
}


class RateLimiter:
    """Redis-backed sliding window rate limiter.

    Uses sorted sets with timestamp scores. Each check:
    1. Removes entries older than the window
    2. Counts remaining entries
    3. If under limit, adds new entry
    """

    def __init__(self, redis_client=None, limits: dict[str, tuple[int, int]] | None = None) -> None:
        self._redis = redis_client
        self._limits = limits or DEFAULT_LIMITS

    def configure(self, overrides: dict[str, tuple[int, int]]) -> None:
        """Override specific route limits."""
        self._limits.update(overrides)

    async def check(self, group_id: str, route: str) -> tuple[bool, int]:
        """Check if request is allowed.

        Returns (allowed, remaining_requests).
        If no Redis client, always allows (local dev mode).
        """
        if self._redis is None:
            return True, -1

        limit_config = self._limits.get(route)
        if not limit_config:
            return True, -1

        max_requests, window_seconds = limit_config
        key = f"rl:{group_id}:{route}"
        now = time.time()
        window_start = now - window_seconds

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}": now})
        pipe.expire(key, window_seconds + 1)
        results = await pipe.execute()

        current_count = results[1]
        remaining = max(0, max_requests - current_count - 1)

        if current_count >= max_requests:
            # Remove the entry we just added
            await self._redis.zrem(key, f"{now}")
            return False, 0

        return True, remaining

    async def get_usage(self, group_id: str, route: str) -> dict:
        """Get current usage stats for a route."""
        if self._redis is None:
            return {"count": 0, "limit": 0, "window": 0}

        limit_config = self._limits.get(route)
        if not limit_config:
            return {"count": 0, "limit": 0, "window": 0}

        max_requests, window_seconds = limit_config
        key = f"rl:{group_id}:{route}"
        now = time.time()
        window_start = now - window_seconds

        await self._redis.zremrangebyscore(key, 0, window_start)
        count = await self._redis.zcard(key)

        return {
            "count": count,
            "limit": max_requests,
            "window": window_seconds,
            "remaining": max(0, max_requests - count),
        }
