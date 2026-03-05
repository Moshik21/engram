"""Per-tenant usage metering — fire-and-forget counters in Redis."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class UsageMeter:
    """Tracks API calls and LLM token usage per tenant.

    Uses Redis hashes with daily keys for cheap aggregation.
    All writes are fire-and-forget (pipeline, no await on result).
    """

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client

    def _day_key(self, group_id: str) -> str:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"usage:{group_id}:{day}"

    async def record_api_call(self, group_id: str, route: str) -> None:
        """Increment API call counter. Fire-and-forget."""
        if self._redis is None:
            return
        key = self._day_key(group_id)
        pipe = self._redis.pipeline()
        pipe.hincrby(key, f"api:{route}", 1)
        pipe.hincrby(key, "api:total", 1)
        pipe.expire(key, 90 * 86400)  # 90-day retention
        await pipe.execute()

    async def record_llm_tokens(
        self, group_id: str, input_tokens: int, output_tokens: int,
    ) -> None:
        """Record LLM token usage. Fire-and-forget."""
        if self._redis is None:
            return
        key = self._day_key(group_id)
        pipe = self._redis.pipeline()
        pipe.hincrby(key, "llm:input_tokens", input_tokens)
        pipe.hincrby(key, "llm:output_tokens", output_tokens)
        pipe.expire(key, 90 * 86400)
        await pipe.execute()

    async def get_daily_usage(self, group_id: str, date: str | None = None) -> dict:
        """Get usage for a specific day (default: today)."""
        if self._redis is None:
            return {}
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"usage:{group_id}:{date}"
        data = await self._redis.hgetall(key)
        result = {}
        for k, v in data.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            v_str = v.decode() if isinstance(v, bytes) else v
            result[k_str] = int(v_str)
        return result
