"""Mode auto-detection: lite vs full based on service availability."""

from __future__ import annotations

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class EngineMode(str, Enum):
    LITE = "lite"
    FULL = "full"
    AUTO = "auto"


async def resolve_mode(requested_mode: str = "auto") -> EngineMode:
    """Determine runtime mode from config or service availability.

    Priority:
    1. Explicit --mode lite or --mode full
    2. ENGRAM_MODE env var
    3. Auto-detect: probe Redis and FalkorDB connectivity
    """
    mode = requested_mode.lower()
    if mode == "lite":
        logger.info("Lite mode selected explicitly")
        return EngineMode.LITE
    if mode == "full":
        logger.info("Full mode selected explicitly — verifying services...")
        if not await _check_falkordb() or not await _check_redis():
            raise RuntimeError(
                "Full mode requested but FalkorDB and/or Redis are not available. "
                "Start them with `docker compose up -d falkordb redis` or use --mode lite."
            )
        return EngineMode.FULL

    env_mode = os.environ.get("ENGRAM_MODE", "").lower()
    if env_mode in ("lite", "full"):
        return await resolve_mode(env_mode)

    logger.info("Auto-detecting mode...")
    falkordb_ok = await _check_falkordb()
    redis_ok = await _check_redis()

    if falkordb_ok and redis_ok:
        logger.info("FalkorDB and Redis detected — using full mode")
        return EngineMode.FULL
    elif falkordb_ok or redis_ok:
        logger.warning(
            "Partial infrastructure detected (FalkorDB=%s, Redis=%s). "
            "Both needed for full mode. Falling back to lite mode.",
            falkordb_ok,
            redis_ok,
        )
        return EngineMode.LITE
    else:
        logger.info("No external services detected — using lite mode")
        return EngineMode.LITE


async def _check_redis() -> bool:
    """Probe Redis connectivity with a 2-second timeout."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(
            os.environ.get("ENGRAM_REDIS__URL", "redis://localhost:6381"),
            socket_connect_timeout=2,
        )
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def _check_falkordb() -> bool:
    """Probe FalkorDB connectivity with a 2-second timeout."""
    try:
        from falkordb import FalkorDB

        db = FalkorDB(
            host=os.environ.get("ENGRAM_FALKORDB__HOST", "localhost"),
            port=int(os.environ.get("ENGRAM_FALKORDB__PORT", "6380")),
            socket_timeout=2,
        )
        db.list_graphs()
        return True
    except Exception:
        return False
