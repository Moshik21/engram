"""Mode auto-detection: lite vs full based on service availability."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from enum import Enum
from time import monotonic

logger = logging.getLogger(__name__)


class EngineMode(str, Enum):
    LITE = "lite"
    FULL = "full"
    HELIX = "helix"
    AUTO = "auto"


async def resolve_mode(requested_mode: str = "auto") -> EngineMode:
    """Determine runtime mode from config or service availability.

    Priority:
    1. Explicit --mode lite, full, or helix
    2. ENGRAM_MODE env var
    3. Auto-detect: probe Helix → FalkorDB+Redis → lite
    """
    mode = requested_mode.lower()
    if mode == "lite":
        logger.info("Lite mode selected explicitly")
        return EngineMode.LITE
    if mode == "full":
        logger.info("Full mode selected explicitly — verifying services...")
        if not await _wait_for_full_services():
            raise RuntimeError(
                "Full mode requested but FalkorDB and/or Redis are not available. "
                "Ensure: (1) Docker services are running: `docker compose up -d falkordb redis`, "
                "(2) full-mode packages are installed: `uv sync --dev` (includes [full] extras). "
                "Or use ENGRAM_MODE=auto to fall back to lite mode gracefully."
            )
        return EngineMode.FULL
    if mode == "helix":
        # Check transport from env var OR dotenv (pydantic-settings reads .env but
        # doesn't set os.environ, so check both)
        transport = os.environ.get("ENGRAM_HELIX__TRANSPORT", "").lower()
        if not transport:
            try:
                from engram.config import EngramConfig
                transport = EngramConfig().helix.transport.lower()
            except Exception:
                transport = "http"
        if transport == "native":
            logger.info("Helix mode with native (PyO3) transport — no network check needed")
            return EngineMode.HELIX
        logger.info("Helix mode selected explicitly — verifying HelixDB...")
        if not await _check_helix():
            raise RuntimeError(
                "Helix mode requested but HelixDB is not reachable. "
                "Ensure HelixDB is running on the configured host/port "
                "(default: localhost:6969). "
                "Or use ENGRAM_MODE=auto to fall back to lite mode gracefully."
            )
        return EngineMode.HELIX

    env_mode = os.environ.get("ENGRAM_MODE", "").lower()
    if env_mode in ("lite", "full", "helix"):
        return await resolve_mode(env_mode)

    logger.info("Auto-detecting mode...")
    # Check Helix first (single service replaces FalkorDB + Redis)
    if await _check_helix():
        logger.info("HelixDB detected — using helix mode")
        return EngineMode.HELIX

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


async def _check_helix() -> bool:
    """Probe HelixDB connectivity with a 2-second timeout."""
    try:
        import socket

        host = os.environ.get("ENGRAM_HELIX__HOST", "localhost")
        port = int(os.environ.get("ENGRAM_HELIX__PORT", "6969"))
        socket.create_connection((host, port), timeout=2)
        return True
    except Exception:
        return False


async def _check_redis() -> bool:
    """Probe Redis connectivity with a 2-second timeout."""
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(
            os.environ.get("ENGRAM_REDIS__URL", "redis://localhost:6381"),
            socket_connect_timeout=2,
        )
        ping_result = r.ping()
        if inspect.isawaitable(ping_result):
            await ping_result
        await r.aclose()
        return True
    except Exception:
        return False


async def _check_falkordb() -> bool:
    """Probe FalkorDB connectivity with a 2-second timeout."""
    try:
        from falkordb import FalkorDB  # type: ignore[import-untyped]

        db = FalkorDB(
            host=os.environ.get("ENGRAM_FALKORDB__HOST", "localhost"),
            port=int(os.environ.get("ENGRAM_FALKORDB__PORT", "6380")),
            password=os.environ.get("ENGRAM_FALKORDB__PASSWORD"),
            socket_timeout=2,
        )
        db.list_graphs()
        return True
    except Exception:
        return False


async def _wait_for_full_services(
    *,
    timeout_seconds: float | None = None,
    retry_interval_seconds: float | None = None,
) -> bool:
    """Wait briefly for explicit full-mode dependencies to become truly ready."""
    timeout = timeout_seconds
    if timeout is None:
        timeout = float(os.environ.get("ENGRAM_FULL_MODE_WAIT_SECONDS", "20"))
    interval = retry_interval_seconds
    if interval is None:
        interval = float(os.environ.get("ENGRAM_FULL_MODE_RETRY_INTERVAL_SECONDS", "1"))

    timeout = max(0.0, timeout)
    interval = max(0.1, interval)
    deadline = monotonic() + timeout
    attempt = 0

    while True:
        attempt += 1
        falkordb_ok = await _check_falkordb()
        redis_ok = await _check_redis()
        if falkordb_ok and redis_ok:
            if attempt > 1:
                logger.info(
                    "Full-mode services became ready after %s attempts",
                    attempt,
                )
            return True
        if monotonic() >= deadline:
            logger.warning(
                "Full-mode services did not become ready within %.1fs (FalkorDB=%s, Redis=%s)",
                timeout,
                falkordb_ok,
                redis_ok,
            )
            return False
        logger.info(
            "Waiting for full-mode services (attempt %s, FalkorDB=%s, Redis=%s)",
            attempt,
            falkordb_ok,
            redis_ok,
        )
        await asyncio.sleep(interval)
