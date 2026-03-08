"""Tests for mode resolution and full-mode dependency waiting."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.storage.resolver import EngineMode, resolve_mode


@pytest.mark.asyncio
async def test_resolve_mode_full_retries_until_services_ready(monkeypatch):
    falkordb = AsyncMock(side_effect=[False, True])
    redis = AsyncMock(side_effect=[False, True])
    sleep = AsyncMock()

    monkeypatch.setattr("engram.storage.resolver._check_falkordb", falkordb)
    monkeypatch.setattr("engram.storage.resolver._check_redis", redis)
    monkeypatch.setattr("engram.storage.resolver.asyncio.sleep", sleep)

    mode = await resolve_mode("full")

    assert mode == EngineMode.FULL
    assert falkordb.await_count == 2
    assert redis.await_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_mode_full_raises_after_timeout(monkeypatch):
    monkeypatch.setenv("ENGRAM_FULL_MODE_WAIT_SECONDS", "0")
    monkeypatch.setattr(
        "engram.storage.resolver._check_falkordb",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "engram.storage.resolver._check_redis",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr("engram.storage.resolver.asyncio.sleep", AsyncMock())

    with pytest.raises(RuntimeError, match="Full mode requested"):
        await resolve_mode("full")
