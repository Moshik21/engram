from __future__ import annotations

import pytest

from engram.config import EngramConfig
from engram.mcp import server as mcp_server


def test_mcp_background_runtime_start_helpers_respect_external_owner():
    config = EngramConfig(_env_file=None)
    config.activation.worker_enabled = True
    config.activation.cue_index_outbox_enabled = True

    try:
        mcp_server.set_background_runtime_managed_externally(False)
        assert mcp_server._should_start_mcp_background_runtime(config) is True
        assert mcp_server._should_start_mcp_cue_index_outbox(config) is True

        mcp_server.set_background_runtime_managed_externally(True)
        assert mcp_server._should_start_mcp_background_runtime(config) is False
        assert mcp_server._should_start_mcp_cue_index_outbox(config) is False
    finally:
        mcp_server.set_background_runtime_managed_externally(False)


@pytest.mark.asyncio
async def test_mcp_lifespan_keeps_process_runtime_between_http_sessions(monkeypatch):
    calls: list[str] = []

    async def fake_init() -> None:
        calls.append("init")
        monkeypatch.setattr(mcp_server, "_manager", object())

    async def fake_shutdown() -> None:
        calls.append("shutdown")
        monkeypatch.setattr(mcp_server, "_manager", None)

    monkeypatch.setattr(mcp_server, "_init", fake_init)
    monkeypatch.setattr(mcp_server, "_shutdown", fake_shutdown)
    monkeypatch.setenv("ENGRAM_MCP_SHUTDOWN_ON_IDLE", "0")
    monkeypatch.setattr(mcp_server, "_manager", None)
    monkeypatch.setattr(mcp_server, "_lifespan_lock", None)
    monkeypatch.setattr(mcp_server, "_lifespan_lock_loop", None)
    monkeypatch.setattr(mcp_server, "_lifespan_refcount", 0)

    async with mcp_server._lifespan(mcp_server.mcp):
        async with mcp_server._lifespan(mcp_server.mcp):
            assert calls == ["init"]
        assert calls == ["init"]

    assert calls == ["init"]

    async with mcp_server._lifespan(mcp_server.mcp):
        assert calls == ["init"]

    assert calls == ["init"]


@pytest.mark.asyncio
async def test_mcp_lifespan_can_shutdown_on_idle_for_tests(monkeypatch):
    calls: list[str] = []

    async def fake_init() -> None:
        calls.append("init")
        monkeypatch.setattr(mcp_server, "_manager", object())

    async def fake_shutdown() -> None:
        calls.append("shutdown")
        monkeypatch.setattr(mcp_server, "_manager", None)

    monkeypatch.setattr(mcp_server, "_init", fake_init)
    monkeypatch.setattr(mcp_server, "_shutdown", fake_shutdown)
    monkeypatch.setenv("ENGRAM_MCP_SHUTDOWN_ON_IDLE", "1")
    monkeypatch.setattr(mcp_server, "_manager", None)
    monkeypatch.setattr(mcp_server, "_lifespan_lock", None)
    monkeypatch.setattr(mcp_server, "_lifespan_lock_loop", None)
    monkeypatch.setattr(mcp_server, "_lifespan_refcount", 0)

    async with mcp_server._lifespan(mcp_server.mcp):
        assert calls == ["init"]

    assert calls == ["init", "shutdown"]
