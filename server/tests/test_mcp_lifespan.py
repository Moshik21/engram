from __future__ import annotations

import pytest

from engram.mcp import server as mcp_server


@pytest.mark.asyncio
async def test_mcp_lifespan_refcounts_overlapping_http_sessions(monkeypatch):
    calls: list[str] = []

    async def fake_init() -> None:
        calls.append("init")

    async def fake_shutdown() -> None:
        calls.append("shutdown")

    monkeypatch.setattr(mcp_server, "_init", fake_init)
    monkeypatch.setattr(mcp_server, "_shutdown", fake_shutdown)
    monkeypatch.setattr(mcp_server, "_lifespan_lock", None)
    monkeypatch.setattr(mcp_server, "_lifespan_lock_loop", None)
    monkeypatch.setattr(mcp_server, "_lifespan_refcount", 0)

    async with mcp_server._lifespan(mcp_server.mcp):
        async with mcp_server._lifespan(mcp_server.mcp):
            assert calls == ["init"]
        assert calls == ["init"]

    assert calls == ["init", "shutdown"]
