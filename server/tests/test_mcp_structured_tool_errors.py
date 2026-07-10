"""Structured MCP errors on the real tool.fn wrap path (not helper-only)."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def public_mcp_server(monkeypatch):
    """Import MCP server with public surface after env is set."""
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "public")
    # Force a clean import if previously loaded under full surface.
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "engram.mcp.server" or name.startswith("engram.mcp.server."):
            del sys.modules[name]
    import engram.mcp.server as server

    importlib.reload(server)
    return server


def test_public_surface_tool_count_and_names(public_mcp_server):
    tools = public_mcp_server.mcp._tool_manager._tools
    names = set(tools)
    assert "get_context" in names
    assert "recall" in names
    assert "remember" in names
    assert "search_entities" not in names
    assert "search_facts" not in names
    assert "get_evaluation_report" not in names


@pytest.mark.asyncio
async def test_uninitialized_get_context_returns_structured_error_json(public_mcp_server):
    """Drive the shipped _install_structured_tool_errors wrap on real tool.fn."""
    tools = public_mcp_server.mcp._tool_manager._tools
    tool = tools["get_context"]
    fn = tool.fn
    assert getattr(fn, "_engram_structured_errors", False) is True

    # Manager intentionally unset (fresh import, no lifespan init).
    assert public_mcp_server._manager is None

    raw = await fn()
    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "not_initialized"
    assert "not initialized" in payload["error"]["message"].lower()


@pytest.mark.asyncio
async def test_uninitialized_remember_returns_structured_error_json(public_mcp_server):
    tools = public_mcp_server.mcp._tool_manager._tools
    fn = tools["remember"].fn
    raw = await fn(content="test durable fact for error path")
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "not_initialized"
