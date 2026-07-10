"""MCP tool surface freeze for golden-loop installs."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from engram.mcp.surface import (
    PUBLIC_CORE_TOOLS,
    PUBLIC_TOOLS,
    apply_mcp_surface,
    resolve_mcp_surface,
)


def test_resolve_mcp_surface_defaults_public(monkeypatch):
    monkeypatch.delenv("ENGRAM_MCP_SURFACE", raising=False)
    assert resolve_mcp_surface() == "public"
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "full")
    assert resolve_mcp_surface() == "full"
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "operator")
    assert resolve_mcp_surface() == "operator"


def test_apply_mcp_surface_public_keeps_golden_loop_only(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "public")
    mcp = FastMCP("test-surface")

    for name in sorted(PUBLIC_CORE_TOOLS | {"search_entities", "trigger_consolidation", "timeline"}):

        def _make(tool_name: str = name):
            @mcp.tool(name=tool_name)
            def _tool() -> str:
                """doc"""
                return tool_name

            return _tool

        _make()

    summary = apply_mcp_surface(mcp)
    kept = set(summary["kept"])
    assert summary["surface"] == "public"
    assert PUBLIC_CORE_TOOLS <= kept
    assert "search_entities" not in kept
    assert "trigger_consolidation" not in kept
    assert "timeline" not in kept


def test_apply_mcp_surface_full_keeps_all(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "full")
    mcp = FastMCP("test-surface-full")

    @mcp.tool()
    def remember() -> str:
        """doc"""
        return "ok"

    @mcp.tool()
    def search_entities() -> str:
        """doc"""
        return "ok"

    summary = apply_mcp_surface(mcp)
    assert summary["surface"] == "full"
    assert summary["removed"] == []
    assert set(summary["kept"]) >= {"remember", "search_entities"}


def test_public_tools_cover_golden_loop():
    for name in (
        "get_context",
        "recall",
        "observe",
        "remember",
        "intend",
        "forget",
        "claim_authority",
    ):
        assert name in PUBLIC_TOOLS
