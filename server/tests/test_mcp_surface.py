"""MCP tool surface freeze for golden-loop installs."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from engram.mcp.surface import (
    FULL_ONLY_TOOLS,
    OPERATOR_TOOLS,
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

    extras = {"search_entities", "trigger_consolidation", "timeline"}
    for name in sorted(PUBLIC_CORE_TOOLS | extras):

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
    expected = {
        "get_context",
        "recall",
        "observe",
        "remember",
        "intend",
        "forget",
        "claim_authority",
        "bootstrap_project",
        "get_runtime_state",
    }
    assert PUBLIC_TOOLS == expected
    assert len(PUBLIC_TOOLS) == 9


def test_operator_includes_polish_tools_not_aliases():
    for name in (
        "timeline",
        "route_question",
        "search_artifacts",
        "observe_image",
        "observe_file",
        "loop_status",
        "loop_apply",
        "loop_clear",
        "loop_propose_from_report",
        "loop_steward_once",
    ):
        assert name in OPERATOR_TOOLS
        assert name not in PUBLIC_TOOLS
    for name in FULL_ONLY_TOOLS:
        assert name not in OPERATOR_TOOLS
        assert name not in PUBLIC_TOOLS


def test_apply_mcp_surface_operator_keeps_polish_drops_eval(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "operator")
    mcp = FastMCP("test-surface-operator")
    for name in (
        "get_context",
        "timeline",
        "search_entities",
        "get_evaluation_report",
        "route_question",
    ):

        def _make(tool_name: str = name):
            @mcp.tool(name=tool_name)
            def _tool() -> str:
                """doc"""
                return tool_name

            return _tool

        _make()

    summary = apply_mcp_surface(mcp)
    kept = set(summary["kept"])
    assert summary["surface"] == "operator"
    assert "get_context" in kept
    assert "timeline" in kept
    assert "route_question" in kept
    assert "search_entities" not in kept
    assert "get_evaluation_report" not in kept
