"""Structured MCP error contract."""

from __future__ import annotations

import json

from engram.mcp.errors import (
    McpToolError,
    format_tool_exception,
    mcp_error_json,
    mcp_error_payload,
)


def test_mcp_error_payload_shape():
    payload = mcp_error_payload("not_initialized", "MCP server not initialized")
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "not_initialized"
    assert "MCP server" in payload["error"]["message"]


def test_mcp_error_json_roundtrip():
    raw = mcp_error_json("validation", "bad arg", details={"field": "limit"})
    data = json.loads(raw)
    assert data["status"] == "error"
    assert data["error"]["details"]["field"] == "limit"


def test_mcp_tool_error_to_json():
    err = McpToolError("not_initialized", "MCP server not initialized")
    data = json.loads(err.to_json())
    assert data["status"] == "error"
    assert data["error"]["code"] == "not_initialized"


def test_format_tool_exception_wraps_generic():
    data = json.loads(format_tool_exception(ValueError("boom")))
    assert data["status"] == "error"
    assert data["error"]["code"] == "internal_error"
    assert "boom" in data["error"]["message"]
