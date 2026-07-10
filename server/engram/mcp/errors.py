"""Structured MCP tool errors.

Agents should receive JSON ``{"status": "error", "error": {...}}`` rather than
opaque RuntimeError text when a tool fails in a recoverable/product way.
"""

from __future__ import annotations

import json
from typing import Any


class McpToolError(Exception):
    """Raised by MCP helpers; convert to structured JSON for tool responses."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            error["details"] = self.details
        return {"status": "error", "error": error}

    def to_json(self) -> str:
        return json.dumps(self.to_payload())


def mcp_error_payload(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured error dict (not raised)."""
    return McpToolError(code, message, details=details).to_payload()


def mcp_error_json(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> str:
    """Build a structured error JSON string for tool return values."""
    return json.dumps(mcp_error_payload(code, message, details=details))


def format_tool_exception(exc: BaseException) -> str:
    """Convert any exception into structured MCP error JSON."""
    if isinstance(exc, McpToolError):
        return exc.to_json()
    return mcp_error_json(
        "internal_error",
        str(exc) or exc.__class__.__name__,
        details={"type": type(exc).__name__},
    )
