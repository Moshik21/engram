"""MCP tool surface registry.

Tools are still registered on the FastMCP app in ``server.py`` via
``@mcp.tool()``. This module documents and re-exports the surface profiles
used to freeze the public golden loop for agent installs.
"""

from engram.mcp.surface import (
    OPERATOR_TOOLS,
    PUBLIC_CORE_TOOLS,
    PUBLIC_ONBOARD_TOOLS,
    PUBLIC_TOOLS,
    apply_mcp_surface,
    resolve_mcp_surface,
)

__all__ = [
    "PUBLIC_CORE_TOOLS",
    "PUBLIC_ONBOARD_TOOLS",
    "PUBLIC_TOOLS",
    "OPERATOR_TOOLS",
    "apply_mcp_surface",
    "resolve_mcp_surface",
]
