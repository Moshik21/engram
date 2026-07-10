"""MCP tool surface profiles — golden loop vs operator vs full.

Product freeze (docs/GOLDEN_LOOP.md): agents should live on the public six
(+ claim_authority for authority routing). The full registry remains available
for operators and eval via ENGRAM_MCP_SURFACE.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

logger = logging.getLogger(__name__)

McpSurface = Literal["public", "operator", "full"]

# Agents must use these for continuity. Everything else is optional surface.
PUBLIC_CORE_TOOLS: frozenset[str] = frozenset(
    {
        "get_context",
        "recall",
        "observe",
        "remember",
        "intend",
        "forget",
        "claim_authority",
    }
)

# Onboarding / session protocol helpers still needed on public installs.
PUBLIC_ONBOARD_TOOLS: frozenset[str] = frozenset(
    {
        "bootstrap_project",
        "get_runtime_state",
    }
)

PUBLIC_TOOLS: frozenset[str] = PUBLIC_CORE_TOOLS | PUBLIC_ONBOARD_TOOLS

# Operator / repair tools kept available on "operator" surface.
# Includes polish tools (timeline, route, artifacts, media observe).
OPERATOR_EXTRA_TOOLS: frozenset[str] = frozenset(
    {
        "mark_identity_core",
        "trigger_consolidation",
        "get_consolidation_status",
        "list_intentions",
        "dismiss_intention",
        "feedback",
        "forget",  # already public
        "get_lifecycle_summary",
        "get_graph_state",
        "adjudicate_evidence",
        "timeline",
        "route_question",
        "search_artifacts",
        "observe_image",
        "observe_file",
    }
)

OPERATOR_TOOLS: frozenset[str] = PUBLIC_TOOLS | OPERATOR_EXTRA_TOOLS

# Full-surface only: deprecated aliases + evaluation. Never public/operator.
FULL_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "search_entities",
        "search_facts",
        "get_evaluation_report",
        "record_session_continuity_evaluation",
        "record_recall_evaluation",
    }
)


def resolve_mcp_surface(raw: str | None = None) -> McpSurface:
    """Resolve surface profile from arg or ENGRAM_MCP_SURFACE env."""
    value = (raw or os.environ.get("ENGRAM_MCP_SURFACE") or "public").strip().lower()
    if value in {"public", "agent", "golden", "core"}:
        return "public"
    if value in {"operator", "ops", "admin"}:
        return "operator"
    if value in {"full", "all", "debug"}:
        return "full"
    logger.warning("Unknown ENGRAM_MCP_SURFACE=%r; defaulting to public", value)
    return "public"


def allowed_tools_for_surface(surface: McpSurface) -> frozenset[str] | None:
    """Return tool allowlist, or None when all tools are allowed."""
    if surface == "full":
        return None
    if surface == "operator":
        return OPERATOR_TOOLS
    return PUBLIC_TOOLS


def apply_mcp_surface(mcp: Any, *, surface: str | None = None) -> dict[str, Any]:
    """Remove non-allowed tools from a FastMCP server in place.

    Returns a summary for logs/tests.
    """
    profile = resolve_mcp_surface(surface)
    allowed = allowed_tools_for_surface(profile)
    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        return {
            "surface": profile,
            "registered": [],
            "removed": [],
            "kept": [],
        }

    registered = [tool.name for tool in manager.list_tools()]
    if allowed is None:
        return {
            "surface": profile,
            "registered": registered,
            "removed": [],
            "kept": registered,
        }

    removed: list[str] = []
    for name in list(registered):
        if name not in allowed:
            try:
                mcp.remove_tool(name)
                removed.append(name)
            except Exception:
                logger.debug("Failed to remove MCP tool %s", name, exc_info=True)

    kept = [tool.name for tool in manager.list_tools()]
    logger.info(
        "MCP surface=%s kept=%d removed=%d",
        profile,
        len(kept),
        len(removed),
    )
    return {
        "surface": profile,
        "registered": registered,
        "removed": sorted(removed),
        "kept": sorted(kept),
    }
