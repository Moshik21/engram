"""AXI command surface for shell-capable AI agents.

AXI mirrors the **public** MCP golden loop — no secret better surface.

Public AXI commands (parity with MCP public tools + install helpers):
  context  ↔ get_context
  recall   ↔ recall
  observe  ↔ observe
  remember ↔ remember
  bootstrap↔ bootstrap_project
  doctor   ↔ readiness (hooks + surface)

Not exposed on AXI (full/operator MCP only): search_entities, search_facts,
evaluation tools, timeline, route_question, etc.
"""

from __future__ import annotations

# Commands agents should treat as the AXI public surface.
PUBLIC_AXI_COMMANDS: frozenset[str] = frozenset(
    {
        "context",
        "recall",
        "observe",
        "remember",
        "bootstrap",
        "doctor",
        "storage",
        "value",
        "hooks",
        "packet-cache",
    }
)

# Explicitly full/debug-only — never a preferred agent path via AXI.
FULL_ONLY_AXI_BLOCKLIST: frozenset[str] = frozenset(
    {
        "search_entities",
        "search_facts",
        "get_evaluation_report",
        "record_recall_evaluation",
        "record_session_continuity_evaluation",
    }
)
