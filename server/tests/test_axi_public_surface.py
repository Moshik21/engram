"""AXI mirrors public MCP tools — no secret better surface."""

from __future__ import annotations

from engram.axi import FULL_ONLY_AXI_BLOCKLIST, PUBLIC_AXI_COMMANDS
from engram.mcp.surface import FULL_ONLY_TOOLS, PUBLIC_CORE_TOOLS


def test_axi_public_commands_cover_write_and_recall():
    for name in ("context", "recall", "observe", "remember"):
        assert name in PUBLIC_AXI_COMMANDS


def test_axi_blocklist_matches_mcp_full_only_aliases():
    # Deprecated aliases / eval must never be preferred AXI paths.
    for name in ("search_entities", "search_facts"):
        assert name in FULL_ONLY_AXI_BLOCKLIST
        assert name in FULL_ONLY_TOOLS


def test_axi_does_not_advertise_full_only_as_public():
    assert FULL_ONLY_AXI_BLOCKLIST.isdisjoint(PUBLIC_AXI_COMMANDS)


def test_mcp_public_core_has_no_full_only_overlap():
    assert FULL_ONLY_TOOLS.isdisjoint(PUBLIC_CORE_TOOLS)
