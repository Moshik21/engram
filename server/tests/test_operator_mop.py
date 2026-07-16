"""Operator hygiene mop protocol; public MCP stays 9 tools."""

from __future__ import annotations

import argparse

from engram.hygiene_cli import configure_hygiene_parser
from engram.mcp.surface import PUBLIC_TOOLS


def test_public_tools_still_golden_nine() -> None:
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
    assert "hygiene" not in PUBLIC_TOOLS
    assert "trigger_consolidation" not in PUBLIC_TOOLS


def test_hygiene_cli_parser_exposes_report_and_mop() -> None:
    parser = argparse.ArgumentParser()
    configure_hygiene_parser(parser)
    args = parser.parse_args(["report", "--format", "json"])
    assert args.action == "report"
    args2 = parser.parse_args(["mop", "--dry-run", "--budget", "50"])
    assert args2.action == "mop"
    assert args2.dry_run is True
    assert args2.budget == 50
