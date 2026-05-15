from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CONSOLIDATION_PRESENTER_BOUNDARIES = {
    ("engram/consolidation_trigger.py", "build_api_consolidation_status_surface"): {
        "serialize_cycle_summary",
    },
    ("engram/consolidation_trigger.py", "build_api_consolidation_history_surface"): {
        "serialize_cycle_summary",
    },
    ("engram/consolidation_trigger.py", "build_api_consolidation_cycle_detail_surface"): {
        "serialize_cycle_detail",
    },
    ("engram/consolidation_trigger.py", "build_mcp_consolidation_trigger_surface"): {
        "serialize_cycle_summary",
    },
    ("engram/consolidation_trigger.py", "build_mcp_consolidation_status_surface"): {
        "serialize_cycle_summary",
    },
    ("engram/consolidation/cli.py", "_print_cycle_result"): {
        "cycle_phase_issue_text",
        "serialize_cycle_summary",
    },
}


def _function_names_used(relative_path: str, function_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == function_name:
            return {name.id for name in ast.walk(node) if isinstance(name, ast.Name)}
    raise AssertionError(f"Function not found: {relative_path}:{function_name}")


@pytest.mark.parametrize(
    ("surface", "expected_names"),
    CONSOLIDATION_PRESENTER_BOUNDARIES.items(),
)
def test_public_consolidation_surfaces_use_shared_presenter(
    surface: tuple[str, str],
    expected_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_names_used(relative_path, function_name)
    missing = expected_names - names_used
    assert missing == set()
