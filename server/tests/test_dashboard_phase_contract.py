from __future__ import annotations

import ast
import re
from pathlib import Path

from engram.consolidation.phase_registry import CONSOLIDATION_PHASE_ORDER


def _dashboard_phase_order() -> tuple[str, ...]:
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "dashboard" / "src" / "constants" / "consolidation.ts"
    text = source.read_text()
    match = re.search(
        r"CONSOLIDATION_PHASE_ORDER\s*=\s*\[(.*?)\]\s+as const",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "dashboard consolidation phase order constant not found"
    parsed = ast.literal_eval(f"[{match.group(1)}]")
    assert isinstance(parsed, list)
    assert all(isinstance(item, str) for item in parsed)
    return tuple(parsed)


def test_dashboard_consolidation_phase_order_matches_backend_registry():
    assert _dashboard_phase_order() == CONSOLIDATION_PHASE_ORDER
