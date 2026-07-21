"""M0.6 grep gate: no compute_activation call may silently drop consolidated_strength.

Two guarantees:
1. A permanent AST-based gate over ``server/engram``: any ``compute_activation``
   call that passes fewer than 4 positional args and no ``consolidated_strength``
   keyword is an offender. Offenders are only tolerated inside an explicit
   per-file allowlist (a ratchet — counts are maximums, never to be raised).
2. A cs-seeded zero-access entity (consolidated_strength > 0, empty
   access_history) is visible to goal priming instead of computing as floor.
"""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import engram
from engram.config import ActivationConfig
from engram.retrieval.goals import identify_active_goals

ENGRAM_ROOT = Path(engram.__file__).parent

# path (relative to engram/) -> max allowed cs-omitting compute_activation calls.
# Every entry is tech debt outside the M0.6 ranking-path fix: thread
# state.consolidated_strength at the call site and ratchet the count down
# (delete the entry at zero). Do NOT add new entries or raise a count.
CS_OMITTED_ALLOWLIST: dict[str, int] = {
    "atlas/builder.py": 1,  # atlas dashboard snapshot; unthreaded
    "retrieval/context_builder.py": 4,  # briefing/context surface; unthreaded
    "retrieval/feedback.py": 1,  # feedback introspection; unthreaded
    # dashboard/time-travel views; one site replays historical timestamps
    # where cs is not time-indexed
    "retrieval/graph_state.py": 6,
    "retrieval/lookup.py": 1,  # entity lookup surface; unthreaded
    "retrieval/prospective.py": 2,  # prospective memory surface; unthreaded
    "retrieval/surprise.py": 1,  # surprise scoring; unthreaded
}


def _cs_omitting_calls(source: str, filename: str) -> int:
    """Count compute_activation calls that omit consolidated_strength."""
    count = 0
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name != "compute_activation":
            continue
        keywords = {kw.arg for kw in node.keywords}
        if len(node.args) < 4 and "consolidated_strength" not in keywords:
            count += 1
    return count


class TestComputeActivationSignatureGate:
    def test_no_cs_omitting_calls_outside_allowlist(self):
        violations: list[str] = []
        for path in sorted(ENGRAM_ROOT.rglob("*.py")):
            rel = path.relative_to(ENGRAM_ROOT).as_posix()
            count = _cs_omitting_calls(path.read_text(), rel)
            if count > CS_OMITTED_ALLOWLIST.get(rel, 0):
                violations.append(
                    f"{rel}: {count} compute_activation call(s) omit consolidated_strength"
                    f" (allowed: {CS_OMITTED_ALLOWLIST.get(rel, 0)})"
                )
        assert not violations, (
            "compute_activation signature drift (M0.6): thread"
            " state.consolidated_strength as the 4th argument instead of"
            " allowlisting:\n" + "\n".join(violations)
        )

    def test_gate_detects_three_arg_call(self):
        source = "act = compute_activation(state.access_history, now, cfg)\n"
        assert _cs_omitting_calls(source, "<synthetic>") == 1

    def test_gate_accepts_threaded_calls(self):
        source = (
            "a = compute_activation(h, now, cfg, state.consolidated_strength)\n"
            "b = engine.compute_activation(h, now, cfg, consolidated_strength=cs)\n"
        )
        assert _cs_omitting_calls(source, "<synthetic>") == 0


# --- Goals visibility: cs-seeded zero-access entity ---


@dataclass
class _Entity:
    id: str
    name: str
    entity_type: str
    attributes: dict = field(default_factory=dict)
    deleted_at: object = None


@dataclass
class _State:
    access_history: list[float] = field(default_factory=list)
    access_count: int = 0
    consolidated_strength: float = 0.0


def _stores(state: _State):
    graph_store = AsyncMock()
    goal = _Entity(id="g1", name="Ship M0.6", entity_type="Goal")

    async def find_entities(entity_type=None, group_id=None, limit=10, **kwargs):
        return [goal] if entity_type == "Goal" else []

    graph_store.find_entities = AsyncMock(side_effect=find_entities)
    graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

    activation_store = AsyncMock()
    activation_store.get_activation = AsyncMock(return_value=state)
    return graph_store, activation_store


class TestCsSeededGoalVisibility:
    @pytest.mark.asyncio
    async def test_cs_seeded_zero_access_goal_is_visible(self):
        """consolidated_strength=0.02, empty access_history -> activation > 0.

        Before M0.6 this path computed act_level = 0.0 (floor) and the goal
        was invisible to priming.
        """
        cfg = ActivationConfig(goal_priming_enabled=True)
        graph_store, activation_store = _stores(_State(consolidated_strength=0.02))

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)

        assert len(goals) == 1
        assert goals[0].entity_id == "g1"
        assert goals[0].activation > 0.0
        assert goals[0].activation >= cfg.goal_priming_activation_floor

    @pytest.mark.asyncio
    async def test_zero_cs_zero_access_goal_stays_invisible(self):
        cfg = ActivationConfig(goal_priming_enabled=True)
        graph_store, activation_store = _stores(_State())

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)

        assert goals == []

    @pytest.mark.asyncio
    async def test_accessed_goal_still_visible(self):
        now = time.time()
        cfg = ActivationConfig(goal_priming_enabled=True)
        graph_store, activation_store = _stores(
            _State(access_history=[now - 10, now - 5], access_count=2)
        )

        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)

        assert len(goals) == 1
        assert goals[0].activation > 0.0
