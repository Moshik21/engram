"""Deterministic evaluation fixtures for recall-need analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.benchmark.metrics import (
    RecallEvalSample,
    false_recall_rate,
    memory_need_precision,
    surfaced_to_used_ratio,
)
from engram.config import ActivationConfig
from engram.retrieval.context import ConversationContext
from engram.retrieval.need import analyze_memory_need


@dataclass(frozen=True)
class MemoryNeedFixture:
    """Single deterministic analyzer fixture."""

    text: str
    expect_recall: bool
    mode: str = "chat"
    expected_need_type: str | None = None
    expected_family: str | None = None
    packets_surfaced: int = 0
    packets_used: int = 0
    false_recalls: int = 0
    recent_turns: list[str] = field(default_factory=list)
    session_entity_names: list[str] = field(default_factory=list)
    context_turns: list[str] = field(default_factory=list)
    cfg_overrides: dict[str, Any] = field(default_factory=dict)
    group_id: str = "default"
    graph_probe: Any = None


@dataclass(frozen=True)
class MemoryNeedFixtureSummary:
    """Aggregated analyzer fixture metrics."""

    precision: float
    recall: float
    false_recall_rate: float
    graph_lift_rate: float
    surfaced_to_used_ratio: float
    family_trigger_counts: dict[str, int]
    family_true_positive_counts: dict[str, int]
    need_type_hits: dict[str, int]


async def evaluate_memory_need_fixtures(
    fixtures: list[MemoryNeedFixture],
) -> MemoryNeedFixtureSummary:
    """Run the analyzer over deterministic fixtures and summarize behavior."""
    recall_samples: list[RecallEvalSample] = []
    family_trigger_counts: dict[str, int] = {}
    family_true_positive_counts: dict[str, int] = {}
    need_type_hits: dict[str, int] = {}
    expected_positive = 0
    true_positive = 0
    total_surfaced = 0
    total_used = 0
    graph_lift_count = 0

    for fixture in fixtures:
        cfg = ActivationConfig(**fixture.cfg_overrides)
        conv_context = _build_context(fixture.context_turns)
        need = await analyze_memory_need(
            fixture.text,
            recent_turns=fixture.recent_turns,
            session_entity_names=fixture.session_entity_names,
            mode=fixture.mode,
            graph_probe=fixture.graph_probe,
            group_id=fixture.group_id,
            conv_context=conv_context,
            cfg=cfg,
        )

        if fixture.expect_recall:
            expected_positive += 1
        if need.should_recall and fixture.expect_recall:
            true_positive += 1
        if need.should_recall:
            family = need.trigger_family or "unknown"
            family_trigger_counts[family] = family_trigger_counts.get(family, 0) + 1
            if fixture.expect_recall:
                family_true_positive_counts[family] = family_true_positive_counts.get(family, 0) + 1
            need_type_hits[need.need_type] = need_type_hits.get(need.need_type, 0) + 1
            if need.decision_path == "graph_lift":
                graph_lift_count += 1

        surfaced = max(0, fixture.packets_surfaced)
        used = min(max(0, fixture.packets_used), surfaced)
        total_surfaced += surfaced
        total_used += used
        recall_samples.append(
            RecallEvalSample(
                recall_triggered=need.should_recall,
                recall_helped=need.should_recall and fixture.expect_recall,
                packets_surfaced=surfaced,
                packets_used=used,
                false_recalls=max(0, fixture.false_recalls),
            )
        )

        if fixture.expected_need_type is not None and fixture.expect_recall:
            assert need.need_type == fixture.expected_need_type
        if fixture.expected_family is not None and fixture.expect_recall:
            assert need.trigger_family == fixture.expected_family

    return MemoryNeedFixtureSummary(
        precision=memory_need_precision(recall_samples),
        recall=(true_positive / expected_positive) if expected_positive else 0.0,
        false_recall_rate=false_recall_rate(recall_samples),
        graph_lift_rate=(graph_lift_count / true_positive) if true_positive else 0.0,
        surfaced_to_used_ratio=surfaced_to_used_ratio(total_surfaced, total_used),
        family_trigger_counts=family_trigger_counts,
        family_true_positive_counts=family_true_positive_counts,
        need_type_hits=need_type_hits,
    )


def _build_context(turns: list[str]) -> ConversationContext | None:
    if not turns:
        return None
    ctx = ConversationContext()
    for turn in turns:
        ctx.add_turn(turn)
    return ctx
