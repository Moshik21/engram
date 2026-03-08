from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.benchmark.memory_need import MemoryNeedFixture, evaluate_memory_need_fixtures
from engram.retrieval.graph_probe import ProbeResult


@pytest.mark.asyncio
async def test_evaluate_memory_need_fixtures_summarizes_precision_and_family_counts():
    fixtures = [
        MemoryNeedFixture(
            text="my son had a great game",
            expect_recall=True,
            expected_need_type="fact_lookup",
            expected_family="pragmatic",
            packets_surfaced=2,
            packets_used=1,
        ),
        MemoryNeedFixture(
            text="Actually, it's not PostgreSQL, it's MySQL",
            expect_recall=True,
            expected_need_type="temporal_update",
            expected_family="structural",
            packets_surfaced=1,
            packets_used=1,
            cfg_overrides={"recall_need_structural_enabled": True},
        ),
        MemoryNeedFixture(
            text="Sarah sent me the final deck",
            expect_recall=True,
            expected_need_type="project_state",
            expected_family="structural",
            packets_surfaced=1,
            packets_used=1,
            cfg_overrides={"recall_need_structural_enabled": True},
        ),
        MemoryNeedFixture(
            text="Really worried about Ben lately",
            expect_recall=True,
            expected_need_type="broad_context",
            expected_family="structural",
            packets_surfaced=1,
            packets_used=1,
            cfg_overrides={"recall_need_structural_enabled": True},
        ),
        MemoryNeedFixture(
            text="Can you write a for loop?",
            expect_recall=False,
            context_turns=[
                "Working on the auth migration",
                "Need to fix the Redis cache layer",
            ],
            cfg_overrides={
                "recall_need_shift_enabled": True,
                "recall_need_impoverishment_enabled": True,
                "recall_need_shift_shadow_only": False,
                "recall_need_impoverishment_shadow_only": False,
            },
        ),
    ]

    summary = await evaluate_memory_need_fixtures(fixtures)

    assert summary.precision == pytest.approx(1.0)
    assert summary.recall == pytest.approx(1.0)
    assert summary.false_recall_rate == pytest.approx(0.0)
    assert summary.graph_lift_rate == pytest.approx(0.0)
    assert summary.surfaced_to_used_ratio == pytest.approx(1.25)
    assert summary.family_trigger_counts["pragmatic"] == 1
    assert summary.family_trigger_counts["structural"] == 3
    assert summary.need_type_hits["fact_lookup"] == 1
    assert summary.need_type_hits["temporal_update"] == 1
    assert summary.need_type_hits["project_state"] == 1
    assert summary.need_type_hits["broad_context"] == 1


@pytest.mark.asyncio
async def test_evaluate_memory_need_fixtures_supports_graph_probe_cases():
    graph_probe = AsyncMock()
    graph_probe.probe = AsyncMock(
        return_value=ProbeResult(
            resonance_score=0.72,
            detected_entities=["ent_will"],
            entity_scores={"ent_will": 0.72},
        )
    )
    fixtures = [
        MemoryNeedFixture(
            text="Will scored",
            expect_recall=True,
            expected_need_type="fact_lookup",
            expected_family="pragmatic",
            cfg_overrides={"recall_need_graph_probe_enabled": True},
            graph_probe=graph_probe,
        )
    ]

    summary = await evaluate_memory_need_fixtures(fixtures)

    assert summary.precision == pytest.approx(1.0)
    assert summary.recall == pytest.approx(1.0)
    assert summary.graph_lift_rate == pytest.approx(1.0)
    graph_probe.probe.assert_awaited_once()
