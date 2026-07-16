"""Cold Decision/identity path must not wait on hybrid first."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from engram.config import ActivationConfig
from engram.retrieval.recall_surface import (
    _run_explicit_recall_with_budget,
    build_api_recall_surface,
)


@pytest.mark.asyncio
async def test_durable_entity_first_skips_hybrid_search() -> None:
    """When exact-name Decision exists, hybrid manager.recall is never called."""
    decision = SimpleNamespace(
        id="ent_dec_1",
        name="Cold Decision hit requires healthy search index",
        entity_type="Decision",
        summary="Product continuity probe",
        identity_core=True,
    )
    graph = SimpleNamespace(
        find_entities_exact_name=AsyncMock(return_value=[decision]),
        find_entity_candidates=AsyncMock(return_value=[]),
    )
    manager = SimpleNamespace(
        _graph=graph,
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        search_entities=AsyncMock(return_value={"entities": []}),
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=True, max_packets=3),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=2000,
            recall_fast_preflight_enabled=True,
            recall_fast_preflight_timeout_ms=50,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
    )

    results, metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id="default",
        query="Cold Decision hit requires healthy search index",
        limit=5,
        cfg=manager.get_memory_need_config(),
        operation_source="axi_recall",
    )

    assert results, "durable first path must return Decision hit"
    assert results[0]["entity"]["id"] == "ent_dec_1"
    assert metadata["fallback_status"] == "durable_entity_first"
    assert "durable_entity_first" in (metadata.get("stage_timings_ms") or {})
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_surface_durable_first_stage_timing() -> None:
    decision = SimpleNamespace(
        id="ent_dec_2",
        name="Prefer identity core durable facts",
        entity_type="Preference",
        summary="Prefer cheap path",
        identity_core=True,
    )
    graph = SimpleNamespace(
        find_entities_exact_name=AsyncMock(return_value=[decision]),
        find_entity_candidates=AsyncMock(return_value=[]),
    )
    manager = SimpleNamespace(
        _graph=graph,
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        search_entities=AsyncMock(return_value={"entities": []}),
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,  # skip packet assembly
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=2000,
            recall_fast_preflight_enabled=False,
            recall_packets_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        get_recall_need_thresholds=Mock(return_value={}),
    )

    response = await build_api_recall_surface(
        manager,
        group_id="default",
        query="Prefer identity core durable facts",
        limit=3,
        operation_source="axi_recall",
    )

    assert response["status"] == "ok"
    assert response.get("lifecycle", {}).get("fallbackStatus") == "durable_entity_first"
    manager.recall.assert_not_awaited()


class TestDurableFirstIntentGate:
    """The fast path may only REPLACE the deep pipeline for durable-lookup
    queries or high-signal hits — never for any query naming a known entity."""

    def _hit(self, entity_type: str, overlap: float) -> dict:
        return {
            "result_type": "entity",
            "entity": {"id": "e1", "name": "X", "type": entity_type},
            "score_breakdown": {"name_overlap": overlap},
        }

    def test_intent_terms_allow_short_circuit(self):
        from engram.retrieval.recall_surface import _durable_first_short_circuit_allowed

        hits = [self._hit("Project", 1.0)]
        assert _durable_first_short_circuit_allowed("what did we decide about storage", hits)

    def test_broad_durable_types_do_not_short_circuit(self):
        from engram.retrieval.recall_surface import _durable_first_short_circuit_allowed

        # "Engram" (Project) / user (Person) name-match must not swallow recall.
        hits = [self._hit("Project", 1.0), self._hit("Person", 1.0)]
        assert not _durable_first_short_circuit_allowed("how is the Engram dashboard wired", hits)

    def test_high_signal_hit_allows_short_circuit(self):
        from engram.retrieval.recall_surface import _durable_first_short_circuit_allowed

        hits = [self._hit("Decision", 0.8)]
        assert _durable_first_short_circuit_allowed("helix native backend", hits)

    def test_low_overlap_high_signal_does_not(self):
        from engram.retrieval.recall_surface import _durable_first_short_circuit_allowed

        hits = [self._hit("Decision", 0.4)]
        assert not _durable_first_short_circuit_allowed("helix native backend", hits)


class TestRescueAggregateTimeout:
    @pytest.mark.asyncio
    async def test_rescue_bounded_by_aggregate_wall(self):
        import asyncio

        from engram.retrieval import recall_surface

        async def slow_inner(*args, **kwargs):
            await asyncio.sleep(30)

        with patch.object(recall_surface, "_durable_entity_name_rescue_inner", slow_inner):
            started = asyncio.get_event_loop().time()
            hits = await recall_surface._durable_entity_name_rescue(
                object(), group_id="g", query="q", limit=5, timeout_seconds=0.05
            )
            elapsed = asyncio.get_event_loop().time() - started
        assert hits == []
        assert elapsed < 2.0
