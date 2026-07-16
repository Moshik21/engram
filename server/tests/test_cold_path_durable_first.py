"""Cold Decision/identity path must not wait on hybrid first."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
