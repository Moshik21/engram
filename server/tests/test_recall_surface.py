from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.config import ActivationConfig
from engram.models.recall import MemoryPacket
from engram.retrieval.recall_surface import (
    build_api_recall_surface,
    build_mcp_explicit_recall_tool_surface,
    build_mcp_recall_surface,
    memory_packet_to_api_dict,
)


@pytest.mark.asyncio
async def test_api_recall_surface_threads_operation_source_into_manager_recall() -> None:
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: SimpleNamespace(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["query"] == "Engram recall"
    manager.recall.assert_awaited_once_with(
        query="Engram recall",
        group_id="native_brain",
        limit=3,
        interaction_type="used",
        interaction_source="axi_recall",
    )


@pytest.mark.asyncio
async def test_mcp_recall_surface_attaches_near_misses_and_surprises() -> None:
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        get_last_near_miss_views=AsyncMock(return_value=[{"entity": "Near Miss"}]),
        get_surprise_connection_views=AsyncMock(return_value=[{"entity": "Surprise"}]),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        cfg=SimpleNamespace(recall_packets_enabled=False),
        resolve_entity_name=AsyncMock(return_value="Entity"),
        get_access_count=AsyncMock(return_value=0),
    )

    assert result["operation"] == "recall"
    assert result["query"] == "Engram recall"
    assert result["lifecycle"]["stage"] == "recall"
    assert result["lifecycle"]["recall_mode"] == "explicit"
    assert result["lifecycle"]["result_count"] == 0
    assert result["lifecycle"]["packet_count"] == 0
    assert result["lifecycle"]["degraded"] is False
    assert result["lifecycle"]["timeout"] is False
    assert result["budget"]["profile"] == "explicit"
    assert result["results"] == []
    assert result["near_misses"] == [{"entity": "Near Miss"}]
    assert result["surprise_connections"] == [{"entity": "Surprise"}]
    manager.recall.assert_awaited_once_with(
        query="Engram recall",
        group_id="native_brain",
        limit=3,
        interaction_type="used",
        interaction_source="mcp_recall",
    )
    manager.get_surprise_connection_views.assert_called_once()
    assert manager.get_surprise_connection_views.call_args.args == ("native_brain",)
    assert manager.get_surprise_connection_views.call_args.kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_api_recall_surface_degrades_when_recall_stage_times_out() -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=100),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"] == []
    assert result["packets"] == []
    assert result["lifecycle"]["degraded"] is True
    assert result["lifecycle"]["timeout"] is True
    assert result["lifecycle"]["skipReason"] == "recall_timeout"
    assert result["budget"]["surface"] == "axi"
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "recall"
    assert sample.source == "axi_recall"
    assert sample.status == "degraded"
    assert sample.timeout is True
    assert sample.degraded is True


@pytest.mark.asyncio
async def test_mcp_recall_surface_owns_entity_name_and_access_count_resolution() -> None:
    manager = SimpleNamespace(
        recall=AsyncMock(
            return_value=[
                {
                    "result_type": "entity",
                    "entity": {"id": "ent_1", "name": "Entity"},
                    "relationships": [
                        {
                            "source_id": "ent_src",
                            "predicate": "RELATED_TO",
                            "target_id": "ent_dst",
                        }
                    ],
                    "score": 0.9,
                }
            ]
        ),
        resolve_entity_name=AsyncMock(side_effect=lambda entity_id, _group_id: entity_id.upper()),
        get_recall_item_access_count=AsyncMock(return_value=4),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        cfg=SimpleNamespace(recall_packets_enabled=False),
    )

    assert result["results"][0]["access_count"] == 4
    assert result["results"][0]["entity_id"] == "ent_1"
    assert result["lifecycle"]["result_count"] == 1
    assert result["results"][0]["related_facts"][0] == {
        "subject": "ENT_SRC",
        "predicate": "RELATED_TO",
        "object": "ENT_DST",
        "polarity": "positive",
    }
    manager.resolve_entity_name.assert_any_await("ent_src", "native_brain")
    manager.resolve_entity_name.assert_any_await("ent_dst", "native_brain")
    manager.get_recall_item_access_count.assert_awaited_once_with("ent_1")


@pytest.mark.asyncio
async def test_mcp_recall_surface_uses_cached_packet_payloads() -> None:
    cached_packet = {
        "packet_type": "state_packet",
        "title": "State: Engram",
        "summary": "Cached project state",
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(
            return_value=SimpleNamespace(packets=[cached_packet])
        ),
        cache_memory_packets=Mock(),
        record_memory_operation=Mock(),
        get_recall_need_thresholds=Mock(return_value=None),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        cfg=SimpleNamespace(recall_packets_enabled=True, recall_packet_explicit_limit=3),
        resolve_entity_name=AsyncMock(return_value="Entity"),
        get_access_count=AsyncMock(return_value=0),
    )

    assert result["packets"] == [cached_packet]
    assert result["lifecycle"]["packet_count"] == 1
    manager.cache_memory_packets.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "packet_cache"
    assert sample.cache_hit is True
    assert sample.packet_count == 1


@pytest.mark.asyncio
async def test_mcp_recall_surface_degrades_when_packet_assembly_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_packet_assembly(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    monkeypatch.setattr(
        "engram.retrieval.recall_surface.assemble_memory_packets",
        slow_packet_assembly,
    )
    manager = SimpleNamespace(
        recall=AsyncMock(
            return_value=[
                {
                    "result_type": "entity",
                    "entity": {"id": "ent_1", "name": "Engram"},
                    "relationships": [],
                    "score": 0.9,
                }
            ]
        ),
        get_cached_memory_packets=Mock(return_value=None),
        cache_memory_packets=Mock(),
        record_memory_operation=Mock(),
        get_recall_need_thresholds=Mock(return_value=None),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=100,
        ),
        resolve_entity_name=AsyncMock(return_value="Entity"),
        get_access_count=AsyncMock(return_value=0),
    )

    assert result["results"]
    assert result["packets"] == []
    assert result["lifecycle"]["packet_count"] == 0
    manager.cache_memory_packets.assert_not_called()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "packet_cache"
    assert sample.status == "degraded"
    assert sample.timeout is True
    assert sample.degraded is True
    assert sample.budget_miss is True
    assert sample.skip_reason == "packet_timeout"


def test_memory_packet_to_api_dict_includes_camel_case_trust_summary() -> None:
    packet = MemoryPacket(
        packet_type="state_packet",
        title="State: Engram",
        summary="Trust is visible",
        why_now="Relevant now.",
        confidence=0.9,
        trust={
            "freshness": "fresh",
            "source": "live_recall",
            "confidence": 0.9,
            "why_now": "Relevant now.",
            "provenance_count": 2,
            "evidence_count": 1,
            "belief_status": "supported",
            "confirmed_count": 1,
            "corrected_count": 1,
            "dismissed_count": 0,
            "last_confirmed_at": "2026-05-21T18:00:00Z",
            "last_corrected_at": "2026-05-21T18:05:00Z",
            "last_dismissed_at": None,
        },
    )

    result = memory_packet_to_api_dict(packet)

    assert result["trust"] == {
        "freshness": "fresh",
        "source": "live_recall",
        "confidence": 0.9,
        "whyNow": "Relevant now.",
        "provenanceCount": 2,
        "evidenceCount": 1,
        "beliefStatus": "supported",
        "confirmedCount": 1,
        "correctedCount": 1,
        "dismissedCount": 0,
        "lastConfirmedAt": "2026-05-21T18:00:00Z",
        "lastCorrectedAt": "2026-05-21T18:05:00Z",
        "lastDismissedAt": None,
    }


@pytest.mark.asyncio
async def test_mcp_explicit_recall_tool_surface_updates_session_and_runs_middleware() -> None:
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
    )
    session = SimpleNamespace(last_recall_time=0.0, auto_recall_primed=False)
    recall_middleware = AsyncMock(
        side_effect=lambda _query, response, **_kwargs: response.update(
            {"recalled_context": {"source": "recall_lite"}}
        )
    )
    perf_values = iter([10.0, 10.1234])

    result = await build_mcp_explicit_recall_tool_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        cfg=SimpleNamespace(recall_packets_enabled=False),
        session=session,
        recall_middleware=recall_middleware,
        perf_counter=lambda: next(perf_values),
        time_source=lambda: 42.5,
    )

    assert result["query_time_ms"] == 123.4
    assert result["recalled_context"] == {"source": "recall_lite"}
    assert session.last_recall_time == 42.5
    assert session.auto_recall_primed is True
    recall_middleware.assert_awaited_once_with("Engram recall", result, tool_name="recall")
