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
        get_last_recall_stage_timings=Mock(
            return_value={"graph_expand": 1.25, "recall_retrieve": 2.5},
        ),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram recall",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["query"] == "Engram recall"
    assert result["diagnostics"]["stageTimingsMs"]["recallSearch"] >= 0
    assert result["diagnostics"]["stageTimingsMs"]["graphExpand"] == 1.25
    assert result["diagnostics"]["stageTimingsMs"]["recallRetrieve"] == 2.5
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
    assert result["diagnostics"]["stage_timings_ms"]["recall_search"] >= 0
    assert result["diagnostics"]["stage_timings_ms"]["recall_present"] >= 0
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
    assert result["diagnostics"]["stageTimingsMs"]["recallSearch"] >= 100
    assert result["budget"]["surface"] == "axi"
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "recall"
    assert sample.source == "axi_recall"
    assert sample.status == "degraded"
    assert sample.timeout is True
    assert sample.degraded is True


@pytest.mark.asyncio
async def test_api_recall_surface_runs_fast_cascade_before_deep_recall() -> None:
    calls = []
    cached_packet = {
        "packet_type": "state_packet",
        "title": "Cached state",
        "summary": "Cached packet is first in the cascade.",
    }
    deep_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_deep",
            "content": "Deep recall result",
            "source": "deep",
            "created_at": None,
        },
        "score": 0.9,
        "score_breakdown": {"semantic": 0.9},
        "linked_entities": [],
    }

    def cached_packets(*_args, **_kwargs):
        calls.append("packet_cache")
        return SimpleNamespace(packets=[cached_packet])

    async def fast_fallback(*_args, **_kwargs):
        calls.append("fallback")
        return [
            {
                "result_type": "episode",
                "episode": {"id": "ep_fast", "content": "Fast", "source": "fast"},
                "score": 0.5,
                "score_breakdown": {"semantic": 0.5},
                "linked_entities": [],
            }
        ]

    async def deep_recall(*_args, **_kwargs):
        calls.append("deep")
        return [deep_result]

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=deep_recall),
        fast_recall_fallback=AsyncMock(side_effect=fast_fallback),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(),
        get_cached_memory_packets=Mock(side_effect=cached_packets),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram latency",
        limit=3,
        operation_source="axi_recall",
    )

    assert calls == ["packet_cache", "fallback", "deep"]
    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_deep"
    assert result["packets"] == [cached_packet]
    assert result["lifecycle"]["fallbackStatus"] == "hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    assert result["diagnostics"]["stageTimingsMs"]["packetCache"] >= 0
    assert result["diagnostics"]["stageTimingsMs"]["recallFallback"] >= 0
    assert result["diagnostics"]["stageTimingsMs"]["recallSearch"] >= 0


@pytest.mark.asyncio
async def test_api_recall_surface_uses_fast_fallback_and_cached_packets_on_timeout() -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    cached_packet = {
        "packet_type": "state_packet",
        "title": "Cached state",
        "summary": "Cached packet survives recall timeout.",
    }
    fallback_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_fast",
            "content": "Fast cue fallback about Engram latency.",
            "source": "cue_fallback",
            "created_at": None,
        },
        "score": 0.82,
        "score_breakdown": {"semantic": 0.82},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[fallback_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=100),
        get_cached_memory_packets=Mock(
            return_value=SimpleNamespace(packets=[cached_packet]),
        ),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram latency",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"][0]["episode"]["id"] == "ep_fast"
    assert result["packets"] == [cached_packet]
    assert result["lifecycle"]["timeout"] is True
    assert result["lifecycle"]["fallbackStatus"] == "hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    assert result["diagnostics"]["stageTimingsMs"]["recallFallback"] >= 0
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="Engram latency",
        group_id="native_brain",
        limit=3,
    )
    recorded_operations = [
        call.args[1].operation for call in manager.record_memory_operation.call_args_list
    ]
    assert recorded_operations == ["packet_cache", "recall"]


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
    assert result["diagnostics"]["stage_timings_ms"]["packet_cache"] >= 0
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
    assert result["diagnostics"]["stage_timings_ms"]["packet_assembly"] >= 100
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
