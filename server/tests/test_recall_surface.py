from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.config import ActivationConfig
from engram.models.recall import MemoryPacket
from engram.retrieval import recall_surface as recall_surface_module
from engram.retrieval.context_builder import _PROJECT_FILE_FALLBACK_PACKET_VERSION
from engram.retrieval.recall_surface import (
    _filter_packets_for_query,
    _packets_satisfy_explicit_query,
    build_api_recall_surface,
    build_mcp_explicit_recall_tool_surface,
    build_mcp_recall_surface,
    cached_explicit_recall_packet_payloads,
    memory_packet_to_api_dict,
)


@pytest.mark.asyncio
async def test_project_file_recall_fallback_uses_dedicated_executor(monkeypatch) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    async def fake_run_project_file_executor(function, /, *args, **kwargs):
        calls.append((function.__name__, args, kwargs))
        return (
            [
                {
                    "packet_type": "project_home",
                    "title": "Project File: README.md",
                    "summary": "Executor-isolated recall fallback.",
                    "trust": {"source": "project_file"},
                }
            ],
            2.5,
        )

    monkeypatch.setattr(
        recall_surface_module,
        "_run_project_file_executor",
        fake_run_project_file_executor,
    )

    task = recall_surface_module._start_project_file_recall_fallback_task(
        query="executor isolation",
        project_path="/tmp/Engram",
        max_packets=1,
    )

    assert task is not None
    packets, duration_ms = await task
    assert packets[0]["summary"] == "Executor-isolated recall fallback."
    assert duration_ms == 2.5
    assert calls == [
        (
            "_build_project_file_recall_fallback_packets",
            (),
            {
                "query": "executor isolation",
                "project_path": "/tmp/Engram",
                "max_packets": 1,
            },
        )
    ]


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
        interaction_type="surfaced",
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
        interaction_type="surfaced",
        interaction_source="mcp_recall",
    )
    manager.get_surprise_connection_views.assert_called_once()
    assert manager.get_surprise_connection_views.call_args.args == ("native_brain",)
    assert manager.get_surprise_connection_views.call_args.kwargs["limit"] == 3


@pytest.mark.asyncio
async def test_api_recall_surface_degrades_when_recall_stage_times_out(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")

    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
        ),
        get_last_recall_stage_timings=Mock(
            return_value={"recall_retrieve_cancelled": 100.0},
        ),
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
    assert result["lifecycle"]["packetCount"] == 1
    assert result["packets"][0]["packet_type"] == "recall_diagnostic"
    assert result["packets"][0]["title"] == "No recalled evidence under budget"
    assert "skip_reason=recall_timeout" in result["packets"][0]["evidence_lines"]
    assert "fallback_status=unavailable" in result["packets"][0]["evidence_lines"]
    assert "recall_retrieve_cancelled_ms=100.0" in result["packets"][0]["evidence_lines"]
    assert result["lifecycle"]["degraded"] is True
    assert result["lifecycle"]["timeout"] is True
    assert result["lifecycle"]["skipReason"] == "recall_timeout"
    assert result["diagnostics"]["stageTimingsMs"]["recallSearch"] >= 100
    assert result["diagnostics"]["stageTimingsMs"]["recallRetrieveCancelled"] == 100.0
    assert result["budget"]["surface"] == "axi"
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "recall"
    assert sample.source == "axi_recall"
    assert sample.status == "degraded"
    assert sample.timeout is True
    assert sample.degraded is True


@pytest.mark.asyncio
async def test_api_recall_surface_filters_irrelevant_fast_fallback_on_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")

    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    fallback_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_irrelevant",
            "content": "A deployment note about another project.",
            "source": "cue_fallback",
            "created_at": None,
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[fallback_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="zzzzquasarflux xylofract wugplinth",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"] == []
    assert result["lifecycle"]["fallbackStatus"] == "filtered"
    assert result["lifecycle"]["fallbackResultCount"] == 0
    assert result["packets"][0]["packet_type"] == "recall_diagnostic"
    assert "fallback_status=filtered" in result["packets"][0]["evidence_lines"]


@pytest.mark.asyncio
async def test_api_recall_surface_does_not_run_fast_fallback_before_successful_recall() -> None:
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
                "episode": {
                    "id": "ep_fast",
                    "content": "Fast fallback about Engram latency.",
                    "source": "fast",
                },
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
        get_memory_need_config=lambda: ActivationConfig(
            recall_fast_preflight_enabled=False,
        ),
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

    assert calls == ["packet_cache", "deep"]
    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_deep"
    assert result["packets"] == [cached_packet]
    assert result["lifecycle"]["fallbackStatus"] == "not_run"
    assert result["lifecycle"]["fallbackResultCount"] == 0
    assert result["diagnostics"]["stageTimingsMs"]["packetCache"] >= 0
    assert result["diagnostics"]["stageTimingsMs"]["recallSearch"] >= 0
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_recall_packet_cache_scope_is_shared_across_sources() -> None:
    cached_packet = {
        "packet_type": "state_packet",
        "title": "State: Engram",
        "summary": "Cached packet is reusable across AXI, REST, and MCP recall.",
    }

    def get_cached_packets(*_args, **kwargs):
        if kwargs["scope"] == "explicit_recall":
            return SimpleNamespace(packets=[cached_packet])
        return None

    manager = SimpleNamespace(
        get_cached_memory_packets=Mock(side_effect=get_cached_packets),
        record_memory_operation=Mock(),
    )
    cfg = ActivationConfig(recall_packet_cache_enabled=True)

    api_packets = await cached_explicit_recall_packet_payloads(
        manager,
        group_id="native_brain",
        query="Engram shared recall cache",
        max_packets=3,
        cfg=cfg,
        operation_source="api_recall",
    )
    mcp_packets = await cached_explicit_recall_packet_payloads(
        manager,
        group_id="native_brain",
        query="Engram shared recall cache",
        max_packets=3,
        cfg=cfg,
        operation_source="mcp_recall",
    )

    assert api_packets == [cached_packet]
    assert mcp_packets == [cached_packet]
    assert [call.kwargs["scope"] for call in manager.get_cached_memory_packets.call_args_list] == [
        "explicit_recall",
        "explicit_recall",
    ]
    samples = [call.args[1] for call in manager.record_memory_operation.call_args_list]
    assert [sample.source for sample in samples] == ["api_recall", "mcp_recall"]
    assert [sample.mode for sample in samples] == ["explicit_recall", "explicit_recall"]
    assert all(sample.cache_hit for sample in samples)


@pytest.mark.asyncio
async def test_api_recall_surface_uses_fast_preflight_on_cache_miss() -> None:
    fallback_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_fast",
            "content": "Engram native PyO3 dogfood performance recall context.",
            "source": "fast",
            "created_at": None,
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[fallback_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=100),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram native PyO3 dogfood performance recall context",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_fast"
    assert result["lifecycle"]["fallbackStatus"] == "fast_preflight_hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    assert result["diagnostics"]["stageTimingsMs"]["recallFastPreflight"] >= 0
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="Engram native PyO3 dogfood performance recall context",
        group_id="native_brain",
        limit=3,
    )


@pytest.mark.asyncio
async def test_fast_preflight_uses_its_own_timeout_budget() -> None:
    async def fallback(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return [fallback_result]

    fallback_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_fast",
            "content": "Engram native PyO3 dogfood performance recall context.",
            "source": "fast",
            "created_at": None,
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(side_effect=fallback),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_fast_fallback_timeout_ms=1,
            recall_fast_preflight_timeout_ms=200,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram native PyO3 dogfood performance recall context",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_fast"
    assert result["lifecycle"]["fallbackStatus"] == "fast_preflight_hit"
    manager.recall.assert_not_awaited()


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
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
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
async def test_api_recall_surface_uses_fast_fallback_when_success_has_low_overlap() -> None:
    weak_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_weak",
            "content": "An older memory value note about a related plan.",
            "source": "deep",
            "created_at": None,
        },
        "score": 0.9,
        "score_breakdown": {"semantic": 0.9},
        "linked_entities": [],
    }
    fallback_result = {
        "result_type": "episode",
        "episode": {
            "id": "ep_exact",
            "content": (
                "Post-fix validation note storage diagnostics skip count scans "
                "capture writes responsive."
            ),
            "source": "cue_fallback",
            "created_at": None,
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[weak_result]),
        fast_recall_fallback=AsyncMock(return_value=[fallback_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_fast_preflight_enabled=False,
        ),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query=(
            "Post-fix validation note storage diagnostics skip count scans "
            "capture writes responsive"
        ),
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_exact"
    assert result["lifecycle"]["fallbackStatus"] == "quality_rescue_hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    assert result["diagnostics"]["stageTimingsMs"]["recallLowOverlapFallback"] >= 0
    manager.fast_recall_fallback.assert_awaited_once_with(
        query=(
            "Post-fix validation note storage diagnostics skip count scans "
            "capture writes responsive"
        ),
        group_id="native_brain",
        limit=3,
    )


@pytest.mark.asyncio
async def test_api_recall_surface_bounds_slow_fast_fallback_on_timeout(monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")

    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    async def slow_fallback(*_args, **_kwargs):
        await asyncio.sleep(1.0)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(side_effect=slow_fallback),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_fallback_timeout_ms=25,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="nomatch recall tail",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"] == []
    assert result["lifecycle"]["fallbackStatus"] == "timeout"
    assert result["lifecycle"]["fallbackResultCount"] == 0
    assert result["diagnostics"]["stageTimingsMs"]["recallFallback"] < 150
    assert result["packets"][0]["packet_type"] == "recall_diagnostic"
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="nomatch recall tail",
        group_id="native_brain",
        limit=3,
    )


@pytest.mark.asyncio
async def test_api_recall_surface_uses_context_packets_when_deep_recall_times_out() -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    context_packet = {
        "packet_type": "project_home",
        "title": "Project Home: Engram",
        "summary": "Cached project packet covers Engram latency work.",
    }
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
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
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["packets"] == [context_packet]
    assert result["lifecycle"]["timeout"] is True
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("session_recent",),
        limit_packets=3,
        sync_persistent=False,
    )
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("identity_core",),
        limit_packets=3,
        sync_persistent=False,
    )
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("project_home",),
        limit_packets=6,
        sync_persistent=False,
    )
    recorded_modes = [call.args[1].mode for call in manager.record_memory_operation.call_args_list]
    assert "context_packet_cache_preflight" in recorded_modes


@pytest.mark.asyncio
async def test_api_recall_surface_deduplicates_context_packet_fallback() -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/install/helix.md",
        "summary": "Helix install docs cover native PyO3.",
        "provenance": ["file:docs/install/helix.md"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet, dict(context_packet)]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Helix",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["packets"] == [context_packet]


@pytest.mark.asyncio
async def test_api_recall_surface_returns_recent_context_packets_on_degraded_cache_miss() -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/memory-value-latency-plan.md",
        "summary": "Native PyO3 dogfood runtime packet cache evidence.",
        "provenance": ["file:docs/memory-value-latency-plan.md"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="unmatched quasarflux",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["packets"] == [context_packet]
    assert result["lifecycle"]["timeout"] is True
    manager.recall.assert_awaited_once()
    recorded_modes = [call.args[1].mode for call in manager.record_memory_operation.call_args_list]
    assert "context_packet_recent_fallback" in recorded_modes


@pytest.mark.asyncio
async def test_api_recall_surface_uses_project_files_when_degraded_cache_is_cold(
    tmp_path,
) -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Native PyO3 dogfood recall should return project context when cache is cold.\n"
    )
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 dogfood recall cold cache",
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["packets"][0]["packet_type"] == "project_home"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert "Native PyO3 dogfood recall" in result["packets"][0]["evidence_lines"][0]
    assert result["diagnostics"]["stageTimingsMs"]["projectFileRecallFallback"] >= 0
    recorded_modes = [call.args[1].mode for call in manager.record_memory_operation.call_args_list]
    assert "project_file_recall_fallback" in recorded_modes


@pytest.mark.asyncio
async def test_api_recall_surface_prebuilds_project_file_fallback_before_timeout(
    monkeypatch,
    tmp_path,
) -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Prebuilt project-file fallback should be ready when recall times out.\n"
    )
    packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/memory-value-latency-plan.md",
        "summary": "Prebuilt fallback packet.",
        "provenance": ["file:docs/memory-value-latency-plan.md"],
    }
    calls: list[bool | None] = []

    def fake_project_file_payloads(*_args, **kwargs):
        calls.append(kwargs.get("cache"))
        time.sleep(0.02)
        return [packet]

    import engram.retrieval.recall_surface as recall_surface

    monkeypatch.setattr(
        recall_surface,
        "project_file_fallback_packet_payloads",
        fake_project_file_payloads,
    )
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_fallback_timeout_ms=1,
            recall_packet_cache_enabled=True,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 dogfood recall cold cache",
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    timings = result["diagnostics"]["stageTimingsMs"]
    assert result["status"] == "degraded"
    assert result["packets"] == [packet]
    assert calls == [False]
    assert timings["projectFileRecallFallback"] < 100
    assert timings["projectFileRecallFallbackWait"] < 50
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert (
        "project_home",
        "native PyO3 dogfood recall cold cache",
        str(tmp_path),
    ) in cache_keys
    assert ("project_home", tmp_path.name, str(tmp_path)) in cache_keys
    assert all(
        call.kwargs.get("persist") is True for call in manager.cache_memory_packets.call_args_list
    )


@pytest.mark.asyncio
async def test_api_recall_surface_uses_project_cwd_when_degraded_cache_is_cold(
    tmp_path,
    monkeypatch,
) -> None:
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    (tmp_path / "README.md").write_text("# Engram\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Cold degraded recall can synthesize project packets from cwd.\n"
    )
    monkeypatch.chdir(tmp_path)
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="cold degraded recall project cwd",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"


@pytest.mark.asyncio
async def test_api_recall_surface_skips_deep_search_when_memory_cache_satisfies_query() -> None:
    context_packet = {
        "packet_type": "episode_packet",
        "title": "Episode: native Helix install",
        "summary": "Stored memory covers native PyO3 Helix install.",
        "episode_ids": ["ep_native_install"],
        "provenance": ["episode:ep_native_install"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 Helix install",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["budget"]["skipReason"] == "cache_satisfied"
    assert result["lifecycle"]["fallbackStatus"] == "cache_satisfied"
    assert result["diagnostics"]["stageTimingsMs"]["cacheSatisfied"] >= 0
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()
    recorded_samples = [call.args[1] for call in manager.record_memory_operation.call_args_list]
    assert any(sample.operation == "recall" and sample.cache_hit for sample in recorded_samples)


def test_explicit_recall_cache_relevance_ignores_generated_why_now() -> None:
    packet = {
        "packet_type": "cue_packet",
        "title": "Latent Memory: ep_old",
        "summary": "Dogfood rolling session proof 20260527 orchid.",
        "episode_ids": ["ep_old"],
        "provenance": ["cue:ep_old"],
        "why_now": (
            "Relevant to the recall query: qvanta noexisting loadedstore miss tail 20260527 probeA"
        ),
    }
    query = "qvanta noexisting loadedstore miss tail 20260527 probeA"

    assert not _packets_satisfy_explicit_query([packet], query=query)
    assert _filter_packets_for_query([packet], query=query, limit=3) == []


def test_project_file_cache_requires_distinctive_marker_for_recall() -> None:
    packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Live dogfood loaded-store recall context evidence from 20260528.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_project_file_fallback_project_path": "/Users/konnermoshier/Engram",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    query = "live dogfood loaded-store recall context trace orchid 20260528 mcp current probe"

    assert _filter_packets_for_query([packet], query=query, limit=3) == []
    assert not _packets_satisfy_explicit_query([packet], query=query)


@pytest.mark.asyncio
async def test_api_recall_surface_uses_session_recent_packet_cache() -> None:
    recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_codex",
        "summary": "Codex dogfood project_path recall adoption should stay fast.",
        "episode_ids": ["ep_codex"],
        "provenance": ["episode:ep_codex", "source:mcp"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
    }
    recent_calls: list[tuple[str, ...]] = []

    def get_recent(
        _group_id: str,
        *,
        scopes: tuple[str, ...],
        limit_packets: int,
        **_kwargs,
    ):
        recent_calls.append(scopes)
        return [recent_packet]

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(side_effect=get_recent),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Codex project_path recall adoption",
        limit=3,
        operation_source="mcp_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [recent_packet]
    assert result["budget"]["skipReason"] == "cache_satisfied"
    assert result["lifecycle"]["fallbackStatus"] == "cache_satisfied"
    assert recent_calls == [
        ("session_recent",),
        ("identity_core",),
        ("project_home",),
    ]
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_does_not_let_project_packets_starve_recent_cache() -> None:
    recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_codex",
        "summary": "User resumed Engram native PyO3 dogfood performance goal again.",
        "episode_ids": ["ep_codex"],
        "provenance": ["episode:ep_codex", "source:mcp"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
    }
    project_packets = [
        {
            "packet_type": "project_home",
            "title": f"Project File: docs/{index}.md",
            "summary": "Engram dogfood performance docs and project packet context.",
            "trust": {"source": "project_file", "freshness": "local"},
        }
        for index in range(8)
    ]

    def get_recent(
        _group_id: str,
        *,
        scopes: tuple[str, ...],
        **_kwargs,
    ):
        if scopes == ("session_recent",):
            return [recent_packet]
        if scopes == ("project_home",):
            return project_packets
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(side_effect=get_recent),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="User resumed Engram native PyO3 dogfood performance goal again",
        limit=3,
        project_path="/Users/konnermoshier/Engram",
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"][0] == recent_packet
    assert result["budget"]["skipReason"] == "cache_satisfied"
    assert result["lifecycle"]["fallbackStatus"] == "cache_satisfied"
    manager.recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_filters_project_file_cache_by_project_path(
    tmp_path,
) -> None:
    engram_project = tmp_path / "Engram"
    machine_project = tmp_path / "MachineShopScheduler"
    engram_project.mkdir()
    machine_project.mkdir()
    wrong_project_packet = {
        "packet_type": "project_home",
        "title": "Project File: package.json",
        "summary": "MachineShopScheduler startup matrix script.",
        "trust": {"source": "project_file", "freshness": "local"},
        "_project_file_fallback_project_path": str(machine_project),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    right_project_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Engram startup matrix 20260527 evidence.",
        "trust": {"source": "project_file", "freshness": "local"},
        "_project_file_fallback_project_path": str(engram_project),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(
            return_value=[wrong_project_packet, right_project_packet]
        ),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="startup matrix 20260527",
        limit=3,
        project_path=str(engram_project),
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    # Project-file cache alone never short-circuits explicit recall; after graph
    # miss, filtered project packets may still surface as fallback.
    assert result["packets"] == [right_project_packet]
    assert result["lifecycle"]["fallbackStatus"] in {
        "project_file_recall_fallback",
        "context_packet_fallback",
    }
    manager.recall.assert_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_ignores_stale_project_file_cache_version(
    tmp_path,
) -> None:
    engram_project = tmp_path / "Engram"
    engram_project.mkdir()
    stale_project_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Noisy stale Engram native PyO3 dogfood performance packet.",
        "trust": {"source": "project_file", "freshness": "local"},
        "_project_file_fallback_project_path": str(engram_project),
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[stale_project_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 dogfood performance",
        limit=3,
        project_path=str(engram_project),
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == []
    assert result["budget"]["skipReason"] is None
    assert result["lifecycle"]["fallbackStatus"] == "miss"
    manager.recall.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_recall_surface_does_not_skip_for_project_file_context_cache() -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/install/helix.md",
        "summary": "Helix install docs cover native PyO3.",
        "provenance": ["file:docs/install/helix.md"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 Helix install",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["budget"]["skipReason"] is None
    assert result["lifecycle"]["fallbackStatus"] == "context_packet_fallback"
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_skips_for_cached_project_file_fallback_packet() -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/install/helix.md",
        "summary": "Generic project-file fallback packet.",
        "trust": {
            "source": "project_file",
            "why": (
                "Synthesized by bounded project-file fallback after recall "
                "returned no usable memory."
            ),
        },
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
        "_project_file_fallback_topic_hint": "native PyO3 Helix install",
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 Helix install",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    # Project-file packets are not cache_satisfied for explicit recall; deep
    # search runs, then packets may reappear as project-file fallback.
    assert result["packets"] == [context_packet]
    assert result["lifecycle"]["fallbackStatus"] in {
        "project_file_recall_fallback",
        "context_packet_fallback",
    }
    manager.recall.assert_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_searches_when_project_file_cache_misses_marker(
    tmp_path,
) -> None:
    engram_project = tmp_path / "Engram"
    engram_project.mkdir()
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/CURRENT_HANDOFF.md",
        "summary": "Live dogfood loaded-store recall context evidence from 20260528.",
        "trust": {"source": "project_file", "freshness": "local"},
        "provenance": ["file:docs/CURRENT_HANDOFF.md"],
        "_project_file_fallback_project_path": str(engram_project),
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query=("live dogfood loaded-store recall context trace orchid 20260528 mcp current probe"),
        limit=3,
        project_path=str(engram_project),
        operation_source="mcp_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["budget"]["skipReason"] is None
    assert result["lifecycle"]["fallbackStatus"] == "context_packet_fallback"
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_skips_when_cached_packets_collectively_satisfy_query() -> None:
    packets = [
        {
            "packet_type": "episode_packet",
            "title": "Episode: native Helix install",
            "summary": "Stored memory covers native PyO3.",
            "episode_ids": ["ep_native"],
            "provenance": ["episode:ep_native"],
        },
        {
            "packet_type": "episode_packet",
            "title": "Episode: recall timeout",
            "summary": "Stored memory covers recall timeout budgets.",
            "episode_ids": ["ep_timeout"],
            "provenance": ["episode:ep_timeout"],
        },
    ]
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=packets),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 recall timeout",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == packets
    assert result["budget"]["skipReason"] == "cache_satisfied"
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_skips_when_cache_matches_separated_compound_terms() -> None:
    context_packet = {
        "packet_type": "episode_packet",
        "title": "Episode: memory value latency",
        "summary": "Stored memory covers dogfood evidence.",
        "evidence_lines": [
            "native PyO3 dogfood cue store timeout deep recall",
            "heading=Memory Value and Latency Plan",
        ],
        "episode_ids": ["ep_latency"],
        "provenance": ["episode:ep_latency"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="cue_store dogfood evidence",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["budget"]["skipReason"] == "cache_satisfied"
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_returns_context_packets_on_empty_success() -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/dogfood-startup-validation-goal.md",
        "summary": "Dogfood evidence mentions cue persistence.",
        "provenance": ["file:docs/dogfood-startup-validation-goal.md"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="cue persistence",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["budget"]["skipReason"] is None
    assert result["lifecycle"]["fallbackStatus"] == "context_packet_fallback"
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_returns_project_packets_on_empty_success(
    tmp_path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Empty loaded-store recall should still return useful project context.\n"
    )
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="empty loaded-store recall project context",
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert result["lifecycle"]["packetCount"] == 1
    assert result["lifecycle"]["fallbackStatus"] == "project_file_recall_fallback"
    assert result["lifecycle"]["degraded"] is True
    assert result["diagnostics"]["stageTimingsMs"]["projectFileRecallFallback"] >= 0
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="empty loaded-store recall project context",
        group_id="native_brain",
        limit=3,
    )
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert (
        "project_home",
        "empty loaded-store recall project context",
        str(tmp_path),
    ) in cache_keys
    assert ("project_home", tmp_path.name, str(tmp_path)) in cache_keys
    assert all(
        call.kwargs.get("persist") is True for call in manager.cache_memory_packets.call_args_list
    )


@pytest.mark.asyncio
async def test_api_recall_surface_waits_for_cold_project_packets_on_empty_success(
    monkeypatch,
    tmp_path,
) -> None:
    packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/memory-value-latency-plan.md",
        "summary": "Cold project fallback packet.",
        "provenance": ["file:docs/memory-value-latency-plan.md"],
    }

    def fake_project_file_payloads(*_args, **_kwargs):
        time.sleep(0.2)
        return [packet]

    import engram.retrieval.recall_surface as recall_surface

    monkeypatch.setattr(
        recall_surface,
        "project_file_fallback_packet_payloads",
        fake_project_file_payloads,
    )
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=1000,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="empty loaded-store recall project context",
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["packets"] == [packet]
    assert result["lifecycle"]["fallbackStatus"] == "project_file_recall_fallback"
    assert result["diagnostics"]["stageTimingsMs"]["projectFileRecallFallback"] >= 200
    assert result["budget"]["budgetMiss"] is False


@pytest.mark.asyncio
async def test_api_recall_surface_uses_project_packets_after_preflight_timeout(
    tmp_path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Explicit recall timeout should return useful project packets.\n",
        encoding="utf-8",
    )

    async def slow_preflight(**_kwargs):
        await asyncio.sleep(0.05)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(side_effect=slow_preflight),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=1000,
            recall_fast_fallback_timeout_ms=1,
            recall_fast_preflight_timeout_ms=1,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="explicit recall timeout useful project packets",
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"], "packets should mirror into items for naive clients"
    assert result.get("resultsSource") == "packets"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    # Soft-hold project files, continue deep recall, then surface project packets.
    assert result["lifecycle"]["fallbackStatus"] == "project_file_recall_fallback"
    assert result["diagnostics"]["stageTimingsMs"]["projectFileRecallFallback"] >= 0
    manager.fast_recall_fallback.assert_awaited_once()
    manager.recall.assert_awaited()


@pytest.mark.asyncio
async def test_api_recall_surface_ignores_weak_session_recent_for_project_file_fallback(
    tmp_path,
) -> None:
    async def slow_recall(**_kwargs):
        await asyncio.sleep(0.2)
        return []

    weak_recent_packet = {
        "packet_type": "recent_observation",
        "title": "Recent Observation: ep_goal",
        "summary": "Dogfood runtime goal resumed.",
        "episode_ids": ["ep_goal"],
        "provenance": ["episode:ep_goal", "source:codex"],
        "trust": {"source": "mcp_observe", "freshness": "fresh"},
        "_cache_scope": "session_recent",
    }
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Dogfood finalize is idempotent after INSERT OR REPLACE metric snapshots.\n"
        "- graph_stats_timeout keeps the human label artifact evaluation bounded.\n"
    )
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=True,
            max_packets=3,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[weak_recent_packet]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query=(
            "dogfood finalize idempotent INSERT OR REPLACE graph_stats_timeout human label artifact"
        ),
        limit=3,
        project_path=str(tmp_path),
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert "Dogfood finalize is idempotent" in result["packets"][0]["summary"]
    assert result["lifecycle"]["fallbackStatus"] == "project_file_recall_fallback"
    assert result["diagnostics"]["stageTimingsMs"]["projectFileRecallFallback"] >= 0
    assert result["packets"][0] != weak_recent_packet


@pytest.mark.asyncio
async def test_api_recall_surface_uses_fast_fallback_on_empty_success() -> None:
    fallback_result = {
        "result_type": "cue_episode",
        "cue": {
            "episode_id": "ep_fast",
            "cue_text": "Engram latency cue",
            "supporting_spans": [],
        },
        "episode": {"id": "ep_fast", "source": "fast", "created_at": None},
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[fallback_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="Engram latency",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["items"][0]["cue"]["episodeId"] == "ep_fast"
    assert result["packets"] == []
    assert result["lifecycle"]["fallbackStatus"] == "hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="Engram latency",
        group_id="native_brain",
        limit=3,
    )


@pytest.mark.asyncio
async def test_api_recall_surface_prefers_project_hits_from_fast_preflight(
    tmp_path,
) -> None:
    project_dir = tmp_path / "Engram"
    project_dir.mkdir()
    wrong_project = {
        "result_type": "episode",
        "episode": {
            "id": "ep_wrong",
            "source": "MachineShopScheduler",
            "content": "MachineShopScheduler insert replace idempotent migration evidence.",
        },
        "score": 1.0,
        "score_breakdown": {"semantic": 1.0},
        "linked_entities": [],
    }
    right_project = {
        "result_type": "episode",
        "episode": {
            "id": "ep_right",
            "source": "project-bootstrap",
            "content": "Engram dogfood finalize idempotent INSERT OR REPLACE evidence.",
        },
        "score": 0.9,
        "score_breakdown": {"semantic": 0.9},
        "linked_entities": [],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[wrong_project, right_project]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=True,
            recall_fast_preflight_timeout_ms=200,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="dogfood finalize idempotent INSERT OR REPLACE",
        limit=3,
        project_path=str(project_dir),
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["lifecycle"]["fallbackStatus"] == "fast_preflight_hit"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0]["episode"]["id"] == "ep_right"
    manager.recall.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_recall_surface_skips_deep_search_when_memory_cache_satisfies_query() -> None:
    context_packet = {
        "packet_type": "episode_packet",
        "title": "Episode: native Helix install",
        "summary": "Stored memory covers native PyO3 Helix install.",
        "episode_ids": ["ep_native_install"],
        "provenance": ["episode:ep_native_install"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="native PyO3 Helix install",
        limit=3,
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["budget"]["skip_reason"] == "cache_satisfied"
    assert result["lifecycle"]["fallback_status"] == "cache_satisfied"
    assert result["diagnostics"]["stage_timings_ms"]["cache_satisfied"] >= 0
    manager.recall.assert_not_awaited()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_recall_surface_returns_context_packets_on_empty_success() -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: docs/dogfood-startup-validation-goal.md",
        "summary": "Dogfood evidence mentions cue persistence.",
        "provenance": ["file:docs/dogfood-startup-validation-goal.md"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="cue persistence",
        limit=3,
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["budget"]["skip_reason"] is None
    assert result["lifecycle"]["fallback_status"] == "context_packet_fallback"
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_recall_surface_uses_context_packets_after_preflight_miss() -> None:
    context_packet = {
        "packet_type": "cue_packet",
        "title": "Latent Memory: active goal",
        "summary": "Current goal tail packet for Engram dogfood continuity.",
        "episode_ids": ["ep_goal_tail"],
        "provenance": ["cue:ep_goal_tail"],
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="zzqx yonderplasm true miss tail",
        limit=3,
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=1000,
            recall_fast_preflight_enabled=True,
            recall_fast_preflight_timeout_ms=200,
        ),
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    # Soft-hold context packets, continue deep recall, then surface as fallback.
    assert result["lifecycle"]["fallback_status"] == "context_packet_fallback"
    assert result["diagnostics"]["stage_timings_ms"]["recall_fast_preflight"] >= 0
    manager.fast_recall_fallback.assert_awaited_once()
    manager.recall.assert_awaited()


@pytest.mark.asyncio
async def test_mcp_recall_surface_uses_project_home_packet_after_preflight_miss(
    tmp_path,
) -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Engram native PyO3 startup notes.",
        "provenance": ["file:README.md"],
        "trust": {"source": "project_file"},
        "_cache_scope": "project_home",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
        "_project_file_fallback_project_path": str(tmp_path),
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="zzqx yonderplasm true miss tail",
        limit=3,
        project_path=str(tmp_path),
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=1000,
            recall_fast_preflight_enabled=True,
            recall_fast_preflight_timeout_ms=200,
        ),
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["lifecycle"]["fallback_status"] in {
        "context_packet_fallback",
        "project_file_recall_fallback",
    }
    manager.fast_recall_fallback.assert_awaited_once()
    manager.recall.assert_awaited()
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("session_recent",),
        limit_packets=3,
        sync_persistent=False,
    )
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("identity_core",),
        limit_packets=3,
        sync_persistent=True,
    )
    manager.get_recent_cached_memory_packets.assert_any_call(
        "native_brain",
        scopes=("project_home",),
        limit_packets=6,
        sync_persistent=True,
    )


@pytest.mark.asyncio
async def test_mcp_recall_surface_uses_context_packets_after_preflight_timeout() -> None:
    context_packet = {
        "packet_type": "project_home",
        "title": "Project File: README.md",
        "summary": "Engram dogfood project home packet.",
        "provenance": ["file:README.md"],
        "trust": {"source": "project_file"},
        "_cache_scope": "project_home",
        "_project_file_fallback_version": _PROJECT_FILE_FALLBACK_PACKET_VERSION,
        "_project_file_fallback_project_path": "/tmp/Engram",
    }

    async def slow_preflight(**_kwargs):
        await asyncio.sleep(0.05)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(side_effect=slow_preflight),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[context_packet]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="zzqx yonderplasm true timeout tail",
        limit=3,
        project_path="/tmp/Engram",
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=1000,
            recall_fast_fallback_timeout_ms=1,
            recall_fast_preflight_timeout_ms=200,
        ),
    )

    assert result["status"] == "ok"
    assert result["packets"] == [context_packet]
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["lifecycle"]["fallback_status"] in {
        "context_packet_fallback",
        "project_file_recall_fallback",
    }
    manager.fast_recall_fallback.assert_awaited_once()
    manager.recall.assert_awaited()


@pytest.mark.asyncio
async def test_mcp_recall_surface_returns_project_packets_on_empty_success(
    tmp_path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- Empty MCP recall should still return useful project context.\n"
    )
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="empty loaded-store recall project context",
        limit=3,
        project_path=str(tmp_path),
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=100,
        ),
    )

    assert result["status"] == "degraded"
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert result["lifecycle"]["packet_count"] == 1
    assert result["lifecycle"]["fallback_status"] == "project_file_recall_fallback"
    assert result["lifecycle"]["degraded"] is True
    assert result["diagnostics"]["stage_timings_ms"]["project_file_recall_fallback"] >= 0
    manager.recall.assert_awaited_once()
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="empty loaded-store recall project context",
        group_id="native_brain",
        limit=3,
    )
    cache_keys = {
        (call.kwargs.get("scope"), call.kwargs.get("topic_hint"), call.kwargs.get("project_path"))
        for call in manager.cache_memory_packets.call_args_list
    }
    assert (
        "project_home",
        "empty loaded-store recall project context",
        str(tmp_path),
    ) in cache_keys
    assert ("project_home", tmp_path.name, str(tmp_path)) in cache_keys
    assert all(
        call.kwargs.get("persist") is True for call in manager.cache_memory_packets.call_args_list
    )


@pytest.mark.asyncio
async def test_mcp_recall_surface_uses_project_packets_after_preflight_timeout(
    tmp_path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "memory-value-latency-plan.md").write_text(
        "# Memory Value and Latency Plan\n\n"
        "- MCP recall timeout should return useful project packets.\n",
        encoding="utf-8",
    )

    async def slow_preflight(**_kwargs):
        await asyncio.sleep(0.05)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[]),
        fast_recall_fallback=AsyncMock(side_effect=slow_preflight),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        get_last_near_miss_views=Mock(return_value=[]),
        get_surprise_connection_views=Mock(return_value=[]),
        cache_memory_packets=Mock(return_value={}),
        record_memory_operation=Mock(),
    )

    result = await build_mcp_recall_surface(
        manager,
        group_id="native_brain",
        query="mcp recall timeout useful project packets",
        limit=3,
        project_path=str(tmp_path),
        cfg=ActivationConfig(
            recall_packets_enabled=True,
            recall_packet_explicit_limit=3,
            recall_budget_explicit_ms=1000,
            recall_fast_fallback_timeout_ms=1,
            recall_fast_preflight_timeout_ms=1,
        ),
    )

    assert result["status"] == "degraded"
    assert result["results"], "packets should mirror into results for naive clients"
    assert result.get("results_source") == "packets"
    assert result["packets"][0]["title"] == "Project File: docs/memory-value-latency-plan.md"
    assert result["lifecycle"]["fallback_status"] == "project_file_recall_fallback"
    assert result["diagnostics"]["stage_timings_ms"]["project_file_recall_fallback"] >= 0
    manager.fast_recall_fallback.assert_awaited_once()
    manager.recall.assert_awaited()


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
        get_cached_memory_packets=Mock(return_value=SimpleNamespace(packets=[cached_packet])),
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


@pytest.mark.asyncio
async def test_mcp_recall_surface_adds_diagnostic_packet_on_empty_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENGRAM_RECALL_PROJECT_FALLBACK", "0")

    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        get_last_near_miss_views=AsyncMock(return_value=[]),
        get_surprise_connection_views=AsyncMock(return_value=[]),
        record_memory_operation=Mock(),
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
    )

    assert result["status"] == "degraded"
    assert result["results"] == []
    assert result["lifecycle"]["packet_count"] == 1
    assert result["packets"][0]["packet_type"] == "recall_diagnostic"
    assert result["packets"][0]["title"] == "No recalled evidence under budget"
    assert "skip_reason=recall_timeout" in result["packets"][0]["evidence_lines"]
    assert "fallback_status=miss" in result["packets"][0]["evidence_lines"]
    assert result["lifecycle"]["timeout"] is True
    assert result["lifecycle"]["skip_reason"] == "recall_timeout"
    assert result["diagnostics"]["stage_timings_ms"]["recall_search"] >= 100


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


@pytest.mark.asyncio
async def test_api_recall_surface_registers_surfaced_cues_for_citation_scan() -> None:
    """The REST/axi lane must feed the mask/cue register like the MCP lane
    does — otherwise cue usage can never accumulate on installs without MCP
    (the capture-census structural-zero finding). Mask-only: response bytes
    are untouched; the buffer learns the surfaced cue."""
    from engram.retrieval.feedback import get_usage_buffer

    buffer = get_usage_buffer()
    buffer.reset()
    cue_result = {
        "result_type": "cue_episode",
        "score": 0.7,
        "cue": {
            "episode_id": "ep_kiln",
            "cue_text": "Melanie booked the kiln for the studio showcase firings",
            "supporting_spans": ["kiln booking confirmed for the showcase"],
        },
        "episode": {"id": "ep_kiln", "source": "mcp_observe"},
    }
    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[cue_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: SimpleNamespace(
            recall_usage_feedback_enabled=True,
        ),
        get_last_recall_stage_timings=Mock(return_value={}),
    )

    await build_api_recall_surface(
        manager,
        group_id="rest_cue_group",
        query="kiln booking",
        limit=3,
        operation_source="axi_recall",
    )

    # Novel reuse of the cue phrase fires (registration happened) ...
    fired = buffer.scan_novel_cue_matches(
        "rest_cue_group",
        "Since Melanie booked the kiln already, plan the second firing early",
        now=time.time() + 60,
    )
    assert [entry.episode_id for entry in fired] == ["ep_kiln"]
    # ... while a verbatim parrot of the surfaced cue text stays masked.
    buffer.reset()


def _partial_episode_result(episode_id: str) -> dict:
    return {
        "result_type": "episode",
        "episode": {
            "id": episode_id,
            "content": "Battery episode candidate materialized before the budget overran.",
            "source": "deep",
            "created_at": None,
        },
        "score": 0.8,
        "score_breakdown": {"semantic": 0.8},
        "linked_entities": [],
    }


@pytest.mark.asyncio
async def test_recall_returns_partial_candidates_on_late_stage_timeout() -> None:
    # Primary search found + materialized candidates, then a downstream stage
    # overran the budget (simulated via a slow recall that wait_for cancels).
    # The salvaged candidates must be returned, and the durable/fast rescue
    # cascade must NOT run.
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    partial = _partial_episode_result("ep_partial")
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_last_recall_stage_timings=Mock(return_value={}),
        get_last_recall_partial_results=Mock(return_value=[partial]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="battery episode candidate",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"][0]["episode"]["id"] == "ep_partial"
    assert result["lifecycle"]["fallbackStatus"] == "partial_on_timeout"
    assert result["lifecycle"]["fallbackResultCount"] == 1
    # Rescue cascade did not fire — the found candidates were preferred.
    manager.fast_recall_fallback.assert_not_awaited()
    group_id, sample = manager.record_memory_operation.call_args.args
    assert sample.status == "degraded"
    assert sample.timeout is True


@pytest.mark.asyncio
async def test_recall_runs_rescue_when_no_partial_candidates_on_timeout() -> None:
    # Primary search found nothing before timing out: the rescue cascade must
    # still run (the partial-return branch must not swallow the empty case).
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_last_recall_stage_timings=Mock(return_value={}),
        get_last_recall_partial_results=Mock(return_value=[]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="nomatch battery tail",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "degraded"
    assert result["items"] == []
    # Rescue cascade DID run because there were no usable candidates.
    manager.fast_recall_fallback.assert_awaited_once_with(
        query="nomatch battery tail",
        group_id="native_brain",
        limit=3,
    )


@pytest.mark.asyncio
async def test_recall_kill_switch_discards_partial_and_runs_rescue() -> None:
    # With the kill switch off, the pre-fix behavior is restored: partial
    # candidates are discarded and the rescue cascade runs.
    async def slow_recall(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []

    partial = _partial_episode_result("ep_partial")
    manager = SimpleNamespace(
        recall=AsyncMock(side_effect=slow_recall),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_last_recall_stage_timings=Mock(return_value={}),
        get_last_recall_partial_results=Mock(return_value=[partial]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=100,
            recall_fast_preflight_enabled=False,
            recall_return_partial_on_timeout=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="battery episode candidate",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["lifecycle"]["fallbackStatus"] != "partial_on_timeout"
    manager.fast_recall_fallback.assert_awaited_once()


@pytest.mark.asyncio
async def test_recall_ok_under_budget_never_consults_partial_getter() -> None:
    # Byte-identical healthy path: a recall that succeeds under budget must
    # never read the partial-results salvage getter, so the flag cannot change
    # the ok-path result.
    deep_result = _partial_episode_result("ep_deep")

    manager = SimpleNamespace(
        recall=AsyncMock(return_value=[deep_result]),
        fast_recall_fallback=AsyncMock(return_value=[]),
        get_last_recall_stage_timings=Mock(return_value={}),
        get_last_recall_partial_results=Mock(return_value=[deep_result]),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(
            enabled=False,
            max_packets=0,
        ),
        get_memory_need_config=lambda: ActivationConfig(
            recall_fast_preflight_enabled=False,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
        record_memory_operation=Mock(),
    )

    result = await build_api_recall_surface(
        manager,
        group_id="native_brain",
        query="battery episode candidate",
        limit=3,
        operation_source="axi_recall",
    )

    assert result["status"] == "ok"
    assert result["items"][0]["episode"]["id"] == "ep_deep"
    assert result["lifecycle"]["fallbackStatus"] == "not_run"
    manager.get_last_recall_partial_results.assert_not_called()
    manager.fast_recall_fallback.assert_not_awaited()


def test_relationship_triple_entity_filter() -> None:
    """Triple-shaped Decisions (graph edges the materializer renders as
    entities) are dropped from the durable rescue; real prose Decisions and
    identities are kept."""
    from engram.retrieval.recall_surface import _is_relationship_triple_entity as f

    # Graph-edge triples -> dropped
    assert f("Engram:recall_profile:all", "Engram -> recall_profile -> all")
    assert f("a:b:c", "")
    assert f("", "Foo -> bar -> baz")
    # Real durable facts -> kept (must not be filtered)
    assert not f("GOLDEN_DECISION_1783643390", "LongMemEval is not product north star")
    assert not f("Konner Moshier", "Founder of Engram; wants fully-local memory")
    assert not f("recall profile", "The recall profile should be set to all for depth")
