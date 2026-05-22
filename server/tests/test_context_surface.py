from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.context_builder import (
    MemoryContextBuilder,
    build_api_context_surface,
    build_mcp_context_surface,
    build_mcp_context_tool_surface,
)
from engram.storage.memory.activation import MemoryActivationStore

CONTEXT_RESULT = {
    "context": "## Active Memory\nAlice works on Engram.",
    "entity_count": 2,
    "fact_count": 3,
    "token_estimate": 42,
    "format": "briefing",
}


@pytest.mark.asyncio
async def test_api_context_surface_maps_rest_keys() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_api_context_surface(
        manager,
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
        operation_source="api_context",
    )

    assert payload == {
        "context": "## Active Memory\nAlice works on Engram.",
        "entityCount": 2,
        "factCount": 3,
        "tokenEstimate": 42,
        "format": "briefing",
    }
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
        operation_source="api_context",
    )


@pytest.mark.asyncio
async def test_api_context_surface_forwards_cached_packet_metadata() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(
        return_value={
            **CONTEXT_RESULT,
            "cached_packets": [
                {
                    "packet_type": "project_home",
                    "title": "Project Home: Engram",
                    "trust": {"source": "cache", "freshness": "fresh"},
                }
            ],
            "packet_cache": {
                "hit": True,
                "packet_count": 1,
                "scopes": {"project_home": 1},
            },
        }
    )

    payload = await build_api_context_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
    )

    assert payload["cachedPackets"][0]["title"] == "Project Home: Engram"
    assert payload["cachedPackets"][0]["trust"]["source"] == "cache"
    assert payload["packetCache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }


@pytest.mark.asyncio
async def test_api_context_surface_maps_degraded_budget_metadata() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(
        return_value={
            **CONTEXT_RESULT,
            "status": "degraded",
            "budget": {
                "profile": "explicit",
                "surface": "rest",
                "mode": "api_context",
                "max_wall_ms": 2000,
                "duration_ms": 2001.0,
                "budget_miss": True,
                "timeout": True,
                "degraded": True,
                "skip_reason": "context_timeout",
            },
            "lifecycle": {
                "stage": "recall",
                "degraded": True,
                "timeout": True,
                "skip_reason": "context_timeout",
            },
        }
    )

    payload = await build_api_context_surface(manager, group_id="native_brain")

    assert payload["status"] == "degraded"
    assert payload["budget"]["maxWallMs"] == 2000
    assert payload["budget"]["skipReason"] == "context_timeout"
    assert payload["lifecycle"]["skipReason"] == "context_timeout"


@pytest.mark.asyncio
async def test_mcp_context_surface_preserves_raw_manager_shape() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Alice",
    )

    assert payload == CONTEXT_RESULT
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=2000,
        topic_hint="Alice",
        project_path=None,
        format="structured",
        operation_source="mcp_context",
    )


@pytest.mark.asyncio
async def test_mcp_context_surface_degrades_on_timeout() -> None:
    async def slow_get_context(**_kwargs):
        await asyncio.sleep(0.15)
        return CONTEXT_RESULT

    manager = MagicMock()
    manager.get_activation_config.return_value = ActivationConfig(recall_budget_explicit_ms=100)
    manager.get_context = AsyncMock(side_effect=slow_get_context)
    manager.record_memory_operation = MagicMock()

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Engram",
        operation_source="mcp_context",
    )

    assert payload["status"] == "degraded"
    assert payload["entity_count"] == 0
    assert payload["budget"]["skip_reason"] == "context_timeout"
    assert payload["budget"]["timeout"] is True
    assert payload["lifecycle"]["degraded"] is True
    group_id, sample = manager.record_memory_operation.call_args.args
    assert group_id == "native_brain"
    assert sample.operation == "context"
    assert sample.source == "mcp_context"
    assert sample.timeout is True


@pytest.mark.asyncio
async def test_mcp_context_tool_surface_runs_middleware_with_context_hint() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)
    recall_middleware = AsyncMock()

    payload = await build_mcp_context_tool_surface(
        manager,
        group_id="native_brain",
        max_tokens=900,
        topic_hint=None,
        project_path="/tmp/engram",
        format="briefing",
        recall_middleware=recall_middleware,
    )

    assert payload == CONTEXT_RESULT
    recall_middleware.assert_awaited_once_with(
        "/tmp/engram",
        payload,
        tool_name="get_context",
    )


@pytest.mark.asyncio
async def test_memory_context_builder_returns_when_topic_recall_times_out() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    activation = MemoryActivationStore(cfg=ActivationConfig())

    async def slow_recall(**_kwargs):
        import asyncio

        await asyncio.sleep(1)
        return []

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(),
        recall=slow_recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    builder._CONTEXT_RECALL_TIMEOUT_SECONDS = 0.01

    result = await builder.get_context(group_id="brain", topic_hint="Engram")

    assert result["format"] == "structured"
    assert result["entity_count"] == 0
    assert "## Recent Activity" in result["context"]


@pytest.mark.asyncio
async def test_memory_context_builder_uses_budget_to_cap_project_expansion() -> None:
    graph = MagicMock()
    activation = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    project = Entity(id="ent_project", name="Engram", entity_type="Project", group_id="brain")
    neighbors = [
        (
            Entity(
                id=f"ent_neighbor_{index}",
                name=f"Neighbor {index}",
                entity_type="Artifact",
                summary="Project artifact",
                group_id="brain",
            ),
            MagicMock(),
        )
        for index in range(8)
    ]
    graph.find_entities = AsyncMock(return_value=[project])
    graph.get_neighbors = AsyncMock(return_value=neighbors)
    graph.get_relationships = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    recall = AsyncMock(return_value=[])

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg,
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=200,
    )

    recall.assert_awaited_once_with(query="Engram", group_id="brain", limit=1)
    activation.get_top_activated.assert_awaited_once_with(group_id="brain", limit=2)
    assert "Neighbor 0" in result["context"]
    assert "Neighbor 1" in result["context"]
    assert "Neighbor 2" not in result["context"]


@pytest.mark.asyncio
async def test_memory_context_builder_includes_cached_project_packets() -> None:
    graph = MagicMock()
    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.get_entity = AsyncMock(return_value=None)
    graph.get_relationships = AsyncMock(return_value=[])
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])
    cache_packets = MagicMock()
    recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Memory runtime",
                },
                "score_breakdown": {},
            }
        ]
    )

    def get_cached_packets(_group_id: str, *, scope: str, **_kwargs):
        if scope == "project_home":
            return SimpleNamespace(
                packets=[
                    {
                        "packet_type": "project_home",
                        "title": "Project Home: Engram",
                        "summary": "Cached Engram project packet.",
                        "why_now": "Project startup context.",
                        "trust": {
                            "source": "cache",
                            "freshness": "recent",
                            "why_now": "Project startup context.",
                        },
                    }
                ]
            )
        return None

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False),
        recall=recall,
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
        get_cached_packets=get_cached_packets,
        cache_packets=cache_packets,
    )

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=800,
    )

    assert "## Cached Memory Packets" in result["context"]
    assert "Project Home: Engram" in result["context"]
    assert result["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert result["cached_packets"][0]["trust"]["source"] == "cache"
    assert cache_packets.called


@pytest.mark.asyncio
async def test_memory_context_builder_skips_project_creation_when_lookup_times_out() -> None:
    graph = MagicMock()
    activation = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    async def slow_find_entities(**_kwargs):
        import asyncio

        await asyncio.sleep(1)
        return []

    graph.find_entities = slow_find_entities
    graph.create_entity = AsyncMock()
    graph.get_neighbors = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_top_activated = AsyncMock(return_value=[])

    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg,
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )
    builder._CONTEXT_GRAPH_LOOKUP_TIMEOUT_SECONDS = 0.01

    result = await builder.get_context(
        group_id="brain",
        project_path="/Users/konnermoshier/Engram",
        max_tokens=200,
    )

    graph.create_entity.assert_not_awaited()
    assert "## Project Context" not in result["context"]
    assert result["format"] == "structured"
