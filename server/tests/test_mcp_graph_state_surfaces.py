from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.graph_state import (
    build_api_graph_neighborhood_surface,
    build_api_temporal_graph_surface,
    build_mcp_entity_neighbors_resource_surface,
    build_mcp_entity_profile_resource_surface,
    build_mcp_graph_state_surface,
    build_mcp_graph_stats_resource_surface,
)


@pytest.mark.asyncio
async def test_mcp_graph_state_surface_forwards_tool_options() -> None:
    manager = MagicMock()
    manager.get_graph_state = AsyncMock(return_value={"stats": {"entities": 3}})

    result = await build_mcp_graph_state_surface(
        manager,
        group_id="native_brain",
        top_n=7,
        include_edges=True,
        entity_types=["Person"],
    )

    assert result == {"stats": {"entities": 3}}
    manager.get_graph_state.assert_awaited_once_with(
        group_id="native_brain",
        top_n=7,
        include_edges=True,
        entity_types=["Person"],
    )


@pytest.mark.asyncio
async def test_mcp_graph_stats_resource_surface_returns_stats_only() -> None:
    manager = MagicMock()
    manager.get_graph_state = AsyncMock(
        return_value={"stats": {"entities": 3}, "top_activated": []}
    )

    result = await build_mcp_graph_stats_resource_surface(manager, group_id="native_brain")

    assert result == {"entities": 3}
    manager.get_graph_state.assert_awaited_once_with(
        group_id="native_brain",
        top_n=10,
        include_edges=False,
        entity_types=None,
    )


@pytest.mark.asyncio
async def test_mcp_entity_resource_surfaces_forward_group() -> None:
    manager = MagicMock()
    manager.get_entity_profile = AsyncMock(return_value={"id": "ent_1"})
    manager.get_entity_neighbors = AsyncMock(return_value={"neighbors": []})

    profile = await build_mcp_entity_profile_resource_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_1",
    )
    neighbors = await build_mcp_entity_neighbors_resource_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_1",
    )

    assert profile == {"id": "ent_1"}
    assert neighbors == {"neighbors": []}
    manager.get_entity_profile.assert_awaited_once_with("ent_1", "native_brain")
    manager.get_entity_neighbors.assert_awaited_once_with("ent_1", "native_brain")


@pytest.mark.asyncio
async def test_api_graph_neighborhood_surface_forwards_options_and_handles_missing() -> None:
    manager = MagicMock()
    manager.get_graph_neighborhood = AsyncMock(return_value={"nodes": []})

    result = await build_api_graph_neighborhood_surface(
        manager,
        group_id="native_brain",
        center="ent_1",
        depth=2,
        max_nodes=50,
        min_activation=0.25,
    )

    assert result.status_code == 200
    assert result.payload == {"nodes": []}
    manager.get_graph_neighborhood.assert_awaited_once_with(
        group_id="native_brain",
        center="ent_1",
        depth=2,
        max_nodes=50,
        min_activation=0.25,
    )

    manager.get_graph_neighborhood = AsyncMock(return_value=None)
    missing = await build_api_graph_neighborhood_surface(
        manager,
        group_id="native_brain",
        center="missing",
        depth=1,
        max_nodes=10,
        min_activation=0.0,
    )

    assert missing.status_code == 404
    assert missing.payload == {"detail": "Entity 'missing' not found"}


@pytest.mark.asyncio
async def test_api_temporal_graph_surface_parses_time_and_handles_errors() -> None:
    manager = MagicMock()
    manager.get_temporal_graph = AsyncMock(return_value={"edges": []})

    result = await build_api_temporal_graph_surface(
        manager,
        group_id="native_brain",
        center="ent_1",
        at="2026-05-15T12:00:00",
        depth=2,
        max_nodes=50,
    )

    assert result.status_code == 200
    assert result.payload == {"edges": []}
    kwargs = manager.get_temporal_graph.await_args.kwargs
    assert kwargs["group_id"] == "native_brain"
    assert kwargs["center"] == "ent_1"
    assert kwargs["at_time"].isoformat() == "2026-05-15T12:00:00"
    assert kwargs["at_label"] == "2026-05-15T12:00:00"
    assert kwargs["depth"] == 2
    assert kwargs["max_nodes"] == 50

    invalid = await build_api_temporal_graph_surface(
        manager,
        group_id="native_brain",
        center="ent_1",
        at="not-time",
        depth=2,
        max_nodes=50,
    )
    assert invalid.status_code == 400
    assert invalid.payload == {"detail": "Invalid ISO 8601 timestamp: 'not-time'"}

    manager.get_temporal_graph = AsyncMock(return_value=None)
    missing = await build_api_temporal_graph_surface(
        manager,
        group_id="native_brain",
        center="missing",
        at="2026-05-15T12:00:00",
        depth=2,
        max_nodes=50,
    )
    assert missing.status_code == 404
    assert missing.payload == {"detail": "Entity 'missing' not found"}
