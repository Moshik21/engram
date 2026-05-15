from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.graph_state import (
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
