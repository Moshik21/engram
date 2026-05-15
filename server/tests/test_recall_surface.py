from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.retrieval.recall_surface import build_mcp_recall_surface


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
