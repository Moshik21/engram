from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.retrieval.recall_surface import (
    build_mcp_explicit_recall_tool_surface,
    build_mcp_recall_surface,
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
