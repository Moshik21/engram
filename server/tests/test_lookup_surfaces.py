from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.lookup import (
    build_api_entity_search_surface,
    build_api_fact_search_surface,
    build_mcp_entity_search_surface,
    build_mcp_entity_search_tool_surface,
    build_mcp_fact_search_surface,
    build_mcp_fact_search_tool_surface,
)

ENTITY_RESULT = {
    "id": "ent_alice",
    "name": "Alice",
    "entity_type": "Person",
    "summary": "Operator",
    "lexical_regime": "named_entity",
    "canonical_identifier": "person:alice",
    "identifier_label": True,
    "activation_score": 0.72,
    "access_count": 4,
    "created_at": "2026-05-15T12:00:00+00:00",
    "updated_at": "2026-05-15T12:30:00+00:00",
}

FACT_RESULT = {
    "subject": "Alice",
    "predicate": "WORKS_AT",
    "object": "Engram",
    "valid_from": "2026-05-01",
    "valid_to": None,
    "confidence": 0.91,
    "source_episode": "ep_123",
    "created_at": "2026-05-15T12:00:00+00:00",
}


@pytest.mark.asyncio
async def test_api_entity_search_surface_maps_camel_case_items() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(return_value=[ENTITY_RESULT])

    payload = await build_api_entity_search_surface(
        manager,
        group_id="native_brain",
        name="Alice",
        entity_type="Person",
        limit=5,
    )

    assert payload["total"] == 1
    assert payload["items"][0]["entityType"] == "Person"
    assert payload["items"][0]["activationCurrent"] == 0.72
    assert payload["items"][0]["identifierLabel"] is True
    manager.search_entities.assert_awaited_once_with(
        group_id="native_brain",
        name="Alice",
        entity_type="Person",
        limit=5,
    )


@pytest.mark.asyncio
async def test_mcp_entity_search_surface_preserves_validation_and_raw_items() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(return_value=[ENTITY_RESULT])

    missing = await build_mcp_entity_search_surface(manager, group_id="native_brain")
    assert missing == {
        "status": "error",
        "message": "At least one of 'name' or 'entity_type' is required.",
    }
    manager.search_entities.assert_not_awaited()

    payload = await build_mcp_entity_search_surface(
        manager,
        group_id="native_brain",
        name="Alice",
        limit=3,
    )

    assert payload == {"entities": [ENTITY_RESULT], "total": 1}
    manager.search_entities.assert_awaited_once_with(
        group_id="native_brain",
        name="Alice",
        entity_type=None,
        limit=3,
    )


@pytest.mark.asyncio
async def test_mcp_entity_search_tool_surface_runs_middleware_after_valid_lookup() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(return_value=[ENTITY_RESULT])
    recall_middleware = AsyncMock()

    missing = await build_mcp_entity_search_tool_surface(
        manager,
        group_id="native_brain",
        recall_middleware=recall_middleware,
    )
    assert missing["status"] == "error"
    recall_middleware.assert_not_awaited()

    payload = await build_mcp_entity_search_tool_surface(
        manager,
        group_id="native_brain",
        name="Alice",
        limit=3,
        recall_middleware=recall_middleware,
    )

    assert payload["entities"] == [ENTITY_RESULT]
    assert payload["total"] == 1
    assert payload["preferRecall"] is True
    assert "deprecationNotice" in payload
    recall_middleware.assert_awaited_once_with(
        "Alice",
        payload,
        tool_name="search_entities",
    )


@pytest.mark.asyncio
async def test_fact_search_surfaces_share_manager_call_with_surface_shapes() -> None:
    manager = MagicMock()
    manager.search_facts = AsyncMock(return_value=[FACT_RESULT])

    api = await build_api_fact_search_surface(
        manager,
        group_id="native_brain",
        query="Alice",
        subject="Alice",
        predicate="works at",
        include_expired=True,
        include_epistemic=True,
        limit=7,
    )
    mcp = await build_mcp_fact_search_surface(
        manager,
        group_id="native_brain",
        query="Alice",
        subject="Alice",
        predicate="works at",
        include_expired=True,
        include_epistemic=True,
        limit=7,
    )

    assert manager.search_facts.await_count == 2
    assert manager.search_facts.await_args.kwargs == {
        "group_id": "native_brain",
        "query": "Alice",
        "subject": "Alice",
        "predicate": "works at",
        "include_expired": True,
        "include_epistemic": True,
        "limit": 7,
    }
    assert api["items"][0]["validFrom"] == "2026-05-01"
    assert api["items"][0]["sourceEpisode"] == "ep_123"
    assert mcp == {"facts": [FACT_RESULT], "total": 1}


@pytest.mark.asyncio
async def test_mcp_fact_search_tool_surface_runs_middleware() -> None:
    manager = MagicMock()
    manager.search_facts = AsyncMock(return_value=[FACT_RESULT])
    recall_middleware = AsyncMock()

    payload = await build_mcp_fact_search_tool_surface(
        manager,
        group_id="native_brain",
        query="Alice",
        subject="Alice",
        predicate="works at",
        include_expired=True,
        include_epistemic=True,
        limit=7,
        recall_middleware=recall_middleware,
    )

    assert payload["facts"] == [FACT_RESULT]
    assert payload["total"] == 1
    assert payload["preferRecall"] is True
    assert "deprecationNotice" in payload
    recall_middleware.assert_awaited_once_with(
        "Alice",
        payload,
        tool_name="search_facts",
    )
