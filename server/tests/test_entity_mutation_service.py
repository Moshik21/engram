from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.retrieval.entity_mutation import EntityMutationService


def _entity(name: str = "Alex") -> SimpleNamespace:
    return SimpleNamespace(
        id="ent_alex",
        name=name,
        entity_type="Person",
        summary="Operator",
        lexical_regime="canonical",
        canonical_identifier="alex",
        identifier_label=True,
        created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 15, 1, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_entity_mutation_service_updates_profile() -> None:
    graph = SimpleNamespace()
    graph.get_entity = AsyncMock(side_effect=[_entity(), _entity("Alex Updated")])
    graph.update_entity = AsyncMock()
    activation = SimpleNamespace(clear_activation=AsyncMock())
    service = EntityMutationService(graph_store=graph, activation_store=activation)

    result = await service.update_entity_profile(
        "ent_alex",
        {"name": "Alex Updated"},
        group_id="brain",
    )

    graph.update_entity.assert_awaited_once_with(
        "ent_alex",
        {"name": "Alex Updated"},
        group_id="brain",
    )
    assert result is not None
    assert result["name"] == "Alex Updated"
    assert result["entityType"] == "Person"


@pytest.mark.asyncio
async def test_entity_mutation_service_update_missing_returns_none() -> None:
    graph = SimpleNamespace(get_entity=AsyncMock(return_value=None), update_entity=AsyncMock())
    activation = SimpleNamespace(clear_activation=AsyncMock())
    service = EntityMutationService(graph_store=graph, activation_store=activation)

    assert await service.update_entity_profile("missing", {}, group_id="brain") is None
    graph.update_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_entity_mutation_service_soft_deletes_and_clears_activation() -> None:
    graph = SimpleNamespace()
    graph.get_entity = AsyncMock(return_value=_entity())
    graph.delete_entity = AsyncMock()
    activation = SimpleNamespace(clear_activation=AsyncMock())
    service = EntityMutationService(graph_store=graph, activation_store=activation)

    result = await service.delete_entity_by_id("ent_alex", group_id="brain")

    graph.delete_entity.assert_awaited_once_with("ent_alex", soft=True, group_id="brain")
    activation.clear_activation.assert_awaited_once_with("ent_alex")
    assert result == {"status": "deleted", "id": "ent_alex", "name": "Alex"}


@pytest.mark.asyncio
async def test_entity_mutation_service_delete_missing_returns_none() -> None:
    graph = SimpleNamespace(get_entity=AsyncMock(return_value=None), delete_entity=AsyncMock())
    activation = SimpleNamespace(clear_activation=AsyncMock())
    service = EntityMutationService(graph_store=graph, activation_store=activation)

    assert await service.delete_entity_by_id("missing", group_id="brain") is None
    graph.delete_entity.assert_not_awaited()
    activation.clear_activation.assert_not_awaited()
