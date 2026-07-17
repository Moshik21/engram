from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.identity_core import IdentityCoreService, build_mcp_identity_core_surface


class FakeGraphStore:
    def __init__(self, entities: list[SimpleNamespace]) -> None:
        self.entities = entities
        self.updates: list[tuple[str, dict, str]] = []

    async def find_entities(self, *, name: str, group_id: str, limit: int):
        assert limit == 1
        return [entity for entity in self.entities if entity.name == name][:limit]

    async def update_entity(self, entity_id: str, updates: dict, *, group_id: str) -> None:
        self.updates.append((entity_id, updates, group_id))


@pytest.mark.asyncio
async def test_identity_core_service_marks_entity() -> None:
    entity = SimpleNamespace(id="ent_1", name="Alex", entity_type="Person")
    graph = FakeGraphStore([entity])
    service = IdentityCoreService(graph_store=graph)

    result = await service.mark_identity_core("Alex", identity_core=True, group_id="brain")

    assert result == {
        "status": "updated",
        "entity": "Alex",
        "entity_type": "Person",
        "identity_core": True,
        "message": "Entity 'Alex' marked as identity core.",
    }
    assert graph.updates == [
        (
            "ent_1",
            {"identity_core": 1, "attributes": json.dumps({"mat_tier": "semantic"})},
            "brain",
        )
    ]


@pytest.mark.asyncio
async def test_identity_core_mark_promotes_mat_tier_in_same_write() -> None:
    """Flagging identity_core auto-promotes the entity to the semantic tier."""
    entity = SimpleNamespace(
        id="ent_1", name="Alex", entity_type="Person", attributes={"hobby": "pottery"}
    )
    graph = FakeGraphStore([entity])
    service = IdentityCoreService(graph_store=graph)

    await service.mark_identity_core("Alex", identity_core=True, group_id="brain")

    assert len(graph.updates) == 1
    _, updates, _ = graph.updates[0]
    assert updates["identity_core"] == 1
    attrs = json.loads(updates["attributes"])
    assert attrs["mat_tier"] == "semantic"
    assert attrs["hobby"] == "pottery"  # existing attributes preserved


@pytest.mark.asyncio
async def test_identity_core_mark_skips_tier_write_when_already_semantic() -> None:
    entity = SimpleNamespace(
        id="ent_1", name="Alex", entity_type="Person", attributes={"mat_tier": "semantic"}
    )
    graph = FakeGraphStore([entity])
    service = IdentityCoreService(graph_store=graph)

    await service.mark_identity_core("Alex", identity_core=True, group_id="brain")

    assert graph.updates == [("ent_1", {"identity_core": 1}, "brain")]


@pytest.mark.asyncio
async def test_identity_core_service_unmarks_entity() -> None:
    entity = SimpleNamespace(id="ent_1", name="Alex", entity_type="Person")
    graph = FakeGraphStore([entity])
    service = IdentityCoreService(graph_store=graph)

    result = await service.mark_identity_core("Alex", identity_core=False, group_id="brain")

    assert result["identity_core"] is False
    assert result["message"] == "Entity 'Alex' removed from identity core."
    assert graph.updates == [("ent_1", {"identity_core": 0}, "brain")]


@pytest.mark.asyncio
async def test_identity_core_service_reports_missing_entity() -> None:
    graph = FakeGraphStore([])
    service = IdentityCoreService(graph_store=graph)

    result = await service.mark_identity_core("Missing", group_id="brain")

    assert result == {"status": "error", "message": "Entity 'Missing' not found."}
    assert graph.updates == []


@pytest.mark.asyncio
async def test_mcp_identity_core_surface_forwards_group_and_flag() -> None:
    manager = MagicMock()
    manager.mark_identity_core = AsyncMock(
        return_value={
            "status": "updated",
            "entity": "Alex",
            "identity_core": True,
        }
    )

    result = await build_mcp_identity_core_surface(
        manager,
        group_id="native_brain",
        entity_name="Alex",
        identity_core=True,
    )

    assert result["status"] == "updated"
    manager.mark_identity_core.assert_awaited_once_with(
        "Alex",
        identity_core=True,
        group_id="native_brain",
    )
