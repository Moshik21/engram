from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.entity_surface import (
    build_api_entity_delete_surface,
    build_api_entity_detail_surface,
    build_api_entity_update_surface,
    entity_not_found_payload,
)


@pytest.mark.asyncio
async def test_entity_detail_surface_reads_active_group() -> None:
    manager = MagicMock()
    manager.get_entity_detail = AsyncMock(return_value={"id": "ent_1", "name": "Alice"})

    result = await build_api_entity_detail_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_1",
    )

    assert result == {"id": "ent_1", "name": "Alice"}
    manager.get_entity_detail.assert_awaited_once_with("ent_1", "native_brain")


@pytest.mark.asyncio
async def test_entity_update_surface_builds_sparse_update_payload() -> None:
    manager = MagicMock()
    manager.update_entity_profile = AsyncMock(
        return_value={"id": "ent_1", "name": "Alice Updated"}
    )

    result = await build_api_entity_update_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_1",
        name="Alice Updated",
        summary=None,
    )

    assert result == {"id": "ent_1", "name": "Alice Updated"}
    manager.update_entity_profile.assert_awaited_once_with(
        "ent_1",
        {"name": "Alice Updated"},
        group_id="native_brain",
    )


@pytest.mark.asyncio
async def test_entity_delete_surface_forwards_group() -> None:
    manager = MagicMock()
    manager.delete_entity_by_id = AsyncMock(return_value={"status": "deleted", "id": "ent_1"})

    result = await build_api_entity_delete_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_1",
    )

    assert result == {"status": "deleted", "id": "ent_1"}
    manager.delete_entity_by_id.assert_awaited_once_with("ent_1", group_id="native_brain")


def test_entity_not_found_payload() -> None:
    assert entity_not_found_payload("ent_missing") == {
        "detail": "Entity 'ent_missing' not found",
    }
