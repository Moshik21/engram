"""REST entity detail and mutation surface helpers."""

from __future__ import annotations

from typing import Any


async def build_api_entity_detail_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
) -> dict | None:
    """Read an entity detail view through the manager facade."""
    return await manager.get_entity_detail(entity_id, group_id)


async def build_api_entity_update_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
    name: str | None = None,
    summary: str | None = None,
) -> dict | None:
    """Update public entity profile fields through the manager facade."""
    updates: dict[str, str] = {}
    if name is not None:
        updates["name"] = name
    if summary is not None:
        updates["summary"] = summary
    return await manager.update_entity_profile(entity_id, updates, group_id=group_id)


async def build_api_entity_delete_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
) -> dict | None:
    """Soft-delete an entity through the manager facade."""
    return await manager.delete_entity_by_id(entity_id, group_id=group_id)


def entity_not_found_payload(entity_id: str) -> dict:
    """Return the shared REST 404 payload for entity surfaces."""
    return {"detail": f"Entity '{entity_id}' not found"}
