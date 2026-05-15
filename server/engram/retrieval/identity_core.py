"""Identity-core mutation helpers."""

from __future__ import annotations

from typing import Any


async def build_mcp_identity_core_surface(
    manager: Any,
    *,
    group_id: str,
    entity_name: str,
    identity_core: bool,
) -> dict:
    """Apply an MCP identity-core mutation and return its public payload."""
    return await manager.mark_identity_core(
        entity_name,
        identity_core=identity_core,
        group_id=group_id,
    )


class IdentityCoreService:
    """Own identity-core protection mutations for a brain group."""

    def __init__(self, *, graph_store: Any) -> None:
        self._graph = graph_store

    async def mark_identity_core(
        self,
        entity_name: str,
        *,
        identity_core: bool = True,
        group_id: str = "default",
    ) -> dict:
        """Mark or unmark an entity as identity core."""
        entities = await self._graph.find_entities(name=entity_name, group_id=group_id, limit=1)
        if not entities:
            return {"status": "error", "message": f"Entity '{entity_name}' not found."}

        entity = entities[0]
        await self._graph.update_entity(
            entity.id,
            {"identity_core": 1 if identity_core else 0},
            group_id=group_id,
        )

        action = "marked as" if identity_core else "removed from"
        return {
            "status": "updated",
            "entity": entity.name,
            "entity_type": entity.entity_type,
            "identity_core": identity_core,
            "message": f"Entity '{entity.name}' {action} identity core.",
        }
