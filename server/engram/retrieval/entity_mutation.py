"""Entity mutation helpers for public REST surfaces."""

from __future__ import annotations

from typing import Any


class EntityMutationService:
    """Own public entity profile updates and soft deletes."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store

    async def update_entity_profile(
        self,
        entity_id: str,
        updates: dict,
        *,
        group_id: str = "default",
    ) -> dict | None:
        entity = await self._graph.get_entity(entity_id, group_id)
        if not entity:
            return None

        if updates:
            await self._graph.update_entity(entity_id, updates, group_id=group_id)

        updated = await self._graph.get_entity(entity_id, group_id)
        if updated is None:
            return None
        return self._entity_summary(updated)

    async def delete_entity_by_id(
        self,
        entity_id: str,
        *,
        group_id: str = "default",
    ) -> dict | None:
        entity = await self._graph.get_entity(entity_id, group_id)
        if not entity:
            return None

        await self._graph.delete_entity(entity_id, soft=True, group_id=group_id)
        await self._activation.clear_activation(entity_id)
        return {"status": "deleted", "id": entity_id, "name": entity.name}

    @staticmethod
    def _entity_summary(entity) -> dict:
        return {
            "id": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "summary": entity.summary,
            "lexicalRegime": entity.lexical_regime,
            "canonicalIdentifier": entity.canonical_identifier,
            "identifierLabel": entity.identifier_label,
            "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
        }
