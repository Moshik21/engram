"""Memory forgetting and correction helpers."""

from __future__ import annotations

import logging
from typing import Any

from engram.utils.dates import utc_now, utc_now_iso

logger = logging.getLogger(__name__)


class MemoryForgettingService:
    """Own entity forgetting and fact invalidation mutations."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store

    async def forget_entity(
        self,
        entity_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Soft-delete an entity and clear its activation."""
        entities = await self._graph.find_entities(name=entity_name, group_id=group_id, limit=1)
        if not entities:
            return {"status": "error", "message": f"Entity '{entity_name}' not found."}

        entity = entities[0]
        await self._graph.delete_entity(entity.id, soft=True, group_id=group_id)
        await self._activation.clear_activation(entity.id)

        logger.info("Forgot entity %s (%s), reason: %s", entity.name, entity.id, reason)
        return {
            "status": "forgotten",
            "target_type": "entity",
            "target": entity.name,
            "valid_to": utc_now_iso(),
            "message": f"Entity '{entity.name}' has been forgotten.",
        }

    async def forget_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Invalidate a specific relationship (fact)."""
        predicate = predicate.upper().replace(" ", "_")

        subject_entities = await self._graph.find_entities(
            name=subject_name,
            group_id=group_id,
            limit=1,
        )
        object_entities = await self._graph.find_entities(
            name=object_name,
            group_id=group_id,
            limit=1,
        )

        if not subject_entities:
            return {"status": "error", "message": f"Subject '{subject_name}' not found."}
        if not object_entities:
            return {"status": "error", "message": f"Object '{object_name}' not found."}

        subject_id = subject_entities[0].id
        object_id = object_entities[0].id

        rels = await self._graph.get_relationships(
            subject_id,
            direction="outgoing",
            predicate=predicate,
            active_only=True,
            group_id=group_id,
        )
        target_rel = next((rel for rel in rels if rel.target_id == object_id), None)

        if not target_rel:
            return {
                "status": "error",
                "message": (f"No active fact found: {subject_name} —{predicate}→ {object_name}."),
            }

        await self._graph.invalidate_relationship(target_rel.id, utc_now(), group_id=group_id)

        logger.info(
            "Forgot fact %s —%s→ %s, reason: %s",
            subject_name,
            predicate,
            object_name,
            reason,
        )
        return {
            "status": "forgotten",
            "target_type": "fact",
            "subject": subject_name,
            "predicate": predicate,
            "object": object_name,
            "valid_to": utc_now_iso(),
            "message": f"Fact '{subject_name} {predicate} {object_name}' has been forgotten.",
        }
