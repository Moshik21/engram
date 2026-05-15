"""Preference-directed feedback recording for memory reinforcement."""

from __future__ import annotations

import inspect
import uuid
from typing import Any

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.relationship import Relationship


class FeedbackRatingError(ValueError):
    """Raised when a public feedback rating is outside the accepted range."""


async def build_explicit_feedback_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
    rating: int,
    comment: str | None,
) -> dict:
    """Validate and record explicit public feedback through the manager facade."""
    if rating < 1 or rating > 5:
        raise FeedbackRatingError("Rating must be between 1 and 5")
    return await manager.record_explicit_feedback(
        group_id=group_id,
        entity_id=entity_id,
        rating=rating,
        comment=comment,
    )


async def build_mcp_explicit_feedback_surface(
    manager: Any,
    *,
    group_id: str,
    entity_id: str,
    rating: int,
    comment: str | None,
) -> dict:
    """Validate, record, and present MCP explicit feedback."""
    try:
        return await build_explicit_feedback_surface(
            manager,
            group_id=group_id,
            entity_id=entity_id,
            rating=rating,
            comment=comment,
        )
    except FeedbackRatingError as e:
        return {"error": str(e)}


class PreferenceFeedbackRecorder:
    """Record explicit user preference feedback as graph reinforcement edges."""

    def __init__(
        self,
        *,
        graph_store: Any,
        cfg: ActivationConfig,
        event_bus: Any = None,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._event_bus = event_bus

    async def record_explicit_feedback(
        self,
        *,
        group_id: str,
        entity_id: str,
        rating: int,
        comment: str | None = None,
    ) -> dict:
        """Create or strengthen preference edges for an explicitly rated entity."""
        entity = await self._graph.get_entity(entity_id, group_id)
        if entity is None:
            raise ValueError(f"Entity {entity_id} not found")

        domain = self._domain_for_type(entity.entity_type)
        pref_entity = await self._get_or_create_preference_profile(group_id)
        edge_weight = abs(rating - 3) / 2.0
        if rating >= 4:
            edge_type = "PREFERS"
        elif rating <= 2:
            edge_type = "AVOIDS"
        else:
            return {
                "status": "neutral",
                "entity_id": entity_id,
                "domain": domain,
                "edge_type": None,
                "edge_weight": 0.0,
            }

        await self._create_or_strengthen_edge(
            pref_entity_id=pref_entity.id,
            entity_id=entity_id,
            edge_type=edge_type,
            edge_weight=edge_weight,
            group_id=group_id,
        )
        await self._publish_feedback_recorded(
            group_id=group_id,
            entity_id=entity_id,
            rating=rating,
            domain=domain,
            edge_type=edge_type,
        )
        return {
            "status": "recorded",
            "entity_id": entity_id,
            "domain": domain,
            "edge_type": edge_type,
            "edge_weight": edge_weight,
        }

    def _domain_for_type(self, entity_type: str) -> str:
        for domain, types in self._cfg.domain_groups.items():
            if entity_type in types:
                return domain
        return "general"

    async def _get_or_create_preference_profile(self, group_id: str):
        prefs = await self._graph.find_entities(
            name="UserPreference",
            entity_type="PreferenceProfile",
            group_id=group_id,
            limit=1,
        )
        if prefs:
            return prefs[0]

        pref_entity = Entity(
            id=f"pref_{uuid.uuid4().hex[:12]}",
            name="UserPreference",
            entity_type="PreferenceProfile",
            summary="User preference profile for preference-directed memory",
            group_id=group_id,
        )
        await self._graph.create_entity(pref_entity)
        return pref_entity

    async def _create_or_strengthen_edge(
        self,
        *,
        pref_entity_id: str,
        entity_id: str,
        edge_type: str,
        edge_weight: float,
        group_id: str,
    ) -> None:
        existing_rels = await self._graph.get_relationships(
            entity_id=pref_entity_id,
            direction="outgoing",
            predicate=edge_type,
            group_id=group_id,
        )
        existing = next((rel for rel in existing_rels if rel.target_id == entity_id), None)
        if existing:
            new_weight = min(1.0, existing.weight + edge_weight * 0.5)
            await self._graph.update_relationship_weight(
                existing.id,
                new_weight,
                group_id=group_id,
            )
            return

        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:12]}",
            source_id=pref_entity_id,
            target_id=entity_id,
            predicate=edge_type,
            weight=edge_weight,
            group_id=group_id,
        )
        await self._graph.create_relationship(rel)

    async def _publish_feedback_recorded(
        self,
        *,
        group_id: str,
        entity_id: str,
        rating: int,
        domain: str,
        edge_type: str,
    ) -> None:
        if not self._event_bus:
            return
        payload = {
            "entity_id": entity_id,
            "rating": rating,
            "domain": domain,
            "edge_type": edge_type,
        }
        try:
            published = self._event_bus.publish(group_id, "feedback.recorded", payload)
        except TypeError:
            published = self._event_bus.publish("feedback.recorded", payload)
        if inspect.isawaitable(published):
            await published
