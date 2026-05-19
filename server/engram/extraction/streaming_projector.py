"""Streaming projector for Project Synapse."""

from __future__ import annotations

import logging
from typing import Any

from engram.events.bus import get_event_bus
from engram.extraction.models import ExtractionResult

logger = logging.getLogger(__name__)


class StreamingEvidenceProjector:
    """Broadcasts incremental evidence discovery events to the event bus.

    Used for 'Real-time Projection' (Phase 5) to give the user immediate
    visual feedback while extraction is still running.
    """

    def __init__(self, group_id: str) -> None:
        self._group_id = group_id
        self._bus = get_event_bus()

    def broadcast_extraction_start(self, episode_id: str) -> None:
        """Signify that a new episode has started extraction."""
        self._bus.publish(
            group_id=self._group_id,
            event_type="streaming.projection_started",
            payload={"episode_id": episode_id},
        )

    def broadcast_result(self, result: ExtractionResult) -> None:
        """Publish incremental chunks of entities and relationships."""
        for entity in result.entities:
            self._bus.publish(
                group_id=self._group_id,
                event_type="streaming.entity_discovered",
                payload={
                    "name": entity.name,
                    "type": entity.entity_type,
                    "summary": entity.summary,
                },
            )

        for rel in result.relationships:
            self._bus.publish(
                group_id=self._group_id,
                event_type="streaming.relationship_discovered",
                payload={
                    "source": rel.subject_text,
                    "predicate": rel.predicate,
                    "target": rel.object_text,
                },
            )

    def broadcast_extraction_complete(self, episode_id: str, stats: dict[str, Any]) -> None:
        """Signify that extraction is done."""
        self._bus.publish(
            group_id=self._group_id,
            event_type="streaming.projection_complete",
            payload={"episode_id": episode_id, "stats": stats},
        )
