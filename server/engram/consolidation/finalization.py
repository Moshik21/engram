"""Post-cycle consolidation finalization services."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from engram.models.consolidation import CycleContext
from engram.utils.dates import utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationFinalizationResult:
    """Summary of non-phase work completed after a consolidation cycle."""

    refreshed_pinned_contexts: int = 0
    invalidated_packet_cache_entries: int = 0

    def event_payload(self) -> dict[str, int]:
        """Return event-safe finalization metrics."""
        return {
            "refreshedPinnedContexts": self.refreshed_pinned_contexts,
            "invalidatedPacketCacheEntries": self.invalidated_packet_cache_entries,
        }


class ConsolidationFinalizationService:
    """Owns successful post-cycle finalization outside the engine loop."""

    def __init__(self, *, graph_manager: object | None) -> None:
        self._graph_manager = graph_manager

    async def refresh_after_cycle(
        self,
        group_id: str,
        *,
        context: CycleContext | None = None,
    ) -> ConsolidationFinalizationResult:
        """Run post-cycle finalizers that should happen after successful consolidation."""
        self._record_storage_count_delta(group_id, context)
        invalidated = self._invalidate_packet_cache(group_id, context)
        refreshed = await self._refresh_pinned_contexts(group_id)
        return ConsolidationFinalizationResult(
            refreshed_pinned_contexts=refreshed,
            invalidated_packet_cache_entries=invalidated,
        )

    def _invalidate_packet_cache(
        self,
        group_id: str,
        context: CycleContext | None,
    ) -> int:
        """Invalidate packet-cache entries touched by consolidation graph mutations."""
        gm = self._graph_manager
        if gm is None or context is None:
            return 0

        invalidate = getattr(gm, "invalidate_memory_packet_cache", None)
        if not callable(invalidate):
            return 0

        entity_ids = set(context.affected_entity_ids)
        entity_ids.update(context.merge_survivor_ids)
        entity_ids.update(context.inferred_edge_entity_ids)
        entity_ids.update(context.pruned_entity_ids)
        entity_ids.update(context.replay_new_entity_ids)
        entity_ids.update(context.dream_seed_ids)
        entity_ids.update(context.dream_association_ids)
        entity_ids.update(context.triage_promoted_ids)
        entity_ids.update(context.matured_entity_ids)
        entity_ids.update(context.schema_entity_ids)
        entity_ids.update(context.microglia_repaired_entity_ids)

        relationship_ids = set(context.microglia_demoted_edge_ids)
        episode_ids = set(context.transitioned_episode_ids)
        if not entity_ids and not relationship_ids and not episode_ids:
            return 0

        return int(
            invalidate(
                group_id,
                entity_ids=sorted(entity_ids) or None,
                episode_ids=sorted(episode_ids) or None,
                relationship_ids=sorted(relationship_ids) or None,
            )
            or 0
        )

    def _record_storage_count_delta(
        self,
        group_id: str,
        context: CycleContext | None,
    ) -> None:
        """Update cached storage counters for known consolidation graph mutations."""
        gm = self._graph_manager
        if gm is None or context is None:
            return
        recorder = getattr(gm, "record_storage_count_delta", None)
        if not callable(recorder):
            return

        new_entity_ids = set(context.replay_new_entity_ids)
        new_entity_ids.update(context.schema_entity_ids)
        entities = len(new_entity_ids) - len(context.pruned_entity_ids)
        relationships = -len(context.microglia_demoted_edge_ids)
        if entities == 0 and relationships == 0:
            return
        try:
            recorder(group_id, entities=entities, relationships=relationships)
        except Exception:
            logger.debug("failed to record consolidation storage count delta", exc_info=True)

    async def _refresh_pinned_contexts(self, group_id: str) -> int:
        """Refresh pinned context intentions after consolidation."""
        gm = self._graph_manager
        if gm is None:
            return 0

        list_intentions = getattr(gm, "list_intentions", None)
        get_context = getattr(gm, "get_context", None)
        update_intention_meta = getattr(gm, "update_intention_meta", None)
        if (
            not callable(list_intentions)
            or not callable(get_context)
            or not callable(update_intention_meta)
        ):
            return 0

        intentions = await list_intentions(group_id=group_id, enabled_only=True)
        refreshed = 0
        for entity in intentions:
            attrs = getattr(entity, "attributes", None)
            if not isinstance(attrs, dict):
                continue
            if attrs.get("trigger_type") != "refresh_context":
                continue
            if attrs.get("refresh_trigger") != "after_consolidation":
                continue

            topic = attrs.get("trigger_text", "")
            if not topic:
                continue

            try:
                context_result = await get_context(
                    group_id=group_id,
                    topic_hint=topic,
                    format="structured",
                )
                pinned_result = _serialize_pinned_context(context_result)
                await update_intention_meta(
                    intention_id=entity.id,
                    group_id=group_id,
                    updates={
                        "pinned_result": pinned_result,
                        "last_refreshed": utc_now_iso(),
                    },
                )
                refreshed += 1
            except Exception:
                logger.debug(
                    "Failed to refresh pinned context %s",
                    getattr(entity, "id", "unknown"),
                )

        if refreshed:
            logger.info("Refreshed %d pinned context(s) for group %s", refreshed, group_id)
        return refreshed


def _serialize_pinned_context(context_result: Any) -> str:
    if isinstance(context_result, dict):
        return json.dumps(context_result)
    return str(context_result)
