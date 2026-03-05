"""Reindex phase: re-embed entities affected by earlier consolidation phases."""

from __future__ import annotations

import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult, ReindexRecord

logger = logging.getLogger(__name__)


class ReindexPhase(ConsolidationPhase):
    """Re-embed entities that were modified by merge or infer phases."""

    @property
    def name(self) -> str:
        return "reindex"

    async def execute(
        self,
        group_id: str,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[ReindexRecord]]:
        t0 = time.perf_counter()

        if context is None:
            return PhaseResult(
                phase=self.name,
                items_processed=0,
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        # Entities to reindex: affected minus pruned (pruned were already removed)
        to_reindex = context.affected_entity_ids - context.pruned_entity_ids
        max_per_cycle = cfg.consolidation_reindex_max_per_cycle

        # Cap at max_per_cycle
        if len(to_reindex) > max_per_cycle:
            to_reindex = set(list(to_reindex)[:max_per_cycle])

        if dry_run:
            # Determine source_phase for each entity
            records: list[ReindexRecord] = []
            for entity_id in to_reindex:
                source = _source_phase(entity_id, context)
                entity = await graph_store.get_entity(entity_id, group_id)
                name = entity.name if entity else "unknown"
                records.append(
                    ReindexRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        entity_id=entity_id,
                        entity_name=name,
                        source_phase=source,
                    )
                )
            return PhaseResult(
                phase=self.name,
                items_processed=len(to_reindex),
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), records

        # Collect entities for batch embedding
        entities_to_reindex = []
        source_map: dict[str, str] = {}
        for entity_id in to_reindex:
            entity = await graph_store.get_entity(entity_id, group_id)
            if entity:
                entities_to_reindex.append(entity)
                source_map[entity_id] = _source_phase(entity_id, context)
            else:
                logger.debug("Reindex: entity %s not found, skipping", entity_id)

        # Batch embed + store (reduces O(N) API calls to O(N/batch_size))
        records = []
        errors = 0
        if entities_to_reindex:
            use_batch = hasattr(search_index, "batch_index_entities")
            if use_batch:
                try:
                    count = await search_index.batch_index_entities(entities_to_reindex)
                    for entity in entities_to_reindex[:count]:
                        records.append(
                            ReindexRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                entity_id=entity.id,
                                entity_name=entity.name,
                                source_phase=source_map.get(entity.id, "unknown"),
                            )
                        )
                except Exception:
                    errors += 1
                    logger.warning(
                        "Batch reindex failed, falling back to one-at-a-time",
                        exc_info=True,
                    )
                    use_batch = False

            if not use_batch:
                for entity in entities_to_reindex:
                    try:
                        await search_index.index_entity(entity)
                        records.append(
                            ReindexRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                entity_id=entity.id,
                                entity_name=entity.name,
                                source_phase=source_map.get(entity.id, "unknown"),
                            )
                        )
                    except Exception:
                        errors += 1
                        logger.warning(
                            "Reindex failed for entity %s (non-fatal)",
                            entity.id,
                            exc_info=True,
                        )

        if errors:
            logger.info("Reindex: %d/%d entities had errors", errors, len(to_reindex))

        return PhaseResult(
            phase=self.name,
            items_processed=len(to_reindex),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _source_phase(entity_id: str, context: CycleContext) -> str:
    """Determine which phase caused this entity to need reindexing."""
    if entity_id in context.merge_survivor_ids:
        return "merge"
    if entity_id in context.inferred_edge_entity_ids:
        return "infer"
    return "unknown"


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
