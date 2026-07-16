"""Reindex phase: re-embed entities affected by earlier consolidation phases.

Also backfills active entities missing hybrid vectors when cycle budget remains
(``consolidation_reindex_fill_missing_vectors``), so HNSW/BM25 coverage does not
stay permanently stuck below entity count.
"""

from __future__ import annotations

import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult, ReindexRecord
from engram.storage.index_completeness import backfill_missing_entity_vectors

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
            context = CycleContext()

        # Entities to reindex: affected minus pruned (pruned were already removed)
        to_reindex = set(context.affected_entity_ids - context.pruned_entity_ids)
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
            remaining = max(0, max_per_cycle - len(to_reindex))
            fill_missing = bool(getattr(cfg, "consolidation_reindex_fill_missing_vectors", True))
            if fill_missing and remaining > 0:
                fill = await backfill_missing_entity_vectors(
                    graph_store,
                    search_index,
                    group_id,
                    max_entities=remaining,
                    dry_run=True,
                )
                for eid in fill.indexed_ids or []:
                    # dry_run leaves indexed_ids empty; surface planned via attempted
                    pass
                # Represent planned fill as processed for operator visibility.
                return PhaseResult(
                    phase=self.name,
                    items_processed=len(to_reindex) + fill.attempted,
                    items_affected=0,
                    duration_ms=_elapsed_ms(t0),
                ), records
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

        # Fill permanent hybrid-index gaps with leftover cycle budget.
        fill_missing = bool(getattr(cfg, "consolidation_reindex_fill_missing_vectors", True))
        remaining = max(0, max_per_cycle - len(records))
        if fill_missing and remaining > 0 and not dry_run:
            already = {r.entity_id for r in records}
            try:
                fill = await backfill_missing_entity_vectors(
                    graph_store,
                    search_index,
                    group_id,
                    max_entities=remaining,
                    dry_run=False,
                )
                for eid in fill.indexed_ids:
                    if eid in already:
                        continue
                    entity = await graph_store.get_entity(eid, group_id)
                    name = entity.name if entity else eid
                    records.append(
                        ReindexRecord(
                            cycle_id=cycle_id,
                            group_id=group_id,
                            entity_id=eid,
                            entity_name=name,
                            source_phase="missing_vector",
                        )
                    )
                if fill.indexed:
                    logger.info(
                        "Reindex fill-missing: indexed=%d failed=%d remaining_budget=%d",
                        fill.indexed,
                        fill.failed,
                        remaining,
                    )
            except Exception:
                errors += 1
                logger.warning("Reindex fill-missing backfill failed", exc_info=True)

        if errors:
            logger.info("Reindex: %d errors (affected+fill path)", errors)

        return PhaseResult(
            phase=self.name,
            items_processed=len(to_reindex) + max(0, len(records) - len(to_reindex)),
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
