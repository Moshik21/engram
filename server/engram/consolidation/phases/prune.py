"""Prune phase: soft-delete dead entities with no relationships and no access."""

from __future__ import annotations

import logging
import time

from engram.activation.engine import compute_activation
from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult, PruneRecord

logger = logging.getLogger(__name__)


class PrunePhase(ConsolidationPhase):
    """Soft-delete entities that are dead: no relationships, no access, old enough."""

    @property
    def name(self) -> str:
        return "prune"

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
    ) -> tuple[PhaseResult, list[PruneRecord]]:
        t0 = time.perf_counter()
        min_age_days = cfg.consolidation_prune_min_age_days
        activation_floor = cfg.consolidation_prune_activation_floor
        max_prunes = cfg.consolidation_prune_max_per_cycle
        now = time.time()

        candidates = await graph_store.get_dead_entities(
            group_id=group_id,
            min_age_days=min_age_days,
            limit=max_prunes,
        )

        records: list[PruneRecord] = []
        for entity in candidates:
            if len(records) >= max_prunes:
                break

            # Double-check activation (graph store access_count may be stale)
            state = await activation_store.get_activation(entity.id)
            if state and state.access_count > cfg.consolidation_prune_min_access_count:
                continue
            if state:
                act_level = compute_activation(
                    state.access_history, now, cfg,
                    state.consolidated_strength,
                )
                if act_level > activation_floor:
                    continue

            if not dry_run:
                await graph_store.delete_entity(entity.id, soft=True, group_id=group_id)
                await activation_store.clear_activation(entity.id)
                await search_index.remove(entity.id)

                # Track pruned entities (NOT added to affected_entity_ids)
                if context is not None:
                    context.pruned_entity_ids.add(entity.id)

            records.append(PruneRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                entity_id=entity.id,
                entity_name=entity.name,
                entity_type=entity.entity_type,
                reason="dead_entity",
            ))

        return PhaseResult(
            phase=self.name,
            items_processed=len(candidates),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
