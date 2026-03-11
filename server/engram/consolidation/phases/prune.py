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
            max_access_count=cfg.consolidation_prune_min_access_count,
        )

        # Goal prune protection: identify goal-related entities
        goal_neighbor_ids: set[str] = set()
        if cfg.goal_priming_enabled and cfg.goal_prune_protection:
            from engram.retrieval.goals import identify_active_goals

            active_goals = await identify_active_goals(
                graph_store,
                activation_store,
                group_id,
                cfg,
            )
            for goal in active_goals:
                goal_neighbor_ids.add(goal.entity_id)
                goal_neighbor_ids.update(goal.neighbor_ids)

        records: list[PruneRecord] = []
        for entity in candidates:
            if len(records) >= max_prunes:
                break

            # Skip identity core entities (should already be excluded by query,
            # but double-check in case of storage backend differences)
            if getattr(entity, "identity_core", False) is True:
                continue

            # Skip goal-related entities
            if entity.id in goal_neighbor_ids:
                continue

            # Double-check activation (graph store access_count may be stale)
            state = await activation_store.get_activation(entity.id)
            if state and state.access_count > cfg.consolidation_prune_min_access_count:
                continue
            if state:
                act_level = compute_activation(
                    state.access_history,
                    now,
                    cfg,
                    state.consolidated_strength,
                )
                if act_level > activation_floor:
                    continue

                # Emotional prune resistance: emotional entities survive with lower activation
                if cfg.emotional_salience_enabled and cfg.emotional_prune_resistance > 0:
                    ent_data = await graph_store.get_entity(entity.id, group_id)
                    if ent_data:
                        attrs = ent_data.attributes if isinstance(ent_data.attributes, dict) else {}
                        emo_composite = attrs.get("emo_composite", 0.0)
                        if isinstance(emo_composite, (int, float)) and emo_composite > 0.5:
                            adjusted_floor = activation_floor - cfg.emotional_prune_resistance
                            if act_level > adjusted_floor:
                                continue

            # Memory tier prune resistance
            if cfg.memory_maturation_enabled:
                ent_data_mat = await graph_store.get_entity(entity.id, group_id)
                if ent_data_mat:
                    mat_attrs = (
                        ent_data_mat.attributes if isinstance(ent_data_mat.attributes, dict) else {}
                    )
                    mat_tier = mat_attrs.get("mat_tier", "episodic")
                    entity_age_days = (now - entity.created_at.timestamp()) / 86400
                    if mat_tier == "semantic":
                        if entity_age_days < cfg.semantic_prune_age_days:
                            continue
                    elif mat_tier == "transitional":
                        if entity_age_days < cfg.episodic_prune_age_days * 2:
                            continue

            if not dry_run:
                await graph_store.delete_entity(entity.id, soft=True, group_id=group_id)
                await activation_store.clear_activation(entity.id)
                await search_index.remove(entity.id)

                # Track pruned entities (NOT added to affected_entity_ids)
                if context is not None:
                    context.pruned_entity_ids.add(entity.id)

            records.append(
                PruneRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    entity_id=entity.id,
                    entity_name=entity.name,
                    entity_type=entity.entity_type,
                    reason="dead_entity",
                )
            )

        return PhaseResult(
            phase=self.name,
            items_processed=len(candidates),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
