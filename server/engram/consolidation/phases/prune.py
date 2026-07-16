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

        # Prune 2.0: age/access-qualified low-value Concepts/Artifacts (budgeted).
        # Still requires zero relationships — never touches identity_core.
        if getattr(cfg, "consolidation_prune_low_value_enabled", True):
            remaining = max(0, max_prunes - len(records))
            if remaining > 0:
                try:
                    from engram.consolidation.hygiene_debt import (
                        LowValueEntityCandidate,
                        select_low_value_prune_candidates,
                    )

                    low_limit = min(
                        remaining,
                        int(getattr(cfg, "consolidation_prune_low_value_max_per_cycle", 50) or 50),
                    )
                    min_lv_age = float(
                        getattr(cfg, "consolidation_prune_low_value_min_age_days", 30.0) or 30.0
                    )
                    max_lv_access = int(
                        getattr(cfg, "consolidation_prune_low_value_max_access", 1) or 1
                    )
                    # Reuse dead-entity scan with slightly looser access — still zero edges.
                    lv_raw = await graph_store.get_dead_entities(
                        group_id=group_id,
                        min_age_days=int(min_lv_age),
                        limit=low_limit * 3,
                        max_access_count=max_lv_access,
                    )
                    already = {r.entity_id for r in records} | goal_neighbor_ids
                    lv_candidates: list[LowValueEntityCandidate] = []
                    for entity in lv_raw or []:
                        if entity.id in already:
                            continue
                        if getattr(entity, "identity_core", False):
                            continue
                        age_days = max(
                            0.0,
                            (now - entity.created_at.timestamp()) / 86400
                            if entity.created_at
                            else min_lv_age + 1,
                        )
                        lv_candidates.append(
                            LowValueEntityCandidate(
                                entity_id=entity.id,
                                entity_type=str(entity.entity_type or ""),
                                access_count=int(entity.access_count or 0),
                                age_days=age_days,
                                identity_core=False,
                                relationship_count=0,
                            )
                        )
                    selected = select_low_value_prune_candidates(
                        lv_candidates,
                        min_age_days=min_lv_age,
                        max_access_count=max_lv_access,
                        max_relationships=0,
                        limit=low_limit,
                    )
                    for cand in selected:
                        if len(records) >= max_prunes:
                            break
                        entity = await graph_store.get_entity(cand.entity_id, group_id)
                        if entity is None or getattr(entity, "identity_core", False):
                            continue
                        # Activation safety net: graph store access_count can be
                        # stale, so honor the live activation record like the
                        # dead-entity pass above before deleting.
                        state = await activation_store.get_activation(cand.entity_id)
                        if state:
                            if state.access_count > max_lv_access:
                                continue
                            act_level = compute_activation(
                                state.access_history,
                                now,
                                cfg,
                                state.consolidated_strength,
                            )
                            if act_level > activation_floor:
                                continue
                        if not dry_run:
                            await graph_store.delete_entity(
                                cand.entity_id, soft=True, group_id=group_id
                            )
                            await activation_store.clear_activation(cand.entity_id)
                            await search_index.remove(cand.entity_id)
                            if context is not None:
                                context.pruned_entity_ids.add(cand.entity_id)
                        records.append(
                            PruneRecord(
                                cycle_id=cycle_id,
                                group_id=group_id,
                                entity_id=cand.entity_id,
                                entity_name=entity.name if entity else cand.entity_id,
                                entity_type=cand.entity_type,
                                reason="low_value_type",
                            )
                        )
                except Exception:
                    logger.debug("Low-value prune expansion failed", exc_info=True)

        return PhaseResult(
            phase=self.name,
            items_processed=len(candidates) + len(records),
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
