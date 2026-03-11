"""Semantic transition phase: graduate episodes from episodic to semantic tier."""

from __future__ import annotations

import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult, SemanticTransitionRecord

logger = logging.getLogger(__name__)


class SemanticTransitionPhase(ConsolidationPhase):
    """Promote episodes based on entity coverage and consolidation cycle count."""

    @property
    def name(self) -> str:
        return "semanticize"

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
    ) -> tuple[PhaseResult, list[SemanticTransitionRecord]]:
        t0 = time.perf_counter()

        if not cfg.episode_transition_enabled:
            return PhaseResult(
                phase=self.name, status="skipped", duration_ms=_elapsed_ms(t0),
            ), []

        episodes = await graph_store.get_episodes(
            group_id=group_id, limit=cfg.episode_transition_max_per_cycle,
        )

        records: list[SemanticTransitionRecord] = []
        processed = 0

        for episode in episodes:
            status_val = (
                episode.status.value if hasattr(episode.status, "value") else episode.status
            )
            if status_val != "completed":
                continue

            if episode.memory_tier == "semantic":
                continue

            processed += 1
            old_tier = episode.memory_tier
            new_cycles = episode.consolidation_cycles + 1

            # Compute entity coverage
            linked_entity_ids = await graph_store.get_episode_entities(episode.id)
            if not linked_entity_ids:
                coverage = 0.0
            else:
                mature_count = 0
                for eid in linked_entity_ids:
                    if context is not None and eid in context.matured_entity_ids:
                        mature_count += 1
                        continue
                    ent = await graph_store.get_entity(eid, group_id)
                    if ent:
                        attrs = ent.attributes if isinstance(ent.attributes, dict) else {}
                        mat_tier = attrs.get("mat_tier", "episodic")
                        if (
                            mat_tier in ("transitional", "semantic")
                            or _context_marks_entity_mature(context, eid, cfg)
                        ):
                            mature_count += 1
                coverage = mature_count / len(linked_entity_ids)

            # Determine new tier
            new_tier = old_tier
            if (
                coverage >= cfg.episode_semantic_coverage
                and new_cycles >= cfg.episode_semantic_min_cycles
            ):
                new_tier = "semantic"
            elif (
                coverage >= cfg.episode_transitional_coverage
                and new_cycles >= cfg.episode_transitional_min_cycles
            ):
                if old_tier == "episodic":
                    new_tier = "transitional"

            updates = {
                "consolidation_cycles": new_cycles,
                "entity_coverage": round(coverage, 4),
            }
            if new_tier != old_tier:
                updates["memory_tier"] = new_tier

            if not dry_run:
                await graph_store.update_episode(episode.id, updates, group_id)

            if new_tier != old_tier:
                if context is not None:
                    context.transitioned_episode_ids.add(episode.id)

                records.append(
                    SemanticTransitionRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        episode_id=episode.id,
                        old_tier=old_tier,
                        new_tier=new_tier,
                        entity_coverage=round(coverage, 4),
                        consolidation_cycles=new_cycles,
                    )
                )

        return PhaseResult(
            phase=self.name,
            items_processed=processed,
            items_affected=len(records),
            duration_ms=_elapsed_ms(t0),
        ), records


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _context_marks_entity_mature(
    context: CycleContext | None,
    entity_id: str,
    cfg: ActivationConfig,
) -> bool:
    if context is None:
        return False
    bundle = context.maturity_feature_cache.get(entity_id)
    if not isinstance(bundle, dict):
        return False
    score = bundle.get("maturity_score")
    if isinstance(score, (int, float)):
        return float(score) >= cfg.maturation_transitional_threshold
    if isinstance(score, str):
        try:
            return float(score) >= cfg.maturation_transitional_threshold
        except ValueError:
            return False
    return False
