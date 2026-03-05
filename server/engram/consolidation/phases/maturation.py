"""Maturation phase: graduate entities from episodic to semantic memory tier."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, MaturationRecord, PhaseResult

logger = logging.getLogger(__name__)


def compute_maturity_score(
    episode_count: int,
    temporal_span_days: float,
    rel_diversity: int,
    access_intervals: list[float],
    cfg: ActivationConfig,
) -> float:
    """Compute maturity score from four signals."""
    source_score = min(1.0, episode_count / 10.0)
    temporal_score = min(1.0, temporal_span_days / 90.0)
    richness_score = min(1.0, rel_diversity / 8.0)

    if len(access_intervals) >= 2:
        mean_interval = sum(access_intervals) / len(access_intervals)
        if mean_interval > 0:
            std_dev = (
                sum((x - mean_interval) ** 2 for x in access_intervals)
                / len(access_intervals)
            ) ** 0.5
            cv = std_dev / mean_interval
            regularity_score = max(0.0, 1.0 - cv)
        else:
            regularity_score = 0.0
    else:
        regularity_score = 0.0

    return (
        cfg.maturation_source_weight * source_score
        + cfg.maturation_temporal_weight * temporal_score
        + cfg.maturation_richness_weight * richness_score
        + cfg.maturation_regularity_weight * regularity_score
    )


class MaturationPhase(ConsolidationPhase):
    """Graduate entities from episodic to transitional to semantic memory tier."""

    @property
    def name(self) -> str:
        return "mature"

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
    ) -> tuple[PhaseResult, list[MaturationRecord]]:
        t0 = time.perf_counter()

        if not cfg.memory_maturation_enabled:
            return PhaseResult(
                phase=self.name, status="skipped", duration_ms=_elapsed_ms(t0),
            ), []

        now = time.time()
        min_age_seconds = cfg.maturation_min_age_days * 86400

        candidates = await graph_store.find_entities(
            group_id=group_id, limit=cfg.maturation_max_per_cycle,
        )

        records: list[MaturationRecord] = []
        for entity in candidates:
            if len(records) >= cfg.maturation_max_per_cycle:
                break

            if entity.deleted_at is not None:
                continue

            attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
            old_tier = attrs.get("mat_tier", "episodic")
            if old_tier == "semantic":
                continue

            # Check age
            entity_age_seconds = now - entity.created_at.timestamp()
            if entity_age_seconds < min_age_seconds:
                continue

            # Identity core auto-promotes to semantic
            if getattr(entity, "identity_core", False):
                new_tier = "semantic"
                maturity_score = 1.0
                episode_count = 0
                temporal_span_days = 0.0
                rel_types: list[str] = []
                regularity = 0.0
            else:
                # Gather data
                episode_count = await graph_store.get_entity_episode_count(
                    entity.id, group_id,
                )
                span_result = await graph_store.get_entity_temporal_span(
                    entity.id, group_id,
                )
                if span_result[0] and span_result[1]:
                    try:
                        min_dt = datetime.fromisoformat(span_result[0])
                        max_dt = datetime.fromisoformat(span_result[1])
                        temporal_span_days = (max_dt - min_dt).total_seconds() / 86400.0
                    except (ValueError, TypeError):
                        temporal_span_days = 0.0
                else:
                    temporal_span_days = 0.0

                rel_types = await graph_store.get_entity_relationship_types(
                    entity.id, group_id,
                )

                # Access intervals from activation store
                state = await activation_store.get_activation(entity.id)
                access_intervals: list[float] = []
                if state and state.access_history and len(state.access_history) >= 2:
                    sorted_hist = sorted(state.access_history)
                    access_intervals = [
                        sorted_hist[i + 1] - sorted_hist[i]
                        for i in range(len(sorted_hist) - 1)
                    ]

                maturity_score = compute_maturity_score(
                    episode_count, temporal_span_days, len(rel_types),
                    access_intervals, cfg,
                )

                # Reconsolidation bonus
                recon_count = attrs.get("recon_count", 0)
                if isinstance(recon_count, (int, float)):
                    maturity_score += min(0.10, recon_count * 0.03)

                # Access regularity for audit
                if len(access_intervals) >= 2:
                    mean_iv = sum(access_intervals) / len(access_intervals)
                    if mean_iv > 0:
                        std_d = (
                            sum((x - mean_iv) ** 2 for x in access_intervals)
                            / len(access_intervals)
                        ) ** 0.5
                        regularity = max(0.0, 1.0 - std_d / mean_iv)
                    else:
                        regularity = 0.0
                else:
                    regularity = 0.0

                # Determine new tier
                if (
                    maturity_score >= cfg.maturation_semantic_threshold
                    and episode_count >= cfg.maturation_min_cycles
                ):
                    new_tier = "semantic"
                elif maturity_score >= cfg.maturation_transitional_threshold:
                    new_tier = "transitional"
                else:
                    continue  # No promotion

            if new_tier == old_tier:
                continue

            if not dry_run:
                attrs["mat_tier"] = new_tier
                attrs["mat_score"] = round(maturity_score, 4)
                await graph_store.update_entity(
                    entity.id,
                    {"attributes": json.dumps(attrs)},
                    group_id,
                )
                if context is not None:
                    context.matured_entity_ids.add(entity.id)
                    context.affected_entity_ids.add(entity.id)

            records.append(
                MaturationRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    entity_id=entity.id,
                    entity_name=entity.name,
                    old_tier=old_tier,
                    new_tier=new_tier,
                    maturity_score=round(maturity_score, 4),
                    source_diversity=episode_count,
                    temporal_span_days=round(temporal_span_days, 2),
                    relationship_richness=len(rel_types),
                    access_regularity=round(regularity, 4),
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
