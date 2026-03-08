"""Maturation phase: graduate entities from episodic to semantic memory tier."""

from __future__ import annotations

import json
import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.maturity_features import (
    compute_maturity_components,
    extract_maturity_features,
    maturity_bundle_changed,
)
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
    return compute_maturity_components(
        episode_count,
        temporal_span_days,
        rel_diversity,
        access_intervals,
        cfg,
    )["maturity_score"]


class MaturationPhase(ConsolidationPhase):
    """Graduate entities from episodic to transitional to semantic memory tier."""

    @property
    def name(self) -> str:
        return "mature"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.memory_maturation_enabled:
            return set()
        return {
            "get_entity_episode_count",
            "get_entity_temporal_span",
            "get_entity_relationship_types",
        }

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
            else:
                new_tier = old_tier

            bundle = await extract_maturity_features(
                entity,
                graph_store,
                activation_store,
                group_id,
                cfg,
                context=context,
                prefer_cached=False,
            )
            maturity_score = float(bundle["maturity_score"])
            episode_count = int(bundle["episode_count"])
            temporal_span_days = float(bundle["temporal_span_days"])
            relationship_richness = int(bundle["relationship_richness"])
            regularity = float(bundle["access_regularity"])

            if new_tier != "semantic":
                if (
                    maturity_score >= cfg.maturation_semantic_threshold
                    and episode_count >= cfg.maturation_min_cycles
                ):
                    new_tier = "semantic"
                elif maturity_score >= cfg.maturation_transitional_threshold:
                    new_tier = "transitional"
                else:
                    new_tier = old_tier

            if new_tier == old_tier:
                if (
                    not dry_run
                    and maturity_bundle_changed(attrs.get("maturity_features_v1"), bundle)
                ):
                    cached_attrs = dict(attrs)
                    cached_attrs["maturity_features_v1"] = bundle
                    cached_attrs["mat_policy_version"] = bundle["policy_version"]
                    cached_attrs["mat_score"] = round(maturity_score, 4)
                    await graph_store.update_entity(
                        entity.id,
                        {"attributes": json.dumps(cached_attrs)},
                        group_id,
                    )
                continue

            if context is not None:
                context.matured_entity_ids.add(entity.id)

            if not dry_run:
                attrs = dict(attrs)
                attrs["mat_tier"] = new_tier
                attrs["mat_score"] = round(maturity_score, 4)
                attrs["maturity_features_v1"] = bundle
                attrs["mat_policy_version"] = bundle["policy_version"]
                await graph_store.update_entity(
                    entity.id,
                    {"attributes": json.dumps(attrs)},
                    group_id,
                )
                if context is not None:
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
                    relationship_richness=relationship_richness,
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
