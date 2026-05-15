"""Calibrate phase: aggregate preference feedback into domain scores."""

from __future__ import annotations

import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CalibrationRecord, CycleContext, PhaseResult

logger = logging.getLogger(__name__)


def _elapsed_ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class CalibratePhase(ConsolidationPhase):
    """Warm tier: aggregate PREFERS/AVOIDS edges into per-domain preference scores."""

    @property
    def name(self) -> str:
        return "calibrate"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        return {"find_entities", "get_relationships", "update_entity"}

    async def execute(
        self,
        group_id: str,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[CalibrationRecord]]:
        t0 = time.perf_counter()

        if not cfg.preference_calibrate_enabled:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=_elapsed_ms(t0),
            ), []

        # Find UserPreference entity
        prefs = await graph_store.find_entities(
            name="UserPreference",
            entity_type="PreferenceProfile",
            group_id=group_id,
            limit=1,
        )
        if not prefs:
            return PhaseResult(
                phase=self.name,
                items_processed=0,
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        pref_entity = prefs[0]
        records: list[CalibrationRecord] = []

        # Get all PREFERS and AVOIDS edges
        prefers_edges = await graph_store.get_relationships(
            entity_id=pref_entity.id,
            direction="outgoing",
            predicate="PREFERS",
            group_id=group_id,
        )
        avoids_edges = await graph_store.get_relationships(
            entity_id=pref_entity.id,
            direction="outgoing",
            predicate="AVOIDS",
            group_id=group_id,
        )

        # Build target entity -> domain mapping
        all_target_ids = list(
            {e.target_id for e in prefers_edges} | {e.target_id for e in avoids_edges}
        )
        if not all_target_ids:
            return PhaseResult(
                phase=self.name,
                items_processed=0,
                items_affected=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        # Batch get entities for domain lookup
        entity_map = {}
        if hasattr(graph_store, "batch_get_entities"):
            entity_map = await graph_store.batch_get_entities(all_target_ids, group_id)
        else:
            for eid in all_target_ids:
                ent = await graph_store.get_entity(eid, group_id)
                if ent:
                    entity_map[eid] = ent

        # Map entity types to domains
        def _get_domain(entity_type: str) -> str:
            for domain, types in cfg.domain_groups.items():
                if entity_type in types:
                    return domain
            return "general"

        # Aggregate per-domain scores
        domain_scores: dict[str, float] = {}
        domain_counts: dict[str, int] = {}

        for edge in prefers_edges:
            ent = entity_map.get(edge.target_id)
            if not ent:
                continue
            domain = _get_domain(ent.entity_type)
            domain_scores[domain] = domain_scores.get(domain, 0.0) + edge.weight
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        for edge in avoids_edges:
            ent = entity_map.get(edge.target_id)
            if not ent:
                continue
            domain = _get_domain(ent.entity_type)
            domain_scores[domain] = domain_scores.get(domain, 0.0) - edge.weight
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Normalize to [-1, 1]
        for domain in domain_scores:
            count = domain_counts.get(domain, 1)
            if count > 0:
                domain_scores[domain] = max(-1.0, min(1.0, domain_scores[domain] / count))

        # Apply decay to previous scores
        existing_attrs = pref_entity.attributes or {}
        prev_scores = existing_attrs.get("domain_preference_scores", {})
        decay = cfg.preference_decay_rate
        for domain in prev_scores:
            if domain not in domain_scores:
                # Decay domains not seen this cycle
                prev_scores[domain] = prev_scores[domain] * (1.0 - decay)
                if abs(prev_scores[domain]) < 0.01:
                    prev_scores[domain] = 0.0

        # Merge: blend new with decayed previous
        for domain, score in domain_scores.items():
            prev = prev_scores.get(domain, 0.0)
            domain_scores[domain] = 0.7 * score + 0.3 * prev * (1.0 - decay)

        # Store on PreferenceProfile entity
        if not dry_run:
            updated_attrs = dict(existing_attrs)
            updated_attrs["domain_preference_scores"] = domain_scores
            updated_attrs["domain_entity_counts"] = domain_counts
            await graph_store.update_entity(
                pref_entity.id,
                {"attributes": updated_attrs},
                group_id,
            )

        # Create audit records
        for domain, score in domain_scores.items():
            records.append(
                CalibrationRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    domain=domain,
                    preference_score=round(score, 4),
                    entity_count=domain_counts.get(domain, 0),
                    decay_applied=decay,
                )
            )

        return PhaseResult(
            phase=self.name,
            items_processed=len(all_target_ids),
            items_affected=len(domain_scores),
            duration_ms=_elapsed_ms(t0),
        ), records
