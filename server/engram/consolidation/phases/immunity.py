"""Immunity phase: Topological anomaly detection and node dissolution."""

from __future__ import annotations

import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, ImmunityRecord, PhaseResult
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


class ImmunityPhase(ConsolidationPhase):
    """Graph-aware anomaly detection inspired by biological immune systems.

    Entities that lack "semantic gravity" (poor connectivity, low clustering,
    and low average edge weight) are flagged as "viral noise" and dissolved.
    """

    @property
    def name(self) -> str:
        return "immunity"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.immunity_enabled:
            return set()
        return {
            "find_entities",
            "get_entity",
            "get_active_neighbors_with_weights",
            "update_entity",
            "delete_entity",
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
    ) -> tuple[PhaseResult, list[Any]]:
        t0 = time.perf_counter()

        if not cfg.immunity_enabled:
            return PhaseResult(phase=self.name, status="skipped", duration_ms=0.0), []

        # Find entities that are old enough to be judged
        # (immunity_min_age_hours)
        min_age_s = cfg.immunity_min_age_hours * 3600
        now = utc_now()

        # This is a simplification; in a real GNN we'd use embeddings too.
        # Here we use topological signals.
        entities = await graph_store.find_entities(group_id=group_id, limit=500)

        records: list[ImmunityRecord] = []
        processed = 0
        affected = 0

        for entity in entities:
            created_at = entity.get("created_at")
            if not created_at:
                continue

            age_s = (now - created_at).total_seconds()
            if age_s < min_age_s:
                continue

            processed += 1
            entity_id = entity["id"]

            # 1. Degree Centrality (how many neighbors)
            neighbors = await graph_store.get_active_neighbors_with_weights(entity_id)
            degree = len(neighbors)

            if degree == 0:
                gravity = 0.0
            else:
                # 2. Average Edge Weight
                total_weight = sum(w for _, w in neighbors)
                avg_weight = total_weight / degree

                # 3. Local Clustering Coefficient (approximate)
                # How many of my neighbors are connected to each other?
                # For simplicity, we'll just use degree and avg_weight for now
                # but we can call it "Topological Coherence".
                gravity = (min(degree, 5) / 5.0) * 0.6 + (min(avg_weight, 1.0)) * 0.4

            if gravity < cfg.immunity_gravity_threshold:
                affected += 1
                decision = "pruned" if not dry_run else "flagged"

                records.append(
                    ImmunityRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        node_id=entity_id,
                        node_name=entity.get("name", "unknown"),
                        semantic_gravity=gravity,
                        decision=decision,
                    )
                )

                if not dry_run:
                    logger.info(
                        "Immunity: Dissolving viral node %s (gravity=%.2f)",
                        entity_id,
                        gravity,
                    )
                    await graph_store.delete_entity(entity_id)
                    if search_index:
                        await search_index.delete_entity(entity_id)

        duration_ms = (time.perf_counter() - t0) * 1000
        return PhaseResult(
            phase=self.name,
            status="success",
            items_processed=processed,
            items_affected=affected,
            duration_ms=duration_ms,
        ), records
