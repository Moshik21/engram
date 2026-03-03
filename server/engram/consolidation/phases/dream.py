"""Dream spreading phase: offline Hebbian reinforcement of associative pathways."""

from __future__ import annotations

import logging
import time
from typing import Any

from engram.activation.engine import compute_activation
from engram.activation.spreading import spread_activation
from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, DreamRecord, PhaseResult

logger = logging.getLogger(__name__)


class DreamSpreadingPhase(ConsolidationPhase):
    """Strengthen associative pathways via offline spreading activation.

    Like biological memory replay during sleep, this phase runs spreading
    activation without a query — using medium-activation entities as seeds.
    Edges traversed during spreading get their weight incremented (Hebbian:
    "neurons that fire together wire together").

    Runs last because it only modifies edge weights which don't affect
    entity embeddings, so no re-reindex is needed.
    """

    @property
    def name(self) -> str:
        return "dream"

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

        if not cfg.consolidation_dream_enabled:
            return PhaseResult(
                phase=self.name, status="skipped", duration_ms=0.0,
            ), []

        now = time.time()

        # 1. Select seeds using bell-curve preference for medium activation
        seeds = await self._select_dream_seeds(
            activation_store, group_id, now, cfg,
        )

        if not seeds:
            elapsed = (time.perf_counter() - t0) * 1000
            return PhaseResult(
                phase=self.name, status="success",
                items_processed=0, items_affected=0,
                duration_ms=round(elapsed, 1),
            ), []

        # Track seed IDs in cycle context
        if context is not None:
            context.dream_seed_ids.update(sid for sid, _ in seeds)

        # 2. Run spreading for each seed and accumulate edge boosts
        edge_boosts: dict[tuple[str, str], float] = {}
        for seed_id, energy in seeds:
            bonuses, _ = await spread_activation(
                [(seed_id, energy)], graph_store, cfg, group_id=group_id,
            )
            seed_boosts = await self._accumulate_edge_boosts(
                seed_id, bonuses, graph_store, group_id, cfg,
            )
            for edge_key, boost in seed_boosts.items():
                edge_boosts[edge_key] = edge_boosts.get(edge_key, 0.0) + boost

        # 3. Apply boosts to edge weights
        records: list[DreamRecord] = []
        edges_boosted = 0
        for (src, tgt), total_boost in edge_boosts.items():
            if total_boost < cfg.consolidation_dream_min_boost:
                continue
            capped_boost = min(total_boost, cfg.consolidation_dream_max_boost_per_edge)

            if not dry_run:
                await graph_store.update_relationship_weight(
                    src, tgt, capped_boost,
                    max_weight=cfg.consolidation_dream_max_edge_weight,
                    group_id=group_id,
                )

            edges_boosted += 1
            records.append(DreamRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                source_entity_id=src,
                target_entity_id=tgt,
                weight_delta=capped_boost,
            ))

        elapsed = (time.perf_counter() - t0) * 1000
        return PhaseResult(
            phase=self.name,
            status="success",
            items_processed=len(seeds),
            items_affected=edges_boosted,
            duration_ms=round(elapsed, 1),
        ), records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _select_dream_seeds(
        self,
        activation_store,
        group_id: str,
        now: float,
        cfg: ActivationConfig,
    ) -> list[tuple[str, float]]:
        """Select seed entities using bell-curve preference for medium activation.

        Entities within the [floor, ceiling] activation band are candidates.
        They are ranked by proximity to the midpoint (closest = highest priority).
        """
        # get_top_activated returns (entity_id, ActivationState) pairs
        all_entities = await activation_store.get_top_activated(
            group_id=group_id, limit=10000, now=now,
        )

        floor = cfg.consolidation_dream_activation_floor
        ceiling = cfg.consolidation_dream_activation_ceiling
        midpoint = cfg.consolidation_dream_activation_midpoint

        candidates: list[tuple[str, float, float]] = []  # (id, activation, distance)
        for entity_id, state in all_entities:
            act = compute_activation(state.access_history, now, cfg)
            if floor <= act <= ceiling:
                distance = abs(act - midpoint)
                candidates.append((entity_id, act, distance))

        # Sort by distance to midpoint (closest first)
        candidates.sort(key=lambda x: x[2])

        # Take top max_seeds, return (entity_id, energy=activation)
        max_seeds = cfg.consolidation_dream_max_seeds
        return [(eid, act) for eid, act, _ in candidates[:max_seeds]]

    async def _accumulate_edge_boosts(
        self,
        seed_id: str,
        bonuses: dict[str, float],
        graph_store,
        group_id: str,
        cfg: ActivationConfig,
    ) -> dict[tuple[str, str], float]:
        """Identify edges traversed during spreading and compute boost amounts.

        An edge (A, B) is considered traversed if both A and B received
        spreading bonuses (or one is the seed). Boost magnitude is
        proportional to the harmonic mean of both endpoints' bonuses.
        """
        reached = set(bonuses.keys()) | {seed_id}
        max_bonus = max(bonuses.values()) if bonuses else 1.0
        if max_bonus <= 0:
            max_bonus = 1.0

        edge_boosts: dict[tuple[str, str], float] = {}

        for node_id in reached:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                node_id, group_id=group_id,
            )
            for neighbor_id, _weight, _predicate in neighbors:
                if neighbor_id not in reached:
                    continue

                # Canonical edge key for deduplication
                edge_key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if edge_key in edge_boosts:
                    continue

                # Compute boost based on harmonic mean of endpoint bonuses
                bonus_a = bonuses.get(edge_key[0], 0.0)
                bonus_b = bonuses.get(edge_key[1], 0.0)

                if bonus_a + bonus_b <= 0:
                    continue

                harmonic = (2.0 * bonus_a * bonus_b) / (bonus_a + bonus_b)
                boost = cfg.consolidation_dream_weight_increment * harmonic / max_bonus
                edge_boosts[edge_key] = boost

        return edge_boosts
