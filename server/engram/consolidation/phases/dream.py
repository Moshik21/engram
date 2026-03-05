"""Dream spreading phase: offline Hebbian reinforcement of associative pathways."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from engram.activation.bfs import _resolve_domain
from engram.activation.engine import compute_activation
from engram.activation.spreading import spread_activation
from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import (
    CycleContext,
    DreamAssociationRecord,
    DreamRecord,
    PhaseResult,
)
from engram.models.relationship import Relationship

logger = logging.getLogger(__name__)


class DreamSpreadingPhase(ConsolidationPhase):
    """Strengthen associative pathways via offline spreading activation.

    Like biological memory replay during sleep, this phase runs spreading
    activation without a query — using medium-activation entities as seeds.
    Edges traversed during spreading get their weight incremented (Hebbian:
    "neurons that fire together wire together").

    When dream associations are enabled, also discovers semantically similar
    but structurally distant cross-domain entity pairs and creates weak
    temporary edges between them, modeling how REM sleep creates novel
    associations.

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
                phase=self.name,
                status="skipped",
                duration_ms=0.0,
            ), []

        now = time.time()

        # 1. Select seeds using bell-curve preference for medium activation
        seeds = await self._select_dream_seeds(
            activation_store,
            group_id,
            now,
            cfg,
        )

        # Track seed IDs in cycle context
        if seeds and context is not None:
            context.dream_seed_ids.update(sid for sid, _ in seeds)

        # 2. Run spreading for each seed and accumulate edge boosts
        edge_boosts: dict[tuple[str, str], float] = {}
        for seed_id, energy in (seeds or []):
            bonuses, _ = await spread_activation(
                [(seed_id, energy)],
                graph_store,
                cfg,
                group_id=group_id,
            )
            seed_boosts = await self._accumulate_edge_boosts(
                seed_id,
                bonuses,
                graph_store,
                group_id,
                cfg,
            )
            for edge_key, boost in seed_boosts.items():
                edge_boosts[edge_key] = edge_boosts.get(edge_key, 0.0) + boost

        # 3. Apply boosts to edge weights
        records: list[Any] = []
        edges_boosted = 0
        for (src, tgt), total_boost in edge_boosts.items():
            if total_boost < cfg.consolidation_dream_min_boost:
                continue
            capped_boost = min(total_boost, cfg.consolidation_dream_max_boost_per_edge)

            if not dry_run:
                await graph_store.update_relationship_weight(
                    src,
                    tgt,
                    capped_boost,
                    max_weight=cfg.consolidation_dream_max_edge_weight,
                    group_id=group_id,
                )

            edges_boosted += 1
            records.append(
                DreamRecord(
                    cycle_id=cycle_id,
                    group_id=group_id,
                    source_entity_id=src,
                    target_entity_id=tgt,
                    weight_delta=capped_boost,
                )
            )

        # 3b. Apply LTD (Long-Term Depression) to unboosted edges of seed entities
        edges_decayed = 0
        if cfg.consolidation_dream_ltd_enabled and seeds and not dry_run:
            edges_decayed = await self._apply_ltd_decay(
                seeds=seeds,
                boosted_edges=edge_boosts,
                graph_store=graph_store,
                group_id=group_id,
                cfg=cfg,
            )

        # 4. Dream associations: discover cross-domain creative connections
        assoc_count = 0
        if cfg.consolidation_dream_associations_enabled:
            assoc_records = await self._find_dream_associations(
                group_id=group_id,
                graph_store=graph_store,
                search_index=search_index,
                cfg=cfg,
                cycle_id=cycle_id,
                dry_run=dry_run,
                context=context,
            )
            assoc_count = len(assoc_records)
            records.extend(assoc_records)

        elapsed = (time.perf_counter() - t0) * 1000
        return PhaseResult(
            phase=self.name,
            status="success",
            items_processed=len(seeds or []),
            items_affected=edges_boosted + assoc_count + edges_decayed,
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
            group_id=group_id,
            limit=10000,
            now=now,
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

        DREAM_ASSOCIATED edges are excluded from Hebbian boosting to prevent
        dream drift over many consolidation cycles.
        """
        reached = set(bonuses.keys()) | {seed_id}
        max_bonus = max(bonuses.values()) if bonuses else 1.0
        if max_bonus <= 0:
            max_bonus = 1.0

        edge_boosts: dict[tuple[str, str], float] = {}

        for node_id in reached:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                node_id,
                group_id=group_id,
            )
            for neighbor_id, _weight, _predicate, *_rest in neighbors:
                # Exclude DREAM_ASSOCIATED edges from Hebbian boosting
                if _predicate == "DREAM_ASSOCIATED":
                    continue

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

    async def _apply_ltd_decay(
        self,
        seeds: list[tuple[str, float]],
        boosted_edges: dict[tuple[str, str], float],
        graph_store,
        group_id: str,
        cfg: ActivationConfig,
    ) -> int:
        """Apply Long-Term Depression to edges NOT activated during spreading.

        Biological analog: synapses that don't fire during sleep replay
        weaken slightly, maintaining discriminability and preventing
        monotonic weight inflation from Hebbian boosting alone.

        Only decays edges connected to seed entities (scope of this dream cycle).
        DREAM_ASSOCIATED edges are excluded (managed by TTL, not LTD).
        """
        decay = cfg.consolidation_dream_ltd_decay
        min_weight = cfg.consolidation_dream_ltd_min_weight
        decayed = 0

        # Collect all seed entity IDs
        seed_ids = {sid for sid, _ in seeds}

        for seed_id in seed_ids:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                seed_id, group_id=group_id,
            )
            for neighbor_id, weight, predicate, *_rest in neighbors:
                # Skip DREAM_ASSOCIATED edges (TTL-managed)
                if predicate == "DREAM_ASSOCIATED":
                    continue

                # Skip edges below the floor
                if weight <= min_weight:
                    continue

                # Check if this edge was boosted
                edge_key = (min(seed_id, neighbor_id), max(seed_id, neighbor_id))
                if edge_key in boosted_edges:
                    continue

                # Apply decay (negative delta)
                capped_decay = min(decay, weight - min_weight)
                if capped_decay <= 0:
                    continue

                await graph_store.update_relationship_weight(
                    seed_id,
                    neighbor_id,
                    -capped_decay,
                    max_weight=cfg.consolidation_dream_max_edge_weight,
                    group_id=group_id,
                )
                decayed += 1

        if decayed:
            logger.info("Dream LTD: decayed %d unboosted edges by %.4f", decayed, decay)
        return decayed

    # ------------------------------------------------------------------
    # Dream Associations
    # ------------------------------------------------------------------

    async def _find_dream_associations(
        self,
        group_id: str,
        graph_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext | None,
    ) -> list[DreamAssociationRecord]:
        """Discover semantically similar but structurally distant cross-domain pairs.

        Algorithm:
        1. Collect eligible entities (not pruned, summary meets min length)
        2. Partition by domain using entity type → domain mapping
        3. Batch retrieve embeddings
        4. Compute cross-domain cosine similarities via matrix multiplication
        5. For each candidate (sorted by similarity desc):
           - Check per-domain-pair quota
           - Check structural proximity
           - Compute surprise score: similarity × (1 - structural_proximity)
           - If surprise >= threshold: create relationship + audit record
        """
        t0 = time.perf_counter()
        max_duration_ms = cfg.consolidation_dream_assoc_max_duration_ms

        # 1. Collect eligible entities
        pruned_ids = context.pruned_entity_ids if context else set()
        entities = await graph_store.find_entities(group_id=group_id, limit=5000)
        eligible = [
            e for e in entities
            if e.id not in pruned_ids
            and e.summary
            and len(e.summary) >= cfg.consolidation_dream_assoc_min_summary_len
        ]

        if len(eligible) < 2:
            return []

        # 2. Partition by domain
        domain_buckets = self._partition_by_domain(eligible, cfg)
        domain_names = list(domain_buckets.keys())
        if len(domain_names) < 2:
            return []

        # 3. Batch retrieve embeddings (text + optional graph structural)
        all_ids = [e.id for e in eligible]
        embeddings = await search_index.get_entity_embeddings(all_ids, group_id=group_id)
        if not embeddings:
            return []

        # 3b. Blend graph embeddings if available (richer structural similarity)
        graph_embeddings: dict[str, list[float]] | None = None
        if hasattr(search_index, "get_graph_embeddings"):
            # Try each method in priority order
            for method in ("node2vec", "transe", "gnn"):
                try:
                    g_embs = await search_index.get_graph_embeddings(
                        all_ids, method=method, group_id=group_id,
                    )
                    if g_embs and len(g_embs) > cfg.consolidation_dream_assoc_min_graph_embeddings:
                        graph_embeddings = g_embs
                        break
                except Exception:
                    continue

        # 4. Compute cross-domain similarities
        candidates = self._compute_cross_domain_similarities(
            domain_buckets, embeddings, cfg,
            graph_embeddings=graph_embeddings,
        )

        # 5. Filter and create associations
        records: list[DreamAssociationRecord] = []
        domain_pair_counts: dict[tuple[str, str], int] = {}
        max_per_cycle = cfg.consolidation_dream_assoc_max_per_cycle
        max_per_pair = cfg.consolidation_dream_assoc_max_per_domain_pair

        for src_id, tgt_id, src_domain, tgt_domain, similarity in candidates:
            # Time budget check
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > max_duration_ms:
                logger.info("Dream associations: time budget exhausted (%.0f ms)", elapsed_ms)
                break

            if len(records) >= max_per_cycle:
                break

            # Per-domain-pair quota
            pair_key = (min(src_domain, tgt_domain), max(src_domain, tgt_domain))
            if domain_pair_counts.get(pair_key, 0) >= max_per_pair:
                continue

            # Check structural proximity
            structural_proximity = 0.0
            try:
                connected = await graph_store.path_exists_within_hops(
                    src_id, tgt_id,
                    max_hops=cfg.consolidation_dream_assoc_structural_max_hops,
                    group_id=group_id,
                )
                if connected:
                    structural_proximity = 1.0
            except Exception:
                pass  # Assume disconnected on error

            # Compute surprise score
            surprise = _compute_surprise_score(similarity, structural_proximity)
            if surprise < cfg.consolidation_dream_assoc_min_surprise:
                continue

            # Look up entity names
            src_entity = next((e for e in eligible if e.id == src_id), None)
            tgt_entity = next((e for e in eligible if e.id == tgt_id), None)
            if not src_entity or not tgt_entity:
                continue

            # Create relationship
            rel_id = None
            if not dry_run:
                valid_to = datetime.utcnow() + timedelta(
                    days=cfg.consolidation_dream_assoc_ttl_days
                )
                rel = Relationship(
                    id=f"rel_{uuid.uuid4().hex[:12]}",
                    source_id=src_id,
                    target_id=tgt_id,
                    predicate="DREAM_ASSOCIATED",
                    weight=cfg.consolidation_dream_assoc_weight,
                    valid_to=valid_to,
                    group_id=group_id,
                    confidence=surprise,
                    source_episode=f"dream:{cycle_id}",
                )
                try:
                    await graph_store.create_relationship(rel)
                    rel_id = rel.id
                except Exception as exc:
                    logger.warning("Failed to create dream association: %s", exc)
                    continue

            record = DreamAssociationRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                source_entity_name=src_entity.name,
                target_entity_name=tgt_entity.name,
                source_domain=src_domain,
                target_domain=tgt_domain,
                surprise_score=surprise,
                embedding_similarity=similarity,
                structural_proximity=structural_proximity,
                relationship_id=rel_id,
            )
            records.append(record)
            domain_pair_counts[pair_key] = domain_pair_counts.get(pair_key, 0) + 1

            # Update cycle context
            if context is not None:
                context.dream_association_ids.add(record.id)

        logger.info(
            "Dream associations: %d candidates → %d created (dry_run=%s)",
            len(candidates),
            len(records),
            dry_run,
        )
        return records

    @staticmethod
    def _partition_by_domain(
        entities: list,
        cfg: ActivationConfig,
    ) -> dict[str, list]:
        """Partition entities into domain buckets based on entity_type → domain mapping.

        Returns {domain: [entity, ...]}. Entities with no domain mapping go to
        'uncategorized'. Each bucket is limited to top_n_per_domain entries.
        """
        buckets: dict[str, list] = {}
        for entity in entities:
            domain = _resolve_domain(entity.entity_type, cfg.domain_groups)
            if domain is None:
                domain = "uncategorized"
            buckets.setdefault(domain, []).append(entity)

        # Trim each bucket to top_n_per_domain
        max_per_domain = cfg.consolidation_dream_assoc_top_n_per_domain
        return {d: es[:max_per_domain] for d, es in buckets.items()}

    @staticmethod
    def _compute_cross_domain_similarities(
        domain_buckets: dict[str, list],
        embeddings: dict[str, list[float]],
        cfg: ActivationConfig,
        graph_embeddings: dict[str, list[float]] | None = None,
    ) -> list[tuple[str, str, str, str, float]]:
        """Compute cosine similarities between entities in different domains.

        Uses numpy matrix multiplication for efficiency. When graph_embeddings
        are available, blends text and graph similarities (70/30 text/graph)
        for richer cross-domain discovery.

        Returns sorted list of (src_id, tgt_id, src_domain, tgt_domain, similarity).
        """
        domain_names = list(domain_buckets.keys())
        candidates: list[tuple[str, str, str, str, float]] = []

        # Blending weight for graph embeddings (graph captures structural
        # similarity that text misses — same position, different names)
        graph_blend = 0.3 if graph_embeddings else 0.0
        text_blend = 1.0 - graph_blend

        for i in range(len(domain_names)):
            for j in range(i + 1, len(domain_names)):
                d1, d2 = domain_names[i], domain_names[j]
                entities_a = domain_buckets[d1]
                entities_b = domain_buckets[d2]

                # Build text embedding matrices
                ids_a = [e.id for e in entities_a if e.id in embeddings]
                ids_b = [e.id for e in entities_b if e.id in embeddings]
                if not ids_a or not ids_b:
                    continue

                mat_a = np.array([embeddings[eid] for eid in ids_a], dtype=np.float32)
                mat_b = np.array([embeddings[eid] for eid in ids_b], dtype=np.float32)

                # Normalize for cosine similarity
                norms_a = np.linalg.norm(mat_a, axis=1, keepdims=True)
                norms_b = np.linalg.norm(mat_b, axis=1, keepdims=True)
                norms_a = np.where(norms_a > 0, norms_a, 1.0)
                norms_b = np.where(norms_b > 0, norms_b, 1.0)
                mat_a = mat_a / norms_a
                mat_b = mat_b / norms_b

                text_sim = mat_a @ mat_b.T

                # Blend with graph embeddings if available
                if graph_embeddings:
                    g_ids_a = [eid for eid in ids_a if eid in graph_embeddings]
                    g_ids_b = [eid for eid in ids_b if eid in graph_embeddings]

                    if g_ids_a and g_ids_b:
                        g_mat_a = np.array(
                            [graph_embeddings[eid] for eid in g_ids_a], dtype=np.float32,
                        )
                        g_mat_b = np.array(
                            [graph_embeddings[eid] for eid in g_ids_b], dtype=np.float32,
                        )
                        g_norms_a = np.linalg.norm(g_mat_a, axis=1, keepdims=True)
                        g_norms_b = np.linalg.norm(g_mat_b, axis=1, keepdims=True)
                        g_norms_a = np.where(g_norms_a > 0, g_norms_a, 1.0)
                        g_norms_b = np.where(g_norms_b > 0, g_norms_b, 1.0)
                        g_mat_a = g_mat_a / g_norms_a
                        g_mat_b = g_mat_b / g_norms_b

                        g_sim = g_mat_a @ g_mat_b.T

                        # Build combined sim for entities that have graph embeddings
                        g_a_map = {eid: gi for gi, eid in enumerate(g_ids_a)}
                        g_b_map = {eid: gi for gi, eid in enumerate(g_ids_b)}

                        for ai in range(len(ids_a)):
                            for bi in range(len(ids_b)):
                                t_sim = float(text_sim[ai, bi])
                                if ids_a[ai] in g_a_map and ids_b[bi] in g_b_map:
                                    gs = float(g_sim[g_a_map[ids_a[ai]], g_b_map[ids_b[bi]]])
                                    combined = text_blend * t_sim + graph_blend * gs
                                else:
                                    combined = t_sim
                                if combined >= 0.2:
                                    candidates.append(
                                        (ids_a[ai], ids_b[bi], d1, d2, combined),
                                    )
                        continue  # Skip the non-blended path below

                # Non-blended path (no graph embeddings for this pair)
                for ai in range(len(ids_a)):
                    for bi in range(len(ids_b)):
                        sim = float(text_sim[ai, bi])
                        if sim >= 0.2:
                            candidates.append((ids_a[ai], ids_b[bi], d1, d2, sim))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[4], reverse=True)
        return candidates


def _compute_surprise_score(similarity: float, structural_proximity: float) -> float:
    """Compute surprise score for a cross-domain entity pair.

    High similarity + structural disconnection = high surprise.
    """
    return similarity * (1.0 - structural_proximity)
