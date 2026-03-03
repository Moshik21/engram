"""Full retrieval pipeline: FTS5 → activation → spreading → scoring."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from engram.activation.spreading import (
    identify_actr_seeds,
    identify_seeds,
    spread_activation,
)
from engram.config import ActivationConfig
from engram.retrieval.router import QueryType, apply_route, classify_query
from engram.retrieval.scorer import (
    ScoredResult,
    score_candidates,
    score_candidates_thompson,
)

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer

logger = logging.getLogger(__name__)


def _scale_limit(base: int, total_entities: int, lo: int, hi: int) -> int:
    """Scale a limit by sqrt(total_entities / 1000), clamped to [lo, hi]."""
    scale = math.sqrt(max(total_entities, 1000) / 1000.0)
    return max(lo, min(int(base * scale), hi))


async def retrieve(
    query: str,
    group_id: str,
    graph_store,
    activation_store,
    search_index,
    cfg: ActivationConfig,
    limit: int = 10,
    enable_routing: bool = True,
    reranker=None,
    working_memory: WorkingMemoryBuffer | None = None,
    community_store=None,
    predicate_cache=None,
) -> list[ScoredResult]:
    """Full retrieval pipeline:

    1. FTS5 → top-K candidates (semantic similarity scores)
    1.5. Classify query and route weights (if enabled)
    2. Batch get activation states
    3. Identify seeds (sem >= seed_threshold)
    4. Spread activation BFS
    5. Score all candidates with composite formula
    5.5. Cross-encoder re-ranking (if enabled)
    5.6. MMR diversity (if enabled)
    6. Return top-N sorted by score
    """
    now = time.time()

    # Save original weight for episode scoring (before routing modifies cfg)
    original_weight_semantic = cfg.weight_semantic

    # Fetch entity count for dynamic pool sizing
    total_entities = 0
    try:
        stats = await graph_store.get_stats(group_id)
        if isinstance(stats, dict):
            total_entities = int(stats.get("entity_count", 0))
    except Exception:
        pass

    # Step 1: Generate candidates (multi-pool or single-pool)
    if cfg.multi_pool_enabled:
        from engram.retrieval.candidate_pool import generate_candidates

        # Pre-classify query for pool composition multipliers
        pre_query_type = await classify_query(query)

        candidates = await generate_candidates(
            query=query,
            group_id=group_id,
            search_index=search_index,
            activation_store=activation_store,
            graph_store=graph_store,
            cfg=cfg,
            now=now,
            working_memory=working_memory,
            total_entities=total_entities,
            query_type=pre_query_type,
        )
        query_type = await classify_query(query, search_results=candidates)
        if enable_routing or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)
        temporal_mode = query_type == QueryType.TEMPORAL
    else:
        # Original single-pool path — scale retrieval_top_k
        top_k = _scale_limit(cfg.retrieval_top_k, total_entities, 5, 500)
        search_results = await search_index.search(
            query=query, group_id=group_id, limit=top_k
        )
        candidates = search_results or []

        # Step 1.5: Classify query and override weights
        query_type = await classify_query(query, search_results=candidates)
        if enable_routing or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)

        # Step 1.6: Temporal bypass — merge activation-based candidates
        temporal_mode = False
        if query_type == QueryType.TEMPORAL:
            temporal_mode = True
            act_limit = _scale_limit(cfg.retrieval_top_k, total_entities, 5, 500)
            top_activated = await activation_store.get_top_activated(
                group_id=group_id, limit=act_limit, now=now,
            )
            existing_ids = {eid for eid, _ in candidates}
            activation_candidates = [
                (eid, 0.0) for eid, _state in top_activated
                if eid not in existing_ids
            ]
            candidates = candidates + activation_candidates

        # Step 1.7: Inject working memory candidates
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            existing_ids = {eid for eid, _ in candidates}
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in existing_ids:
                    candidates.append((item_id, 0.1 * recency_score))
                    existing_ids.add(item_id)

    # Step 1.1: Episode search (runs in both modes)
    episode_candidates: list[ScoredResult] = []
    if cfg.episode_retrieval_enabled and hasattr(search_index, "search_episodes"):
        try:
            ep_results = await search_index.search_episodes(
                query=query,
                group_id=group_id,
                limit=cfg.episode_retrieval_max * 3,
            )
            for ep_id, sem_sim in ep_results:
                ep_score = original_weight_semantic * sem_sim * cfg.episode_retrieval_weight
                episode_candidates.append(
                    ScoredResult(
                        node_id=ep_id,
                        score=ep_score,
                        semantic_similarity=sem_sim,
                        activation=0.0,
                        spreading=0.0,
                        edge_proximity=0.0,
                        exploration_bonus=0.0,
                        result_type="episode",
                    )
                )
        except Exception as e:
            logger.warning("Episode search failed (non-fatal): %s", e)

    if not candidates:
        return []

    # Step 2: Batch get activation states
    entity_ids = [eid for eid, _ in candidates]
    activation_states = await activation_store.batch_get(entity_ids)

    # Step 3: Identify seeds (strategy-dependent)
    if cfg.spreading_strategy == "actr":
        # ACT-R: seeds come from working memory only (not search results)
        if working_memory is not None and cfg.working_memory_enabled:
            seeds = identify_actr_seeds(working_memory, now, cfg)
        else:
            seeds = []
        seed_node_ids = {nid for nid, _ in seeds}
        context_gate = None  # context gate is a BFS/PPR concept
    else:
        seeds = identify_seeds(
            candidates, activation_states, now, cfg,
            temporal_mode=temporal_mode,
        )
        seed_node_ids = {nid for nid, _ in seeds}

        # Step 3.5: Add working memory entities as additional seeds
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in seed_node_ids:
                    energy = cfg.working_memory_seed_energy * recency_score
                    if energy > 0.0:
                        seeds.append((item_id, energy))
                        seed_node_ids.add(item_id)

        # Step 3.7: Build context gate (if enabled)
        context_gate = None
        if cfg.context_gating_enabled and predicate_cache is not None:
            query_emb = getattr(search_index, '_last_query_vec', None)
            if query_emb:
                from engram.activation.context_gate import build_context_gate

                context_gate = build_context_gate(query_emb, predicate_cache, cfg)

    # Step 4: Spread activation
    bonuses, hop_distances = await spread_activation(
        seeds, graph_store, cfg, group_id=group_id,
        community_store=community_store,
        context_gate=context_gate,
    )

    # Step 4.5: Merge spreading-discovered entities with real semantic similarity
    existing_ids = {eid for eid, _ in candidates}
    new_ids = [
        nid for nid in bonuses
        if nid not in existing_ids and bonuses[nid] > 0.0
    ]
    if new_ids:
        new_states = await activation_store.batch_get(new_ids)
        activation_states.update(new_states)
        discovered_sims = await search_index.compute_similarity(
            query=query, entity_ids=new_ids, group_id=group_id,
        )
        candidates = candidates + [
            (nid, discovered_sims.get(nid, 0.0)) for nid in new_ids
        ]

    # Step 5: Score all candidates
    if cfg.ts_enabled:
        scored = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses=bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
        )
    else:
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses=bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
        )

    # Step 5.5: Cross-encoder re-ranking (if enabled)
    if cfg.reranker_enabled and reranker is not None:
        try:
            # Fetch entity summaries for reranking
            docs: list[tuple[str, str]] = []
            for sr in scored[:cfg.reranker_top_n * 2]:
                entity = await graph_store.get_entity(sr.node_id, group_id)
                text = ""
                if entity:
                    text = entity.name
                    if entity.summary:
                        text = f"{entity.name}: {entity.summary}"
                docs.append((sr.node_id, text))

            if docs:
                reranked = await reranker.rerank(
                    query, docs, top_n=cfg.reranker_top_n,
                )
                # Build reranked order
                rerank_order = {eid: i for i, (eid, _) in enumerate(reranked)}
                scored.sort(
                    key=lambda sr: rerank_order.get(sr.node_id, len(scored)),
                )
        except Exception as e:
            logger.warning("Reranking failed (non-fatal): %s", e)

    # Step 5.6: MMR diversity (if enabled)
    if cfg.mmr_enabled:
        try:
            from engram.retrieval.mmr import apply_mmr

            # Fetch entity embeddings for MMR
            entity_embeddings: dict[str, list[float]] = {}
            mmr_ids = [sr.node_id for sr in scored[:cfg.retrieval_top_n * 2]]
            if hasattr(search_index, '_vectors') and hasattr(search_index, '_embeddings_enabled'):
                if search_index._embeddings_enabled:
                    from engram.storage.sqlite.vectors import unpack_vector
                    for eid in mmr_ids:
                        cursor = await search_index._vectors.db.execute(
                            "SELECT embedding, dimensions FROM embeddings "
                            "WHERE id = ? AND group_id = ?",
                            (eid, group_id),
                        )
                        row = await cursor.fetchone()
                        if row:
                            entity_embeddings[eid] = unpack_vector(
                                row["embedding"], row["dimensions"],
                            )

            scored = apply_mmr(
                scored, entity_embeddings,
                lambda_param=cfg.mmr_lambda,
                top_n=min(limit, cfg.retrieval_top_n),
            )
        except Exception as e:
            logger.warning("MMR diversity failed (non-fatal): %s", e)

    # Step 6: Return top-N, mixing entities and episodes
    top_n = min(limit, cfg.retrieval_top_n)
    if episode_candidates and cfg.episode_retrieval_enabled:
        # Take top entities up to (limit - episode_retrieval_max)
        entity_limit = max(1, top_n - cfg.episode_retrieval_max)
        entity_results = scored[:entity_limit]
        # Fill remaining slots with top episodes up to episode_retrieval_max
        episode_candidates.sort(key=lambda r: r.score, reverse=True)
        ep_results_final = episode_candidates[:cfg.episode_retrieval_max]
        # Combine and re-sort by score
        results = entity_results + ep_results_final
        results.sort(key=lambda r: r.score, reverse=True)
    else:
        results = scored[:top_n]

    # Step 7: Record Thompson Sampling feedback (if enabled)
    if cfg.ts_enabled:
        from engram.activation.feedback import (
            record_negative_feedback,
            record_positive_feedback,
        )

        returned_ids = {r.node_id for r in results}
        all_candidate_ids = {eid for eid, _ in candidates}
        for eid in returned_ids:
            await record_positive_feedback(eid, activation_store, cfg)
        for eid in all_candidate_ids - returned_ids:
            await record_negative_feedback(eid, activation_store, cfg)

    return results
