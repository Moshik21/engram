"""Retrieval method configurations and run_retrieval() wrapper for A/B benchmarks.

Defines retrieval strategies and a wrapper that executes a single retrieval
pass without recording access — so activation state is not contaminated
between methods.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from engram.activation.spreading import identify_seeds, spread_activation
from engram.config import ActivationConfig
from engram.retrieval.router import QueryType, apply_route, classify_query
from engram.retrieval.scorer import (
    ScoredResult,
    score_candidates,
    score_candidates_thompson,
)
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMethod:
    """A named retrieval configuration for benchmarking."""

    name: str
    description: str
    config: ActivationConfig
    spreading_enabled: bool
    routing_enabled: bool = False
    requires_consolidation: bool = False


# ---------------------------------------------------------------------------
# Predefined methods
# ---------------------------------------------------------------------------

METHOD_FULL_ENGRAM = RetrievalMethod(
    name="Full Engram",
    description=(
        "Complete pipeline: semantic search + ACT-R activation + "
        "spreading activation + edge proximity scoring."
    ),
    config=ActivationConfig(
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_NO_SPREADING = RetrievalMethod(
    name="No Spreading",
    description=(
        "Semantic search with ACT-R base-level activation but no "
        "spreading activation or edge proximity."
    ),
    config=ActivationConfig(
        weight_semantic=0.50,
        weight_activation=0.50,
        weight_spreading=0.00,
        weight_edge_proximity=0.00,
    ),
    spreading_enabled=False,
)

METHOD_PURE_SEARCH = RetrievalMethod(
    name="Pure Search",
    description=(
        "Baseline: only semantic/FTS similarity, no activation signals."
    ),
    config=ActivationConfig(
        weight_semantic=1.00,
        weight_activation=0.00,
        weight_spreading=0.00,
        weight_edge_proximity=0.00,
        exploration_weight=0.00,
        rediscovery_weight=0.00,
    ),
    spreading_enabled=False,
)

METHOD_SEARCH_RECENCY = RetrievalMethod(
    name="Search+Recency",
    description=(
        "Semantic search weighted with ACT-R recency decay but no "
        "spreading activation or edge proximity."
    ),
    config=ActivationConfig(
        weight_semantic=0.70,
        weight_activation=0.30,
        weight_spreading=0.00,
        weight_edge_proximity=0.00,
    ),
    spreading_enabled=False,
)

METHOD_ROUTED = RetrievalMethod(
    name="Routed",
    description=(
        "Full pipeline with query-type routing: weights adapt based on "
        "query classification (direct/temporal/associative/default)."
    ),
    config=ActivationConfig(
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_THOMPSON = RetrievalMethod(
    name="Thompson",
    description=(
        "Full pipeline with Thompson Sampling exploration: "
        "Beta-distribution posterior for adaptive per-entity exploration."
    ),
    config=ActivationConfig(
        ts_enabled=True,
        ts_weight=0.08,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_PPR = RetrievalMethod(
    name="PPR",
    description=(
        "Personalized PageRank spreading: smooth graph-based relevance "
        "distribution without fixed hop cutoff."
    ),
    config=ActivationConfig(
        spreading_strategy="ppr",
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_FAN_SPREAD = RetrievalMethod(
    name="Fan Spread",
    description=(
        "Full pipeline with aggressive fan-based S_ji associative strength "
        "(fan_s_max=4.5) replacing 1/sqrt(degree) dampening."
    ),
    config=ActivationConfig(
        fan_s_max=4.5,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_RRF = RetrievalMethod(
    name="RRF Fusion",
    description=(
        "Full pipeline with Reciprocal Rank Fusion replacing "
        "linear weighted merge in hybrid search."
    ),
    config=ActivationConfig(
        use_rrf=True,
        rrf_k=60,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_MMR = RetrievalMethod(
    name="MMR Diversity",
    description=(
        "Full pipeline with Maximal Marginal Relevance diversity "
        "re-ranking (lambda=0.7)."
    ),
    config=ActivationConfig(
        mmr_enabled=True,
        mmr_lambda=0.7,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_LINEAR = RetrievalMethod(
    name="Linear Merge",
    description=(
        "Full pipeline with linear weighted merge replacing "
        "Reciprocal Rank Fusion in hybrid search."
    ),
    config=ActivationConfig(
        use_rrf=False,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
)

METHOD_COMMUNITY = RetrievalMethod(
    name="Community",
    description=(
        "Full pipeline with community-aware spreading: bridge edges "
        "boosted 1.5x, intra-cluster edges dampened to 0.7x."
    ),
    config=ActivationConfig(
        community_spreading_enabled=True,
        community_bridge_boost=1.5,
        community_intra_dampen=0.7,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_CONTEXT_GATED = RetrievalMethod(
    name="Context-Gated",
    description=(
        "Full pipeline with context-gated spreading: edge weights "
        "modulated by query-predicate cosine similarity (floor=0.3)."
    ),
    config=ActivationConfig(
        context_gating_enabled=True,
        context_gate_floor=0.3,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_MULTI_POOL = RetrievalMethod(
    name="Multi-Pool",
    description=(
        "Multi-pool candidate generation: text search + activation + "
        "graph neighborhood + working memory merged via RRF."
    ),
    config=ActivationConfig(
        multi_pool_enabled=True,
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_FULL_STACK = RetrievalMethod(
    name="Full Stack",
    description=(
        "Everything enabled: multi-pool candidates, community-aware "
        "spreading (bridge boost 1.5x, intra dampen 0.7x), "
        "context-gated edges, fan_s_max=3.5, spread_energy_budget=75.0, "
        "episode retrieval, query routing."
    ),
    config=ActivationConfig(
        # Multi-pool candidate generation (#8)
        multi_pool_enabled=True,
        # Fan dampening (#1) — default 3.5
        fan_s_max=3.5,
        # Increased energy budget (#4) — default 50.0, increase for deeper spread
        spread_energy_budget=75.0,
        # Community-aware spreading (#9)
        community_spreading_enabled=True,
        community_bridge_boost=1.5,
        community_intra_dampen=0.7,
        # Context-gated spreading (#10)
        context_gating_enabled=True,
        context_gate_floor=0.3,
        # Scoring weights (#3)
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
)

METHOD_POST_CONSOLIDATION = RetrievalMethod(
    name="Post-Consolidation",
    description=(
        "Full Engram pipeline after entity merging, pruning, "
        "edge inference, and history compaction."
    ),
    config=ActivationConfig(
        weight_semantic=0.40,
        weight_activation=0.25,
        weight_spreading=0.20,
        weight_edge_proximity=0.15,
    ),
    spreading_enabled=True,
    routing_enabled=True,
    requires_consolidation=True,
)

ALL_METHODS: list[RetrievalMethod] = [
    METHOD_FULL_ENGRAM,
    METHOD_NO_SPREADING,
    METHOD_PURE_SEARCH,
    METHOD_SEARCH_RECENCY,
    METHOD_ROUTED,
    METHOD_PPR,
    METHOD_THOMPSON,
    METHOD_FAN_SPREAD,
    METHOD_RRF,
    METHOD_MMR,
    METHOD_LINEAR,
    METHOD_COMMUNITY,
    METHOD_MULTI_POOL,
    METHOD_CONTEXT_GATED,
    METHOD_FULL_STACK,
    METHOD_POST_CONSOLIDATION,
]


# ---------------------------------------------------------------------------
# Retrieval wrapper
# ---------------------------------------------------------------------------


async def run_retrieval(
    query: str,
    group_id: str,
    graph_store: GraphStore,
    activation_store: ActivationStore,
    search_index: SearchIndex,
    method: RetrievalMethod,
    limit: int = 10,
    reranker=None,
    now: float | None = None,
    working_memory: WorkingMemoryBuffer | None = None,
    community_store=None,
    predicate_cache=None,
) -> list[ScoredResult]:
    """Execute a single retrieval pass for the given method.

    Steps:
        1. Search for candidates via ``search_index.search()``.
        1.5. Classify query; if TEMPORAL, merge activation-based candidates.
        1.7. Inject working memory candidates (if provided).
        1.8. Search episodes (if available) and merge as scored results.
        2. Batch-get activation states for all candidate IDs.
        3. If ``method.spreading_enabled``: identify seeds and spread
           activation through the graph.
        3.5. Add working memory entities as additional seeds.
        4. Score all candidates (Thompson or deterministic).
        5. Cross-encoder re-ranking (if enabled).
        6. MMR diversity (if enabled).
        7. Return the top *limit* results sorted by score descending.

    CRITICAL: This function does **not** call ``record_access()`` so that
    activation state is not contaminated between methods during benchmarking.
    """
    cfg = method.config
    now = now if now is not None else time.time()

    # 1. Generate candidates (multi-pool or single-pool)
    if cfg.multi_pool_enabled:
        from engram.retrieval.candidate_pool import generate_candidates

        candidates = await generate_candidates(
            query=query,
            group_id=group_id,
            search_index=search_index,
            activation_store=activation_store,
            graph_store=graph_store,
            cfg=cfg,
            now=now,
            working_memory=working_memory,
        )
        query_type = await classify_query(query, search_results=candidates or [])
        if method.routing_enabled or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)
        temporal_mode = query_type == QueryType.TEMPORAL
    else:
        # Original single-pool path
        candidates = await search_index.search(
            query, group_id=group_id, limit=cfg.retrieval_top_k,
        )

        # 1.5. Always classify query for temporal bypass
        query_type = await classify_query(query, search_results=candidates or [])

        # Override weights if routing enabled, or always for TEMPORAL queries
        if method.routing_enabled or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)

        # 1.6. Temporal bypass: merge activation-based candidates for recency queries
        temporal_mode = False
        if query_type == QueryType.TEMPORAL:
            temporal_mode = True
            top_activated = await activation_store.get_top_activated(
                group_id=group_id, limit=cfg.retrieval_top_k, now=now,
            )
            existing_ids = {eid for eid, _ in candidates} if candidates else set()
            activation_candidates = [
                (eid, 0.0) for eid, _state in top_activated
                if eid not in existing_ids
            ]
            if candidates:
                candidates = candidates + activation_candidates
            else:
                candidates = activation_candidates

        # 1.7. Inject working memory candidates
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            existing_ids = {eid for eid, _ in candidates} if candidates else set()
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in existing_ids:
                    candidates.append((item_id, 0.1 * recency_score))
                    existing_ids.add(item_id)

    if not candidates:
        return []

    # 1.8. Episode search — collect episode results to merge at the end
    episode_results: list[ScoredResult] = []
    if cfg.episode_retrieval_enabled and hasattr(search_index, 'search_episodes'):
        try:
            ep_hits = await search_index.search_episodes(
                query, group_id=group_id, limit=cfg.episode_retrieval_max,
            )
            for ep_id, sem_sim in ep_hits:
                score = cfg.weight_semantic * sem_sim * cfg.episode_retrieval_weight
                episode_results.append(ScoredResult(
                    node_id=ep_id,
                    score=score,
                    semantic_similarity=sem_sim,
                    activation=0.0,
                    spreading=0.0,
                    edge_proximity=0.0,
                    exploration_bonus=0.0,
                    result_type="episode",
                ))
        except Exception as e:
            logger.warning("Episode search failed (non-fatal): %s", e)

    # 2. Batch-get activation states for all candidate IDs
    candidate_ids = [node_id for node_id, _ in candidates]
    activation_states = await activation_store.batch_get(candidate_ids)

    # 3. Spreading activation (conditional)
    if method.spreading_enabled:
        seeds = identify_seeds(
            candidates, activation_states, now, cfg,
            temporal_mode=temporal_mode,
        )
        seed_node_ids = {node_id for node_id, _ in seeds}

        # 3.5. Add working memory entities as additional seeds
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in seed_node_ids:
                    energy = cfg.working_memory_seed_energy * recency_score
                    if energy > 0.0:
                        seeds.append((item_id, energy))
                        seed_node_ids.add(item_id)

        # Build context gate (if enabled)
        context_gate = None
        if cfg.context_gating_enabled and predicate_cache is not None:
            query_emb = getattr(search_index, '_last_query_vec', None)
            if query_emb:
                from engram.activation.context_gate import build_context_gate

                context_gate = build_context_gate(
                    query_emb, predicate_cache, cfg,
                )

        spreading_bonuses, hop_distances = await spread_activation(
            seeds,
            graph_store,  # neighbor_provider
            cfg,
            group_id=group_id,
            community_store=community_store,
            context_gate=context_gate,
        )

        # 3.7. Merge spreading-discovered entities with real semantic similarity
        existing_ids = {eid for eid, _ in candidates}
        new_ids = [
            nid for nid in spreading_bonuses
            if nid not in existing_ids and spreading_bonuses[nid] > 0.0
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
    else:
        spreading_bonuses: dict[str, float] = {}
        hop_distances: dict[str, int] = {}
        seed_node_ids: set[str] = set()

    # 4. Score candidates (Thompson Sampling or deterministic)
    if cfg.ts_enabled:
        scored = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses=spreading_bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
        )
    else:
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses=spreading_bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
        )

    # 5. Cross-encoder re-ranking (if enabled)
    if cfg.reranker_enabled and reranker is not None:
        try:
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
                rerank_order = {eid: i for i, (eid, _) in enumerate(reranked)}
                scored.sort(
                    key=lambda sr: rerank_order.get(sr.node_id, len(scored)),
                )
        except Exception as e:
            logger.warning("Reranking failed (non-fatal): %s", e)

    # 6. MMR diversity (if enabled)
    if cfg.mmr_enabled:
        try:
            from engram.retrieval.mmr import apply_mmr

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

    # 7. Merge episode results into scored list
    if episode_results:
        scored.extend(episode_results)
        scored.sort(key=lambda sr: sr.score, reverse=True)

    # 8. Return top `limit` results (already sorted by score)
    return scored[:limit]
