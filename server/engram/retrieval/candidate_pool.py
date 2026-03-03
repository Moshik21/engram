"""Multi-pool candidate generation: search + activation + graph + working memory."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from engram.config import ActivationConfig

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer
    from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

logger = logging.getLogger(__name__)


async def _search_pool(
    query: str,
    group_id: str,
    search_index: SearchIndex,
    limit: int,
) -> list[tuple[str, float]]:
    """Pool 1: semantic/FTS search candidates."""
    try:
        results = await search_index.search(
            query=query, group_id=group_id, limit=limit,
        )
        return results or []
    except Exception as e:
        logger.warning("Search pool failed (non-fatal): %s", e)
        return []


async def _activation_pool(
    group_id: str,
    activation_store: ActivationStore,
    limit: int,
    now: float,
) -> list[tuple[str, float]]:
    """Pool 2: top activated entities by ACT-R base-level activation.

    get_top_activated already sorts by computed activation internally.
    We re-compute here to get the actual score for RRF ranking.
    """
    try:
        from engram.activation.engine import compute_activation
        from engram.config import ActivationConfig

        top = await activation_store.get_top_activated(
            group_id=group_id, limit=limit, now=now,
        )
        cfg = ActivationConfig()
        results: list[tuple[str, float]] = []
        for eid, state in top:
            act = compute_activation(state.access_history, now, cfg)
            results.append((eid, act))
        return results
    except Exception as e:
        logger.warning("Activation pool failed (non-fatal): %s", e)
        return []


async def _graph_neighborhood_pool(
    seed_ids: list[str],
    group_id: str,
    graph_store: GraphStore,
    max_neighbors: int,
    pool_limit: int,
) -> list[tuple[str, float]]:
    """Pool 3: 1-hop neighbors of top search seeds, ranked by fan-in count."""
    try:
        if not seed_ids:
            return []

        # Count how many seeds each neighbor connects to (fan-in)
        fan_in: dict[str, int] = {}
        seed_set = set(seed_ids)

        for sid in seed_ids:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                entity_id=sid, group_id=group_id,
            )
            for nid, _weight, _pred in neighbors[:max_neighbors]:
                if nid not in seed_set:
                    fan_in[nid] = fan_in.get(nid, 0) + 1

        # Sort by fan-in descending, take top pool_limit
        ranked = sorted(fan_in.items(), key=lambda x: x[1], reverse=True)
        return [(nid, float(count)) for nid, count in ranked[:pool_limit]]
    except Exception as e:
        logger.warning("Graph neighborhood pool failed (non-fatal): %s", e)
        return []


async def _working_memory_pool(
    working_memory: WorkingMemoryBuffer,
    group_id: str,
    graph_store: GraphStore,
    now: float,
    max_neighbors: int,
    pool_limit: int,
) -> list[tuple[str, float]]:
    """Pool 4: working memory entities + dampened 1-hop neighbors."""
    try:
        wm_candidates = working_memory.get_candidates(now)
        if not wm_candidates:
            return []

        results: dict[str, float] = {}
        wm_ids: set[str] = set()

        # Add WM entities with their recency score
        for item_id, recency_score, _item_type in wm_candidates:
            results[item_id] = recency_score
            wm_ids.add(item_id)

        # Expand 1-hop neighbors with 0.5x dampening
        for item_id, recency_score, _item_type in wm_candidates:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                entity_id=item_id, group_id=group_id,
            )
            for nid, _weight, _pred in neighbors[:max_neighbors]:
                if nid not in wm_ids:
                    dampened = recency_score * 0.5
                    # Keep the higher score if already seen
                    if nid not in results or dampened > results[nid]:
                        results[nid] = dampened

        # Sort by score descending, take top pool_limit
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:pool_limit]
    except Exception as e:
        logger.warning("Working memory pool failed (non-fatal): %s", e)
        return []


def _merge_pools_rrf(
    pools: list[list[tuple[str, float]]],
    rrf_k: int,
    limit: int,
) -> list[str]:
    """Merge ranked lists via Reciprocal Rank Fusion.

    score(d) = sum over pools: 1 / (rrf_k + rank_i(d))
    Returns entity IDs ordered by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}

    for pool in pools:
        for rank, (eid, _score) in enumerate(pool):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [eid for eid, _ in ranked[:limit]]


async def generate_candidates(
    query: str,
    group_id: str,
    search_index: SearchIndex,
    activation_store: ActivationStore,
    graph_store: GraphStore,
    cfg: ActivationConfig,
    now: float | None = None,
    working_memory: WorkingMemoryBuffer | None = None,
) -> list[tuple[str, float]]:
    """Orchestrate multi-pool candidate generation.

    Returns (entity_id, real_semantic_similarity) tuples in RRF-merged order.
    """
    if now is None:
        now = time.time()

    # Step 1: Run search + activation pools concurrently
    search_results, activation_results = await asyncio.gather(
        _search_pool(query, group_id, search_index, cfg.pool_search_limit),
        _activation_pool(group_id, activation_store, cfg.pool_activation_limit, now),
    )

    # Step 2: Graph neighborhood from top search seeds (sequential)
    seed_ids = [eid for eid, _ in search_results[:cfg.pool_graph_seed_count]]
    graph_results = await _graph_neighborhood_pool(
        seed_ids, group_id, graph_store,
        cfg.pool_graph_max_neighbors, cfg.pool_graph_limit,
    )

    # Step 3: Working memory pool (if provided)
    wm_results: list[tuple[str, float]] = []
    if working_memory is not None and cfg.working_memory_enabled:
        wm_results = await _working_memory_pool(
            working_memory, group_id, graph_store,
            now, cfg.pool_wm_max_neighbors, cfg.pool_wm_limit,
        )

    # Step 4: Merge non-empty pools via RRF
    pools = [p for p in [search_results, activation_results, graph_results, wm_results] if p]
    if not pools:
        return []

    merged_ids = _merge_pools_rrf(pools, cfg.rrf_k, cfg.pool_total_limit)

    # Step 5: Build semantic score map from search results
    search_scores: dict[str, float] = {eid: score for eid, score in search_results}

    # Backfill real semantic scores for non-search entities
    non_search_ids = [eid for eid in merged_ids if eid not in search_scores]
    backfilled: dict[str, float] = {}
    if non_search_ids:
        try:
            backfilled = await search_index.compute_similarity(
                query=query, entity_ids=non_search_ids, group_id=group_id,
            )
        except Exception as e:
            logger.warning("Semantic backfill failed (non-fatal): %s", e)

    # Step 6: Return (entity_id, real_semantic_similarity) in RRF order
    return [
        (eid, search_scores.get(eid, backfilled.get(eid, 0.0)))
        for eid in merged_ids
    ]
