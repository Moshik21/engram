"""Graph-Connected Maximal Marginal Relevance (GC-MMR) re-ranking."""

from __future__ import annotations

import numpy as np

from engram.retrieval.scorer import ScoredResult


async def apply_gc_mmr(
    results: list[ScoredResult],
    graph_store,
    group_id: str,
    entity_embeddings: dict[str, list[float]],
    lambda_rel: float = 0.7,
    lambda_div: float = 0.2,
    lambda_conn: float = 0.1,
    top_n: int = 10,
) -> list[ScoredResult]:
    """Re-rank results using Graph-Connected MMR.

    At each step, select the result maximizing:
        GC-MMR(d) = λ_rel * rel - λ_div * max_sim(d, selected) + λ_conn * connectivity

    Where connectivity measures how well a candidate connects to the already-selected set
    via graph edges.

    Args:
        results: Scored retrieval results (already sorted by relevance).
        graph_store: Graph store for neighbor lookups.
        group_id: Group ID for graph queries.
        entity_embeddings: {entity_id: embedding_vector} for cosine similarity.
        lambda_rel: Weight for relevance (higher = more relevant results).
        lambda_div: Weight for diversity penalty (higher = more diverse results).
        lambda_conn: Weight for graph connectivity bonus (higher = more connected results).
        top_n: Number of results to return.

    Returns:
        Re-ranked list of ScoredResult.
    """
    if not results:
        return []
    if len(results) <= 1:
        return results[:top_n]

    top_n = min(top_n, len(results))

    # Pre-fetch neighbor adjacency for all result entity IDs (single batch)
    # neighbor_map: {entity_id: {neighbor_id: edge_weight}}
    result_ids = {r.node_id for r in results}
    neighbor_map: dict[str, dict[str, float]] = {}
    for nid in result_ids:
        try:
            neighbors = await graph_store.get_active_neighbors_with_weights(nid, group_id)
            neighbor_map[nid] = {n[0]: n[1] for n in neighbors if n[0] in result_ids}
        except Exception:
            neighbor_map[nid] = {}

    # Build normalized embedding vectors
    embeddings: dict[str, np.ndarray] = {}
    for nid in result_ids:
        if nid in entity_embeddings and entity_embeddings[nid]:
            vec = np.asarray(entity_embeddings[nid], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                embeddings[nid] = vec / norm

    # Normalize relevance scores to [0, 1]
    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    score_range = max_score - min_score if max_score > min_score else 1.0

    selected: list[ScoredResult] = []
    remaining = list(results)

    for _ in range(top_n):
        if not remaining:
            break

        best_idx = 0
        best_gc_mmr = float("-inf")

        for i, candidate in enumerate(remaining):
            # Relevance component (normalized)
            rel = (candidate.score - min_score) / score_range

            # Diversity component: max cosine similarity to any selected result
            max_sim = 0.0
            if selected and candidate.node_id in embeddings:
                cand_vec = embeddings[candidate.node_id]
                for sel in selected:
                    if sel.node_id in embeddings:
                        sim = float(np.dot(cand_vec, embeddings[sel.node_id]))
                        max_sim = max(max_sim, sim)

            # Connectivity component: graph edges to selected set
            conn = 0.0
            if selected:
                cand_neighbors = neighbor_map.get(candidate.node_id, {})
                total_weight = 0.0
                for sel in selected:
                    # Check if candidate is a neighbor of selected entity
                    if sel.node_id in cand_neighbors:
                        total_weight += cand_neighbors[sel.node_id]
                    # Check reverse direction too
                    sel_neighbors = neighbor_map.get(sel.node_id, {})
                    if candidate.node_id in sel_neighbors:
                        total_weight += sel_neighbors[candidate.node_id]
                conn = total_weight / max(1, len(selected))

            gc_mmr_score = lambda_rel * rel - lambda_div * max_sim + lambda_conn * conn

            if gc_mmr_score > best_gc_mmr:
                best_gc_mmr = gc_mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
