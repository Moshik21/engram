"""Maximal Marginal Relevance (MMR) diversity re-ranking."""

from __future__ import annotations

import numpy as np

from engram.retrieval.scorer import ScoredResult


def apply_mmr(
    results: list[ScoredResult],
    entity_embeddings: dict[str, list[float]],
    lambda_param: float = 0.7,
    top_n: int = 10,
) -> list[ScoredResult]:
    """Re-rank results using Maximal Marginal Relevance.

    At each step, select the result maximizing:
        MMR(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)

    Args:
        results: Scored retrieval results (already sorted by relevance).
        entity_embeddings: {entity_id: embedding_vector} for cosine similarity.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
        top_n: Number of results to return.

    Returns:
        Re-ranked list of ScoredResult.
    """
    if not results:
        return []

    if len(results) <= 1:
        return results[:top_n]

    top_n = min(top_n, len(results))

    # Build normalized embedding vectors for candidates that have embeddings
    candidate_ids = [r.node_id for r in results]
    embeddings: dict[str, np.ndarray] = {}
    for nid in candidate_ids:
        if nid in entity_embeddings and entity_embeddings[nid]:
            vec = np.asarray(entity_embeddings[nid], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                embeddings[nid] = vec / norm

    # If no embeddings available, return as-is
    if not embeddings:
        return results[:top_n]

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
        best_mmr = float("-inf")

        for i, candidate in enumerate(remaining):
            # Relevance component (normalized)
            rel = (candidate.score - min_score) / score_range

            # Diversity component: max similarity to any selected result
            max_sim = 0.0
            if selected and candidate.node_id in embeddings:
                cand_vec = embeddings[candidate.node_id]
                for sel in selected:
                    if sel.node_id in embeddings:
                        sim = float(np.dot(cand_vec, embeddings[sel.node_id]))
                        max_sim = max(max_sim, sim)

            mmr_score = lambda_param * rel - (1.0 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected
