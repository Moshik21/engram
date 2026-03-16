"""Embedding-based relevance confidence scorer.

Computes how well each retrieval result answers the query — orthogonal to
the composite ``score`` which measures memory strength (ACT-R + spreading +
edge proximity + semantic).  Uses cosine similarity between query and
best-available text representation.  Zero additional LLM cost.
"""

from __future__ import annotations

import logging

import numpy as np

from engram.embeddings.provider import EmbeddingProvider
from engram.retrieval.scorer import ScoredResult

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Full-dimension cosine similarity between two vectors."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class RelevanceScorer:
    """Scores retrieval results by query-relevance (not memory strength).

    For entities, reuses the ``semantic_similarity`` already computed during
    vector search (zero extra cost).  For episodes with chunk text, batch-
    embeds chunk texts and picks the best chunk.  Falls back to episode-level
    similarity when chunks are unavailable.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    async def score_results(
        self,
        query: str,
        results: list[ScoredResult],
        entity_summaries: dict[str, str],
        episode_contents: dict[str, str],
        chunk_texts: dict[str, str],
        query_vec: list[float] | None = None,
    ) -> None:
        """Compute ``relevance_confidence`` for each result (mutates in-place).

        Parameters
        ----------
        query:
            The user's recall query.
        results:
            Scored results from the retrieval pipeline.
        entity_summaries:
            Mapping of entity node_id → summary text.
        episode_contents:
            Mapping of episode node_id → episode content text.
        chunk_texts:
            Mapping of episode node_id → best chunk text (from chunk_context).
        query_vec:
            Pre-computed query embedding (reused from search to avoid a
            redundant embed call).  If ``None``, will be computed.
        """
        if not results:
            return

        # Get or compute query vector
        if query_vec is None or not query_vec:
            query_vec = await self._provider.embed_query(query)
        if not query_vec:
            # No embeddings available — leave relevance at 0.0
            return

        # Collect texts that need embedding (episodes with chunk/content text)
        texts_to_embed: list[str] = []
        text_node_ids: list[str] = []

        for sr in results:
            if sr.result_type in {"episode", "cue_episode"}:
                # Prefer chunk text (more precise), fall back to episode content
                text = chunk_texts.get(sr.node_id) or episode_contents.get(sr.node_id)
                if text:
                    texts_to_embed.append(text)
                    text_node_ids.append(sr.node_id)

        # Batch embed episode/chunk texts
        text_vecs: dict[str, list[float]] = {}
        if texts_to_embed:
            try:
                vecs = await self._provider.embed(texts_to_embed)
                for node_id, vec in zip(text_node_ids, vecs):
                    if vec:
                        text_vecs[node_id] = vec
            except Exception:
                logger.debug("Failed to embed episode texts for relevance", exc_info=True)

        # Score each result
        for sr in results:
            if sr.result_type in {"episode", "cue_episode"}:
                if sr.node_id in text_vecs:
                    sr.relevance_confidence = cosine_similarity(query_vec, text_vecs[sr.node_id])
                elif sr.semantic_similarity > 0:
                    # Fall back to episode-level semantic similarity from search
                    sr.relevance_confidence = sr.semantic_similarity
            else:
                # Entity: semantic_similarity IS the cosine sim from vector search
                sr.relevance_confidence = sr.semantic_similarity


def compute_answer_containment(
    gold_answer_vec: list[float],
    evidence_vecs: list[list[float]],
) -> float:
    """Max cosine similarity between gold answer and evidence vectors.

    Used by the benchmark evaluator to judge answer correctness without
    an LLM.  Returns 0.0 if no evidence vectors are provided.
    """
    if not gold_answer_vec or not evidence_vecs:
        return 0.0

    best = 0.0
    for ev in evidence_vecs:
        if ev:
            sim = cosine_similarity(gold_answer_vec, ev)
            if sim > best:
                best = sim
    return best
