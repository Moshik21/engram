"""Context-gated spreading activation.

Computes per-edge gate factors based on cosine similarity between the
query embedding and each predicate's natural-language embedding. This
makes spreading activation situation-aware -- energy flows preferentially
through edges that are semantically relevant to the current query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engram.config import ActivationConfig
    from engram.embeddings.provider import EmbeddingProvider


class ContextGate:
    """Per-query context gate for spreading activation.

    Computes ``floor + (1 - floor) * max(0, cosine_sim(query, predicate))``
    for each predicate, with internal memoization.
    """

    def __init__(
        self,
        query_embedding: list[float],
        predicate_embeddings: dict[str, list[float]],
        floor: float = 0.3,
    ) -> None:
        self._floor = floor
        self._cache: dict[str, float] = {}

        # Pre-normalize query vector
        q = np.array(query_embedding, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        self._query_norm = q / q_norm if q_norm > 0 else q

        # Pre-normalize predicate vectors
        self._pred_norms: dict[str, np.ndarray] = {}
        for pred, emb in predicate_embeddings.items():
            v = np.array(emb, dtype=np.float64)
            v_norm = np.linalg.norm(v)
            self._pred_norms[pred] = v / v_norm if v_norm > 0 else v

    def gate(self, predicate: str) -> float:
        """Return the context gate factor for a predicate.

        Unknown predicates (not in the predicate embedding cache) return 1.0
        (no gating).
        """
        if predicate in self._cache:
            return self._cache[predicate]

        if predicate not in self._pred_norms:
            return 1.0

        sim = float(np.dot(self._query_norm, self._pred_norms[predicate]))
        value = self._floor + (1.0 - self._floor) * max(0.0, sim)
        self._cache[predicate] = value
        return value


class PredicateEmbeddingCache:
    """Shared cache of predicate natural-name embeddings.

    Initialized once at startup (or corpus load for benchmarks). Embeds the
    ~16 predicate natural names using the configured EmbeddingProvider.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] = {}
        self._initialized: bool = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def get_embeddings(self) -> dict[str, list[float]]:
        return dict(self._embeddings)

    def set_embeddings(self, embeddings: dict[str, list[float]]) -> None:
        """Inject embeddings directly (for testing/benchmarks)."""
        self._embeddings = dict(embeddings)
        self._initialized = bool(embeddings)

    async def initialize(
        self,
        cfg: ActivationConfig,
        provider: EmbeddingProvider,
    ) -> None:
        """Embed all predicate natural names from config. Idempotent."""
        if self._initialized:
            return

        natural_names = cfg.predicate_natural_names
        if not natural_names:
            self._initialized = True
            return

        predicates = list(natural_names.keys())
        texts = [natural_names[p] for p in predicates]

        embeddings = await provider.embed(texts)
        if not embeddings:
            # Provider returned empty (e.g. NoopProvider) -- graceful degradation
            self._initialized = True
            return

        for pred, emb in zip(predicates, embeddings):
            if emb:  # Skip empty embeddings
                self._embeddings[pred] = emb

        self._initialized = True


def build_context_gate(
    query_embedding: list[float] | None,
    predicate_cache: PredicateEmbeddingCache | None,
    cfg: ActivationConfig,
) -> ContextGate | None:
    """Factory: build a ContextGate if inputs are available.

    Returns None if query embedding is missing, predicate cache is
    uninitialized/empty, or context gating is disabled.
    """
    if not cfg.context_gating_enabled:
        return None

    if not query_embedding:
        return None

    if predicate_cache is None or not predicate_cache.initialized:
        return None

    pred_embs = predicate_cache.get_embeddings()
    if not pred_embs:
        return None

    return ContextGate(
        query_embedding=query_embedding,
        predicate_embeddings=pred_embs,
        floor=cfg.context_gate_floor,
    )
