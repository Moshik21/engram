"""Hybrid search index combining FTS5 keyword search with vector similarity."""

from __future__ import annotations

import asyncio
import logging

from engram.config import ActivationConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore, cosine_similarity, unpack_vector

logger = logging.getLogger(__name__)


class HybridSearchIndex:
    """Wraps FTS5 + SQLiteVectorStore. Implements SearchIndex protocol.

    Scores are weighted merge: fts_weight * fts_score + vec_weight * vec_score.
    Falls back to FTS5-only when embeddings are unavailable.
    """

    def __init__(
        self,
        fts: FTS5SearchIndex,
        vector_store: SQLiteVectorStore,
        provider: EmbeddingProvider,
        fts_weight: float = 0.3,
        vec_weight: float = 0.7,
        cfg: ActivationConfig | None = None,
    ) -> None:
        self._fts = fts
        self._vectors = vector_store
        self._provider = provider
        self._fts_weight = fts_weight
        self._vec_weight = vec_weight
        self._embeddings_enabled = provider.dimension() > 0
        self._cfg = cfg or ActivationConfig()
        self._last_query_vec: list[float] | None = None

    async def initialize(self, db=None) -> None:
        """Initialize both FTS5 and vector store."""
        await self._fts.initialize(db=db)
        await self._vectors.initialize(db=db)

    async def index_entity(self, entity: Entity) -> None:
        """Index an entity for both FTS5 and vector search."""
        # FTS5 is maintained via triggers — no-op
        # Embed entity name + summary if provider available
        if self._embeddings_enabled and entity.name:
            try:
                text = entity.name
                if entity.summary:
                    text = f"{entity.name}: {entity.summary}"
                embeddings = await self._provider.embed([text])
                if embeddings:
                    await self._vectors.upsert(
                        entity.id, "entity", entity.group_id, text, embeddings[0]
                    )
            except Exception as e:
                logger.warning("Failed to embed entity %s: %s", entity.id, e)

    async def index_episode(self, episode: Episode) -> None:
        """Index an episode for vector search."""
        if self._embeddings_enabled and episode.content:
            try:
                embeddings = await self._provider.embed([episode.content])
                if embeddings:
                    await self._vectors.upsert(
                        episode.id, "episode", episode.group_id, episode.content, embeddings[0]
                    )
            except Exception as e:
                logger.warning("Failed to embed episode %s: %s", episode.id, e)

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Search using hybrid FTS5 + vector scoring.

        Runs FTS5 and query embedding concurrently via asyncio.gather.
        Returns (entity_id, score) pairs normalized to 0.0-1.0.
        """
        self._last_query_vec = None

        # If embeddings not available, fall back to FTS5-only
        if not self._embeddings_enabled:
            fts_results = await self._fts.search(
                query=query, entity_types=entity_types, group_id=group_id, limit=limit * 2
            )
            return fts_results[:limit]

        # Check if we have any embeddings for this group
        if group_id and not await self._vectors.has_embeddings(group_id):
            fts_results = await self._fts.search(
                query=query, entity_types=entity_types, group_id=group_id, limit=limit * 2
            )
            return fts_results[:limit]

        # Run FTS5 search and query embedding concurrently
        try:
            fts_results, query_vec = await asyncio.gather(
                self._fts.search(
                    query=query, entity_types=entity_types,
                    group_id=group_id, limit=limit * 2,
                ),
                self._provider.embed_query(query),
            )
        except Exception as e:
            logger.warning("Parallel search failed, falling back to FTS5: %s", e)
            fts_results = await self._fts.search(
                query=query, entity_types=entity_types, group_id=group_id, limit=limit * 2
            )
            return fts_results[:limit]

        if not query_vec:
            return fts_results[:limit]

        self._last_query_vec = query_vec

        # Vector search uses the pre-computed query embedding
        try:
            vec_results = await self._vectors.search(
                query_vec, group_id or "default", content_type="entity", limit=limit * 2
            )
        except Exception as e:
            logger.warning("Vector search failed, falling back to FTS5: %s", e)
            return fts_results[:limit]

        if not vec_results:
            return fts_results[:limit]

        return self._merge_results(fts_results, vec_results, limit)

    def _merge_results(
        self,
        fts_results: list[tuple[str, float]],
        vec_results: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Merge FTS5 and vector results.

        When use_rrf is True (default), uses Reciprocal Rank Fusion:
            score(d) = Σ 1/(k + rank_i(d))
        Otherwise falls back to linear weighted merge.
        """
        if self._cfg.use_rrf:
            return self._merge_rrf(fts_results, vec_results, limit)
        return self._merge_linear(fts_results, vec_results, limit)

    def _merge_rrf(
        self,
        fts_results: list[tuple[str, float]],
        vec_results: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))."""
        k = self._cfg.rrf_k
        rrf_scores: dict[str, float] = {}

        for rank, (eid, _) in enumerate(fts_results, start=1):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank)

        for rank, (eid, _) in enumerate(vec_results, start=1):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (k + rank)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Normalize to 0-1
        if merged:
            max_score = merged[0][1]
            if max_score > 0:
                merged = [(eid, score / max_score) for eid, score in merged]

        return merged[:limit]

    def _merge_linear(
        self,
        fts_results: list[tuple[str, float]],
        vec_results: list[tuple[str, float]],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Linear weighted merge (legacy behavior)."""
        # Normalize FTS scores to 0-1
        fts_scores: dict[str, float] = {}
        if fts_results:
            fts_max = max(s for _, s in fts_results) if fts_results else 1.0
            for eid, score in fts_results:
                fts_scores[eid] = score / fts_max if fts_max > 0 else 0.0

        # Vector scores are already cosine similarity in [-1, 1], normalize to [0, 1]
        vec_scores: dict[str, float] = {}
        if vec_results:
            for eid, score in vec_results:
                vec_scores[eid] = max(0.0, (score + 1.0) / 2.0)  # map [-1,1] to [0,1]

        # Combine all candidate IDs
        all_ids = set(fts_scores.keys()) | set(vec_scores.keys())

        merged: list[tuple[str, float]] = []
        for eid in all_ids:
            fts_s = fts_scores.get(eid, 0.0)
            vec_s = vec_scores.get(eid, 0.0)
            combined = self._fts_weight * fts_s + self._vec_weight * vec_s
            merged.append((eid, combined))

        # Sort by combined score descending
        merged.sort(key=lambda x: x[1], reverse=True)

        # Normalize to 0-1
        if merged:
            max_score = merged[0][1]
            if max_score > 0:
                merged = [(eid, score / max_score) for eid, score in merged]

        return merged[:limit]

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> dict[str, float]:
        """Compute semantic similarity between query and stored entity embeddings.

        If query_embedding is provided, uses it directly instead of re-embedding.
        Falls back to self._last_query_vec from most recent search() if available.
        """
        if not self._embeddings_enabled or not entity_ids:
            return {}

        query_vec = query_embedding or self._last_query_vec
        if not query_vec:
            try:
                query_vec = await self._provider.embed_query(query)
                if not query_vec:
                    return {}
            except Exception:
                return {}

        results: dict[str, float] = {}
        for eid in entity_ids:
            cursor = await self._vectors.db.execute(
                "SELECT embedding, dimensions FROM embeddings WHERE id = ? AND group_id = ?",
                (eid, group_id or "default"),
            )
            row = await cursor.fetchone()
            if row:
                vec = unpack_vector(row["embedding"], row["dimensions"])
                results[eid] = max(0.0, cosine_similarity(query_vec, vec))
        return results

    async def search_episodes(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search episodes using hybrid FTS5 + vector scoring.

        Runs FTS5 episode search and vector search concurrently.
        Returns (episode_id, score) pairs normalized to 0.0-1.0.
        """
        # If embeddings not available, fall back to FTS5-only
        if not self._embeddings_enabled:
            return await self._fts.search_episodes(
                query=query, group_id=group_id, limit=limit,
            )

        # Check if we have any embeddings for this group
        if group_id and not await self._vectors.has_embeddings(group_id):
            return await self._fts.search_episodes(
                query=query, group_id=group_id, limit=limit,
            )

        # Run FTS5 episode search and query embedding concurrently
        try:
            fts_results, query_vec = await asyncio.gather(
                self._fts.search_episodes(
                    query=query, group_id=group_id, limit=limit * 2,
                ),
                self._provider.embed_query(query),
            )
        except Exception as e:
            logger.warning("Parallel episode search failed, falling back to FTS5: %s", e)
            return await self._fts.search_episodes(
                query=query, group_id=group_id, limit=limit,
            )

        if not query_vec:
            return fts_results[:limit]

        # Vector search for episodes
        try:
            vec_results = await self._vectors.search(
                query_vec, group_id or "default", content_type="episode", limit=limit * 2,
            )
        except Exception as e:
            logger.warning("Vector episode search failed, falling back to FTS5: %s", e)
            return fts_results[:limit]

        if not vec_results:
            return fts_results[:limit]

        return self._merge_results(fts_results, vec_results, limit)

    async def remove(self, entity_id: str) -> None:
        """Remove entity from both FTS5 and vector store."""
        await self._fts.remove(entity_id)
        await self._vectors.remove(entity_id)
