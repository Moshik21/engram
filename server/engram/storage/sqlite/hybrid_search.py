"""Hybrid search index combining FTS5 keyword search with vector similarity."""

from __future__ import annotations

import asyncio
import logging

from engram.config import ActivationConfig
from engram.embeddings.provider import EmbeddingProvider, truncate_vectors
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
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
        storage_dim: int = 0,
        embed_provider: str = "",
        embed_model: str = "",
    ) -> None:
        self._fts = fts
        self._vectors = vector_store
        self._provider = provider
        self._fts_weight = fts_weight
        self._vec_weight = vec_weight
        self._embeddings_enabled = provider.dimension() > 0
        self._cfg = cfg or ActivationConfig()
        self._last_query_vec: list[float] | None = None
        self._storage_dim = storage_dim
        self._embed_provider = embed_provider
        self._embed_model = embed_model

    async def initialize(self, db=None) -> None:
        """Initialize both FTS5 and vector store."""
        await self._fts.initialize(db=db)
        await self._vectors.initialize(db=db)
        await self._check_embedding_version()

    async def index_entity(self, entity: Entity) -> None:
        """Index an entity for both FTS5 and vector search."""
        if self._embeddings_enabled and entity.name:
            try:
                text = entity.name
                if entity.summary:
                    text = f"{entity.name}: {entity.summary}"
                embeddings = await self._provider.embed([text])
                if embeddings:
                    if self._storage_dim > 0:
                        embeddings = truncate_vectors(embeddings, self._storage_dim)
                    await self._vectors.upsert(
                        entity.id, "entity", entity.group_id, text, embeddings[0],
                        embed_provider=self._embed_provider,
                        embed_model=self._embed_model,
                    )
            except Exception as e:
                logger.warning("Failed to embed entity %s: %s", entity.id, e)

    async def index_episode(self, episode: Episode) -> None:
        """Index an episode for vector search."""
        if self._embeddings_enabled and episode.content:
            try:
                embeddings = await self._provider.embed([episode.content])
                if embeddings:
                    if self._storage_dim > 0:
                        embeddings = truncate_vectors(embeddings, self._storage_dim)
                    await self._vectors.upsert(
                        episode.id, "episode", episode.group_id, episode.content,
                        embeddings[0],
                        embed_provider=self._embed_provider,
                        embed_model=self._embed_model,
                    )
            except Exception as e:
                logger.warning("Failed to embed episode %s: %s", episode.id, e)

    async def index_episode_cue(self, cue: EpisodeCue) -> None:
        """Index cue text for vector search."""
        if not self._embeddings_enabled:
            return
        if not cue.cue_text:
            try:
                await self._vectors.remove(cue.episode_id, content_type="episode_cue")
            except Exception as e:
                logger.warning("Failed to remove cue embedding %s: %s", cue.episode_id, e)
            return
        try:
            embeddings = await self._provider.embed([cue.cue_text])
            if not embeddings:
                return
            if self._storage_dim > 0:
                embeddings = truncate_vectors(embeddings, self._storage_dim)
            await self._vectors.upsert(
                cue.episode_id,
                "episode_cue",
                cue.group_id,
                cue.cue_text,
                embeddings[0],
                embed_provider=self._embed_provider,
                embed_model=self._embed_model,
            )
        except Exception as e:
            logger.warning("Failed to embed cue %s: %s", cue.episode_id, e)

    async def batch_index_entities(self, entities: list[Entity]) -> int:
        """Batch-embed and index multiple entities. Returns count indexed."""
        if not self._embeddings_enabled or not entities:
            return 0
        texts, valid = [], []
        for e in entities:
            if e.name:
                t = e.name
                if e.summary:
                    t = f"{e.name}: {e.summary}"
                texts.append(t)
                valid.append(e)
        if not texts:
            return 0
        embeddings = await self._provider.embed(texts)
        if not embeddings:
            return 0
        if self._storage_dim > 0:
            embeddings = truncate_vectors(embeddings, self._storage_dim)
        items: list[tuple[str, str, str, str | None, list[float]]] = [
            (e.id, "entity", e.group_id, t, vec)
            for e, t, vec in zip(valid, texts, embeddings)
        ]
        await self._vectors.batch_upsert(
            items,
            embed_provider=self._embed_provider,
            embed_model=self._embed_model,
        )
        return len(items)

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
                    query=query,
                    entity_types=entity_types,
                    group_id=group_id,
                    limit=limit * 2,
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

        # Truncate query vector to storage dimension for consistent comparison
        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]
        self._last_query_vec = query_vec

        # Vector search uses the pre-computed query embedding
        try:
            vec_results = await self._vectors.search(
                query_vec, group_id or "default", content_type="entity",
                limit=limit * 2, storage_dim=self._storage_dim,
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

        # Truncate query vector to storage dimension
        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]

        results: dict[str, float] = {}
        for eid in entity_ids:
            cursor = await self._vectors.db.execute(
                "SELECT embedding, dimensions FROM embeddings WHERE id = ? AND group_id = ?",
                (eid, group_id or "default"),
            )
            row = await cursor.fetchone()
            if row:
                vec = unpack_vector(row["embedding"], row["dimensions"])
                # Truncate stored vector if it's larger than storage_dim
                if self._storage_dim > 0 and len(vec) > self._storage_dim:
                    vec = vec[:self._storage_dim]
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
                query=query,
                group_id=group_id,
                limit=limit,
            )

        # Check if we have any embeddings for this group
        if group_id and not await self._vectors.has_embeddings(group_id):
            return await self._fts.search_episodes(
                query=query,
                group_id=group_id,
                limit=limit,
            )

        # Run FTS5 episode search and query embedding concurrently
        try:
            fts_results, query_vec = await asyncio.gather(
                self._fts.search_episodes(
                    query=query,
                    group_id=group_id,
                    limit=limit * 2,
                ),
                self._provider.embed_query(query),
            )
        except Exception as e:
            logger.warning("Parallel episode search failed, falling back to FTS5: %s", e)
            return await self._fts.search_episodes(
                query=query,
                group_id=group_id,
                limit=limit,
            )

        if not query_vec:
            return fts_results[:limit]

        # Truncate query vector to storage dimension
        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]

        # Vector search for episodes
        try:
            vec_results = await self._vectors.search(
                query_vec,
                group_id or "default",
                content_type="episode",
                limit=limit * 2,
                storage_dim=self._storage_dim,
            )
        except Exception as e:
            logger.warning("Vector episode search failed, falling back to FTS5: %s", e)
            return fts_results[:limit]

        if not vec_results:
            return fts_results[:limit]

        return self._merge_results(fts_results, vec_results, limit)

    async def search_episode_cues(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search episode cues using hybrid FTS5 + vector scoring."""
        if not self._embeddings_enabled:
            return await self._fts.search_episode_cues(query=query, group_id=group_id, limit=limit)

        if group_id and not await self._vectors.has_embeddings(group_id):
            return await self._fts.search_episode_cues(query=query, group_id=group_id, limit=limit)

        try:
            fts_results, query_vec = await asyncio.gather(
                self._fts.search_episode_cues(
                    query=query,
                    group_id=group_id,
                    limit=limit * 2,
                ),
                self._provider.embed_query(query),
            )
        except Exception as e:
            logger.warning("Parallel cue search failed, falling back to FTS5: %s", e)
            return await self._fts.search_episode_cues(query=query, group_id=group_id, limit=limit)

        if not query_vec:
            return fts_results[:limit]
        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]

        try:
            vec_results = await self._vectors.search(
                query_vec,
                group_id or "default",
                content_type="episode_cue",
                limit=limit * 2,
                storage_dim=self._storage_dim,
            )
        except Exception as e:
            logger.warning("Vector cue search failed, falling back to FTS5: %s", e)
            return fts_results[:limit]

        if not vec_results:
            return fts_results[:limit]

        return self._merge_results(fts_results, vec_results, limit)

    async def close(self) -> None:
        """No-op — connection is shared with the graph store."""
        pass

    async def get_entity_embeddings(
        self,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Batch retrieve entity embeddings from vector store."""
        if not self._embeddings_enabled or not entity_ids:
            return {}
        results: dict[str, list[float]] = {}
        placeholders = ",".join("?" * len(entity_ids))
        gid = group_id or "default"
        cursor = await self._vectors.db.execute(
            f"SELECT id, embedding, dimensions FROM embeddings "
            f"WHERE id IN ({placeholders}) AND group_id = ? AND content_type = 'entity'",
            [*entity_ids, gid],
        )
        for row in await cursor.fetchall():
            vec = unpack_vector(row["embedding"], row["dimensions"])
            results[row["id"]] = vec
        return results

    async def get_graph_embeddings(
        self,
        entity_ids: list[str],
        method: str = "node2vec",
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Retrieve graph structural embeddings for entities."""
        if not entity_ids:
            return {}
        from engram.embeddings.graph.storage import GraphEmbeddingStore

        store = GraphEmbeddingStore()
        return await store.get_embeddings(
            self._vectors.db, entity_ids, method, group_id or "default",
        )

    async def remove(self, entity_id: str) -> None:
        """Remove entity from both FTS5 and vector store."""
        await self._fts.remove(entity_id)
        await self._vectors.remove(entity_id)

    async def _check_embedding_version(self) -> None:
        """Warn if stored embeddings have different dimensions or provider."""
        if not self._embeddings_enabled:
            return
        try:
            cursor = await self._vectors.db.execute(
                "SELECT dimensions, embed_provider, embed_model, COUNT(*) as cnt "
                "FROM embeddings GROUP BY dimensions, embed_provider, embed_model"
            )
            rows = await cursor.fetchall()
            current_dim = self._storage_dim or self._provider.dimension()
            for row in rows:
                if row["dimensions"] != current_dim and row["cnt"] > 0:
                    logger.warning(
                        "Embedding dimension mismatch: %d vectors with dim=%d "
                        "(current=%d). Run consolidation reindex to update.",
                        row["cnt"], row["dimensions"], current_dim,
                    )
                stored_provider = row["embed_provider"] if "embed_provider" in row.keys() else ""
                if stored_provider and stored_provider != self._embed_provider and row["cnt"] > 0:
                    logger.warning(
                        "Embedding provider mismatch: %d vectors from '%s' "
                        "(current='%s'). Run consolidation reindex to update.",
                        row["cnt"], stored_provider, self._embed_provider,
                    )
        except Exception:
            pass  # Non-fatal — table may not have versioning columns yet
