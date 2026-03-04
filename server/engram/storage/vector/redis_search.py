"""Redis Search HNSW vector index for full mode."""

from __future__ import annotations

import logging
import struct

from engram.config import EmbeddingConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.models.entity import Entity
from engram.models.episode import Episode

logger = logging.getLogger(__name__)


def pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector into binary (4 bytes per float, little-endian)."""
    return struct.pack(f"<{len(vec)}f", *vec)


class RedisSearchIndex:
    """HNSW vector index using Redis Search (FT.CREATE / FT.SEARCH).

    Implements the SearchIndex protocol for full mode.
    """

    INDEX_NAME = "engram_vectors"

    def __init__(
        self,
        redis,
        provider: EmbeddingProvider,
        config: EmbeddingConfig,
    ) -> None:
        self._redis = redis
        self._provider = provider
        self._config = config
        self._embeddings_enabled = provider.dimension() > 0

    async def initialize(self) -> None:
        """Create the FT.CREATE index idempotently."""
        if not self._embeddings_enabled:
            logger.info("RedisSearchIndex: embeddings disabled (NoopProvider), skipping index")
            return

        # Check if index already exists
        try:
            await self._redis.execute_command("FT.INFO", self.INDEX_NAME)
            logger.info("RedisSearchIndex: index '%s' already exists", self.INDEX_NAME)
            return
        except Exception:
            pass  # Index does not exist, create it

        dim = self._provider.dimension()
        m = self._config.hnsw_m
        ef_construction = self._config.hnsw_ef_construction

        try:
            await self._redis.execute_command(
                "FT.CREATE",
                self.INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                "engram:",
                "SCHEMA",
                "group_id",
                "TAG",
                "content_type",
                "TAG",
                "source_id",
                "TAG",
                "text",
                "TEXT",
                "entity_type",
                "TAG",
                "created_at",
                "NUMERIC",
                "SORTABLE",
                "embedding",
                "VECTOR",
                "HNSW",
                "10",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(dim),
                "DISTANCE_METRIC",
                "COSINE",
                "M",
                str(m),
                "EF_CONSTRUCTION",
                str(ef_construction),
            )
            logger.info(
                "RedisSearchIndex: created index '%s' (dim=%d, M=%d, ef=%d)",
                self.INDEX_NAME,
                dim,
                m,
                ef_construction,
            )
        except Exception as e:
            logger.warning("RedisSearchIndex: FT.CREATE failed: %s", e)

    def _hash_key(self, group_id: str, content_type: str, item_id: str) -> str:
        return f"engram:{group_id}:vec:{content_type}:{item_id}"

    async def index_entity(self, entity: Entity) -> None:
        """Embed and index an entity."""
        if not self._embeddings_enabled or not entity.name:
            return

        text = entity.name
        if entity.summary:
            text = f"{entity.name}: {entity.summary}"

        try:
            embeddings = await self._provider.embed([text])
            if not embeddings:
                return

            key = self._hash_key(entity.group_id, "entity", entity.id)
            vec_bytes = pack_vector(embeddings[0])
            created_ts = entity.created_at.timestamp() if entity.created_at else 0.0

            await self._redis.hset(
                key,
                mapping={
                    "group_id": entity.group_id,
                    "content_type": "entity",
                    "source_id": entity.id,
                    "text": text,
                    "entity_type": entity.entity_type or "",
                    "created_at": str(created_ts),
                    "embedding": vec_bytes,
                },
            )
        except Exception as e:
            logger.warning("Failed to index entity %s: %s", entity.id, e)

    async def index_episode(self, episode: Episode) -> None:
        """Embed and index an episode."""
        if not self._embeddings_enabled or not episode.content:
            return

        try:
            embeddings = await self._provider.embed([episode.content])
            if not embeddings:
                return

            key = self._hash_key(episode.group_id, "episode", episode.id)
            vec_bytes = pack_vector(embeddings[0])
            created_ts = episode.created_at.timestamp() if episode.created_at else 0.0

            await self._redis.hset(
                key,
                mapping={
                    "group_id": episode.group_id,
                    "content_type": "episode",
                    "source_id": episode.id,
                    "text": episode.content,
                    "entity_type": "",
                    "created_at": str(created_ts),
                    "embedding": vec_bytes,
                },
            )
        except Exception as e:
            logger.warning("Failed to index episode %s: %s", episode.id, e)

    async def _text_search(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Full-text search using the indexed TEXT field in Redis Search."""
        filter_parts = []
        if group_id:
            filter_parts.append(f"@group_id:{{{group_id}}}")
        filter_parts.append("@content_type:{entity}")

        # Tokenize and escape query for Redis Search
        tokens = [t.strip("?!.,;:'\"") for t in query.split() if len(t) > 2]
        if not tokens:
            return []
        text_clause = "|".join(tokens)  # OR search
        filter_parts.append(f"@text:({text_clause})")

        ft_query = " ".join(filter_parts)

        try:
            result = await self._redis.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                ft_query,
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "2",
                "source_id",
                "score",
            )
        except Exception as e:
            logger.debug("Redis text search failed (non-fatal): %s", e)
            return []

        if not result or result[0] == 0:
            return []

        # Parse results — source_id with a fixed baseline score
        results: list[tuple[str, float]] = []
        i = 1
        while i < len(result):
            i += 1  # skip key
            if i >= len(result):
                break
            fields = result[i]
            i += 1
            field_dict = {}
            for j in range(0, len(fields), 2):
                k = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                v = (
                    fields[j + 1].decode()
                    if isinstance(fields[j + 1], bytes)
                    else fields[j + 1]
                )
                field_dict[k] = v
            source_id = field_dict.get("source_id", "")
            if source_id:
                results.append((source_id, 0.5))  # baseline score for text match

        return results

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """KNN vector search with group_id TAG filter, supplemented by text search."""
        if not self._embeddings_enabled:
            # Embeddings disabled — fall back to text-only search
            try:
                return await self._text_search(query, group_id=group_id, limit=limit)
            except Exception:
                return []

        try:
            query_vec = await self._provider.embed_query(query)
            if not query_vec:
                # Embedding failed — fall back to text search
                try:
                    return await self._text_search(
                        query, group_id=group_id, limit=limit
                    )
                except Exception:
                    return []
        except Exception as e:
            logger.warning("Failed to embed query: %s", e)
            # Embedding failed — fall back to text search
            try:
                return await self._text_search(query, group_id=group_id, limit=limit)
            except Exception:
                return []

        vec_bytes = pack_vector(query_vec)

        # Build filter
        filter_parts = []
        if group_id:
            # TAG values use {} syntax in Redis Search
            filter_parts.append(f"@group_id:{{{group_id}}}")
        if entity_types:
            joined = "|".join(entity_types)
            filter_parts.append(f"@entity_type:{{{joined}}}")
        filter_parts.append("@content_type:{entity}")
        filter_str = " ".join(filter_parts) if filter_parts else "*"

        knn_query = f"({filter_str})=>[KNN {limit} @embedding $blob AS score]"

        try:
            result = await self._redis.execute_command(
                "FT.SEARCH",
                self.INDEX_NAME,
                knn_query,
                "PARAMS",
                "2",
                "blob",
                vec_bytes,
                "SORTBY",
                "score",
                "LIMIT",
                "0",
                str(limit),
                "RETURN",
                "2",
                "source_id",
                "score",
                "DIALECT",
                "2",
            )
        except Exception as e:
            logger.warning("RedisSearchIndex search failed: %s", e)
            return []

        # Parse FT.SEARCH response:
        # [total_count, key1, [field, value, ...], key2, [field, value, ...], ...]
        if not result or result[0] == 0:
            return []

        results: list[tuple[str, float]] = []
        i = 1
        while i < len(result):
            # key
            i += 1
            if i >= len(result):
                break
            fields = result[i]
            i += 1

            # Parse field-value pairs
            field_dict: dict[str, str] = {}
            for j in range(0, len(fields), 2):
                k = fields[j].decode() if isinstance(fields[j], bytes) else fields[j]
                v = fields[j + 1].decode() if isinstance(fields[j + 1], bytes) else fields[j + 1]
                field_dict[k] = v

            source_id = field_dict.get("source_id", "")
            # Redis COSINE distance is [0, 2]. Convert to similarity [0, 1]
            distance = float(field_dict.get("score", "1.0"))
            similarity = 1.0 - (distance / 2.0)

            if source_id:
                results.append((source_id, similarity))

        # Supplement with text search results when KNN is sparse
        if len(results) < limit:
            try:
                text_results = await self._text_search(
                    query,
                    group_id=group_id,
                    limit=limit,
                )
                existing = {eid for eid, _ in results}
                for eid, score in text_results:
                    if eid not in existing:
                        results.append((eid, score))
                        existing.add(eid)
            except Exception:
                pass  # text search is best-effort

        return results

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]:
        """Compute similarity between query and stored entity embeddings via HNSW."""
        if not self._embeddings_enabled or not entity_ids:
            return {}
        try:
            query_vec = await self._provider.embed_query(query)
            if not query_vec:
                return {}
        except Exception:
            return {}

        results: dict[str, float] = {}
        gid = group_id or "default"

        for eid in entity_ids:
            key = self._hash_key(gid, "entity", eid)
            try:
                stored = await self._redis.hget(key, "embedding")
                if stored is None:
                    continue
                stored_vec = list(struct.unpack(f"<{len(stored) // 4}f", stored))
                # Compute cosine similarity
                dot = sum(a * b for a, b in zip(query_vec, stored_vec))
                norm_q = sum(a * a for a in query_vec) ** 0.5
                norm_s = sum(a * a for a in stored_vec) ** 0.5
                if norm_q > 0 and norm_s > 0:
                    results[eid] = max(0.0, dot / (norm_q * norm_s))
            except Exception:
                continue

        return results

    async def close(self) -> None:
        """No-op — Redis client lifecycle is managed externally."""
        pass

    async def remove(self, entity_id: str) -> None:
        """Remove all vector entries for an entity."""
        # Scan for matching keys and delete them
        pattern = f"engram:*:vec:entity:{entity_id}"
        async for key in self._redis.scan_iter(match=pattern, count=100):
            await self._redis.delete(key)

        # Also check episode keys
        pattern_ep = f"engram:*:vec:episode:{entity_id}"
        async for key in self._redis.scan_iter(match=pattern_ep, count=100):
            await self._redis.delete(key)
