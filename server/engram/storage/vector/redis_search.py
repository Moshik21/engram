"""Redis Search HNSW vector index for full mode."""

from __future__ import annotations

import logging
import struct

from engram.config import EmbeddingConfig
from engram.embeddings.provider import EmbeddingProvider, truncate_vectors
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.utils.attachments import get_first_image_attachment

logger = logging.getLogger(__name__)


def pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector into binary (4 bytes per float, little-endian)."""
    return struct.pack(f"<{len(vec)}f", *vec)


class RedisSearchIndex:
    """HNSW vector index using Redis Search (FT.CREATE / FT.SEARCH).

    Implements the SearchIndex protocol for full mode.
    """

    INDEX_NAME = "engram_vectors"
    KEY_PREFIX = "engram:"

    def __init__(
        self,
        redis,
        provider: EmbeddingProvider,
        config: EmbeddingConfig,
        storage_dim: int = 0,
        embed_provider: str = "",
        embed_model: str = "",
        *,
        index_name: str | None = None,
        key_prefix: str | None = None,
    ) -> None:
        self._redis = redis
        self._provider = provider
        self._config = config
        self._embeddings_enabled = provider.dimension() > 0
        self._storage_dim = storage_dim
        self._embed_provider = embed_provider
        self._embed_model = embed_model
        self._index_name = index_name or self.INDEX_NAME
        prefix = key_prefix or self.KEY_PREFIX
        self._key_prefix = prefix if prefix.endswith(":") else f"{prefix}:"

    async def initialize(self) -> None:
        """Create the FT.CREATE index idempotently."""
        if not self._embeddings_enabled:
            logger.info("RedisSearchIndex: embeddings disabled (NoopProvider), skipping index")
            return

        # Check if index already exists
        try:
            await self._redis.execute_command("FT.INFO", self._index_name)
            logger.info("RedisSearchIndex: index '%s' already exists", self._index_name)
            return
        except Exception:
            pass  # Index does not exist, create it

        dim = self._storage_dim if self._storage_dim > 0 else self._provider.dimension()
        m = self._config.hnsw_m
        ef_construction = self._config.hnsw_ef_construction

        try:
            await self._redis.execute_command(
                "FT.CREATE",
                self._index_name,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                self._key_prefix,
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
                self._index_name,
                dim,
                m,
                ef_construction,
            )
        except Exception as e:
            logger.warning("RedisSearchIndex: FT.CREATE failed: %s", e)

    def _hash_key(self, group_id: str, content_type: str, item_id: str) -> str:
        return f"{self._key_prefix}{group_id}:vec:{content_type}:{item_id}"

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
            if self._storage_dim > 0:
                embeddings = truncate_vectors(embeddings, self._storage_dim)

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
                    "embed_provider": self._embed_provider,
                    "embed_model": self._embed_model,
                },
            )
        except Exception as e:
            logger.warning("Failed to index entity %s: %s", entity.id, e)

    async def index_episode(self, episode: Episode) -> None:
        """Embed and index an episode.

        When the episode has image attachments and the provider supports
        multimodal embedding, the first image is embedded together with
        the text for a richer representation.
        """
        if not self._embeddings_enabled or not episode.content:
            return

        try:
            embedding: list[float] | None = None

            # Try multimodal embedding for episodes with image attachments
            if episode.attachments and hasattr(self._provider, "embed_multimodal"):
                image_data = get_first_image_attachment(episode.attachments)
                if image_data is not None:
                    image_bytes, image_mime = image_data
                    embedding = await self._provider.embed_multimodal(
                        text=episode.content,
                        image_bytes=image_bytes,
                        image_mime=image_mime,
                    )
                    if embedding and self._storage_dim > 0:
                        embedding = truncate_vectors([embedding], self._storage_dim)[0]

            # Fall back to text-only embedding
            if not embedding:
                embeddings = await self._provider.embed([episode.content])
                if not embeddings:
                    return
                embedding = embeddings[0]
                if self._storage_dim > 0:
                    embedding = truncate_vectors([embedding], self._storage_dim)[0]

            key = self._hash_key(episode.group_id, "episode", episode.id)
            vec_bytes = pack_vector(embedding)
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
                    "embed_provider": self._embed_provider,
                    "embed_model": self._embed_model,
                },
            )
        except Exception as e:
            logger.warning("Failed to index episode %s: %s", episode.id, e)

    async def index_episode_cue(self, cue: EpisodeCue) -> None:
        """Embed and index cue text."""
        key = self._hash_key(cue.group_id, "episode_cue", cue.episode_id)
        if not cue.cue_text:
            try:
                await self._redis.delete(key)
            except Exception as e:
                logger.warning("Failed to remove cue %s: %s", cue.episode_id, e)
            return
        if not self._embeddings_enabled:
            return

        try:
            embeddings = await self._provider.embed([cue.cue_text])
            if not embeddings:
                return
            if self._storage_dim > 0:
                embeddings = truncate_vectors(embeddings, self._storage_dim)

            vec_bytes = pack_vector(embeddings[0])
            created_ts = cue.created_at.timestamp() if cue.created_at else 0.0

            await self._redis.hset(
                key,
                mapping={
                    "group_id": cue.group_id,
                    "content_type": "episode_cue",
                    "source_id": cue.episode_id,
                    "text": cue.cue_text,
                    "entity_type": "",
                    "created_at": str(created_ts),
                    "embedding": vec_bytes,
                    "embed_provider": self._embed_provider,
                    "embed_model": self._embed_model,
                },
            )
        except Exception as e:
            logger.warning("Failed to index cue %s: %s", cue.episode_id, e)

    @staticmethod
    def _decode_value(value) -> str:
        if isinstance(value, bytes):
            return value.decode()
        return str(value)

    def _build_filter(
        self,
        *,
        content_type: str,
        group_id: str | None = None,
        entity_types: list[str] | None = None,
    ) -> str:
        parts = []
        if group_id:
            parts.append(f"@group_id:{{{group_id}}}")
        if entity_types:
            parts.append(f"@entity_type:{{{'|'.join(entity_types)}}}")
        parts.append(f"@content_type:{{{content_type}}}")
        return " ".join(parts) if parts else "*"

    def _parse_ft_rows(self, result) -> list[dict[str, str]]:
        if not result or result[0] == 0:
            return []

        rows: list[dict[str, str]] = []
        i = 1
        while i < len(result):
            i += 1  # skip key
            if i >= len(result):
                break
            fields = result[i]
            i += 1
            field_dict = {}
            for j in range(0, len(fields), 2):
                field_dict[self._decode_value(fields[j])] = self._decode_value(fields[j + 1])
            rows.append(field_dict)
        return rows

    async def _text_search(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
        content_type: str = "entity",
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Full-text search using the indexed TEXT field in Redis Search."""
        filter_parts = [
            self._build_filter(
                content_type=content_type,
                group_id=group_id,
                entity_types=entity_types,
            )
        ]

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
                self._index_name,
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

        # Parse results — source_id with a fixed baseline score
        results: list[tuple[str, float]] = []
        for field_dict in self._parse_ft_rows(result):
            source_id = field_dict.get("source_id", "")
            if source_id:
                results.append((source_id, 0.5))  # baseline score for text match

        return results

    async def _search_content(
        self,
        query: str,
        *,
        content_type: str,
        group_id: str | None = None,
        limit: int = 20,
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._embeddings_enabled:
            return await self._text_search(
                query,
                group_id=group_id,
                limit=limit,
                content_type=content_type,
                entity_types=entity_types,
            )

        try:
            query_vec = await self._provider.embed_query(query)
            if not query_vec:
                return await self._text_search(
                    query,
                    group_id=group_id,
                    limit=limit,
                    content_type=content_type,
                    entity_types=entity_types,
                )
        except Exception as e:
            logger.warning("Failed to embed %s query: %s", content_type, e)
            return await self._text_search(
                query,
                group_id=group_id,
                limit=limit,
                content_type=content_type,
                entity_types=entity_types,
            )

        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]
        vec_bytes = pack_vector(query_vec)
        filter_str = self._build_filter(
            content_type=content_type,
            group_id=group_id,
            entity_types=entity_types,
        )
        knn_query = f"({filter_str})=>[KNN {limit} @embedding $blob AS score]"

        try:
            result = await self._redis.execute_command(
                "FT.SEARCH",
                self._index_name,
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
            logger.warning("RedisSearchIndex %s search failed: %s", content_type, e)
            return await self._text_search(
                query,
                group_id=group_id,
                limit=limit,
                content_type=content_type,
                entity_types=entity_types,
            )

        results: list[tuple[str, float]] = []
        for field_dict in self._parse_ft_rows(result):
            source_id = field_dict.get("source_id", "")
            distance = float(field_dict.get("score", "1.0"))
            similarity = 1.0 - (distance / 2.0)
            if source_id:
                results.append((source_id, similarity))

        if len(results) >= limit:
            return results[:limit]

        try:
            text_results = await self._text_search(
                query,
                group_id=group_id,
                limit=limit,
                content_type=content_type,
                entity_types=entity_types,
            )
        except Exception:
            return results

        existing = {source_id for source_id, _score in results}
        for source_id, score in text_results:
            if source_id in existing:
                continue
            results.append((source_id, score))
            existing.add(source_id)
            if len(results) >= limit:
                break
        return results

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """KNN vector search with group_id TAG filter, supplemented by text search."""
        return await self._search_content(
            query,
            content_type="entity",
            group_id=group_id,
            limit=limit,
            entity_types=entity_types,
        )

    async def search_episodes(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search raw episode embeddings/text in Redis Search."""
        return await self._search_content(
            query,
            content_type="episode",
            group_id=group_id,
            limit=limit,
        )

    async def search_episode_cues(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search cue embeddings/text in Redis Search."""
        return await self._search_content(
            query,
            content_type="episode_cue",
            group_id=group_id,
            limit=limit,
        )

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

        # Truncate query vector to storage dimension
        if self._storage_dim > 0:
            query_vec = truncate_vectors([query_vec], self._storage_dim)[0]

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

    async def batch_index_entities(self, entities: list[Entity]) -> int:
        """Batch-embed and index multiple entities via Redis pipeline."""
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
        try:
            embeddings = await self._provider.embed(texts)
            if not embeddings:
                return 0
            if self._storage_dim > 0:
                embeddings = truncate_vectors(embeddings, self._storage_dim)
            pipe = self._redis.pipeline(transaction=False)
            for e, t, vec in zip(valid, texts, embeddings):
                key = self._hash_key(e.group_id, "entity", e.id)
                vec_bytes = pack_vector(vec)
                created_ts = e.created_at.timestamp() if e.created_at else 0.0
                pipe.hset(
                    key,
                    mapping={
                        "group_id": e.group_id,
                        "content_type": "entity",
                        "source_id": e.id,
                        "text": t,
                        "entity_type": e.entity_type or "",
                        "created_at": str(created_ts),
                        "embedding": vec_bytes,
                        "embed_provider": self._embed_provider,
                        "embed_model": self._embed_model,
                    },
                )
            await pipe.execute()
            return len(valid)
        except Exception as e:
            logger.warning("Batch index entities failed: %s", e)
            return 0

    async def close(self) -> None:
        """No-op — Redis client lifecycle is managed externally."""
        pass

    async def get_entity_embeddings(
        self,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Batch retrieve entity embeddings from Redis hashes."""
        if not self._embeddings_enabled or not entity_ids:
            return {}
        gid = group_id or "default"
        results: dict[str, list[float]] = {}
        pipe = self._redis.pipeline(transaction=False)
        keys = []
        for eid in entity_ids:
            key = self._hash_key(gid, "entity", eid)
            keys.append((eid, key))
            pipe.hget(key, "vector")
        raw_values = await pipe.execute()
        dim = self._config.dimensions
        for (eid, _), raw in zip(keys, raw_values):
            if raw:
                vec = list(struct.unpack(f"<{dim}f", raw))
                results[eid] = vec
        return results

    async def remove(self, entity_id: str) -> None:
        """Remove all vector entries for an entity."""
        # Scan for matching keys and delete them
        pattern = f"{self._key_prefix}*:vec:entity:{entity_id}"
        async for key in self._redis.scan_iter(match=pattern, count=100):
            await self._redis.delete(key)

        # Also check episode keys
        pattern_ep = f"{self._key_prefix}*:vec:episode:{entity_id}"
        async for key in self._redis.scan_iter(match=pattern_ep, count=100):
            await self._redis.delete(key)

    async def delete_group(self, group_id: str) -> None:
        """Remove all vector keys for a given *group_id*.

        Keys follow the pattern ``{prefix}{group_id}:vec:{type}:{id}``.
        We SCAN with that pattern and delete in batches of 200 to avoid
        blocking Redis for too long.
        """
        pattern = f"{self._key_prefix}{group_id}:vec:*"
        batch: list[bytes | str] = []
        try:
            async for key in self._redis.scan_iter(match=pattern, count=200):
                batch.append(key)
                if len(batch) >= 200:
                    await self._redis.delete(*batch)
                    batch = []
            if batch:
                await self._redis.delete(*batch)
        except Exception as exc:
            logger.warning(
                "RedisSearchIndex.delete_group(%s) failed: %s",
                group_id,
                exc,
            )
