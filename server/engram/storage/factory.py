"""Factory that builds the correct backend triple for the resolved mode."""

from __future__ import annotations

import logging
import os

from engram.config import EngramConfig
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.storage.resolver import EngineMode

logger = logging.getLogger(__name__)


def create_stores(
    mode: EngineMode, config: EngramConfig
) -> tuple[GraphStore, ActivationStore, SearchIndex]:
    """Create the storage backend triple based on mode."""
    if mode == EngineMode.LITE:
        from engram.embeddings.provider import NoopProvider, VoyageProvider
        from engram.storage.memory.activation import MemoryActivationStore
        from engram.storage.sqlite.graph import SQLiteGraphStore
        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        db_path = str(config.get_sqlite_path())

        encryptor = None
        if config.encryption.enabled and config.encryption.master_key:
            from engram.security.encryption import FieldEncryptor

            encryptor = FieldEncryptor(config.encryption.master_key)

        # Resolve embedding API key
        api_key = config.embedding.api_key or os.environ.get("VOYAGE_API_KEY", "")

        if api_key:
            provider = VoyageProvider(
                api_key=api_key,
                model=config.embedding.model,
                dimensions=config.embedding.dimensions,
                batch_size=config.embedding.batch_size,
            )
            logger.info("Embedding provider: VoyageProvider (%s)", config.embedding.model)
        else:
            provider = NoopProvider()
            logger.warning(
                "No VOYAGE_API_KEY found — vector search disabled. "
                "Set VOYAGE_API_KEY or ENGRAM_EMBEDDING__API_KEY to enable."
            )

        fts = FTS5SearchIndex(db_path)
        vectors = SQLiteVectorStore(db_path)

        search_index = HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=provider,
            fts_weight=config.embedding.fts_weight,
            vec_weight=config.embedding.vec_weight,
        )

        return (
            SQLiteGraphStore(db_path, encryptor=encryptor),
            MemoryActivationStore(cfg=config.activation),
            search_index,
        )
    else:  # EngineMode.FULL
        import redis.asyncio as aioredis

        from engram.embeddings.provider import NoopProvider, VoyageProvider
        from engram.storage.falkordb.graph import FalkorDBGraphStore
        from engram.storage.redis.activation import RedisActivationStore
        from engram.storage.vector.redis_search import RedisSearchIndex

        encryptor = None
        if config.encryption.enabled and config.encryption.master_key:
            from engram.security.encryption import FieldEncryptor

            encryptor = FieldEncryptor(config.encryption.master_key)

        # Shared Redis client (activation + vector search use same Redis instance)
        # decode_responses=False because embedding vectors are raw bytes
        redis_url = config.redis.url
        redis_client = aioredis.from_url(redis_url, decode_responses=False)

        # Embedding provider (same VoyageProvider/NoopProvider logic as lite mode)
        api_key = config.embedding.api_key or os.environ.get("VOYAGE_API_KEY", "")

        if api_key:
            provider = VoyageProvider(
                api_key=api_key,
                model=config.embedding.model,
                dimensions=config.embedding.dimensions,
                batch_size=config.embedding.batch_size,
            )
            logger.info("Embedding provider: VoyageProvider (%s)", config.embedding.model)
        else:
            provider = NoopProvider()
            logger.warning(
                "No VOYAGE_API_KEY found — vector search disabled. "
                "Set VOYAGE_API_KEY or ENGRAM_EMBEDDING__API_KEY to enable."
            )

        return (
            FalkorDBGraphStore(config.falkordb, encryptor=encryptor),
            RedisActivationStore(redis_client, cfg=config.activation),
            RedisSearchIndex(redis_client, provider=provider, config=config.embedding),
        )
