"""Factory that builds the correct backend triple for the resolved mode."""

from __future__ import annotations

import logging
import os
from typing import cast

from engram.config import EngramConfig
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.storage.resolver import EngineMode

logger = logging.getLogger(__name__)


def _create_embedding_provider(config: EngramConfig):
    """Resolve embedding provider from config with automatic fallback.

    Resolution order for ``"auto"`` (default):
    1. Voyage — if VOYAGE_API_KEY is set
    2. Local (FastEmbed) — always available (default dependency)
    3. Noop — vector search disabled, FTS5 still works
    """
    from engram.embeddings.provider import NoopProvider, VoyageProvider

    provider_type = config.embedding.provider.lower()

    # Normalize legacy "voyage" default to "auto" behavior
    if provider_type in ("voyage", "auto"):
        api_key = config.embedding.api_key or os.environ.get("VOYAGE_API_KEY", "")
        if api_key:
            logger.info("Embedding provider: VoyageProvider (%s)", config.embedding.model)
            return VoyageProvider(
                api_key=api_key,
                model=config.embedding.model,
                dimensions=config.embedding.dimensions,
                batch_size=config.embedding.batch_size,
            )
        if provider_type == "voyage":
            # Explicit "voyage" without key — warn and fall back
            logger.warning("No VOYAGE_API_KEY — falling back to local embeddings")
        provider_type = "local"

    if provider_type == "local":
        try:
            from engram.embeddings.provider import FastEmbedProvider

            provider = FastEmbedProvider(model=config.embedding.local_model)
            # Override dimensions so downstream (e.g. Redis HNSW) uses correct value
            config.embedding.dimensions = provider.dimension()
            logger.info(
                "Embedding provider: FastEmbedProvider (%s, %dd)",
                config.embedding.local_model,
                provider.dimension(),
            )
            return provider
        except ImportError:
            logger.warning(
                "fastembed not installed — vector search disabled. "
                "Install with: pip install fastembed"
            )

    # noop or fallback
    logger.info("Embedding provider: NoopProvider (vector search disabled)")
    return NoopProvider()


def create_stores(
    mode: EngineMode, config: EngramConfig
) -> tuple[GraphStore, ActivationStore, SearchIndex]:
    """Create the storage backend triple based on mode."""
    if mode == EngineMode.LITE:
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

        provider = _create_embedding_provider(config)

        # Compute effective storage dimension
        storage_dim = config.embedding.storage_dimensions
        if storage_dim > 0 and storage_dim >= provider.dimension():
            storage_dim = 0  # no truncation if storage >= native
        if storage_dim > 0 and config.embedding.provider == "voyage":
            logger.warning(
                "Matryoshka truncation (storage_dimensions=%d) is not supported "
                "by Voyage models. Only Nomic Embed v1.5 supports this. "
                "Set ENGRAM_EMBEDDING__STORAGE_DIMENSIONS=0 or use provider=local.",
                storage_dim,
            )

        # Embedding metadata for versioning
        provider_type = config.embedding.provider.lower()
        embed_meta_provider = provider_type
        embed_meta_model = (
            config.embedding.model
            if provider_type == "voyage"
            else config.embedding.local_model
            if provider_type == "local"
            else "noop"
        )

        fts = FTS5SearchIndex(db_path)
        vectors = SQLiteVectorStore(db_path)

        search_index = HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=provider,
            fts_weight=config.embedding.fts_weight,
            vec_weight=config.embedding.vec_weight,
            storage_dim=storage_dim,
            embed_provider=embed_meta_provider,
            embed_model=embed_meta_model,
        )

        return (
            SQLiteGraphStore(db_path, encryptor=encryptor),
            MemoryActivationStore(cfg=config.activation),
            search_index,
        )
    else:  # EngineMode.FULL
        import redis.asyncio as aioredis

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
        redis_kwargs: dict = {"decode_responses": False}
        if config.redis.ssl_cert_reqs:
            import ssl as _ssl

            reqs_map = {
                "required": _ssl.CERT_REQUIRED,
                "optional": _ssl.CERT_OPTIONAL,
                "none": _ssl.CERT_NONE,
            }
            redis_kwargs["ssl_cert_reqs"] = reqs_map.get(
                config.redis.ssl_cert_reqs,
                _ssl.CERT_REQUIRED,
            )
        redis_client = aioredis.from_url(redis_url, **redis_kwargs)

        provider = _create_embedding_provider(config)

        # Compute effective storage dimension
        storage_dim = config.embedding.storage_dimensions
        if storage_dim > 0 and storage_dim >= provider.dimension():
            storage_dim = 0
        if storage_dim > 0 and config.embedding.provider == "voyage":
            logger.warning(
                "Matryoshka truncation (storage_dimensions=%d) is not supported "
                "by Voyage models. Use provider=local for Matryoshka support.",
                storage_dim,
            )

        provider_type = config.embedding.provider.lower()
        embed_meta_provider = provider_type
        embed_meta_model = (
            config.embedding.model
            if provider_type == "voyage"
            else config.embedding.local_model
            if provider_type == "local"
            else "noop"
        )

        return cast(
            tuple[GraphStore, ActivationStore, SearchIndex],
            (
                FalkorDBGraphStore(config.falkordb, encryptor=encryptor),
                RedisActivationStore(redis_client, cfg=config.activation),
                RedisSearchIndex(
                    redis_client,
                    provider=provider,
                    config=config.embedding,
                    storage_dim=storage_dim,
                    embed_provider=embed_meta_provider,
                    embed_model=embed_meta_model,
                ),
            ),
        )
