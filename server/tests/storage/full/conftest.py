"""Test fixtures for full mode storage tests (FalkorDB + Redis)."""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from engram.config import ActivationConfig, EmbeddingConfig, FalkorDBConfig

# Skip all tests in this module if Docker services are not available
pytestmark = pytest.mark.requires_docker


@pytest.fixture
def falkordb_config() -> FalkorDBConfig:
    """FalkorDB config with unique graph name per test run."""
    return FalkorDBConfig(
        host=os.environ.get("ENGRAM_FALKORDB__HOST", "localhost"),
        port=int(os.environ.get("ENGRAM_FALKORDB__PORT", "6380")),
        password=os.environ.get("ENGRAM_FALKORDB__PASSWORD", "engram_dev"),
        graph_name=f"engram_test_{uuid.uuid4().hex[:8]}",
    )


@pytest_asyncio.fixture
async def redis_client():
    """Async Redis client for tests (uses db=1 to isolate from production)."""
    try:
        import redis.asyncio as aioredis
    except ImportError:
        pytest.skip("redis package not installed (install with: uv pip install -e '.[full]')")

    url = os.environ.get("ENGRAM_REDIS__URL", "redis://:engram_dev@localhost:6381/1")
    client = aioredis.from_url(url, decode_responses=False)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available")
    yield client
    # Cleanup: flush test database
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def falkordb_graph_store(falkordb_config):
    """Initialized FalkorDB graph store with cleanup."""
    try:
        from engram.storage.falkordb.graph import FalkorDBGraphStore
    except ImportError:
        pytest.skip("falkordb package not installed (install with: uv pip install -e '.[full]')")

    store = FalkorDBGraphStore(falkordb_config)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("FalkorDB not available")
    yield store
    # Cleanup: delete all nodes and relationships in test graph
    try:
        await store._query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass
    await store.close()


@pytest_asyncio.fixture
async def redis_activation_store(redis_client):
    """Redis activation store with cleanup."""
    from engram.storage.redis.activation import RedisActivationStore

    store = RedisActivationStore(redis_client, cfg=ActivationConfig())
    yield store
    # Cleanup is handled by redis_client fixture (flushdb)


@pytest_asyncio.fixture
async def redis_search_index(redis_client):
    """Redis search index with NoopProvider."""
    from engram.embeddings.provider import NoopProvider
    from engram.storage.vector.redis_search import RedisSearchIndex

    provider = NoopProvider()
    config = EmbeddingConfig()
    index = RedisSearchIndex(redis_client, provider=provider, config=config)
    await index.initialize()
    yield index
