"""Test fixtures for HelixDB storage tests.

All tests require a running HelixDB instance on localhost:6969.
They are marked with ``requires_helix`` and will be skipped when
HelixDB is not reachable.
"""

from __future__ import annotations

import socket
from uuid import uuid4

import pytest
import pytest_asyncio  # noqa: F401  -- needed for async fixture decorator

from engram.config import EmbeddingConfig, HelixDBConfig


def helix_available() -> bool:
    """Return True if a HelixDB instance is listening on the default port."""
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not helix_available(), reason="HelixDB not available on localhost:6969"),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def helix_config() -> HelixDBConfig:
    """HelixDB config pointing at the local instance."""
    return HelixDBConfig(
        host="localhost",
        port=6969,
        verbose=False,
    )


@pytest.fixture
def test_group_id() -> str:
    """Unique group ID per test to guarantee isolation."""
    return f"test_{uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def helix_graph_store(helix_config: HelixDBConfig, test_group_id: str):
    """Initialized HelixGraphStore with per-test cleanup."""
    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(helix_config)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")

    yield store

    # Cleanup: remove all data for this test's group
    try:
        await store.delete_group(test_group_id)
    except Exception:
        pass
    await store.close()


@pytest_asyncio.fixture
async def helix_search_index(helix_config: HelixDBConfig, test_group_id: str):
    """Initialized HelixSearchIndex with a noop embedding provider."""
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    provider = NoopProvider()
    embed_config = EmbeddingConfig()
    index = HelixSearchIndex(
        helix_config=helix_config,
        provider=provider,
        embed_config=embed_config,
        storage_dim=0,
        embed_provider="noop",
        embed_model="noop",
    )
    try:
        await index.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")

    yield index

    # Cleanup
    try:
        await index.delete_group(test_group_id)
    except Exception:
        pass
    await index.close()
