"""Shared test fixtures.

All graph/search fixtures require a running HelixDB instance
and are guarded by the ``requires_helix`` marker.
"""

from __future__ import annotations

import asyncio
import os
import socket
from uuid import uuid4

import pytest
import pytest_asyncio

from engram.config import ActivationConfig, EmbeddingConfig, EngramConfig, HelixDBConfig
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.security.encryption import FieldEncryptor
from engram.storage.memory.activation import MemoryActivationStore


def _helix_available() -> bool:
    """Return True if a HelixDB instance is listening on the default port."""
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config() -> EngramConfig:
    return EngramConfig(
        helix=HelixDBConfig(host="localhost", port=6969),
        _env_file=None,
    )


@pytest.fixture
def helix_config() -> HelixDBConfig:
    return HelixDBConfig(host="localhost", port=6969)


@pytest.fixture
def test_group_id() -> str:
    """Unique group ID per test to guarantee isolation."""
    return f"test_{uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def graph_store(helix_config):
    if not _helix_available():
        pytest.skip("HelixDB not available on localhost:6969")

    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(helix_config)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def activation_store() -> MemoryActivationStore:
    return MemoryActivationStore(cfg=ActivationConfig())


@pytest_asyncio.fixture
async def search_index(helix_config):
    if not _helix_available():
        pytest.skip("HelixDB not available on localhost:6969")

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
    await index.close()


class MockExtractor(EntityExtractor):
    """Extractor that returns canned results without calling Claude."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self._result = result or ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


@pytest_asyncio.fixture
async def encrypted_graph_store(helix_config):
    if not _helix_available():
        pytest.skip("HelixDB not available on localhost:6969")

    from engram.storage.helix.graph import HelixGraphStore

    key = os.urandom(32).hex()
    encryptor = FieldEncryptor(key)
    store = HelixGraphStore(helix_config, encryptor=encryptor)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def graph_manager(
    graph_store,
    activation_store: MemoryActivationStore,
    search_index,
) -> GraphManager:
    extractor = MockExtractor(
        ExtractionResult(
            entities=[
                {"name": "Python", "entity_type": "Technology", "summary": "Programming language"},
                {"name": "FastAPI", "entity_type": "Technology", "summary": "Web framework"},
            ],
            relationships=[
                {
                    "source": "FastAPI",
                    "target": "Python",
                    "predicate": "BUILT_WITH",
                    "weight": 1.0,
                },
            ],
        )
    )
    return GraphManager(graph_store, activation_store, search_index, extractor)
