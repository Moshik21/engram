"""Shared test fixtures.

All graph/search fixtures require a running HelixDB instance
and are guarded by the ``requires_helix`` marker.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

# Tests exercise the full MCP registry (operator + eval tools). Product installs
# default to ENGRAM_MCP_SURFACE=public (golden loop only).
os.environ.setdefault("ENGRAM_MCP_SURFACE", "full")

import pytest
import pytest_asyncio

from engram.config import ActivationConfig, EmbeddingConfig, EngramConfig, HelixDBConfig
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.security.encryption import FieldEncryptor
from engram.storage.helix.availability import (
    helix_available as _helix_probe,
)
from engram.storage.helix.availability import (
    helix_native_available,
)
from engram.storage.memory.activation import MemoryActivationStore


def _helix_available() -> bool:
    """True when native PyO3 (preferred) or HTTP Helix is usable.

    Product path is native data-dir; HTTP :6969 remains a secondary
    compatibility probe for Docker/HTTP integration tests.
    """
    return bool(_helix_probe(prefer_native=True).get("available"))


def _test_helix_config(tmp_path: Path | None = None) -> HelixDBConfig:
    """Prefer native data-dir fixtures; fall back to HTTP :6969.

    Never point at the live product data-dir — tests use a disposable path.
    Pass ``tmp_path`` for per-test isolation: Helix BM25 doc IDs are
    content-derived, so re-creating the same content on a shared dir raises
    "document already exists" (previously swallowed by the transport, which
    let cross-test contamination pass silently).
    """
    if helix_native_available():
        data_dir = os.environ.get("ENGRAM_TEST_HELIX_DATA_DIR")
        if not data_dir and tmp_path is not None:
            data_dir = str(tmp_path / "helix-native")
        if not data_dir:
            data_dir = str(Path(os.environ.get("TMPDIR", "/tmp")) / "engram-helix-pytest-native")
        os.makedirs(data_dir, exist_ok=True)
        return HelixDBConfig(transport="native", data_dir=data_dir, verbose=False)
    return HelixDBConfig(host="localhost", port=6969, transport="http")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config(tmp_path: Path) -> EngramConfig:
    return EngramConfig(
        helix=_test_helix_config(tmp_path),
        _env_file=None,
    )


@pytest.fixture
def helix_config(tmp_path: Path) -> HelixDBConfig:
    return _test_helix_config(tmp_path)


@pytest.fixture
def test_group_id() -> str:
    """Unique group ID per test to guarantee isolation."""
    return f"test_{uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def graph_store(helix_config):
    if not _helix_available():
        pytest.skip("Helix not available (native data-dir or HTTP :6969)")

    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(helix_config)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("Helix not available or helix package not installed")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def activation_store() -> MemoryActivationStore:
    return MemoryActivationStore(cfg=ActivationConfig())


@pytest_asyncio.fixture
async def search_index(helix_config):
    if not _helix_available():
        pytest.skip("Helix not available (native data-dir or HTTP :6969)")

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
        pytest.skip("Helix not available or helix package not installed")
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
        pytest.skip("Helix not available (native data-dir or HTTP :6969)")

    from engram.storage.helix.graph import HelixGraphStore

    key = os.urandom(32).hex()
    encryptor = FieldEncryptor(key)
    store = HelixGraphStore(helix_config, encryptor=encryptor)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("Helix not available or helix package not installed")
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
