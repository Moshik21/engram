"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio

from engram.config import ActivationConfig, EngramConfig
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.security.encryption import FieldEncryptor
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def config(tmp_path: Path) -> EngramConfig:
    return EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "test.db")},
    )


@pytest_asyncio.fixture
async def graph_store(tmp_path: Path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(str(tmp_path / "test.db"))
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def activation_store() -> MemoryActivationStore:
    return MemoryActivationStore(cfg=ActivationConfig())


@pytest_asyncio.fixture
async def search_index(graph_store: SQLiteGraphStore) -> FTS5SearchIndex:
    index = FTS5SearchIndex(graph_store._db_path)
    await index.initialize(db=graph_store._db)
    return index


class MockExtractor(EntityExtractor):
    """Extractor that returns canned results without calling Claude."""

    def __init__(self, result: ExtractionResult | None = None) -> None:
        self._result = result or ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


@pytest_asyncio.fixture
async def encrypted_graph_store(tmp_path: Path) -> SQLiteGraphStore:
    key = os.urandom(32).hex()
    encryptor = FieldEncryptor(key)
    store = SQLiteGraphStore(str(tmp_path / "encrypted_test.db"), encryptor=encryptor)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def graph_manager(
    graph_store: SQLiteGraphStore,
    activation_store: MemoryActivationStore,
    search_index: FTS5SearchIndex,
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
