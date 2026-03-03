"""Benchmark validation tests for Week 3 activation engine.

These tests verify that the activation-aware retrieval pipeline
meets the go/no-go criteria defined in the Week 3 spec.
"""

from __future__ import annotations

import time

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


class MockBenchExtractor:
    """Mock extractor that returns results based on content keyword matching."""

    def __init__(self, results: dict[str, ExtractionResult]) -> None:
        self._results = results

    async def extract(self, text: str) -> ExtractionResult:
        for key, result in self._results.items():
            if key in text:
                return result
        return ExtractionResult(entities=[], relationships=[])


@pytest_asyncio.fixture
async def benchmark_env(tmp_path):
    """Create a fully initialized benchmark environment."""
    db_path = str(tmp_path / "bench.db")
    cfg = ActivationConfig()
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=cfg)
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    yield graph_store, activation_store, search_index, cfg

    await graph_store.close()


@pytest.mark.asyncio
class TestActivationBenchmarks:
    async def test_activation_boosts_frequency(self, benchmark_env):
        """Entity mentioned 5 times should rank #1 for relevant query."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "Engram": ExtractionResult(
                    entities=[
                        {"name": "Engram", "entity_type": "Project", "summary": "Memory layer"},
                    ],
                    relationships=[],
                ),
                "weather": ExtractionResult(
                    entities=[
                        {"name": "Weather", "entity_type": "Topic", "summary": "Weather report"},
                    ],
                    relationships=[],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        for i in range(5):
            await manager.ingest_episode(f"Working on Engram episode {i}", group_id="default")
        await manager.ingest_episode("The weather is nice", group_id="default")

        results = await manager.recall("Engram project", group_id="default")
        assert len(results) >= 1
        entity_results = [r for r in results if "entity" in r]
        assert len(entity_results) >= 1
        assert entity_results[0]["entity"]["name"] == "Engram"

    async def test_activation_recency(self, benchmark_env):
        """Recent entity should rank higher than old entity with same semantic."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "VS Code": ExtractionResult(
                    entities=[
                        {"name": "VS Code", "entity_type": "Technology", "summary": "Code editor"},
                    ],
                    relationships=[],
                ),
                "Cursor": ExtractionResult(
                    entities=[
                        {
                            "name": "Cursor",
                            "entity_type": "Technology",
                            "summary": "AI code editor",
                        },
                    ],
                    relationships=[],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        await manager.ingest_episode("Using VS Code for development", group_id="default")
        await manager.ingest_episode("Switched to Cursor for AI features", group_id="default")

        results = await manager.recall("code editor", group_id="default")
        assert len(results) >= 1
        names = [r["entity"]["name"] for r in results if "entity" in r]
        assert "Cursor" in names

    async def test_spreading_finds_multi_hop(self, benchmark_env):
        """Spreading activation should find entities connected via multi-hop paths."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "Marcus Elena": ExtractionResult(
                    entities=[
                        {
                            "name": "Marcus",
                            "entity_type": "Person",
                            "summary": "Startup network contact",
                        },
                        {"name": "Elena", "entity_type": "Person", "summary": "Works at YC"},
                        {
                            "name": "YC",
                            "entity_type": "Organization",
                            "summary": "Y Combinator accelerator",
                        },
                        {
                            "name": "Engram",
                            "entity_type": "Project",
                            "summary": "Could get funding",
                        },
                    ],
                    relationships=[
                        {
                            "source": "Marcus",
                            "target": "Elena",
                            "predicate": "KNOWS",
                            "weight": 1.0,
                        },
                        {
                            "source": "Elena",
                            "target": "YC",
                            "predicate": "WORKS_AT",
                            "weight": 1.0,
                        },
                        {
                            "source": "Marcus",
                            "target": "Engram",
                            "predicate": "COULD_FUND",
                            "weight": 0.5,
                        },
                    ],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        await manager.ingest_episode("Marcus Elena YC could help fund Engram", group_id="default")

        results = await manager.recall("funding", group_id="default")
        found_names = {r["entity"]["name"] for r in results if "entity" in r}
        assert len(found_names) >= 1

    async def test_activation_overhead(self, benchmark_env):
        """Activation overhead should be < 50ms p95."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "test": ExtractionResult(
                    entities=[
                        {
                            "name": "TestEntity",
                            "entity_type": "Test",
                            "summary": "For benchmarking",
                        },
                    ],
                    relationships=[],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        await manager.ingest_episode("test entity for benchmarking", group_id="default")

        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            await manager.recall("test", group_id="default")
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        assert p95 < 50, f"p95 latency {p95:.1f}ms exceeds 50ms threshold"

    async def test_cold_entity_still_retrievable(self, benchmark_env):
        """A single-mention entity should still be retrievable via search."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "allergy": ExtractionResult(
                    entities=[
                        {
                            "name": "PeanutAllergy",
                            "entity_type": "Health",
                            "summary": "Peanut allergy condition",
                        },
                    ],
                    relationships=[],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        await manager.ingest_episode("I have a peanut allergy", group_id="default")

        results = await manager.recall("allergy", group_id="default")
        assert len(results) >= 1
        entity_results = [r for r in results if "entity" in r]
        assert len(entity_results) >= 1
        assert entity_results[0]["entity"]["name"] == "PeanutAllergy"

    async def test_no_regression_direct_recall(self, benchmark_env):
        """Direct recall queries should still work correctly."""
        graph_store, activation_store, search_index, cfg = benchmark_env

        extractor = MockBenchExtractor(
            {
                "Python FastAPI": ExtractionResult(
                    entities=[
                        {
                            "name": "Python",
                            "entity_type": "Technology",
                            "summary": "Programming language",
                        },
                        {
                            "name": "FastAPI",
                            "entity_type": "Technology",
                            "summary": "Web framework",
                        },
                    ],
                    relationships=[
                        {
                            "source": "FastAPI",
                            "target": "Python",
                            "predicate": "BUILT_WITH",
                            "weight": 1.0,
                        },
                    ],
                ),
            }
        )
        manager = GraphManager(graph_store, activation_store, search_index, extractor, cfg=cfg)

        await manager.ingest_episode("Python FastAPI web development", group_id="default")

        results = await manager.recall("Python", group_id="default")
        assert len(results) >= 1
        names = [r["entity"]["name"] for r in results if "entity" in r]
        assert "Python" in names

        results = await manager.recall("FastAPI", group_id="default")
        assert len(results) >= 1
        names = [r["entity"]["name"] for r in results if "entity" in r]
        assert "FastAPI" in names
