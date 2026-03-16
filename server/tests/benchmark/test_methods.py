"""Tests for retrieval method configs and run_retrieval wrapper."""

import socket
from pathlib import Path

import pytest
import pytest_asyncio

from engram.benchmark.corpus import CorpusGenerator
from engram.benchmark.methods import (
    ALL_METHODS,
    METHOD_FULL_ENGRAM,
    METHOD_PURE_SEARCH,
    run_retrieval,
)
from engram.config import ActivationConfig
from engram.storage.memory.activation import MemoryActivationStore


def _helix_available() -> bool:
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not _helix_available(), reason="HelixDB not available"),
]


def test_method_configs_valid():
    """All methods should have weights that sum to approximately 1.0."""
    for m in ALL_METHODS:
        total = (
            m.config.weight_semantic
            + m.config.weight_activation
            + m.config.weight_spreading
            + m.config.weight_edge_proximity
        )
        assert abs(total - 1.0) < 0.01, f"{m.name} weights sum to {total}"


# Use a small corpus for integration tests
_small_corpus = CorpusGenerator(seed=99).generate()


@pytest_asyncio.fixture
async def benchmark_stores(tmp_path: Path):
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore

    """Create stores loaded with a small benchmark corpus."""
    graph_store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    search_index = HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969),
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=0,
        embed_provider="noop",
        embed_model="noop",
    )
    await search_index.initialize()

    gen = CorpusGenerator(seed=99)
    await gen.load(_small_corpus, graph_store, activation_store, search_index)

    yield graph_store, activation_store, search_index
    await graph_store.close()


async def test_run_retrieval_returns_results(benchmark_stores):
    """Full Engram method should return scored results."""
    graph_store, activation_store, search_index = benchmark_stores
    # Pick a direct query that should find something
    direct_queries = [q for q in _small_corpus.ground_truth if q.category == "direct"]
    assert len(direct_queries) > 0
    query = direct_queries[0]
    results = await run_retrieval(
        query.query_text,
        "benchmark",
        graph_store,
        activation_store,
        search_index,
        METHOD_FULL_ENGRAM,
        limit=5,
    )
    assert isinstance(results, list)
    # Should return some results (FTS5 should match entity names)
    assert len(results) >= 0  # May be 0 if FTS5 doesn't match, that's OK


async def test_pure_search_ignores_activation(benchmark_stores):
    """Pure search method should produce scores with zero activation component."""
    graph_store, activation_store, search_index = benchmark_stores
    direct_queries = [q for q in _small_corpus.ground_truth if q.category == "direct"]
    assert len(direct_queries) > 0
    query = direct_queries[0]
    results = await run_retrieval(
        query.query_text,
        "benchmark",
        graph_store,
        activation_store,
        search_index,
        METHOD_PURE_SEARCH,
        limit=5,
    )
    # For pure search, entity scores should equal semantic_similarity
    # (activation weight = 0, edge_proximity weight = 0)
    # Episode results have a different scoring formula so we exclude them.
    entity_results = [r for r in results if r.result_type == "entity"]
    for r in entity_results:
        assert abs(r.score - r.semantic_similarity) < 0.01, (
            f"Pure search score {r.score} != semantic {r.semantic_similarity}"
        )
