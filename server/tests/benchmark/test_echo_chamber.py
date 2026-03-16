"""Tests for echo chamber benchmark."""

from __future__ import annotations

import socket

import pytest

from engram.benchmark.metrics import gini_coefficient


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


class TestGiniCoefficient:
    def test_uniform_is_zero(self):
        """All equal values → Gini = 0."""
        assert gini_coefficient([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_extreme_inequality(self):
        """One value dominates → Gini near 1."""
        values = [0.0] * 99 + [100.0]
        gini = gini_coefficient(values)
        assert gini > 0.95

    def test_empty_is_zero(self):
        """Empty list → Gini = 0."""
        assert gini_coefficient([]) == 0.0

    def test_all_zeros(self):
        """All zeros → Gini = 0."""
        assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0

    def test_moderate_inequality(self):
        """Moderate spread → Gini between 0 and 1."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        gini = gini_coefficient(values)
        assert 0.0 < gini < 1.0


class TestEchoChamberSmoke:
    @pytest.mark.asyncio
    async def test_smoke_20_queries(self, tmp_path):
        from engram.config import EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex

        """Run a small echo chamber and verify it produces valid results."""
        from engram.benchmark.corpus import generate_corpus
        from engram.benchmark.echo_chamber import run_echo_chamber
        from engram.config import ActivationConfig
        from engram.storage.memory.activation import MemoryActivationStore
        graph_store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph_store.initialize()
        search_index = HelixSearchIndex(
     helix_config=HelixDBConfig(host="localhost", port=6969),
     provider=NoopProvider(),
     embed_config=EmbeddingConfig(),
     storage_dim=0,
     embed_provider="noop",
     embed_model="noop",
 )
        await search_index.initialize()
        cfg = ActivationConfig()
        activation_store = MemoryActivationStore(cfg)

        corpus = generate_corpus(seed=42)
        for entity in corpus.entities:
            await graph_store.create_entity(entity)
            await search_index.index_entity(entity)
        for rel in corpus.relationships:
            await graph_store.create_relationship(rel)

        # Build query pools from ground truth
        hot = [entity.name for entity in corpus.entities[:3]]
        diverse = [entity.name for entity in corpus.entities[3:6]]
        entity_ids = [e.id for e in corpus.entities]

        result = await run_echo_chamber(
            hot_queries=hot,
            diverse_queries=diverse,
            corpus_entity_ids=entity_ids,
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            total_queries=20,
            snapshot_interval=10,
        )

        assert result.total_queries == 20
        assert len(result.snapshots) >= 2  # at q10 and q20
        assert 0.0 <= result.final_coverage <= 1.0
        assert 0.0 <= result.final_gini <= 1.0
        assert result.final_surfaced_count >= result.final_used_count
        # Verify snapshot structure
        for s in result.snapshots:
            assert s.query_index > 0
            assert len(s.top10_ids) <= 10
            assert 0.0 <= s.top10_jaccard <= 1.0
            assert s.surfaced_count >= s.used_count
            assert s.surfaced_to_used_ratio >= 0.0

        await graph_store.close()

    @pytest.mark.asyncio
    async def test_usage_policy_tracks_surfaced_vs_used(self, tmp_path):
        from engram.config import EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex

        """Custom usage policies should reduce reinforcement below surfaced volume."""
        from engram.benchmark.corpus import generate_corpus
        from engram.benchmark.echo_chamber import run_echo_chamber
        from engram.config import ActivationConfig
        from engram.storage.memory.activation import MemoryActivationStore
        graph_store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph_store.initialize()
        search_index = HelixSearchIndex(
     helix_config=HelixDBConfig(host="localhost", port=6969),
     provider=NoopProvider(),
     embed_config=EmbeddingConfig(),
     storage_dim=0,
     embed_provider="noop",
     embed_model="noop",
 )
        await search_index.initialize()
        cfg = ActivationConfig()
        activation_store = MemoryActivationStore(cfg)

        corpus = generate_corpus(seed=7)
        for entity in corpus.entities:
            await graph_store.create_entity(entity)
            await search_index.index_entity(entity)
        for rel in corpus.relationships:
            await graph_store.create_relationship(rel)

        hot = [entity.name for entity in corpus.entities[:3]]
        diverse = [entity.name for entity in corpus.entities[3:6]]
        entity_ids = [e.id for e in corpus.entities]

        def usage_policy(_query: str, results: list) -> set[str]:
            for result in results:
                if result.result_type == "entity":
                    return {result.node_id}
            return set()

        result = await run_echo_chamber(
            hot_queries=hot,
            diverse_queries=diverse,
            corpus_entity_ids=entity_ids,
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            group_id="benchmark",
            total_queries=20,
            snapshot_interval=10,
            usage_policy=usage_policy,
        )

        assert result.final_surfaced_count > 0
        assert result.final_used_count > 0
        assert result.final_surfaced_count >= result.final_used_count
        assert result.final_surfaced_to_used_ratio >= 1.0
        assert result.snapshots[-1].surfaced_count == result.final_surfaced_count
        assert result.snapshots[-1].used_count == result.final_used_count

        await graph_store.close()
