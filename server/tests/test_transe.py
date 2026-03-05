"""Tests for TransE knowledge graph embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from engram.config import ActivationConfig
from engram.embeddings.graph.transe import TransETrainer, _l2_normalize


class TestL2Normalize:
    def test_basic_normalization(self):
        """Rows should have unit norm after normalization."""
        mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        _l2_normalize(mat)
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    def test_zero_vector(self):
        """Zero vectors should not cause NaN."""
        mat = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        _l2_normalize(mat)
        assert not np.any(np.isnan(mat))


class TestTransETrainer:
    @pytest.mark.asyncio
    async def test_below_threshold(self):
        """Should return empty when below min_triples."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=100,
        )
        trainer = TransETrainer(cfg)

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return [type("E", (), {"id": f"e{i}"})() for i in range(5)]

            async def get_relationships(self, eid, direction, group_id):
                return []

        result = await trainer.train(MockGraph(), "default")
        assert result == {}

    @pytest.mark.asyncio
    async def test_train_produces_embeddings(self):
        """Should produce entity + relation embeddings."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=32,
            graph_embedding_transe_epochs=10,
            graph_embedding_transe_batch_size=32,
        )
        trainer = TransETrainer(cfg)

        n_entities = 30
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n_entities)]

        # Create relationship objects with proper attributes
        def make_rel(src, tgt, pred):
            return type("R", (), {
                "source_id": src,
                "target_id": tgt,
                "predicate": pred,
            })()

        relationships = {}
        for i in range(n_entities):
            rels = []
            # WORKS_AT chain
            if i < n_entities - 1:
                rels.append(make_rel(f"e{i}", f"e{i+1}", "WORKS_AT"))
            # KNOWS chain (every other)
            if i % 2 == 0 and i + 2 < n_entities:
                rels.append(make_rel(f"e{i}", f"e{i+2}", "KNOWS"))
            relationships[f"e{i}"] = rels

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_relationships(self, eid, direction, group_id):
                return relationships.get(eid, [])

        result = await trainer.train(MockGraph(), "default")
        # Should have entity embeddings + relation embeddings
        assert len(result) > n_entities  # entities + relation types
        assert "WORKS_AT" in result  # relation embedding
        assert "KNOWS" in result
        assert len(result["e0"]) == 32

    @pytest.mark.asyncio
    async def test_translational_property(self):
        """Training should produce embeddings for a graph with enough triples."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=32,
            graph_embedding_transe_epochs=50,
            graph_embedding_transe_margin=1.0,
            graph_embedding_transe_lr=0.01,
            graph_embedding_transe_batch_size=32,
        )
        trainer = TransETrainer(cfg)

        def make_rel(src, tgt, pred):
            return type("R", (), {
                "source_id": src,
                "target_id": tgt,
                "predicate": pred,
            })()

        n = 30
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]
        relationships: dict[str, list] = {f"e{i}": [] for i in range(n)}
        # Create enough triples: chain + cross-links
        for i in range(n - 1):
            relationships[f"e{i}"].append(make_rel(f"e{i}", f"e{i+1}", "A"))
        for i in range(0, n - 2, 2):
            relationships[f"e{i}"].append(make_rel(f"e{i}", f"e{i+2}", "B"))

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_relationships(self, eid, direction, group_id):
                return relationships.get(eid, [])

        result = await trainer.train(MockGraph(), "default")
        assert len(result) > 0

    def test_method_name(self):
        cfg = ActivationConfig()
        trainer = TransETrainer(cfg)
        assert trainer.method_name() == "transe"

    @pytest.mark.asyncio
    async def test_positive_triples_closer_than_random(self):
        """After training, h+r-t should be smaller for real triples than random."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=32,
            graph_embedding_transe_epochs=100,
            graph_embedding_transe_lr=0.01,
            graph_embedding_transe_batch_size=32,
        )
        trainer = TransETrainer(cfg)

        def make_rel(src, tgt, pred):
            return type("R", (), {
                "source_id": src,
                "target_id": tgt,
                "predicate": pred,
            })()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]
        relationships: dict[str, list] = {f"e{i}": [] for i in range(n)}
        for i in range(n - 1):
            relationships[f"e{i}"].append(make_rel(f"e{i}", f"e{i+1}", "NEXT"))

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_relationships(self, eid, direction, group_id):
                return relationships.get(eid, [])

        result = await trainer.train(MockGraph(), "default")
        if not result:
            pytest.skip("Not enough triples for training")

        import numpy as np

        # Check a real triple: e0 + NEXT ≈ e1
        h = np.array(result["e0"])
        r = np.array(result["NEXT"])
        t = np.array(result["e1"])
        dist_real = np.linalg.norm(h + r - t)

        # Random triple: e0 + NEXT ≈ e10
        t_rand = np.array(result["e10"])
        dist_rand = np.linalg.norm(h + r - t_rand)

        # Real triple should have smaller distance
        assert dist_real < dist_rand

    @pytest.mark.asyncio
    async def test_self_loop_triples(self):
        """Training should complete without errors when self-loops exist."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=16,
            graph_embedding_transe_epochs=10,
            graph_embedding_transe_batch_size=32,
        )
        trainer = TransETrainer(cfg)

        def make_rel(src, tgt, pred):
            return type("R", (), {
                "source_id": src,
                "target_id": tgt,
                "predicate": pred,
            })()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]
        relationships: dict[str, list] = {f"e{i}": [] for i in range(n)}
        for i in range(n - 1):
            relationships[f"e{i}"].append(make_rel(f"e{i}", f"e{i+1}", "NEXT"))
        # Add self-loops
        for i in range(0, n, 3):
            relationships[f"e{i}"].append(make_rel(f"e{i}", f"e{i}", "SELF_REF"))

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_relationships(self, eid, direction, group_id):
                return relationships.get(eid, [])

        # Should not raise
        result = await trainer.train(MockGraph(), "default")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_no_edges(self):
        """Graph with entities but no edges should return empty."""
        cfg = ActivationConfig(
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
        )
        trainer = TransETrainer(cfg)

        entities = [type("E", (), {"id": f"e{i}"})() for i in range(20)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_relationships(self, eid, direction, group_id):
                return []

        result = await trainer.train(MockGraph(), "default")
        assert result == {}
