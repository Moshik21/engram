"""Tests for GNN (GraphSAGE) trainer and numpy inference."""

from __future__ import annotations

import numpy as np
import pytest

from engram.embeddings.graph.gnn_inference import GraphSAGEInference


class TestGraphSAGEInference:
    def test_basic_forward(self):
        """Forward pass should produce correct output shape."""
        input_dim = 8
        output_dim = 4
        n_nodes = 5

        weights = {
            "layer_0_self_weight": np.random.randn(input_dim, output_dim).astype(np.float32),
            "layer_0_neigh_weight": np.random.randn(input_dim, output_dim).astype(np.float32),
        }

        model = GraphSAGEInference(weights)
        assert model.num_layers == 1

        features = np.random.randn(n_nodes, input_dim).astype(np.float32)
        adj = {0: [1, 2], 1: [0], 2: [0, 3], 3: [2, 4], 4: [3]}

        output = model.forward(features, adj)
        assert output.shape == (n_nodes, output_dim)

    def test_two_layer(self):
        """Two-layer model should produce output from final layer dim."""
        weights = {
            "layer_0_self_weight": np.random.randn(8, 16).astype(np.float32),
            "layer_0_neigh_weight": np.random.randn(8, 16).astype(np.float32),
            "layer_1_self_weight": np.random.randn(16, 4).astype(np.float32),
            "layer_1_neigh_weight": np.random.randn(16, 4).astype(np.float32),
        }

        model = GraphSAGEInference(weights)
        assert model.num_layers == 2

        features = np.random.randn(3, 8).astype(np.float32)
        adj = {0: [1], 1: [0, 2], 2: [1]}

        output = model.forward(features, adj)
        assert output.shape == (3, 4)

    def test_isolated_nodes(self):
        """Isolated nodes should still get embeddings (via self-projection)."""
        weights = {
            "layer_0_self_weight": np.eye(4, dtype=np.float32),
            "layer_0_neigh_weight": np.eye(4, dtype=np.float32),
        }

        model = GraphSAGEInference(weights)
        features = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        adj = {0: [], 1: []}  # No neighbors

        output = model.forward(features, adj)
        assert output.shape == (2, 4)
        # Should be normalized self-projection
        assert np.linalg.norm(output[0]) > 0

    def test_output_normalized(self):
        """Output should be L2-normalized per layer."""
        weights = {
            "layer_0_self_weight": np.random.randn(4, 4).astype(np.float32),
            "layer_0_neigh_weight": np.random.randn(4, 4).astype(np.float32),
        }

        model = GraphSAGEInference(weights)
        features = np.random.randn(5, 4).astype(np.float32)
        adj = {0: [1, 2], 1: [0], 2: [0], 3: [4], 4: [3]}

        output = model.forward(features, adj)
        norms = np.linalg.norm(output, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), atol=1e-5)


class TestFeatureBuilder:
    @pytest.mark.asyncio
    async def test_build_feature_matrix(self):
        """Should build matrix from entity embeddings."""
        from engram.embeddings.graph.feature_builder import build_feature_matrix

        entity_ids = ["e1", "e2", "e3"]

        class MockSearchIndex:
            async def get_entity_embeddings(self, ids, group_id):
                return {
                    "e1": [1.0, 2.0, 3.0],
                    "e3": [4.0, 5.0, 6.0],
                    # e2 missing — should get zeros
                }

        matrix = await build_feature_matrix(entity_ids, MockSearchIndex(), "default")
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(matrix[1], [0.0, 0.0, 0.0])  # missing
        np.testing.assert_allclose(matrix[2], [4.0, 5.0, 6.0])

    @pytest.mark.asyncio
    async def test_empty_embeddings(self):
        """Should return 768-dim zeros when no embeddings found."""
        from engram.embeddings.graph.feature_builder import build_feature_matrix

        class MockSearchIndex:
            async def get_entity_embeddings(self, ids, group_id):
                return {}

        matrix = await build_feature_matrix(["e1"], MockSearchIndex(), "default")
        assert matrix.shape == (1, 768)


# GNN trainer tests require torch — mark them
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
class TestGNNTrainer:
    @pytest.mark.asyncio
    async def test_below_threshold(self):
        from engram.config import ActivationConfig
        from engram.embeddings.graph.gnn import GNNTrainer

        cfg = ActivationConfig(
            graph_embedding_gnn_enabled=True,
            graph_embedding_gnn_min_entities=500,
        )
        trainer = GNNTrainer(cfg)

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return [type("E", (), {"id": f"e{i}"})() for i in range(10)]

        result = await trainer.train(MockGraph(), "default")
        assert result == {}

    @pytest.mark.asyncio
    async def test_train_produces_embeddings(self):
        from engram.config import ActivationConfig
        from engram.embeddings.graph.gnn import GNNTrainer

        cfg = ActivationConfig(
            graph_embedding_gnn_enabled=True,
            graph_embedding_gnn_min_entities=50,
            graph_embedding_gnn_output_dim=16,
            graph_embedding_gnn_hidden_dim=32,
            graph_embedding_gnn_epochs=5,
        )
        trainer = GNNTrainer(cfg)

        n = 60
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        result = await trainer.train(MockGraph(), "default")
        assert len(result) == n
        assert len(result["e0"]) == 16

    def test_method_name(self):
        from engram.config import ActivationConfig
        from engram.embeddings.graph.gnn import GNNTrainer

        cfg = ActivationConfig()
        trainer = GNNTrainer(cfg)
        assert trainer.method_name() == "gnn"
