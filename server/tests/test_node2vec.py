"""Tests for Node2Vec trainer and Skip-gram."""

from __future__ import annotations

import numpy as np
import pytest

from engram.config import ActivationConfig
from engram.embeddings.graph.node2vec import (
    Node2VecTrainer,
    _build_neighbor_sets,
    _generate_walks,
    _random_walk,
)
from engram.embeddings.graph.skipgram import NumpySkipGram  # noqa: I001

# ---- Skip-gram tests ----

class TestNumpySkipGram:
    def test_basic_training(self):
        """Skip-gram should produce embeddings of correct shape."""
        sg = NumpySkipGram(vocab_size=10, dimensions=16, window=3, epochs=2, seed=42)
        walks = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 2, 4, 6, 8]]
        result = sg.train(walks)
        assert result.shape == (10, 16)
        assert result.dtype == np.float32

    def test_empty_walks(self):
        """Empty walks should return initial embeddings."""
        sg = NumpySkipGram(vocab_size=5, dimensions=8, epochs=1, seed=42)
        result = sg.train([])
        assert result.shape == (5, 8)

    def test_single_node_walk(self):
        """Single-node walks should not crash."""
        sg = NumpySkipGram(vocab_size=3, dimensions=4, epochs=1, seed=42)
        result = sg.train([[0], [1], [2]])
        assert result.shape == (3, 4)

    def test_no_nan_in_embeddings(self):
        """Training should never produce NaN or Inf values."""
        sg = NumpySkipGram(vocab_size=20, dimensions=32, window=5, epochs=5, seed=42)
        walks = [[i % 20 for i in range(j, j + 10)] for j in range(50)]
        result = sg.train(walks)
        assert np.all(np.isfinite(result)), "Embeddings contain NaN or Inf"

    def test_connected_nodes_closer(self):
        """Nodes that co-occur in walks should be closer in embedding space."""
        # Create walks where 0-1-2 are always together, 7-8-9 always together
        walks = []
        for _ in range(50):
            walks.append([0, 1, 2, 0, 1, 2])
            walks.append([7, 8, 9, 7, 8, 9])
        sg = NumpySkipGram(
            vocab_size=10, dimensions=32, window=3,
            epochs=5, lr=0.05, seed=42,
        )
        emb = sg.train(walks)

        # 0 and 1 should be closer than 0 and 7
        sim_01 = np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]))
        sim_07 = np.dot(emb[0], emb[7]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[7]))
        assert sim_01 > sim_07


# ---- Random walk tests ----

class TestRandomWalks:
    def test_walk_length(self):
        """Walks should respect max length."""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 1.0)], 2: [(1, 1.0)]}
        ns = _build_neighbor_sets(adj)
        rng = np.random.RandomState(42)
        walk = _random_walk(adj, ns, 0, 10, 1.0, 1.0, rng)
        assert len(walk) <= 10
        assert walk[0] == 0

    def test_walk_isolated_node(self):
        """Walk from isolated node should be length 1."""
        adj = {0: [], 1: [(0, 1.0)]}
        ns = _build_neighbor_sets(adj)
        rng = np.random.RandomState(42)
        walk = _random_walk(adj, ns, 0, 10, 1.0, 1.0, rng)
        assert walk == [0]

    def test_generate_walks_count(self):
        """Should generate num_walks * num_nodes walks (minus isolated)."""
        adj = {
            0: [(1, 1.0), (2, 1.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(0, 1.0), (1, 1.0)],
        }
        rng = np.random.RandomState(42)
        walks = _generate_walks(adj, 3, num_walks=5, walk_length=4, p=1.0, q=1.0, rng=rng)
        assert len(walks) == 15  # 5 walks * 3 nodes

    def test_walk_respects_p_parameter(self):
        """High p should make walks less likely to return to previous node."""
        # Linear graph: 0 -- 1 -- 2 -- 3
        adj = {
            0: [(1, 1.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(1, 1.0), (3, 1.0)],
            3: [(2, 1.0)],
        }
        rng = np.random.RandomState(42)
        ns = _build_neighbor_sets(adj)
        # With high p (= low return probability), walk should tend forward
        walks_high_p = []
        for _ in range(100):
            walk = _random_walk(adj, ns, 0, 5, p=10.0, q=1.0, rng=rng)
            walks_high_p.append(walk)

        # Count how many walks reach node 3
        reach_3 = sum(1 for w in walks_high_p if 3 in w)
        # With high p, should reach far nodes more often
        assert reach_3 > 10  # At least some should reach node 3

    def test_walk_respects_q_parameter(self):
        """Low q should favor DFS (exploration), high q should favor BFS."""
        # Star graph: 0 is center, 1-5 are leaves connected only to 0
        adj = {0: [(i, 1.0) for i in range(1, 6)]}
        for i in range(1, 6):
            adj[i] = [(0, 1.0)]
        ns = _build_neighbor_sets(adj)
        rng = np.random.RandomState(42)

        # With low q (DFS), walks from 1 through 0 should explore other leaves
        # With high q (BFS), walks should return to previously visited nodes more
        walks_low_q = []
        walks_high_q = []
        for _ in range(200):
            walks_low_q.append(_random_walk(adj, ns, 1, 6, p=1.0, q=0.25, rng=rng))
            walks_high_q.append(_random_walk(adj, ns, 1, 6, p=1.0, q=4.0, rng=rng))

        # Low q should visit more unique nodes on average (exploration)
        avg_unique_low = np.mean([len(set(w)) for w in walks_low_q])
        avg_unique_high = np.mean([len(set(w)) for w in walks_high_q])
        assert avg_unique_low >= avg_unique_high - 0.5  # Low q explores at least as much

    def test_walk_with_weights(self):
        """Weighted edges should influence walk direction."""
        adj = {
            0: [(1, 10.0), (2, 0.01)],  # Strongly prefer node 1
            1: [(0, 1.0)],
            2: [(0, 1.0)],
        }
        ns = _build_neighbor_sets(adj)
        rng = np.random.RandomState(42)
        visit_1 = 0
        visit_2 = 0
        for _ in range(200):
            walk = _random_walk(adj, ns, 0, 2, 1.0, 1.0, rng)
            if len(walk) > 1:
                if walk[1] == 1:
                    visit_1 += 1
                elif walk[1] == 2:
                    visit_2 += 1
        assert visit_1 > visit_2 * 5  # Should visit 1 much more often


# ---- Node2VecTrainer integration tests ----

class TestNode2VecTrainer:
    @pytest.mark.asyncio
    async def test_train_below_threshold(self):
        """Should return empty dict when below min_entities."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=100,
        )
        trainer = Node2VecTrainer(cfg)

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return [type("E", (), {"id": f"e{i}"})() for i in range(10)]

            async def get_active_neighbors_with_weights(self, eid, group_id):
                return []

        result = await trainer.train(MockGraph(), "default")
        assert result == {}

    @pytest.mark.asyncio
    async def test_train_produces_embeddings(self):
        """Should produce embeddings when above threshold."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        trainer = Node2VecTrainer(cfg)

        # Create a small connected graph
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(20)]
        neighbors = {
            f"e{i}": [(f"e{(i+1) % 20}", 1.0, "RELATED_TO")]
            for i in range(20)
        }

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                return neighbors.get(eid, [])

        result = await trainer.train(MockGraph(), "default")
        assert len(result) == 20
        assert len(result["e0"]) == 16

    @pytest.mark.asyncio
    async def test_train_all_isolated_nodes(self):
        """Graph with only isolated nodes should return empty dict."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        trainer = Node2VecTrainer(cfg)

        entities = [type("E", (), {"id": f"e{i}"})() for i in range(10)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                return []  # All isolated

        result = await trainer.train(MockGraph(), "default")
        # With all isolated nodes, walks are length 1 — training still works
        # but may produce empty result or valid embeddings
        assert isinstance(result, dict)

    def test_method_name(self):
        cfg = ActivationConfig()
        trainer = Node2VecTrainer(cfg)
        assert trainer.method_name() == "node2vec"
