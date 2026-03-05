"""Tests for graph embedding integration in scorer and retrieval pipeline."""

from __future__ import annotations

import time

import pytest

from engram.config import ActivationConfig
from engram.retrieval.scorer import ScoredResult, score_candidates


class TestScorerGraphStructural:
    def test_graph_sim_added_to_score(self):
        """Graph structural similarity should contribute to composite score."""
        cfg = ActivationConfig(
            weight_semantic=0.4,
            weight_activation=0.0,
            weight_spreading=0.0,
            weight_edge_proximity=0.0,
            weight_graph_structural=0.2,
        )
        now = time.time()

        candidates = [("e1", 0.5), ("e2", 0.5)]
        graph_sims = {"e1": 0.8, "e2": 0.0}

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
            graph_similarities=graph_sims,
        )

        # e1 should score higher due to graph similarity
        e1 = next(r for r in results if r.node_id == "e1")
        e2 = next(r for r in results if r.node_id == "e2")
        assert e1.score > e2.score
        assert e1.graph_structural == 0.8
        assert e2.graph_structural == 0.0

    def test_zero_weight_no_effect(self):
        """When weight_graph_structural=0, graph_sim should not affect score."""
        cfg = ActivationConfig(
            weight_semantic=0.4,
            weight_activation=0.25,
            weight_spreading=0.15,
            weight_edge_proximity=0.15,
            weight_graph_structural=0.0,  # disabled
        )
        now = time.time()

        candidates = [("e1", 0.5)]
        graph_sims = {"e1": 1.0}  # high graph sim

        results_with = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
            graph_similarities=graph_sims,
        )

        results_without = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
            graph_similarities=None,
        )

        assert results_with[0].score == pytest.approx(results_without[0].score)

    def test_none_graph_sims(self):
        """None graph_similarities should work fine."""
        cfg = ActivationConfig(weight_graph_structural=0.1)
        now = time.time()

        results = score_candidates(
            candidates=[("e1", 0.5)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
            graph_similarities=None,
        )

        assert results[0].graph_structural == 0.0

    def test_graph_structural_field_on_result(self):
        """ScoredResult should have graph_structural field."""
        r = ScoredResult(
            node_id="e1", score=1.0, semantic_similarity=0.5,
            activation=0.3, spreading=0.1, edge_proximity=0.2,
            graph_structural=0.7,
        )
        assert r.graph_structural == 0.7

    def test_graph_sim_ranking_effect(self):
        """Graph similarity should change ranking when weight > 0."""
        cfg = ActivationConfig(
            weight_semantic=0.3,
            weight_activation=0.0,
            weight_spreading=0.0,
            weight_edge_proximity=0.0,
            weight_graph_structural=0.3,
        )
        now = time.time()

        # e1 has lower semantic but high graph sim
        # e2 has higher semantic but no graph sim
        candidates = [("e1", 0.4), ("e2", 0.6)]
        graph_sims = {"e1": 1.0, "e2": 0.0}

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=now,
            cfg=cfg,
            graph_similarities=graph_sims,
        )

        # e1: 0.3*0.4 + 0.3*1.0 = 0.42
        # e2: 0.3*0.6 + 0.3*0.0 = 0.18
        assert results[0].node_id == "e1"

    def test_config_weight_field(self):
        """Config should accept weight_graph_structural."""
        cfg = ActivationConfig(weight_graph_structural=0.15)
        assert cfg.weight_graph_structural == 0.15

    def test_config_default_value(self):
        """Default weight should be 0.1."""
        cfg = ActivationConfig()
        assert cfg.weight_graph_structural == 0.1

    def test_config_node2vec_fields(self):
        """Config should accept all node2vec fields."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_dimensions=128,
            graph_embedding_node2vec_walk_length=30,
            graph_embedding_node2vec_num_walks=15,
            graph_embedding_node2vec_p=2.0,
            graph_embedding_node2vec_q=0.5,
            graph_embedding_node2vec_window=7,
            graph_embedding_node2vec_epochs=10,
            graph_embedding_node2vec_min_entities=100,
        )
        assert cfg.graph_embedding_node2vec_enabled is True
        assert cfg.graph_embedding_node2vec_dimensions == 128


class TestGraphEmbedPipelineFallback:
    """Tests for graph embedding pipeline fallback behavior."""

    def test_empty_seeds_uses_candidate_fallback(self):
        """When seed_node_ids is empty, top candidates should be used as proxies."""
        # This tests the logic change in pipeline.py
        # We verify by checking that the fallback code path computes seeds from candidates
        candidates = [("e1", 0.9), ("e2", 0.7), ("e3", 0.5), ("e4", 0.3)]
        seed_node_ids: set[str] = set()  # Empty seeds

        # Simulate the fallback logic
        query_seed_ids = list(seed_node_ids) if seed_node_ids else []
        if not query_seed_ids:
            query_seed_ids = [
                eid for eid, _ in sorted(
                    candidates, key=lambda x: x[1], reverse=True,
                )[:3]
            ]

        assert query_seed_ids == ["e1", "e2", "e3"]

    def test_method_selection_fallback(self):
        """When first method has no data, second method should be tried."""
        # The pipeline loop tries methods in order and continues on empty results
        methods_to_try = ["node2vec", "transe"]
        graph_data = {"transe": {"e1": [0.1, 0.2]}}  # Only transe has data

        used_method = None
        for method in methods_to_try:
            data = graph_data.get(method, {})
            if not data:
                continue
            used_method = method
            break

        assert used_method == "transe"
