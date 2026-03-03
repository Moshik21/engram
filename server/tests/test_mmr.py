"""Tests for Maximal Marginal Relevance diversity re-ranking."""

from engram.retrieval.mmr import apply_mmr
from engram.retrieval.scorer import ScoredResult


def _make_result(node_id: str, score: float) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
    )


class TestMMR:
    def test_identical_embeddings_penalizes_duplicates(self):
        """Results with identical embeddings should be penalized."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
            _make_result("e3", 0.5),
        ]
        embeddings = {
            "e1": [1.0, 0.0, 0.0],
            "e2": [1.0, 0.0, 0.0],
            "e3": [0.0, 1.0, 0.0],
        }
        reranked = apply_mmr(results, embeddings, lambda_param=0.5, top_n=3)
        assert reranked[0].node_id == "e1"
        assert reranked[1].node_id == "e3"
        assert reranked[2].node_id == "e2"

    def test_diverse_embeddings_preserves_order(self):
        """Results with orthogonal embeddings are not penalized."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
            _make_result("e3", 0.8),
        ]
        embeddings = {
            "e1": [1.0, 0.0, 0.0],
            "e2": [0.0, 1.0, 0.0],
            "e3": [0.0, 0.0, 1.0],
        }
        reranked = apply_mmr(results, embeddings, lambda_param=0.7, top_n=3)
        assert reranked[0].node_id == "e1"
        assert reranked[1].node_id == "e2"

    def test_lambda_1_no_diversity(self):
        """lambda=1.0 means pure relevance, no diversity penalty."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
            _make_result("e3", 0.8),
        ]
        embeddings = {
            "e1": [1.0, 0.0],
            "e2": [1.0, 0.0],
            "e3": [0.0, 1.0],
        }
        reranked = apply_mmr(results, embeddings, lambda_param=1.0, top_n=3)
        assert reranked[0].node_id == "e1"
        assert reranked[1].node_id == "e2"
        assert reranked[2].node_id == "e3"

    def test_lambda_0_max_diversity(self):
        """lambda=0.0 means pure diversity, no relevance."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
            _make_result("e3", 0.8),
        ]
        embeddings = {
            "e1": [1.0, 0.0],
            "e2": [0.99, 0.1],
            "e3": [0.0, 1.0],
        }
        reranked = apply_mmr(results, embeddings, lambda_param=0.0, top_n=3)
        assert reranked[0].node_id == "e1"
        assert reranked[1].node_id == "e3"

    def test_empty_results(self):
        """Empty input returns empty output."""
        assert apply_mmr([], {}, lambda_param=0.7) == []

    def test_single_result(self):
        """Single result is returned as-is."""
        results = [_make_result("e1", 1.0)]
        reranked = apply_mmr(
            results, {"e1": [1.0, 0.0]}, lambda_param=0.7, top_n=5,
        )
        assert len(reranked) == 1
        assert reranked[0].node_id == "e1"

    def test_missing_embeddings_handled(self):
        """Results without embeddings are still included."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
        ]
        embeddings = {"e1": [1.0, 0.0]}
        reranked = apply_mmr(results, embeddings, lambda_param=0.7, top_n=2)
        assert len(reranked) == 2

    def test_no_embeddings_returns_original(self):
        """When no embeddings available, returns original order."""
        results = [
            _make_result("e1", 1.0),
            _make_result("e2", 0.9),
            _make_result("e3", 0.8),
        ]
        reranked = apply_mmr(results, {}, lambda_param=0.7, top_n=2)
        assert len(reranked) == 2
        assert reranked[0].node_id == "e1"
        assert reranked[1].node_id == "e2"

    def test_top_n_limits_output(self):
        """top_n parameter limits number of results returned."""
        results = [
            _make_result(f"e{i}", 1.0 - i * 0.1) for i in range(10)
        ]
        embeddings = {
            f"e{i}": [float(i == j) for j in range(10)] for i in range(10)
        }
        reranked = apply_mmr(results, embeddings, lambda_param=0.7, top_n=3)
        assert len(reranked) == 3
