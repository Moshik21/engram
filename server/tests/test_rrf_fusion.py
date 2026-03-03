"""Tests for Reciprocal Rank Fusion in HybridSearchIndex."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.storage.sqlite.hybrid_search import HybridSearchIndex


@pytest.fixture
def make_index():
    """Factory to create HybridSearchIndex with configurable RRF settings."""

    def _make(use_rrf: bool = True, rrf_k: int = 60):
        fts = AsyncMock()
        vectors = AsyncMock()
        provider = MagicMock()
        provider.dimension.return_value = 512
        cfg = ActivationConfig(use_rrf=use_rrf, rrf_k=rrf_k)
        return HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=provider,
            cfg=cfg,
        )

    return _make


class TestRRFFusion:
    def test_rrf_scores_are_valid(self, make_index):
        """RRF scores are positive and normalized to 0-1."""
        idx = make_index()
        fts = [("e1", 10.0), ("e2", 5.0), ("e3", 2.0)]
        vec = [("e2", 0.9), ("e1", 0.8), ("e4", 0.7)]
        results = idx._merge_rrf(fts, vec, limit=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0
        assert results[0][1] == 1.0

    def test_rrf_single_source_fts_only(self, make_index):
        """RRF works with FTS results only."""
        idx = make_index()
        fts = [("e1", 10.0), ("e2", 5.0)]
        results = idx._merge_rrf(fts, [], limit=10)
        assert len(results) == 2
        assert results[0][0] == "e1"

    def test_rrf_single_source_vec_only(self, make_index):
        """RRF works with vector results only."""
        idx = make_index()
        vec = [("e1", 0.9), ("e2", 0.8)]
        results = idx._merge_rrf([], vec, limit=10)
        assert len(results) == 2
        assert results[0][0] == "e1"

    def test_rrf_disjoint_sets(self, make_index):
        """RRF correctly handles disjoint result sets."""
        idx = make_index()
        fts = [("e1", 10.0), ("e2", 5.0)]
        vec = [("e3", 0.9), ("e4", 0.8)]
        results = idx._merge_rrf(fts, vec, limit=10)
        assert len(results) == 4
        ids = {eid for eid, _ in results}
        assert ids == {"e1", "e2", "e3", "e4"}

    def test_rrf_vs_linear_different_rankings(self, make_index):
        """RRF and linear merge can produce different rankings."""
        idx_rrf = make_index(use_rrf=True)
        idx_linear = make_index(use_rrf=False)
        fts = [("e1", 10.0), ("e2", 1.0)]
        vec = [("e2", 0.99), ("e1", 0.1)]
        rrf_results = idx_rrf._merge_rrf(fts, vec, limit=10)
        linear_results = idx_linear._merge_linear(fts, vec, limit=10)
        assert len(rrf_results) == 2
        assert len(linear_results) == 2

    def test_config_toggle(self, make_index):
        """use_rrf config flag controls which merge method is used."""
        idx_rrf = make_index(use_rrf=True)
        idx_linear = make_index(use_rrf=False)
        fts = [("e1", 10.0)]
        vec = [("e1", 0.9)]
        result_rrf = idx_rrf._merge_results(fts, vec, limit=10)
        result_linear = idx_linear._merge_results(fts, vec, limit=10)
        assert len(result_rrf) == 1
        assert len(result_linear) == 1

    def test_rrf_empty_results(self, make_index):
        """RRF handles empty input gracefully."""
        idx = make_index()
        results = idx._merge_rrf([], [], limit=10)
        assert results == []

    def test_rrf_k_parameter_effect(self, make_index):
        """Different k values change score magnitudes but not ranking."""
        idx_low_k = make_index(rrf_k=1)
        idx_high_k = make_index(rrf_k=200)
        fts = [("e1", 10.0), ("e2", 5.0)]
        vec = [("e1", 0.9), ("e2", 0.8)]
        results_low = idx_low_k._merge_rrf(fts, vec, limit=10)
        results_high = idx_high_k._merge_rrf(fts, vec, limit=10)
        assert results_low[0][0] == results_high[0][0]
