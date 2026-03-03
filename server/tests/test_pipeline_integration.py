"""Integration tests for pipeline with reranker, MMR, and new benchmark methods."""

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.pipeline import retrieve
from engram.retrieval.reranker import NoopReranker


def _mock_search_index(results=None):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=results or [("e1", 0.9), ("e2", 0.7)])
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


def _mock_graph_store(neighbors=None):
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(
        return_value=neighbors or [],
    )
    store.get_entity = AsyncMock(return_value=Entity(
        id="e1", name="Test", entity_type="Thing",
        summary="A test entity", group_id="default",
    ))
    return store


def _mock_activation_store():
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value={})
    store.get_activation = AsyncMock(return_value=None)
    store.set_activation = AsyncMock()
    return store


class TestPipelineWithReranker:
    @pytest.mark.asyncio
    async def test_pipeline_with_reranker(self):
        """Pipeline calls reranker when enabled."""
        cfg = ActivationConfig(reranker_enabled=True, reranker_top_n=10)
        reranker = NoopReranker()

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
            reranker=reranker,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_without_reranker(self):
        """Pipeline works without reranker (backward compat)."""
        cfg = ActivationConfig(reranker_enabled=False)

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_reranker_enabled_but_none(self):
        """Pipeline handles reranker_enabled=True but no reranker provided."""
        cfg = ActivationConfig(reranker_enabled=True)

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
            reranker=None,
        )
        assert len(results) > 0


class TestPipelineWithMMR:
    @pytest.mark.asyncio
    async def test_pipeline_with_mmr_enabled(self):
        """Pipeline applies MMR when enabled."""
        cfg = ActivationConfig(mmr_enabled=True, mmr_lambda=0.7)

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_without_mmr(self):
        """Pipeline works without MMR (backward compat)."""
        cfg = ActivationConfig(mmr_enabled=False)

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
        )
        assert len(results) > 0


class TestPipelineWithBoth:
    @pytest.mark.asyncio
    async def test_pipeline_with_reranker_and_mmr(self):
        """Pipeline applies both reranker and MMR."""
        cfg = ActivationConfig(
            reranker_enabled=True, reranker_top_n=10,
            mmr_enabled=True, mmr_lambda=0.7,
        )

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
            reranker=NoopReranker(),
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_backward_compat(self):
        """Default config (no reranker, no MMR) works as before."""
        cfg = ActivationConfig()

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index(),
            cfg=cfg,
        )
        assert len(results) > 0
        assert results[0].node_id == "e1"
        assert results[0].score > 0


class TestBenchmarkMethods:
    def test_fan_spread_method_config(self):
        """Fan spread method has correct config (aggressive fan_s_max=2.5)."""
        from engram.benchmark.methods import METHOD_FAN_SPREAD
        assert METHOD_FAN_SPREAD.spreading_enabled is True
        assert METHOD_FAN_SPREAD.config.fan_s_max == 4.5

    def test_rrf_method_config(self):
        """RRF method has correct config."""
        from engram.benchmark.methods import METHOD_RRF
        assert METHOD_RRF.config.use_rrf is True
        assert METHOD_RRF.config.rrf_k == 60
        assert METHOD_RRF.spreading_enabled is True

    def test_mmr_method_config(self):
        """MMR method has correct config."""
        from engram.benchmark.methods import METHOD_MMR
        assert METHOD_MMR.config.mmr_enabled is True
        assert METHOD_MMR.config.mmr_lambda == 0.7

    def test_all_methods_includes_new(self):
        """ALL_METHODS includes the new benchmark methods."""
        from engram.benchmark.methods import ALL_METHODS
        names = [m.name for m in ALL_METHODS]
        assert "Fan Spread" in names
        assert "RRF Fusion" in names
        assert "MMR Diversity" in names
        assert "Linear Merge" in names
        assert "Context-Gated" in names
        assert "Post-Consolidation" in names
        assert len(ALL_METHODS) == 16

    def test_linear_merge_method_config(self):
        """Linear merge method has use_rrf=False."""
        from engram.benchmark.methods import METHOD_LINEAR
        assert METHOD_LINEAR.config.use_rrf is False
        assert METHOD_LINEAR.spreading_enabled is True

    def test_community_method_config(self):
        """Community method has correct config."""
        from engram.benchmark.methods import METHOD_COMMUNITY
        assert METHOD_COMMUNITY.config.community_spreading_enabled is True
        assert METHOD_COMMUNITY.config.community_bridge_boost == 1.5
        assert METHOD_COMMUNITY.config.community_intra_dampen == 0.7
        assert METHOD_COMMUNITY.spreading_enabled is True
        assert METHOD_COMMUNITY.routing_enabled is True

    def test_context_gated_method_config(self):
        """Context-gated method has correct config."""
        from engram.benchmark.methods import METHOD_CONTEXT_GATED
        assert METHOD_CONTEXT_GATED.config.context_gating_enabled is True
        assert METHOD_CONTEXT_GATED.config.context_gate_floor == 0.3
        assert METHOD_CONTEXT_GATED.spreading_enabled is True
        assert METHOD_CONTEXT_GATED.routing_enabled is True
