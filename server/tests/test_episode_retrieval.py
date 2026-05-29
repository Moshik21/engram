"""Tests for episode retrieval as retrieval targets (Item 6)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.retrieval.pipeline import retrieve
from engram.retrieval.scorer import ScoredResult
from engram.utils.dates import utc_now
from tests.conftest import _helix_available

# ── ScoredResult tests ──────────────────────────────────────────────


class TestScoredResultType:
    def test_result_type_defaults_to_entity(self):
        """ScoredResult.result_type defaults to 'entity' for backward compat."""
        sr = ScoredResult(
            node_id="e1",
            score=0.5,
            semantic_similarity=0.5,
            activation=0.3,
            spreading=0.1,
            edge_proximity=0.0,
        )
        assert sr.result_type == "entity"

    def test_result_type_can_be_episode(self):
        """ScoredResult.result_type can be set to 'episode'."""
        sr = ScoredResult(
            node_id="ep_123",
            score=0.4,
            semantic_similarity=0.5,
            activation=0.0,
            spreading=0.0,
            edge_proximity=0.0,
            result_type="episode",
        )
        assert sr.result_type == "episode"


# ── Config tests ────────────────────────────────────────────────────


class TestEpisodeRetrievalConfig:
    def test_default_episode_retrieval_enabled(self):
        """Episode retrieval is enabled by default."""
        cfg = ActivationConfig()
        assert cfg.episode_retrieval_enabled is True

    def test_default_episode_retrieval_weight(self):
        """Episode retrieval weight defaults to 0.8."""
        cfg = ActivationConfig()
        assert cfg.episode_retrieval_weight == 0.8

    def test_default_episode_retrieval_max(self):
        """Episode retrieval max defaults to 5."""
        cfg = ActivationConfig()
        assert cfg.episode_retrieval_max == 5

    def test_default_core_tier_excludes_entities(self):
        """Core episode-vector tier is the default: no entity slots in top-k,
        passage_first strategy. Pins the core-tier separation defaults."""
        cfg = ActivationConfig()
        assert cfg.passage_first_entity_budget == 0
        assert cfg.retrieval_strategy == "passage_first"


# ── HelixSearchIndex search_episodes tests ──────────────────────────


@pytest.mark.requires_helix
@pytest.mark.skipif(not _helix_available(), reason="HelixDB not available")
class TestHelixSearchEpisodes:
    @pytest.mark.asyncio
    async def test_search_episodes_returns_results(self):
        """HelixSearchIndex.search_episodes returns episode results."""
        from engram.config import EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex

        graph = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph.initialize()
        search = HelixSearchIndex(
            helix_config=HelixDBConfig(host="localhost", port=6969),
            provider=NoopProvider(),
            embed_config=EmbeddingConfig(),
            storage_dim=0,
            embed_provider="noop",
            embed_model="noop",
        )
        await search.initialize()

        ep = Episode(
            id="ep_test1",
            content="Alice works at TechCorp on machine learning projects",
            source="test",
            status=EpisodeStatus.COMPLETED,
            group_id="default",
            created_at=utc_now(),
        )
        await graph.create_episode(ep)
        results = await search.search_episodes("machine learning", group_id="default")
        assert len(results) >= 1
        assert results[0][0] == "ep_test1"
        assert 0.0 <= results[0][1] <= 1.0
        await graph.close()
        await search.close()

    @pytest.mark.asyncio
    async def test_search_episodes_empty_query(self):
        """HelixSearchIndex.search_episodes handles empty query."""
        from engram.config import EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.storage.helix.search import HelixSearchIndex

        search = HelixSearchIndex(
            helix_config=HelixDBConfig(host="localhost", port=6969),
            provider=NoopProvider(),
            embed_config=EmbeddingConfig(),
            storage_dim=0,
            embed_provider="noop",
            embed_model="noop",
        )
        await search.initialize()
        results = await search.search_episodes("", group_id="default")
        assert results == []
        await search.close()


# ── Pipeline episode tests ──────────────────────────────────────────


def _mock_search_index_with_episodes(
    entity_results=None,
    episode_results=None,
    cue_results=None,
):
    """Create a mock search index that supports search_episodes."""
    idx = AsyncMock()
    idx.search = AsyncMock(
        return_value=entity_results if entity_results is not None else [("e1", 0.9), ("e2", 0.7)],
    )
    idx.search_episodes = AsyncMock(
        return_value=(
            episode_results if episode_results is not None else [("ep_1", 0.8), ("ep_2", 0.6)]
        ),
    )
    idx.search_episode_cues = AsyncMock(
        return_value=cue_results if cue_results is not None else [],
    )
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


class _SearchIndexMissingEpisodes:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9)])
        self.search_episode_cues = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False


class _SearchIndexMissingCues:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9)])
        self.search_episodes = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False


class _SlowPrimarySearchIndex:
    def __init__(self):
        self.search = AsyncMock(side_effect=self._slow_search)
        self.search_episodes = AsyncMock(return_value=[("ep_slow", 0.2)])
        self.search_episode_cues = AsyncMock(return_value=[("ep_slow_cue", 0.2)])
        self.search_episodes_fast = AsyncMock(return_value=[("ep_fast", 0.9)])
        self.search_episode_cues_fast = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False

    async def _slow_search(self, **_kwargs):
        await asyncio.sleep(0.2)
        return [("e_slow", 0.9)]


class _SlowEmptyPrimarySearchIndex:
    def __init__(self):
        self.search = AsyncMock(side_effect=self._slow_search)
        self.search_episodes = AsyncMock(return_value=[])
        self.search_episode_cues = AsyncMock(return_value=[])
        self.search_episodes_fast = AsyncMock(return_value=[])
        self.search_episode_cues_fast = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False

    async def _slow_search(self, **_kwargs):
        await asyncio.sleep(0.2)
        return []


class _SlowChunkSearchIndex:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9)])
        self.search_episodes = AsyncMock(return_value=[("ep_1", 0.7)])
        self.search_episode_cues = AsyncMock(return_value=[])
        self.search_episode_chunks = AsyncMock(side_effect=self._slow_chunk_search)
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False

    async def _slow_chunk_search(self, **_kwargs):
        await asyncio.sleep(0.2)
        return [
            {
                "episode_id": "ep_chunk",
                "chunk_text": "late chunk match",
                "score": 0.95,
            }
        ]


class _SlowEpisodeSearchIndex:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9)])
        self.search_episodes = AsyncMock(side_effect=self._slow_episode_search)
        self.search_episode_cues = AsyncMock(return_value=[])
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False

    async def _slow_episode_search(self, **_kwargs):
        await asyncio.sleep(0.2)
        return [("ep_slow", 0.95)]


class _SlowCueSearchIndex:
    def __init__(self):
        self.search = AsyncMock(return_value=[("e1", 0.9)])
        self.search_episodes = AsyncMock(return_value=[])
        self.search_episode_cues = AsyncMock(side_effect=self._slow_cue_search)
        self.compute_similarity = AsyncMock(return_value={})
        self._embeddings_enabled = False

    async def _slow_cue_search(self, **_kwargs):
        await asyncio.sleep(0.2)
        return [("ep_slow_cue", 0.95)]


class _SlowReranker:
    async def rerank(self, *_args, **_kwargs):
        await asyncio.sleep(0.2)
        return []


class _WorkingMemoryStub:
    def get_candidates(self, _now):
        return [("wm_zero", 1.0, "entity")]


def _mock_graph_store():
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    store.get_entity = AsyncMock(
        return_value=Entity(
            id="e1",
            name="Test",
            entity_type="Thing",
            summary="A test entity",
            group_id="default",
        )
    )
    store.get_episode_by_id = AsyncMock(
        return_value=Episode(
            id="ep_1",
            content="Test episode content that is quite long",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.CUE_ONLY,
            group_id="default",
            created_at=utc_now(),
        )
    )
    store.get_episode_cue = AsyncMock(
        return_value=EpisodeCue(
            episode_id="ep_1",
            group_id="default",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text="mentions: Test",
            first_spans=["Test episode content that is quite long"],
            hit_count=0,
            route_reason="entity_dense",
        )
    )
    store.get_episode_entities = AsyncMock(return_value=["e1", "e2"])
    return store


def _mock_activation_store():
    store = AsyncMock()
    store.batch_get = AsyncMock(return_value={})
    store.get_activation = AsyncMock(return_value=None)
    store.set_activation = AsyncMock()
    store.record_access = AsyncMock()
    store.get_top_activated = AsyncMock(return_value=[])
    return store


class TestPipelineEpisodeRetrieval:
    @pytest.mark.asyncio
    async def test_episode_retrieval_enabled_returns_episodes(self):
        """Pipeline with episode_retrieval_enabled=True returns episode results."""
        cfg = ActivationConfig(episode_retrieval_enabled=True, episode_retrieval_max=2)
        stage_timings = {}

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )
        episode_results = [r for r in results if r.result_type == "episode"]
        assert len(episode_results) > 0
        assert stage_timings["recall_embed"] >= 0

    @pytest.mark.asyncio
    async def test_default_core_tier_returns_no_entities(self):
        """Default config (core episode-vector tier) keeps entities out of the
        top-k: every result is an episode or cue_episode, never an entity."""
        cfg = ActivationConfig()  # passage_first_entity_budget defaults to 0
        assert cfg.passage_first_entity_budget == 0

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(
                entity_results=[("e1", 0.95), ("e2", 0.9)],
                episode_results=[("ep_1", 0.8), ("ep_2", 0.6)],
            ),
            cfg=cfg,
        )
        assert results
        assert all(r.result_type in {"episode", "cue_episode"} for r in results)
        assert not any(r.result_type == "entity" for r in results)

    @pytest.mark.asyncio
    async def test_explicit_depth_path_returns_entities(self):
        """Opt-in depth/graph tier (passage_first_entity_budget >= 1 plus a
        nonzero weight_graph_structural) surfaces entity results in the top-k."""
        cfg = ActivationConfig(
            passage_first_entity_budget=3,
            weight_graph_structural=0.1,
        )
        assert cfg.passage_first_entity_budget == 3

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(
                entity_results=[("e1", 0.95), ("e2", 0.9)],
                episode_results=[("ep_1", 0.8), ("ep_2", 0.6)],
            ),
            cfg=cfg,
        )
        entity_results = [r for r in results if r.result_type == "entity"]
        assert len(entity_results) > 0

    @pytest.mark.asyncio
    async def test_primary_search_timeout_uses_fast_episode_fallback(self):
        """A slow primary entity search does not consume the full recall path."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=True,
            chunk_search_enabled=True,
            retrieval_primary_search_timeout_ms=25,
            recall_planner_enabled=True,
        )
        search = _SlowPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert [result.node_id for result in results] == ["ep_fast"]
        assert results[0].result_type == "episode"
        assert stage_timings["recall_primary_search_timeout"] >= 25
        assert stage_timings["recall_planner_skipped_primary_timeout"] == 0.0
        assert "recall_primary_search" not in stage_timings
        assert "recall_planner_timeout" not in stage_timings
        search.search_episodes_fast.assert_awaited_once()
        search.search_episode_cues_fast.assert_awaited_once()
        search.search_episodes.assert_not_awaited()
        search.search_episode_cues.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_primary_timeout_zero_semantic_pool_returns_special_results(self):
        """Zero-score side pools do not hide episode/cue hits behind graph scoring."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=True,
            chunk_search_enabled=False,
            retrieval_primary_search_timeout_ms=25,
            working_memory_enabled=True,
            recall_planner_enabled=True,
        )
        activation = _mock_activation_store()
        search = _SlowPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            working_memory=_WorkingMemoryStub(),
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_candidate_count"] == 1.0
        assert stage_timings["recall_candidate_max_score"] == 0.0
        assert stage_timings["recall_planner_skipped_primary_timeout"] == 0.0
        assert stage_timings["recall_zero_semantic_special_deferred"] == 1.0
        assert "recall_planner_timeout" not in stage_timings
        assert "recall_activation_state" not in stage_timings
        assert "recall_spread" not in stage_timings
        activation.batch_get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stats_timeout_falls_back_to_default_pool_sizes(self):
        """Slow corpus stats do not consume the recall budget."""
        async def slow_stats(_group_id):
            await asyncio.sleep(0.2)
            return {"entity_count": 100_000}

        cfg = ActivationConfig(
            retrieval_stats_timeout_ms=25,
            retrieval_primary_search_timeout_ms=0,
        )
        graph = _mock_graph_store()
        graph.get_stats = AsyncMock(side_effect=slow_stats)
        stage_timings = {}

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert stage_timings["recall_stats_timeout"] >= 25
        assert "recall_stats" not in stage_timings

    @pytest.mark.asyncio
    async def test_graph_query_expansion_timeout_uses_original_query(self, monkeypatch):
        """Slow graph expansion is bounded before primary search."""
        async def slow_expand(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return "expanded query"

        monkeypatch.setattr(
            "engram.retrieval.graph_expansion.expand_query_from_graph",
            slow_expand,
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            graph_query_expansion_timeout_ms=25,
        )
        search = _mock_search_index_with_episodes(entity_results=[("e1", 0.9)])
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert stage_timings["graph_expand_timeout"] >= 25
        assert "graph_expand" not in stage_timings
        assert search.search.call_args.kwargs["query"] == "native latency"

    @pytest.mark.asyncio
    async def test_stats_timeout_skips_graph_expansion_and_caps_primary_search(
        self,
        monkeypatch,
    ):
        """A slow graph preflight avoids stacking expansion plus full primary timeout."""
        expand_calls = 0

        async def slow_stats(_group_id):
            await asyncio.sleep(0.2)
            return {"entity_count": 100_000}

        async def slow_expand(*_args, **_kwargs):
            nonlocal expand_calls
            expand_calls += 1
            await asyncio.sleep(0.2)
            return "expanded query"

        monkeypatch.setattr(
            "engram.retrieval.graph_expansion.expand_query_from_graph",
            slow_expand,
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_stats_timeout_ms=25,
            retrieval_primary_search_timeout_ms=200,
            retrieval_primary_search_timeout_after_probe_timeout_ms=40,
        )
        graph = _mock_graph_store()
        graph.get_stats = AsyncMock(side_effect=slow_stats)
        search = _SlowEmptyPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="Native Engram latency",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert expand_calls == 0
        assert stage_timings["recall_stats_timeout"] >= 25
        assert stage_timings["graph_expand_skipped_stats_timeout"] == 0.0
        assert stage_timings["recall_primary_search_effective_timeout_ms"] == 40
        assert 40 <= stage_timings["recall_primary_search_timeout"] < 100
        assert "graph_expand_timeout" not in stage_timings

    @pytest.mark.asyncio
    async def test_graph_expansion_timeout_caps_primary_search(self, monkeypatch):
        """A graph expansion timeout bounds the next primary search attempt."""
        expand_calls = 0

        async def slow_expand(*_args, **_kwargs):
            nonlocal expand_calls
            expand_calls += 1
            await asyncio.sleep(0.2)
            return "expanded query"

        monkeypatch.setattr(
            "engram.retrieval.graph_expansion.expand_query_from_graph",
            slow_expand,
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            graph_query_expansion_timeout_ms=25,
            retrieval_primary_search_timeout_ms=200,
            retrieval_primary_search_timeout_after_probe_timeout_ms=40,
        )
        search = _SlowEmptyPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="Native Engram latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert expand_calls == 1
        assert stage_timings["graph_expand_timeout"] >= 25
        assert stage_timings["recall_primary_search_effective_timeout_ms"] == 40
        assert 40 <= stage_timings["recall_primary_search_timeout"] < 100

    @pytest.mark.asyncio
    async def test_probe_timeout_skips_secondary_graph_scoring(self, monkeypatch):
        """Responsive primary hits are returned without graph-heavy enhancers."""
        async def slow_expand(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return "expanded query"

        async def slow_get_entity(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return Entity(
                id="e1",
                name="Test",
                entity_type="Thing",
                summary="A test entity",
                group_id="default",
            )

        async def slow_find_entities(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return []

        monkeypatch.setattr(
            "engram.retrieval.graph_expansion.expand_query_from_graph",
            slow_expand,
        )
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            graph_query_expansion_timeout_ms=25,
            retrieval_graph_pool_timeout_ms=25,
            retrieval_entity_attributes_timeout_ms=25,
            emotional_salience_enabled=True,
            goal_priming_enabled=True,
            cross_domain_penalty_enabled=True,
            gc_mmr_enabled=True,
        )
        graph = _mock_graph_store()
        graph.get_entity = AsyncMock(side_effect=slow_get_entity)
        graph.find_entities = AsyncMock(side_effect=slow_find_entities)
        search = _mock_search_index_with_episodes(entity_results=[("e1", 0.9)])
        stage_timings = {}

        results = await retrieve(
            query="Native Engram latency",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["graph_expand_timeout"] >= 25
        assert stage_timings["recall_graph_pool_skipped_probe_timeout"] == 0.0
        assert stage_timings["recall_goal_priming_skipped_probe_timeout"] == 0.0
        assert stage_timings["recall_cross_domain_seed_skipped_probe_timeout"] == 0.0
        assert stage_timings["recall_spread_skipped_probe_timeout"] == 0.0
        assert stage_timings["recall_entity_attributes_skipped_probe_timeout"] == 0.0
        assert stage_timings["recall_gc_mmr_skipped_probe_timeout"] == 0.0
        assert "recall_graph_pool_timeout" not in stage_timings
        assert "recall_goal_priming_timeout" not in stage_timings
        assert "recall_cross_domain_seed_timeout" not in stage_timings
        assert "recall_spread_timeout" not in stage_timings
        assert "recall_entity_attributes_timeout" not in stage_timings
        assert "recall_gc_mmr" not in stage_timings
        graph.get_entity.assert_not_awaited()
        graph.find_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_reranker_does_not_materialize_documents(self):
        """The pass-through reranker must not spend graph reads building docs."""
        from engram.retrieval.reranker import NoopReranker

        async def slow_get_entity(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return Entity(
                id="e1",
                name="Test",
                entity_type="Thing",
                summary="A test entity",
                group_id="default",
            )

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            emotional_salience_enabled=False,
            mmr_enabled=False,
        )
        graph = _mock_graph_store()
        graph.get_entity = AsyncMock(side_effect=slow_get_entity)
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(entity_results=[("e1", 0.9)]),
            cfg=cfg,
            reranker=NoopReranker(),
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_reranker_skipped_noop"] == 0.0
        assert "recall_reranker" not in stage_timings
        graph.get_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_primary_timeout_empty_candidates_skips_planner_and_entity_match(self):
        """A no-evidence primary timeout avoids secondary graph-heavy miss probes."""
        async def slow_find_entities(**_kwargs):
            await asyncio.sleep(0.2)
            return []

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_primary_search_timeout_ms=25,
            recall_planner_enabled=True,
        )
        graph = _mock_graph_store()
        graph.find_entities = AsyncMock(side_effect=slow_find_entities)
        search = _SlowEmptyPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="zzzzquasarflux xylofract",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_primary_search_timeout"] >= 25
        assert stage_timings["recall_planner_skipped_primary_timeout"] == 0.0
        assert stage_timings["recall_entity_match_skipped_primary_timeout"] == 0.0
        assert "recall_planner_timeout" not in stage_timings
        assert "recall_entity_match_timeout" not in stage_timings
        graph.find_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_zero_semantic_working_memory_candidates_short_circuit_graph_scoring(self):
        """Zero-score non-semantic pools do not force graph-heavy miss scoring."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_primary_search_timeout_ms=25,
            working_memory_enabled=True,
        )
        activation = _mock_activation_store()
        search = _SlowEmptyPrimarySearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="zzzzquasarflux xylofract",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            working_memory=_WorkingMemoryStub(),
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_primary_search_timeout"] >= 25
        assert stage_timings["recall_working_memory_candidate_count"] == 1.0
        assert stage_timings["recall_candidate_count"] == 1.0
        assert stage_timings["recall_candidate_max_score"] == 0.0
        assert stage_timings["recall_zero_semantic_short_circuit"] == 0.0
        assert "recall_activation_state" not in stage_timings
        assert "recall_spread" not in stage_timings
        activation.batch_get.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_episode_search_timeout_preserves_entity_results(self):
        """Slow episode search is bounded and does not erase entity results."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_episode_search_timeout_ms=25,
        )
        search = _SlowEpisodeSearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert "ep_slow" not in {result.node_id for result in results}
        assert stage_timings["recall_episode_search_timeout"] >= 25
        assert "recall_episode_search" not in stage_timings
        assert search.search_episodes.await_count == 1

    @pytest.mark.asyncio
    async def test_cue_search_timeout_preserves_entity_results(self):
        """Slow cue search is bounded and does not erase entity results."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            chunk_search_enabled=False,
            retrieval_cue_search_timeout_ms=25,
        )
        search = _SlowCueSearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert "ep_slow_cue" not in {result.node_id for result in results}
        assert stage_timings["recall_cue_search_timeout"] >= 25
        assert "recall_cue_search" not in stage_timings
        assert search.search_episode_cues.await_count == 1

    @pytest.mark.asyncio
    async def test_activation_state_timeout_preserves_search_results(self):
        """Slow activation-state loading is bounded and recall still scores results."""
        async def slow_batch_get(_entity_ids):
            await asyncio.sleep(0.2)
            return {}

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_activation_state_timeout_ms=25,
        )
        activation = _mock_activation_store()
        activation.batch_get = AsyncMock(side_effect=slow_batch_get)
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=activation,
            search_index=_mock_search_index_with_episodes(entity_results=[("e1", 0.9)]),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_activation_state_timeout"] >= 25
        assert "recall_activation_state" not in stage_timings
        assert activation.batch_get.await_count == 1

    @pytest.mark.asyncio
    async def test_entity_match_timeout_preserves_empty_result(self):
        """Slow name-match fallback is bounded when semantic search returns nothing."""
        async def slow_find_entities(**_kwargs):
            await asyncio.sleep(0.2)
            return []

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_entity_match_timeout_ms=25,
        )
        graph = _mock_graph_store()
        graph.find_entities = AsyncMock(side_effect=slow_find_entities)
        stage_timings = {}

        results = await retrieve(
            query="zzzzquasarflux xylofract",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(entity_results=[]),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_entity_match_timeout"] >= 25
        assert "recall_entity_match" not in stage_timings

    @pytest.mark.asyncio
    async def test_spread_timeout_preserves_search_results(self, monkeypatch):
        """Slow graph spreading is bounded and recall still returns semantic results."""
        async def slow_spread(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return {"e2": 0.5}, {"e2": 1}

        monkeypatch.setattr("engram.retrieval.pipeline.spread_activation", slow_spread)

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_spread_timeout_ms=25,
        )
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(entity_results=[("e1", 0.9)]),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_spread_timeout"] >= 25
        assert "recall_spread" not in stage_timings

    @pytest.mark.asyncio
    async def test_entity_attributes_timeout_preserves_search_results(self):
        """Slow entity-attribute loading is bounded and recall still returns results."""
        async def slow_get_entity(_entity_id, _group_id):
            await asyncio.sleep(0.2)
            return Entity(
                id="e1",
                name="Test",
                entity_type="Thing",
                summary="A test entity",
                group_id="default",
            )

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_entity_attributes_timeout_ms=25,
        )
        graph = _mock_graph_store()
        graph.get_entity = AsyncMock(side_effect=slow_get_entity)
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=graph,
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(entity_results=[("e1", 0.9)]),
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_entity_attributes_timeout"] >= 25
        assert "recall_entity_attributes" not in stage_timings

    @pytest.mark.asyncio
    async def test_graph_similarity_timeout_preserves_search_results(self):
        """Slow graph-structural similarity is bounded and recall still returns results."""
        async def slow_graph_embeddings(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            return {}

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_graph_similarity_timeout_ms=25,
            weight_graph_structural=0.1,
        )
        search = _mock_search_index_with_episodes(entity_results=[("e1", 0.9)])
        search.get_graph_embeddings = AsyncMock(side_effect=slow_graph_embeddings)
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_graph_similarity_timeout"] >= 25
        assert "recall_graph_similarity" not in stage_timings

    @pytest.mark.asyncio
    async def test_reranker_timeout_preserves_search_results(self):
        """Slow reranker preparation/execution is bounded after scoring."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            chunk_search_enabled=False,
            retrieval_reranker_timeout_ms=25,
            mmr_enabled=False,
        )
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(entity_results=[("e1", 0.9)]),
            cfg=cfg,
            reranker=_SlowReranker(),
            stage_timings_ms=stage_timings,
        )

        assert results
        assert results[0].node_id == "e1"
        assert stage_timings["recall_reranker_timeout"] >= 25
        assert "recall_reranker" not in stage_timings

    @pytest.mark.asyncio
    async def test_chunk_search_timeout_preserves_base_results(self):
        """Slow chunk search is bounded and does not erase other recall results."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=False,
            chunk_search_enabled=True,
            retrieval_chunk_search_timeout_ms=25,
        )
        search = _SlowChunkSearchIndex()
        stage_timings = {}

        results = await retrieve(
            query="native latency",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=search,
            cfg=cfg,
            stage_timings_ms=stage_timings,
        )

        assert results
        assert "ep_chunk" not in {result.node_id for result in results}
        assert stage_timings["recall_chunk_search_timeout"] >= 25
        assert "recall_chunk_search" not in stage_timings
        assert search.search_episode_chunks.await_count == 1

    @pytest.mark.asyncio
    async def test_episode_retrieval_disabled_no_episodes(self):
        """Pipeline with episode_retrieval_enabled=False returns no episodes."""
        cfg = ActivationConfig(episode_retrieval_enabled=False)

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(),
            cfg=cfg,
        )
        episode_results = [r for r in results if r.result_type == "episode"]
        assert len(episode_results) == 0

    @pytest.mark.asyncio
    async def test_episode_retrieval_max_caps_count(self):
        """episode_retrieval_max caps the number of episodes in results."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=1,
        )

        idx = _mock_search_index_with_episodes(
            episode_results=[("ep_1", 0.9), ("ep_2", 0.8), ("ep_3", 0.7)],
        )

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=idx,
            cfg=cfg,
        )
        episode_results = [r for r in results if r.result_type == "episode"]
        assert len(episode_results) <= 1

    @pytest.mark.asyncio
    async def test_episode_retrieval_weight_discount(self):
        """episode_retrieval_weight discount is applied to episode scores."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_weight=0.5,
            retrieval_strategy="hybrid",
        )

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(
                episode_results=[("ep_1", 1.0)],
            ),
            cfg=cfg,
        )
        ep_results = [r for r in results if r.result_type == "episode"]
        if ep_results:
            # score = weight_semantic * sem_sim * episode_retrieval_weight
            # = 0.40 * 1.0 * 0.5 = 0.20
            assert ep_results[0].score == pytest.approx(
                cfg.weight_semantic * 1.0 * 0.5,
                abs=0.01,
            )

    @pytest.mark.asyncio
    async def test_episode_no_search_episodes_method(self):
        """Pipeline raises when episode retrieval is enabled but unsupported."""
        cfg = ActivationConfig(episode_retrieval_enabled=True)

        with pytest.raises(RuntimeError, match="search_episodes"):
            await retrieve(
                query="test query",
                group_id="default",
                graph_store=_mock_graph_store(),
                activation_store=_mock_activation_store(),
                search_index=_SearchIndexMissingEpisodes(),
                cfg=cfg,
            )

    @pytest.mark.asyncio
    async def test_cue_no_search_episode_cues_method(self):
        """Pipeline raises when cue recall is enabled but unsupported."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
        )

        with pytest.raises(RuntimeError, match="search_episode_cues"):
            await retrieve(
                query="test query",
                group_id="default",
                graph_store=_mock_graph_store(),
                activation_store=_mock_activation_store(),
                search_index=_SearchIndexMissingCues(),
                cfg=cfg,
            )

    @pytest.mark.asyncio
    async def test_cue_recall_enabled_returns_cue_episodes(self):
        """Pipeline returns cue-backed episode results when cue recall is enabled."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            cue_recall_max=2,
        )

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(
                cue_results=[("ep_1", 0.85)],
            ),
            cfg=cfg,
        )

        cue_results = [r for r in results if r.result_type == "cue_episode"]
        assert len(cue_results) == 1
        assert cue_results[0].node_id == "ep_1"
        assert cue_results[0].score == pytest.approx(
            cfg.weight_semantic * 0.85 * cfg.cue_recall_weight,
            abs=0.01,
        )

    @pytest.mark.asyncio
    async def test_episode_and_cue_results_coexist(self):
        """Pipeline can return raw episode and cue-backed packets together."""
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=2,
            cue_recall_enabled=True,
            cue_recall_max=2,
            working_memory_enabled=False,
        )

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(
                entity_results=[],
                episode_results=[("ep_raw", 0.75)],
                cue_results=[("ep_cue", 0.9)],
            ),
            cfg=cfg,
            limit=4,
        )

        by_type = {result.result_type: result.node_id for result in results}
        assert by_type["episode"] == "ep_raw"
        assert by_type["cue_episode"] == "ep_cue"


# ── GraphManager.recall() tests ─────────────────────────────────────


class TestGraphManagerRecallEpisodes:
    @pytest.mark.asyncio
    async def test_fast_recall_fallback_materializes_cue_then_episode_hits(self):
        """GraphManager fast fallback skips graph expansion and returns cue/episode hits."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_episode_by_id = AsyncMock(
            side_effect=lambda episode_id, _group_id: Episode(
                id=episode_id,
                content=f"content for {episode_id}",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            )
        )
        graph.get_episode_cue = AsyncMock(
            side_effect=lambda episode_id, _group_id: EpisodeCue(
                episode_id=episode_id,
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text=f"cue for {episode_id}",
                first_spans=[f"content for {episode_id}"],
                route_reason="fallback_test",
            )
        )
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            episode_results=[("ep_2", 0.7)],
            cue_results=[("ep_1", 0.9)],
        )
        extractor = AsyncMock()
        gm = GraphManager(graph, activation, search, extractor, cfg=ActivationConfig())

        results = await gm.fast_recall_fallback(
            query="Engram latency",
            group_id="default",
            limit=2,
        )

        assert [result["result_type"] for result in results] == ["cue_episode", "episode"]
        assert results[0]["cue"]["episode_id"] == "ep_1"
        assert results[1]["episode"]["id"] == "ep_2"
        search.search.assert_not_awaited()
        search.search_episode_cues.assert_awaited_once()
        search.search_episodes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fast_recall_fallback_prefers_fast_search_methods(self):
        """GraphManager uses backend fast search when available for fallback."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_episode_by_id = AsyncMock(
            side_effect=lambda episode_id, _group_id: Episode(
                id=episode_id,
                content=f"content for {episode_id}",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            )
        )
        graph.get_episode_cue = AsyncMock(
            side_effect=lambda episode_id, _group_id: EpisodeCue(
                episode_id=episode_id,
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text=f"cue for {episode_id}",
                first_spans=[f"content for {episode_id}"],
                route_reason="fallback_test",
            )
        )
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            episode_results=[("ep_slow", 0.2)],
            cue_results=[("ep_slow_cue", 0.3)],
        )
        search.search_episode_cues_fast = AsyncMock(return_value=[("ep_1", 0.9)])
        search.search_episodes_fast = AsyncMock(return_value=[("ep_2", 0.7)])
        extractor = AsyncMock()
        gm = GraphManager(graph, activation, search, extractor, cfg=ActivationConfig())

        results = await gm.fast_recall_fallback(
            query="Engram latency",
            group_id="default",
            limit=2,
        )

        assert [result["result_type"] for result in results] == ["cue_episode", "episode"]
        assert results[0]["cue"]["episode_id"] == "ep_1"
        assert results[1]["episode"]["id"] == "ep_2"
        search.search_episode_cues_fast.assert_awaited_once()
        search.search_episodes_fast.assert_awaited_once()
        search.search_episode_cues.assert_not_awaited()
        search.search_episodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fast_recall_fallback_prefers_direct_record_materialization(self):
        """GraphManager avoids graph lookups when backend returns fallback records."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_episode_by_id = AsyncMock()
        graph.get_episode_entities = AsyncMock()
        graph.get_episode_cue = AsyncMock()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            episode_results=[("ep_slow", 0.2)],
            cue_results=[("ep_slow_cue", 0.3)],
        )
        search.search_episode_cue_records_fast = AsyncMock(
            return_value=[
                {
                    "episode_id": "ep_1",
                    "group_id": "default",
                    "cue_text": "Engram latency cue",
                    "first_spans_json": '["Engram latency cue"]',
                    "projection_state": "cue_only",
                    "route_reason": "fallback_test",
                    "_score": 0.9,
                }
            ]
        )
        search.search_episode_records_fast = AsyncMock(
            return_value=[
                {
                    "episode_id": "ep_2",
                    "group_id": "default",
                    "content": "Engram latency episode",
                    "source": "test",
                    "created_at": "2026-05-26T00:00:00",
                    "projection_state": "cue_only",
                    "_score": 0.7,
                }
            ]
        )
        extractor = AsyncMock()
        gm = GraphManager(graph, activation, search, extractor, cfg=ActivationConfig())

        results = await gm.fast_recall_fallback(
            query="Engram latency",
            group_id="default",
            limit=2,
        )

        assert [result["result_type"] for result in results] == ["cue_episode", "episode"]
        assert results[0]["cue"]["episode_id"] == "ep_1"
        assert results[0]["cue"]["supporting_spans"] == ["Engram latency cue"]
        assert results[1]["episode"]["id"] == "ep_2"
        assert results[1]["episode"]["content"] == "Engram latency episode"
        graph.get_episode_by_id.assert_not_awaited()
        graph.get_episode_entities.assert_not_awaited()
        graph.get_episode_cue.assert_not_awaited()
        search.search_episode_cue_records_fast.assert_awaited_once()
        search.search_episode_records_fast.assert_awaited_once()
        search.search_episode_cues.assert_not_awaited()
        search.search_episodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fast_recall_fallback_skips_legacy_when_record_search_is_empty(self):
        """Direct record BM25 misses should not repeat the same legacy BM25 work."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_episode_by_id = AsyncMock()
        graph.get_episode_entities = AsyncMock()
        graph.get_episode_cue = AsyncMock()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            episode_results=[("ep_legacy", 0.7)],
            cue_results=[("ep_legacy_cue", 0.8)],
        )
        search.search_episode_cue_records_fast = AsyncMock(return_value=[])
        search.search_episode_records_fast = AsyncMock(return_value=[])
        extractor = AsyncMock()
        gm = GraphManager(graph, activation, search, extractor, cfg=ActivationConfig())

        results = await gm.fast_recall_fallback(
            query="zzqx yonderplasm no evidence",
            group_id="default",
            limit=2,
        )

        assert results == []
        graph.get_episode_by_id.assert_not_awaited()
        graph.get_episode_entities.assert_not_awaited()
        graph.get_episode_cue.assert_not_awaited()
        search.search_episode_cue_records_fast.assert_awaited_once()
        search.search_episode_records_fast.assert_awaited_once()
        search.search_episode_cues.assert_not_awaited()
        search.search_episodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fast_recall_fallback_uses_ready_record_search_when_peer_stalls(self):
        """A slow cue-record lookup should not starve a ready episode-record hit."""
        from engram.graph_manager import GraphManager

        async def slow_cue_records(**_kwargs):
            await asyncio.sleep(0.2)
            return [
                {
                    "episode_id": "ep_slow_cue",
                    "group_id": "default",
                    "cue_text": "slow cue",
                    "first_spans_json": '["slow cue"]',
                    "projection_state": "cue_only",
                    "_score": 0.9,
                }
            ]

        graph = _mock_graph_store()
        graph.get_episode_by_id = AsyncMock()
        graph.get_episode_entities = AsyncMock()
        graph.get_episode_cue = AsyncMock()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            episode_results=[("ep_slow", 0.2)],
            cue_results=[("ep_slow_cue", 0.3)],
        )
        search.search_episode_cue_records_fast = AsyncMock(side_effect=slow_cue_records)
        search.search_episode_records_fast = AsyncMock(
            return_value=[
                {
                    "episode_id": "ep_ready",
                    "group_id": "default",
                    "content": "ready episode",
                    "source": "test",
                    "created_at": "2026-05-26T00:00:00",
                    "projection_state": "cue_only",
                    "_score": 0.7,
                }
            ]
        )
        extractor = AsyncMock()
        gm = GraphManager(graph, activation, search, extractor, cfg=ActivationConfig())

        results = await asyncio.wait_for(
            gm.fast_recall_fallback(
                query="Engram latency",
                group_id="default",
                limit=1,
            ),
            timeout=0.05,
        )

        assert [result["result_type"] for result in results] == ["episode"]
        assert results[0]["episode"]["id"] == "ep_ready"
        graph.get_episode_by_id.assert_not_awaited()
        graph.get_episode_entities.assert_not_awaited()
        graph.get_episode_cue.assert_not_awaited()
        search.search_episode_cue_records_fast.assert_awaited_once()
        search.search_episode_records_fast.assert_awaited_once()
        search.search_episode_cues.assert_not_awaited()
        search.search_episodes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_recall_formats_episode_results(self):
        """GraphManager.recall() formats episode results correctly."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes()
        extractor = AsyncMock()

        # Opt into the depth/graph tier (passage_first_entity_budget >= 1) so
        # entities are reachable in the top-k alongside episodes. The default
        # core tier (budget=0) keeps entities out of the needle top-k.
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=2,
            passage_first_entity_budget=3,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall("test query", group_id="default")

        episode_results = [r for r in results if r.get("result_type") == "episode"]
        entity_results = [r for r in results if "entity" in r]

        # Should have both types
        assert len(entity_results) > 0
        assert len(episode_results) > 0

        # Episode result format check
        ep = episode_results[0]
        assert "episode" in ep
        assert "id" in ep["episode"]
        assert "content" in ep["episode"]
        assert "source" in ep["episode"]
        assert "created_at" in ep["episode"]
        assert "score" in ep
        assert "score_breakdown" in ep
        assert "linked_entities" in ep
        assert ep["result_type"] == "episode"

    @pytest.mark.asyncio
    async def test_recall_entity_results_are_typed_and_prime_neighbors(self):
        """GraphManager.recall() tags entity results and retrieval priming sees them."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("e_neighbor", 0.8, "RELATES_TO", "Thing")],
        )
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[("e1", 0.95)],
            episode_results=[],
            cue_results=[],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            retrieval_priming_enabled=True,
            retrieval_priming_top_n=1,
            retrieval_priming_boost=0.15,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall("test query", group_id="default")

        assert results[0]["result_type"] == "entity"
        assert results[0]["entity"]["id"] == "e1"
        assert gm._priming_buffer["e_neighbor"][0] == pytest.approx(
            cfg.retrieval_priming_boost * 0.8,
        )

    @pytest.mark.asyncio
    async def test_recall_entity_results_preserve_relationship_polarity(self):
        """GraphManager.recall() should preserve relationship polarity in entity results."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_relationships = AsyncMock(
            return_value=[
                Relationship(
                    id="rel_neg",
                    source_id="e1",
                    target_id="e2",
                    predicate="USES",
                    polarity="negative",
                    group_id="default",
                )
            ]
        )
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[("e1", 0.95)],
            episode_results=[],
            cue_results=[],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=False,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall("test query", group_id="default")

        assert results[0]["relationships"][0]["predicate"] == "USES"
        assert results[0]["relationships"][0]["polarity"] == "negative"

    @pytest.mark.asyncio
    async def test_recall_current_state_queries_prefer_entities_over_episodes(self):
        """Current-state queries should suppress historical episode hits
        when entity state exists."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[("e1", 0.95)],
            episode_results=[("ep_1", 0.9)],
            cue_results=[],
        )
        extractor = AsyncMock()

        # Depth tier (passage_first_entity_budget >= 1) so entity state can
        # surface; current-state suppression then drops the historical episode.
        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=False,
            working_memory_enabled=False,
            passage_first_entity_budget=3,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        # Query names the surfaced entity (mock entity is "Test"), so the
        # entity is relevant to the current-state question and episodes are
        # suppressed in favor of entity state.
        results = await gm.recall(
            "Which framework does Test use now?",
            group_id="default",
        )

        assert results
        assert all(result["result_type"] == "entity" for result in results)

    @pytest.mark.asyncio
    async def test_recall_no_record_access_for_episodes(self):
        """GraphManager.recall() does not record access for episodes."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        activation = _mock_activation_store()

        # Only return episodes, no entities
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[("ep_1", 0.9)],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            episode_retrieval_max=3,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await gm.recall("test query", group_id="default")

        # record_access should not be called for episodes
        activation.record_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_recall_formats_cue_results_and_tracks_hits(self):
        """GraphManager.recall() formats cue-backed results and increments hit counts."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[],
            cue_results=[("ep_1", 0.9)],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            cue_recall_max=2,
            cue_recall_hit_threshold=2,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall("test query", group_id="default")

        assert len(results) == 1
        cue_result = results[0]
        assert cue_result["result_type"] == "cue_episode"
        assert cue_result["cue"]["episode_id"] == "ep_1"
        assert cue_result["cue"]["cue_text"] == "mentions: Test"
        assert cue_result["cue"]["supporting_spans"] == ["Test episode content that is quite long"]
        graph.update_episode_cue.assert_awaited()

    @pytest.mark.asyncio
    async def test_recall_skips_merged_episode_results(self):
        """Merged episodes are suppressed even if a backend returns stale hits."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        episodes = {
            "ep_active": Episode(
                id="ep_active",
                content="Active episode content",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            ),
            "ep_merged": Episode(
                id="ep_merged",
                content="Merged episode content",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.MERGED,
                group_id="default",
                created_at=utc_now(),
            ),
        }
        graph.get_episode_by_id = AsyncMock(
            side_effect=lambda episode_id, group_id="default": episodes[episode_id]
        )
        graph.get_episode_entities = AsyncMock(return_value=[])
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[("ep_merged", 0.95), ("ep_active", 0.8)],
            cue_results=[],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=False,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall("test query", group_id="default", limit=5)

        assert [result["episode"]["id"] for result in results] == ["ep_active"]

    @pytest.mark.asyncio
    async def test_recall_promotes_hot_cue_to_scheduled_projection(self):
        """Cue hits past threshold promote an episode into scheduled projection."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        graph.get_episode_cue = AsyncMock(
            return_value=EpisodeCue(
                episode_id="ep_1",
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text="mentions: Test",
                first_spans=["Test episode content that is quite long"],
                hit_count=1,
                route_reason="entity_dense",
            )
        )
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[],
            cue_results=[("ep_1", 0.9)],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            cue_recall_hit_threshold=2,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await gm.recall("test query", group_id="default")

        graph.update_episode.assert_awaited_once_with(
            "ep_1",
            {
                "status": EpisodeStatus.QUEUED.value,
                "error": None,
                "projection_state": EpisodeProjectionState.SCHEDULED.value,
                "last_projection_reason": "cue_recall_hits",
            },
            group_id="default",
        )
        graph.update_episode_cue.assert_awaited_once()
        cue_updates = graph.update_episode_cue.await_args.args[1]
        assert cue_updates["projection_state"] == EpisodeProjectionState.SCHEDULED
        assert cue_updates["route_reason"] == "cue_recall_hits"
        assert cue_updates["hit_count"] == 2

    @pytest.mark.asyncio
    async def test_recall_selected_feedback_can_promote_cue_before_hit_threshold(self):
        """Strong cue selection signals can schedule projection before raw hit threshold."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        cue = EpisodeCue(
            episode_id="ep_1",
            group_id="default",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text="mentions: Test",
            first_spans=["Test episode content that is quite long"],
            hit_count=0,
            selected_count=0,
            policy_score=0.65,
            route_reason="entity_dense",
        )
        graph.get_episode_cue = AsyncMock(return_value=cue)

        async def _update_cue(_episode_id, updates, group_id="default"):
            del group_id
            for key, value in updates.items():
                setattr(cue, key, value.value if hasattr(value, "value") else value)

        graph.update_episode_cue = AsyncMock(side_effect=_update_cue)
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[],
            cue_results=[("ep_1", 0.9)],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            cue_recall_hit_threshold=5,
            cue_policy_learning_enabled=True,
            cue_policy_schedule_threshold=0.8,
            cue_policy_select_weight=0.4,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall(
            "test query",
            group_id="default",
            interaction_type="selected",
        )

        assert results[0]["cue"]["projection_state"] == EpisodeProjectionState.SCHEDULED.value
        assert cue.selected_count == 1
        assert cue.hit_count == 1
        graph.update_episode.assert_awaited()
        graph.update_episode_cue.assert_awaited()

    @pytest.mark.asyncio
    async def test_recall_tracks_cue_near_misses(self):
        """Cue near-misses are exposed and fed back into cue policy state."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        episodes = {
            "ep_1": Episode(
                id="ep_1",
                content="Primary cue episode",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            ),
            "ep_2": Episode(
                id="ep_2",
                content="Near miss cue episode",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            ),
        }
        graph.get_episode_by_id = AsyncMock(
            side_effect=lambda episode_id, group_id="default": episodes[episode_id]
        )
        cues = {
            "ep_1": EpisodeCue(
                episode_id="ep_1",
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text="mentions: Primary",
                first_spans=["Primary cue episode"],
                hit_count=0,
                route_reason="entity_dense",
            ),
            "ep_2": EpisodeCue(
                episode_id="ep_2",
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text="mentions: Near miss",
                first_spans=["Near miss cue episode"],
                near_miss_count=0,
                route_reason="entity_dense",
            ),
        }
        graph.get_episode_cue = AsyncMock(
            side_effect=lambda episode_id, group_id="default": cues[episode_id]
        )

        async def _update_cue(episode_id, updates, group_id="default"):
            del group_id
            cue = cues[episode_id]
            for key, value in updates.items():
                setattr(cue, key, value.value if hasattr(value, "value") else value)

        graph.update_episode_cue = AsyncMock(side_effect=_update_cue)
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes(
            entity_results=[],
            episode_results=[],
            cue_results=[("ep_1", 0.91), ("ep_2", 0.63)],
        )
        extractor = AsyncMock()

        cfg = ActivationConfig(
            episode_retrieval_enabled=False,
            cue_recall_enabled=True,
            cue_recall_max=2,
            conv_context_enabled=True,
            conv_near_miss_enabled=True,
            conv_near_miss_window=1,
            cue_policy_learning_enabled=True,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        await gm.recall("test query", group_id="default", limit=1)

        assert gm._last_near_misses[0]["result_type"] == "cue_episode"
        assert gm._last_near_misses[0]["cue"]["episode_id"] == "ep_2"
        assert cues["ep_2"].near_miss_count == 1
        graph.update_episode_cue.assert_awaited()


# ── Protocol tests ──────────────────────────────────────────────────


class TestSearchIndexProtocol:
    def test_search_episodes_in_protocol(self):
        """SearchIndex protocol includes search_episodes method."""
        import inspect

        from engram.storage.protocols import SearchIndex

        members = dict(inspect.getmembers(SearchIndex))
        assert "search_episodes" in members

    def test_search_episode_cues_in_protocol(self):
        """SearchIndex protocol includes search_episode_cues method."""
        import inspect

        from engram.storage.protocols import SearchIndex

        members = dict(inspect.getmembers(SearchIndex))
        assert "search_episode_cues" in members
