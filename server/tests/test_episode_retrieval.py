"""Tests for episode retrieval as retrieval targets (Item 6)."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.retrieval.pipeline import retrieve
from engram.retrieval.scorer import ScoredResult

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
        """Episode retrieval max defaults to 3."""
        cfg = ActivationConfig()
        assert cfg.episode_retrieval_max == 3


# ── FTS5 search_episodes tests ──────────────────────────────────────


class TestFTS5SearchEpisodes:
    @pytest.mark.asyncio
    async def test_search_episodes_returns_results(self):
        """FTS5SearchIndex.search_episodes returns results from episodes_fts."""
        from engram.storage.sqlite.graph import SQLiteGraphStore
        from engram.storage.sqlite.search import FTS5SearchIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            graph = SQLiteGraphStore(db_path)
            await graph.initialize()

            fts = FTS5SearchIndex(db_path)
            await fts.initialize(db=graph.db)

            # Create an episode
            ep = Episode(
                id="ep_test1",
                content="Alice works at TechCorp on machine learning projects",
                source="test",
                status=EpisodeStatus.COMPLETED,
                group_id="default",
                created_at=datetime.utcnow(),
            )
            await graph.create_episode(ep)

            results = await fts.search_episodes("machine learning", group_id="default")
            assert len(results) >= 1
            assert results[0][0] == "ep_test1"
            assert 0.0 <= results[0][1] <= 1.0

            await graph.close()

    @pytest.mark.asyncio
    async def test_search_episodes_empty_query(self):
        """FTS5SearchIndex.search_episodes handles empty query."""
        from engram.storage.sqlite.search import FTS5SearchIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            from engram.storage.sqlite.graph import SQLiteGraphStore

            graph = SQLiteGraphStore(db_path)
            await graph.initialize()

            fts = FTS5SearchIndex(db_path)
            await fts.initialize(db=graph.db)

            results = await fts.search_episodes("", group_id="default")
            assert results == []

            await graph.close()


# ── HybridSearchIndex search_episodes tests ─────────────────────────


class TestHybridSearchEpisodes:
    @pytest.mark.asyncio
    async def test_search_episodes_fts_fallback(self):
        """HybridSearchIndex.search_episodes falls back to FTS5 when no embeddings."""
        from engram.embeddings.provider import NoopProvider
        from engram.storage.sqlite.graph import SQLiteGraphStore
        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            graph = SQLiteGraphStore(db_path)
            await graph.initialize()

            fts = FTS5SearchIndex(db_path)
            vectors = SQLiteVectorStore(db_path)
            provider = NoopProvider()
            hybrid = HybridSearchIndex(fts, vectors, provider)
            await hybrid.initialize(db=graph.db)

            # Create episodes
            ep = Episode(
                id="ep_hybrid1",
                content="Bob researches quantum computing algorithms",
                source="test",
                status=EpisodeStatus.COMPLETED,
                group_id="default",
                created_at=datetime.utcnow(),
            )
            await graph.create_episode(ep)

            results = await hybrid.search_episodes("quantum computing", group_id="default")
            assert len(results) >= 1
            assert results[0][0] == "ep_hybrid1"

            await graph.close()


# ── Pipeline episode tests ──────────────────────────────────────────


def _mock_search_index_with_episodes(
    entity_results=None,
    episode_results=None,
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
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


def _mock_graph_store():
    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
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
            group_id="default",
            created_at=datetime.utcnow(),
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

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=_mock_search_index_with_episodes(),
            cfg=cfg,
        )
        episode_results = [r for r in results if r.result_type == "episode"]
        assert len(episode_results) > 0

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
        """Pipeline handles search_index without search_episodes gracefully."""
        cfg = ActivationConfig(episode_retrieval_enabled=True)

        # Mock without search_episodes attribute
        idx = AsyncMock()
        idx.search = AsyncMock(return_value=[("e1", 0.9)])
        idx.compute_similarity = AsyncMock(return_value={})
        idx._embeddings_enabled = False
        del idx.search_episodes  # Remove the auto-generated attribute

        results = await retrieve(
            query="test query",
            group_id="default",
            graph_store=_mock_graph_store(),
            activation_store=_mock_activation_store(),
            search_index=idx,
            cfg=cfg,
        )
        # Should work without errors, no episodes returned
        episode_results = [r for r in results if r.result_type == "episode"]
        assert len(episode_results) == 0


# ── GraphManager.recall() tests ─────────────────────────────────────


class TestGraphManagerRecallEpisodes:
    @pytest.mark.asyncio
    async def test_recall_formats_episode_results(self):
        """GraphManager.recall() formats episode results correctly."""
        from engram.graph_manager import GraphManager

        graph = _mock_graph_store()
        activation = _mock_activation_store()
        search = _mock_search_index_with_episodes()
        extractor = AsyncMock()

        cfg = ActivationConfig(episode_retrieval_enabled=True, episode_retrieval_max=2)
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


# ── Protocol tests ──────────────────────────────────────────────────


class TestSearchIndexProtocol:
    def test_search_episodes_in_protocol(self):
        """SearchIndex protocol includes search_episodes method."""
        import inspect

        from engram.storage.protocols import SearchIndex

        members = dict(inspect.getmembers(SearchIndex))
        assert "search_episodes" in members
