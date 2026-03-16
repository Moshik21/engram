"""Tests for episode retrieval as retrieval targets (Item 6)."""

from __future__ import annotations

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


# ── HelixSearchIndex search_episodes tests ──────────────────────────


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

        cfg = ActivationConfig(
            episode_retrieval_enabled=True,
            cue_recall_enabled=False,
            working_memory_enabled=False,
        )
        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        results = await gm.recall(
            "Which framework does Falcon Dashboard use now?",
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

        graph.update_episode.assert_awaited()
        graph.update_episode_cue.assert_awaited()

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
