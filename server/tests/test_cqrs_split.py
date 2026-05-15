"""Tests for CQRS split: store_episode + project_episode."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.extractor import (
    MAX_EXTRACTION_INPUT_CHARS,
    ExtractionResult,
    ExtractionStatus,
)
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus


def _make_manager(
    entities: list[Entity] | None = None,
    extract_result=None,
    *,
    content: str = "Test content about Python",
    cfg: ActivationConfig | None = None,
):
    """Create a GraphManager with mocked stores."""
    graph = AsyncMock()
    graph.create_episode = AsyncMock(return_value="ep_test")
    graph.update_episode = AsyncMock()
    graph.get_episode_by_id = AsyncMock(
        return_value=Episode(
            id="ep_test",
            content=content,
            source="test",
            status=EpisodeStatus.QUEUED,
            group_id="default",
        )
    )
    graph.create_entity = AsyncMock(return_value="ent_test")
    graph.get_entity = AsyncMock(return_value=None)
    graph.find_entity_candidates = AsyncMock(return_value=entities or [])
    graph.link_episode_entity = AsyncMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.find_conflicting_relationships = AsyncMock(return_value=[])
    graph.create_relationship = AsyncMock()
    graph.get_stats = AsyncMock(return_value={"entities": 0, "relationships": 0, "episodes": 0})

    activation = AsyncMock()
    activation.record_access = AsyncMock()

    search = AsyncMock()
    search.index_entity = AsyncMock()
    search.index_episode = AsyncMock()

    extractor = AsyncMock()
    if extract_result:
        extractor.extract = AsyncMock(return_value=extract_result)
    else:
        result = MagicMock()
        result.entities = [
            {"name": "Python", "entity_type": "Technology", "summary": "A language"},
        ]
        result.relationships = []
        extractor.extract = AsyncMock(return_value=result)

    cfg = cfg or ActivationConfig()

    manager = GraphManager(
        graph_store=graph,
        activation_store=activation,
        search_index=search,
        extractor=extractor,
        cfg=cfg,
    )
    return manager


@pytest.mark.asyncio
class TestStoreEpisode:
    async def test_returns_episode_id(self):
        manager = _make_manager()
        ep_id = await manager.store_episode("Test content", "default", "test")
        assert ep_id.startswith("ep_")

    async def test_creates_episode_queued(self):
        manager = _make_manager()
        await manager.store_episode("Test content", "default", "test")
        manager._graph.create_episode.assert_called_once()
        call_args = manager._graph.create_episode.call_args[0][0]
        assert call_args.status == EpisodeStatus.QUEUED

    async def test_no_entities_created(self):
        manager = _make_manager()
        await manager.store_episode("Test content about Python", "default", "test")
        manager._graph.create_entity.assert_not_called()

    async def test_no_extraction(self):
        manager = _make_manager()
        await manager.store_episode("Test content", "default", "test")
        manager._extractor.extract.assert_not_called()


@pytest.mark.asyncio
class TestProjectEpisode:
    async def test_delegates_to_projection_service(self):
        manager = _make_manager()
        proposed_entities = [{"name": "Python", "entity_type": "Technology"}]
        proposed_relationships = [{"source": "Python", "predicate": "USED_FOR", "target": "AI"}]
        manager._projection_service.project_episode = AsyncMock()

        await manager.project_episode(
            "ep_test",
            "custom",
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier="fast",
        )

        manager._projection_service.project_episode.assert_awaited_once_with(
            "ep_test",
            group_id="custom",
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier="fast",
        )

    async def test_creates_entities(self):
        manager = _make_manager()
        ep_id = await manager.store_episode("Test content", "default", "test")
        result = await manager.project_episode(ep_id, "default")
        manager._extractor.extract.assert_called_once()
        manager._graph.create_entity.assert_called()
        assert result.outcome == "projected"
        assert result.episode_status == EpisodeStatus.COMPLETED
        assert result.projection_state == EpisodeProjectionState.PROJECTED
        assert result.entity_count == 1
        assert result.relationship_count == 0

    async def test_duplicate_projection_returns_skipped_result(self):
        content = "Test content about Python"
        manager = _make_manager(content=content)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        manager._content_hashes.add(content_hash)

        result = await manager.project_episode("ep_test", "default")

        assert result.outcome == "skipped"
        assert result.reason == "duplicate_content"
        assert result.episode_status == EpisodeStatus.COMPLETED
        assert result.projection_state == EpisodeProjectionState.CUE_ONLY
        manager._extractor.extract.assert_not_called()

    async def test_sets_completed_status(self):
        manager = _make_manager()
        ep_id = await manager.store_episode("Test content", "default", "test")
        await manager.project_episode(ep_id, "default")
        # Should have been called with COMPLETED status
        update_calls = manager._graph.update_episode.call_args_list
        statuses = [
            c[1].get("updates", c[0][1] if len(c[0]) > 1 else {}).get("status")
            for c in update_calls
        ]
        assert EpisodeStatus.COMPLETED.value in statuses

    async def test_raises_on_missing_episode(self):
        manager = _make_manager()
        manager._graph.get_episode_by_id = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match="Episode not found"):
            await manager.project_episode("ep_nonexistent", "default")

    async def test_sets_failed_on_extraction_error(self):
        manager = _make_manager()
        manager._extractor.extract = AsyncMock(side_effect=RuntimeError("LLM down"))
        ep_id = await manager.store_episode("Test content", "default", "test")
        with pytest.raises(RuntimeError, match="LLM down"):
            await manager.project_episode(ep_id, "default")
        # Should have set FAILED status
        update_calls = manager._graph.update_episode.call_args_list
        statuses = [
            c[1].get("updates", c[0][1] if len(c[0]) > 1 else {}).get("status")
            for c in update_calls
        ]
        assert EpisodeStatus.FAILED.value in statuses

    async def test_retryable_extractor_result_sets_retrying_and_does_not_commit_hash(self):
        manager = _make_manager(
            extract_result=ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.API_ERROR,
                error="API down",
            )
        )
        ep_id = await manager.store_episode("Test content", "default", "test")
        with pytest.raises(Exception, match="extractor_api_error"):
            await manager.project_episode(ep_id, "default")

        update_calls = manager._graph.update_episode.call_args_list
        statuses = [
            c[1].get("updates", c[0][1] if len(c[0]) > 1 else {}).get("status")
            for c in update_calls
        ]
        assert EpisodeStatus.RETRYING.value in statuses
        assert manager._content_hashes == set()
        assert manager._content_hashes_inflight == set()

    async def test_long_episode_uses_targeted_projection_plan(self):
        correction = "Correction: I actually moved to Phoenix in 2024 and no longer live in Mesa."
        filler = "Earlier note: I lived in Mesa and commuted to Tempe. "
        content = (filler * 180) + correction
        assert correction not in content[:MAX_EXTRACTION_INPUT_CHARS]

        manager = _make_manager(
            content=content,
            cfg=ActivationConfig(projection_planner_enabled=True),
        )
        ep_id = await manager.store_episode(content, "default", "test")

        await manager.project_episode(ep_id, "default")

        projected_text = manager._extractor.extract.await_args.args[0]
        assert len(projected_text) <= MAX_EXTRACTION_INPUT_CHARS
        assert correction in projected_text


@pytest.mark.asyncio
class TestIngestEpisodeWrapper:
    async def test_returns_episode_id(self):
        manager = _make_manager()
        ep_id = await manager.ingest_episode("Test content", "default", "test")
        assert ep_id.startswith("ep_")

    async def test_extraction_runs(self):
        manager = _make_manager()
        await manager.ingest_episode("Test content", "default", "test")
        manager._extractor.extract.assert_called_once()

    async def test_survives_extraction_failure(self):
        """ingest_episode should not raise even if project_episode fails."""
        manager = _make_manager()
        manager._extractor.extract = AsyncMock(side_effect=RuntimeError("LLM down"))
        ep_id = await manager.ingest_episode("Test content", "default", "test")
        assert ep_id.startswith("ep_")
