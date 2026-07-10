"""Tests for primary Recall result materialization."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer


def _score(
    node_id: str,
    *,
    result_type: str,
    score: float = 0.8,
) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.1,
        spreading=0.0,
        edge_proximity=0.2,
        result_type=result_type,
    )


def _episode(
    episode_id: str,
    *,
    projection_state: EpisodeProjectionState = EpisodeProjectionState.CUE_ONLY,
) -> Episode:
    return Episode(
        id=episode_id,
        content=f"{episode_id} content",
        source="test",
        status=EpisodeStatus.COMPLETED,
        projection_state=projection_state,
        group_id="default",
    )


def _materializer(graph: AsyncMock) -> tuple[RecallPrimaryResultMaterializer, dict[str, object]]:
    cue_feedback = AsyncMock()
    access_recorder = AsyncMock()
    interaction_recorder = MagicMock()
    materializer = RecallPrimaryResultMaterializer(
        graph_store=graph,
        result_builder=RecallResultBuilder(ActivationConfig()),
        cue_feedback_recorder=cue_feedback,
        entity_access_recorder=access_recorder,
        interaction_recorder=interaction_recorder,
        working_memory_updater=RecallWorkingMemoryUpdater(),
    )
    return materializer, {
        "cue_feedback": cue_feedback,
        "access_recorder": access_recorder,
        "interaction_recorder": interaction_recorder,
    }


@pytest.mark.asyncio
async def test_materializes_entity_result_with_access_feedback_and_working_memory() -> None:
    graph = AsyncMock()
    entity = Entity(id="ent_react", name="React", entity_type="Technology")
    relationship = Relationship(
        id="rel_1",
        source_id=entity.id,
        target_id="ent_ui",
        predicate="USES",
    )
    graph.get_entity = AsyncMock(return_value=entity)
    graph.get_relationships = AsyncMock(return_value=[relationship])
    materializer, deps = _materializer(graph)
    working_memory = WorkingMemoryBuffer()

    result = await materializer.materialize(
        [_score(entity.id, result_type="entity")],
        group_id="default",
        query="React",
        record_access=True,
        interaction_type="used",
        interaction_source="chat_tool_use",
        now=123.0,
        working_memory=working_memory,
    )

    assert result.results[0]["result_type"] == "entity"
    assert result.results[0]["entity"]["id"] == entity.id
    assert working_memory.size == 1
    deps["access_recorder"].record_entity_access.assert_awaited_once_with(
        entity,
        group_id="default",
        query="React",
        source="chat_tool_use",
        timestamp=123.0,
    )
    deps["interaction_recorder"].record_entity_interaction.assert_called_once_with(
        group_id="default",
        entity=entity,
        interaction_type="used",
        source="chat_tool_use",
        query="React",
        score=0.8,
        recorded_access=True,
    )


@pytest.mark.asyncio
async def test_materializes_cue_episode_and_records_cue_feedback() -> None:
    graph = AsyncMock()
    episode = _episode("ep_cue")
    cue = EpisodeCue(
        episode_id=episode.id,
        group_id="default",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="mentions: React",
    )
    graph.get_episode_by_id = AsyncMock(return_value=episode)
    graph.get_episode_entities = AsyncMock(return_value=["ent_react"])
    graph.get_episode_cue = AsyncMock(return_value=cue)
    materializer, deps = _materializer(graph)
    working_memory = WorkingMemoryBuffer()

    result = await materializer.materialize(
        [_score(episode.id, result_type="cue_episode")],
        group_id="default",
        query="React",
        record_access=False,
        interaction_type="surfaced",
        interaction_source="auto_recall",
        now=123.0,
        working_memory=working_memory,
    )

    assert result.results[0]["result_type"] == "cue_episode"
    assert result.results[0]["episode"]["id"] == episode.id
    assert result.seen_episode_ids == {episode.id}
    assert working_memory.size == 1
    deps["cue_feedback"].record_cue_feedback.assert_awaited_once()
    call = deps["cue_feedback"].record_cue_feedback.await_args
    assert call.args[0] == episode
    assert call.args[1] == 0.8
    assert call.args[2] == "React"
    assert call.kwargs.get("interaction_type") == "surfaced"


@pytest.mark.asyncio
async def test_skips_merged_episodes() -> None:
    graph = AsyncMock()
    graph.get_episode_by_id = AsyncMock(
        return_value=_episode("ep_merged", projection_state=EpisodeProjectionState.MERGED)
    )
    graph.get_episode_entities = AsyncMock()
    materializer, _deps = _materializer(graph)

    result = await materializer.materialize(
        [_score("ep_merged", result_type="episode")],
        group_id="default",
        query="React",
        record_access=False,
        interaction_type=None,
        interaction_source="recall",
        now=123.0,
        working_memory=None,
    )

    assert result.results == []
    assert result.seen_episode_ids == set()
    graph.get_episode_entities.assert_not_called()


@pytest.mark.asyncio
async def test_materialize_entity_bounds_slow_relationships() -> None:
    async def slow_relationships(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return [Relationship(id="rel_slow", source_id="ent_react", target_id="ent_ui")]

    graph = AsyncMock()
    entity = Entity(id="ent_react", name="React", entity_type="Technology")
    graph.get_entity = AsyncMock(return_value=entity)
    graph.get_relationships = AsyncMock(side_effect=slow_relationships)
    materializer, deps = _materializer(graph)
    stage_timings: dict[str, float] = {}

    result = await materializer.materialize(
        [_score(entity.id, result_type="entity")],
        group_id="default",
        query="React",
        record_access=True,
        interaction_type="used",
        interaction_source="chat_tool_use",
        now=123.0,
        working_memory=None,
        graph_timeout_seconds=0.01,
        side_effect_timeout_seconds=0.01,
        stage_timings_ms=stage_timings,
    )

    assert result.results[0]["result_type"] == "entity"
    assert result.results[0]["relationships"] == []
    assert stage_timings["recall_materialize_relationships_timeout"] >= 10
    deps["access_recorder"].record_entity_access.assert_awaited_once()


@pytest.mark.asyncio
async def test_materialize_entity_skips_slow_entity_lookup() -> None:
    async def slow_get_entity(*_args, **_kwargs):
        await asyncio.sleep(0.2)
        return Entity(id="ent_react", name="React", entity_type="Technology")

    graph = AsyncMock()
    graph.get_entity = AsyncMock(side_effect=slow_get_entity)
    graph.get_relationships = AsyncMock(return_value=[])
    materializer, deps = _materializer(graph)
    stage_timings: dict[str, float] = {}

    result = await materializer.materialize(
        [_score("ent_react", result_type="entity")],
        group_id="default",
        query="React",
        record_access=True,
        interaction_type="used",
        interaction_source="chat_tool_use",
        now=123.0,
        working_memory=None,
        graph_timeout_seconds=0.01,
        side_effect_timeout_seconds=0.01,
        stage_timings_ms=stage_timings,
    )

    assert result.results == []
    assert stage_timings["recall_materialize_entity_timeout"] >= 10
    graph.get_relationships.assert_not_called()
    deps["access_recorder"].record_entity_access.assert_not_awaited()
