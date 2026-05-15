"""Primary Recall result materialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engram.models.episode import EpisodeProjectionState
from engram.retrieval.feedback import (
    RecallCueFeedbackRecorder,
    RecallEntityAccessRecorder,
    RecallInteractionRecorder,
)
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer
from engram.storage.protocols import GraphStore


@dataclass
class RecallPrimaryMaterialization:
    """Materialized primary Recall results plus episode IDs already emitted."""

    results: list[dict[str, Any]]
    seen_episode_ids: set[str]


class RecallPrimaryResultMaterializer:
    """Turn scored primary candidates into raw Recall result dictionaries."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        result_builder: RecallResultBuilder,
        cue_feedback_recorder: RecallCueFeedbackRecorder,
        entity_access_recorder: RecallEntityAccessRecorder,
        interaction_recorder: RecallInteractionRecorder,
        working_memory_updater: RecallWorkingMemoryUpdater,
    ) -> None:
        self._graph = graph_store
        self._result_builder = result_builder
        self._cue_feedback_recorder = cue_feedback_recorder
        self._entity_access_recorder = entity_access_recorder
        self._interaction_recorder = interaction_recorder
        self._working_memory_updater = working_memory_updater

    async def materialize(
        self,
        scored_results: Sequence[ScoredResult],
        *,
        group_id: str,
        query: str,
        record_access: bool,
        interaction_type: str | None,
        interaction_source: str,
        now: float,
        working_memory: WorkingMemoryBuffer | None,
    ) -> RecallPrimaryMaterialization:
        results: list[dict[str, Any]] = []
        seen_episode_ids: set[str] = set()

        for scored_result in scored_results:
            if scored_result.result_type in {"episode", "cue_episode"}:
                await self._materialize_episode_result(
                    scored_result,
                    results=results,
                    seen_episode_ids=seen_episode_ids,
                    group_id=group_id,
                    query=query,
                    interaction_type=interaction_type,
                    now=now,
                    working_memory=working_memory,
                )
                continue

            await self._materialize_entity_result(
                scored_result,
                results=results,
                group_id=group_id,
                query=query,
                record_access=record_access,
                interaction_type=interaction_type,
                interaction_source=interaction_source,
                now=now,
                working_memory=working_memory,
            )

        return RecallPrimaryMaterialization(
            results=results,
            seen_episode_ids=seen_episode_ids,
        )

    async def _materialize_episode_result(
        self,
        scored_result: ScoredResult,
        *,
        results: list[dict[str, Any]],
        seen_episode_ids: set[str],
        group_id: str,
        query: str,
        interaction_type: str | None,
        now: float,
        working_memory: WorkingMemoryBuffer | None,
    ) -> None:
        if scored_result.node_id in seen_episode_ids:
            return

        episode = await self._graph.get_episode_by_id(scored_result.node_id, group_id)
        if episode is None or self._is_merged_episode(episode):
            return

        seen_episode_ids.add(episode.id)
        linked_entities = await self._graph.get_episode_entities(
            scored_result.node_id,
            group_id=group_id,
        )
        self._working_memory_updater.add_result(
            working_memory,
            item_id=scored_result.node_id,
            item_type="episode",
            score=scored_result.score,
            query=query,
            now=now,
        )

        if scored_result.result_type == "cue_episode":
            cue = await self._graph.get_episode_cue(scored_result.node_id, group_id)
            if cue is None:
                return
            await self._cue_feedback_recorder.record_cue_feedback(
                episode,
                scored_result.score,
                query,
                interaction_type=interaction_type,
            )
            cue = await self._graph.get_episode_cue(episode.id, group_id) or cue
            results.append(
                self._result_builder.cue_episode_result(
                    episode,
                    cue,
                    scored_result,
                    linked_entities=linked_entities,
                    hit_increment=1,
                )
            )
            return

        results.append(
            self._result_builder.episode_result(
                episode,
                scored_result,
                linked_entities=linked_entities,
            )
        )

    async def _materialize_entity_result(
        self,
        scored_result: ScoredResult,
        *,
        results: list[dict[str, Any]],
        group_id: str,
        query: str,
        record_access: bool,
        interaction_type: str | None,
        interaction_source: str,
        now: float,
        working_memory: WorkingMemoryBuffer | None,
    ) -> None:
        entity = await self._graph.get_entity(scored_result.node_id, group_id)
        if entity is None:
            return

        relationships = await self._graph.get_relationships(
            scored_result.node_id,
            group_id=group_id,
        )
        if record_access:
            await self._entity_access_recorder.record_entity_access(
                entity,
                group_id=group_id,
                query=query,
                source=interaction_source,
                timestamp=now,
            )

        self._working_memory_updater.add_result(
            working_memory,
            item_id=scored_result.node_id,
            item_type="entity",
            score=scored_result.score,
            query=query,
            now=now,
        )
        results.append(
            self._result_builder.entity_result(
                entity,
                relationships,
                scored_result,
            )
        )
        self._interaction_recorder.record_entity_interaction(
            group_id=group_id,
            entity=entity,
            interaction_type=interaction_type,
            source=interaction_source,
            query=query,
            score=scored_result.score,
            recorded_access=record_access,
        )

    @staticmethod
    def _is_merged_episode(episode: object) -> bool:
        return (
            RecallResultBuilder.episode_projection_state_value(episode)
            == EpisodeProjectionState.MERGED.value
        )
