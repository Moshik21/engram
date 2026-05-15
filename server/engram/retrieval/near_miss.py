"""Near-miss recall result helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engram.models.episode import Episode, EpisodeProjectionState
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.feedback import RecallCueFeedbackRecorder
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult
from engram.storage.protocols import GraphStore


@dataclass(frozen=True)
class CueNearMissContext:
    """Episode/cue pair eligible for near-miss feedback."""

    episode: Episode
    cue: EpisodeCue


class RecallNearMissBuilder:
    """Build near-miss entries while leaving feedback side effects to callers."""

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph = graph_store

    async def entity_near_miss(
        self,
        scored_result: ScoredResult,
        *,
        group_id: str,
    ) -> dict[str, Any] | None:
        entity = await self._graph.get_entity(scored_result.node_id, group_id)
        if entity is None:
            return None
        return {
            "result_type": "entity",
            "entity": {"name": entity.name, "type": entity.entity_type},
            "score": round(scored_result.score, 4),
        }

    async def cue_context(
        self,
        scored_result: ScoredResult,
        *,
        group_id: str,
    ) -> CueNearMissContext | None:
        episode = await self._graph.get_episode_by_id(scored_result.node_id, group_id)
        cue = await self._graph.get_episode_cue(scored_result.node_id, group_id)
        if episode is None or not isinstance(cue, EpisodeCue):
            return None
        if self._is_merged_episode(episode):
            return None
        return CueNearMissContext(episode=episode, cue=cue)

    @staticmethod
    def cue_near_miss(cue: EpisodeCue, scored_result: ScoredResult) -> dict[str, Any]:
        return {
            "result_type": "cue_episode",
            "cue": RecallResultBuilder.cue_result_payload(cue),
            "score": round(scored_result.score, 4),
        }

    async def episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        cue = await self._graph.get_episode_cue(episode_id, group_id)
        return cue if isinstance(cue, EpisodeCue) else None

    @staticmethod
    def _is_merged_episode(episode: object) -> bool:
        return (
            RecallResultBuilder.episode_projection_state_value(episode)
            == EpisodeProjectionState.MERGED.value
        )


class RecallNearMissMaterializer:
    """Materialize near-miss entries and apply cue near-miss feedback."""

    def __init__(
        self,
        *,
        near_miss_builder: RecallNearMissBuilder,
        cue_feedback_recorder: RecallCueFeedbackRecorder,
    ) -> None:
        self._near_miss_builder = near_miss_builder
        self._cue_feedback_recorder = cue_feedback_recorder

    async def materialize(
        self,
        scored_results: Sequence[ScoredResult],
        *,
        group_id: str,
        query: str,
        interaction_type: str | None,
    ) -> list[dict[str, Any]]:
        near_misses: list[dict[str, Any]] = []
        for scored_result in scored_results:
            if scored_result.result_type == "entity":
                near_miss = await self._near_miss_builder.entity_near_miss(
                    scored_result,
                    group_id=group_id,
                )
                if near_miss:
                    near_misses.append(near_miss)
                continue

            if scored_result.result_type != "cue_episode":
                continue

            near_miss_context = await self._near_miss_builder.cue_context(
                scored_result,
                group_id=group_id,
            )
            if near_miss_context is None:
                continue

            await self._cue_feedback_recorder.record_cue_feedback(
                near_miss_context.episode,
                scored_result.score,
                query,
                interaction_type=interaction_type,
                near_miss=True,
            )
            cue = (
                await self._near_miss_builder.episode_cue(
                    near_miss_context.episode.id,
                    group_id,
                )
                or near_miss_context.cue
            )
            near_misses.append(
                self._near_miss_builder.cue_near_miss(cue, scored_result)
            )
        return near_misses
