"""Episode expansion for recall results."""

from __future__ import annotations

from typing import Any

from engram.config import ActivationConfig
from engram.models.episode import EpisodeProjectionState
from engram.retrieval.result_builder import RecallResultBuilder
from engram.storage.protocols import GraphStore


class RecallEpisodeTraversal:
    """Expand recall results with graph-linked and temporally adjacent episodes."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        cfg: ActivationConfig,
        result_builder: RecallResultBuilder,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._result_builder = result_builder

    async def append_entity_linked_episodes(
        self,
        results: list[dict[str, Any]],
        *,
        group_id: str,
        seen_episode_ids: set[str],
        candidate_entity_scores: list[tuple[str, float]] | None = None,
    ) -> None:
        """Follow top entity hits to linked episodes and append synthetic results."""
        if not self._cfg.entity_episode_traversal_enabled:
            return

        if self._cfg.entity_episode_traversal_source == "candidates" and candidate_entity_scores:
            entity_scores = sorted(
                candidate_entity_scores,
                key=lambda item: item[1],
                reverse=True,
            )[: self._cfg.entity_episode_max_entities]
        else:
            entity_scores = self._top_entity_scores(results)
        for entity_id, entity_score in entity_scores:
            try:
                linked_episode_ids = await self._graph.get_episodes_for_entity(
                    entity_id,
                    group_id=group_id,
                    limit=self._cfg.entity_episode_max_per_entity,
                )
            except Exception:
                continue

            for episode_id in linked_episode_ids:
                if episode_id in seen_episode_ids:
                    continue

                episode = await self._graph.get_episode_by_id(episode_id, group_id)
                if episode is None or self._is_merged_episode(episode):
                    continue

                seen_episode_ids.add(episode.id)
                linked_entities = await self._graph.get_episode_entities(
                    episode.id,
                    group_id=group_id,
                )
                results.append(
                    self._result_builder.synthetic_episode_result(
                        episode,
                        score=entity_score * self._cfg.entity_episode_weight,
                        score_breakdown={
                            "semantic": 0.0,
                            "activation": 0.0,
                            "edge_proximity": 0.0,
                            "exploration_bonus": 0.0,
                            "entity_traversal": True,
                            "parent_entity_id": entity_id,
                        },
                        linked_entities=linked_entities,
                    )
                )

    async def append_temporal_episodes(
        self,
        results: list[dict[str, Any]],
        *,
        group_id: str,
        seen_episode_ids: set[str],
    ) -> None:
        """Append temporally adjacent episodes for top episode hits."""
        if not self._cfg.temporal_contiguity_enabled:
            return

        episode_scores = self._top_episode_scores(results)
        for episode_id, episode_score in episode_scores:
            try:
                adjacent_episodes = await self._graph.get_adjacent_episodes(
                    episode_id,
                    group_id=group_id,
                    limit=self._cfg.temporal_contiguity_max_adjacent,
                )
            except Exception:
                continue

            for adjacent_episode in adjacent_episodes:
                if adjacent_episode.id in seen_episode_ids:
                    continue
                if self._is_merged_episode(adjacent_episode):
                    continue

                seen_episode_ids.add(adjacent_episode.id)
                linked_entities = await self._graph.get_episode_entities(
                    adjacent_episode.id,
                    group_id=group_id,
                )
                results.append(
                    self._result_builder.synthetic_episode_result(
                        adjacent_episode,
                        score=episode_score * self._cfg.temporal_contiguity_weight,
                        score_breakdown={
                            "semantic": 0.0,
                            "activation": 0.0,
                            "edge_proximity": 0.0,
                            "exploration_bonus": 0.0,
                            "temporal_contiguity": True,
                            "parent_episode_id": episode_id,
                        },
                        linked_entities=linked_entities,
                    )
                )

    def _top_entity_scores(self, results: list[dict[str, Any]]) -> list[tuple[str, float]]:
        entity_scores: list[tuple[str, float]] = []
        for result in results:
            if result.get("result_type") != "entity":
                continue
            entity_payload = result.get("entity")
            if isinstance(entity_payload, dict) and entity_payload.get("id"):
                entity_scores.append((entity_payload["id"], self._score(result)))
        entity_scores.sort(key=lambda item: item[1], reverse=True)
        return entity_scores[: self._cfg.entity_episode_max_entities]

    def _top_episode_scores(self, results: list[dict[str, Any]]) -> list[tuple[str, float]]:
        episode_scores: list[tuple[str, float]] = []
        for result in results:
            if result.get("result_type") != "episode":
                continue
            episode_payload = result.get("episode")
            if isinstance(episode_payload, dict) and episode_payload.get("id"):
                episode_scores.append((episode_payload["id"], self._score(result)))
        episode_scores.sort(key=lambda item: item[1], reverse=True)
        return episode_scores[: self._cfg.temporal_contiguity_max_adjacent]

    @staticmethod
    def _score(result: dict[str, Any]) -> float:
        value = result.get("score", 0.0)
        return float(value) if isinstance(value, int | float) else 0.0

    @staticmethod
    def _is_merged_episode(episode: object) -> bool:
        return (
            RecallResultBuilder.episode_projection_state_value(episode)
            == EpisodeProjectionState.MERGED.value
        )
