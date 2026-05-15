"""Post-materialization Recall processing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engram.retrieval.confidence import RecallConfidenceApplier
from engram.retrieval.context import (
    ConversationContext,
    RecallConversationFingerprintRecorder,
)
from engram.retrieval.episode_traversal import RecallEpisodeTraversal
from engram.retrieval.near_miss import RecallNearMissMaterializer
from engram.retrieval.priming import RecallPrimingUpdater
from engram.retrieval.result_selection import filter_current_state_results
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer


@dataclass
class RecallPostProcessResult:
    """Final Recall results plus materialized near misses."""

    results: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]


class RecallPostProcessor:
    """Run Recall side effects after primary result materialization."""

    def __init__(
        self,
        *,
        episode_traversal: RecallEpisodeTraversal,
        working_memory_updater: RecallWorkingMemoryUpdater,
        priming_updater: RecallPrimingUpdater,
        near_miss_materializer: RecallNearMissMaterializer,
        confidence_applier: RecallConfidenceApplier,
        fingerprint_recorder: RecallConversationFingerprintRecorder,
    ) -> None:
        self._episode_traversal = episode_traversal
        self._working_memory_updater = working_memory_updater
        self._priming_updater = priming_updater
        self._near_miss_materializer = near_miss_materializer
        self._confidence_applier = confidence_applier
        self._fingerprint_recorder = fingerprint_recorder

    async def process(
        self,
        results: list[dict[str, Any]],
        *,
        group_id: str,
        query: str,
        seen_episode_ids: set[str],
        near_miss_results: Sequence[ScoredResult],
        now: float,
        working_memory: WorkingMemoryBuffer | None,
        priming_buffer: dict[str, tuple[float, float]],
        conv_context: ConversationContext | None,
        interaction_type: str | None,
        interaction_source: str,
    ) -> RecallPostProcessResult:
        """Apply episode expansion, recall side effects, and final scoring."""
        await self._episode_traversal.append_entity_linked_episodes(
            results,
            group_id=group_id,
            seen_episode_ids=seen_episode_ids,
        )
        await self._episode_traversal.append_temporal_episodes(
            results,
            group_id=group_id,
            seen_episode_ids=seen_episode_ids,
        )

        filtered_results = filter_current_state_results(query, results)

        self._working_memory_updater.add_query(
            working_memory,
            query=query,
            now=now,
        )
        await self._priming_updater.update(
            filtered_results,
            group_id=group_id,
            priming_buffer=priming_buffer,
        )
        near_misses = await self._near_miss_materializer.materialize(
            near_miss_results,
            group_id=group_id,
            query=query,
            interaction_type=interaction_type,
        )
        await self._confidence_applier.apply(query=query, results=filtered_results)
        await self._fingerprint_recorder.record_recall_query(
            conv_context,
            query,
            interaction_source=interaction_source,
        )

        return RecallPostProcessResult(
            results=filtered_results,
            near_misses=near_misses,
        )
