"""Adjacent-turn batching helpers for the background episode worker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.ingestion.projection_state import sync_projection_state
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.storage.protocols import GraphStore, SearchIndex
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingEpisode:
    """A queued episode awaiting worker-side batching or processing."""

    episode_id: str
    content: str
    source: str
    arrived_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True)
class EpisodeBatchMergeResult:
    """Result of merging adjacent auto-captured turns."""

    primary_episode_id: str
    merged_content: str
    retired_episode_ids: tuple[str, ...]


class EpisodeWorkerBatchMerger:
    """Merge adjacent auto turns and keep cue state aligned."""

    def __init__(
        self,
        graph: GraphStore,
        search: SearchIndex,
        cfg: ActivationConfig,
    ) -> None:
        self._graph = graph
        self._search = search
        self._cfg = cfg

    async def merge(
        self,
        batch: list[PendingEpisode],
        group_id: str,
    ) -> EpisodeBatchMergeResult:
        """Merge multiple adjacent auto turns into the first episode."""
        primary = batch[0]
        merged_content = "\n\n".join(ep.content for ep in batch)

        try:
            await self._graph.update_episode(
                primary.episode_id,
                {"content": merged_content},
                group_id=group_id,
            )
        except Exception:
            logger.warning("Worker: failed to merge batch content", exc_info=True)

        await self.rebuild_episode_cue(primary.episode_id, group_id)

        retired_ids: list[str] = []
        for ep in batch[1:]:
            await self.retire_merged_episode(ep.episode_id, primary.episode_id, group_id)
            retired_ids.append(ep.episode_id)

        return EpisodeBatchMergeResult(
            primary_episode_id=primary.episode_id,
            merged_content=merged_content,
            retired_episode_ids=tuple(retired_ids),
        )

    async def rebuild_episode_cue(
        self,
        episode_id: str,
        group_id: str,
    ) -> None:
        """Regenerate cue text after worker-side episode content changes."""
        if not self._cfg.cue_layer_enabled:
            return

        get_episode = getattr(self._graph, "get_episode_by_id", None)
        upsert_cue = getattr(self._graph, "upsert_episode_cue", None)
        if get_episode is None or upsert_cue is None:
            return

        try:
            stored_episode = await get_episode(episode_id, group_id)
        except Exception:
            logger.warning(
                "Worker: failed to load merged episode %s for cue rebuild",
                episode_id,
                exc_info=True,
            )
            return
        if stored_episode is None:
            return

        episode = self._coerce_episode(stored_episode, episode_id, group_id)
        cue = build_episode_cue(episode, self._cfg)
        if cue is None:
            return

        previous_cue = await self._get_existing_cue(episode_id, group_id)
        if previous_cue is not None:
            cue.hit_count = previous_cue.hit_count
            cue.surfaced_count = previous_cue.surfaced_count
            cue.selected_count = previous_cue.selected_count
            cue.used_count = previous_cue.used_count
            cue.near_miss_count = previous_cue.near_miss_count
            cue.policy_score = max(cue.policy_score, previous_cue.policy_score)
            cue.projection_attempts = previous_cue.projection_attempts
            cue.last_hit_at = previous_cue.last_hit_at
            cue.last_feedback_at = previous_cue.last_feedback_at
            cue.last_projected_at = previous_cue.last_projected_at
            cue.created_at = previous_cue.created_at

        try:
            await upsert_cue(cue)
            await sync_projection_state(
                self._graph,
                episode_id,
                group_id=group_id,
                state=cue.projection_state,
                reason=cue.route_reason,
                cue_layer_enabled=self._cfg.cue_layer_enabled,
                sync_cue=False,
                log_prefix="Worker",
            )
            index_episode_cue = getattr(self._search, "index_episode_cue", None)
            if self._cfg.cue_vector_index_enabled and index_episode_cue is not None:
                await index_episode_cue(cue)
        except Exception:
            logger.warning(
                "Worker: failed to rebuild cue for %s",
                episode_id,
                exc_info=True,
            )

    async def retire_merged_episode(
        self,
        episode_id: str,
        primary_episode_id: str,
        group_id: str,
    ) -> None:
        """Retire merged-away cue state so secondary turns stop surfacing."""
        merged_reason = f"merged_into:{primary_episode_id}"

        try:
            await sync_projection_state(
                self._graph,
                episode_id,
                group_id=group_id,
                state=EpisodeProjectionState.MERGED,
                reason=merged_reason,
                episode_updates={
                    "status": "completed",
                },
                cue_layer_enabled=False,
                log_prefix="Worker",
            )
        except Exception:
            logger.warning(
                "Worker: failed to retire merged episode %s",
                episode_id,
                exc_info=True,
            )
            return

        if not self._cfg.cue_layer_enabled:
            return

        get_cue = getattr(self._graph, "get_episode_cue", None)
        update_cue = getattr(self._graph, "update_episode_cue", None)
        if get_cue is None or update_cue is None:
            return

        try:
            cue = await get_cue(episode_id, group_id)
        except Exception:
            logger.warning(
                "Worker: failed to load merged-away cue for %s",
                episode_id,
                exc_info=True,
            )
            return
        if cue is None:
            return

        retired_updates = {
            "projection_state": EpisodeProjectionState.MERGED,
            "route_reason": merged_reason,
            "cue_text": "",
            "entity_mentions": [],
            "temporal_markers": [],
            "quote_spans": [],
            "contradiction_keys": [],
            "first_spans": [],
        }
        try:
            await update_cue(episode_id, retired_updates, group_id=group_id)
            index_episode_cue = getattr(self._search, "index_episode_cue", None)
            if self._cfg.cue_vector_index_enabled and index_episode_cue is not None:
                retired_cue = (
                    cue.model_copy(update=retired_updates)
                    if hasattr(cue, "model_copy")
                    else cue.copy(update=retired_updates)
                )
                await index_episode_cue(retired_cue)
        except Exception:
            logger.warning(
                "Worker: failed to retire cue for merged episode %s",
                episode_id,
                exc_info=True,
            )

    async def _get_existing_cue(self, episode_id: str, group_id: str) -> Any | None:
        get_cue = getattr(self._graph, "get_episode_cue", None)
        if get_cue is None or not callable(get_cue):
            return None
        try:
            return await get_cue(episode_id, group_id)
        except Exception:
            return None

    @staticmethod
    def _coerce_episode(stored_episode: Any, episode_id: str, group_id: str) -> Episode:
        if isinstance(stored_episode, Episode):
            return stored_episode

        raw_status = getattr(stored_episode, "status", EpisodeStatus.QUEUED)
        if not isinstance(raw_status, EpisodeStatus):
            try:
                raw_status = EpisodeStatus(str(raw_status))
            except ValueError:
                raw_status = EpisodeStatus.QUEUED

        raw_projection_state = getattr(
            stored_episode,
            "projection_state",
            EpisodeProjectionState.QUEUED,
        )
        if not isinstance(raw_projection_state, EpisodeProjectionState):
            try:
                raw_projection_state = EpisodeProjectionState(str(raw_projection_state))
            except ValueError:
                raw_projection_state = EpisodeProjectionState.QUEUED

        return Episode(
            id=getattr(stored_episode, "id", episode_id),
            content=getattr(stored_episode, "content", ""),
            source=getattr(stored_episode, "source", None),
            status=raw_status,
            group_id=getattr(stored_episode, "group_id", group_id),
            session_id=getattr(stored_episode, "session_id", None),
            created_at=getattr(stored_episode, "created_at", None) or utc_now(),
            updated_at=getattr(stored_episode, "updated_at", None),
            error=getattr(stored_episode, "error", None),
            retry_count=getattr(stored_episode, "retry_count", 0),
            processing_duration_ms=getattr(
                stored_episode,
                "processing_duration_ms",
                None,
            ),
            encoding_context=getattr(stored_episode, "encoding_context", None),
            memory_tier=getattr(stored_episode, "memory_tier", "episodic"),
            consolidation_cycles=getattr(
                stored_episode,
                "consolidation_cycles",
                0,
            ),
            entity_coverage=getattr(stored_episode, "entity_coverage", 0.0),
            projection_state=raw_projection_state,
            last_projection_reason=getattr(
                stored_episode,
                "last_projection_reason",
                None,
            ),
            last_projected_at=getattr(stored_episode, "last_projected_at", None),
        )
