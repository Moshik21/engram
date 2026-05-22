"""Capture and cue storage service for raw episodes."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.ingestion.cue_index_outbox import CueIndexOutbox
from engram.ingestion.projection_state import sync_projection_state
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.storage.protocols import GraphStore, SearchIndex
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)

EventPublisher = Callable[[str, str, dict], None]
DecisionMaterializer = Callable[..., Awaitable[None]]
StorageCountRecorder = Callable[..., None]


class EpisodeCaptureService:
    """Store raw experience and create the cue layer used by later projection."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
        publish_event: EventPublisher,
        materialize_decisions: DecisionMaterializer,
        record_storage_counts: StorageCountRecorder | None = None,
        cue_index_outbox: CueIndexOutbox | None = None,
    ) -> None:
        self._graph = graph_store
        self._search = search_index
        self._cfg = cfg
        self._publish = publish_event
        self._materialize_decisions = materialize_decisions
        self._record_storage_counts = record_storage_counts
        self._cue_index_outbox = cue_index_outbox or _create_cue_index_outbox(cfg)
        self._cue_index_tasks: set[asyncio.Task[None]] = set()
        self._last_stage_timings_ms: dict[str, float] = {}

    async def store_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
        conversation_date: datetime | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Store a raw episode and optional deterministic cue metadata."""
        stage_timings: dict[str, float] = {}
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = Episode(
            id=episode_id,
            content=content,
            source=source,
            status=EpisodeStatus.QUEUED,
            projection_state=EpisodeProjectionState.QUEUED,
            group_id=group_id,
            session_id=session_id,
            conversation_date=conversation_date,
            created_at=utc_now(),
            attachments=attachments or [],
        )
        capture_started = time_perf_counter()
        await self._graph.create_episode(episode)
        stage_timings["capture_store"] = _elapsed_ms(capture_started)
        self._record_storage_delta(group_id, episodes=1)
        self._publish_queued(episode)

        if self._cfg.cue_layer_enabled:
            await self._store_episode_cue(episode, stage_timings=stage_timings)

        if (
            self._cfg.decision_graph_enabled
            and not _is_auto_capture_source(source)
            and content.strip()
        ):
            try:
                await self._materialize_decisions(
                    content,
                    episode_id=episode_id,
                    group_id=group_id,
                )
            except Exception:
                logger.warning("Failed to materialize conversation decisions", exc_info=True)
        self._last_stage_timings_ms = stage_timings
        return episode_id

    def last_stage_timings(self) -> dict[str, float]:
        """Return stage timings for the latest capture call."""
        return dict(self._last_stage_timings_ms)

    def _publish_queued(self, episode: Episode) -> None:
        created_at = episode.created_at.isoformat() + "Z" if episode.created_at else ""
        self._publish(
            episode.group_id,
            "episode.queued",
            {
                "episode": {
                    "episodeId": episode.id,
                    "content": episode.content[:200] if episode.content else "",
                    "source": episode.source or "unknown",
                    "status": "queued",
                    "createdAt": created_at,
                    "updatedAt": created_at,
                    "entities": [],
                    "factsCount": 0,
                    "processingDurationMs": None,
                    "error": None,
                    "retryCount": 0,
                },
            },
        )

    async def _store_episode_cue(
        self,
        episode: Episode,
        *,
        stage_timings: dict[str, float],
    ) -> None:
        try:
            cue = build_episode_cue(episode, self._cfg)
            if cue is not None and hasattr(self._graph, "upsert_episode_cue"):
                cue_store_started = time_perf_counter()
                await self._graph.upsert_episode_cue(cue)
                stage_timings["cue_store"] = _elapsed_ms(cue_store_started)
                self._record_storage_delta(episode.group_id, cues=1)
                if self._cfg.cue_vector_index_enabled and hasattr(
                    self._search,
                    "index_episode_cue",
                ):
                    enqueue_started = time_perf_counter()
                    self._enqueue_episode_cue_index(cue, stage_timings)
                    stage_timings["cue_index_enqueue"] = _elapsed_ms(enqueue_started)
                await sync_projection_state(
                    self._graph,
                    episode.id,
                    group_id=episode.group_id,
                    state=cue.projection_state,
                    reason=cue.route_reason,
                    cue_layer_enabled=self._cfg.cue_layer_enabled,
                    sync_cue=False,
                    log_prefix="Capture",
                )
                self._publish(
                    episode.group_id,
                    "episode.cued",
                    {
                        "episodeId": episode.id,
                        "projectionState": cue.projection_state.value,
                        "cueScore": cue.cue_score,
                        "projectionPriority": cue.projection_priority,
                        "routeReason": cue.route_reason,
                    },
                )
                if cue.projection_state == EpisodeProjectionState.SCHEDULED:
                    projection_enqueue_started = time_perf_counter()
                    self._publish(
                        episode.group_id,
                        "episode.projection_scheduled",
                        {
                            "episodeId": episode.id,
                            "reason": cue.route_reason,
                            "projectionState": cue.projection_state.value,
                        },
                    )
                    stage_timings["projection_enqueue"] = _elapsed_ms(
                        projection_enqueue_started,
                    )
            elif cue is None:
                await sync_projection_state(
                    self._graph,
                    episode.id,
                    group_id=episode.group_id,
                    state=EpisodeProjectionState.CUE_ONLY,
                    reason="system_discourse",
                    cue_layer_enabled=self._cfg.cue_layer_enabled,
                    sync_cue=False,
                    log_prefix="Capture",
                )
        except Exception:
            logger.warning("Failed to generate/store episode cue", exc_info=True)

    async def _index_episode_cue_best_effort(
        self,
        cue,
        stage_timings: dict[str, float] | None = None,
    ) -> None:
        """Index cue vectors without letting embedding latency block capture."""
        index_cue = getattr(self._search, "index_episode_cue", None)
        if not callable(index_cue):
            self._mark_cue_index_failed(cue, "index_episode_cue unavailable")
            return
        timeout_ms = int(getattr(self._cfg, "capture_cue_vector_index_timeout_ms", 0) or 0)
        index_started = time_perf_counter()
        indexed = False
        error: str | None = None
        try:
            if timeout_ms > 0:
                await asyncio.wait_for(index_cue(cue), timeout=timeout_ms / 1000)
            else:
                await index_cue(cue)
            indexed = True
        except TimeoutError:
            error = f"timeout after {timeout_ms}ms"
            logger.warning(
                "Timed out indexing episode cue %s after %sms; capture continuing",
                cue.episode_id,
                timeout_ms,
            )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            logger.warning("Failed to index episode cue %s", cue.episode_id, exc_info=True)
        finally:
            if stage_timings is not None:
                stage_timings["cue_index"] = _elapsed_ms(index_started)
            if indexed:
                self._mark_cue_index_done(cue)
            elif error is not None:
                self._mark_cue_index_failed(cue, error)

    def _enqueue_episode_cue_index(
        self,
        cue: Any,
        stage_timings: dict[str, float],
    ) -> None:
        """Schedule cue vector indexing after capture has acknowledged the write."""
        self._persist_cue_index_work(cue, stage_timings)
        try:
            task = asyncio.create_task(
                self._index_episode_cue_best_effort(cue, stage_timings),
            )
        except RuntimeError:
            logger.warning(
                "No running event loop for episode cue %s indexing; skipping",
                cue.episode_id,
            )
            return
        self._cue_index_tasks.add(task)
        task.add_done_callback(self._cue_index_tasks.discard)

    async def drain_cue_indexing(self) -> None:
        """Wait for queued cue-indexing tasks; intended for tests and shutdown hooks."""
        if not self._cue_index_tasks:
            return
        await asyncio.gather(*tuple(self._cue_index_tasks), return_exceptions=True)

    async def drain_cue_index_outbox(self, *, limit: int | None = None) -> int:
        """Replay durable cue-indexing work from previous process lifetimes."""
        if self._cue_index_outbox is None:
            return 0
        replay_limit = limit or int(getattr(self._cfg, "cue_index_outbox_replay_limit", 100))
        items = self._cue_index_outbox.pending(limit=replay_limit)
        for item in items:
            await self._index_episode_cue_best_effort(item.cue)
        return len(items)

    def cue_index_outbox_pending_count(self) -> int:
        """Return queued durable cue-indexing work, if enabled."""
        if self._cue_index_outbox is None:
            return 0
        try:
            return self._cue_index_outbox.pending_count()
        except Exception:
            logger.debug("failed to count cue index outbox rows", exc_info=True)
            return 0

    def _record_storage_delta(self, group_id: str, **deltas: int) -> None:
        if self._record_storage_counts is None:
            return
        try:
            self._record_storage_counts(group_id, **deltas)
        except Exception:
            logger.debug("failed to record storage count delta", exc_info=True)

    def _persist_cue_index_work(
        self,
        cue: Any,
        stage_timings: dict[str, float],
    ) -> None:
        if self._cue_index_outbox is None:
            return
        started = time_perf_counter()
        try:
            self._cue_index_outbox.enqueue(cue)
        except Exception:
            logger.warning(
                "Failed to enqueue cue %s for durable vector indexing",
                getattr(cue, "episode_id", "unknown"),
                exc_info=True,
            )
        finally:
            stage_timings["cue_index_outbox_enqueue"] = _elapsed_ms(started)

    def _mark_cue_index_done(self, cue: Any) -> None:
        if self._cue_index_outbox is None:
            return
        try:
            self._cue_index_outbox.mark_done(
                episode_id=cue.episode_id,
                group_id=cue.group_id,
            )
        except Exception:
            logger.debug("failed to mark cue index outbox row done", exc_info=True)

    def _mark_cue_index_failed(self, cue: Any, error: str) -> None:
        if self._cue_index_outbox is None:
            return
        try:
            self._cue_index_outbox.mark_failed(
                episode_id=cue.episode_id,
                group_id=cue.group_id,
                error=error,
            )
        except Exception:
            logger.debug("failed to mark cue index outbox row failed", exc_info=True)


def _is_auto_capture_source(source: str | None) -> bool:
    """Return whether a source is a hook/bootstrap capture that should project later."""
    return bool(source and source.startswith("auto:"))


def _create_cue_index_outbox(cfg: ActivationConfig) -> CueIndexOutbox | None:
    if not getattr(cfg, "cue_index_outbox_enabled", False):
        return None
    configured = getattr(cfg, "cue_index_outbox_path", "") or ""
    if not configured:
        return None
    return CueIndexOutbox(Path(configured))


def time_perf_counter() -> float:
    return asyncio.get_running_loop().time()


def _elapsed_ms(started: float) -> float:
    return round((time_perf_counter() - started) * 1000, 4)
