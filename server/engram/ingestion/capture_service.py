"""Capture and cue storage service for raw episodes."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.ingestion.projection_state import sync_projection_state
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.storage.protocols import GraphStore, SearchIndex
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)

EventPublisher = Callable[[str, str, dict], None]
DecisionMaterializer = Callable[..., Awaitable[None]]


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
    ) -> None:
        self._graph = graph_store
        self._search = search_index
        self._cfg = cfg
        self._publish = publish_event
        self._materialize_decisions = materialize_decisions

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
        await self._graph.create_episode(episode)
        self._publish_queued(episode)

        if self._cfg.cue_layer_enabled:
            await self._store_episode_cue(episode)

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
        return episode_id

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

    async def _store_episode_cue(self, episode: Episode) -> None:
        try:
            cue = build_episode_cue(episode, self._cfg)
            if cue is not None and hasattr(self._graph, "upsert_episode_cue"):
                await self._graph.upsert_episode_cue(cue)
                if self._cfg.cue_vector_index_enabled and hasattr(
                    self._search,
                    "index_episode_cue",
                ):
                    await self._index_episode_cue_best_effort(cue)
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
                    self._publish(
                        episode.group_id,
                        "episode.projection_scheduled",
                        {
                            "episodeId": episode.id,
                            "reason": cue.route_reason,
                            "projectionState": cue.projection_state.value,
                        },
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

    async def _index_episode_cue_best_effort(self, cue) -> None:
        """Index cue vectors without letting embedding latency block capture."""
        index_cue = getattr(self._search, "index_episode_cue", None)
        if not callable(index_cue):
            return
        timeout_ms = int(getattr(self._cfg, "capture_cue_vector_index_timeout_ms", 0) or 0)
        try:
            if timeout_ms > 0:
                await asyncio.wait_for(index_cue(cue), timeout=timeout_ms / 1000)
            else:
                await index_cue(cue)
        except TimeoutError:
            logger.warning(
                "Timed out indexing episode cue %s after %sms; capture continuing",
                cue.episode_id,
                timeout_ms,
            )
        except Exception:
            logger.warning("Failed to index episode cue %s", cue.episode_id, exc_info=True)


def _is_auto_capture_source(source: str | None) -> bool:
    """Return whether a source is a hook/bootstrap capture that should project later."""
    return bool(source and source.startswith("auto:"))
