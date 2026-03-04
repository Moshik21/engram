"""Event-driven background processor for QUEUED episodes.

Subscribes to episode.queued events via EventBus,
scores each episode, and selectively runs extraction.
Supports adjacent turn batching for auto-captured content.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.triage import _llm_judge_score, personal_narrative_boost
from engram.events.bus import EventBus
from engram.extraction.discourse import classify_discourse
from engram.graph_manager import GraphManager

logger = logging.getLogger(__name__)

# Batching window for adjacent auto-captured turns
_BATCH_WINDOW_SECS = 30


class _PendingEpisode:
    """A queued episode awaiting batching or processing."""

    __slots__ = ("episode_id", "content", "source", "arrived_at")

    def __init__(self, episode_id: str, content: str, source: str) -> None:
        self.episode_id = episode_id
        self.content = content
        self.source = source
        self.arrived_at = time.monotonic()


class EpisodeWorker:
    """Background worker that processes QUEUED episodes from EventBus events."""

    def __init__(
        self,
        graph_manager: GraphManager,
        cfg: ActivationConfig,
    ) -> None:
        self._manager = graph_manager
        self._cfg = cfg
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._event_bus: EventBus | None = None
        self._group_id: str | None = None
        # Batch buffer for auto-captured turns
        self._batch_buffer: list[_PendingEpisode] = []
        self._batch_timer: asyncio.Task | None = None

    def start(self, group_id: str, event_bus: EventBus) -> None:
        """Subscribe to events and start processing."""
        if self._task is not None:
            return
        self._event_bus = event_bus
        self._group_id = group_id
        self._queue = event_bus.subscribe(group_id)
        self._task = asyncio.create_task(self._consume(group_id))

    async def stop(self) -> None:
        """Cancel worker task and unsubscribe."""
        if self._batch_timer:
            self._batch_timer.cancel()
            self._batch_timer = None
        # Flush any remaining batch
        if self._batch_buffer and self._group_id:
            await self._flush_batch(self._group_id)
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._event_bus and self._queue and self._group_id:
            self._event_bus.unsubscribe(self._group_id, self._queue)
        self._queue = None

    async def _consume(self, group_id: str) -> None:
        """Main loop: consume events, score, optionally extract."""
        assert self._queue is not None
        while True:
            try:
                event = await self._queue.get()
                event_type = event.get("type", "")

                if event_type != "episode.queued":
                    continue

                episode_id = event.get("payload", {}).get("episode", {}).get("episodeId")
                if not episode_id:
                    continue

                # Get content — for auto: sources, fetch full content from DB
                content = event.get("payload", {}).get("episode", {}).get("content", "")
                source = event.get("payload", {}).get("episode", {}).get("source", "")

                if source.startswith("auto:") and len(content) < 200:
                    full_episode = await self._manager._graph.get_episode_by_id(
                        episode_id, group_id
                    )
                    if full_episode and full_episode.content:
                        content = full_episode.content

                # Check for system meta-commentary before scoring
                discourse = classify_discourse(content)
                if discourse == "system":
                    await self._manager._graph.update_episode(
                        episode_id,
                        {"status": "completed", "skipped_meta": True},
                        group_id=group_id,
                    )
                    logger.debug("Worker: skipped meta-discourse episode %s", episode_id)
                    continue

                # Auto-captured episodes: buffer for adjacent turn batching
                if source.startswith("auto:") and source in ("auto:prompt", "auto:response"):
                    pending = _PendingEpisode(episode_id, content, source)
                    self._batch_buffer.append(pending)
                    self._schedule_batch_flush(group_id)
                    continue

                if not self._cfg.triage_enabled:
                    # No triage — extract everything
                    await self._process(episode_id, group_id)
                    continue

                # Score the episode content
                score = await self._score(content)

                if score >= self._cfg.triage_min_score:
                    await self._process(episode_id, group_id)
                else:
                    # Mark as completed without extraction
                    await self._manager._graph.update_episode(
                        episode_id,
                        {"status": "completed", "skipped_triage": True},
                        group_id=group_id,
                    )

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Episode worker error", exc_info=True)

    def _schedule_batch_flush(self, group_id: str) -> None:
        """Schedule a batch flush after the batching window."""
        if self._batch_timer is not None:
            return  # Timer already running
        self._batch_timer = asyncio.create_task(self._batch_flush_after(group_id))

    async def _batch_flush_after(self, group_id: str) -> None:
        """Wait for batching window, then flush."""
        try:
            await asyncio.sleep(_BATCH_WINDOW_SECS)
            await self._flush_batch(group_id)
        except asyncio.CancelledError:
            pass
        finally:
            self._batch_timer = None

    async def _flush_batch(self, group_id: str) -> None:
        """Process buffered auto-captured episodes, merging adjacent turns."""
        if not self._batch_buffer:
            return

        batch = self._batch_buffer[:]
        self._batch_buffer.clear()

        if len(batch) == 1:
            # Single episode — process normally
            ep = batch[0]
            await self._process_auto_episode(ep, group_id)
            return

        # Multiple episodes — merge content for scoring, process the richest one
        merged_content = "\n\n".join(ep.content for ep in batch)
        primary = batch[0]  # Use first episode as the container

        # Update the primary episode's content with the merged text
        try:
            await self._manager._graph.update_episode(
                primary.episode_id,
                {"content": merged_content},
                group_id=group_id,
            )
        except Exception:
            logger.warning("Worker: failed to merge batch content", exc_info=True)

        # Score the merged content
        if self._cfg.triage_enabled:
            score = await self._score(merged_content)
            if score >= self._cfg.triage_min_score:
                await self._process(primary.episode_id, group_id)
            else:
                await self._manager._graph.update_episode(
                    primary.episode_id,
                    {"status": "completed", "skipped_triage": True},
                    group_id=group_id,
                )
        else:
            await self._process(primary.episode_id, group_id)

        # Mark remaining episodes as completed (merged into primary)
        for ep in batch[1:]:
            try:
                await self._manager._graph.update_episode(
                    ep.episode_id,
                    {"status": "completed"},
                    group_id=group_id,
                )
            except Exception:
                pass

    async def _process_auto_episode(
        self, ep: _PendingEpisode, group_id: str
    ) -> None:
        """Process a single auto-captured episode with triage."""
        if self._cfg.triage_enabled:
            score = await self._score(ep.content)
            if score >= self._cfg.triage_min_score:
                await self._process(ep.episode_id, group_id)
            else:
                await self._manager._graph.update_episode(
                    ep.episode_id,
                    {"status": "completed", "skipped_triage": True},
                    group_id=group_id,
                )
        else:
            await self._process(ep.episode_id, group_id)

    async def _process(self, episode_id: str, group_id: str) -> None:
        """Run extraction on an episode, swallowing errors."""
        try:
            await self._manager.project_episode(episode_id, group_id)
        except Exception:
            logger.warning(
                "Worker: extraction failed for %s", episode_id, exc_info=True
            )

    async def _score(self, content: str) -> float:
        """Score episode content using LLM judge or heuristics."""
        if not content:
            return 0.0

        if self._cfg.triage_llm_judge_enabled:
            result = await asyncio.to_thread(
                _llm_judge_score, content, self._cfg.triage_llm_judge_model,
            )
            return result["score"]

        # Lightweight heuristic scoring (same as TriagePhase)
        length_score = min(len(content) / 500, 1.0) * 0.3
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        keyword_score = min(caps / 10, 1.0) * 0.3
        novelty_score = 0.2
        personal_score = personal_narrative_boost(content, self._cfg)
        return length_score + keyword_score + novelty_score + personal_score
