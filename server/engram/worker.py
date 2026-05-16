"""Event-driven background processor for projection work.

Subscribes to episode lifecycle events via EventBus,
scores queued episodes and executes scheduled projections.
Supports adjacent turn batching for auto-captured content.

Three-tier confidence routing (when multi-signal enabled):
  - High confidence (>worker_extract_threshold): extract immediately
  - Low confidence (<worker_skip_threshold): skip immediately
  - Middle: leave as QUEUED for triage phase to batch-process

Worker NEVER calls LLM — all scoring is deterministic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.discourse import classify_discourse
from engram.graph_manager import GraphManager
from engram.ingestion.worker_batching import EpisodeWorkerBatchMerger, PendingEpisode
from engram.ingestion.worker_events import load_full_auto_content, parse_episode_worker_event
from engram.ingestion.worker_routing import EpisodeWorkerProjectionRouter
from engram.ingestion.worker_runtime import EpisodeWorkerRuntimeStores
from engram.ingestion.worker_scoring import EpisodeWorkerScoringService

logger = logging.getLogger(__name__)

# Batching window for adjacent auto-captured turns
_BATCH_WINDOW_SECS = 30


class EpisodeWorker:
    """Background worker that processes queued and scheduled episodes."""

    def __init__(
        self,
        graph_manager: GraphManager,
        cfg: ActivationConfig,
        *,
        stores: EpisodeWorkerRuntimeStores | None = None,
    ) -> None:
        self._manager = graph_manager
        self._cfg = cfg
        runtime_stores = stores or graph_manager.get_episode_worker_runtime_stores()
        self._graph = runtime_stores.graph
        self._activation = runtime_stores.activation
        self._search = runtime_stores.search
        self._batch_merger = EpisodeWorkerBatchMerger(self._graph, self._search, self._cfg)
        self._routing = EpisodeWorkerProjectionRouter(self._graph, self._cfg)
        self._scoring = EpisodeWorkerScoringService(
            graph=self._graph,
            activation=self._activation,
            search=self._search,
            cfg=self._cfg,
        )
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._event_bus: EventBus | None = None
        self._group_id: str | None = None
        # Batch buffer for auto-captured turns
        self._batch_buffer: list[PendingEpisode] = []
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
                worker_event = parse_episode_worker_event(event)
                if worker_event is None:
                    continue

                if worker_event.is_scheduled_projection:
                    if await self._routing.should_skip_projection(
                        worker_event.episode_id,
                        group_id,
                    ):
                        continue
                    logger.debug(
                        "Worker: processing scheduled projection %s",
                        worker_event.episode_id,
                    )
                    await self._process(worker_event.episode_id, group_id)
                    continue

                if await self._routing.should_skip_projection(
                    worker_event.episode_id,
                    group_id,
                    skip_scheduled=True,
                ):
                    continue

                worker_event = await load_full_auto_content(
                    self._graph,
                    worker_event,
                    group_id,
                )

                # Check for system meta-commentary before scoring
                discourse = classify_discourse(worker_event.content)
                if discourse == "system":
                    await self._routing.skip_system_discourse(
                        worker_event.episode_id,
                        group_id,
                    )
                    logger.debug(
                        "Worker: skipped meta-discourse episode %s",
                        worker_event.episode_id,
                    )
                    continue

                # Auto-captured episodes: buffer for adjacent turn batching
                if worker_event.is_auto_turn:
                    pending = PendingEpisode(
                        worker_event.episode_id,
                        worker_event.content,
                        worker_event.source,
                    )
                    self._batch_buffer.append(pending)
                    self._schedule_batch_flush(group_id)
                    continue

                if not self._cfg.triage_enabled:
                    # No triage — extract everything
                    await self._process(worker_event.episode_id, group_id)
                    continue

                decision, signals = await self._scoring.score(
                    worker_event.content,
                    group_id,
                )
                if await self._routing.route_decision(
                    worker_event.episode_id,
                    decision,
                    group_id,
                ):
                    await self._process(worker_event.episode_id, group_id, signals)

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

        merge_result = await self._batch_merger.merge(batch, group_id)

        # Score the merged content
        if self._cfg.triage_enabled:
            decision, signals = await self._scoring.score(
                merge_result.merged_content,
                group_id,
            )
            if await self._routing.route_decision(
                merge_result.primary_episode_id,
                decision,
                group_id,
            ):
                await self._process(merge_result.primary_episode_id, group_id, signals)
        else:
            await self._process(merge_result.primary_episode_id, group_id)

    async def _process_auto_episode(self, ep: PendingEpisode, group_id: str) -> None:
        """Process a single auto-captured episode with triage."""
        if self._cfg.triage_enabled:
            decision, signals = await self._scoring.score(ep.content, group_id)
            if await self._routing.route_decision(ep.episode_id, decision, group_id):
                await self._process(ep.episode_id, group_id, signals)
        else:
            await self._process(ep.episode_id, group_id)

    async def _process(
        self,
        episode_id: str,
        group_id: str,
        signals: Any | None = None,
    ) -> None:
        """Run extraction on an episode, swallowing errors."""
        try:
            await self._manager.project_episode(episode_id, group_id)
            await self._scoring.record_projection_outcome(episode_id, group_id, signals)
        except Exception:
            logger.warning("Worker: extraction failed for %s", episode_id, exc_info=True)
