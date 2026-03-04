"""Event-driven background processor for QUEUED episodes.

Subscribes to episode.queued events via EventBus,
scores each episode, and selectively runs extraction.
"""

from __future__ import annotations

import asyncio
import logging
import re

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.discourse import classify_discourse
from engram.graph_manager import GraphManager

logger = logging.getLogger(__name__)


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

                # Check for system meta-commentary before scoring
                content = event.get("payload", {}).get("episode", {}).get("content", "")
                discourse = classify_discourse(content)
                if discourse == "system":
                    await self._manager._graph.update_episode(
                        episode_id,
                        {"status": "completed", "skipped_meta": True},
                        group_id=group_id,
                    )
                    logger.debug("Worker: skipped meta-discourse episode %s", episode_id)
                    continue

                if not self._cfg.triage_enabled:
                    # No triage — extract everything
                    await self._process(episode_id, group_id)
                    continue

                # Score the episode content
                score = self._score(content)

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

    async def _process(self, episode_id: str, group_id: str) -> None:
        """Run extraction on an episode, swallowing errors."""
        try:
            await self._manager.project_episode(episode_id, group_id)
        except Exception:
            logger.warning(
                "Worker: extraction failed for %s", episode_id, exc_info=True
            )

    @staticmethod
    def _score(content: str) -> float:
        """Lightweight scoring (same heuristics as TriagePhase)."""
        if not content:
            return 0.0
        # Length signal
        length_score = min(len(content) / 500, 1.0) * 0.3
        # Keyword density (capitalized words, numbers)
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        keyword_score = min(caps / 10, 1.0) * 0.3
        # Base novelty (no DB lookup in hot path — assume moderate)
        novelty_score = 0.2
        return length_score + keyword_score + novelty_score
