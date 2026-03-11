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
import re
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.cues import build_episode_cue
from engram.extraction.discourse import classify_discourse
from engram.graph_manager import GraphManager
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.goals import compute_goal_triage_boost, identify_active_goals
from engram.retrieval.triage_policy import TriageDecision, apply_episode_utility_policy
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from engram.retrieval.triage_scorer import TriageScorer

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
    """Background worker that processes queued and scheduled episodes."""

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
        # Multi-signal scorer (lazy-init, shared with triage phase via same config)
        self._scorer: TriageScorer | None = None

    def _get_scorer(self) -> TriageScorer | None:
        """Lazy-init multi-signal scorer."""
        if self._scorer is None and self._cfg.triage_multi_signal_enabled:
            from engram.retrieval.triage_scorer import get_shared_triage_scorer

            self._scorer = get_shared_triage_scorer(self._cfg)
        return self._scorer

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

                if event_type not in {"episode.queued", "episode.projection_scheduled"}:
                    continue

                if event_type == "episode.projection_scheduled":
                    episode_id = event.get("payload", {}).get("episodeId")
                    if not episode_id:
                        continue
                    if await self._should_skip_projection(episode_id, group_id):
                        continue
                    logger.debug("Worker: processing scheduled projection %s", episode_id)
                    await self._process(episode_id, group_id)
                    continue

                episode_id = event.get("payload", {}).get("episode", {}).get("episodeId")
                if not episode_id:
                    continue
                if await self._should_skip_projection(
                    episode_id,
                    group_id,
                    skip_scheduled=True,
                ):
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
                        {
                            "status": "completed",
                            "skipped_meta": True,
                            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
                            "last_projection_reason": "system_discourse",
                        },
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

                decision, signals = await self._score(content, group_id)
                await self._route_episode(episode_id, decision, group_id, signals)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Episode worker error", exc_info=True)

    async def _load_episode(self, episode_id: str, group_id: str) -> Any | None:
        """Best-effort episode lookup for projection-state guards."""
        get_episode = getattr(self._manager._graph, "get_episode_by_id", None)
        if get_episode is None:
            return None
        try:
            episode = await get_episode(episode_id, group_id)
        except Exception:
            return None
        if episode is None:
            return None
        if isinstance(episode, dict):
            return SimpleNamespace(**episode)
        return episode

    async def _should_skip_projection(
        self,
        episode_id: str,
        group_id: str,
        *,
        skip_scheduled: bool = False,
    ) -> bool:
        """Skip duplicate worker work when an episode is already scheduled or done."""
        episode = await self._load_episode(episode_id, group_id)
        if episode is None:
            return False

        state = getattr(episode, "projection_state", None)
        if isinstance(state, EpisodeProjectionState):
            state = state.value

        if state in {
            EpisodeProjectionState.PROJECTING.value,
            EpisodeProjectionState.PROJECTED.value,
            EpisodeProjectionState.DEAD_LETTER.value,
        }:
            return True

        if skip_scheduled and state == EpisodeProjectionState.SCHEDULED.value:
            return True

        return False

    async def _route_episode(
        self,
        episode_id: str,
        decision: TriageDecision,
        group_id: str,
        signals: Any | None = None,
    ) -> None:
        """Three-tier confidence routing: extract / defer / skip."""
        if decision.action == "extract":
            logger.debug(
                "Worker: extract immediately %s (score=%.3f)",
                episode_id,
                decision.score,
            )
            await self._process(episode_id, group_id, signals)
        elif decision.action == "skip":
            logger.debug(
                "Worker: skip %s (score=%.3f)",
                episode_id,
                decision.score,
            )
            await self._manager._graph.update_episode(
                episode_id,
                {
                    "status": "completed",
                    "skipped_triage": True,
                    "projection_state": EpisodeProjectionState.CUE_ONLY.value,
                    "last_projection_reason": "worker_skip_threshold",
                },
                group_id=group_id,
            )
            await self._sync_cue_projection_state(
                episode_id,
                group_id,
                EpisodeProjectionState.CUE_ONLY,
                "worker_skip_threshold",
            )
        else:
            logger.debug(
                "Worker: defer to triage %s (score=%.3f)",
                episode_id,
                decision.score,
            )
            await self._manager._graph.update_episode(
                episode_id,
                {
                    "projection_state": EpisodeProjectionState.SCHEDULED.value,
                    "last_projection_reason": "worker_deferred_to_triage",
                },
                group_id=group_id,
            )
            await self._sync_cue_projection_state(
                episode_id,
                group_id,
                EpisodeProjectionState.SCHEDULED,
                "worker_deferred_to_triage",
            )

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

        await self._rebuild_episode_cue(primary.episode_id, group_id)

        for ep in batch[1:]:
            await self._retire_merged_episode(ep.episode_id, primary.episode_id, group_id)

        # Score the merged content
        if self._cfg.triage_enabled:
            decision, signals = await self._score(merged_content, group_id)
            await self._route_episode(primary.episode_id, decision, group_id, signals)
        else:
            await self._process(primary.episode_id, group_id)

    async def _process_auto_episode(self, ep: _PendingEpisode, group_id: str) -> None:
        """Process a single auto-captured episode with triage."""
        if self._cfg.triage_enabled:
            decision, signals = await self._score(ep.content, group_id)
            await self._route_episode(ep.episode_id, decision, group_id, signals)
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
            await self._record_projection_outcome(episode_id, signals)
        except Exception:
            logger.warning("Worker: extraction failed for %s", episode_id, exc_info=True)

    async def _score(
        self,
        content: str,
        group_id: str = "default",
    ) -> tuple[TriageDecision, Any | None]:
        """Score episode content. Multi-signal scorer preferred, heuristic fallback.

        Worker NEVER calls LLM — that's reserved for triage phase escalation.
        """
        if not content:
            return (
                apply_episode_utility_policy(
                    "",
                    self._cfg,
                    0.0,
                    discourse_class="world",
                    mode="worker" if self._cfg.triage_multi_signal_enabled else "phase",
                    score_source="empty",
                ),
                None,
            )

        discourse_class = classify_discourse(content)

        scorer = self._get_scorer()
        if scorer is not None:
            signals = await scorer.score(
                content=content,
                search_index=getattr(self._manager, "_search", None),
                graph_store=getattr(self._manager, "_graph", None),
                activation_store=getattr(self._manager, "_activation", None),
                group_id=group_id,
            )
            return (
                apply_episode_utility_policy(
                    content,
                    self._cfg,
                    signals.composite,
                    discourse_class=discourse_class,
                    mode="worker",
                    score_source="multi_signal",
                ),
                signals,
            )

        length_score = min(len(content) / 500, 1.0) * 0.25
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", content))
        keyword_score = min(caps / 10, 1.0) * 0.20
        novelty_score = 0.15

        # Emotional salience signal
        emotional_score = 0.0
        if self._cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience

            salience = compute_emotional_salience(content)
            emotional_score = salience.composite * self._cfg.emotional_triage_weight
            base_score = length_score + keyword_score + novelty_score + emotional_score
            if salience.composite >= self._cfg.triage_personal_floor_threshold:
                return (
                    apply_episode_utility_policy(
                        content,
                        self._cfg,
                        max(base_score, self._cfg.triage_personal_floor),
                        discourse_class=discourse_class,
                        mode="phase",
                        score_source="heuristic",
                    ),
                    None,
                )

        base_score = length_score + keyword_score + novelty_score + emotional_score

        # Goal-relevance boost
        if self._cfg.goal_priming_enabled:
            try:
                goals = await identify_active_goals(
                    self._manager._graph,
                    self._manager._activation,
                    group_id,
                    self._cfg,
                )
                base_score += compute_goal_triage_boost(content, goals, self._cfg)
            except Exception:
                logger.debug("Worker: goal boost failed", exc_info=True)

        return (
            apply_episode_utility_policy(
                content,
                self._cfg,
                base_score,
                discourse_class=discourse_class,
                mode="phase",
                score_source="heuristic",
            ),
            None,
        )

    async def _record_projection_outcome(
        self,
        episode_id: str,
        signals: Any | None,
    ) -> None:
        """Feed successful worker projections back into the shared scorer."""
        if signals is None:
            return
        get_episode_entities = getattr(self._manager._graph, "get_episode_entities", None)
        if get_episode_entities is None:
            return
        try:
            entity_ids = await get_episode_entities(episode_id)
        except Exception:
            return
        if isinstance(entity_ids, list):
            scorer = self._get_scorer()
            if scorer is not None:
                scorer.record_outcome(signals, len(entity_ids))

    async def _sync_cue_projection_state(
        self,
        episode_id: str,
        group_id: str,
        state: EpisodeProjectionState,
        reason: str,
    ) -> None:
        """Keep cue metadata aligned with worker routing decisions."""
        if not self._cfg.cue_layer_enabled:
            return
        update_cue = getattr(self._manager._graph, "update_episode_cue", None)
        if update_cue is None or not callable(update_cue):
            return
        try:
            await update_cue(
                episode_id,
                {
                    "projection_state": state,
                    "route_reason": reason,
                },
                group_id=group_id,
            )
        except Exception:
            logger.warning(
                "Worker: failed to sync cue state for %s",
                episode_id,
                exc_info=True,
            )

    async def _rebuild_episode_cue(
        self,
        episode_id: str,
        group_id: str,
    ) -> None:
        """Regenerate cue text after worker-side episode content changes."""
        if not self._cfg.cue_layer_enabled:
            return

        get_episode = getattr(self._manager._graph, "get_episode_by_id", None)
        upsert_cue = getattr(self._manager._graph, "upsert_episode_cue", None)
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

        episode = stored_episode
        if not isinstance(episode, Episode):
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

            episode = Episode(
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

        cue = build_episode_cue(episode, self._cfg)
        if cue is None:
            return

        get_cue = getattr(self._manager._graph, "get_episode_cue", None)
        previous_cue = None
        if get_cue is not None and callable(get_cue):
            try:
                previous_cue = await get_cue(episode_id, group_id)
            except Exception:
                previous_cue = None

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
            await self._manager._graph.update_episode(
                episode_id,
                {
                    "projection_state": cue.projection_state.value,
                    "last_projection_reason": cue.route_reason,
                },
                group_id=group_id,
            )
            if self._cfg.cue_vector_index_enabled and hasattr(
                self._manager._search, "index_episode_cue"
            ):
                await self._manager._search.index_episode_cue(cue)
        except Exception:
            logger.warning(
                "Worker: failed to rebuild cue for %s",
                episode_id,
                exc_info=True,
            )

    async def _retire_merged_episode(
        self,
        episode_id: str,
        primary_episode_id: str,
        group_id: str,
    ) -> None:
        """Retire merged-away cue state so secondary turns stop surfacing."""
        merged_reason = f"merged_into:{primary_episode_id}"

        try:
            await self._manager._graph.update_episode(
                episode_id,
                {
                    "status": "completed",
                    "projection_state": EpisodeProjectionState.MERGED.value,
                    "last_projection_reason": merged_reason,
                },
                group_id=group_id,
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

        get_cue = getattr(self._manager._graph, "get_episode_cue", None)
        update_cue = getattr(self._manager._graph, "update_episode_cue", None)
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
            if self._cfg.cue_vector_index_enabled and hasattr(
                self._manager._search, "index_episode_cue"
            ):
                retired_cue = (
                    cue.model_copy(update=retired_updates)
                    if hasattr(cue, "model_copy")
                    else cue.copy(update=retired_updates)
                )
                await self._manager._search.index_episode_cue(retired_cue)
        except Exception:
            logger.warning(
                "Worker: failed to retire cue for merged episode %s",
                episode_id,
                exc_info=True,
            )
