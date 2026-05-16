"""Projection-state routing helpers for the background episode worker."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from engram.config import ActivationConfig
from engram.ingestion.projection_state import sync_projection_state
from engram.models.episode import EpisodeProjectionState
from engram.retrieval.triage_policy import TriageDecision
from engram.storage.protocols import GraphStore

logger = logging.getLogger(__name__)


class EpisodeWorkerProjectionRouter:
    """Apply worker routing decisions to episode and cue projection state."""

    def __init__(self, graph: GraphStore, cfg: ActivationConfig) -> None:
        self._graph = graph
        self._cfg = cfg

    async def should_skip_projection(
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

    async def route_decision(
        self,
        episode_id: str,
        decision: TriageDecision,
        group_id: str,
    ) -> bool:
        """Apply skip/defer state changes and return whether to project now."""
        if decision.action == "extract":
            logger.debug(
                "Worker: extract immediately %s (score=%.3f)",
                episode_id,
                decision.score,
            )
            return True

        if decision.action == "skip":
            logger.debug(
                "Worker: skip %s (score=%.3f)",
                episode_id,
                decision.score,
            )
            await sync_projection_state(
                self._graph,
                episode_id,
                group_id=group_id,
                state=EpisodeProjectionState.CUE_ONLY,
                reason="worker_skip_threshold",
                episode_updates={
                    "status": "completed",
                    "skipped_triage": True,
                },
                cue_layer_enabled=self._cfg.cue_layer_enabled,
                cue_reason="worker_skip_threshold",
                log_prefix="Worker",
            )
            return False

        logger.debug(
            "Worker: defer to triage %s (score=%.3f)",
            episode_id,
            decision.score,
        )
        await sync_projection_state(
            self._graph,
            episode_id,
            group_id=group_id,
            state=EpisodeProjectionState.SCHEDULED,
            reason="worker_deferred_to_triage",
            cue_layer_enabled=self._cfg.cue_layer_enabled,
            cue_reason="worker_deferred_to_triage",
            log_prefix="Worker",
        )
        return False

    async def skip_system_discourse(self, episode_id: str, group_id: str) -> None:
        """Mark system meta-discourse as cue-only instead of projecting it."""
        await sync_projection_state(
            self._graph,
            episode_id,
            group_id=group_id,
            state=EpisodeProjectionState.CUE_ONLY,
            reason="system_discourse",
            episode_updates={
                "status": "completed",
                "skipped_meta": True,
            },
            cue_layer_enabled=self._cfg.cue_layer_enabled,
            cue_reason="system_discourse",
            log_prefix="Worker",
        )

    async def _load_episode(self, episode_id: str, group_id: str) -> Any | None:
        get_episode = getattr(self._graph, "get_episode_by_id", None)
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
