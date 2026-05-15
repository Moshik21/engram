"""Shared projection-state synchronization for episodes and cue metadata."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from engram.models.episode import EpisodeProjectionState

logger = logging.getLogger(__name__)


async def sync_projection_state(
    graph_store: Any,
    episode_id: str,
    *,
    group_id: str,
    state: EpisodeProjectionState,
    reason: str | None = None,
    last_projected_at: datetime | None = None,
    episode_updates: Mapping[str, Any] | None = None,
    cue_layer_enabled: bool = True,
    cue_reason: str | None = None,
    cue_updates: Mapping[str, Any] | None = None,
    sync_cue: bool = True,
    log_prefix: str = "Projection state",
) -> None:
    """Update episode projection state and keep cue metadata aligned when possible."""
    episode_payload: dict[str, Any] = dict(episode_updates or {})
    episode_payload["projection_state"] = state.value
    if reason is not None:
        episode_payload["last_projection_reason"] = reason
    if last_projected_at is not None:
        episode_payload["last_projected_at"] = last_projected_at.isoformat()

    await graph_store.update_episode(episode_id, episode_payload, group_id=group_id)

    if not cue_layer_enabled or not sync_cue:
        return
    update_cue = getattr(graph_store, "update_episode_cue", None)
    if update_cue is None or not callable(update_cue):
        return

    cue_payload: dict[str, Any] = {"projection_state": state}
    if cue_reason is not None:
        cue_payload["route_reason"] = cue_reason
    cue_payload.update(cue_updates or {})

    try:
        await update_cue(episode_id, cue_payload, group_id=group_id)
    except Exception:
        logger.warning(
            "%s: failed to sync cue state for %s",
            log_prefix,
            episode_id,
            exc_info=True,
        )
