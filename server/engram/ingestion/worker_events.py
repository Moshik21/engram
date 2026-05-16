"""Event parsing helpers for the background episode worker."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from engram.storage.protocols import GraphStore

EPISODE_QUEUED_EVENT = "episode.queued"
EPISODE_PROJECTION_SCHEDULED_EVENT = "episode.projection_scheduled"


@dataclass(frozen=True)
class EpisodeWorkerEvent:
    """Normalized worker event payload."""

    kind: str
    episode_id: str
    content: str = ""
    source: str = ""

    @property
    def is_scheduled_projection(self) -> bool:
        return self.kind == EPISODE_PROJECTION_SCHEDULED_EVENT

    @property
    def is_auto_turn(self) -> bool:
        return self.source in {"auto:prompt", "auto:response"}


def parse_episode_worker_event(event: dict[str, Any]) -> EpisodeWorkerEvent | None:
    """Parse an EventBus payload into the worker's route-neutral event shape."""
    event_type = event.get("type", "")
    payload = event.get("payload", {})
    if event_type == EPISODE_PROJECTION_SCHEDULED_EVENT:
        episode_id = payload.get("episodeId")
        if not episode_id:
            return None
        return EpisodeWorkerEvent(kind=event_type, episode_id=episode_id)

    if event_type != EPISODE_QUEUED_EVENT:
        return None

    episode = payload.get("episode", {})
    episode_id = episode.get("episodeId")
    if not episode_id:
        return None
    return EpisodeWorkerEvent(
        kind=event_type,
        episode_id=episode_id,
        content=episode.get("content", ""),
        source=episode.get("source", ""),
    )


async def load_full_auto_content(
    graph: GraphStore,
    event: EpisodeWorkerEvent,
    group_id: str,
) -> EpisodeWorkerEvent:
    """Load full stored content for compact auto-capture event payloads."""
    if not event.source.startswith("auto:") or len(event.content) >= 200:
        return event

    full_episode = await graph.get_episode_by_id(event.episode_id, group_id)
    if full_episode and full_episode.content:
        return replace(event, content=full_episode.content)
    return event
