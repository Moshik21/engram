"""Offline capture queue replay for the Capture stage."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

DrainQueue = Callable[[], list[dict]]
DedupCheck = Callable[[str], bool]
EpisodeStore = Callable[..., Awaitable[str]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfflineReplayResult:
    """Summary of replaying queued offline captures."""

    replayed: int
    skipped: int
    total: int

    def as_payload(self) -> dict[str, int]:
        return {
            "replayed": self.replayed,
            "skipped": self.skipped,
            "total": self.total,
        }


class OfflineReplayService:
    """Drain offline capture entries into the active brain group."""

    def __init__(
        self,
        *,
        drain_queue: DrainQueue,
        dedup_check: DedupCheck,
        store_episode: EpisodeStore,
    ) -> None:
        self._drain_queue = drain_queue
        self._dedup_check = dedup_check
        self._store_episode = store_episode

    async def replay_queue(self, *, group_id: str) -> OfflineReplayResult:
        entries = self._drain_queue()
        replayed = 0
        skipped = 0

        for entry in entries:
            content = entry.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 10:
                skipped += 1
                continue
            if self._dedup_check(content):
                skipped += 1
                continue
            try:
                await self._store_episode(
                    content=content,
                    group_id=group_id,
                    source=entry.get("source", "offline:replay"),
                    session_id=entry.get("session_id"),
                )
                replayed += 1
            except Exception:
                logger.warning("Failed to replay queue entry", exc_info=True)
                skipped += 1

        return OfflineReplayResult(
            replayed=replayed,
            skipped=skipped,
            total=len(entries),
        )


async def build_api_offline_replay_surface(
    *,
    drain_queue: DrainQueue,
    dedup_check: DedupCheck,
    store_episode: EpisodeStore,
    group_id: str,
) -> dict[str, int | str]:
    """Replay queued offline captures and return the REST acknowledgement payload."""
    replay_service = OfflineReplayService(
        drain_queue=drain_queue,
        dedup_check=dedup_check,
        store_episode=store_episode,
    )
    result = await replay_service.replay_queue(group_id=group_id)
    return {
        "status": "replayed",
        **result.as_payload(),
    }
