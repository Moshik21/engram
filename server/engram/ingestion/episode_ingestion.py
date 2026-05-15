"""Capture-to-project orchestration for one-shot episode ingestion."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime

from engram.models.episode import Attachment

EpisodeStore = Callable[..., Awaitable[str]]
EpisodeProjector = Callable[..., Awaitable[object]]


class EpisodeIngestionService:
    """Own the store-then-project compatibility workflow."""

    def __init__(
        self,
        *,
        store_episode: EpisodeStore,
        project_episode: EpisodeProjector,
    ) -> None:
        self._store_episode = store_episode
        self._project_episode = project_episode

    async def ingest_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
        conversation_date: datetime | None = None,
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Store an episode, attempt projection, and return the stored episode ID."""
        episode_id = await self._store_episode(
            content,
            group_id,
            source,
            session_id,
            conversation_date=conversation_date,
            attachments=attachments,
        )
        try:
            await self._project_episode(
                episode_id,
                group_id,
                proposed_entities=proposed_entities,
                proposed_relationships=proposed_relationships,
                model_tier=model_tier,
            )
        except Exception:
            # Projection records FAILED status itself; ingest callers keep the episode ID.
            pass
        return episode_id
