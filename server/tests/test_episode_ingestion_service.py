"""Tests for one-shot episode ingestion orchestration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.ingestion.episode_ingestion import EpisodeIngestionService
from engram.models.episode import Attachment


@pytest.mark.asyncio
async def test_ingestion_service_stores_then_projects_with_proposals():
    store_episode = AsyncMock(return_value="ep_123")
    project_episode = AsyncMock()
    service = EpisodeIngestionService(
        store_episode=store_episode,
        project_episode=project_episode,
    )
    conversation_date = datetime(2026, 5, 15, 12, 0, 0)
    attachments = [
        Attachment(
            mime_type="text/plain",
            data_url="file:///tmp/note.txt",
            description="hello",
        )
    ]

    episode_id = await service.ingest_episode(
        content="Alice works at Google",
        group_id="brain",
        source="dashboard",
        session_id="session_1",
        conversation_date=conversation_date,
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        proposed_relationships=[
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
        ],
        model_tier="opus",
        attachments=attachments,
    )

    assert episode_id == "ep_123"
    store_episode.assert_awaited_once_with(
        "Alice works at Google",
        "brain",
        "dashboard",
        "session_1",
        conversation_date=conversation_date,
        attachments=attachments,
    )
    project_episode.assert_awaited_once_with(
        "ep_123",
        "brain",
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        proposed_relationships=[
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
        ],
        model_tier="opus",
    )


@pytest.mark.asyncio
async def test_ingestion_service_returns_episode_id_when_projection_fails():
    store_episode = AsyncMock(return_value="ep_failed_projection")
    project_episode = AsyncMock(side_effect=RuntimeError("projection failed"))
    service = EpisodeIngestionService(
        store_episode=store_episode,
        project_episode=project_episode,
    )

    episode_id = await service.ingest_episode("content", group_id="brain")

    assert episode_id == "ep_failed_projection"
    project_episode.assert_awaited_once()
