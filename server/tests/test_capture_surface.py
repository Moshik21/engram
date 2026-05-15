from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.ingestion.capture_surface import (
    build_observation_attachment,
    ingest_projecting_memory,
    parse_conversation_date,
    store_observation,
)


def test_parse_conversation_date_accepts_iso_and_ignores_bad_values() -> None:
    parsed = parse_conversation_date("2026-05-15T12:34:56")

    assert isinstance(parsed, datetime)
    assert parse_conversation_date("not-a-date") is None
    assert parse_conversation_date(None) is None


def test_build_observation_attachment_preserves_payload() -> None:
    attachment = build_observation_attachment(
        mime_type="image/png",
        data_url="data:image/png;base64,abc",
        description="lathe setup",
    )

    assert attachment.mime_type == "image/png"
    assert attachment.data_url == "data:image/png;base64,abc"
    assert attachment.description == "lathe setup"


@pytest.mark.asyncio
async def test_store_observation_forwards_optional_capture_fields() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    conv_dt = parse_conversation_date("2026-05-15T12:34:56")

    episode_id = await store_observation(
        manager,
        content="Observed operator preference.",
        group_id="native_brain",
        source="mcp",
        session_id="sess_1",
        conversation_date=conv_dt,
        pass_session_id=True,
        pass_conversation_date=True,
    )

    assert episode_id == "ep_observe"
    manager.store_episode.assert_awaited_once_with(
        content="Observed operator preference.",
        group_id="native_brain",
        source="mcp",
        session_id="sess_1",
        conversation_date=conv_dt,
    )


@pytest.mark.asyncio
async def test_ingest_projecting_memory_can_preserve_empty_attachment_arg() -> None:
    manager = MagicMock()
    manager.ingest_episode = AsyncMock(return_value="ep_remember")

    episode_id = await ingest_projecting_memory(
        manager,
        content="Alice works at Engram.",
        group_id="native_brain",
        source="mcp",
        session_id="sess_1",
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        proposed_relationships=[
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"},
        ],
        model_tier="opus",
        attachments=None,
        pass_session_id=True,
        pass_attachments=True,
    )

    assert episode_id == "ep_remember"
    manager.ingest_episode.assert_awaited_once_with(
        content="Alice works at Engram.",
        group_id="native_brain",
        source="mcp",
        conversation_date=None,
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        proposed_relationships=[
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"},
        ],
        model_tier="opus",
        session_id="sess_1",
        attachments=None,
    )
