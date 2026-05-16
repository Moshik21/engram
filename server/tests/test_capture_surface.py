from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_surface import (
    build_api_attachment_observe_write_surface,
    build_api_observe_write_surface,
    build_api_remember_write_surface,
    build_mcp_attachment_observe_write_surface,
    build_mcp_observe_write_surface,
    build_mcp_remember_write_surface,
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


@pytest.mark.asyncio
async def test_build_api_observe_write_surface_presents_observed_payload() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")

    response = await build_api_observe_write_surface(
        manager,
        content="Observed operator preference.",
        group_id="native_brain",
        source="dashboard",
        conversation_date="2026-05-15T12:34:56",
    )

    manager.store_episode.assert_awaited_once_with(
        content="Observed operator preference.",
        group_id="native_brain",
        source="dashboard",
        conversation_date=parse_conversation_date("2026-05-15T12:34:56"),
    )
    assert response["status"] == "observed"
    assert response["operation"] == "observe"
    assert response["episodeId"] == "ep_observe"
    assert response["lifecycle"]["stage"] == "cue"


@pytest.mark.asyncio
async def test_build_api_attachment_observe_write_surface_presents_legacy_episode_id() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_image")

    response = await build_api_attachment_observe_write_surface(
        manager,
        data_url="data:image/png;base64,abc",
        mime_type="image/png",
        attachment_kind="image",
        fallback_content="[image: image/png]",
        group_id="native_brain",
        description="control panel sketch",
        source="api",
    )

    manager.store_episode.assert_awaited_once()
    call_kwargs = manager.store_episode.await_args.kwargs
    assert call_kwargs["content"] == "control panel sketch"
    assert call_kwargs["attachments"][0].mime_type == "image/png"
    assert response["status"] == "stored"
    assert response["operation"] == "observe"
    assert response["episode_id"] == "ep_image"
    assert response["lifecycle"]["attachmentKind"] == "image"


@pytest.mark.asyncio
async def test_build_api_remember_write_surface_loads_client_adjudications() -> None:
    manager = MagicMock()
    manager.ingest_episode = AsyncMock(return_value="ep_remember")
    manager.edge_adjudication_client_enabled = MagicMock(return_value=True)
    manager.get_episode_adjudications = AsyncMock(
        return_value=[{"request_id": "adj_1", "candidate_evidence": []}]
    )

    response = await build_api_remember_write_surface(
        manager,
        content="Alice works at Engram.",
        group_id="native_brain",
        source="dashboard",
        conversation_date="2026-05-15T12:34:56",
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        model_tier="opus",
    )

    manager.ingest_episode.assert_awaited_once()
    call_kwargs = manager.ingest_episode.await_args.kwargs
    assert call_kwargs["conversation_date"] == parse_conversation_date(
        "2026-05-15T12:34:56"
    )
    assert call_kwargs["proposed_entities"] == [{"name": "Alice", "entity_type": "Person"}]
    assert call_kwargs["model_tier"] == "opus"
    assert response["status"] == "remembered"
    assert response["operation"] == "remember"
    assert response["adjudicationRequests"][0]["requestId"] == "adj_1"


@pytest.mark.asyncio
async def test_build_mcp_remember_write_surface_runs_capture_project_side_effects() -> None:
    manager = MagicMock()
    manager.ingest_episode = AsyncMock(return_value="ep_remember")
    manager.get_episode_adjudications = AsyncMock(
        return_value=[{"request_id": "adj_1", "candidate_evidence": []}]
    )
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)
    ingest_live_turn = AsyncMock()
    recall_middleware = AsyncMock(
        side_effect=lambda _content, response, **_kwargs: response.update(
            {"recalled_context": {"source": "recall_lite"}}
        )
    )

    response = await build_mcp_remember_write_surface(
        manager,
        content="Alice works at Engram.",
        group_id="native_brain",
        session=session,
        source="mcp",
        conversation_date="2026-05-15T12:34:56",
        proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
        proposed_relationships=[
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"},
        ],
        model_tier="opus",
        activation_cfg=ActivationConfig(
            evidence_extraction_enabled=True,
            edge_adjudication_client_enabled=True,
        ),
        ingest_live_turn=ingest_live_turn,
        recall_middleware=recall_middleware,
    )

    assert session.episode_count == 1
    assert session.last_activity is not None
    manager.ingest_episode.assert_awaited_once()
    call_kwargs = manager.ingest_episode.await_args.kwargs
    assert call_kwargs["group_id"] == "native_brain"
    assert call_kwargs["session_id"] == "sess_1"
    assert call_kwargs["conversation_date"] == parse_conversation_date(
        "2026-05-15T12:34:56"
    )
    assert call_kwargs["proposed_entities"] == [{"name": "Alice", "entity_type": "Person"}]
    assert call_kwargs["model_tier"] == "opus"
    ingest_live_turn.assert_awaited_once_with(manager, "Alice works at Engram.", source="remember")
    recall_middleware.assert_awaited_once()
    assert response["status"] == "stored"
    assert response["operation"] == "remember"
    assert response["adjudication_requests"][0]["request_id"] == "adj_1"
    assert response["recalled_context"] == {"source": "recall_lite"}


@pytest.mark.asyncio
async def test_build_mcp_observe_write_surface_runs_capture_side_effects() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)
    ingest_live_turn = AsyncMock()
    recall_middleware = AsyncMock()

    response = await build_mcp_observe_write_surface(
        manager,
        content="Observed an operator preference.",
        group_id="native_brain",
        session=session,
        source="mcp",
        conversation_date="2026-05-15T12:34:56",
        ingest_live_turn=ingest_live_turn,
        recall_middleware=recall_middleware,
    )

    manager.store_episode.assert_awaited_once_with(
        content="Observed an operator preference.",
        group_id="native_brain",
        source="mcp",
        session_id="sess_1",
        conversation_date=parse_conversation_date("2026-05-15T12:34:56"),
    )
    assert session.episode_count == 1
    ingest_live_turn.assert_awaited_once_with(
        manager,
        "Observed an operator preference.",
        source="observe",
    )
    recall_middleware.assert_awaited_once()
    assert response["operation"] == "observe"
    assert response["lifecycle"]["stage"] == "cue"


@pytest.mark.asyncio
async def test_build_mcp_attachment_observe_write_surface_preserves_attachment_kind() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_image")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)

    response = await build_mcp_attachment_observe_write_surface(
        manager,
        data_url="data:image/png;base64,abc",
        mime_type="image/png",
        attachment_kind="image",
        fallback_content="Image observation",
        group_id="native_brain",
        session=session,
        description="panel sketch",
        source="mcp",
    )

    manager.store_episode.assert_awaited_once()
    call_kwargs = manager.store_episode.await_args.kwargs
    assert call_kwargs["content"] == "panel sketch"
    assert call_kwargs["session_id"] == "sess_1"
    assert call_kwargs["attachments"][0].mime_type == "image/png"
    assert call_kwargs["attachments"][0].data_url == "data:image/png;base64,abc"
    assert response["lifecycle"]["attachment_kind"] == "image"
    assert response["message"] == "Image stored for background processing."
    assert session.episode_count == 1
