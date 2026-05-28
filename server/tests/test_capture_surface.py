from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion import capture_surface
from engram.ingestion.capture_surface import (
    attach_api_capture_diagnostics,
    attach_mcp_capture_diagnostics,
    build_api_attachment_observe_write_surface,
    build_api_auto_observe_surface,
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


def _latest_memory_operation(manager: MagicMock) -> tuple[str, Any]:
    group_id, sample = manager.record_memory_operation.call_args.args
    return group_id, sample


def test_parse_conversation_date_accepts_iso_and_ignores_bad_values() -> None:
    parsed = parse_conversation_date("2026-05-15T12:34:56")

    assert isinstance(parsed, datetime)
    assert parse_conversation_date("not-a-date") is None
    assert parse_conversation_date(None) is None


def test_cache_recent_observation_keeps_rolling_session_packets() -> None:
    manager = MagicMock()
    manager.get_recent_cached_memory_packets.return_value = [
        {
            "packet_type": "recent_observation",
            "title": "Recent Observation: ep_old",
            "summary": "Older amber dogfood session_recent recall proof.",
            "episode_ids": ["ep_old"],
            "provenance": ["episode:ep_old"],
            "_cache_scope": "session_recent",
        },
        {
            "packet_type": "recent_observation",
            "title": "Recent Observation: ep_new",
            "summary": "Duplicate newest packet should be removed.",
            "episode_ids": ["ep_new"],
            "provenance": ["episode:ep_new"],
            "_cache_scope": "session_recent",
        },
    ]

    capture_surface._cache_recent_observation_packet(
        manager,
        group_id="native_brain",
        episode_id="ep_new",
        content="New dogfood session_recent recall proof.",
        source="codex",
        packet_source="mcp_observe",
    )

    manager.get_recent_cached_memory_packets.assert_called_once_with(
        "native_brain",
        scopes=("session_recent",),
        limit_packets=4,
    )
    packets = manager.cache_memory_packets.call_args.kwargs["packets"]
    assert [packet["episode_ids"] for packet in packets] == [["ep_new"], ["ep_old"]]
    assert all("_cache_scope" not in packet for packet in packets)


def test_build_observation_attachment_preserves_payload() -> None:
    attachment = build_observation_attachment(
        mime_type="image/png",
        data_url="data:image/png;base64,abc",
        description="lathe setup",
    )

    assert attachment.mime_type == "image/png"
    assert attachment.data_url == "data:image/png;base64,abc"
    assert attachment.description == "lathe setup"


def test_capture_diagnostics_expose_stage_timings_for_api_and_mcp() -> None:
    manager = SimpleNamespace(
        get_last_capture_stage_timings=lambda: {
            "capture_store": 1.25,
            "cue_index": 3.5,
            "projection_enqueue": 0.75,
        },
    )

    api_response = attach_api_capture_diagnostics({"status": "observed"}, manager)
    mcp_response = attach_mcp_capture_diagnostics({"status": "stored"}, manager)

    assert api_response["diagnostics"]["stageTimingsMs"] == {
        "captureStore": 1.25,
        "cueIndex": 3.5,
        "projectionEnqueue": 0.75,
    }
    assert mcp_response["diagnostics"]["stage_timings_ms"] == {
        "capture_store": 1.25,
        "cue_index": 3.5,
        "projection_enqueue": 0.75,
    }


def test_capture_diagnostics_mark_deferred_raw_capture_lifecycle() -> None:
    manager = SimpleNamespace(
        get_last_capture_stage_timings=lambda: {"capture_store_timeout": 1002.0},
    )

    api_response = attach_api_capture_diagnostics(
        {
            "status": "observed",
            "lifecycle": {
                "stage": "cue",
                "captureStatus": "stored",
                "projectionStatus": "queued",
            },
        },
        manager,
    )
    mcp_response = attach_mcp_capture_diagnostics(
        {
            "status": "stored",
            "lifecycle": {
                "stage": "cue",
                "capture_status": "stored",
                "projection_status": "queued",
            },
        },
        manager,
    )

    assert api_response["lifecycle"]["captureStatus"] == "deferred"
    assert api_response["lifecycle"]["projectionStatus"] == "pending"
    assert mcp_response["lifecycle"]["capture_status"] == "deferred"
    assert mcp_response["lifecycle"]["projection_status"] == "pending"


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
    group_id, sample = _latest_memory_operation(manager)
    assert group_id == "native_brain"
    assert sample.operation == "observe"
    assert sample.source == "api_observe"
    assert sample.mode == "api_observe"
    assert sample.status == "ok"
    assert sample.result_count == 1


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
    group_id, sample = _latest_memory_operation(manager)
    assert group_id == "native_brain"
    assert sample.operation == "remember"
    assert sample.source == "api_remember"
    assert sample.mode == "api_remember"
    assert sample.status == "ok"
    assert sample.result_count == 1


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
    group_id, sample = _latest_memory_operation(manager)
    assert group_id == "native_brain"
    assert sample.operation == "remember"
    assert sample.source == "mcp_remember"
    assert sample.mode == "mcp_remember"
    assert sample.status == "ok"
    assert sample.result_count == 1


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
        capture_store_timeout_ms=capture_surface._AGENT_WRITE_CAPTURE_STORE_TIMEOUT_MS,
    )
    manager.cache_memory_packets.assert_called_once()
    cache_kwargs = manager.cache_memory_packets.call_args.kwargs
    assert cache_kwargs["scope"] == "session_recent"
    assert cache_kwargs["topic_hint"] is None
    assert cache_kwargs["project_path"] is None
    assert cache_kwargs["persist"] is False
    assert cache_kwargs["packets"][0]["packet_type"] == "recent_observation"
    assert cache_kwargs["packets"][0]["episode_ids"] == ["ep_observe"]
    assert "Observed an operator preference." in cache_kwargs["packets"][0]["summary"]
    assert session.episode_count == 1
    ingest_live_turn.assert_awaited_once_with(
        manager,
        "Observed an operator preference.",
        source="observe",
    )
    recall_middleware.assert_awaited_once()
    assert response["operation"] == "observe"
    assert response["lifecycle"]["stage"] == "cue"
    group_id, sample = _latest_memory_operation(manager)
    assert group_id == "native_brain"
    assert sample.operation == "observe"
    assert sample.source == "mcp_observe"
    assert sample.mode == "mcp_observe"
    assert sample.status == "ok"
    assert sample.result_count == 1


@pytest.mark.asyncio
async def test_build_mcp_observe_write_surface_bounds_slow_side_effects(
    monkeypatch,
) -> None:
    monkeypatch.setattr(capture_surface, "_MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS", 0.01)
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)

    live_turn_finished = asyncio.Event()

    async def slow_live_turn(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        live_turn_finished.set()

    async def slow_recall_middleware(*_args, **_kwargs):
        await asyncio.sleep(0.05)

    response = await build_mcp_observe_write_surface(
        manager,
        content="Observed an operator preference that should not block on side effects.",
        group_id="native_brain",
        session=session,
        source="mcp",
        ingest_live_turn=slow_live_turn,
        recall_middleware=slow_recall_middleware,
    )

    assert response["operation"] == "observe"
    assert response["diagnostics"]["stage_timings_ms"]["live_turn_timeout"] >= 0
    assert response["diagnostics"]["stage_timings_ms"]["recall_middleware_timeout"] >= 0
    assert session.episode_count == 1
    await asyncio.wait_for(live_turn_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_build_mcp_observe_write_surface_runs_side_effects_concurrently(
    monkeypatch,
) -> None:
    monkeypatch.setattr(capture_surface, "_MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS", 0.05)
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)

    async def slow_side_effect(*_args, **_kwargs):
        await asyncio.sleep(0.2)

    started = asyncio.get_running_loop().time()
    response = await build_mcp_observe_write_surface(
        manager,
        content="Observed an operator preference with slow write-side enrichment.",
        group_id="native_brain",
        session=session,
        source="mcp",
        ingest_live_turn=slow_side_effect,
        recall_middleware=slow_side_effect,
    )
    elapsed = asyncio.get_running_loop().time() - started

    timings = response["diagnostics"]["stage_timings_ms"]
    assert elapsed < 0.12
    assert timings["live_turn_timeout"] < 100
    assert timings["recall_middleware_timeout"] < 100
    assert session.episode_count == 1


@pytest.mark.asyncio
async def test_build_mcp_observe_write_surface_bounds_default_recall_side_effect() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)

    async def slow_recall_middleware(*_args, **_kwargs):
        await asyncio.sleep(1)

    started = asyncio.get_running_loop().time()
    response = await build_mcp_observe_write_surface(
        manager,
        content="Observed an operator preference that should not pay a long recall tax.",
        group_id="native_brain",
        session=session,
        source="mcp",
        ingest_live_turn=AsyncMock(),
        recall_middleware=slow_recall_middleware,
    )
    elapsed = asyncio.get_running_loop().time() - started

    timings = response["diagnostics"]["stage_timings_ms"]
    assert capture_surface._MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS == 0.075
    assert elapsed < 0.2
    assert timings["recall_middleware_timeout"] < 200
    assert session.episode_count == 1


@pytest.mark.asyncio
async def test_build_mcp_observe_write_surface_uses_short_live_turn_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setattr(capture_surface, "_MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(capture_surface, "_MCP_WRITE_LIVE_TURN_TIMEOUT_SECONDS", 0.01)
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    session = SimpleNamespace(session_id="sess_1", episode_count=0, last_activity=None)
    live_turn_finished = asyncio.Event()

    async def slow_live_turn(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        live_turn_finished.set()

    started = asyncio.get_running_loop().time()
    response = await build_mcp_observe_write_surface(
        manager,
        content="Observed an operator preference with slow live-turn fingerprinting.",
        group_id="native_brain",
        session=session,
        source="mcp",
        ingest_live_turn=slow_live_turn,
        recall_middleware=AsyncMock(),
    )
    elapsed = asyncio.get_running_loop().time() - started

    timings = response["diagnostics"]["stage_timings_ms"]
    assert elapsed < 0.08
    assert timings["live_turn_timeout"] < 50
    assert "recall_middleware_timeout" not in timings
    await asyncio.wait_for(live_turn_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_build_api_auto_observe_surface_records_skip_metrics() -> None:
    manager = MagicMock()

    response = await build_api_auto_observe_surface(
        manager,
        content="short",
        group_id="native_brain",
        source="auto:prompt",
    )

    assert response["status"] == "skipped"
    assert response["reason"] == "too_short"
    group_id, sample = _latest_memory_operation(manager)
    assert group_id == "native_brain"
    assert sample.operation == "observe"
    assert sample.source == "api_auto_observe"
    assert sample.mode == "api_auto_observe"
    assert sample.status == "skipped"
    assert sample.skip_reason == "too_short"
    assert sample.result_count == 0


@pytest.mark.asyncio
async def test_build_api_auto_observe_surface_caches_session_recent_packet() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_auto")

    response = await build_api_auto_observe_surface(
        manager,
        content="[user|Engram] capture this prompt for immediate context",
        group_id="native_brain",
        source="auto:prompt",
        session_id="sess-123",
        auto_observe_enabled=True,
        dedup_check=lambda _content: False,
    )

    assert response["status"] == "observed"
    manager.store_episode.assert_awaited_once_with(
        content="[user|Engram] capture this prompt for immediate context",
        group_id="native_brain",
        source="auto:prompt",
        session_id="sess-123",
        conversation_date=None,
        capture_store_timeout_ms=capture_surface._AGENT_WRITE_CAPTURE_STORE_TIMEOUT_MS,
    )
    manager.cache_memory_packets.assert_called_once()
    cache_kwargs = manager.cache_memory_packets.call_args.kwargs
    assert cache_kwargs["scope"] == "session_recent"
    assert cache_kwargs["topic_hint"] is None
    assert cache_kwargs["project_path"] is None
    assert cache_kwargs["persist"] is False
    assert cache_kwargs["packets"][0]["episode_ids"] == ["ep_auto"]
    assert cache_kwargs["packets"][0]["trust"]["source"] == "api_auto_observe"
    assert "immediate context" in cache_kwargs["packets"][0]["summary"]


@pytest.mark.asyncio
async def test_build_api_observe_write_surface_ignores_metrics_record_failure() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    manager.record_memory_operation.side_effect = RuntimeError("metrics unavailable")

    response = await build_api_observe_write_surface(
        manager,
        content="Observed operator preference.",
        group_id="native_brain",
        source="dashboard",
    )

    assert response["status"] == "observed"
    assert response["episodeId"] == "ep_observe"
    manager.record_memory_operation.assert_called_once()


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
