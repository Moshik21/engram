from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from engram.retrieval.chat_runtime import (
    ChatResponseTurnResult,
    build_api_chat_stream_response_surface,
    build_chat_sse_event,
    stream_api_chat_sse_events,
)
from engram.storage.sqlite.conversations import ConversationNotFoundError


async def _collect_stream(stream):
    return [chunk async for chunk in stream]


def _parse_sse_chunks(chunks: list[str]) -> list[dict]:
    events = []
    for chunk in chunks:
        if chunk == "data: [DONE]\n\n":
            continue
        assert chunk.startswith("data: ")
        events.append(json.loads(chunk.removeprefix("data: ").strip()))
    return events


def test_build_chat_sse_event_serializes_one_event() -> None:
    assert build_chat_sse_event({"type": "finish", "finishReason": "stop"}) == (
        'data: {"type": "finish", "finishReason": "stop"}\n\n'
    )


@pytest.mark.asyncio
async def test_build_api_chat_stream_response_surface_returns_rate_limit_payload() -> None:
    rate_limiter = SimpleNamespace(check=AsyncMock(return_value=(False, 0)))

    surface = await build_api_chat_stream_response_surface(
        SimpleNamespace(),
        group_id="brain_a",
        message="hello",
        history=[],
        conversation_store=SimpleNamespace(),
        conversation_id=None,
        session_date=None,
        rate_limiter=rate_limiter,
        event_bus=None,
    )

    assert surface.status_code == 429
    assert surface.payload == {"detail": "Rate limit exceeded for chat", "remaining": 0}
    assert surface.stream is None


@pytest.mark.asyncio
async def test_build_api_chat_stream_response_surface_returns_conversation_not_found() -> None:
    conversation_store = SimpleNamespace(
        get_conversation=AsyncMock(side_effect=ConversationNotFoundError("missing"))
    )

    surface = await build_api_chat_stream_response_surface(
        SimpleNamespace(),
        group_id="brain_a",
        message="hello",
        history=[],
        conversation_store=conversation_store,
        conversation_id="foreign_conv",
        session_date=None,
        rate_limiter=None,
        event_bus=None,
    )

    assert surface.status_code == 404
    assert surface.payload == {"detail": "Conversation not found"}
    assert surface.stream is None


@pytest.mark.asyncio
async def test_stream_api_chat_sse_events_runs_turn_and_schedules_persistence() -> None:
    client = SimpleNamespace()
    client_factory = Mock(return_value=client)
    manager = SimpleNamespace()
    conversation_store = SimpleNamespace()
    event_bus = SimpleNamespace()
    stream_logger = SimpleNamespace(exception=Mock())
    turn = ChatResponseTurnResult(
        stream_events=[{"type": "text-start", "id": "text_0"}],
        final_text="Engram remembers.",
        recall_results=[{"result_type": "entity", "entity": {"id": "ent_1"}}],
    )
    run_turn = AsyncMock(return_value=turn)
    schedule_persistence = Mock()

    with (
        patch("engram.retrieval.chat_runtime.run_chat_response_turn", run_turn),
        patch(
            "engram.retrieval.chat_runtime.schedule_chat_turn_persistence",
            schedule_persistence,
        ),
    ):
        chunks = await _collect_stream(
            stream_api_chat_sse_events(
                manager=manager,
                group_id="brain_a",
                message="What changed?",
                history=[],
                conversation_store=conversation_store,
                conversation_id="conv_1",
                event_bus=event_bus,
                session_entity_names=["Engram"],
                client_factory=client_factory,
                max_history_messages=4,
                max_tool_turns=2,
                stream_logger=stream_logger,
            )
        )

    assert _parse_sse_chunks(chunks) == [
        {"type": "start"},
        {"type": "start-step"},
        {"type": "text-start", "id": "text_0"},
        {"type": "finish-step"},
        {"type": "finish", "finishReason": "stop", "conversationId": "conv_1"},
    ]
    assert chunks[-1] == "data: [DONE]\n\n"
    client_factory.assert_called_once_with()
    run_turn.assert_awaited_once_with(
        client,
        manager=manager,
        group_id="brain_a",
        message="What changed?",
        history=[],
        event_bus=event_bus,
        session_entity_names=["Engram"],
        max_history_messages=4,
        max_tool_turns=2,
    )
    schedule_persistence.assert_called_once_with(
        conversation_store,
        conversation_id="conv_1",
        group_id="brain_a",
        user_message="What changed?",
        assistant_message="Engram remembers.",
        recall_results=[{"result_type": "entity", "entity": {"id": "ent_1"}}],
    )
    stream_logger.exception.assert_not_called()


@pytest.mark.asyncio
async def test_stream_api_chat_sse_events_turns_errors_into_finish_events() -> None:
    client = SimpleNamespace()
    stream_logger = SimpleNamespace(exception=Mock())
    run_turn = AsyncMock(side_effect=RuntimeError("API key invalid"))
    schedule_persistence = Mock()

    with (
        patch("engram.retrieval.chat_runtime.run_chat_response_turn", run_turn),
        patch(
            "engram.retrieval.chat_runtime.schedule_chat_turn_persistence",
            schedule_persistence,
        ),
    ):
        chunks = await _collect_stream(
            stream_api_chat_sse_events(
                manager=SimpleNamespace(),
                group_id="brain_a",
                message="test",
                history=None,
                conversation_store=None,
                conversation_id=None,
                event_bus=None,
                client_factory=Mock(return_value=client),
                stream_logger=stream_logger,
            )
        )

    assert _parse_sse_chunks(chunks) == [
        {"type": "start"},
        {"type": "start-step"},
        {"type": "error", "errorText": "API key invalid"},
        {"type": "finish", "finishReason": "error"},
    ]
    assert chunks[-1] == "data: [DONE]\n\n"
    stream_logger.exception.assert_called_once_with("Chat stream error")
    schedule_persistence.assert_not_called()
