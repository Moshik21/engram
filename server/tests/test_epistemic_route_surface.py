from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.epistemic_route import (
    build_question_route_surface,
    recent_route_turn_contents,
)


def test_recent_route_turn_contents_accepts_strings_and_chat_messages() -> None:
    history = [
        "Older Redis note",
        SimpleNamespace(content="We debated native Helix."),
        SimpleNamespace(content=""),
        "Current Engram route question",
    ]

    assert recent_route_turn_contents(history, limit=2) == [
        "We debated native Helix.",
        "Current Engram route question",
    ]


@pytest.mark.asyncio
async def test_build_question_route_surface_normalizes_route_contract() -> None:
    manager = MagicMock()
    manager.route_question = AsyncMock(return_value={"questionFrame": {"mode": "inspect"}})

    result = await build_question_route_surface(
        manager,
        group_id="native_brain",
        question="Where did the OpenClaw decision come from?",
        project_path="/tmp/project",
        history=[SimpleNamespace(content="We mentioned OpenClaw."), "Need sources."],
        session_entity_names=["OpenClaw"],
        surface="rest",
    )

    assert result == {"questionFrame": {"mode": "inspect"}}
    assert manager.route_question.await_args.kwargs == {
        "group_id": "native_brain",
        "project_path": "/tmp/project",
        "recent_turns": ["We mentioned OpenClaw.", "Need sources."],
        "session_entity_names": ["OpenClaw"],
        "surface": "rest",
    }
