from __future__ import annotations

from unittest.mock import MagicMock

from engram.models.recall import MemoryNeed
from engram.retrieval.chat_feedback import (
    build_memory_grounding_retry_system_prompt,
    is_generic_memory_free_response,
    should_retry_chat_response,
)


def test_generic_memory_free_response_detects_short_generic_replies() -> None:
    assert is_generic_memory_free_response("Got it, happy to help with that.") is True
    assert (
        is_generic_memory_free_response("Engram should keep PyO3 native as the main path.")
        is False
    )
    assert (
        is_generic_memory_free_response(
            "Got it. " + "This detailed answer has enough specific follow-up text. " * 20
        )
        is False
    )


def test_should_retry_chat_response_requires_enabled_relevant_unused_memory() -> None:
    manager = MagicMock()
    manager.recall_need_post_response_safety_net_enabled.return_value = True
    need = MemoryNeed(need_type="project_state", should_recall=True, confidence=0.9)
    recall_results = [
        {
            "entity": {
                "id": "ent_engram",
                "name": "Engram",
                "type": "Project",
                "summary": "Brain runtime",
            },
            "score": 0.91,
        }
    ]

    assert (
        should_retry_chat_response(
            manager,
            chat_need=need,
            response_text="Got it, we can work through it.",
            recall_results=recall_results,
        )
        is True
    )
    assert (
        should_retry_chat_response(
            manager,
            chat_need=need,
            response_text="Engram should stay focused on the brain runtime.",
            recall_results=recall_results,
        )
        is False
    )

    manager.recall_need_post_response_safety_net_enabled.return_value = False
    assert (
        should_retry_chat_response(
            manager,
            chat_need=need,
            response_text="Got it.",
            recall_results=recall_results,
        )
        is False
    )


def test_build_memory_grounding_retry_system_prompt_appends_instruction() -> None:
    prompt = [{"type": "text", "text": "base"}]
    need = MemoryNeed(need_type="open_loop", should_recall=True, confidence=0.8)

    retry_prompt = build_memory_grounding_retry_system_prompt(
        prompt,
        chat_need=need,
        prior_response="Got it.",
    )

    assert retry_prompt[:-1] == prompt
    assert "open_loop" in retry_prompt[-1]["text"]
    assert "Got it." in retry_prompt[-1]["text"]
