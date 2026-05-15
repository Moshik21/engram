"""Knowledge-chat recall feedback and retry policy helpers."""

from __future__ import annotations

from typing import Any

from engram.models.recall import MemoryNeed
from engram.retrieval.feedback import (
    extract_recall_targets,
    partition_recall_targets_by_usage,
)


async def apply_chat_recall_feedback(
    manager: Any,
    *,
    group_id: str,
    query: str,
    response_text: str,
    recall_results: list[dict],
) -> None:
    """Upgrade selected chat memories to used/dismissed based on the final response."""
    if not manager.recall_usage_feedback_enabled():
        return
    if not response_text or not recall_results:
        return

    target_lookup = {
        target["lookup_id"]: target for target in extract_recall_targets(recall_results)
    }
    if not target_lookup:
        return

    used_targets, dismissed_targets = partition_recall_targets_by_usage(
        response_text,
        recall_results,
    )
    if used_targets:
        await manager.apply_memory_interaction(
            [target["lookup_id"] for target in used_targets],
            group_id=group_id,
            interaction_type="used",
            source="chat_response",
            query=query,
            result_lookup=target_lookup,
        )
    if dismissed_targets:
        await manager.apply_memory_interaction(
            [target["lookup_id"] for target in dismissed_targets],
            group_id=group_id,
            interaction_type="dismissed",
            source="chat_response",
            query=query,
            result_lookup=target_lookup,
        )


def is_generic_memory_free_response(response_text: str) -> bool:
    """Detect short generic replies that likely failed to use relevant memory."""
    lowered = " ".join(response_text.lower().split())
    if not lowered:
        return False
    generic_prefixes = (
        "that makes sense",
        "got it",
        "understood",
        "thanks for sharing",
        "that sounds",
        "i understand",
        "you're right",
        "it makes sense",
    )
    generic_fragments = (
        "let me know if you want help",
        "if you'd like, i can help",
        "happy to help",
        "we can work through it",
    )
    if len(lowered.split()) > 80:
        return False
    if any(lowered.startswith(prefix) for prefix in generic_prefixes):
        return True
    return any(fragment in lowered for fragment in generic_fragments)


def should_retry_chat_response(
    manager: Any,
    *,
    chat_need: MemoryNeed | None,
    response_text: str,
    recall_results: list[dict],
) -> bool:
    """Whether the knowledge-chat safety net should retry once."""
    if not manager.recall_need_post_response_safety_net_enabled():
        return False
    if chat_need is None or not chat_need.should_recall:
        return False
    if not is_generic_memory_free_response(response_text):
        return False
    used_targets, _dismissed_targets = partition_recall_targets_by_usage(
        response_text,
        recall_results,
    )
    return not used_targets


def build_memory_grounding_retry_system_prompt(
    system_prompt: list[dict],
    *,
    chat_need: MemoryNeed,
    prior_response: str,
) -> list[dict]:
    """Append a stronger memory-grounding instruction to a chat system prompt."""
    retry_system = list(system_prompt)
    retry_system.append(
        {
            "type": "text",
            "text": (
                "The previous draft stayed too generic for a memory-relevant turn. "
                "Revise once and ground the answer in specific remembered facts, "
                "people, timelines, or project state when available. If memory "
                "still does not help, say that plainly instead of giving a generic "
                f"reassurance. Need type: {chat_need.need_type}. Prior draft: "
                f"{prior_response[:400]}"
            ),
        }
    )
    return retry_system
