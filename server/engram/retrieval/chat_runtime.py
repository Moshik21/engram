"""Knowledge-chat runtime helpers for memory need and live context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from engram.models.recall import MemoryNeed
from engram.retrieval.context import (
    ingest_manager_conversation_turn,
    manager_conversation_context,
    manager_conversation_turn_count,
)
from engram.retrieval.control import resolve_manager_recall_need_thresholds
from engram.retrieval.need import analyze_memory_need

DEFAULT_MAX_HISTORY_MESSAGES = 10


@dataclass(frozen=True)
class ApiChatRateLimitSurface:
    """REST chat rate-limit payload plus HTTP status."""

    status_code: int
    payload: dict


def build_api_chat_rate_limit_surface(remaining: int) -> ApiChatRateLimitSurface:
    """Return the REST chat rate-limit response surface."""
    return ApiChatRateLimitSurface(
        status_code=429,
        payload={"detail": "Rate limit exceeded for chat", "remaining": remaining},
    )


def build_chat_runtime_policy(manager: Any) -> Any:
    """Return the route-facing knowledge-chat runtime policy."""
    return manager.get_chat_runtime_policy()


async def gather_chat_epistemic_evidence(
    manager: Any,
    *,
    message: str,
    group_id: str,
    history: Sequence[Any] | None,
    session_entity_names: list[str],
    memory_need: MemoryNeed | None,
) -> Any:
    """Gather epistemic evidence for a REST chat turn through the manager facade."""
    return await manager.gather_epistemic_evidence(
        message,
        group_id=group_id,
        project_path=None,
        recent_turns=recent_chat_turn_contents(history, limit=6),
        session_entity_names=session_entity_names,
        surface="rest",
        memory_need=memory_need,
    )


async def build_chat_context_surface(
    manager: Any,
    *,
    group_id: str,
    topic_hint: str | None,
    max_tokens: int = 1000,
) -> dict:
    """Build the baseline REST chat context through the manager facade."""
    return await manager.get_context(
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
    )


async def analyze_chat_memory_need(
    message: str,
    manager: Any,
    history: Sequence[Any] | None = None,
    session_entity_names: list[str] | None = None,
    *,
    group_id: str,
) -> MemoryNeed:
    """Analyze whether a knowledge-chat turn likely needs memory."""
    cfg = manager.get_memory_need_config()
    graph_probe = (
        manager.get_recall_need_graph_probe()
        if manager.recall_need_graph_probe_enabled()
        else None
    )
    return await analyze_memory_need(
        message,
        recent_turns=recent_chat_turn_contents(history, limit=6),
        session_entity_names=session_entity_names or [],
        mode="chat",
        graph_probe=graph_probe,
        group_id=group_id,
        conv_context=manager_conversation_context(manager),
        cfg=cfg,
        thresholds=await resolve_manager_recall_need_thresholds(manager, group_id),
    )


def build_chat_memory_guidance(need: MemoryNeed) -> str:
    """Produce a short system-prompt hint for memory usage on a chat turn."""
    if not need.should_recall:
        return (
            "Memory does not look required for this turn. Answer directly unless you need "
            "specific prior facts, commitments, or project history."
        )
    return (
        f"Memory is likely relevant for this turn ({need.need_type}). "
        "Use recall/search tools before answering when prior context could change the answer."
    )


async def hydrate_chat_context(
    manager: Any,
    history: Sequence[Any] | None,
    message: str,
    *,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
) -> None:
    """Ground live conversation context in chat history before recall/tool use."""
    conv_context = manager_conversation_context(manager)
    if conv_context is None:
        return
    if manager_conversation_turn_count(manager) == 0 and history:
        for msg in history[-max_history_messages:]:
            source = "chat_user" if getattr(msg, "role", "") == "user" else "chat_assistant"
            await ingest_manager_conversation_turn(
                manager,
                getattr(msg, "content", ""),
                source=source,
            )
    await ingest_manager_conversation_turn(
        manager,
        message,
        source="chat_user",
    )


async def record_chat_assistant_turn(manager: Any, message: str) -> None:
    """Record an assistant response as a live conversation turn."""
    conv_context = manager_conversation_context(manager)
    if conv_context is None or not message.strip():
        return
    await ingest_manager_conversation_turn(
        manager,
        message,
        source="chat_assistant",
    )


def recent_chat_turn_contents(history: Sequence[Any] | None, *, limit: int) -> list[str]:
    """Return recent non-empty chat message content."""
    return [
        str(content)
        for content in (getattr(msg, "content", "") for msg in (history or []))
        if str(content).strip()
    ][-limit:]
