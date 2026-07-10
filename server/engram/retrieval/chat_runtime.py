"""Knowledge-chat runtime helpers for memory need and live context."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic

from engram.models.recall import MemoryNeed
from engram.retrieval.chat_events import build_chat_tool_stream_events
from engram.retrieval.chat_feedback import (
    apply_chat_recall_feedback,
    should_retry_chat_response,
)
from engram.retrieval.chat_persistence import (
    chat_conversation_not_found_payload,
    resolve_chat_conversation,
    schedule_chat_turn_persistence,
)
from engram.retrieval.chat_tools import (
    CHAT_TOOLS,
    extract_message_text,
    retry_memory_grounded_response,
    run_chat_tool_use_loop,
)
from engram.retrieval.context import (
    ingest_manager_conversation_turn,
    manager_conversation_context,
    manager_conversation_top_entity_names,
    manager_conversation_turn_count,
)
from engram.retrieval.control import (
    record_manager_memory_need_analysis,
    resolve_manager_recall_need_thresholds,
)
from engram.retrieval.epistemic import render_epistemic_summary
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.need import analyze_memory_need

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_MESSAGES = 10
DEFAULT_MAX_TOOL_TURNS = 3


@dataclass(frozen=True)
class ApiChatRateLimitSurface:
    """REST chat rate-limit payload plus HTTP status."""

    status_code: int
    payload: dict


@dataclass(frozen=True)
class ApiChatStreamResponseSurface:
    """REST chat response surface with either an error payload or SSE stream."""

    status_code: int
    payload: dict[str, Any] | None = None
    stream: AsyncIterator[str] | None = None
    media_type: str = "text/event-stream"


@dataclass(frozen=True)
class ChatResponseTurnResult:
    """Route-neutral result for a completed knowledge-chat memory turn."""

    stream_events: list[dict[str, Any]]
    final_text: str
    recall_results: list[dict[str, Any]]


def build_chat_sse_event(data: Mapping[str, Any]) -> str:
    """Format one AI SDK SSE event for the REST chat transport."""
    return f"data: {json.dumps(dict(data))}\n\n"


def build_api_chat_rate_limit_surface(remaining: int) -> ApiChatRateLimitSurface:
    """Return the REST chat rate-limit response surface."""
    return ApiChatRateLimitSurface(
        status_code=429,
        payload={"detail": "Rate limit exceeded for chat", "remaining": remaining},
    )


async def check_api_chat_rate_limit(
    rate_limiter: Any | None,
    *,
    group_id: str,
    action: str = "chat",
) -> ApiChatRateLimitSurface | None:
    """Return a REST rate-limit surface when chat is not allowed."""
    if rate_limiter is None:
        return None

    allowed, remaining = await rate_limiter.check(group_id, action)
    if allowed:
        return None
    return build_api_chat_rate_limit_surface(remaining)


async def build_api_chat_stream_response_surface(
    manager: Any,
    *,
    group_id: str,
    message: str,
    history: Sequence[Any] | None,
    conversation_store: Any | None,
    conversation_id: str | None,
    session_date: str | None,
    rate_limiter: Any | None,
    event_bus: Any | None,
    stream_logger: Any | None = None,
) -> ApiChatStreamResponseSurface:
    """Build the REST chat response surface outside the FastAPI route."""
    rate_limit_result = await check_api_chat_rate_limit(
        rate_limiter,
        group_id=group_id,
    )
    if rate_limit_result is not None:
        return ApiChatStreamResponseSurface(
            status_code=rate_limit_result.status_code,
            payload=rate_limit_result.payload,
        )

    conversation = await resolve_chat_conversation(
        conversation_store,
        group_id=group_id,
        conversation_id=conversation_id,
        message=message,
        session_date=session_date,
    )
    if conversation.not_found:
        return ApiChatStreamResponseSurface(
            status_code=404,
            payload=chat_conversation_not_found_payload(),
        )

    return ApiChatStreamResponseSurface(
        status_code=200,
        stream=stream_api_chat_sse_events(
            manager=manager,
            group_id=group_id,
            message=message,
            history=history,
            conversation_store=conversation_store,
            conversation_id=conversation.conversation_id,
            event_bus=event_bus,
            session_entity_names=manager_conversation_top_entity_names(manager),
            stream_logger=stream_logger,
        ),
    )


async def run_chat_response_turn(
    client: Any,
    manager: Any,
    *,
    group_id: str,
    message: str,
    history: Sequence[Any] | None,
    event_bus: Any | None = None,
    session_entity_names: list[str] | None = None,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_tool_turns: int = 3,
    context_max_tokens: int = 1000,
    text_part_id: str = "text_0",
    text_chunk_size: int = 20,
) -> ChatResponseTurnResult:
    """Run a full REST knowledge-chat memory turn outside the transport route."""
    chat_policy = build_chat_runtime_policy(manager)
    await hydrate_chat_context(
        manager,
        history,
        message,
        max_history_messages=max_history_messages,
    )

    chat_need: MemoryNeed | None = None
    epistemic_bundle = None
    topic_hint: str | None = message
    entity_names = session_entity_names or []

    if chat_policy.recall_need_analyzer_enabled:
        chat_need = await analyze_chat_memory_need(
            message,
            manager,
            history,
            session_entity_names=entity_names,
            group_id=group_id,
        )
        await record_manager_memory_need_analysis(manager, group_id, chat_need)
        if chat_policy.recall_telemetry_enabled and event_bus is not None:
            publish_memory_need_analysis(
                event_bus,
                group_id,
                chat_need,
                source="knowledge_chat",
                mode="chat",
                turn_text=message,
            )
        topic_hint = chat_need.query_hint if chat_need.should_recall else None

    if chat_policy.epistemic_routing_enabled:
        epistemic_bundle = await gather_chat_epistemic_evidence(
            manager,
            message=message,
            group_id=group_id,
            history=history,
            session_entity_names=entity_names,
            memory_need=chat_need,
        )

    context_result = await build_chat_context_surface(
        manager,
        group_id=group_id,
        max_tokens=context_max_tokens,
        topic_hint=topic_hint,
    )
    system_prompt = build_chat_system_prompt_surface(
        context=context_result["context"],
        memory_need=chat_need,
        epistemic_bundle=epistemic_bundle,
    )
    messages = build_chat_messages(
        history,
        message,
        max_history_messages=max_history_messages,
    )

    tool_loop = await run_chat_tool_use_loop(
        client,
        manager=manager,
        group_id=group_id,
        system_prompt=system_prompt,
        messages=messages,
        tools=CHAT_TOOLS,
        initial_recall_results=list(getattr(epistemic_bundle, "memory_results", []) or []),
        max_tool_turns=max_tool_turns,
    )
    final_text = extract_message_text(tool_loop.response.content)

    if final_text and should_retry_chat_response(
        manager,
        chat_need=chat_need,
        response_text=final_text,
        recall_results=tool_loop.recall_results,
    ):
        if chat_need is None:
            raise RuntimeError("chat_need missing during retry path")
        final_text = await retry_memory_grounded_response(
            client,
            system_prompt=system_prompt,
            loop_messages=tool_loop.loop_messages,
            chat_need=chat_need,
            prior_response=final_text,
        )

    if final_text:
        await apply_chat_recall_feedback(
            manager,
            group_id=group_id,
            query=message,
            response_text=final_text,
            recall_results=tool_loop.recall_results,
        )
        await record_chat_assistant_turn(manager, final_text)

    stream_events = [
        *build_chat_tool_stream_events(tool_loop.recall_results, tool_loop.facts),
        *build_chat_text_stream_events(
            final_text,
            text_part_id=text_part_id,
            chunk_size=text_chunk_size,
        ),
    ]
    return ChatResponseTurnResult(
        stream_events=stream_events,
        final_text=final_text,
        recall_results=tool_loop.recall_results,
    )


async def stream_api_chat_sse_events(
    *,
    manager: Any,
    group_id: str,
    message: str,
    history: Sequence[Any] | None,
    conversation_store: Any | None,
    conversation_id: str | None,
    event_bus: Any | None,
    session_entity_names: list[str] | None = None,
    client_factory: Any | None = None,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    max_tool_turns: int = DEFAULT_MAX_TOOL_TURNS,
    stream_logger: logging.Logger | None = None,
) -> AsyncIterator[str]:
    """Run the REST knowledge-chat turn and yield AI SDK SSE frames."""
    log = stream_logger or logger
    try:
        yield build_chat_sse_event({"type": "start"})
        yield build_chat_sse_event({"type": "start-step"})

        client = (client_factory or anthropic.AsyncAnthropic)()
        turn = await run_chat_response_turn(
            client,
            manager=manager,
            group_id=group_id,
            message=message,
            history=history,
            event_bus=event_bus,
            session_entity_names=session_entity_names,
            max_history_messages=max_history_messages,
            max_tool_turns=max_tool_turns,
        )

        for stream_event in turn.stream_events:
            yield build_chat_sse_event(stream_event)

        yield build_chat_sse_event({"type": "finish-step"})
        yield build_chat_sse_event(
            {
                "type": "finish",
                "finishReason": "stop",
                **({"conversationId": conversation_id} if conversation_id else {}),
            }
        )

        schedule_chat_turn_persistence(
            conversation_store,
            conversation_id=conversation_id,
            group_id=group_id,
            user_message=message,
            assistant_message=turn.final_text,
            recall_results=turn.recall_results,
        )
    except Exception as exc:
        log.exception("Chat stream error")
        yield build_chat_sse_event({"type": "error", "errorText": str(exc)})
        yield build_chat_sse_event({"type": "finish", "finishReason": "error"})

    yield "data: [DONE]\n\n"


def build_chat_text_stream_events(
    text: str,
    *,
    text_part_id: str = "text_0",
    chunk_size: int = 20,
) -> list[dict[str, Any]]:
    """Build AI SDK v6 text events for a completed chat response."""
    if not text:
        return []
    return [
        {"type": "text-start", "id": text_part_id},
        *[
            {
                "type": "text-delta",
                "id": text_part_id,
                "delta": text[index : index + chunk_size],
            }
            for index in range(0, len(text), chunk_size)
        ],
        {"type": "text-end", "id": text_part_id},
    ]


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
        manager.get_recall_need_graph_probe() if manager.recall_need_graph_probe_enabled() else None
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


def build_chat_system_prompt_surface(
    *,
    context: str,
    memory_need: MemoryNeed | None,
    epistemic_bundle: Any | None,
) -> list[dict[str, Any]]:
    """Build the Anthropic system prompt for REST knowledge chat."""
    memory_guidance = (
        build_chat_memory_guidance(memory_need)
        if memory_need is not None
        else (
            "Use memory tools when prior context matters. Do not guess when tools "
            "can provide precise answers."
        )
    )
    static_preamble = (
        "You are a helpful assistant with access to the user's memory graph. "
        "You have tools to search the graph — use them to answer questions accurately.\n\n"
        "Guidelines:\n"
        "- Use recall as the primary retrieval tool (pass lookup_kind='entities' or "
        "'facts' for structured lookups)\n"
        "- search_entities and search_facts are deprecated compat aliases only\n"
        "- Do not use search_facts as the primary path for project reconcile questions "
        "when artifacts or runtime are required\n"
        f"- {memory_guidance}\n\n"
        "Below is baseline context about the user:\n\n"
    )
    system_prompt: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": static_preamble,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": context,
        },
    ]
    if epistemic_bundle is not None:
        contract_guidance = "\n".join(
            f"- {item}" for item in epistemic_bundle.answer_contract.guidance[:4]
        )
        system_prompt.append(
            {
                "type": "text",
                "text": (
                    "Epistemic routing and gathered evidence for this turn:\n\n"
                    f"{render_epistemic_summary(epistemic_bundle)}\n\n"
                    "Answer-contract guidance for this turn:\n"
                    f"{contract_guidance}"
                ),
            }
        )
    return system_prompt


def build_chat_messages(
    history: Sequence[Any] | None,
    message: str,
    *,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
) -> list[dict[str, str]]:
    """Build the sliding-window Anthropic message list for REST knowledge chat."""
    messages: list[dict[str, str]] = []
    if history:
        for msg in history[-max_history_messages:]:
            messages.append(
                {
                    "role": str(getattr(msg, "role", "")),
                    "content": str(getattr(msg, "content", "")),
                }
            )
    messages.append({"role": "user", "content": message})
    return messages


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
