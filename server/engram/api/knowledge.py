"""Knowledge Management API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

import anthropic
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engram.api.deps import (
    get_config,
    get_conversation_store,
    get_manager,
    get_notification_surface_service,
    get_rate_limiter,
)
from engram.events.bus import get_event_bus
from engram.ingestion.adjudication_surface import (
    build_api_adjudication_resolution_surface,
    load_client_enabled_episode_adjudication_requests,
)
from engram.ingestion.capture_surface import (
    build_observation_attachment,
    ingest_projecting_memory,
    parse_conversation_date,
    store_observation,
)
from engram.ingestion.dedup import CaptureDedupCache
from engram.ingestion.offline_replay import build_api_manager_offline_replay_surface
from engram.ingestion.presenter import (
    memory_write_contract,
    present_api_memory_write,
    present_api_observe_skip,
)
from engram.ingestion.project_bootstrap import (
    build_project_bootstrap_surface,
    project_bootstrap_http_status,
)
from engram.models.recall import MemoryNeed
from engram.notifications.surface import (
    build_api_notification_dismiss_surface,
    build_api_notifications_surface,
)
from engram.retrieval.artifacts import build_api_artifact_search_surface
from engram.retrieval.chat_events import build_chat_tool_stream_events
from engram.retrieval.chat_feedback import (
    apply_chat_recall_feedback,
    should_retry_chat_response,
)
from engram.retrieval.chat_persistence import (
    chat_conversation_not_found_payload,
    persist_chat_turn,
    resolve_chat_conversation,
)
from engram.retrieval.chat_runtime import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    analyze_chat_memory_need,
    build_api_chat_rate_limit_surface,
    build_chat_context_surface,
    build_chat_messages,
    build_chat_runtime_policy,
    build_chat_system_prompt_surface,
    gather_chat_epistemic_evidence,
    hydrate_chat_context,
    record_chat_assistant_turn,
)
from engram.retrieval.chat_tools import (
    CHAT_TOOLS,
    extract_message_text,
    retry_memory_grounded_response,
    run_chat_tool_use_loop,
)
from engram.retrieval.context import (
    manager_conversation_top_entity_names,
)
from engram.retrieval.context_builder import build_api_context_surface
from engram.retrieval.control import record_manager_memory_need_analysis
from engram.retrieval.epistemic_route import build_question_route_surface
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.forgetting import build_api_forget_response_surface
from engram.retrieval.lookup import build_api_fact_search_surface
from engram.retrieval.preference_feedback import (
    build_api_explicit_feedback_surface,
)
from engram.retrieval.prospective import (
    build_api_create_intention_response_surface,
    build_api_dismiss_intention_response_surface,
    build_intention_list_surface,
)
from engram.retrieval.recall_surface import build_api_recall_surface
from engram.retrieval.runtime_state import build_runtime_state_surface
from engram.security.middleware import get_tenant
from engram.utils.offline_queue import drain_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# ─── Notifications ──────────────────────────────────────────────


class DismissBody(BaseModel):
    ids: list[str]


@router.get("/notifications")
async def get_notifications(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    since: float = Query(0.0, ge=0.0),
) -> JSONResponse:
    """Return pending or recent notifications."""
    tenant = get_tenant(request)
    group_id = tenant.group_id

    service = get_notification_surface_service()
    payload = build_api_notifications_surface(
        service,
        group_id=group_id,
        limit=limit,
        since=since,
    )
    return JSONResponse(content=payload)


@router.post("/notifications/dismiss")
async def dismiss_notifications(request: Request, body: DismissBody) -> JSONResponse:
    """Dismiss one or more notifications by ID."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    service = get_notification_surface_service()
    payload = build_api_notification_dismiss_surface(
        service,
        group_id=group_id,
        ids=body.ids,
    )
    return JSONResponse(content=payload)


# ─── Dedup cache for auto-observe ────────────────────────────────

_DEDUP = CaptureDedupCache(ttl_seconds=300.0, max_entries=1000)
_DEDUP_CACHE = _DEDUP.cache  # compatibility handle for existing tests
_DEDUP_TTL = _DEDUP.ttl_seconds


def _dedup_check(content: str) -> bool:
    """Return True if content was seen in the last 5 minutes (skip it)."""
    return _DEDUP.check(content)


# ─── Request bodies ──────────────────────────────────────────────


class ObserveBody(BaseModel):
    content: str
    source: str = "dashboard"
    conversation_date: str | None = None


class AutoObserveBody(BaseModel):
    content: str
    source: str = "auto:prompt"
    project: str = "unknown"
    role: str = "user"
    session_id: str | None = None
    conversation_date: str | None = None


class ObserveImageRequest(BaseModel):
    image_data: str  # base64 encoded
    mime_type: str = "image/png"
    description: str = ""
    source: str = "api"


class ObserveFileRequest(BaseModel):
    file_data: str  # base64 encoded
    mime_type: str  # required
    description: str = ""
    source: str = "api"


class RememberBody(BaseModel):
    content: str
    source: str = "dashboard"
    conversation_date: str | None = None
    proposed_entities: list[dict] | None = None
    proposed_relationships: list[dict] | None = None
    model_tier: str = "default"


class AdjudicateBody(BaseModel):
    request_id: str
    entities: list[dict] | None = None
    relationships: list[dict] | None = None
    reject_evidence_ids: list[str] | None = None
    model_tier: str = "default"
    rationale: str | None = None


class FactRef(BaseModel):
    subject: str
    predicate: str
    object: str


class IntendBody(BaseModel):
    trigger_text: str
    action_text: str
    trigger_type: str = "activation"
    entity_names: list[str] | None = None
    threshold: float | None = None
    priority: str = "normal"
    context: str | None = None
    see_also: list[str] | None = None
    refresh_trigger: str = "manual"


class BootstrapBody(BaseModel):
    project_path: str
    session_id: str | None = None


class ForgetBody(BaseModel):
    entity_name: str | None = None
    fact: FactRef | None = None
    reason: str | None = None


class FeedbackBody(BaseModel):
    entity_id: str
    rating: int
    comment: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    message: str
    history: list[ChatMessage] | None = None
    conversation_id: str | None = None
    session_date: str | None = None


class RouteBody(BaseModel):
    question: str
    project_path: str | None = None
    history: list[ChatMessage] | None = None


def _get_conv_top_entity_names(manager) -> list[str]:
    return manager_conversation_top_entity_names(manager)


# ─── Endpoints ───────────────────────────────────────────────────


@router.post("/observe")
async def observe(request: Request, body: ObserveBody) -> JSONResponse:
    """Store content without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    episode_id = await store_observation(
        manager,
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=parse_conversation_date(body.conversation_date),
        pass_conversation_date=True,
    )

    return JSONResponse(
        content=present_api_memory_write(
            memory_write_contract("observe", episode_id),
            status="observed",
        ),
    )


@router.post("/auto-observe")
async def auto_observe(request: Request, body: AutoObserveBody) -> JSONResponse:
    """Auto-capture endpoint with dedup. Used by Claude Code hooks."""
    tenant = get_tenant(request)
    group_id = tenant.group_id

    if not get_config().server.auto_observe_enabled:
        return JSONResponse(
            content=present_api_observe_skip("skipped", reason="disabled"),
        )

    if not body.content or len(body.content.strip()) < 10:
        return JSONResponse(
            content=present_api_observe_skip("skipped", reason="too_short"),
        )

    if _dedup_check(body.content):
        return JSONResponse(content=present_api_observe_skip("dedup_skipped"))

    manager = get_manager()

    episode_id = await store_observation(
        manager,
        content=body.content,
        group_id=group_id,
        source=body.source,
        session_id=body.session_id,
        conversation_date=parse_conversation_date(body.conversation_date),
        pass_conversation_date=True,
    )

    return JSONResponse(
        content=present_api_memory_write(
            memory_write_contract("observe", episode_id),
            status="observed",
        ),
    )


@router.post("/observe-image")
async def observe_image(request: Request, body: ObserveImageRequest) -> JSONResponse:
    """Store an image observation without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    attachment = build_observation_attachment(
        mime_type=body.mime_type,
        data_url=body.image_data,
        description=body.description,
    )
    episode_id = await store_observation(
        manager,
        content=body.description or f"[image: {body.mime_type}]",
        group_id=group_id,
        source=body.source,
        attachments=[attachment],
    )

    return JSONResponse(
        content=present_api_memory_write(
            memory_write_contract("observe", episode_id, attachment_kind="image"),
            status="stored",
            include_legacy_episode_id=True,
        ),
    )


@router.post("/observe-file")
async def observe_file(request: Request, body: ObserveFileRequest) -> JSONResponse:
    """Store a file observation without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    attachment = build_observation_attachment(
        mime_type=body.mime_type,
        data_url=body.file_data,
        description=body.description,
    )
    episode_id = await store_observation(
        manager,
        content=body.description or f"[file: {body.mime_type}]",
        group_id=group_id,
        source=body.source,
        attachments=[attachment],
    )

    return JSONResponse(
        content=present_api_memory_write(
            memory_write_contract("observe", episode_id, attachment_kind="file"),
            status="stored",
            include_legacy_episode_id=True,
        ),
    )


@router.post("/replay-queue")
async def replay_queue(request: Request) -> JSONResponse:
    """Replay entries from the offline capture queue (~/.engram/capture-queue.jsonl).

    Drains the queue file atomically and ingests each entry via store_episode().
    Returns the count of replayed entries.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_manager_offline_replay_surface(
        manager,
        drain_queue=drain_queue,
        dedup_check=_dedup_check,
        group_id=group_id,
    )
    return JSONResponse(content=payload)


@router.post("/remember")
async def remember(request: Request, body: RememberBody) -> JSONResponse:
    """Ingest content with full extraction (slow path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    episode_id = await ingest_projecting_memory(
        manager,
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=parse_conversation_date(body.conversation_date),
        proposed_entities=body.proposed_entities,
        proposed_relationships=body.proposed_relationships,
        model_tier=body.model_tier,
    )
    adjudications = await load_client_enabled_episode_adjudication_requests(
        manager,
        episode_id=episode_id,
        group_id=group_id,
    )
    return JSONResponse(
        content=present_api_memory_write(
            memory_write_contract(
                "remember",
                episode_id,
                adjudication_requests=adjudications,
            ),
            status="remembered",
        ),
    )


@router.post("/adjudicate")
async def adjudicate(request: Request, body: AdjudicateBody) -> JSONResponse:
    """Resolve a previously created edge-adjudication request."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_adjudication_resolution_surface(
        manager,
        group_id=group_id,
        request_id=body.request_id,
        entities=body.entities,
        relationships=body.relationships,
        reject_evidence_ids=body.reject_evidence_ids,
        model_tier=body.model_tier,
        rationale=body.rationale,
    )
    return JSONResponse(content=payload)


@router.get("/recall")
async def recall(
    request: Request,
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=50, description="Max results"),
) -> JSONResponse:
    """Retrieve relevant memories using activation-aware scoring."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    payload = await build_api_recall_surface(
        manager,
        group_id=group_id,
        query=q,
        limit=limit,
    )
    return JSONResponse(content=payload)


@router.get("/facts")
async def search_facts(
    request: Request,
    q: str = Query("", description="Search query"),
    subject: str | None = Query(None, description="Filter by subject entity"),
    predicate: str | None = Query(None, description="Filter by predicate"),
    include_expired: bool = Query(False, description="Include expired facts"),
    include_epistemic: bool = Query(
        False,
        description="Include internal epistemic graph facts such as decision/documentation edges",
    ),
    limit: int = Query(10, ge=1, le=100, description="Max results"),
) -> JSONResponse:
    """Search for facts/relationships in the knowledge graph."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_fact_search_surface(
        manager,
        group_id=group_id,
        query=q,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )

    return JSONResponse(content=payload)


@router.get("/context")
async def get_context(
    request: Request,
    max_tokens: int = Query(2000, ge=100, le=10000, description="Token budget"),
    topic_hint: str | None = Query(None, description="Topic to bias context toward"),
    project_path: str | None = Query(None, description="Project directory for auto topic hint"),
    format: str = Query("structured", description="Output format: structured or briefing"),
) -> JSONResponse:
    """Get a pre-assembled context summary of active memories."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_context_surface(
        manager,
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
    )

    return JSONResponse(content=payload)


@router.post("/forget")
async def forget(request: Request, body: ForgetBody) -> JSONResponse:
    """Forget an entity or a specific fact."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await build_api_forget_response_surface(
        manager,
        group_id=group_id,
        entity_name=body.entity_name,
        fact=body.fact,
        reason=body.reason,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


@router.post("/feedback")
async def post_feedback(request: Request, body: FeedbackBody) -> JSONResponse:
    """Record explicit user feedback on an entity."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    result = await build_api_explicit_feedback_surface(
        manager,
        group_id=group_id,
        entity_id=body.entity_id,
        rating=body.rating,
        comment=body.comment,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


@router.post("/bootstrap")
async def bootstrap_project(request: Request, body: BootstrapBody) -> JSONResponse:
    """Bootstrap a project: create Project entity and observe key files. Idempotent."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await build_project_bootstrap_surface(
        manager,
        group_id=group_id,
        project_path=body.project_path,
        session_id=body.session_id,
    )

    return JSONResponse(status_code=project_bootstrap_http_status(result), content=result)


@router.post("/route")
async def route_knowledge_question(request: Request, body: RouteBody) -> JSONResponse:
    """Return the deterministic epistemic route for a question."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    session_entity_names = _get_conv_top_entity_names(manager)
    result = await build_question_route_surface(
        manager,
        group_id=group_id,
        question=body.question,
        project_path=body.project_path,
        history=body.history,
        session_entity_names=session_entity_names,
        surface="rest",
    )
    return JSONResponse(content=result)


@router.get("/artifacts/search")
async def search_artifacts(
    request: Request,
    q: str = Query(..., min_length=1, description="Artifact search query"),
    project_path: str | None = Query(None, description="Optional project path filter"),
    limit: int = Query(5, ge=1, le=20),
) -> JSONResponse:
    """Search bootstrapped project artifacts."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    payload = await build_api_artifact_search_surface(
        manager,
        group_id=group_id,
        query=q,
        project_path=project_path,
        limit=limit,
    )
    return JSONResponse(content=payload)


@router.get("/runtime")
async def get_runtime_state(
    request: Request,
    project_path: str | None = Query(None, description="Optional project path context"),
) -> JSONResponse:
    """Return effective runtime/config state and artifact freshness."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    result = await build_runtime_state_surface(
        manager,
        group_id=group_id,
        project_path=project_path,
    )
    return JSONResponse(content=result)


@router.post("/intentions")
async def create_intention(request: Request, body: IntendBody) -> JSONResponse:
    """Create a graph-embedded intention (prospective memory trigger)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await build_api_create_intention_response_surface(
        manager,
        group_id=group_id,
        trigger_text=body.trigger_text,
        action_text=body.action_text,
        trigger_type=body.trigger_type,
        entity_names=body.entity_names,
        threshold=body.threshold,
        priority=body.priority,
        context=body.context,
        see_also=body.see_also,
        refresh_trigger=body.refresh_trigger,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


@router.get("/intentions")
async def list_intentions(
    request: Request,
    enabled_only: bool = Query(True, description="Filter to enabled intentions only"),
) -> JSONResponse:
    """List active prospective memory intentions."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_intention_list_surface(
        manager,
        group_id=group_id,
        enabled_only=enabled_only,
        surface="api",
    )
    return JSONResponse(content=payload)


@router.delete("/intentions/{intention_id}")
async def dismiss_intention(
    request: Request,
    intention_id: str,
    hard: bool = Query(False, description="Permanently delete instead of soft-disable"),
) -> JSONResponse:
    """Dismiss (disable or delete) a prospective memory intention."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await build_api_dismiss_intention_response_surface(
        manager,
        group_id=group_id,
        intention_id=intention_id,
        hard=hard,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


def _sse(data: dict) -> str:
    """Format a single SSE event line."""
    return f"data: {json.dumps(data)}\n\n"


MAX_TOOL_TURNS = 3  # Cap agentic loop iterations
MAX_HISTORY_MESSAGES = DEFAULT_MAX_HISTORY_MESSAGES  # Sliding window for cost control

@router.post("/chat", response_model=None)
async def chat(request: Request, body: ChatBody) -> StreamingResponse | JSONResponse:
    """Stream a chat response using AI SDK v6 UIMessageStream protocol.

    Uses an agentic tool-use loop so the LLM can query the memory graph
    autonomously (recall, search_entities, search_facts) before answering.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id

    # Rate limit chat endpoint (guards against runaway API costs)
    rate_limiter = get_rate_limiter()
    if rate_limiter:
        allowed, remaining = await rate_limiter.check(group_id, "chat")
        if not allowed:
            result = build_api_chat_rate_limit_surface(remaining)
            return JSONResponse(
                status_code=result.status_code,
                content=result.payload,
            )

    manager = get_manager()
    chat_policy = build_chat_runtime_policy(manager)

    conversation_id = body.conversation_id
    try:
        conv_store = get_conversation_store()
    except RuntimeError:
        conv_store = None

    conversation = await resolve_chat_conversation(
        conv_store,
        group_id=group_id,
        conversation_id=conversation_id,
        message=body.message,
        session_date=body.session_date,
    )
    if conversation.not_found:
        return JSONResponse(status_code=404, content=chat_conversation_not_found_payload())
    conversation_id = conversation.conversation_id

    await hydrate_chat_context(manager, body.history, body.message)

    chat_need: MemoryNeed | None = None
    epistemic_bundle = None
    topic_hint: str | None = body.message
    session_entity_names = _get_conv_top_entity_names(manager)

    if chat_policy.recall_need_analyzer_enabled:
        chat_need = await analyze_chat_memory_need(
            body.message,
            manager,
            body.history,
            session_entity_names=session_entity_names,
            group_id=group_id,
        )
        await record_manager_memory_need_analysis(manager, group_id, chat_need)
        if chat_policy.recall_telemetry_enabled:
            publish_memory_need_analysis(
                get_event_bus(),
                group_id,
                chat_need,
                source="knowledge_chat",
                mode="chat",
                turn_text=body.message,
            )
        topic_hint = chat_need.query_hint if chat_need.should_recall else None

    if chat_policy.epistemic_routing_enabled:
        epistemic_bundle = await gather_chat_epistemic_evidence(
            manager,
            message=body.message,
            group_id=group_id,
            history=body.history,
            session_entity_names=session_entity_names,
            memory_need=chat_need,
        )

    # Baseline context for the system prompt (1000 token budget to control costs)
    context_result = await build_chat_context_surface(
        manager,
        group_id=group_id,
        max_tokens=1000,
        topic_hint=topic_hint,
    )

    system_prompt = build_chat_system_prompt_surface(
        context=context_result["context"],
        memory_need=chat_need,
        epistemic_bundle=epistemic_bundle,
    )
    messages = build_chat_messages(
        body.history,
        body.message,
        max_history_messages=MAX_HISTORY_MESSAGES,
    )

    text_part_id = "text_0"

    async def event_stream():
        try:
            yield _sse({"type": "start"})
            yield _sse({"type": "start-step"})

            client = anthropic.AsyncAnthropic()

            tool_loop = await run_chat_tool_use_loop(
                client,
                manager=manager,
                group_id=group_id,
                system_prompt=cast(Any, system_prompt),
                messages=cast(Any, messages),
                tools=cast(Any, CHAT_TOOLS),
                initial_recall_results=list(getattr(epistemic_bundle, "memory_results", []) or []),
                max_tool_turns=MAX_TOOL_TURNS,
            )
            response = tool_loop.response
            loop_messages = tool_loop.loop_messages
            all_recall_results = tool_loop.recall_results
            all_facts = tool_loop.facts

            # Emit synthetic tool events for UI components
            for tool_event in build_chat_tool_stream_events(all_recall_results, all_facts):
                yield _sse(tool_event)

            # Stream the final text response
            final_text = extract_message_text(response.content)

            if final_text:
                if should_retry_chat_response(
                    manager,
                    chat_need=chat_need,
                    response_text=final_text,
                    recall_results=all_recall_results,
                ):
                    if chat_need is None:
                        raise RuntimeError("chat_need missing during retry path")
                    final_text = await retry_memory_grounded_response(
                        client,
                        system_prompt=system_prompt,
                        loop_messages=loop_messages,
                        chat_need=chat_need,
                        prior_response=final_text,
                    )
                await apply_chat_recall_feedback(
                    manager,
                    group_id=group_id,
                    query=body.message,
                    response_text=final_text,
                    recall_results=all_recall_results,
                )
                await record_chat_assistant_turn(manager, final_text)
                yield _sse({"type": "text-start", "id": text_part_id})
                # Emit in chunks for streaming feel
                chunk_size = 20
                for i in range(0, len(final_text), chunk_size):
                    yield _sse(
                        {
                            "type": "text-delta",
                            "id": text_part_id,
                            "delta": final_text[i : i + chunk_size],
                        }
                    )
                yield _sse({"type": "text-end", "id": text_part_id})

            yield _sse({"type": "finish-step"})
            yield _sse(
                {
                    "type": "finish",
                    "finishReason": "stop",
                    **({"conversationId": conversation_id} if conversation_id else {}),
                }
            )

            # Fire-and-forget: persist messages + tag entities
            if conv_store and conversation_id and final_text:

                async def _persist():
                    try:
                        await persist_chat_turn(
                            conv_store,
                            conversation_id=conversation_id,
                            group_id=group_id,
                            user_message=body.message,
                            assistant_message=final_text,
                            recall_results=all_recall_results,
                        )
                    except Exception:
                        logger.warning("Failed to persist chat messages", exc_info=True)

                asyncio.create_task(_persist())

        except Exception as e:
            logger.exception("Chat stream error")
            yield _sse({"type": "error", "errorText": str(e)})
            yield _sse({"type": "finish", "finishReason": "error"})

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
