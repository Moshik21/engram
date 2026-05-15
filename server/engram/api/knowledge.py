"""Knowledge Management API endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
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
from engram.ingestion.adjudication_surface import load_episode_adjudication_requests
from engram.ingestion.dedup import CaptureDedupCache
from engram.ingestion.offline_replay import OfflineReplayService
from engram.ingestion.presenter import (
    memory_write_contract,
    present_api_memory_write,
    present_api_observe_skip,
)
from engram.models.episode import Attachment
from engram.models.recall import MemoryNeed
from engram.retrieval.artifacts import build_api_artifact_search_surface
from engram.retrieval.chat_events import build_chat_tool_events, raw_recall_from_chat_item
from engram.retrieval.chat_feedback import (
    apply_chat_recall_feedback,
    build_memory_grounding_retry_system_prompt,
    should_retry_chat_response,
)
from engram.retrieval.chat_persistence import (
    persist_chat_turn,
    resolve_chat_conversation,
)
from engram.retrieval.chat_runtime import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    analyze_chat_memory_need,
    build_chat_memory_guidance,
    hydrate_chat_context,
    recent_chat_turn_contents,
    record_chat_assistant_turn,
)
from engram.retrieval.chat_tools import execute_chat_tool
from engram.retrieval.context import (
    manager_conversation_top_entity_names,
)
from engram.retrieval.control import record_manager_memory_need_analysis
from engram.retrieval.epistemic import render_epistemic_summary
from engram.retrieval.epistemic_route import build_question_route_surface
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.forgetting import build_api_forget_surface
from engram.retrieval.preference_feedback import (
    FeedbackRatingError,
    build_explicit_feedback_surface,
)
from engram.retrieval.prospective import (
    build_api_create_intention_surface,
    build_api_dismiss_intention_surface,
    build_intention_list_surface,
)
from engram.retrieval.recall_surface import build_api_recall_surface
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
    if service is None:
        return JSONResponse(content={"notifications": []})

    return JSONResponse(
        content={
            "notifications": service.list_notifications(
                group_id=group_id,
                limit=limit,
                since=since,
            )
        }
    )


@router.post("/notifications/dismiss")
async def dismiss_notifications(request: Request, body: DismissBody) -> JSONResponse:
    """Dismiss one or more notifications by ID."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    service = get_notification_surface_service()
    if service is None:
        return JSONResponse(content={"dismissed": 0})

    count = service.dismiss_notifications(group_id=group_id, ids=body.ids)
    return JSONResponse(content={"dismissed": count})


# ─── Dedup cache for auto-observe ────────────────────────────────

_DEDUP = CaptureDedupCache(ttl_seconds=300.0, max_entries=1000)
_DEDUP_CACHE = _DEDUP.cache  # compatibility handle for existing tests
_DEDUP_TTL = _DEDUP.ttl_seconds


def _extract_message_text(blocks: object) -> str:
    """Join text-bearing Anthropic blocks without assuming concrete block types."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


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

    conv_dt = None
    if body.conversation_date:
        try:
            conv_dt = datetime.fromisoformat(body.conversation_date)
        except (ValueError, TypeError):
            pass
    episode_id = await manager.store_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=conv_dt,
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

    conv_dt = None
    if body.conversation_date:
        try:
            conv_dt = datetime.fromisoformat(body.conversation_date)
        except (ValueError, TypeError):
            pass
    episode_id = await manager.store_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
        session_id=body.session_id,
        conversation_date=conv_dt,
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

    attachment = Attachment(
        mime_type=body.mime_type,
        data_url=body.image_data,
        description=body.description,
    )
    episode_id = await manager.store_episode(
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

    attachment = Attachment(
        mime_type=body.mime_type,
        data_url=body.file_data,
        description=body.description,
    )
    episode_id = await manager.store_episode(
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

    replay_service = OfflineReplayService(
        drain_queue=drain_queue,
        dedup_check=_dedup_check,
        store_episode=manager.store_episode,
    )
    result = await replay_service.replay_queue(group_id=group_id)

    return JSONResponse(
        content={
            "status": "replayed",
            **result.as_payload(),
        }
    )


@router.post("/remember")
async def remember(request: Request, body: RememberBody) -> JSONResponse:
    """Ingest content with full extraction (slow path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    conv_dt = None
    if body.conversation_date:
        try:
            conv_dt = datetime.fromisoformat(body.conversation_date)
        except (ValueError, TypeError):
            pass
    episode_id = await manager.ingest_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=conv_dt,
        proposed_entities=body.proposed_entities,
        proposed_relationships=body.proposed_relationships,
        model_tier=body.model_tier,
    )
    adjudications: list[dict] = []
    if manager.edge_adjudication_client_enabled():
        adjudications = await load_episode_adjudication_requests(
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

    outcome = await manager.submit_adjudication_resolution(
        body.request_id,
        entities=body.entities,
        relationships=body.relationships,
        reject_evidence_ids=body.reject_evidence_ids,
        source="client_adjudication",
        model_tier=body.model_tier,
        rationale=body.rationale,
        group_id=group_id,
    )
    return JSONResponse(
        content={
            "status": outcome.status,
            "requestId": outcome.request_id,
            "committedIds": outcome.committed_ids,
            "supersededEvidenceIds": outcome.superseded_evidence_ids,
            "replacementEvidenceIds": outcome.replacement_evidence_ids,
        },
    )


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

    results = await manager.search_facts(
        group_id=group_id,
        query=q,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )

    items = []
    for r in results:
        items.append(
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "validFrom": r.get("valid_from"),
                "validTo": r.get("valid_to"),
                "confidence": r.get("confidence"),
                "sourceEpisode": r.get("source_episode"),
                "createdAt": r.get("created_at"),
            }
        )

    return JSONResponse(content={"items": items})


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

    result = await manager.get_context(
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
    )

    return JSONResponse(
        content={
            "context": result["context"],
            "entityCount": result["entity_count"],
            "factCount": result["fact_count"],
            "tokenEstimate": result["token_estimate"],
            "format": result.get("format", "structured"),
        }
    )


@router.post("/forget")
async def forget(request: Request, body: ForgetBody) -> JSONResponse:
    """Forget an entity or a specific fact."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    try:
        result = await build_api_forget_surface(
            manager,
            group_id=group_id,
            entity_name=body.entity_name,
            fact=body.fact,
            reason=body.reason,
        )
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Provide either entity_name or fact."},
        )

    status_code = 200 if result.get("status") != "error" else 404
    return JSONResponse(status_code=status_code, content=result)


@router.post("/feedback")
async def post_feedback(request: Request, body: FeedbackBody) -> JSONResponse:
    """Record explicit user feedback on an entity."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    try:
        result = await build_explicit_feedback_surface(
            manager,
            group_id=group_id,
            entity_id=body.entity_id,
            rating=body.rating,
            comment=body.comment,
        )
        return JSONResponse(result)
    except FeedbackRatingError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@router.post("/bootstrap")
async def bootstrap_project(request: Request, body: BootstrapBody) -> JSONResponse:
    """Bootstrap a project: create Project entity and observe key files. Idempotent."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await manager.bootstrap_project(
        project_path=body.project_path,
        group_id=group_id,
        session_id=body.session_id,
    )

    status_code = 200 if result.get("status") != "skipped" else 400
    return JSONResponse(status_code=status_code, content=result)


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
    result = await manager.get_runtime_state(
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

    try:
        payload = await build_api_create_intention_surface(
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
        return JSONResponse(content=payload)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})


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

    try:
        payload = await build_api_dismiss_intention_surface(
            manager,
            group_id=group_id,
            intention_id=intention_id,
            hard=hard,
        )
        return JSONResponse(content=payload)
    except Exception:
        return JSONResponse(status_code=404, content={"detail": "Intention not found"})


def _sse(data: dict) -> str:
    """Format a single SSE event line."""
    return f"data: {json.dumps(data)}\n\n"


def _emit_tool(tool_call_id: str, tool_name: str, input_data: dict) -> str:
    """Emit AI SDK v6 synthetic tool call (input-available + output-available)."""
    inp = _sse(
        {
            "type": "tool-input-available",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "input": input_data,
            "dynamic": True,
        }
    )
    out = _sse(
        {
            "type": "tool-output-available",
            "toolCallId": tool_call_id,
            "output": "displayed",
            "dynamic": True,
        }
    )
    return inp + out


MAX_TOOL_TURNS = 3  # Cap agentic loop iterations
MAX_HISTORY_MESSAGES = DEFAULT_MAX_HISTORY_MESSAGES  # Sliding window for cost control

CHAT_TOOLS = [
    {
        "name": "recall",
        "description": (
            "Search memories by semantic similarity + activation. "
            "Returns scored entities with summaries and relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-20)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_entities",
        "description": (
            "Find specific entities by name (fuzzy match) or type. "
            "Use when you know or suspect the name of an entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name to search for"},
                "entity_type": {
                    "type": "string",
                    "description": "Filter by type (Person, Technology, etc.)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
            },
        },
    },
    {
        "name": "search_facts",
        "description": (
            "Search for relationships/facts in the knowledge graph. "
            "Can filter by subject entity name and/or predicate type "
            "(e.g., PARENT_OF, WORKS_AT, LIVES_IN). Internal epistemic "
            "decision/artifact edges stay hidden unless include_epistemic=true."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "default": "",
                },
                "subject": {
                    "type": "string",
                    "description": "Filter by subject entity name",
                },
                "predicate": {
                    "type": "string",
                    "description": "Filter by relationship type",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results",
                    "default": 10,
                },
                "include_epistemic": {
                    "type": "boolean",
                    "description": (
                        "Debug-only: include decision/artifact graph facts. "
                        "Do not use this as the primary path for project reconcile answers."
                    ),
                    "default": False,
                },
            },
        },
    },
]


async def _execute_tool(manager, group_id: str, tool_name: str, tool_input: dict) -> str:
    """Execute a chat tool call against the manager. Returns JSON string."""
    payload = await execute_chat_tool(
        manager,
        group_id=group_id,
        tool_name=tool_name,
        tool_input=tool_input,
    )
    return json.dumps(payload)


async def _retry_memory_grounded_response(
    client,
    *,
    system_prompt: list[dict],
    loop_messages: list[dict],
    chat_need: MemoryNeed,
    prior_response: str,
) -> str:
    """Retry once with a stronger memory-grounding instruction."""
    retry_system = build_memory_grounding_retry_system_prompt(
        system_prompt,
        chat_need=chat_need,
        prior_response=prior_response,
    )
    retry_response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=cast(Any, retry_system),
        messages=cast(Any, loop_messages),
    )
    retry_text = _extract_message_text(retry_response.content)
    return retry_text or prior_response


def _build_tool_events(recall_results: list, facts: list) -> str:
    """Analyze recall results and emit synthetic tool events for rich components."""
    return "".join(
        _emit_tool(f"tc_{idx}", event.name, event.input)
        for idx, event in enumerate(build_chat_tool_events(recall_results, facts), start=1)
    )


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
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded for chat", "remaining": remaining},
            )

    manager = get_manager()
    chat_policy = manager.get_chat_runtime_policy()

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
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
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
        epistemic_bundle = await manager.gather_epistemic_evidence(
            body.message,
            group_id=group_id,
            project_path=None,
            recent_turns=recent_chat_turn_contents(body.history, limit=6),
            session_entity_names=session_entity_names,
            surface="rest",
            memory_need=chat_need,
        )

    # Baseline context for the system prompt (1000 token budget to control costs)
    context_result = await manager.get_context(
        group_id=group_id,
        max_tokens=1000,
        topic_hint=topic_hint,
    )

    # Build system prompt with memory context + tool-use guidance
    memory_guidance = (
        build_chat_memory_guidance(chat_need)
        if chat_need is not None
        else (
            "Use memory tools when prior context matters. Do not guess when tools "
            "can provide precise answers."
        )
    )
    static_preamble = (
        "You are a helpful assistant with access to the user's memory graph. "
        "You have tools to search the graph — use them to answer questions accurately.\n\n"
        "Guidelines:\n"
        "- Use search_facts for user-facing relationships like family members or work history\n"
        "- Do not use search_facts as the primary path for project reconcile questions "
        "when artifacts or runtime are required\n"
        "- Use search_entities to find entities by name\n"
        "- Use recall for general semantic search\n"
        f"- {memory_guidance}\n\n"
        "Below is baseline context about the user:\n\n"
    )
    system_prompt = [
        {
            "type": "text",
            "text": static_preamble,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": context_result["context"],
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

    # Build messages (sliding window to control token costs)
    messages: list[dict] = []
    if body.history:
        history = body.history
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": body.message})

    text_part_id = "text_0"

    async def event_stream():
        try:
            yield _sse({"type": "start"})
            yield _sse({"type": "start-step"})

            client = anthropic.AsyncAnthropic()

            # Accumulate tool results for UI components
            all_recall_results: list[dict] = list(
                getattr(epistemic_bundle, "memory_results", []) or []
            )
            all_facts: list[dict] = []

            # --- Agentic tool-use loop (non-streaming) ---
            loop_messages = list(messages)
            for _turn in range(MAX_TOOL_TURNS):
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=cast(Any, system_prompt),
                    messages=cast(Any, loop_messages),
                    tools=cast(Any, CHAT_TOOLS),
                )

                if response.stop_reason != "tool_use":
                    # Final response — extract text and stream it
                    break

                # Process tool calls
                assistant_content = response.content
                tool_results = []
                for block in assistant_content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input

                    # Execute tool
                    try:
                        result_str = await _execute_tool(
                            manager,
                            group_id,
                            tool_name,
                            tool_input,
                        )
                    except Exception as e:
                        logger.warning("Chat tool %s failed: %s", tool_name, e)
                        result_str = json.dumps({"error": str(e)})

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

                    # Accumulate for UI components
                    try:
                        parsed = json.loads(result_str)
                        if tool_name == "recall":
                            for item in parsed.get("results", []):
                                all_recall_results.append(raw_recall_from_chat_item(item))
                        elif tool_name == "search_facts":
                            all_facts.extend(parsed.get("facts", []))
                    except Exception:
                        pass

                # Append assistant + tool results for next turn
                loop_messages.append({"role": "assistant", "content": assistant_content})
                loop_messages.append({"role": "user", "content": tool_results})
            else:
                # Exhausted turns — do one final call without tools
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=cast(Any, system_prompt),
                    messages=cast(Any, loop_messages),
                )

            # Emit synthetic tool events for UI components
            tool_events = _build_tool_events(all_recall_results, all_facts)
            if tool_events:
                yield tool_events

            # Stream the final text response
            final_text = _extract_message_text(response.content)

            if final_text:
                if should_retry_chat_response(
                    manager,
                    chat_need=chat_need,
                    response_text=final_text,
                    recall_results=all_recall_results,
                ):
                    if chat_need is None:
                        raise RuntimeError("chat_need missing during retry path")
                    final_text = await _retry_memory_grounded_response(
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
