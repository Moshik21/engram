"""Knowledge Management API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engram.api.deps import (
    get_config,
    get_manager,
    get_notification_surface_service,
    get_optional_conversation_store,
    get_rate_limiter,
)
from engram.events.bus import get_event_bus
from engram.ingestion.adjudication_surface import build_api_adjudication_resolution_surface
from engram.ingestion.capture_surface import (
    build_api_attachment_observe_write_surface,
    build_api_auto_observe_request_surface,
    build_api_observe_write_surface,
    build_api_remember_write_surface,
)
from engram.ingestion.dedup import CaptureDedupCache
from engram.ingestion.offline_replay import build_api_manager_offline_replay_surface
from engram.ingestion.project_bootstrap import (
    build_project_bootstrap_surface,
    project_bootstrap_http_status,
)
from engram.notifications.surface import (
    build_api_notification_dismiss_surface,
    build_api_notifications_surface,
)
from engram.retrieval.artifacts import build_api_artifact_search_surface
from engram.retrieval.chat_runtime import (
    build_api_chat_stream_response_surface,
)
from engram.retrieval.context import (
    manager_conversation_top_entity_names,
)
from engram.retrieval.context_builder import build_api_context_surface
from engram.retrieval.epistemic_route import build_question_route_surface
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

    payload = await build_api_observe_write_surface(
        manager,
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=body.conversation_date,
    )
    return JSONResponse(content=payload)


@router.post("/auto-observe")
async def auto_observe(request: Request) -> JSONResponse:
    """Auto-capture endpoint with dedup. Used by Claude Code hooks."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_auto_observe_request_surface(
        manager,
        request=request,
        group_id=group_id,
        auto_observe_enabled=get_config().server.auto_observe_enabled,
        dedup_check=_dedup_check,
    )
    return JSONResponse(content=payload)


@router.post("/observe-image")
async def observe_image(request: Request, body: ObserveImageRequest) -> JSONResponse:
    """Store an image observation without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_attachment_observe_write_surface(
        manager,
        data_url=body.image_data,
        mime_type=body.mime_type,
        attachment_kind="image",
        fallback_content=f"[image: {body.mime_type}]",
        group_id=group_id,
        description=body.description,
        source=body.source,
    )
    return JSONResponse(content=payload)


@router.post("/observe-file")
async def observe_file(request: Request, body: ObserveFileRequest) -> JSONResponse:
    """Store a file observation without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_attachment_observe_write_surface(
        manager,
        data_url=body.file_data,
        mime_type=body.mime_type,
        attachment_kind="file",
        fallback_content=f"[file: {body.mime_type}]",
        group_id=group_id,
        description=body.description,
        source=body.source,
    )
    return JSONResponse(content=payload)


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

    payload = await build_api_remember_write_surface(
        manager,
        content=body.content,
        group_id=group_id,
        source=body.source,
        conversation_date=body.conversation_date,
        proposed_entities=body.proposed_entities,
        proposed_relationships=body.proposed_relationships,
        model_tier=body.model_tier,
    )
    return JSONResponse(content=payload)


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


@router.post("/chat", response_model=None)
async def chat(request: Request, body: ChatBody) -> StreamingResponse | JSONResponse:
    """Stream a chat response using AI SDK v6 UIMessageStream protocol.

    Uses an agentic tool-use loop so the LLM can query the memory graph
    autonomously (recall, search_entities, search_facts) before answering.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id
    surface = await build_api_chat_stream_response_surface(
        get_manager(),
        group_id=group_id,
        message=body.message,
        history=body.history,
        conversation_store=get_optional_conversation_store(),
        conversation_id=body.conversation_id,
        session_date=body.session_date,
        rate_limiter=get_rate_limiter(),
        event_bus=get_event_bus(),
        stream_logger=logger,
    )

    if surface.payload is not None:
        return JSONResponse(
            status_code=surface.status_code,
            content=surface.payload,
        )

    return StreamingResponse(
        surface.stream,
        media_type=surface.media_type,
    )
