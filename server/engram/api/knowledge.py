"""Knowledge Management API endpoints."""

from __future__ import annotations

import json
import logging

import anthropic
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engram.api.deps import get_manager
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# ─── Request bodies ──────────────────────────────────────────────


class ObserveBody(BaseModel):
    content: str
    source: str = "dashboard"


class RememberBody(BaseModel):
    content: str
    source: str = "dashboard"


class FactRef(BaseModel):
    subject: str
    predicate: str
    object: str


class ForgetBody(BaseModel):
    entity_name: str | None = None
    fact: FactRef | None = None
    reason: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    message: str
    history: list[ChatMessage] | None = None


# ─── Endpoints ───────────────────────────────────────────────────


@router.post("/observe")
async def observe(request: Request, body: ObserveBody) -> JSONResponse:
    """Store content without extraction (fast path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    episode_id = await manager.store_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
    )

    return JSONResponse(content={"status": "observed", "episodeId": episode_id})


@router.post("/remember")
async def remember(request: Request, body: RememberBody) -> JSONResponse:
    """Ingest content with full extraction (slow path)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    episode_id = await manager.ingest_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
    )

    return JSONResponse(content={"status": "remembered", "episodeId": episode_id})


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

    results = await manager.recall(query=q, group_id=group_id, limit=limit)

    items = []
    for r in results:
        result_type = r.get("result_type", "entity")
        if result_type == "episode":
            ep = r["episode"]
            items.append({
                "resultType": "episode",
                "episode": {
                    "id": ep["id"],
                    "content": ep["content"],
                    "source": ep.get("source"),
                    "createdAt": ep.get("created_at"),
                },
                "score": r["score"],
                "scoreBreakdown": {
                    "semantic": r["score_breakdown"]["semantic"],
                    "activation": r["score_breakdown"]["activation"],
                    "edgeProximity": r["score_breakdown"]["edge_proximity"],
                    "explorationBonus": r["score_breakdown"]["exploration_bonus"],
                },
            })
        else:
            ent = r["entity"]
            items.append({
                "resultType": "entity",
                "entity": {
                    "id": ent["id"],
                    "name": ent["name"],
                    "entityType": ent["type"],
                    "summary": ent.get("summary"),
                },
                "score": r["score"],
                "scoreBreakdown": {
                    "semantic": r["score_breakdown"]["semantic"],
                    "activation": r["score_breakdown"]["activation"],
                    "edgeProximity": r["score_breakdown"]["edge_proximity"],
                    "explorationBonus": r["score_breakdown"]["exploration_bonus"],
                },
                "relationships": r.get("relationships", []),
            })

    return JSONResponse(content={"items": items, "query": q})


@router.get("/facts")
async def search_facts(
    request: Request,
    q: str = Query("", description="Search query"),
    subject: str | None = Query(None, description="Filter by subject entity"),
    predicate: str | None = Query(None, description="Filter by predicate"),
    include_expired: bool = Query(False, description="Include expired facts"),
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
        limit=limit,
    )

    items = []
    for r in results:
        items.append({
            "subject": r["subject"],
            "predicate": r["predicate"],
            "object": r["object"],
            "validFrom": r.get("valid_from"),
            "validTo": r.get("valid_to"),
            "confidence": r.get("confidence"),
            "sourceEpisode": r.get("source_episode"),
            "createdAt": r.get("created_at"),
        })

    return JSONResponse(content={"items": items})


@router.get("/context")
async def get_context(
    request: Request,
    max_tokens: int = Query(2000, ge=100, le=10000, description="Token budget"),
    topic_hint: str | None = Query(None, description="Topic to bias context toward"),
) -> JSONResponse:
    """Get a pre-assembled context summary of active memories."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await manager.get_context(
        group_id=group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
    )

    return JSONResponse(content={
        "context": result["context"],
        "entityCount": result["entity_count"],
        "factCount": result["fact_count"],
        "tokenEstimate": result["token_estimate"],
    })


@router.post("/forget")
async def forget(request: Request, body: ForgetBody) -> JSONResponse:
    """Forget an entity or a specific fact."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    if body.entity_name:
        result = await manager.forget_entity(
            entity_name=body.entity_name,
            group_id=group_id,
            reason=body.reason,
        )
    elif body.fact:
        result = await manager.forget_fact(
            subject_name=body.fact.subject,
            predicate=body.fact.predicate,
            object_name=body.fact.object,
            group_id=group_id,
            reason=body.reason,
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"detail": "Provide either entity_name or fact."},
        )

    status_code = 200 if result.get("status") != "error" else 404
    return JSONResponse(status_code=status_code, content=result)


@router.post("/chat")
async def chat(request: Request, body: ChatBody) -> StreamingResponse:
    """Stream a chat response augmented with memory context (SSE)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    # Gather memory context
    recall_results = await manager.recall(query=body.message, group_id=group_id, limit=5)
    context_result = await manager.get_context(
        group_id=group_id,
        max_tokens=1500,
        topic_hint=body.message,
    )

    # Build system prompt with memory context
    system_prompt = (
        "You are a helpful assistant with access to the user's memory graph. "
        "Use the following context to inform your responses.\n\n"
        f"{context_result['context']}"
    )

    # Build messages
    messages = []
    if body.history:
        for msg in body.history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": body.message})

    # Build sources from recall results
    sources = []
    for r in recall_results:
        if r.get("result_type") == "episode":
            ep = r["episode"]
            sources.append({
                "type": "episode",
                "id": ep["id"],
                "content": ep["content"][:200],
                "score": r["score"],
            })
        else:
            ent = r["entity"]
            sources.append({
                "type": "entity",
                "id": ent["id"],
                "name": ent["name"],
                "summary": ent.get("summary"),
                "score": r["score"],
            })

    async def event_stream():
        try:
            client = anthropic.AsyncAnthropic()
            async with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    chunk = json.dumps({"type": "text", "content": text})
                    yield f"data: {chunk}\n\n"

            # Send sources
            sources_event = json.dumps({"type": "sources", "items": sources})
            yield f"data: {sources_event}\n\n"
        except Exception as e:
            error_event = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_event}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
