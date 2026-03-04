"""Knowledge Management API endpoints."""

from __future__ import annotations

import hashlib
import json
import logging
import time

import anthropic
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engram.api.deps import get_manager
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


# ─── Dedup cache for auto-observe ────────────────────────────────

_DEDUP_CACHE: dict[str, float] = {}  # content_hash → timestamp
_DEDUP_TTL = 300  # 5 minutes


def _dedup_check(content: str) -> bool:
    """Return True if content was seen in the last 5 minutes (skip it)."""
    now = time.time()
    # Evict stale entries periodically
    if len(_DEDUP_CACHE) > 1000:
        stale = [k for k, ts in _DEDUP_CACHE.items() if now - ts > _DEDUP_TTL]
        for k in stale:
            del _DEDUP_CACHE[k]

    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    if content_hash in _DEDUP_CACHE and now - _DEDUP_CACHE[content_hash] < _DEDUP_TTL:
        return True

    _DEDUP_CACHE[content_hash] = now
    return False


# ─── Request bodies ──────────────────────────────────────────────


class ObserveBody(BaseModel):
    content: str
    source: str = "dashboard"


class AutoObserveBody(BaseModel):
    content: str
    source: str = "auto:prompt"
    project: str = "unknown"
    role: str = "user"
    session_id: str | None = None


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


@router.post("/auto-observe")
async def auto_observe(request: Request, body: AutoObserveBody) -> JSONResponse:
    """Auto-capture endpoint with dedup. Used by Claude Code hooks."""
    tenant = get_tenant(request)
    group_id = tenant.group_id

    if not body.content or len(body.content.strip()) < 10:
        return JSONResponse(content={"status": "skipped", "reason": "too_short"})

    if _dedup_check(body.content):
        return JSONResponse(content={"status": "dedup_skipped"})

    manager = get_manager()

    episode_id = await manager.store_episode(
        content=body.content,
        group_id=group_id,
        source=body.source,
        session_id=body.session_id,
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

    return JSONResponse(content={
        "context": result["context"],
        "entityCount": result["entity_count"],
        "factCount": result["fact_count"],
        "tokenEstimate": result["token_estimate"],
        "format": result.get("format", "structured"),
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


def _sse(data: dict) -> str:
    """Format a single SSE event line."""
    return f"data: {json.dumps(data)}\n\n"


def _emit_tool(tool_call_id: str, tool_name: str, input_data: dict) -> str:
    """Emit AI SDK v6 synthetic tool call (input-available + output-available)."""
    inp = _sse({
        "type": "tool-input-available",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "input": input_data,
        "dynamic": True,
    })
    out = _sse({
        "type": "tool-output-available",
        "toolCallId": tool_call_id,
        "output": "displayed",
        "dynamic": True,
    })
    return inp + out


def _build_tool_events(recall_results: list, facts: list) -> str:
    """Analyze recall results and emit synthetic tool events for rich components."""
    lines = ""
    tc_idx = 0

    # --- show_entities: when recall has entity results ---
    entities = []
    for r in recall_results:
        if r.get("result_type") != "episode":
            ent = r["entity"]
            entities.append({
                "id": ent["id"],
                "name": ent["name"],
                "entityType": ent.get("type", "Other"),
                "summary": ent.get("summary"),
                "score": round(r["score"], 3),
                "activation": round(r["score_breakdown"].get("activation", 0), 3),
            })

    if entities:
        tc_idx += 1
        lines += _emit_tool(f"tc_{tc_idx}", "show_entities", {"entities": entities})

    # --- show_relationship_graph: entity with 3+ relationships ---
    for r in recall_results:
        if r.get("result_type") == "episode":
            continue
        rels = r.get("relationships", [])
        if len(rels) >= 3:
            ent = r["entity"]
            nodes = [{"id": ent["id"], "name": ent["name"], "type": ent.get("type", "Other")}]
            edges = []
            seen_ids = {ent["id"]}
            for rel in rels[:12]:  # cap at 12 edges
                target_id = rel.get("target_id", "")
                source_id = rel.get("source_id", "")
                other_id = target_id if source_id == ent["id"] else source_id
                if other_id and other_id not in seen_ids:
                    other_name = other_id
                    for e in entities:
                        if e["id"] == other_id:
                            other_name = e["name"]
                            break
                    nodes.append({"id": other_id, "name": other_name, "type": "Other"})
                    seen_ids.add(other_id)
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "predicate": rel.get("predicate", "RELATED"),
                    "weight": rel.get("weight", 1.0),
                })
            tc_idx += 1
            lines += _emit_tool(f"tc_{tc_idx}", "show_relationship_graph", {
                "centralEntity": ent["name"],
                "nodes": nodes,
                "edges": edges,
            })
            break  # only one graph per response

    # --- show_facts: when facts search returns results ---
    if facts:
        fact_items = []
        for f in facts[:10]:  # cap at 10
            fact_items.append({
                "subject": f["subject"],
                "predicate": f["predicate"],
                "object": f["object"],
                "confidence": f.get("confidence"),
            })
        tc_idx += 1
        lines += _emit_tool(f"tc_{tc_idx}", "show_facts", {"facts": fact_items})

    # --- show_activation_chart: when 3+ entities ---
    if len(entities) >= 3:
        chart_entities = [
            {"name": e["name"], "entityType": e["entityType"], "activation": e["activation"]}
            for e in sorted(entities, key=lambda x: x["activation"], reverse=True)[:8]
        ]
        tc_idx += 1
        lines += _emit_tool(f"tc_{tc_idx}", "show_activation_chart", {"entities": chart_entities})

    # --- show_timeline: when episodes in recall ---
    episodes = []
    for r in recall_results:
        if r.get("result_type") == "episode":
            ep = r["episode"]
            episodes.append({
                "id": ep["id"],
                "content": ep["content"][:200],
                "source": ep.get("source"),
                "createdAt": ep.get("created_at"),
                "score": round(r["score"], 3),
            })

    if episodes:
        tc_idx += 1
        lines += _emit_tool(f"tc_{tc_idx}", "show_timeline", {"episodes": episodes})

    return lines


@router.post("/chat")
async def chat(request: Request, body: ChatBody) -> StreamingResponse:
    """Stream a chat response using AI SDK v6 UIMessageStream protocol."""
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

    # Search facts related to the query
    facts = []
    try:
        facts = await manager.search_facts(
            group_id=group_id,
            query=body.message,
            limit=10,
        )
    except Exception:
        pass  # facts are optional enrichment

    # Build system prompt with memory context (cached static prefix + dynamic context)
    static_preamble = (
        "You are a helpful assistant with access to the user's memory graph. "
        "Use the following context to inform your responses.\n\n"
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

    # Build messages
    messages = []
    if body.history:
        for msg in body.history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": body.message})

    # Pre-build synthetic tool events
    tool_events = _build_tool_events(recall_results, facts)

    # Generate a text part ID for the streaming text
    text_part_id = "text_0"

    async def event_stream():
        try:
            # Start message
            yield _sse({"type": "start"})
            yield _sse({"type": "start-step"})

            # Emit synthetic tool calls first (rich components)
            if tool_events:
                yield tool_events

            # Stream LLM text response
            yield _sse({"type": "text-start", "id": text_part_id})

            client = anthropic.AsyncAnthropic()
            async with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield _sse({
                        "type": "text-delta",
                        "id": text_part_id,
                        "delta": text,
                    })

            yield _sse({"type": "text-end", "id": text_part_id})

            # Finish step and message
            yield _sse({"type": "finish-step"})
            yield _sse({"type": "finish", "finishReason": "stop"})
        except Exception as e:
            yield _sse({"type": "error", "errorText": str(e)})
            yield _sse({"type": "finish", "finishReason": "error"})

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
