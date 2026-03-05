"""Knowledge Management API endpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

import anthropic
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from engram.api.deps import get_conversation_store, get_manager
from engram.security.middleware import get_tenant
from engram.utils.offline_queue import drain_queue

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


class IntendBody(BaseModel):
    trigger_text: str
    action_text: str
    trigger_type: str = "activation"
    entity_names: list[str] | None = None
    threshold: float | None = None
    priority: str = "normal"
    context: str | None = None
    see_also: list[str] | None = None


class BootstrapBody(BaseModel):
    project_path: str
    session_id: str | None = None


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
    conversation_id: str | None = None
    session_date: str | None = None


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


@router.post("/replay-queue")
async def replay_queue(request: Request) -> JSONResponse:
    """Replay entries from the offline capture queue (~/.engram/capture-queue.jsonl).

    Drains the queue file atomically and ingests each entry via store_episode().
    Returns the count of replayed entries.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    entries = drain_queue()
    replayed = 0
    skipped = 0

    for entry in entries:
        content = entry.get("content", "")
        if not content or len(content.strip()) < 10:
            skipped += 1
            continue
        if _dedup_check(content):
            skipped += 1
            continue
        try:
            await manager.store_episode(
                content=content,
                group_id=entry.get("group_id", group_id),
                source=entry.get("source", "offline:replay"),
                session_id=entry.get("session_id"),
            )
            replayed += 1
        except Exception:
            logger.warning("Failed to replay queue entry", exc_info=True)
            skipped += 1

    return JSONResponse(content={
        "status": "replayed",
        "replayed": replayed,
        "skipped": skipped,
        "total": len(entries),
    })


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


@router.post("/intentions")
async def create_intention(request: Request, body: IntendBody) -> JSONResponse:
    """Create a graph-embedded intention (prospective memory trigger)."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    try:
        intention_id = await manager.create_intention(
            trigger_text=body.trigger_text,
            action_text=body.action_text,
            trigger_type=body.trigger_type,
            entity_names=body.entity_names,
            threshold=body.threshold,
            priority=body.priority,
            group_id=group_id,
            context=body.context,
            see_also=body.see_also,
        )
        return JSONResponse(content={
            "status": "created",
            "intentionId": intention_id,
            "triggerText": body.trigger_text,
            "actionText": body.action_text,
        })
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

    intention_entities = await manager.list_intentions(
        group_id=group_id, enabled_only=enabled_only,
    )

    cfg = manager._cfg
    items = []

    if cfg.prospective_graph_embedded:
        from engram.activation.engine import compute_activation
        from engram.models.prospective import IntentionMeta

        now = time.time()
        for entity in intention_entities:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            state = await manager._activation.get_activation(entity.id)
            activation = 0.0
            if state:
                activation = compute_activation(state.access_history, now, cfg)
            warmth_ratio = (
                activation / meta.activation_threshold
                if meta.activation_threshold > 0 else 0.0
            )

            item = {
                "id": entity.id,
                "triggerText": meta.trigger_text,
                "actionText": meta.action_text,
                "triggerType": meta.trigger_type,
                "threshold": meta.activation_threshold,
                "fireCount": meta.fire_count,
                "maxFires": meta.max_fires,
                "enabled": meta.enabled,
                "priority": meta.priority,
                "expiresAt": meta.expires_at,
                "warmthRatio": round(warmth_ratio, 4),
                "linkedEntityIds": meta.trigger_entity_ids,
            }
            if meta.context is not None:
                item["context"] = meta.context
            if meta.see_also is not None:
                item["seeAlso"] = meta.see_also
            items.append(item)
    else:
        for i in intention_entities:
            items.append({
                "id": i.id,
                "triggerText": i.trigger_text,
                "actionText": i.action_text,
                "triggerType": i.trigger_type,
                "threshold": i.threshold,
                "fireCount": i.fire_count,
                "maxFires": i.max_fires,
                "enabled": i.enabled,
            })

    return JSONResponse(content={"intentions": items, "total": len(items)})


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
        await manager.dismiss_intention(intention_id, group_id, hard=hard)
        return JSONResponse(content={
            "status": "dismissed",
            "intentionId": intention_id,
            "hard": hard,
        })
    except Exception as e:
        return JSONResponse(status_code=404, content={"detail": str(e)})


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


MAX_TOOL_TURNS = 3  # Cap agentic loop iterations
MAX_HISTORY_MESSAGES = 10  # Sliding window for conversation history (cost control)

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
            "(e.g., PARENT_OF, WORKS_AT, LIVES_IN)."
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
            },
        },
    },
]


async def _execute_tool(manager, group_id: str, tool_name: str, tool_input: dict) -> str:
    """Execute a chat tool call against the manager. Returns JSON string."""
    logger.info("Chat tool call: %s(%s)", tool_name, json.dumps(tool_input))
    if tool_name == "recall":
        results = await manager.recall(
            query=tool_input["query"],
            group_id=group_id,
            limit=min(tool_input.get("limit", 5), 20),
        )
        # Summarize for the LLM
        items = []
        for r in results:
            if r.get("result_type") == "episode":
                ep = r["episode"]
                items.append({
                    "type": "episode",
                    "content": ep["content"][:300],
                    "source": ep.get("source"),
                    "score": round(r["score"], 3),
                })
            else:
                ent = r["entity"]
                sb = r.get("score_breakdown", {})
                items.append({
                    "type": "entity",
                    "name": ent["name"],
                    "entityType": ent.get("type"),
                    "summary": ent.get("summary"),
                    "id": ent.get("id", ""),
                    "score": round(r["score"], 3),
                    "activation": round(sb.get("activation", 0), 3),
                    "relationships": [
                        {
                            "predicate": rel.get("predicate"),
                            "target": rel.get("target_name", rel.get("target_id", "")),
                            "source": rel.get("source_name", rel.get("source_id", "")),
                        }
                        for rel in r.get("relationships", [])[:10]
                    ],
                })
        return json.dumps({"results": items, "total": len(items)})

    elif tool_name == "search_entities":
        results = await manager.search_entities(
            group_id=group_id,
            name=tool_input.get("name"),
            entity_type=tool_input.get("entity_type"),
            limit=min(tool_input.get("limit", 10), 20),
        )
        items = []
        for ent in results:
            items.append({
                "name": ent.get("name", ""),
                "entityType": ent.get("type", ""),
                "summary": ent.get("summary"),
                "id": ent.get("id", ""),
            })
        return json.dumps({"entities": items, "total": len(items)})

    elif tool_name == "search_facts":
        # Over-fetch then deduplicate — duplicate (subject, predicate, object)
        # triples are common and can crowd out unique results at low limits.
        requested_limit = min(tool_input.get("limit", 10), 20)
        results = await manager.search_facts(
            group_id=group_id,
            query=tool_input.get("query", ""),
            subject=tool_input.get("subject"),
            predicate=tool_input.get("predicate"),
            limit=requested_limit * 2,  # over-fetch to survive duplicates
        )
        seen: set[tuple[str, str, str]] = set()
        items = []
        for f in results:
            key = (f["subject"], f["predicate"], f["object"])
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "subject": f["subject"],
                "predicate": f["predicate"],
                "object": f["object"],
                "confidence": f.get("confidence"),
            })
            if len(items) >= requested_limit:
                break
        logger.info(
            "Chat search_facts returned %d unique facts (from %d raw)",
            len(items), len(results),
        )
        return json.dumps({"facts": items, "total": len(items)})

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


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
    """Stream a chat response using AI SDK v6 UIMessageStream protocol.

    Uses an agentic tool-use loop so the LLM can query the memory graph
    autonomously (recall, search_entities, search_facts) before answering.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id

    # Rate limit chat endpoint (guards against runaway API costs)
    from engram.main import _app_state

    rate_limiter = _app_state.get("rate_limiter")
    if rate_limiter:
        allowed, remaining = await rate_limiter.check(group_id, "chat")
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded for chat", "remaining": remaining},
            )

    manager = get_manager()

    # Baseline context for the system prompt (1000 token budget to control costs)
    context_result = await manager.get_context(
        group_id=group_id,
        max_tokens=1000,
        topic_hint=body.message,
    )

    # Build system prompt with memory context + tool-use guidance
    static_preamble = (
        "You are a helpful assistant with access to the user's memory graph. "
        "You have tools to search the graph — use them to answer questions accurately.\n\n"
        "Guidelines:\n"
        "- Use search_facts to find specific relationships (e.g., family members, work history)\n"
        "- Use search_entities to find entities by name\n"
        "- Use recall for general semantic search\n"
        "- Always search before answering. Do not guess from context alone when tools "
        "can provide precise answers.\n\n"
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

    # Build messages (sliding window to control token costs)
    messages: list[dict] = []
    if body.history:
        history = body.history
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": body.message})

    # Resolve or create conversation
    conversation_id = body.conversation_id
    try:
        conv_store = get_conversation_store()
    except RuntimeError:
        conv_store = None

    if conv_store and not conversation_id:
        title = body.message[:60].strip()
        conversation_id = await conv_store.create_conversation(
            group_id=group_id,
            session_date=body.session_date,
            title=title,
        )

    text_part_id = "text_0"

    async def event_stream():
        try:
            yield _sse({"type": "start"})
            yield _sse({"type": "start-step"})

            client = anthropic.AsyncAnthropic()

            # Accumulate tool results for UI components
            all_recall_results: list[dict] = []
            all_facts: list[dict] = []

            # --- Agentic tool-use loop (non-streaming) ---
            loop_messages = list(messages)
            for _turn in range(MAX_TOOL_TURNS):
                response = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=loop_messages,
                    tools=CHAT_TOOLS,
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
                            manager, group_id, tool_name, tool_input,
                        )
                    except Exception as e:
                        logger.warning("Chat tool %s failed: %s", tool_name, e)
                        result_str = json.dumps({"error": str(e)})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

                    # Accumulate for UI components
                    try:
                        parsed = json.loads(result_str)
                        if tool_name == "recall":
                            for item in parsed.get("results", []):
                                all_recall_results.append(_to_raw_recall(item))
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
                    system=system_prompt,
                    messages=loop_messages,
                )

            # Emit synthetic tool events for UI components
            tool_events = _build_tool_events(all_recall_results, all_facts)
            if tool_events:
                yield tool_events

            # Stream the final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            if final_text:
                yield _sse({"type": "text-start", "id": text_part_id})
                # Emit in chunks for streaming feel
                chunk_size = 20
                for i in range(0, len(final_text), chunk_size):
                    yield _sse({
                        "type": "text-delta",
                        "id": text_part_id,
                        "delta": final_text[i : i + chunk_size],
                    })
                yield _sse({"type": "text-end", "id": text_part_id})

            yield _sse({"type": "finish-step"})
            yield _sse({
                "type": "finish",
                "finishReason": "stop",
                **({"conversationId": conversation_id} if conversation_id else {}),
            })

            # Fire-and-forget: persist messages + tag entities
            if conv_store and conversation_id and final_text:
                async def _persist():
                    try:
                        await conv_store.add_messages_bulk(conversation_id, [
                            {"role": "user", "content": body.message},
                            {"role": "assistant", "content": final_text},
                        ])
                        # Tag entities discovered during tool use
                        entity_ids: set[str] = set()
                        for r in all_recall_results:
                            if r.get("result_type") != "episode":
                                eid = r.get("entity", {}).get("id")
                                if eid:
                                    entity_ids.add(eid)
                        for eid in entity_ids:
                            await conv_store.tag_entity(conversation_id, eid)
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


def _to_raw_recall(item: dict) -> dict:
    """Convert summarized tool result back to raw recall format for _build_tool_events."""
    if item.get("type") == "episode":
        return {
            "result_type": "episode",
            "episode": {
                "id": "",
                "content": item.get("content", ""),
                "source": item.get("source"),
                "created_at": None,
            },
            "score": item.get("score", 0),
            "score_breakdown": {
                "semantic": 0,
                "activation": 0,
                "edge_proximity": 0,
                "exploration_bonus": 0,
            },
        }
    else:
        return {
            "result_type": "entity",
            "entity": {
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "type": item.get("entityType", "Other"),
                "summary": item.get("summary"),
            },
            "score": item.get("score", 0),
            "score_breakdown": {
                "semantic": 0,
                "activation": item.get("activation", 0),
                "edge_proximity": 0,
                "exploration_bonus": 0,
            },
            "relationships": [
                {
                    "predicate": rel.get("predicate"),
                    "target_id": rel.get("target", ""),
                    "source_id": rel.get("source", ""),
                }
                for rel in item.get("relationships", [])
            ],
        }
