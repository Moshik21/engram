"""MCP server for Engram — native stdio transport."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Literal, cast

from mcp.server.fastmcp import FastMCP

from engram.config import ActivationConfig, EngramConfig
from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.evaluation.label_service import (
    persist_recall_eval_sample,
    persist_session_continuity_sample,
)
from engram.evaluation.presenter import (
    present_recall_sample_write,
    present_session_sample_write,
)
from engram.evaluation.report_service import build_brain_loop_evaluation_surface
from engram.evaluation.store import (
    SQLiteEvaluationStore,
)
from engram.events.bus import get_event_bus
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.ingestion.adjudication_surface import load_episode_adjudication_requests
from engram.ingestion.presenter import memory_write_contract, present_mcp_memory_write
from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT
from engram.models.episode import Attachment
from engram.notifications.surface import get_notification_surface_service_from_state
from engram.retrieval.artifacts import build_mcp_artifact_search_surface
from engram.retrieval.auto_recall import (
    RECALL_TOOLS,
    WRITE_TOOLS,
    RecallCooldown,
    apply_mcp_recall_enrichment,
    compact_auto_recall_surface,
    compact_lite_auto_recall_surface,
    extract_recall_query,
    plan_mcp_recall_middleware,
    plan_session_prime,
    should_recall_for_tool,
)
from engram.retrieval.context import (
    ingest_manager_conversation_turn,
    manager_conversation_context,
    manager_conversation_embed_fn,
    manager_conversation_recent_turns,
    manager_conversation_top_entity_names,
)
from engram.retrieval.control import (
    record_manager_memory_need_analysis,
    resolve_manager_recall_need_thresholds,
)
from engram.retrieval.epistemic_route import build_question_route_surface
from engram.retrieval.feedback import publish_memory_need_analysis
from engram.retrieval.forgetting import build_mcp_forget_surface
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.packets import assemble_memory_packets
from engram.retrieval.preference_feedback import (
    FeedbackRatingError,
    build_explicit_feedback_surface,
)
from engram.retrieval.prospective import (
    build_intention_list_surface,
    build_mcp_create_intention_surface,
    build_mcp_dismiss_intention_surface,
)
from engram.retrieval.recall_surface import build_mcp_recall_surface
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode, resolve_mode
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Initialize storage on the same event loop as the MCP server."""
    await _init()
    try:
        yield
    finally:
        await _shutdown()


mcp = FastMCP("engram", instructions=ENGRAM_SYSTEM_PROMPT, lifespan=_lifespan)


# ─── Session State ──────────────────────────────────────────────────


@dataclass
class SessionState:
    """Tracks session-level metadata for the MCP server lifetime."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: str = field(default_factory=lambda: os.environ.get("ENGRAM_GROUP_ID", "default"))
    started_at: datetime = field(default_factory=utc_now)
    episode_count: int = 0
    last_activity: datetime = field(default_factory=utc_now)
    auto_recall_primed: bool = False
    last_recall_time: float = 0.0
    recall_cache: dict = field(default_factory=dict)  # entity_id -> (timestamp, compact_result)


# Module-level state initialized on startup
_manager: GraphManager | None = None
_group_id: str = "default"
_session: SessionState | None = None
_recall_cooldown: RecallCooldown | None = None
_activation_cfg: ActivationConfig | None = None
_evaluation_store: SQLiteEvaluationStore | None = None
_consolidation_store: Any | None = None
_episode_worker: Any | None = None
_redis_publisher: Any | None = None


async def _init() -> None:
    """Initialize storage and services."""
    global _manager, _group_id, _session, _recall_cooldown
    global _activation_cfg, _evaluation_store, _consolidation_store
    global _episode_worker, _redis_publisher

    config = EngramConfig()
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    search_initializer = cast(Any, search_index).initialize
    if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
        await search_initializer(db=graph_store._db)
    else:
        await search_initializer()

    extractor = create_extractor(config)
    event_bus = get_event_bus()
    _manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        extractor,
        cfg=config.activation,
        event_bus=event_bus,
        runtime_mode=mode.value,
    )
    _group_id = os.environ.get("ENGRAM_GROUP_ID", config.default_group_id)
    _session = SessionState(group_id=_group_id)
    _activation_cfg = config.activation
    _recall_cooldown = RecallCooldown(
        max_per_minute=config.activation.auto_recall_max_per_minute,
        cooldown_seconds=config.activation.auto_recall_cooldown_seconds,
    )
    _evaluation_store = SQLiteEvaluationStore(str(config.get_sqlite_path()))
    if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
        await _evaluation_store.initialize(db=graph_store._db)
    else:
        await _evaluation_store.initialize()
    _consolidation_store = await _create_consolidation_store(mode, config, graph_store)

    # Background episode worker
    from engram.worker import EpisodeWorker

    if config.activation.worker_enabled:
        _episode_worker = EpisodeWorker(_manager, config.activation)
        _episode_worker.start(_group_id, event_bus)

    # In full mode, bridge events to Redis so the REST API/dashboard can see them
    if mode == EngineMode.FULL:
        from engram.events.redis_bridge import create_publisher

        publisher = await create_publisher(_group_id, redis_url=config.redis.url)
        if publisher:
            _redis_publisher = publisher
            event_bus.add_on_publish_hook(publisher)

    logger.info(
        "Engram MCP server initialized (%s mode, session=%s)", mode.value, _session.session_id
    )


async def _shutdown() -> None:
    """Close MCP-owned runtime resources."""
    global _manager, _session, _recall_cooldown, _activation_cfg
    global _evaluation_store, _consolidation_store, _episode_worker, _redis_publisher

    if _episode_worker is not None:
        await _episode_worker.stop()
        _episode_worker = None

    if _redis_publisher is not None:
        get_event_bus().remove_on_publish_hook(_redis_publisher)
        await _redis_publisher.close()
        _redis_publisher = None

    await _maybe_close(_evaluation_store)
    _evaluation_store = None
    await _maybe_close(_consolidation_store)
    _consolidation_store = None

    if _manager is not None:
        await _maybe_close(getattr(_manager, "_search", None))
        await _maybe_close(getattr(_manager, "_activation", None))
        await _maybe_close(getattr(_manager, "_graph", None))
        _manager = None

    _session = None
    _recall_cooldown = None
    _activation_cfg = None


async def _maybe_close(resource: Any) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _create_consolidation_store(
    mode: EngineMode,
    config: EngramConfig,
    graph_store: Any,
) -> Any:
    """Create the consolidation audit store for MCP-side evaluation reports."""
    if mode == EngineMode.HELIX:
        from engram.storage.helix.consolidation import HelixConsolidationStore

        store = HelixConsolidationStore(
            config.helix,
            client=getattr(graph_store, "_helix_client", None),
        )
        await store.initialize()
        return store

    if config.postgres.dsn:
        from engram.storage.postgres.consolidation import PostgresConsolidationStore

        store = PostgresConsolidationStore(
            config.postgres.dsn,
            min_pool_size=config.postgres.min_pool_size,
            max_pool_size=config.postgres.max_pool_size,
        )
        await store.initialize()
        return store

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(str(config.get_sqlite_path()))
    if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
        await store.initialize(db=graph_store._db)
    else:
        await store.initialize()
    return store


def _get_manager() -> GraphManager:
    if _manager is None:
        raise RuntimeError("MCP server not initialized")
    return _manager


def _get_session() -> SessionState:
    if _session is None:
        raise RuntimeError("MCP server not initialized")
    return _session


def _get_evaluation_store() -> SQLiteEvaluationStore:
    if _evaluation_store is None:
        raise RuntimeError("EvaluationStore not initialized")
    return _evaluation_store


def _get_conv_context(manager: GraphManager):
    """Return the concrete conversation context if one is configured."""
    return manager_conversation_context(manager)


def _get_conv_embed_fn(manager: GraphManager):
    """Return the embedding function used for live conversation turns, if available."""
    return manager_conversation_embed_fn(manager)


def _get_conv_top_entity_names(manager: GraphManager) -> list[str]:
    return manager_conversation_top_entity_names(manager)


def _get_conv_recent_turns(manager: GraphManager, limit: int) -> list[str]:
    return manager_conversation_recent_turns(manager, limit)


def _get_graph_probe(manager: GraphManager):
    return manager.get_recall_need_graph_probe()


async def _ingest_live_turn(
    manager: GraphManager,
    text: str,
    *,
    source: str,
) -> None:
    """Record a live turn in conversation context with embedding when possible."""
    conv_context = _get_conv_context(manager)
    if conv_context is None:
        return
    await ingest_manager_conversation_turn(manager, text, source=source)


# ─── AutoRecall Helpers ──────────────────────────────────────────────


def _extract_recall_query(content: str) -> str:
    """Extract a recall query from content: proper nouns first, then first sentence."""
    return extract_recall_query(content)


async def _auto_recall_lite(
    content: str,
    manager: GraphManager,
    cfg: ActivationConfig,
) -> dict | None:
    """Dispatch auto-recall to lite or medium based on config.

    - lite: FTS5-only entity probe (~3-5ms)
    - medium: FTS5 + embedding rerank (~8-15ms, better disambiguation)
    """
    if len(content) < 20:
        return None

    session = _get_session()
    level = getattr(cfg, "auto_recall_level", "lite")

    try:
        if level == "medium" and hasattr(manager, "recall_medium"):
            results = await manager.recall_medium(
                text=content,
                group_id=_group_id,
                session_cache=session.recall_cache,
                token_budget=cfg.auto_recall_token_budget,
                cache_ttl=cfg.auto_recall_cache_ttl_seconds,
            )
        else:
            results = await manager.recall_lite(
                text=content,
                group_id=_group_id,
                session_cache=session.recall_cache,
                token_budget=cfg.auto_recall_token_budget,
                cache_ttl=cfg.auto_recall_cache_ttl_seconds,
            )
    except Exception:
        logger.debug("auto_recall failed", exc_info=True)
        return None

    return compact_lite_auto_recall_surface(results, level=level)


async def _auto_recall_full(
    content: str,
    manager: GraphManager,
    cfg: ActivationConfig,
) -> dict | None:
    """Full recall pipeline piggybacked on observe/remember calls.

    Kept for backwards compatibility and explicit recall paths.
    Uses need analysis, cooldown, topic dedup, packet assembly.
    """
    if not cfg.auto_recall_enabled:
        return None

    need = None
    query = ""
    if cfg.recall_need_analyzer_enabled:
        recent_turns: list[str] = []
        conv_context = _get_conv_context(manager)
        if conv_context is not None:
            recent_turns = _get_conv_recent_turns(manager, cfg.conv_multi_query_turns)
        session_entity_names = _get_conv_top_entity_names(manager)
        need = await analyze_memory_need(
            content,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            mode="auto_recall",
            graph_probe=_get_graph_probe(manager) if cfg.recall_need_graph_probe_enabled else None,
            group_id=_group_id,
            conv_context=conv_context,
            cfg=cfg,
            thresholds=await resolve_manager_recall_need_thresholds(manager, _group_id),
        )
        await record_manager_memory_need_analysis(manager, _group_id, need)
        if cfg.recall_telemetry_enabled:
            publish_memory_need_analysis(
                get_event_bus(),
                _group_id,
                need,
                source="auto_recall",
                mode="auto_recall",
                turn_text=content,
            )
        if not need.should_recall:
            return None
        query = need.query_hint or _extract_recall_query(content)
    else:
        query = _extract_recall_query(content)

    if not query:
        return None

    session = _get_session()
    cooldown = _recall_cooldown

    # Cooldown check
    now = time.time()
    if cooldown and cooldown.is_throttled(query, now):
        return None

    # Skip if explicit recall was recent (within 30s)
    if session.last_recall_time and (now - session.last_recall_time) < 30.0:
        return None

    # Topic-shift-aware recall limit (Wave 3)
    recall_limit = cfg.auto_recall_limit
    conv_context = _get_conv_context(manager)
    if (
        cfg.conv_topic_shift_enabled
        and conv_context is not None
        and conv_context.detect_topic_shift()
    ):
        recall_limit = cfg.conv_topic_shift_recall_boost
        conv_context.acknowledge_shift()

    try:
        interaction_type = None
        record_access = True
        if cfg.recall_telemetry_enabled or cfg.recall_usage_feedback_enabled:
            interaction_type = "surfaced"
        if cfg.recall_usage_feedback_enabled:
            record_access = False
        results = await manager.recall(
            query=query,
            group_id=_group_id,
            limit=recall_limit,
            record_access=record_access,
            interaction_type=interaction_type,
            interaction_source="auto_recall",
            memory_need=need,
        )
    except Exception:
        logger.debug("auto_recall failed", exc_info=True)
        return None

    packets = []
    if cfg.recall_packets_enabled:
        packets = [
            packet.to_dict()
            for packet in await assemble_memory_packets(
                results,
                query,
                mode="auto_surface",
                memory_need=need,
                max_packets=cfg.recall_packet_auto_limit,
                resolve_entity_name=lambda entity_id: manager.resolve_entity_name(
                    entity_id,
                    _group_id,
                ),
            )
        ]

    response = compact_auto_recall_surface(
        results,
        query=query,
        packets=packets,
        min_score=cfg.auto_recall_min_score,
    )
    if response is None:
        return None

    if cooldown:
        cooldown.record(query, now)

    return response


async def _session_prime(
    content: str | None,
    manager: GraphManager,
    cfg: ActivationConfig,
) -> dict | None:
    """Auto-prime context on first tool call in a session."""
    if not cfg.auto_recall_session_prime:
        return None
    session = _get_session()
    plan = plan_session_prime(
        content,
        cfg,
        already_primed=session.auto_recall_primed,
    )
    if plan is None:
        return None
    session.auto_recall_primed = True

    try:
        result = await manager.get_context(
            group_id=_group_id,
            max_tokens=plan.max_tokens,
            topic_hint=plan.topic_hint,
            format="structured",
        )
        return result
    except Exception:
        logger.debug("session_prime failed", exc_info=True)
        return None


async def _load_consolidation_evaluation_inputs(
    manager: GraphManager,
    group_id: str,
    cycle_limit: int,
) -> tuple[list[Any], list[Any]]:
    """Read recent consolidation context from the active MCP runtime store."""
    return await ConsolidationAuditReader(_consolidation_store).evaluation_context(
        group_id,
        cycle_limit=max(1, cycle_limit),
    )


# ─── Recall Middleware ───────────────────────────────────────────────

_RECALL_TOOLS = RECALL_TOOLS
_WRITE_TOOLS = WRITE_TOOLS


def _should_recall(tool_name: str, cfg: ActivationConfig | None) -> bool:
    """Unified gate: should this tool get recall context?"""
    return should_recall_for_tool(tool_name, cfg)


async def _serialize_intentions(manager: GraphManager) -> list[dict] | None:
    """Serialize and drain triggered intentions from manager."""
    drain_triggered_intention_views = getattr(
        manager,
        "drain_triggered_intention_views",
        None,
    )
    if not callable(drain_triggered_intention_views):
        return None
    result = drain_triggered_intention_views()
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, list) and result else None


def _serialize_notifications(cfg: ActivationConfig, group_id: str) -> list[dict] | None:
    """Serialize proactive memory notifications."""
    service = get_notification_surface_service_from_state()
    if service is None:
        return None
    return service.mcp_notifications(cfg=cfg, group_id=group_id)


async def _recall_middleware(
    content: str,
    response: dict,
    *,
    tool_name: str,
    auto_observe: bool = False,
) -> None:
    """Unified recall middleware — replaces per-tool piggyback blocks.

    Attaches recalled_context, session_context, triggered_intentions,
    and memory_notifications to any tool response.
    """
    cfg = _activation_cfg
    plan = plan_mcp_recall_middleware(
        content,
        tool_name=tool_name,
        cfg=cfg,
        auto_observe=auto_observe,
    )
    if not plan.should_recall:
        # Even when recall is disabled, surface notifications for get_context
        if plan.surface_notifications_when_recall_disabled and cfg:
            apply_mcp_recall_enrichment(
                response,
                memory_notifications=_serialize_notifications(cfg, _group_id),
            )
        return
    assert cfg is not None
    manager = _get_manager()

    # Auto-observe long content (e.g. route_question questions)
    if plan.auto_observe_content:
        try:
            await manager.store_episode(content, _group_id, source="tool_piggyback")
        except Exception:
            logger.debug("middleware auto-observe failed", exc_info=True)

    # Update conversation fingerprint (read tools only; write tools do it inline)
    if plan.ingest_live_turn:
        await _ingest_live_turn(manager, content, source="tool_piggyback")

    # Session prime (first call only)
    prime = await _session_prime(content, manager, cfg)

    # Recall
    recalled = await _auto_recall_lite(content, manager, cfg)

    # Triggered intentions
    intentions = await _serialize_intentions(manager)

    # Memory notifications
    notifs = _serialize_notifications(cfg, _group_id)
    apply_mcp_recall_enrichment(
        response,
        session_context=prime,
        recalled_context=recalled,
        triggered_intentions=intentions,
        memory_notifications=notifs,
    )


# ─── Tools ──────────────────────────────────────────────────────────


@mcp.tool()
async def remember(
    content: str,
    source: str = "mcp",
    conversation_date: str | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    image_data: str | None = None,
    image_mime: str = "image/png",
) -> str:
    """Store a memory. Extracts entities and relationships from the text.

    Args:
        content: The text to remember (conversation excerpt, fact, note, etc.)
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")
        conversation_date: Optional ISO 8601 date string for when the conversation happened
        proposed_entities: Optional client-proposed entities
            [{"name": ..., "entity_type": ...}]
        proposed_relationships: Optional client-proposed relationships
            [{"subject": ..., "predicate": ..., "object": ...}]
        model_tier: The calling model tier (opus/sonnet/haiku/default)
            for confidence scoring
        image_data: Optional base64 encoded image to attach to the memory
        image_mime: MIME type of the attached image (default "image/png")

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    cfg = _activation_cfg
    conv_dt = None
    if conversation_date:
        try:
            conv_dt = datetime.fromisoformat(conversation_date)
        except (ValueError, TypeError):
            pass
    attachments: list[Attachment] = []
    if image_data:
        attachments.append(
            Attachment(
                mime_type=image_mime,
                data_url=image_data,
                description=content[:200] if content else "",
            )
        )
    episode_id = await manager.ingest_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=conv_dt,
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
        model_tier=model_tier,
        attachments=attachments or None,
    )
    session.episode_count += 1
    session.last_activity = utc_now()
    # Update conversation context (Wave 2)
    await _ingest_live_turn(manager, content, source="remember")
    message = "Memory received. Entities and relationships extracted."
    adjudications: list[dict] = []
    # Add remember_outcome for v2 pipeline
    if cfg and cfg.evidence_extraction_enabled:
        message = "Memory received. Evidence extracted and evaluated."
        if cfg.edge_adjudication_client_enabled:
            adjudications = await load_episode_adjudication_requests(
                manager,
                episode_id=episode_id,
                group_id=_group_id,
            )
    response = present_mcp_memory_write(
        memory_write_contract(
            "remember",
            episode_id,
            adjudication_requests=adjudications,
        ),
        message=message,
    )
    await _recall_middleware(content, response, tool_name="remember")
    return json.dumps(response)


@mcp.tool()
async def feedback(
    entity_id: str,
    rating: int,
    comment: str | None = None,
) -> str:
    """Rate an entity to influence future memory retrieval.

    Args:
        entity_id: The entity to rate
        rating: 1-5 scale (1=strongly avoid, 3=neutral, 5=strongly prefer)
        comment: Optional comment explaining the rating

    Returns:
        JSON with status, entity_id, domain, edge_type, edge_weight.
    """
    manager = _get_manager()
    try:
        result = await build_explicit_feedback_surface(
            manager,
            group_id=_group_id,
            entity_id=entity_id,
            rating=rating,
            comment=comment,
        )
    except FeedbackRatingError as e:
        return json.dumps({"error": str(e)})
    return json.dumps(result)


@mcp.tool()
async def adjudicate_evidence(
    request_id: str,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
    reject_evidence_ids: list[str] | None = None,
    model_tier: str = "default",
    rationale: str | None = None,
) -> str:
    """Resolve a structured edge-adjudication work item.

    Args:
        request_id: The adjudication request returned by remember()
        entities: Optional adjudicated entity proposals
        relationships: Optional adjudicated relationship proposals
        reject_evidence_ids: Optional evidence rows to explicitly reject
        model_tier: The calling model tier (opus/sonnet/haiku/default)
        rationale: Optional short rationale for provenance

    Returns:
        JSON with request status and committed IDs.
    """
    manager = _get_manager()
    outcome = await manager.submit_adjudication_resolution(
        request_id,
        entities=entities,
        relationships=relationships,
        reject_evidence_ids=reject_evidence_ids,
        source="client_adjudication",
        model_tier=model_tier,
        rationale=rationale,
        group_id=_group_id,
    )
    return json.dumps(
        {
            "status": outcome.status,
            "request_id": outcome.request_id,
            "committed_ids": outcome.committed_ids,
            "superseded_evidence_ids": outcome.superseded_evidence_ids,
            "replacement_evidence_ids": outcome.replacement_evidence_ids,
        },
    )


@mcp.tool()
async def observe(content: str, source: str = "mcp", conversation_date: str | None = None) -> str:
    """Store raw text without extraction. Fast path for bulk capture.

    Args:
        content: The text to store (conversation excerpt, fact, note, etc.)
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")
        conversation_date: Optional ISO 8601 date string for when the conversation happened

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    conv_dt = None
    if conversation_date:
        try:
            conv_dt = datetime.fromisoformat(conversation_date)
        except (ValueError, TypeError):
            pass
    episode_id = await manager.store_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=conv_dt,
    )
    session.episode_count += 1
    session.last_activity = utc_now()
    # Update conversation context (Wave 2)
    await _ingest_live_turn(manager, content, source="observe")
    response = present_mcp_memory_write(
        memory_write_contract("observe", episode_id),
        message="Stored for background processing.",
    )
    await _recall_middleware(content, response, tool_name="observe")
    return json.dumps(response)


@mcp.tool()
async def observe_image(
    image_data: str,
    mime_type: str = "image/png",
    description: str = "",
    source: str = "mcp",
) -> str:
    """Store an image in memory. Fast path — no extraction.

    Args:
        image_data: Base64 encoded image data
        mime_type: MIME type of the image (default "image/png")
        description: Optional text description of what the image shows
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    content = description or "Image observation"
    attachment = Attachment(
        mime_type=mime_type,
        data_url=image_data,
        description=description,
    )
    episode_id = await manager.store_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
        attachments=[attachment],
    )
    session.episode_count += 1
    session.last_activity = utc_now()
    return json.dumps(
        present_mcp_memory_write(
            memory_write_contract("observe", episode_id, attachment_kind="image"),
            message="Image stored for background processing.",
        ),
    )


@mcp.tool()
async def observe_file(
    file_data: str,
    mime_type: str,
    description: str = "",
    source: str = "mcp",
) -> str:
    """Store a file (PDF, audio, video) in memory. Fast path — no extraction.

    Args:
        file_data: Base64 encoded file data
        mime_type: MIME type of the file (e.g., "application/pdf", "audio/mp3", "video/mp4")
        description: Optional text description of the file contents
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    content = description or "File observation"
    attachment = Attachment(
        mime_type=mime_type,
        data_url=file_data,
        description=description,
    )
    episode_id = await manager.store_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
        attachments=[attachment],
    )
    session.episode_count += 1
    session.last_activity = utc_now()
    return json.dumps(
        present_mcp_memory_write(
            memory_write_contract("observe", episode_id, attachment_kind="file"),
            message="File stored for background processing.",
        ),
    )


@mcp.tool()
async def recall(query: str, limit: int = 5) -> str:
    """Retrieve memories relevant to a query using activation-aware search.

    Args:
        query: What to search for in memory
        limit: Maximum number of results to return (default 5)

    Returns:
        JSON with results array, total_candidates, and query_time_ms.
    """
    manager = _get_manager()
    session = _get_session()
    cfg = _activation_cfg or ActivationConfig()
    t0 = time.perf_counter()
    async def _resolve_entity_name(entity_id: str) -> str:
        return await manager.resolve_entity_name(entity_id, _group_id)

    async def _get_access_count(entity_id: str) -> int:
        if not entity_id:
            return 0
        value = await manager.get_recall_item_access_count(entity_id)
        return value if isinstance(value, int) else 0

    response_dict = await build_mcp_recall_surface(
        manager,
        group_id=_group_id,
        query=query,
        limit=limit,
        cfg=cfg,
        resolve_entity_name=_resolve_entity_name,
        get_access_count=_get_access_count,
    )
    query_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    session.last_recall_time = time.time()
    session.auto_recall_primed = True
    response_dict["query_time_ms"] = query_time_ms

    # Append near-misses if available (Wave 2)
    near_misses = manager.get_last_near_miss_views()
    if inspect.isawaitable(near_misses):
        near_misses = await near_misses
    if isinstance(near_misses, list) and near_misses:
        response_dict["near_misses"] = near_misses

    surprises = manager.get_surprise_connection_views(_group_id, now=time.time(), limit=3)
    if inspect.isawaitable(surprises):
        surprises = await surprises
    if isinstance(surprises, list) and surprises:
        response_dict["surprise_connections"] = surprises

    await _recall_middleware(query, response_dict, tool_name="recall")
    return json.dumps(response_dict)


@mcp.tool()
async def search_entities(
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 10,
) -> str:
    """Search for specific entities by name or type.

    Args:
        name: Entity name to search for (fuzzy matching)
        entity_type: Filter by entity type (e.g., "Person", "Technology")
        limit: Maximum results (default 10)

    Returns:
        JSON with entities array and total count.
    """
    if not name and not entity_type:
        return json.dumps(
            {
                "status": "error",
                "message": "At least one of 'name' or 'entity_type' is required.",
            }
        )
    manager = _get_manager()
    entities = await manager.search_entities(
        group_id=_group_id, name=name, entity_type=entity_type, limit=limit
    )
    result = {"entities": entities, "total": len(entities)}
    await _recall_middleware(name or entity_type or "", result, tool_name="search_entities")
    return json.dumps(result)


@mcp.tool()
async def search_facts(
    query: str,
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    include_epistemic: bool = False,
    limit: int = 10,
) -> str:
    """Search for facts and relationships in the knowledge graph.

    Args:
        query: What to search for
        subject: Filter by subject entity name
        predicate: Filter by relationship type (e.g., "WORKS_AT", "LIVES_IN")
        include_expired: Include expired/invalidated facts (default False)
        include_epistemic: Include internal decision/artifact graph facts (debug only)
        limit: Maximum results (default 10)

    Returns:
        JSON with facts array and total count.
    """
    manager = _get_manager()
    facts = await manager.search_facts(
        group_id=_group_id,
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        include_epistemic=include_epistemic,
        limit=limit,
    )
    result = {"facts": facts, "total": len(facts)}
    await _recall_middleware(query, result, tool_name="search_facts")
    return json.dumps(result)


@mcp.tool()
async def forget(
    entity_name: str | None = None,
    fact: dict | None = None,
    reason: str | None = None,
) -> str:
    """Mark a memory, entity, or fact as forgotten (soft delete).

    Provide exactly one of entity_name or fact.

    Args:
        entity_name: Name of entity to forget
        fact: Dict with "subject", "predicate", "object" keys to forget a specific fact
        reason: Optional reason for forgetting

    Returns:
        JSON with status and details.
    """
    manager = _get_manager()
    result = await build_mcp_forget_surface(
        manager,
        group_id=_group_id,
        entity_name=entity_name,
        fact=fact,
        reason=reason,
    )
    return json.dumps(result)


@mcp.tool()
async def get_context(
    max_tokens: int = 2000,
    topic_hint: str | None = None,
    project_path: str | None = None,
    format: str = "structured",
) -> str:
    """Return a pre-assembled context string of the most activated memories.

    Args:
        max_tokens: Token budget for context (default 2000)
        topic_hint: Optional topic to bias which memories are loaded
        project_path: Project directory; leaf name used as topic hint
        format: Output format — "structured" (markdown) or "briefing" (LLM narrative)

    Returns:
        JSON with context markdown, entity_count, fact_count, token_estimate.
    """
    manager = _get_manager()
    result = await manager.get_context(
        group_id=_group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
        project_path=project_path,
        format=format,
    )
    # Middleware handles recall + notifications; fallback for get_context built in
    await _recall_middleware(topic_hint or project_path or "", result, tool_name="get_context")
    return json.dumps(result)


@mcp.tool()
async def bootstrap_project(project_path: str) -> str:
    """Auto-observe key project files and create a Project entity.

    Reads README, package.json, pyproject.toml, Makefile, .env.example, and
    CLAUDE.md from the project directory and stores them as episodes. Creates
    a Project entity in the knowledge graph so spreading activation can reach
    project-relevant entities.

    Safe to call every session — files are only re-observed if the last
    bootstrap was more than 24 hours ago (staleness check).

    Args:
        project_path: Absolute path to the project directory.

    Returns:
        JSON with status (bootstrapped/refreshed/already_bootstrapped),
        project_entity_id, and files_observed.
    """
    manager = _get_manager()
    session = _get_session()
    result = await manager.bootstrap_project(
        project_path=project_path,
        group_id=_group_id,
        session_id=session.session_id,
    )
    return json.dumps(result)


@mcp.tool()
async def route_question(
    question: str,
    project_path: str | None = None,
    history: list[str] | None = None,
) -> str:
    """Classify a question as remember, inspect, or reconcile."""
    manager = _get_manager()
    session_entity_names = _get_conv_top_entity_names(manager)
    result = await build_question_route_surface(
        manager,
        group_id=_group_id,
        question=question,
        project_path=project_path,
        history=history,
        session_entity_names=session_entity_names,
        surface="mcp",
    )
    await _recall_middleware(question, result, tool_name="route_question", auto_observe=True)
    return json.dumps(result)


@mcp.tool()
async def search_artifacts(
    query: str,
    project_path: str | None = None,
    limit: int = 5,
) -> str:
    """Search the bootstrapped artifact substrate."""
    manager = _get_manager()
    result = await build_mcp_artifact_search_surface(
        manager,
        group_id=_group_id,
        query=query,
        project_path=project_path,
        limit=limit,
    )
    await _recall_middleware(query, result, tool_name="search_artifacts")
    return json.dumps(result)


@mcp.tool()
async def get_runtime_state(project_path: str | None = None) -> str:
    """Return effective runtime/config state and artifact freshness."""
    manager = _get_manager()
    result = await manager.get_runtime_state(
        group_id=_group_id,
        project_path=project_path,
    )
    return json.dumps(result)


@mcp.tool()
async def get_graph_state(
    top_n: int = 20,
    include_edges: bool = False,
    entity_types: list[str] | None = None,
) -> str:
    """Return current graph statistics and top-activated nodes.

    Args:
        top_n: Number of top-activated entities to return (default 20)
        include_edges: Include relationship edges for top entities (default False)
        entity_types: Filter by entity types (e.g., ["Person", "Technology"])

    Returns:
        JSON with stats, top_activated, edges (if requested), and group_id.
    """
    manager = _get_manager()
    result = await manager.get_graph_state(
        group_id=_group_id,
        top_n=top_n,
        include_edges=include_edges,
        entity_types=entity_types,
    )
    return json.dumps(result)


@mcp.tool()
async def get_lifecycle_summary(episode_limit: int = 5, cycle_limit: int = 10) -> str:
    """Return the shared Capture -> Cue -> Project -> Recall -> Consolidate summary."""
    manager = _get_manager()
    consolidation_engine = None
    consolidation_reader = None
    if _consolidation_store is not None:
        consolidation_engine = SimpleNamespace(
            is_running=False,
        )
        consolidation_reader = ConsolidationAuditReader(_consolidation_store)
    summary = await manager.get_lifecycle_summary(
        group_id=_group_id,
        consolidation_engine=consolidation_engine,
        consolidation_reader=consolidation_reader,
        activation_config=_activation_cfg,
        episode_limit=max(1, episode_limit),
        cycle_limit=max(1, cycle_limit),
    )
    return json.dumps(summary)


@mcp.tool()
async def record_recall_evaluation(
    recall_triggered: bool,
    recall_helped: bool,
    recall_needed: bool | None = None,
    packets_surfaced: int = 0,
    packets_used: int = 0,
    false_recalls: int = 0,
    source: str = "mcp",
    query: str | None = None,
    notes: str | None = None,
) -> str:
    """Record a labeled recall-quality sample for the current brain.

    Args:
        recall_triggered: Whether the agent chose to recall memory
        recall_helped: Whether the recalled memory helped the task
        recall_needed: Whether memory should have been recalled for this task
        packets_surfaced: Number of memory packets/results surfaced
        packets_used: Number of surfaced packets/results actually used
        false_recalls: Number of surfaced memories judged misleading or irrelevant
        source: Label source, defaults to "mcp"
        query: Optional query or task that triggered recall
        notes: Optional short operator/agent note

    Returns:
        JSON acknowledgement with the persisted sample contract.
    """
    store = _get_evaluation_store()
    sample = await persist_recall_eval_sample(
        store,
        group_id=_group_id,
        recall_triggered=recall_triggered,
        recall_helped=recall_helped,
        recall_needed=recall_needed,
        packets_surfaced=packets_surfaced,
        packets_used=packets_used,
        false_recalls=false_recalls,
        source=source,
        query=query,
        notes=notes,
    )
    return json.dumps(present_recall_sample_write(sample, surface="mcp"))


@mcp.tool()
async def record_session_continuity_evaluation(
    baseline_score: float,
    memory_score: float,
    open_loop_expected: bool = False,
    open_loop_recovered: bool = False,
    temporal_expected: bool = False,
    temporal_correct: bool = False,
    source: str = "mcp",
    scenario: str | None = None,
    notes: str | None = None,
) -> str:
    """Record a labeled multi-turn/session-continuity sample.

    Args:
        baseline_score: Score without Engram memory
        memory_score: Score with Engram memory
        open_loop_expected: Whether the task required resurfacing unresolved work
        open_loop_recovered: Whether memory recovered that open loop
        temporal_expected: Whether the task required newer-vs-older fact handling
        temporal_correct: Whether memory handled the temporal fact correctly
        source: Label source, defaults to "mcp"
        scenario: Optional scenario name
        notes: Optional short operator/agent note

    Returns:
        JSON acknowledgement with the persisted sample contract.
    """
    store = _get_evaluation_store()
    sample = await persist_session_continuity_sample(
        store,
        group_id=_group_id,
        baseline_score=baseline_score,
        memory_score=memory_score,
        open_loop_expected=open_loop_expected,
        open_loop_recovered=open_loop_recovered,
        temporal_expected=temporal_expected,
        temporal_correct=temporal_correct,
        source=source,
        scenario=scenario,
        notes=notes,
    )
    return json.dumps(present_session_sample_write(sample, surface="mcp"))


@mcp.tool()
async def get_evaluation_report(
    cycle_limit: int = 10,
    sample_limit: int = 500,
) -> str:
    """Return the local Capture -> Cue -> Project -> Recall -> Consolidate report.

    Args:
        cycle_limit: Recent consolidation cycles to include
        sample_limit: Saved recall/session evaluation samples to include

    Returns:
        JSON brain-loop evaluation report for the current group.
    """
    manager = _get_manager()
    store = _get_evaluation_store()
    recent_cycles, calibration_snapshots = await _load_consolidation_evaluation_inputs(
        manager,
        _group_id,
        cycle_limit=max(1, cycle_limit),
    )
    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id=_group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        sample_limit=max(1, sample_limit),
        snapshot_source="mcp_report",
    )
    return json.dumps(report)


@mcp.tool()
async def mark_identity_core(entity_name: str, identity_core: bool = True) -> str:
    """Mark or unmark an entity as part of the user's identity core.

    Identity core entities are protected from pruning and always included
    in context loading.

    Args:
        entity_name: Name of the entity to mark
        identity_core: True to protect, False to unprotect

    Returns:
        JSON with status and entity details.
    """
    manager = _get_manager()
    result = await manager.mark_identity_core(
        entity_name,
        identity_core=identity_core,
        group_id=_group_id,
    )
    return json.dumps(result)


@mcp.tool()
async def trigger_consolidation(dry_run: bool = True) -> str:
    """Trigger a memory consolidation cycle.

    Merges duplicates, infers edges, prunes dead entities, compacts histories.

    Args:
        dry_run: If true (default), report what would change without modifying data.

    Returns:
        JSON with cycle_id and status.
    """
    manager = _get_manager()

    from engram.consolidation.presenter import (
        serialize_cycle_summary,
    )

    store = await _get_mcp_consolidation_trigger_store(manager)

    trigger_result = await manager.trigger_consolidation_cycle(
        group_id=_group_id,
        trigger="mcp",
        dry_run=dry_run,
        consolidation_store=store,
    )

    result = serialize_cycle_summary(trigger_result.cycle)
    result["cycle_id"] = result.pop("id")
    result["graph_stats"] = trigger_result.graph_stats

    return json.dumps(result)


async def _get_mcp_consolidation_trigger_store(manager: GraphManager) -> Any | None:
    """Return the active MCP audit store, with a lite shared-DB fallback."""
    if _consolidation_store is not None:
        return _consolidation_store

    db = manager.get_consolidation_shared_db()
    if db is None:
        return None

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(":memory:")
    await store.initialize(db=db)
    return store


@mcp.tool()
async def get_consolidation_status() -> str:
    """Check whether a consolidation cycle is running and get the latest cycle summary.

    Returns:
        JSON with is_running flag and latest cycle summary if available.
    """
    from engram.consolidation.presenter import serialize_cycle_summary

    # In MCP mode, consolidation runs synchronously so is_running is always false
    result = {
        "is_running": False,
        "message": "Use trigger_consolidation to run a cycle. "
        "In MCP mode, cycles run synchronously.",
    }
    latest_cycle = await ConsolidationAuditReader(_consolidation_store).latest_cycle(_group_id)
    if latest_cycle is not None:
        result["latest_cycle"] = serialize_cycle_summary(latest_cycle)
    return json.dumps(result)


# ─── Prospective Memory (Wave 4) ─────────────────────────────────────


@mcp.tool()
async def intend(
    trigger_text: str,
    action_text: str,
    trigger_type: str = "activation",
    entity_names: list[str] | None = None,
    threshold: float | None = None,
    priority: str = "normal",
    context: str | None = None,
    see_also: list[str] | None = None,
    refresh_trigger: str = "manual",
) -> str:
    """Create a graph-embedded intention (prospective memory trigger).

    The intention fires automatically during remember/observe when related
    entities are activated in the knowledge graph. Use for "remind me when
    X happens" or "next time I work on Y, tell me Z" behavior.

    Use trigger_type="refresh_context" with refresh_trigger="after_consolidation"
    for persistent context queries (pinned contexts) that auto-refresh after
    each consolidation cycle. The trigger_text serves as the topic query.
    Results are cached on the intention and included in get_context() output.

    Args:
        trigger_text: Natural language description of the trigger condition
            (e.g., "auth module", "Python upgrades", "job interview").
            For pinned contexts, this is the topic query.
        action_text: What to surface when the trigger fires
            (e.g., "Check the XSS fix before deploying").
            For pinned contexts, a short label for the pinned query.
        trigger_type: "activation" (spreading activation threshold),
            "entity_mention" (fires when named entity appears in content),
            or "refresh_context" (pinned context query).
        entity_names: Entity names to link via TRIGGERED_BY edges. These
            entities activate the intention through graph spreading.
        threshold: Activation threshold override (0.0-1.0, default 0.5)
        priority: "low", "normal", "high", or "critical" (affects ordering)
        context: Rich background the agent needs at fire time. When provided,
            the agent can act on the intention without additional recall/search.
        see_also: Breadcrumb topic hints ("cliffhangers"). The agent will
            mention these as conversational hooks rather than searching them.
        refresh_trigger: "manual" (default) or "after_consolidation" (auto-refresh
            pinned context after each consolidation cycle).

    Returns:
        JSON with status, intention_id, linked entities, and threshold.
    """
    manager = _get_manager()
    try:
        payload = await build_mcp_create_intention_surface(
            manager,
            group_id=_group_id,
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            entity_names=entity_names,
            threshold=threshold,
            priority=priority,
            context=context,
            see_also=see_also,
            refresh_trigger=refresh_trigger,
        )
        return json.dumps(payload)
    except ValueError as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def dismiss_intention(intention_id: str, hard: bool = False) -> str:
    """Dismiss (disable or delete) a prospective memory intention.

    Args:
        intention_id: The ID of the intention to dismiss.
        hard: If true, permanently delete. If false (default), soft-disable.

    Returns:
        JSON with status and intention_id.
    """
    manager = _get_manager()
    try:
        payload = await build_mcp_dismiss_intention_surface(
            manager,
            group_id=_group_id,
            intention_id=intention_id,
            hard=hard,
        )
        return json.dumps(payload)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
async def list_intentions(enabled_only: bool = True) -> str:
    """List active prospective memory intentions with warmth info.

    Args:
        enabled_only: If true, only return enabled and non-expired intentions.

    Returns:
        JSON with intentions array and total count.
    """
    manager = _get_manager()
    payload = await build_intention_list_surface(
        manager,
        group_id=_group_id,
        enabled_only=enabled_only,
        surface="mcp",
    )

    return json.dumps(payload)


# ─── Resources ──────────────────────────────────────────────────────


@mcp.resource("engram://graph/stats")
async def graph_stats_resource() -> str:
    """Current graph statistics: entity counts, relationship counts, type distribution."""
    manager = _get_manager()
    state = await manager.get_graph_state(group_id=_group_id, top_n=10, include_edges=False)
    return json.dumps(state["stats"])


@mcp.resource("engram://entity/{entity_id}")
async def entity_profile_resource(entity_id: str) -> str:
    """Full profile for a specific entity including activation and relationships."""
    manager = _get_manager()
    return json.dumps(await manager.get_entity_profile(entity_id, _group_id))


@mcp.resource("engram://entity/{entity_id}/neighbors")
async def entity_neighbors_resource(entity_id: str) -> str:
    """Entities directly connected to the specified entity with relationship details."""
    manager = _get_manager()
    return json.dumps(await manager.get_entity_neighbors(entity_id, _group_id))


# ─── Prompts ────────────────────────────────────────────────────────


@mcp.prompt()
def engram_system(
    persona: str = "assistant",
    auto_remember: str = "important",
) -> str:
    """System prompt instructions for using Engram memory tools in conversation."""
    return ENGRAM_SYSTEM_PROMPT


@mcp.prompt()
def engram_context_loader(topic: str | None = None) -> str:
    """Pre-load memory context at conversation start."""
    hint = f' Pass topic_hint="{topic}".' if topic else ""
    return f"{ENGRAM_CONTEXT_LOADER_PROMPT}{hint}"


# ─── Entry point ────────────────────────────────────────────────────


def main(
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Entry point for MCP server (stdio or streamable-http)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    transport_name = transport or os.environ.get("ENGRAM_TRANSPORT", "stdio")
    if transport_name not in {"stdio", "sse", "streamable-http"}:
        raise ValueError(f"Unsupported transport: {transport_name}")
    transport_literal = cast(
        Literal["stdio", "sse", "streamable-http"],
        transport_name,
    )

    if transport_literal in ("streamable-http", "sse"):
        mcp.settings.host = host or os.environ.get("ENGRAM_MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(port or os.environ.get("ENGRAM_MCP_PORT", "8200"))
        mcp.settings.stateless_http = True
        logger.info(
            "Starting MCP %s server at http://%s:%s/mcp",
            transport_literal,
            mcp.settings.host,
            mcp.settings.port,
        )

    mcp.run(transport=transport_literal)


if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser()
    _parser.add_argument("--transport", default=None)
    _parser.add_argument("--host", default=None)
    _parser.add_argument("--port", type=int, default=None)
    _args = _parser.parse_args()
    main(transport=_args.transport, host=_args.host, port=_args.port)
