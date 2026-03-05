"""MCP server for Engram — native stdio transport."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from engram.config import ActivationConfig, EngramConfig
from engram.events.bus import get_event_bus
from engram.extraction.extractor import EntityExtractor
from engram.graph_manager import GraphManager
from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Initialize storage on the same event loop as the MCP server."""
    await _init()
    yield


mcp = FastMCP("engram", instructions=ENGRAM_SYSTEM_PROMPT, lifespan=_lifespan)


# ─── Session State ──────────────────────────────────────────────────


@dataclass
class SessionState:
    """Tracks session-level metadata for the MCP server lifetime."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: str = field(default_factory=lambda: os.environ.get("ENGRAM_GROUP_ID", "default"))
    started_at: datetime = field(default_factory=datetime.utcnow)
    episode_count: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    auto_recall_primed: bool = False
    last_recall_time: float = 0.0


class RecallCooldown:
    """Rate limiter + topic dedup for auto-recall."""

    def __init__(self, max_per_minute: int = 3, cooldown_seconds: float = 60.0) -> None:
        self._entries: deque[tuple[set[str], float]] = deque(maxlen=20)
        self.max_per_minute = max_per_minute
        self.cooldown_seconds = cooldown_seconds

    def _tokenize(self, query: str) -> set[str]:
        return {w.lower() for w in query.split() if len(w) > 2}

    def is_throttled(self, query: str, now: float) -> bool:
        # Rate limit: count entries in last 60s
        recent = [t for _, t in self._entries if now - t < 60.0]
        if len(recent) >= self.max_per_minute:
            return True
        # Topic dedup: check token overlap within cooldown window
        tokens = self._tokenize(query)
        if not tokens:
            return False
        for prev_tokens, ts in self._entries:
            if now - ts > self.cooldown_seconds:
                continue
            if not prev_tokens:
                continue
            overlap = len(tokens & prev_tokens) / max(len(tokens | prev_tokens), 1)
            if overlap > 0.5:
                return True
        return False

    def record(self, query: str, now: float) -> None:
        self._entries.append((self._tokenize(query), now))


# Module-level state initialized on startup
_manager: GraphManager | None = None
_group_id: str = "default"
_session: SessionState | None = None
_recall_cooldown: RecallCooldown | None = None
_activation_cfg: ActivationConfig | None = None


async def _init() -> None:
    """Initialize storage and services."""
    global _manager, _group_id, _session, _recall_cooldown, _activation_cfg

    config = EngramConfig()
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    if hasattr(search_index, "initialize"):
        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await search_index.initialize(db=graph_store._db)
        else:
            await search_index.initialize()

    extractor = EntityExtractor()
    event_bus = get_event_bus()
    _manager = GraphManager(
        graph_store, activation_store, search_index, extractor,
        cfg=config.activation, event_bus=event_bus,
    )
    _group_id = os.environ.get("ENGRAM_GROUP_ID", config.default_group_id)
    _session = SessionState(group_id=_group_id)
    _activation_cfg = config.activation
    _recall_cooldown = RecallCooldown(
        max_per_minute=config.activation.auto_recall_max_per_minute,
        cooldown_seconds=config.activation.auto_recall_cooldown_seconds,
    )

    # Background episode worker
    from engram.worker import EpisodeWorker

    if config.activation.worker_enabled:
        _worker = EpisodeWorker(_manager, config.activation)
        _worker.start(_group_id, event_bus)

    # In full mode, bridge events to Redis so the REST API/dashboard can see them
    if mode == EngineMode.FULL:
        from engram.events.redis_bridge import create_publisher

        publisher = await create_publisher(_group_id, redis_url=config.redis.url)
        if publisher:
            event_bus.add_on_publish_hook(publisher)

    logger.info(
        "Engram MCP server initialized (%s mode, session=%s)", mode.value, _session.session_id
    )


def _get_manager() -> GraphManager:
    if _manager is None:
        raise RuntimeError("MCP server not initialized")
    return _manager


def _get_session() -> SessionState:
    if _session is None:
        raise RuntimeError("MCP server not initialized")
    return _session


# ─── AutoRecall Helpers ──────────────────────────────────────────────


def _extract_recall_query(content: str) -> str:
    """Extract a recall query from content: proper nouns first, then first sentence."""
    if len(content) < 20:
        return ""
    # Extract capitalized proper nouns
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", content)
    if proper_nouns:
        return " ".join(proper_nouns)[:200]
    # Fallback: first meaningful sentence
    first_sentence = content.split(".")[0].strip()
    return first_sentence[:200]


async def _auto_recall(
    content: str, manager: GraphManager, cfg: ActivationConfig,
) -> dict | None:
    """Piggyback lightweight recall on observe/remember calls."""
    if not cfg.auto_recall_enabled:
        return None

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
    if (cfg.conv_topic_shift_enabled
            and hasattr(manager, '_conv_context') and manager._conv_context is not None
            and manager._conv_context.detect_topic_shift()):
        recall_limit = cfg.conv_topic_shift_recall_boost
        manager._conv_context.acknowledge_shift()

    try:
        results = await manager.recall(
            query=query, group_id=_group_id, limit=recall_limit,
        )
    except Exception:
        logger.debug("auto_recall failed", exc_info=True)
        return None

    # Filter: skip episodes, skip below min_score
    entities = []
    for r in results:
        if r.get("result_type") == "episode":
            continue
        if r.get("score", 0) < cfg.auto_recall_min_score:
            continue
        entity = r["entity"]
        # Compact format — no expensive relationship resolution
        facts = []
        for rel in r.get("relationships", [])[:3]:
            facts.append(f"{rel.get('predicate', '?')}")
        entry: dict = {
            "name": entity["name"],
            "type": entity["type"],
            "summary": (entity.get("summary") or "")[:100],
        }
        if facts:
            entry["top_facts"] = facts
        entities.append(entry)

    if not entities:
        return None

    if cooldown:
        cooldown.record(query, now)

    return {
        "source": "auto_recall",
        "query_used": query,
        "entities": entities,
    }


async def _session_prime(
    content: str | None, manager: GraphManager, cfg: ActivationConfig,
) -> dict | None:
    """Auto-prime context on first tool call in a session."""
    if not cfg.auto_recall_session_prime:
        return None
    session = _get_session()
    if session.auto_recall_primed:
        return None
    session.auto_recall_primed = True

    topic = None
    if content:
        topic = _extract_recall_query(content) or None

    try:
        result = await manager.get_context(
            group_id=_group_id,
            max_tokens=cfg.auto_recall_session_prime_max_tokens,
            topic_hint=topic,
            format="briefing",
        )
        return result
    except Exception:
        logger.debug("session_prime failed", exc_info=True)
        return None


# ─── Tools ──────────────────────────────────────────────────────────


@mcp.tool()
async def remember(content: str, source: str = "mcp") -> str:
    """Store a memory. Extracts entities and relationships from the text.

    Args:
        content: The text to remember (conversation excerpt, fact, note, etc.)
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    cfg = _activation_cfg
    episode_id = await manager.ingest_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
    )
    session.episode_count += 1
    session.last_activity = datetime.utcnow()
    # Update conversation context (Wave 2)
    if hasattr(manager, '_conv_context') and manager._conv_context is not None:
        manager._conv_context.add_turn(content)
    response: dict = {
        "status": "stored",
        "episode_id": episode_id,
        "message": "Memory received. Entities and relationships extracted.",
    }
    if cfg and cfg.auto_recall_on_remember:
        prime = await _session_prime(content, manager, cfg)
        if prime:
            response["session_context"] = prime
        recalled = await _auto_recall(content, manager, cfg)
        if recalled:
            response["recalled_context"] = recalled
    # Surface triggered intentions (Wave 4)
    if manager._triggered_intentions:
        response["triggered_intentions"] = [
            {
                "trigger": m.trigger_text,
                "action": m.action_text,
                "similarity": round(m.similarity, 4),
                "matched_via": m.matched_via,
                **({"context": m.context} if m.context else {}),
                **({"see_also": m.see_also} if m.see_also else {}),
            }
            for m in manager._triggered_intentions
        ]
        manager._triggered_intentions = []
    return json.dumps(response)


@mcp.tool()
async def observe(content: str, source: str = "mcp") -> str:
    """Store raw text without extraction. Fast path for bulk capture.

    Args:
        content: The text to store (conversation excerpt, fact, note, etc.)
        source: Where this memory came from (e.g., "claude_desktop", "claude_code")

    Returns:
        JSON with status, episode_id, and message.
    """
    manager = _get_manager()
    session = _get_session()
    cfg = _activation_cfg
    episode_id = await manager.store_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
    )
    session.episode_count += 1
    session.last_activity = datetime.utcnow()
    # Update conversation context (Wave 2)
    if hasattr(manager, '_conv_context') and manager._conv_context is not None:
        manager._conv_context.add_turn(content)
    response: dict = {
        "status": "stored",
        "episode_id": episode_id,
        "message": "Stored for background processing.",
    }
    if cfg and cfg.auto_recall_on_observe:
        prime = await _session_prime(content, manager, cfg)
        if prime:
            response["session_context"] = prime
        recalled = await _auto_recall(content, manager, cfg)
        if recalled:
            response["recalled_context"] = recalled
    # Surface triggered intentions (Wave 4)
    if manager._triggered_intentions:
        response["triggered_intentions"] = [
            {
                "trigger": m.trigger_text,
                "action": m.action_text,
                "similarity": round(m.similarity, 4),
                "matched_via": m.matched_via,
                **({"context": m.context} if m.context else {}),
                **({"see_also": m.see_also} if m.see_also else {}),
            }
            for m in manager._triggered_intentions
        ]
        manager._triggered_intentions = []
    return json.dumps(response)


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
    t0 = time.perf_counter()
    results = await manager.recall(query=query, group_id=_group_id, limit=limit)
    query_time_ms = round((time.perf_counter() - t0) * 1000, 1)
    session.last_recall_time = time.time()
    session.auto_recall_primed = True

    formatted = []
    for r in results:
        if r.get("result_type") == "episode":
            continue
        entity = r["entity"]
        # Resolve relationship target/source IDs to entity names
        related_facts = []
        for rel in r.get("relationships", []):
            source_name = await manager.resolve_entity_name(rel["source_id"], _group_id)
            target_name = await manager.resolve_entity_name(rel["target_id"], _group_id)
            related_facts.append(
                {
                    "subject": source_name,
                    "predicate": rel["predicate"],
                    "object": target_name,
                }
            )

        state = await manager._activation.get_activation(entity["id"])
        access_count = state.access_count if state else 0

        formatted.append(
            {
                "entity": entity["name"],
                "entity_type": entity["type"],
                "summary": entity.get("summary"),
                "composite_score": round(r["score"], 4),
                "score_breakdown": {
                    k: round(v, 4) for k, v in r.get("score_breakdown", {}).items()
                },
                "related_facts": related_facts,
                "access_count": access_count,
            }
        )

    response_dict: dict = {
        "results": formatted,
        "total_candidates": len(formatted),
        "query_time_ms": query_time_ms,
    }

    # Append near-misses if available (Wave 2)
    if hasattr(manager, '_last_near_misses') and manager._last_near_misses:
        response_dict["near_misses"] = manager._last_near_misses

    # Surface surprise connections (Wave 3)
    if hasattr(manager, '_surprise_cache') and manager._surprise_cache is not None:
        surprises = manager._surprise_cache.get(_group_id, time.time())
        if surprises:
            response_dict["surprise_connections"] = [
                {
                    "entity": s.entity_name,
                    "connected_to": s.connected_to_name,
                    "relationship": s.predicate,
                    "surprise_score": round(s.surprise_score, 4),
                }
                for s in surprises[:3]
            ]

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
    return json.dumps({"entities": entities, "total": len(entities)})


@mcp.tool()
async def search_facts(
    query: str,
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    limit: int = 10,
) -> str:
    """Search for facts and relationships in the knowledge graph.

    Args:
        query: What to search for
        subject: Filter by subject entity name
        predicate: Filter by relationship type (e.g., "WORKS_AT", "LIVES_IN")
        include_expired: Include expired/invalidated facts (default False)
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
        limit=limit,
    )
    return json.dumps({"facts": facts, "total": len(facts)})


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
    if (entity_name is None) == (fact is None):
        return json.dumps(
            {
                "status": "error",
                "message": "Provide exactly one of 'entity_name' or 'fact'.",
            }
        )
    manager = _get_manager()
    if entity_name:
        result = await manager.forget_entity(entity_name, _group_id, reason=reason)
    else:
        result = await manager.forget_fact(
            subject_name=fact["subject"],
            predicate=fact["predicate"],
            object_name=fact["object"],
            group_id=_group_id,
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
    entities = await manager._graph.find_entities(
        name=entity_name, group_id=_group_id, limit=1
    )
    if not entities:
        return json.dumps(
            {"status": "error", "message": f"Entity '{entity_name}' not found."}
        )
    entity = entities[0]
    await manager._graph.update_entity(
        entity.id,
        {"identity_core": 1 if identity_core else 0},
        group_id=_group_id,
    )
    action = "marked as" if identity_core else "removed from"
    return json.dumps(
        {
            "status": "updated",
            "entity": entity.name,
            "entity_type": entity.entity_type,
            "identity_core": identity_core,
            "message": f"Entity '{entity.name}' {action} identity core.",
        }
    )


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

    from engram.consolidation.engine import ConsolidationEngine
    from engram.consolidation.store import SQLiteConsolidationStore

    # Create a one-shot engine for MCP context
    store = None
    if hasattr(manager._graph, "_db"):
        store = SQLiteConsolidationStore(":memory:")
        await store.initialize(db=manager._graph._db)

    engine = ConsolidationEngine(
        manager._graph,
        manager._activation,
        manager._search,
        cfg=manager._cfg,
        consolidation_store=store,
        extractor=manager._extractor,
    )

    # Get graph stats for context
    graph_stats = await manager._graph.get_stats(_group_id)

    cycle = await engine.run_cycle(
        group_id=_group_id,
        trigger="mcp",
        dry_run=dry_run,
    )

    total_processed = sum(pr.items_processed for pr in cycle.phase_results)
    total_affected = sum(pr.items_affected for pr in cycle.phase_results)
    description = (
        f"{'Dry run' if cycle.dry_run else 'Live'} cycle: "
        f"{total_processed} items processed, {total_affected} affected "
        f"across {len(cycle.phase_results)} phases"
    )

    return json.dumps(
        {
            "cycle_id": cycle.id,
            "status": cycle.status,
            "dry_run": cycle.dry_run,
            "graph_stats": graph_stats,
            "phases": [
                {
                    "phase": pr.phase,
                    "status": pr.status,
                    "items_processed": pr.items_processed,
                    "items_affected": pr.items_affected,
                }
                for pr in cycle.phase_results
            ],
            "total_duration_ms": cycle.total_duration_ms,
            "summary": {
                "total_processed": total_processed,
                "total_affected": total_affected,
                "description": description,
            },
        }
    )


@mcp.tool()
async def get_consolidation_status() -> str:
    """Check whether a consolidation cycle is running and get the latest cycle summary.

    Returns:
        JSON with is_running flag and latest cycle summary if available.
    """
    # In MCP mode, consolidation runs synchronously so is_running is always false
    return json.dumps(
        {
            "is_running": False,
            "message": "Use trigger_consolidation to run a cycle. "
            "In MCP mode, cycles run synchronously.",
        }
    )


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
) -> str:
    """Create a graph-embedded intention (prospective memory trigger).

    The intention fires automatically during remember/observe when related
    entities are activated in the knowledge graph. Use for "remind me when
    X happens" or "next time I work on Y, tell me Z" behavior.

    Args:
        trigger_text: Natural language description of the trigger condition
            (e.g., "auth module", "Python upgrades", "job interview")
        action_text: What to surface when the trigger fires
            (e.g., "Check the XSS fix before deploying")
        trigger_type: "activation" (spreading activation threshold) or
            "entity_mention" (fires when named entity appears in content)
        entity_names: Entity names to link via TRIGGERED_BY edges. These
            entities activate the intention through graph spreading.
        threshold: Activation threshold override (0.0-1.0, default 0.5)
        priority: "low", "normal", "high", or "critical" (affects ordering)
        context: Rich background the agent needs at fire time. When provided,
            the agent can act on the intention without additional recall/search.
        see_also: Breadcrumb topic hints ("cliffhangers"). The agent will
            mention these as conversational hooks rather than searching them.

    Returns:
        JSON with status, intention_id, linked entities, and threshold.
    """
    manager = _get_manager()
    try:
        intention_id = await manager.create_intention(
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            entity_names=entity_names,
            threshold=threshold,
            priority=priority,
            group_id=_group_id,
            context=context,
            see_also=see_also,
        )
        return json.dumps({
            "status": "created",
            "intention_id": intention_id,
            "trigger_type": trigger_type,
            "linked_entities": entity_names or [],
            "threshold": threshold or manager._cfg.prospective_activation_threshold,
            "message": f"Intention set: will fire when '{trigger_text}' activates.",
        })
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
        await manager.dismiss_intention(intention_id, _group_id, hard=hard)
        return json.dumps({
            "status": "dismissed",
            "intention_id": intention_id,
            "hard": hard,
            "message": f"Intention {intention_id} {'deleted' if hard else 'disabled'}.",
        })
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
    cfg = manager._cfg
    intentions = await manager.list_intentions(
        group_id=_group_id, enabled_only=enabled_only,
    )

    items = []
    if cfg.prospective_graph_embedded:
        from engram.activation.engine import compute_activation
        from engram.models.prospective import IntentionMeta

        now = time.time()
        for entity in intentions:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            # Compute warmth ratio
            state = await manager._activation.get_activation(entity.id)
            activation = 0.0
            if state:
                activation = compute_activation(state.access_history, now, cfg)
            warmth_ratio = (
                activation / meta.activation_threshold
                if meta.activation_threshold > 0 else 0.0
            )

            # Classify warmth
            levels = cfg.prospective_warmth_levels
            if warmth_ratio >= 1.0:
                warmth_label = "hot"
            elif warmth_ratio >= levels[2]:
                warmth_label = "warm"
            elif warmth_ratio >= levels[1]:
                warmth_label = "warming"
            elif warmth_ratio >= levels[0]:
                warmth_label = "cool"
            else:
                warmth_label = "dormant"

            items.append({
                "id": entity.id,
                "trigger_text": meta.trigger_text,
                "action_text": meta.action_text,
                "trigger_type": meta.trigger_type,
                "threshold": meta.activation_threshold,
                "fire_count": meta.fire_count,
                "max_fires": meta.max_fires,
                "enabled": meta.enabled,
                "priority": meta.priority,
                "expires_at": meta.expires_at,
                "warmth_ratio": round(warmth_ratio, 4),
                "warmth_label": warmth_label,
                "linked_entity_ids": meta.trigger_entity_ids,
            })
    else:
        # v1 fallback
        for i in intentions:
            items.append({
                "id": i.id,
                "trigger_text": i.trigger_text,
                "action_text": i.action_text,
                "trigger_type": i.trigger_type,
                "entity_name": i.entity_name,
                "threshold": i.threshold,
                "fire_count": i.fire_count,
                "max_fires": i.max_fires,
                "enabled": i.enabled,
                "expires_at": i.expires_at.isoformat() if i.expires_at else None,
            })

    return json.dumps({"intentions": items, "total": len(items)})


# ─── Resources ──────────────────────────────────────────────────────


@mcp.resource("engram://graph/stats")
async def graph_stats_resource() -> str:
    """Current graph statistics: entity counts, relationship counts, type distribution."""
    manager = _get_manager()
    stats = await manager._graph.get_stats(_group_id)
    type_counts = await manager._graph.get_entity_type_counts(_group_id)
    stats["entity_type_distribution"] = type_counts
    return json.dumps(stats)


@mcp.resource("engram://entity/{entity_id}")
async def entity_profile_resource(entity_id: str) -> str:
    """Full profile for a specific entity including activation and relationships."""
    manager = _get_manager()
    entity = await manager._graph.get_entity(entity_id, _group_id)
    if not entity:
        return json.dumps({"error": "Entity not found", "entity_id": entity_id})

    state = await manager._activation.get_activation(entity_id)
    rels = await manager._graph.get_relationships(entity_id, active_only=True, group_id=_group_id)

    facts = []
    for r in rels:
        target_name = await manager.resolve_entity_name(r.target_id, _group_id)
        source_name = await manager.resolve_entity_name(r.source_id, _group_id)
        if r.source_id == entity_id:
            facts.append(
                {
                    "predicate": r.predicate,
                    "object": target_name,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                }
            )
        else:
            facts.append(
                {
                    "predicate": r.predicate,
                    "subject": source_name,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                }
            )

    from engram.activation.engine import compute_activation

    now = time.time()
    activation_score = 0.0
    if state:
        activation_score = compute_activation(state.access_history, now, manager._cfg)

    return json.dumps(
        {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "summary": entity.summary,
            "activation": {
                "current": round(activation_score, 4),
                "access_count": state.access_count if state else 0,
                "last_accessed": state.last_accessed if state else None,
            },
            "facts": facts,
            "created_at": entity.created_at.isoformat() if entity.created_at else None,
            "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
        }
    )


@mcp.resource("engram://entity/{entity_id}/neighbors")
async def entity_neighbors_resource(entity_id: str) -> str:
    """Entities directly connected to the specified entity with relationship details."""
    manager = _get_manager()
    neighbors = await manager._graph.get_neighbors(entity_id, hops=1, group_id=_group_id)
    result = []
    for entity, rel in neighbors:
        result.append(
            {
                "entity": {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                },
                "relationship": {
                    "predicate": rel.predicate,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "weight": rel.weight,
                },
            }
        )
    return json.dumps(result)


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


def main() -> None:
    """Entry point for MCP stdio server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    transport = os.environ.get("ENGRAM_TRANSPORT", "stdio")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
