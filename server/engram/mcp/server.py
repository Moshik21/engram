"""MCP server for Engram — native stdio transport."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from engram.config import EngramConfig
from engram.extraction.extractor import EntityExtractor
from engram.graph_manager import GraphManager
from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode

logger = logging.getLogger(__name__)

mcp = FastMCP("engram")


# ─── Session State ──────────────────────────────────────────────────


@dataclass
class SessionState:
    """Tracks session-level metadata for the MCP server lifetime."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: str = "default"
    started_at: datetime = field(default_factory=datetime.utcnow)
    episode_count: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)


# Module-level state initialized on startup
_manager: GraphManager | None = None
_group_id: str = "default"
_session: SessionState | None = None


async def _init() -> None:
    """Initialize storage and services."""
    global _manager, _group_id, _session

    config = EngramConfig()
    mode = EngineMode.LITE  # MCP stdio always uses lite
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    # Share the db connection with FTS5
    if hasattr(search_index, "initialize"):
        if hasattr(graph_store, "_db"):
            await search_index.initialize(db=graph_store._db)
        else:
            await search_index.initialize()

    extractor = EntityExtractor()
    _manager = GraphManager(
        graph_store, activation_store, search_index, extractor, cfg=config.activation
    )
    _group_id = os.environ.get("ENGRAM_GROUP_ID", config.default_group_id)
    _session = SessionState(group_id=_group_id)

    logger.info("Engram MCP server initialized (lite mode, session=%s)", _session.session_id)


def _get_manager() -> GraphManager:
    if _manager is None:
        raise RuntimeError("MCP server not initialized")
    return _manager


def _get_session() -> SessionState:
    if _session is None:
        raise RuntimeError("MCP server not initialized")
    return _session


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
    episode_id = await manager.ingest_episode(
        content=content,
        group_id=_group_id,
        source=source,
        session_id=session.session_id,
    )
    session.episode_count += 1
    session.last_activity = datetime.utcnow()
    return json.dumps(
        {
            "status": "stored",
            "episode_id": episode_id,
            "message": "Memory received. Entities and relationships extracted.",
        }
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
    t0 = time.perf_counter()
    results = await manager.recall(query=query, group_id=_group_id, limit=limit)
    query_time_ms = round((time.perf_counter() - t0) * 1000, 1)

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

    return json.dumps(
        {
            "results": formatted,
            "total_candidates": len(formatted),
            "query_time_ms": query_time_ms,
        }
    )


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
async def get_context(max_tokens: int = 2000, topic_hint: str | None = None) -> str:
    """Return a pre-assembled context string of the most activated memories.

    Args:
        max_tokens: Token budget for context (default 2000)
        topic_hint: Optional topic to bias which memories are loaded

    Returns:
        JSON with context markdown, entity_count, fact_count, token_estimate.
    """
    manager = _get_manager()
    result = await manager.get_context(
        group_id=_group_id, max_tokens=max_tokens, topic_hint=topic_hint
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

    cycle = await engine.run_cycle(
        group_id=_group_id,
        trigger="mcp",
        dry_run=dry_run,
    )

    return json.dumps(
        {
            "cycle_id": cycle.id,
            "status": cycle.status,
            "dry_run": cycle.dry_run,
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

    # Run initialization before starting the server
    asyncio.get_event_loop().run_until_complete(_init())

    transport = os.environ.get("ENGRAM_TRANSPORT", "stdio")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
