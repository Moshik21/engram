# 09 - Lite Mode: Zero-Infrastructure Developer Experience

## Problem Statement

The current Engram architecture requires Docker Compose with FalkorDB + Redis to reach "first memory." This creates friction for:

1. **New users evaluating Engram** -- they want to try it in 60 seconds, not provision containers.
2. **Claude Desktop users** -- non-technical users who just want memory for their Claude, not a DevOps exercise.
3. **Dashboard contributors** -- frontend developers who need a running backend without understanding the full stack.
4. **CI/CD environments** -- testing pipelines should not need container orchestration.

Lite mode eliminates this friction: `pip install engram` is the only prerequisite.

---

## Architecture Overview

```
                         ┌───────────────────────────────────────────┐
                         │           engram.main entrypoint           │
                         │                                           │
                         │   --mode lite  |  --mode full  |  auto   │
                         └──────────────────┬────────────────────────┘
                                            │
                                    ┌───────▼────────┐
                                    │  Mode Resolver  │
                                    │  (auto-detect)  │
                                    └───────┬────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
            ┌───────▼───────┐       ┌───────▼───────┐      ┌───────▼───────┐
            │  GraphStore   │       │ ActivationStore│      │  SearchIndex  │
            │  (Protocol)   │       │  (Protocol)    │      │  (Protocol)   │
            └───────┬───────┘       └───────┬───────┘      └───────┬───────┘
                    │                       │                       │
        ┌───────────┴──────────┐  ┌─────────┴─────────┐  ┌─────────┴─────────┐
        │                      │  │                    │  │                    │
   ┌────▼─────┐  ┌─────────┐  │  │  ┌──────────┐     │  │  ┌─────────────┐  │
   │ SQLite   │  │FalkorDB │  │  │  │ In-memory │     │  │  │ SQLite FTS5 │  │
   │ Graph    │  │ Graph   │  │  │  │ Dict      │     │  │  │             │  │
   │ Adapter  │  │ Client  │  │  │  └──────────┘     │  │  └─────────────┘  │
   └──────────┘  └─────────┘  │  │  ┌──────────┐     │  │  ┌─────────────┐  │
                  LITE  FULL   │  │  │ Redis    │     │  │  │ Vector      │  │
                               │  │  │ State    │     │  │  │ Embeddings  │  │
                               │  │  └──────────┘     │  │  └─────────────┘  │
                               │  │   LITE   FULL     │  │   LITE    FULL    │
                               │  │                    │  │                    │
                               └──┘                    └──┘                    │
                                                                              └──┘
```

### Key Principle: Same MCP Tool Interface

The MCP tools (`remember`, `recall`, `search_entities`, `search_facts`, `forget`, `get_context`, `get_graph_state`) remain identical regardless of mode. Claude never knows or cares whether it is talking to lite or full mode. The abstraction layer sits beneath the Graph Manager.

---

## Abstraction Layer Design

### Protocol Definitions (Python Protocols)

Three protocols define the contracts that both lite and full implementations must satisfy.

```python
# engram/storage/protocols.py

from typing import Protocol, runtime_checkable
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.models.episode import Episode
from engram.models.activation import ActivationState

@runtime_checkable
class GraphStore(Protocol):
    """Persistent storage for entities, relationships, and episodes."""

    async def initialize(self) -> None:
        """Create tables/indexes. Idempotent."""
        ...

    async def create_entity(self, entity: Entity) -> str:
        """Insert entity, return its ID."""
        ...

    async def get_entity(self, entity_id: str) -> Entity | None:
        ...

    async def update_entity(self, entity_id: str, updates: dict) -> None:
        ...

    async def delete_entity(self, entity_id: str, soft: bool = True) -> None:
        ...

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        ...

    async def create_relationship(self, rel: Relationship) -> str:
        ...

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",  # "outgoing" | "incoming" | "both"
        predicate: str | None = None,
    ) -> list[Relationship]:
        ...

    async def invalidate_relationship(self, rel_id: str, valid_to: datetime) -> None:
        """Set valid_to on a relationship (temporal invalidation)."""
        ...

    async def get_neighbors(
        self, entity_id: str, hops: int = 1
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops and the connecting relationships."""
        ...

    async def create_episode(self, episode: Episode) -> str:
        ...

    async def get_episodes(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]:
        ...

    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        """Create MENTIONED_IN edge."""
        ...

    async def get_stats(self, group_id: str | None = None) -> dict:
        """Return counts: entities, relationships, episodes."""
        ...

    async def export_all(self, group_id: str | None = None) -> dict:
        """Export entire graph as JSON for migration."""
        ...

    async def import_all(self, data: dict) -> None:
        """Import graph JSON. Used for lite-to-full migration."""
        ...


@runtime_checkable
class ActivationStore(Protocol):
    """Hot-path activation state storage."""

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        ...

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        ...

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        ...

    async def batch_set(self, states: dict[str, ActivationState]) -> None:
        ...

    async def get_top_activated(
        self, group_id: str | None = None, limit: int = 20
    ) -> list[tuple[str, ActivationState]]:
        ...

    async def decay_sweep(self, decay_fn: callable, threshold: float = 0.01) -> int:
        """Apply decay to all states. Return count of decayed entries."""
        ...

    async def snapshot_to_graph(self, graph_store: GraphStore) -> None:
        """Persist activation state back to graph store for durability."""
        ...


@runtime_checkable
class SearchIndex(Protocol):
    """Text/semantic search over entities and facts."""

    async def index_entity(self, entity: Entity) -> None:
        ...

    async def index_relationship(self, rel: Relationship) -> None:
        ...

    async def index_episode(self, episode: Episode) -> None:
        ...

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Return list of (entity_id, relevance_score) pairs."""
        ...

    async def remove(self, entity_id: str) -> None:
        ...
```

### Implementation Matrix

| Protocol        | Lite Implementation        | Full Implementation           |
|-----------------|----------------------------|-------------------------------|
| `GraphStore`    | `SQLiteGraphStore`         | `FalkorDBGraphStore`          |
| `ActivationStore` | `MemoryActivationStore` | `RedisActivationStore`        |
| `SearchIndex`   | `FTS5SearchIndex`          | `VectorSearchIndex`           |

---

## Lite Mode Implementations

### SQLiteGraphStore

A single SQLite file (`~/.engram/engram.db` by default) stores the entire graph using relational tables that model graph semantics.

```sql
-- engram/storage/sqlite/schema.sql

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    summary         TEXT,
    attributes      TEXT,            -- JSON blob
    group_id        TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,    -- ISO 8601
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT,            -- soft delete
    activation_base REAL NOT NULL DEFAULT 0.5,
    activation_current REAL NOT NULL DEFAULT 0.5,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id);

CREATE TABLE IF NOT EXISTS relationships (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES entities(id),
    target_id       TEXT NOT NULL REFERENCES entities(id),
    predicate       TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    valid_from      TEXT,
    valid_to        TEXT,
    created_at      TEXT NOT NULL,
    source_episode  TEXT,
    group_id        TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate);

CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    source          TEXT,             -- "claude_desktop", "claude_code", etc.
    group_id        TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_entities (
    episode_id      TEXT NOT NULL REFERENCES episodes(id),
    entity_id       TEXT NOT NULL REFERENCES entities(id),
    PRIMARY KEY (episode_id, entity_id)
);

-- FTS5 virtual table for text search
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name,
    summary,
    entity_type,
    content=entities,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content,
    content=episodes,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(rowid, name, summary, entity_type)
    VALUES (new.rowid, new.name, new.summary, new.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, summary, entity_type)
    VALUES ('delete', old.rowid, old.name, old.summary, old.entity_type);
    INSERT INTO entities_fts(rowid, name, summary, entity_type)
    VALUES (new.rowid, new.name, new.summary, new.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;
```

**Graph traversal** (e.g., `get_neighbors` with N hops) uses recursive CTEs:

```sql
-- 2-hop neighbor query
WITH RECURSIVE neighbors(entity_id, depth) AS (
    -- Seed: the starting entity
    SELECT :start_id, 0
    UNION ALL
    -- Recurse: follow relationships in both directions
    SELECT
        CASE WHEN r.source_id = n.entity_id THEN r.target_id ELSE r.source_id END,
        n.depth + 1
    FROM neighbors n
    JOIN relationships r ON (r.source_id = n.entity_id OR r.target_id = n.entity_id)
    WHERE n.depth < :max_hops
      AND r.valid_to IS NULL
      AND r.source_id != r.target_id
)
SELECT DISTINCT e.*, n.depth
FROM neighbors n
JOIN entities e ON e.id = n.entity_id
WHERE e.deleted_at IS NULL AND n.depth > 0;
```

**Performance characteristics:**
- SQLite handles 100K+ entities comfortably with proper indexing.
- WAL mode enabled for concurrent read/write (single-writer, multiple readers).
- Connection pooling via `aiosqlite` with `PRAGMA journal_mode=WAL`.
- Recursive CTEs for 2-hop traversal complete in <10ms for typical personal graphs (<10K nodes).

### MemoryActivationStore

A Python `dict[str, ActivationState]` held in process memory. This is acceptable for lite mode because:

1. Personal graphs are small (hundreds to low thousands of entities).
2. Activation state is derived from graph data -- it can be reconstructed on startup.
3. Periodic snapshots write activation columns back to the `entities` table in SQLite.

```python
# engram/storage/memory/activation.py

class MemoryActivationStore:
    def __init__(self):
        self._states: dict[str, ActivationState] = {}
        self._lock = asyncio.Lock()

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return self._states.get(entity_id)

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        self._states[entity_id] = state

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        return {eid: self._states[eid] for eid in entity_ids if eid in self._states}

    async def batch_set(self, states: dict[str, ActivationState]) -> None:
        self._states.update(states)

    async def get_top_activated(
        self, group_id: str | None = None, limit: int = 20
    ) -> list[tuple[str, ActivationState]]:
        items = sorted(
            self._states.items(),
            key=lambda x: x[1].current_activation,
            reverse=True,
        )
        return items[:limit]

    async def decay_sweep(self, decay_fn, threshold: float = 0.01) -> int:
        count = 0
        for eid, state in self._states.items():
            new_state = decay_fn(state)
            if abs(new_state.current_activation - state.current_activation) > threshold:
                self._states[eid] = new_state
                count += 1
        return count

    async def snapshot_to_graph(self, graph_store) -> None:
        """Persist current activation state to SQLite entity rows."""
        for eid, state in self._states.items():
            await graph_store.update_entity(eid, {
                "activation_base": state.base_activation,
                "activation_current": state.current_activation,
                "access_count": state.access_count,
                "last_accessed": state.last_accessed.isoformat() if state.last_accessed else None,
            })

    async def load_from_graph(self, graph_store) -> None:
        """Reconstruct activation state from SQLite on startup."""
        # Query all entities with activation data
        # Populate self._states
        ...
```

**Startup behavior:** On process start, `MemoryActivationStore.load_from_graph()` reads activation columns from the `entities` table to warm the in-memory state. Snapshots run every 60 seconds (configurable) and on graceful shutdown.

### FTS5SearchIndex

SQLite FTS5 provides full-text search with BM25 ranking as a replacement for vector embeddings in lite mode.

```python
# engram/storage/sqlite/search.py

class FTS5SearchIndex:
    """Full-text search using SQLite FTS5 with BM25 ranking."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Search entities using FTS5 BM25 scoring.
        Returns (entity_id, relevance_score) pairs normalized to 0.0-1.0.
        """
        # FTS5 query with BM25 ranking
        sql = """
            SELECT e.id, rank
            FROM entities_fts fts
            JOIN entities e ON e.rowid = fts.rowid
            WHERE entities_fts MATCH :query
              AND e.deleted_at IS NULL
        """
        params = {"query": self._prepare_query(query)}

        if entity_types:
            placeholders = ",".join(f":type_{i}" for i in range(len(entity_types)))
            sql += f" AND e.entity_type IN ({placeholders})"
            for i, t in enumerate(entity_types):
                params[f"type_{i}"] = t

        if group_id:
            sql += " AND e.group_id = :group_id"
            params["group_id"] = group_id

        sql += " ORDER BY rank LIMIT :limit"
        params["limit"] = limit

        rows = await self._execute(sql, params)
        # Normalize BM25 scores to 0.0-1.0 range
        if not rows:
            return []
        max_score = abs(rows[0][1])  # BM25 returns negative scores
        return [
            (row[0], abs(row[1]) / max_score if max_score > 0 else 0.0)
            for row in rows
        ]

    def _prepare_query(self, query: str) -> str:
        """Convert natural language query to FTS5 query syntax.
        Uses prefix matching and implicit OR for multi-word queries.
        """
        tokens = query.strip().split()
        # Use prefix matching for each token, OR them together
        return " OR ".join(f'"{t}"*' for t in tokens if t)
```

**Tradeoffs vs vector embeddings:**
- FTS5 handles keyword/entity-name searches well (the primary use case for `search_entities`).
- FTS5 misses semantic similarity (e.g., "payment" would not match "Stripe" unless co-occurring).
- Acceptable for lite mode because the activation engine compensates: associatively connected nodes get boosted via spreading activation even without semantic matching.
- Migration to full mode adds vector embeddings on top, improving semantic recall.

---

## Auto-Detection Logic

The mode resolver determines which backend to use at startup. It follows a clear priority chain.

```python
# engram/storage/resolver.py

import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class EngineMode(str, Enum):
    LITE = "lite"
    FULL = "full"
    AUTO = "auto"


async def resolve_mode(requested_mode: str = "auto") -> EngineMode:
    """
    Determine the runtime mode based on explicit config or service availability.

    Priority:
    1. Explicit --mode lite or --mode full (respect user intent, fail if full unavailable)
    2. ENGRAM_MODE env var ("lite" | "full")
    3. Auto-detect: probe Redis and FalkorDB connectivity
    """
    # 1. Explicit mode from CLI or env var
    mode = requested_mode.lower()
    if mode == "lite":
        logger.info("Lite mode selected explicitly")
        return EngineMode.LITE
    if mode == "full":
        logger.info("Full mode selected explicitly -- verifying services...")
        if not await _check_falkordb() or not await _check_redis():
            raise RuntimeError(
                "Full mode requested but FalkorDB and/or Redis are not available. "
                "Start them with `docker compose up -d falkordb redis` or use --mode lite."
            )
        return EngineMode.FULL

    # 2. Check env var
    env_mode = os.environ.get("ENGRAM_MODE", "").lower()
    if env_mode in ("lite", "full"):
        return await resolve_mode(env_mode)

    # 3. Auto-detect
    logger.info("Auto-detecting mode...")
    falkordb_ok = await _check_falkordb()
    redis_ok = await _check_redis()

    if falkordb_ok and redis_ok:
        logger.info("FalkorDB and Redis detected -- using full mode")
        return EngineMode.FULL
    elif falkordb_ok or redis_ok:
        logger.warning(
            "Partial infrastructure detected (FalkorDB=%s, Redis=%s). "
            "Both are needed for full mode. Falling back to lite mode.",
            falkordb_ok, redis_ok,
        )
        return EngineMode.LITE
    else:
        logger.info("No external services detected -- using lite mode")
        return EngineMode.LITE


async def _check_redis() -> bool:
    """Probe Redis connectivity with a 2-second timeout."""
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(
            os.environ.get("ENGRAM_REDIS_URL", "redis://localhost:6379"),
            socket_connect_timeout=2,
        )
        await r.ping()
        await r.aclose()
        return True
    except Exception:
        return False


async def _check_falkordb() -> bool:
    """Probe FalkorDB connectivity with a 2-second timeout."""
    try:
        from falkordb import FalkorDB
        db = FalkorDB(
            host=os.environ.get("ENGRAM_FALKORDB_HOST", "localhost"),
            port=int(os.environ.get("ENGRAM_FALKORDB_PORT", "6379")),
            socket_timeout=2,
        )
        db.list_graphs()
        return True
    except Exception:
        return False
```

### Startup Sequence

```
python -m engram.main [--mode lite|full|auto]
        │
        ▼
  ┌─────────────┐
  │ Load config  │  <-- config.yaml / env vars / CLI args
  └──────┬──────┘
         ▼
  ┌──────────────┐
  │ resolve_mode │  <-- auto-detect or explicit
  └──────┬───────┘
         ▼
  ┌──────────────────┐
  │ Create backends  │
  │ based on mode:   │
  │                  │
  │ lite:            │
  │   SQLiteGraphStore(db_path)
  │   MemoryActivationStore()
  │   FTS5SearchIndex(db_path)
  │                  │
  │ full:            │
  │   FalkorDBGraphStore(url)
  │   RedisActivationStore(url)
  │   VectorSearchIndex(config)
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │ Initialize stores│  <-- create tables, connect, warm caches
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │ Build services   │  <-- GraphManager, ActivationEngine, Retrieval
  │ (inject backends)│      all receive Protocol instances
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │ Start servers    │  <-- MCP server + FastAPI (dashboard API)
  └──────────────────┘
```

### Dependency Injection

```python
# engram/storage/factory.py

from engram.storage.protocols import GraphStore, ActivationStore, SearchIndex
from engram.storage.resolver import EngineMode


def create_stores(mode: EngineMode, config: dict) -> tuple[GraphStore, ActivationStore, SearchIndex]:
    """Factory that builds the correct backend triple for the resolved mode."""
    if mode == EngineMode.LITE:
        from engram.storage.sqlite.graph import SQLiteGraphStore
        from engram.storage.memory.activation import MemoryActivationStore
        from engram.storage.sqlite.search import FTS5SearchIndex

        db_path = config.get("sqlite_path", "~/.engram/engram.db")
        db_path = os.path.expanduser(db_path)

        return (
            SQLiteGraphStore(db_path),
            MemoryActivationStore(),
            FTS5SearchIndex(db_path),
        )
    else:
        from engram.storage.falkordb.graph import FalkorDBGraphStore
        from engram.storage.redis.activation import RedisActivationStore
        from engram.storage.vector.search import VectorSearchIndex

        return (
            FalkorDBGraphStore(config["falkordb_url"]),
            RedisActivationStore(config["redis_url"]),
            VectorSearchIndex(config["embedding_config"]),
        )
```

---

## Configuration

### Minimal Config (Lite Mode)

A single environment variable gets you started:

```bash
export ENGRAM_API_KEY=sk-ant-...   # Claude API key for entity extraction
python -m engram.main              # auto-detects lite mode, starts MCP server
```

### Full Config Reference

```yaml
# config.yaml (optional -- all values have sensible defaults)

mode: auto                          # "lite" | "full" | "auto"

api_key: ${ENGRAM_API_KEY}          # Claude API key (required)
extraction_model: claude-haiku-4-5  # Model for entity extraction

# Lite mode settings
sqlite:
  path: ~/.engram/engram.db         # Database location
  wal_mode: true                    # WAL for concurrent access

# Full mode settings (only used in full mode)
falkordb:
  host: localhost
  port: 6379
  graph_name: engram

redis:
  url: redis://localhost:6379/0

embeddings:
  provider: voyage                   # "voyage" | "openai" | "local"
  model: voyage-3-lite
  dimensions: 512

# Server settings
server:
  host: 0.0.0.0
  port: 8420
  transport: sse                     # "sse" | "stdio"

# Activation engine (same for both modes)
activation:
  decay_rate: 0.1
  spread_factor: 0.3
  spread_hops: 2
  snapshot_interval_seconds: 60

# Dashboard proxy (for local frontend dev)
dashboard:
  api_url: http://localhost:8420
  ws_url: ws://localhost:8420/ws

# Multi-tenant
default_group_id: default
```

### Environment Variable Overrides

Every config key can be overridden via env var with `ENGRAM_` prefix:

| Env Var | Config Key | Default |
|---------|-----------|---------|
| `ENGRAM_API_KEY` | `api_key` | (required) |
| `ENGRAM_MODE` | `mode` | `auto` |
| `ENGRAM_SQLITE_PATH` | `sqlite.path` | `~/.engram/engram.db` |
| `ENGRAM_REDIS_URL` | `redis.url` | `redis://localhost:6379/0` |
| `ENGRAM_FALKORDB_HOST` | `falkordb.host` | `localhost` |
| `ENGRAM_FALKORDB_PORT` | `falkordb.port` | `6379` |
| `ENGRAM_SERVER_PORT` | `server.port` | `8420` |
| `ENGRAM_SERVER_TRANSPORT` | `server.transport` | `sse` |

---

## Quickstart Flow (3 Steps)

### For Users

```bash
# Step 1: Install
pip install engram

# Step 2: Set your Claude API key
export ENGRAM_API_KEY=sk-ant-...

# Step 3: Start
python -m engram.main
```

Output:
```
Engram v0.1.0
Mode: lite (SQLite + in-memory activation)
Database: ~/.engram/engram.db
MCP server: http://localhost:8420/mcp (SSE)
Dashboard: http://localhost:8420

Ready. Add to Claude Desktop config:
{
  "mcpServers": {
    "engram": {
      "url": "http://localhost:8420/mcp"
    }
  }
}
```

### For Claude Code Users

```bash
pip install engram
export ENGRAM_API_KEY=sk-ant-...
python -m engram.main --transport stdio
```

Or in Claude Code's MCP config:
```json
{
  "mcpServers": {
    "engram": {
      "command": "python",
      "args": ["-m", "engram.main", "--transport", "stdio"],
      "env": {
        "ENGRAM_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

---

## Migration: Lite to Full

When users outgrow lite mode (large graphs, need vector search, want the dashboard with live WebSocket updates at scale), they migrate to full mode.

### Export/Import Flow

```
┌──────────────┐     export_all()     ┌──────────────┐     import_all()     ┌──────────────┐
│  Lite Mode   │  ───────────────►    │  JSON File   │  ───────────────►    │  Full Mode   │
│  (SQLite)    │                      │  engram.json │                      │  (FalkorDB)  │
└──────────────┘                      └──────────────┘                      └──────────────┘
```

### CLI Commands

```bash
# Export from lite mode (while lite server is running or using DB directly)
python -m engram.migrate export --output engram-export.json

# Start full stack
docker compose up -d

# Import into full mode
python -m engram.migrate import --input engram-export.json --target full

# Verify
python -m engram.migrate verify --source engram-export.json --target full
```

### Export Format

```json
{
  "version": "1.0",
  "exported_at": "2026-02-27T12:00:00Z",
  "source_mode": "lite",
  "entities": [
    {
      "id": "ent_abc123",
      "name": "ReadyCheck",
      "entity_type": "Project",
      "summary": "Meeting prep SaaS with Stripe integration",
      "attributes": {"status": "in_progress"},
      "group_id": "default",
      "created_at": "2026-02-20T10:00:00Z",
      "updated_at": "2026-02-27T09:00:00Z",
      "activation_base": 0.8,
      "access_count": 15,
      "last_accessed": "2026-02-27T09:00:00Z"
    }
  ],
  "relationships": [
    {
      "id": "rel_def456",
      "source_id": "ent_abc123",
      "target_id": "ent_ghi789",
      "predicate": "INTEGRATES",
      "weight": 1.0,
      "valid_from": "2026-02-20T10:00:00Z",
      "valid_to": null,
      "source_episode": "ep_xyz"
    }
  ],
  "episodes": [
    {
      "id": "ep_xyz",
      "content": "Working on ReadyCheck Stripe integration...",
      "source": "claude_desktop",
      "group_id": "default",
      "created_at": "2026-02-20T10:00:00Z"
    }
  ],
  "episode_entities": [
    {"episode_id": "ep_xyz", "entity_id": "ent_abc123"}
  ]
}
```

### What Migrates

| Data | Migrated | Notes |
|------|----------|-------|
| Entities + attributes | Yes | Direct mapping |
| Relationships + temporal data | Yes | Cypher CREATE from JSON |
| Episodes | Yes | Direct mapping |
| Episode-entity links | Yes | MENTIONED_IN edges |
| Activation base/access_count | Yes | Copied to node properties |
| Current activation state | No | Recomputed from base + recency in full mode |
| FTS5 index | No | Rebuilt as vector embeddings in full mode |

---

## Dashboard Local Development

For frontend contributors who do not want to run the full Docker stack.

### Prerequisites

```bash
# Backend (lite mode)
pip install engram
export ENGRAM_API_KEY=sk-ant-...
python -m engram.main --mode lite

# In another terminal -- frontend
cd dashboard/
cp .env.example .env.local
npm install
npm run dev
```

### `.env.example` (dashboard)

```bash
# Dashboard local dev config
VITE_API_URL=http://localhost:8420
VITE_WS_URL=ws://localhost:8420/ws
```

### Seed Data for Development

```bash
# Populate the lite-mode database with demo data for UI development
python -m engram.seed_demo --episodes 30 --entities 50

# Or use a fixture file
python -m engram.seed_demo --from-file fixtures/demo_graph.json
```

The seed script creates realistic entities (people, projects, technologies, concepts), relationships between them, and pre-computed activation states so the dashboard has interesting data to render immediately.

### Vite Proxy Config (Optional)

If CORS is an issue during local dev, the Vite config can proxy API calls:

```typescript
// dashboard/vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8420',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8420',
        ws: true,
      },
    },
  },
});
```

---

## Lite Mode Limitations

Transparent to MCP clients but important for documentation:

| Capability | Lite Mode | Full Mode |
|-----------|-----------|-----------|
| Graph storage | SQLite (single file) | FalkorDB (Redis module) |
| Graph queries | Recursive CTEs | Native Cypher traversal |
| Text search | FTS5 (keyword + BM25) | Vector embeddings (semantic) |
| Activation state | In-memory dict | Redis (sub-ms, persistent) |
| Concurrent writers | 1 (SQLite WAL) | Many (FalkorDB + Redis) |
| Activation durability | Periodic snapshot to SQLite | Redis persistence + snapshots |
| Dashboard WebSocket | Supported (polling fallback) | Supported (native push) |
| Max practical graph size | ~50K entities | ~500K+ entities |
| Semantic similarity | No (keyword matching only) | Yes (vector cosine similarity) |
| Process crash recovery | Lose unsnapshot'd activation | Redis persists activation |

### Retrieval Quality Impact

In lite mode the retrieval scoring formula adjusts weights to compensate for keyword-only search:

```
# Full mode (semantic search available)
score = semantic_similarity * 0.4
      + current_activation * 0.3
      + recency_score * 0.2
      + frequency_score * 0.1

# Lite mode (FTS5 keyword search)
score = fts5_bm25_score * 0.25
      + current_activation * 0.35
      + recency_score * 0.25
      + frequency_score * 0.15
```

The activation engine carries more weight in lite mode, partially compensating for the lack of semantic matching. Spreading activation means that conceptually related nodes (connected in the graph) still get boosted even when the keyword search does not directly match them.

---

## File Layout

New files for the lite mode abstraction layer:

```
server/engram/
├── storage/
│   ├── __init__.py
│   ├── protocols.py              # GraphStore, ActivationStore, SearchIndex protocols
│   ├── resolver.py               # Mode auto-detection logic
│   ├── factory.py                # Backend factory (create_stores)
│   ├── sqlite/
│   │   ├── __init__.py
│   │   ├── graph.py              # SQLiteGraphStore implementation
│   │   ├── search.py             # FTS5SearchIndex implementation
│   │   └── schema.sql            # DDL for tables + FTS5 + triggers
│   ├── memory/
│   │   ├── __init__.py
│   │   └── activation.py         # MemoryActivationStore implementation
│   ├── falkordb/
│   │   ├── __init__.py
│   │   └── graph.py              # FalkorDBGraphStore implementation
│   ├── redis/
│   │   ├── __init__.py
│   │   └── activation.py         # RedisActivationStore implementation
│   └── vector/
│       ├── __init__.py
│       └── search.py             # VectorSearchIndex implementation
├── migrate/
│   ├── __init__.py
│   ├── export.py                 # Export graph to JSON
│   ├── import_.py                # Import graph from JSON
│   └── verify.py                 # Verify migration integrity
└── seed_demo.py                  # Demo data generator
```

---

## Testing Strategy

### Unit Tests

Each protocol implementation gets its own test suite using the same test cases:

```python
# tests/storage/conftest.py
import pytest

@pytest.fixture(params=["sqlite", "falkordb"])
def graph_store(request, tmp_path):
    if request.param == "sqlite":
        from engram.storage.sqlite.graph import SQLiteGraphStore
        return SQLiteGraphStore(str(tmp_path / "test.db"))
    else:
        pytest.importorskip("falkordb")
        from engram.storage.falkordb.graph import FalkorDBGraphStore
        return FalkorDBGraphStore("localhost:6379")
```

This ensures both implementations pass the same behavioral tests.

### Integration Tests

```bash
# Run lite-mode tests (no Docker needed)
pytest tests/ -m "not requires_docker"

# Run full-mode tests (needs Docker stack)
docker compose up -d falkordb redis
pytest tests/ -m "requires_docker"

# Run all tests
docker compose up -d falkordb redis
pytest tests/
```

---

## Coordination Notes

### For Config Agent (Task #1)

The mode detection system needs these config fields in the Pydantic schema:
- `mode: Literal["lite", "full", "auto"]` with default `"auto"`
- `sqlite.path: str` with default `"~/.engram/engram.db"`
- Env var mapping: `ENGRAM_MODE` -> `mode`, `ENGRAM_SQLITE_PATH` -> `sqlite.path`
- The `resolve_mode()` function should use the validated config, not raw env vars directly.

### For Embedding Agent (Task #4)

Lite mode replaces vector embeddings with SQLite FTS5:
- The `SearchIndex` protocol is the abstraction boundary. `VectorSearchIndex` (full mode) and `FTS5SearchIndex` (lite mode) both implement it.
- The `search()` method returns `list[tuple[str, float]]` -- entity ID and a normalized 0-1 relevance score -- regardless of whether the score comes from BM25 or cosine similarity.
- Lite mode adjusts retrieval scoring weights to compensate (activation gets +5%, recency gets +5%, text match gets -15% relative to semantic similarity).
- The embedding agent should design `VectorSearchIndex` to conform to the same `SearchIndex` protocol.
