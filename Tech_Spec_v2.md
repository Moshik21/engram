# ENGRAM: Activation-Based Memory Layer for AI Agents

*Architecture Spec & Build Plan*
*v0.2 - February 2026*
*Refined from 10-perspective technical review*

---

## Project Summary

Engram is an open-source memory layer for AI agents that combines temporal knowledge graphs with spreading activation retrieval and a visual dashboard. Unlike existing solutions (Mem0, Zep/Graphiti, Letta) that rely on static similarity search, Engram uses activation dynamics -- recency, frequency, and associative proximity -- to surface contextually relevant memories the way human recall works.

The dashboard is the product's face. It makes memory visible, explorable, and shareable. The activation engine is the technical moat. The MCP integration makes it immediately useful for Claude Desktop and Claude Code users.

---

## Competitive Positioning

| Product | Memory Type | Visualization | Activation Dynamics | MCP Support | Pricing |
|---------|-----------|---------------|-------------------|-------------|---------|
| Mem0 | Vector + KV + Graph (paid) | API dashboard only | No | Yes | Free / $19 / $249/mo |
| Zep/Graphiti | Temporal KG (Neo4j) | Neo4j browser (raw) | No | Yes (experimental) | Open source + enterprise |
| Letta (MemGPT) | Hierarchical blocks | None | No | Limited | Open source |
| Mastra | Observational | None | No | No (LangChain plugin) | Open source |
| MCP Memory Server | Flat KG (SQLite) | memory-visualizer (basic) | No | Yes (native) | Free |
| **Engram** | **Temporal KG + Activation** | **Purpose-built dashboard** | **Yes (core feature)** | **Yes (primary)** | **Open source + hosted** |

### Where Engram wins:

1. **Only product with activation-based retrieval.** Everyone else does cosine similarity or keyword search. Engram surfaces memories based on how "activated" they are through ACT-R power-law decay, frequency reinforcement, and bounded spreading activation from connected nodes.

2. **Only product with a purpose-built visual dashboard.** Mem0 has an API metrics dashboard. Zep dumps into Neo4j browser. Nobody lets you watch your AI's understanding of you grow in real time with a beautiful, interactive graph.

3. **Claude-first.** Every competitor defaults to OpenAI. Engram is built for the Claude ecosystem from day one, with Claude doing entity extraction and graph construction natively.

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    MCP CLIENTS                           │
│  Claude Desktop  |  Claude Code  |  Cursor  |  VS Code  │
└──────────┬───────────────────────────────────┬───────────┘
           │      MCP Protocol (stdio/SSE)     │
           ▼                                   ▼
┌──────────────────────────────────────────────────────────┐
│                  ENGRAM MCP SERVER                        │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │  Episode     │  │  Memory      │  │  Activation     │ │
│  │  Ingestion   │  │  Retrieval   │  │  Engine         │ │
│  │  (async)     │  │              │  │  (ACT-R)        │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘ │
│         │                │                    │          │
│  ┌──────▼────────────────▼────────────────────▼────────┐ │
│  │              GRAPH MANAGER                           │ │
│  │  Entity Extraction | Relationship Resolution        │ │
│  │  Temporal Tracking | Conflict Detection              │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌──────────────────────▼──────────────────────────────┐ │
│  │           ABSTRACTION LAYER (Protocols)              │ │
│  │  GraphStore | ActivationStore | SearchIndex          │ │
│  │  (FalkorDB or SQLite) (Redis or Dict) (HNSW or FTS) │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐ │
│  │  Auth        │  │  Rate       │  │  Tenant          │ │
│  │  Middleware   │  │  Limiter    │  │  Context         │ │
│  └─────────────┘  └─────────────┘  └──────────────────┘ │
└─────────────────────────┼────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │  FalkorDB  │ │  Redis     │ │  Voyage AI │
    │  (Graph +  │ │  (Activate │ │  (Embed)   │
    │   Vectors) │ │  + Queue)  │ │            │
    └────────────┘ └────────────┘ └────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼                             ▼
    ┌────────────────┐          ┌─────────────────┐
    │  REST API      │          │  WebSocket API   │
    │  (FastAPI)     │          │  (live updates)  │
    └───────┬────────┘          └────────┬────────┘
            │                            │
            ▼                            ▼
    ┌─────────────────────────────────────────────┐
    │            ENGRAM DASHBOARD                   │
    │  React + Zustand + react-force-graph         │
    │  Real-time graph visualization               │
    │  Activation heatmaps                         │
    │  Timeline view                               │
    │  Memory search + explore                     │
    └─────────────────────────────────────────────┘
```

### Dual-Mode Architecture

Engram runs in two modes, selectable via config or auto-detected:

| Component | Full Mode | Lite Mode |
|-----------|-----------|-----------|
| Graph Store | FalkorDB (Redis module) | SQLite with graph adapter |
| Activation State | Redis (sub-ms reads) | In-memory dict |
| Search Index | Redis Search HNSW (vector) | SQLite FTS5 (keyword) |
| Embedding | Voyage AI API | Disabled (FTS5 fallback) |
| Infrastructure | Docker Compose | `pip install engram` |
| Target User | Production, SaaS | Eval, dev, CI, casual use |

Both modes expose identical MCP tools. Claude never knows which mode is running. The abstraction layer (`GraphStore`, `ActivationStore`, `SearchIndex` protocols) makes the backends interchangeable.

**Lite mode quickstart:**
```bash
pip install engram
export ANTHROPIC_API_KEY=sk-ant-...
python -m engram.mcp.server  # stdio MCP server, ready in <2 seconds
```

---

## Core Components

### 1. Episode Ingestion Pipeline (Async)

Every conversation turn becomes an "episode" that flows through async extraction. The `remember` MCP tool returns immediately (~10ms) with an episode ID. All heavy processing happens in a background worker.

```
MCP Client         MCP Server          Redis Stream       Worker Pool
    |                   |                   |                   |
    |  remember(text)   |                   |                   |
    |------------------>|                   |                   |
    |                   |  Persist raw      |                   |
    |                   |  episode (10ms)   |                   |
    |                   |  XADD to queue ---|                   |
    |  {episode_id,     |                   |                   |
    |   status:"queued"}|                   |                   |
    |<------------------|                   |                   |
    |                   |                   |  XREADGROUP       |
    |                   |                   |------------------>|
    |                   |                   |                   |
    |                   |                   |    Claude API     |
    |                   |                   |    extraction     |
    |                   |                   |    (1-3s)         |
    |                   |                   |                   |
    |                   |                   |    Entity dedup   |
    |                   |                   |    Graph writes   |
    |                   |                   |    Embed vectors  |
    |                   |                   |    Update activ.  |
    |                   |                   |                   |
    |                   |                   |    WebSocket      |
    |                   |                   |    notify         |
```

**Episode lifecycle states:**
```
queued → extracting → resolving → writing → embedding → completed
                                                    └→ dead_letter (on failure)
```

**Error handling:**
- Claude API failures: retry with exponential backoff (3 attempts, 1s/4s/16s)
- Permanent failures: move to dead-letter queue, surface in dashboard
- Embedding failures: skip embedding, mark as `embedding_pending`, retry in background
- Partial extraction: store what succeeded, log what failed

Entity extraction uses Claude with structured output. System prompt forces consistent entity typing:

**Entity Types:**
- Person (name, role, relationship to user)
- Organization (name, type, user's connection)
- Project (name, status, technologies, goals)
- Concept (name, domain, related concepts)
- Preference (category, value, strength, context)
- Location (name, significance to user)
- Event (name, date, participants, outcome)
- Tool/Technology (name, proficiency, use context)

### 2. Activation Engine (ACT-R Power-Law Model)

This is what makes Engram different. Every node in the graph has an activation level computed lazily from its access history using the ACT-R base-level learning equation.

**Design principles:**
- **ACT-R grounding.** Uses the well-studied power-law decay model backed by decades of cognitive science (Anderson & Schooler, 1991).
- **Orthogonal signals.** Three independent scoring dimensions: semantic similarity, activation, edge proximity.
- **Lazy evaluation.** No background decay sweeps. Activation computed on read from stored timestamps.
- **Bounded spreading.** Firing threshold, degree normalization, visited set, energy budget.

**Activation State per Node:**

```python
@dataclass
class ActivationState:
    node_id: str
    access_history: list[float]   # Unix timestamps, capped at MAX_HISTORY_SIZE (200)
    spreading_bonus: float = 0.0  # Transient, reset per retrieval cycle
    last_accessed: float = 0.0    # For O(1) staleness checks
    access_count: int = 0         # Lifetime count, for dashboard display
```

**Redis storage layout:**
```
engram:{group_id}:activation:{node_id} -> Hash {
    access_history: JSON array of float timestamps,
    access_count: int,
    last_accessed: float
}
```

#### Base-Level Activation (Power-Law Decay)

```
B_i(t) = ln( Σ (t - t_j)^{-d} )    for j = 1..n
```

Where:
- `t_j` = timestamp of the j-th access
- `d` = decay exponent (default **0.5**, the standard ACT-R value)
- `n` = number of recorded accesses

Normalized to [0, 1] via sigmoid:

```
activation_i(t) = sigmoid( (B_i(t) - B_mid) / B_scale )
```

**Calibrated so that:**
- Node accessed once 10 seconds ago → activation ~0.85
- Node accessed once 1 hour ago → activation ~0.50
- Node accessed once 7 days ago → activation ~0.10
- Node accessed 10 times over the past week → activation ~0.75

**Why power-law, not exponential?** Exponential decay forgets too aggressively -- anything older than a few hours drops to near-zero. Power-law produces long-tail behavior matching human memory: heavily-accessed items retain activation for weeks, while one-off mentions fade within hours.

**Why not the original hyperbolic formula?** The original `base * 1/(1 + d*log(t+1))` decays so slowly that a node accessed once a month ago retains ~70% activation. Nothing ever becomes dormant. The ACT-R model produces proper decay curves validated across decades of cognitive research.

#### Frequency Reinforcement

Frequency is already encoded in the base-level equation -- each access adds a term to the sum, raising B_i. No separate reinforcement step is needed.

Access events that record to history:
- Node mentioned in ingested episode → Yes
- Node returned in retrieval results → Yes
- Node receives spreading activation → No (prevents phantom reinforcement)
- User views node in dashboard → Yes

#### Spreading Activation (Bounded)

When retrieval identifies seed nodes via semantic search, activation spreads outward through the graph with these safeguards:

| Safeguard | Parameter | Default | Purpose |
|-----------|-----------|---------|---------|
| Firing threshold | `spread_firing_threshold` | 0.05 | Prevents negligible activations from propagating |
| Degree normalization | `sqrt(out_degree)` | n/a | Hub nodes don't dominate; energy divided by sqrt(degree) |
| Visited set | `visited: set` | n/a | Each node enqueued at most once; prevents cycles |
| Energy budget | `spread_energy_budget` | 5.0 | Total energy cap per retrieval; bounds worst-case cost |
| Max hops | `spread_max_hops` | 2 | Hard depth limit |
| Per-hop decay | `spread_decay_per_hop` | 0.5 | Each hop halves the energy |
| Temporal filter | `valid_to` on edges | n/a | Expired relationships excluded from traversal |

**Why sqrt(out_degree)?** Linear normalization (1/degree) makes hubs useless for spreading. No normalization makes hubs dominate everything. sqrt is the standard middle ground from GNN message-passing (GraphSAGE) -- hubs spread more than leaves, but not proportionally.

#### Retrieval Scoring (3 Orthogonal Signals)

```
score_i = 0.50 × semantic_similarity_i
        + 0.35 × activation_i
        + 0.15 × edge_proximity_i
```

| Signal | Weight | Range | What it captures |
|--------|--------|-------|-----------------|
| `semantic_similarity` | 0.50 | [0, 1] | Content relevance to query (cosine similarity) |
| `activation` | 0.35 | [0, 1] | Recency + frequency + associative priming (ACT-R + spreading) |
| `edge_proximity` | 0.15 | [0, 1] | Structural closeness to seed nodes (0.5^hops) |

**Why 3 signals instead of the original 4?** The original scorer had `current_activation` (0.3) plus a separate `recency_score` (0.2) and `frequency_score` (0.1). This double-counts recency and frequency, which are already encoded in the ACT-R activation formula. Collapsing to 3 orthogonal signals eliminates the bias toward recent items that would drown out semantically strong but older results.

#### Retrieval Flow

```
Query arrives
    │
    ▼
Embed query → vector search → top-K candidates (semantic_similarity scores)
    │
    ▼
Compute activation for each candidate (lazy, from access_history)
    │
    ▼
Identify seeds = candidates with semantic_similarity ≥ 0.3
    │
    ▼
Spread activation from seeds (2 hops, bounded)
    │
    ▼
Composite score = 0.50 × semantic + 0.35 × activation + 0.15 × edge_proximity
    │
    ▼
Return top-N by score
    │
    ▼
Record access for returned nodes
```

**Latency budget:** Target <200ms p99 for `recall`. The vector search step (~35ms) and spreading activation (~5ms with budget cap) are the main costs. Total retrieval: 40-80ms typical.

### 3. Embedding Strategy

**The missing piece from v1.** The architecture now has a concrete vector storage backend.

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default model | Voyage AI `voyage-3-lite` (512d) | Best cost/quality tradeoff; activation engine re-ranks results |
| Vector storage | Redis Search HNSW | Already in stack, sub-ms KNN, no new infrastructure |
| What gets embedded | Entity summaries + episode content | Entities for recall, episodes for fact search |
| Timing | Eager, batched during async ingestion | Retrieval must be fast; no lazy embedding on query path |
| Tenant isolation | Single global index, TAG-filtered on group_id | Simpler ops, same performance at personal scale |
| Provider abstraction | `EmbeddingProvider` ABC | Swap Voyage/OpenAI/local via config |
| Lite mode | SQLite FTS5 keyword search | Zero external API dependency |

**HNSW index parameters:**
- Dimensions: 512 (voyage-3-lite)
- Distance metric: COSINE
- m: 16, ef_construction: 200, ef_runtime: 50
- Estimated memory: ~200MB at 100K vectors

**Hybrid search:** Optional BM25 + semantic via reciprocal rank fusion for queries containing specific entity names.

### 4. Graph Store (FalkorDB)

FalkorDB over Neo4j for several reasons:
- Lighter weight (runs as Redis module, single container)
- Faster for personal-scale graphs (<100K nodes)
- Free, no license complications

**Schema:**

```cypher
// Nodes
(:Entity {
    id: string,
    name: string,
    entity_type: string,
    summary: string,
    created_at: datetime,
    updated_at: datetime,
    access_count: int,
    last_accessed: datetime,
    group_id: string
})

(:Episode {
    id: string,
    content: string,          // encrypted at rest
    source: string,
    status: string,           // queued|extracting|completed|dead_letter
    created_at: datetime,
    group_id: string,
    session_id: string
})

// Edges
[:RELATES_TO {
    predicate: string,
    weight: float,
    valid_from: datetime,
    valid_to: datetime,       // null = still valid
    created_at: datetime,
    source_episode: string
}]

[:MENTIONED_IN {
    created_at: datetime,
    extraction_confidence: float
}]
```

**Indexes (created at startup):**

```cypher
CREATE INDEX FOR (e:Entity) ON (e.group_id)
CREATE INDEX FOR (e:Entity) ON (e.name)
CREATE INDEX FOR (e:Entity) ON (e.entity_type)
CREATE INDEX FOR (e:Entity) ON (e.created_at)
CREATE INDEX FOR (e:Entity) ON (e.updated_at)
CREATE INDEX FOR (e:Entity) ON (e.last_accessed)
CREATE INDEX FOR (ep:Episode) ON (ep.group_id)
CREATE INDEX FOR (ep:Episode) ON (ep.created_at)
CREATE INDEX FOR (ep:Episode) ON (ep.source)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.predicate)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_from)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_to)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.weight)
```

### 5. Security & Multi-Tenancy

**Axiom: an unauthenticated or cross-tenant memory leak is a ship-stopping bug.**

#### Authentication

| Deployment | Auth Method | Token Lifetime |
|-----------|-------------|---------------|
| Self-hosted | Bearer token (from config/env) | Long-lived |
| SaaS: MCP clients | API key (`X-API-Key` header) | Long-lived, rotatable |
| SaaS: Dashboard | JWT (`engram_session` cookie) | 15 min access, 7 day refresh |

Auth is disabled by default in local dev (`ENGRAM_AUTH_ENABLED=false`) but the `TenantContext` middleware still runs, injecting a default group_id. This means tenant isolation logic is always exercised.

#### Mandatory TenantContext

Every request (except `/health`) passes through `TenantContextMiddleware`, which resolves credentials into an immutable `TenantContext`:

```python
class TenantContext(BaseModel):
    group_id: str
    user_id: str | None = None
    role: str = "owner"
    auth_method: str   # "bearer" | "api_key" | "jwt" | "none"

    class Config:
        frozen = True
```

**Key guarantee:** The `GraphStore`, `ActivationStore`, and `SearchIndex` all require `group_id` as a parameter. There is no method that omits it. A developer who tries to query across tenants must bypass the entire class.

#### Middleware Stack Order

```
Request → SecurityHeaders → CORS → RateLimit → TenantContext → Handler
```

#### PII & Encryption

- Episode content and entity summaries encrypted at rest with AES-256-GCM
- Per-tenant encryption keys derived from master key via HKDF
- PII detection flag during Claude extraction (`pii_detected`, `pii_categories`)
- GDPR Article 17 hard-delete cascade across graph, Redis, vectors, and auth records
- Audit logging via structlog (no PII in log entries)

#### Secrets Management

```
Priority (highest to lowest):
1. Environment variables (ENGRAM_AUTH_SECRET, ENGRAM_MASTER_KEY, etc.)
2. .env file (gitignored)
3. config.yaml (non-secret values only)
```

Config loader refuses to start if secrets are hardcoded in YAML. Docker secrets `_FILE` suffix supported.

### 6. MCP Server

Native stdio transport via `mcp.server.fastmcp` -- no `mcp-remote` bridge required. HTTP/SSE secondary for hosted/multi-user deployments.

#### Tools

| Tool | Description | Async | Side Effects |
|------|-------------|-------|-------------|
| `remember` | Ingest an episode. Returns immediately with episode_id; extraction runs async. | Yes | Creates episode, triggers pipeline |
| `recall` | Activation-aware retrieval. Returns memories with composite scores and score breakdowns. | No | Updates activation state |
| `search_entities` | Find entities by name or type. Fuzzy matching. Does not trigger spreading activation. | No | None (read-only) |
| `search_facts` | Find temporal facts/relationships. Supports include_expired flag. | No | None (read-only) |
| `forget` | Soft-delete: sets valid_to to now, decays activation to zero. | No | Invalidates data |
| `get_context` | Pre-assembled context string of most activated memories. Token-budgeted. | No | Updates activation state |
| `get_graph_state` | Graph statistics + top-activated nodes for dashboard. | No | None (read-only) |

All tools have full JSON Schema definitions for input parameters and response shapes. See `refined/06_mcp_protocol.md` for complete schemas.

#### MCP Resources & Prompts

Beyond tools, Engram uses MCP's `resources` and `prompts` primitives:

**Resources:**
- `engram://graph/stats` -- graph statistics (entity count, relationship count, etc.)
- `engram://entity/{id}` -- entity detail with activation state
- `engram://entity/{id}/neighbors` -- 1-hop neighborhood

**Prompts:**
- `engram-system` -- system prompt template instructing Claude to use `remember` after meaningful exchanges and `get_context` at conversation start

#### Session Scoping

`group_id` is resolved per-transport:
- **stdio (local):** `ENGRAM_GROUP_ID` env var
- **HTTP/SSE (hosted):** JWT `group_id` claim via TenantContextMiddleware

Each MCP connection generates a `session_id` (UUID v4) attached to every episode for conversation tracking.

#### Transport Configuration

**Claude Desktop (local stdio):**
```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
      "env": {
        "ENGRAM_GROUP_ID": "personal",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

### 7. Dashboard (React App)

**Tech Stack:** React 18 + TypeScript + Zustand + react-force-graph-3d + Tailwind CSS + Recharts + Vite

**State Management:** Zustand with typed slices for graph data, selected node, time position, activation overlay toggle, and WebSocket event buffer. Components subscribe to specific slices to avoid unnecessary re-renders.

#### Views

**a) Graph Explorer (Main View)**
- Interactive force-directed graph (react-force-graph-3d, WebGL)
- Nodes sized by activation level, colored by entity type
- Edge thickness by relationship weight
- Hover: entity summary, activation score, last accessed
- Click: expand node, show connected entities and facts
- Activation heatmap overlay toggle
- **Subgraph loading:** `GET /api/graph?center={id}&depth=2&maxNodes=200` -- neighborhood-based, not full graph. Progressive expand-on-click.
- 2D fallback mode for accessibility and low-end devices
- Real-time updates via WebSocket

**b) Timeline View**
- Horizontal timeline showing entity creation/modification
- Temporal validity ranges for facts
- Episode markers with source labels
- **Time scrubber:** server-side query `GET /api/graph/at?timestamp={iso}` evaluates bi-temporal validity
- Filter graph to point-in-time state

**c) Memory Feed**
- Reverse-chronological list of extracted facts and entities
- Source conversation labels
- Quick actions: correct, delete, merge entities
- Filter by entity type, date range, source

**d) Activation Monitor**
- Live activation levels bar chart (top 20 nodes)
- Spreading activation animation during queries
- Decay curve visualization
- "Currently activated" vs "dormant" node counts

**e) Stats & Insights**
- Total entities, relationships, facts
- Memory growth over time
- Most connected entities (hub nodes)
- Entity type distribution

#### WebSocket Protocol

Defined event types with payload schemas:

| Event | Payload | Trigger |
|-------|---------|---------|
| `node:created` | `{id, name, entityType, summary}` | Ingestion completes |
| `node:updated` | `{id, changedFields}` | Entity summary changes |
| `node:activation_updated` | `{id, activation, previousActivation}` | Retrieval or ingestion |
| `edge:created` | `{source, target, predicate, weight}` | New relationship |
| `edge:invalidated` | `{source, target, predicate, validTo}` | Contradiction detected |
| `episode:queued` | `{episodeId, source}` | remember called |
| `episode:completed` | `{episodeId, entitiesCreated, factsExtracted}` | Pipeline finishes |
| `episode:failed` | `{episodeId, error, stage}` | Pipeline error |

Reconnection: client tracks last sequence number, requests missed events or falls back to full REST refetch.

### 8. Configuration

All configuration via a single Pydantic Settings model supporting env vars (ENGRAM_ prefix), .env files, and YAML.

**Key config sections:**

```yaml
claude:
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-haiku-4-5-20251001"   # Haiku for extraction (cost-efficient)
  extraction_model: "claude-sonnet-4-6" # Sonnet for complex episodes

redis:
  host: "localhost"
  port: 6381

falkordb:
  host: "localhost"
  port: 6380

activation:
  decay_exponent: 0.5
  max_history_size: 200
  B_mid: -0.5
  B_scale: 1.0
  spread_max_hops: 2
  spread_decay_per_hop: 0.5
  spread_firing_threshold: 0.05
  spread_energy_budget: 5.0
  weight_semantic: 0.50
  weight_activation: 0.35
  weight_edge_proximity: 0.15
  seed_threshold: 0.3
  retrieval_top_k: 50
  retrieval_top_n: 10

embedding:
  provider: "voyage"
  model: "voyage-3-lite"
  dimensions: 512

server:
  host: "0.0.0.0"
  port: 8787
  mode: "auto"                  # "auto" | "full" | "lite"
  log_level: "info"

auth:
  enabled: false                # true for production
  mode: "self_hosted"           # "self_hosted" | "saas"
  bearer_token: "${ENGRAM_AUTH_SECRET}"
  default_group_id: "default"

encryption:
  enabled: false                # true for production
  master_key: "${ENGRAM_MASTER_KEY}"

cors:
  allowed_origins: ["http://localhost:3000", "http://localhost:5173"]

rate_limiting:
  enabled: false
```

---

## Data Flow Example

User in Claude Desktop says: "I've been working on ReadyCheck, trying to get the Stripe integration done this week. Also started looking at retatrutide protocols for body comp."

**Ingestion (async):**

1. `remember` called → episode persisted, queued to Redis Stream → returns `{episode_id, status: "queued"}` in ~10ms
2. Background worker picks up episode. Claude extracts:
   - Entities: ReadyCheck (Project), Stripe (Tool/Technology), retatrutide (Concept)
   - Relationships: User → WORKS_ON → ReadyCheck, ReadyCheck → INTEGRATES → Stripe, User → RESEARCHING → retatrutide
   - Facts: "ReadyCheck Stripe integration in progress (as of Feb 2026)", "User exploring retatrutide for body composition"
3. Graph Manager:
   - ReadyCheck already exists? Update last_accessed, record access
   - Stripe node exists? Link to ReadyCheck with INTEGRATES edge
   - retatrutide exists? Strengthen via new access, add new fact
4. Embed episode content + new/updated entity summaries (batched Voyage API call)
5. WebSocket notifies dashboard: `episode:completed` with entity/fact counts

**Later retrieval:**

User asks: "What was I doing with payments?"

1. Embed query → vector search finds: Stripe, ReadyCheck, payment-related nodes
2. Compute activation for each (lazy, from access_history)
3. Identify seeds (semantic_similarity ≥ 0.3)
4. Spread activation: Stripe → ReadyCheck (0.7 edge weight) → SaaS launch timeline
5. Composite score: 0.50 × semantic + 0.35 × activation + 0.15 × edge_proximity
6. Return: "You're integrating Stripe into ReadyCheck, your meeting prep SaaS. This was in progress as of last week. ReadyCheck has an 8-week launch target."

The spreading activation brought in the launch timeline even though the user didn't ask about it. That's associative recall.

---

## Build Plan: 8 Weeks

### Week 1: Foundation + Infrastructure

**Goal: Running graph store + basic extraction + CI from day one**

- [ ] Repo structure (monorepo: /server, /dashboard, /mcp)
- [ ] Docker Compose with named volumes, health checks, depends_on conditions
  - FalkorDB (with HEALTHCHECK), Redis (with HEALTHCHECK), restart policies
- [ ] GitHub Actions CI: lint (ruff + mypy), unit tests, docker compose smoke test
- [ ] Pydantic Settings config model with all sections (activation, embedding, auth, etc.)
- [ ] FalkorDB index DDL script (14 indexes), run at startup via `ensure_indexes()`
- [ ] Python project setup (FastAPI, async)
- [ ] TenantContext middleware (P0 security -- even with auth disabled)
- [ ] group_id enforcement in all Cypher queries
- [ ] Bearer token auth (can be disabled for dev)
- [ ] .gitignore with secrets protection
- [ ] Claude API integration for entity extraction
- [ ] Basic graph CRUD operations against FalkorDB
- [ ] Lite mode: SQLite graph adapter + in-memory activation store
- [ ] Write 5 test episodes and verify graph construction
- [ ] Simple CLI tool to ingest text and dump graph state

**Deliverable:** Can feed text in, see entities and relationships. CI green from day one. Both full and lite modes work.

### Week 2: Temporal Model + Conflict Resolution

**Goal: Facts have time validity, conflicts are detected**

- [ ] Bi-temporal model on edges (valid_from, valid_to, created_at, expired_at)
- [ ] Temporal extraction from natural language ("started last week", "since January")
- [ ] Conflict detection: same subject + predicate with different object → invalidate old
- [ ] Entity deduplication (fuzzy name matching + embedding similarity)
- [ ] PII encryption at rest (AES-256-GCM, per-tenant keys)
- [ ] PII detection flag during extraction
- [ ] Integration tests: 20 episodes across 5 conversations, verify temporal accuracy

**Deliverable:** Graph correctly handles evolving information. "I moved to Denver" invalidates "lives in Mesa."

### Week 3: Activation Engine + Benchmark

**Goal: Working activation engine + competitive benchmark (VALIDATE HYPOTHESIS)**

- [ ] ActivationState model + Redis storage with {group_id}: prefix
- [ ] ACT-R base-level activation (power-law decay, lazy evaluation)
- [ ] Spreading activation (bounded: threshold, degree normalization, visited set, budget)
- [ ] 3-signal composite scorer (semantic, activation, edge_proximity)
- [ ] Embedding pipeline: Voyage-3-lite, Redis Search HNSW index
- [ ] Retrieval endpoint: query → activated results
- [ ] **COMPETITIVE BENCHMARK (go/no-go gate):**
  - 50-episode dataset across 10 conversation threads
  - 20 queries with human-annotated ground truth
  - Compare: Engram activation vs pure vector vs keyword vs graph traversal
  - Metrics: precision@5, recall@10, MRR, latency p50/p99
  - **Go/no-go: activation must beat vector search by ≥15% on precision@5**
  - If no-go: assess whether dashboard alone is sufficient differentiator
- [ ] Unit economics model: cost per ingestion, cost per retrieval, monthly cost per user

**Deliverable:** Validated (or invalidated) hypothesis with reproducible benchmark. This is the most important week.

### Week 4: MCP Server

**Goal: Claude Desktop and Claude Code can use Engram as memory**

- [ ] MCP server implementation (native stdio via mcp.server.fastmcp)
  - All 7 tools with full JSON Schema
  - MCP resources (graph stats, entity detail, neighbors)
  - MCP prompts (engram-system, engram-context-loader)
- [ ] Async ingestion via Redis Streams (remember returns immediately)
- [ ] Session tracking (session_id per MCP connection)
- [ ] Claude Desktop config integration guide
- [ ] Claude Code integration guide
- [ ] Rate limiting on MCP endpoints
- [ ] Test: 10-turn conversation in Claude Desktop, verify memory persists
- [ ] Test: start new conversation, verify recall works across sessions

**Deliverable:** Working memory for Claude Desktop. Converse, close, open new, Claude remembers.

### Week 5: Dashboard - Graph Explorer

**Goal: Beautiful, interactive graph visualization**

- [ ] React project setup (Vite + TypeScript + Tailwind + Zustand)
- [ ] Zustand store with typed slices (graph, selection, time, activation, websocket)
- [ ] FastAPI REST endpoints:
  - `GET /api/graph?center={id}&depth=2&maxNodes=200` (subgraph neighborhood)
  - `GET /api/entities/{id}`
  - `GET /api/entities/{id}/neighbors`
  - `GET /api/stats`
  - `GET /api/episodes` (paginated)
- [ ] WebSocket endpoint with defined event contract + sequence numbers
- [ ] CORS configuration
- [ ] Graph Explorer view:
  - react-force-graph-3d integration
  - Node sizing by activation, coloring by entity type
  - Edge rendering with relationship labels
  - Hover tooltips, click to expand/focus
  - Activation heatmap overlay toggle
- [ ] Search bar: find and zoom to entities
- [ ] 2D fallback mode

**Deliverable:** Open localhost:3000, see your AI's memory as an interactive 3D graph. This is the screenshot moment.

### Week 6: Dashboard - Timeline + Feed

**Goal: Temporal exploration and memory management**

- [ ] Timeline view:
  - Horizontal timeline with entity creation markers
  - Fact validity ranges (bars showing when facts were true)
  - Episode markers with source labels
  - Time scrubber: `GET /api/graph/at?timestamp={iso}` server-side temporal query
- [ ] Memory Feed:
  - Reverse-chronological extracted facts
  - Quick actions: edit, delete, merge entities
  - Filter by entity type, date range, source
- [ ] Stats panel:
  - Total entities, relationships, facts
  - Growth chart, most connected entities, entity type distribution
- [ ] GDPR endpoints: `GET /api/gdpr/export`, `DELETE /api/gdpr/erase`

**Deliverable:** Full dashboard with three views. Can explore memory temporally and manage it.

### Week 7: Activation Monitor + Polish

**Goal: Real-time activation visualization, production polish**

- [ ] Activation Monitor view:
  - Live activation levels bar chart (top 20 nodes)
  - Spreading activation animation during queries
  - Decay curve visualization
  - Activated vs dormant counts
- [ ] Dashboard polish:
  - Dark mode
  - Smooth animations/transitions
  - Loading states, error handling
  - Mobile-responsive (tablet)
- [ ] Docker Compose production config
- [ ] One-command startup: `docker compose up`
- [ ] Backup/restore script (`scripts/backup.sh`)
- [ ] Health check endpoints
- [ ] Audit logging (structlog)
- [ ] README with screenshots, quickstart, architecture diagram

**Deliverable:** Production-quality local product. `docker compose up`, configure Claude, start using.

### Week 8: Launch Prep

**Goal: Open source release + initial content**

- [ ] GitHub repo:
  - Clean README with hero screenshot
  - CONTRIBUTING.md
  - LICENSE (Apache 2.0)
  - Architecture docs
  - API reference with JSON Schema documentation
- [ ] Benchmark results document (from Week 3, polished)
- [ ] Blog post: "Why AI Memory Needs Spreading Activation"
- [ ] Twitter/X thread with dashboard GIFs
- [ ] Product Hunt prep
- [ ] Hacker News Show HN post draft
- [ ] Demo video (2-3 min): install, configure, converse, show dashboard

**Deliverable:** Public launch. Repo live, content published, demo available.

---

## Post-Launch Roadmap

### Month 3: Hosted Service (SaaS)

- Cloud-hosted FalkorDB + Redis per user
- Dashboard at app.engram.dev
- JWT auth for dashboard, API key for MCP clients
- Free tier: 1 graph, 1,000 entities, 5,000 episodes
- Pro tier ($15/mo): unlimited entities/episodes, multiple graphs
- Team tier ($49/mo): shared graphs, multi-user
- **Pre-launch:** model unit economics (cost per ingestion, per retrieval, per user/month)

### Month 4: Advanced Features

- Multi-model support (OpenAI, Gemini for extraction)
- Import/export (bring your Mem0 data in, export your graph)
- Custom entity types (user-defined schema)
- Webhooks for graph events
- Plugin system for custom activation rules
- Edge activation (Phase 2 -- edges have their own access history)

### Month 5: Ecosystem

- VS Code extension (inline memory panel)
- Browser extension (capture web research into graph)
- API for third-party integrations
- Community entity extractors
- **Community strategy:** Discord from day one, contributor-friendly plugin system, memory template gallery

---

## Technical Decisions & Rationale

**Why FalkorDB over Neo4j?**
Lighter weight, runs as Redis module, free for commercial use, faster for personal-scale graphs (<100K nodes). Neo4j is overkill for individual memory and has licensing complexity.

**Why Redis for activation state?**
Sub-millisecond reads. Activation needs to be checked on every retrieval. Also serves as the async ingestion queue (Redis Streams), WebSocket pub/sub fan-out, rate limiting backend, and embedding vector index (Redis Search HNSW).

**Why ACT-R for decay instead of the original hyperbolic formula?**
The original formula `1/(1+d*log(t+1))` decays so slowly that nothing ever goes dormant. ACT-R's power-law decay is backed by 40+ years of cognitive science and produces the long-tail behavior observed in human memory: heavily-accessed items persist, one-off mentions fade.

**Why 3 scoring signals instead of 4?**
The original `recency_score` and `frequency_score` double-count signals already encoded in the ACT-R activation formula. Collapsing to 3 orthogonal signals (semantic, activation, edge proximity) eliminates bias and produces cleaner rankings.

**Why async ingestion?**
Synchronous Claude API calls (1-3s) would block the MCP client on every `remember` call. Async returns in ~10ms, with extraction running in a background worker pool. Also enables retry, dead-letter handling, and batching.

**Why Voyage-3-lite for embeddings?**
512 dimensions keeps storage compact. Quality is sufficient for candidate retrieval since the activation engine re-ranks results. $0.02/1M tokens is negligible compared to Claude extraction costs.

**Why dual-mode (full + lite)?**
`docker compose up` is a 30-minute setup for new users. `pip install engram` is 2 minutes. Lite mode collapses time-to-first-memory from 30 min to 2 min, which is the difference between someone trying Engram and closing the tab.

**Why native stdio MCP transport?**
The original spec relied on `mcp-remote` as a bridge for Claude Desktop. This adds a process layer, complicates error handling, and creates a fragile dependency. Native stdio via `mcp.server.fastmcp` eliminates all of this.

**Why bearer token auth even for self-hosted?**
Personal memories served over open HTTP is a security gap even on localhost. A single bearer token adds negligible friction and prevents network-adjacent access.

**Why Apache 2.0 license?**
Maximum adoption. Lets companies use it internally without legal friction. Industry standard for this category.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Activation doesn't beat vector search | Low | **Critical** | **Week 3 benchmark is a go/no-go gate.** If <15% improvement, pivot to dashboard as primary differentiator. |
| Claude API costs for extraction | Medium | Medium | Use Haiku for simple episodes, Sonnet for complex. Batch extraction. Async pipeline enables rate limit compliance. |
| FalkorDB performance at scale | Low | Medium | Personal graphs stay small; benchmark at 50K nodes in Week 3. |
| Competitor ships activation features | Medium | High | Move fast, ship first, build community. Dashboard + DX are the real moat. |
| Dashboard complexity slows timeline | Medium | Medium | Ship Graph Explorer first (Week 5), other views are incremental. |
| MCP protocol changes | Low | Low | Native stdio + abstracted transport for easy updates. |
| Embedding API outage during ingestion | Medium | Low | Embedding failures non-blocking: mark as pending, retry in background. |
| Redis data loss | Low | High | AOF with appendfsync everysec. Activation recovery from FalkorDB on startup. Named Docker volumes with backup script. |

---

## Success Metrics

**Week 3 (benchmark gate):**
- Activation retrieval ≥15% better than vector-only on precision@5
- Reproducible benchmark with published dataset and methodology
- Unit economics model shows viable cost per user

**Week 8 (launch):**
- Working product: Docker compose up → full stack running
- Lite mode: pip install → memory in 2 minutes
- 4+ dashboard views functional
- README with <5 min quickstart

**Month 3:**
- 500+ GitHub stars
- 50+ active users (self-hosted)
- 10+ issues/PRs from community
- Hosted service beta with 20+ signups

**Month 6:**
- 2,000+ GitHub stars
- 100+ monthly active hosted users
- $1K+ MRR (re-evaluate pricing based on unit economics)
- Featured in 2+ AI/dev newsletters

---

## File & Repo Structure

```
engram/
├── docker-compose.yml           # Full stack: FalkorDB + Redis + Server + Dashboard
├── docker-compose.dev.yml       # Dev overrides (hot reload, debug)
├── .github/
│   └── workflows/
│       └── ci.yml               # Lint + test + smoke test (from Week 1)
├── config.example.yaml          # Configuration template (no secrets)
├── .env.example                 # Environment variable template
├── .gitignore                   # Secrets protection
├── README.md
├── LICENSE                      # Apache 2.0
├── CONTRIBUTING.md
├── scripts/
│   ├── backup.sh                # Redis BGSAVE + volume snapshot
│   ├── seed_demo.py             # Generate demo data
│   └── benchmark.py             # Run retrieval benchmarks
├── docs/
│   ├── architecture.md
│   ├── activation-engine.md
│   ├── api-reference.md
│   └── benchmarks.md
├── server/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── engram/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app + middleware stack
│   │   ├── config.py            # Pydantic Settings model
│   │   ├── models/
│   │   │   ├── entity.py
│   │   │   ├── episode.py
│   │   │   ├── relationship.py
│   │   │   ├── activation.py
│   │   │   └── tenant.py        # TenantContext model
│   │   ├── extraction/
│   │   │   ├── claude_extractor.py
│   │   │   ├── prompts.py
│   │   │   └── resolver.py      # Entity dedup & conflict resolution
│   │   ├── graph/
│   │   │   ├── protocol.py      # GraphStore protocol (ABC)
│   │   │   ├── falkordb_store.py # Full mode: FalkorDB
│   │   │   ├── sqlite_store.py  # Lite mode: SQLite
│   │   │   ├── indexes.py       # Index DDL + ensure_indexes()
│   │   │   ├── temporal.py      # Bi-temporal logic
│   │   │   └── queries.py       # Common Cypher queries
│   │   ├── activation/
│   │   │   ├── engine.py        # ACT-R activation + spreading
│   │   │   ├── protocol.py      # ActivationStore protocol (ABC)
│   │   │   ├── redis_store.py   # Full mode: Redis
│   │   │   └── memory_store.py  # Lite mode: in-memory dict
│   │   ├── retrieval/
│   │   │   ├── scorer.py        # 3-signal composite scoring
│   │   │   ├── protocol.py      # SearchIndex protocol (ABC)
│   │   │   ├── vector_search.py # Full mode: Redis Search HNSW
│   │   │   ├── fts_search.py    # Lite mode: SQLite FTS5
│   │   │   ├── embeddings.py    # EmbeddingProvider ABC + implementations
│   │   │   └── context.py       # Context assembly for get_context
│   │   ├── ingestion/
│   │   │   ├── pipeline.py      # Async worker pipeline
│   │   │   ├── queue.py         # Redis Stream producer/consumer
│   │   │   └── dead_letter.py   # Failed episode handling
│   │   ├── security/
│   │   │   ├── middleware.py     # TenantContext + SecurityHeaders + RateLimit
│   │   │   ├── encryption.py    # AES-256-GCM field encryption
│   │   │   └── gdpr.py          # Erasure + export services
│   │   ├── api/
│   │   │   ├── routes.py        # REST endpoints (subgraph, timeline, stats)
│   │   │   └── websocket.py     # WebSocket with event contract
│   │   └── mcp/
│   │       ├── server.py        # MCP server (native stdio + HTTP/SSE)
│   │       └── tools.py         # MCP tool definitions with JSON Schema
│   └── tests/
│       ├── test_extraction.py
│       ├── test_activation.py
│       ├── test_retrieval.py
│       ├── test_security.py
│       └── benchmarks/
│           ├── dataset.json     # 50 episodes, 20 queries with ground truth
│           └── run_benchmark.py # Reproducible comparison framework
├── dashboard/
│   ├── package.json
│   ├── Dockerfile
│   ├── src/
│   │   ├── App.tsx
│   │   ├── store/
│   │   │   ├── types.ts         # Zustand store type definitions
│   │   │   ├── index.ts         # Store creation with slices
│   │   │   └── slices/
│   │   │       ├── graph.ts
│   │   │       ├── selection.ts
│   │   │       ├── time.ts
│   │   │       └── websocket.ts
│   │   ├── components/
│   │   │   ├── GraphExplorer.tsx
│   │   │   ├── Timeline.tsx
│   │   │   ├── MemoryFeed.tsx
│   │   │   ├── ActivationMonitor.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   └── NodeDetail.tsx
│   │   ├── hooks/
│   │   │   ├── useGraph.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useActivation.ts
│   │   ├── api/
│   │   │   └── client.ts        # REST + WebSocket client
│   │   └── styles/
│   │       └── globals.css
│   └── public/
└── refined/                      # Detailed implementation specs (10 docs)
    ├── 01_config_and_indexes.md
    ├── 02_activation_engine.md
    ├── 03_async_ingestion.md
    ├── 04_embedding_strategy.md
    ├── 05_security_model.md
    ├── 06_mcp_protocol.md
    ├── 07_frontend_architecture.md
    ├── 08_devops_infrastructure.md
    ├── 09_lite_mode.md
    └── 10_benchmark_methodology.md
```

---

*This is a living document. The 10 detailed implementation specs in `/refined/` contain complete code examples, JSON schemas, and configuration details for each component. Update as architecture decisions evolve during build.*
