<p align="center">
  <h1 align="center">Engram</h1>
  <p align="center">
    <strong>Memory layer for AI agents.</strong><br>
    Temporal knowledge graphs + ACT-R spreading activation + memory consolidation.
  </p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#mcp-integration">MCP Integration</a> &middot;
  <a href="#dashboard">Dashboard</a> &middot;
  <a href="#api-reference">API</a> &middot;
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/react-19-61dafb" alt="React 19">
  <img src="https://img.shields.io/badge/tests-1212_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License">
</p>

---

AI agents forget everything between sessions. Engram fixes that.

Engram is an open-source memory layer that gives AI agents persistent, searchable, activation-aware memory. It builds a temporal knowledge graph from conversations, uses cognitive science (ACT-R) to determine what's relevant, and runs offline consolidation cycles inspired by how biological memory works during sleep.

**Key capabilities:**

- **Observe** conversations cheaply with background processing that selectively extracts high-value content
- **Remember** critical facts and relationships as a knowledge graph with full LLM extraction
- **Recall** with activation-aware retrieval that prioritizes recent, frequent, and contextually relevant memories
- **Consolidate** memory offline: triage queued episodes, merge duplicates, infer missing relationships, prune stale entities, strengthen associative pathways
- **Visualize** the knowledge graph in real-time with a 3D neural brain dashboard

## Quickstart

### Option 1: MCP Server (recommended for Claude Code / Cursor / Windsurf)

```bash
cd server
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run python -m engram.mcp.server
```

This starts Engram in **lite mode** (zero dependencies beyond Python) using SQLite for storage. Add it to your MCP client config:

<details>
<summary><strong>Claude Code config</strong></summary>

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "ENGRAM_GROUP_ID": "my-project"
      }
    }
  }
}
```

</details>

<details>
<summary><strong>Cursor / Windsurf config</strong></summary>

Same MCP protocol. Add to your editor's MCP server configuration:

```json
{
  "engram": {
    "command": "uv",
    "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
    "env": {
      "ANTHROPIC_API_KEY": "sk-ant-...",
      "ENGRAM_GROUP_ID": "my-project"
    }
  }
}
```

</details>

### Option 2: Full Stack with Dashboard

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required), optionally VOYAGE_API_KEY
docker compose up -d
```

Opens: **Dashboard** at `http://localhost:3000`, **API** at `http://localhost:8100`

Full mode runs FalkorDB (graph database), Redis (activation store + vector search), the FastAPI server, and the React dashboard.

**Verify it's working:**

```bash
curl -s http://localhost:8100/health | python3 -m json.tool
# Should show: {"status": "ok", "mode": "full", ...}

# Open the dashboard
open http://localhost:3000
```

### Option 3: MCP + Dashboard (recommended — best experience)

Run the MCP server locally for AI memory **and** Docker for the live dashboard. You talk to Claude, it remembers, and you watch the knowledge graph grow in real-time.

```bash
# 1. Start Docker stack (dashboard + databases)
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and VOYAGE_API_KEY (optional)
docker compose up -d

# 2. Configure MCP server with Redis bridge
```

Add to your MCP client config (Claude Code, Claude Desktop, Cursor, etc.):

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "ENGRAM_MODE": "full",
        "ENGRAM_FALKORDB__HOST": "localhost",
        "ENGRAM_FALKORDB__PORT": "6380",
        "ENGRAM_FALKORDB__PASSWORD": "engram_dev",
        "ENGRAM_REDIS__URL": "redis://:engram_dev@localhost:6381/0",
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "standard"
      }
    }
  }
}
```

The MCP server connects to Docker's FalkorDB and Redis on mapped ports (6380, 6381). A Redis pub/sub bridge automatically forwards events from the MCP server to the dashboard — so when Claude stores a memory, you see the new entities appear live in the 3D graph.

### Option 4: REST API Only

```bash
cd server
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run uvicorn engram.main:app --port 8100
```

## How It Works

### Architecture

Engram runs in two modes with identical APIs:

| Layer | Lite Mode (default) | Full Mode (Docker) |
|-------|--------------------|--------------------|
| Graph | SQLite (WAL, FTS5) | FalkorDB (Cypher) |
| Activation | In-memory dict | Redis hashes (7-day TTL) |
| Search | FTS5 + numpy cosine | Redis Search HNSW (512d) |
| Embeddings | Optional (Voyage AI) | Optional (Voyage AI) |

Mode is auto-detected: probes Redis + FalkorDB with a 2s timeout, falls back to lite.

### API Keys

Engram uses external APIs for two things: extracting structure from text and (optionally) embedding entities for semantic search.

#### Anthropic API (required)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Get a key at console.anthropic.com
```

**What it does**: When you `remember` something (or when the background worker promotes an `observe`d episode), Engram sends the text to Claude Haiku (`claude-haiku-4-5-20251001`) to extract entities, relationships, and temporal markers. This is what turns unstructured text into a knowledge graph. Without it, Engram can't extract structure from memories.

**Cost**: Claude Haiku is Anthropic's fastest, cheapest model. A typical extraction uses ~500-1,500 input tokens and ~200-800 output tokens. At Haiku's pricing (~$0.80/1M input, ~$4/1M output), that's roughly **$0.001-0.005 per memory extracted**. With the `observe` + triage flow, only ~35% of stored content triggers extraction, reducing costs significantly. Consolidation phases (replay, LLM validation) also use Haiku when enabled.

#### Voyage AI (optional)

```bash
export VOYAGE_API_KEY=pa-...   # Get a key at dash.voyageai.com
```

**What it does**: Embeds each entity into a 512-dimensional vector (`voyage-4-lite` model) for semantic search. Engram fuses FTS5 keyword results with vector cosine similarity via RRF, improving retrieval for associative and semantic queries.

**Without it**: Engram still works — retrieval uses FTS5 keyword matching, ACT-R activation, and spreading activation. You lose semantic similarity but keep everything else.

**Cost**: ~$0.01 per 1M tokens embedded. Embeddings are computed once per entity (re-computed only during consolidation reindex).

| | Anthropic (required) | Voyage AI (optional) |
|---|---|---|
| **Purpose** | Entity extraction from text | Semantic vector search |
| **Model** | Claude Haiku | voyage-4-lite (512d) |
| **When called** | `remember`, promoted `observe`, consolidation replay/validation | Entity creation + reindex |
| **Cost per call** | ~$0.001-0.005 | ~$0.00001 |
| **Without it** | Engram can't ingest memories | Falls back to keyword search |

### Data Flow

Engram has two ingestion paths — a cheap fast path and a full extraction path:

```
observe("talked about the Quantum project today")     remember("Alice works at Acme Corp")
    │                                                      │
    ▼                                                      ▼
  QUEUED episode (~5ms, no LLM)                         QUEUED → immediate extraction
    │                                                      │
    ▼                                                      ▼
  Background Worker scores content                      Entity Extraction (Claude Haiku)
    │                                                      │  → Alice [Person]
    ├─ score >= threshold ──▶ project_episode()            │  → Acme Corp [Organization]
    │     (same extraction as remember)                    │  → Alice ──WORKS_AT──▶ Acme Corp
    │                                                      ▼
    └─ score < threshold ──▶ stored, not extracted       Graph Write + Activation + Embedding
                              (searchable via FTS)           │
                                                             ▼
                                                         WebSocket → Dashboard updates
```

The system prompt biases the LLM toward `observe` for most content, reserving `remember` for high-signal items (identity facts, explicit preferences, corrections). This reduces LLM extraction costs by ~65% while still capturing critical information immediately.

### Meta-Commentary Filtering

Engram includes a 5-layer defense against meta-contamination — when debugging discussions or system telemetry get extracted as real-world facts (e.g., "Kallon has activation score 0.91" polluting Kallon's entity summary).

| Layer | Where | What It Does |
|-------|-------|-------------|
| **Discourse classifier** | Worker, Triage, `project_episode()` | Regex-based gate classifies content as `world`, `hybrid`, or `system`. Pure system-discourse episodes are skipped before extraction. |
| **Extraction prompt** | LLM extraction | Instructions tell Claude to ignore system metrics and return empty results for meta-commentary. |
| **Epistemic mode tagging** | LLM extraction + entity loop | Entities tagged `"meta"` by the LLM are skipped during graph writes. |
| **Summary merge guard** | `_merge_entity_attributes()` | Meta-contaminated summaries are rejected for protected entity types (Person, CreativeWork, Location, Event, Organization) but allowed for technical types (Technology, Concept, Software). |
| **MCP prompt warning** | System prompt | Instructs AI agents not to store debugging output, activation scores, or system telemetry as memories. |

This prevents every debugging session from degrading the knowledge graph. The `observe` path is especially protected since meta-commentary tends to be keyword-dense and would otherwise score high on triage heuristics.

### ACT-R Activation

Memory retrieval is based on the [ACT-R cognitive architecture](https://en.wikipedia.org/wiki/ACT-R). Each entity has an activation level computed lazily from its access history:

```
B_i(t) = ln(consolidated_strength + Σ (t - t_j)^(-0.5))
```

Entities accessed recently and frequently have higher activation. This is normalized via sigmoid to [0, 1] and combined with other signals for retrieval scoring.

### Retrieval Pipeline

When you `recall("What does Alice work on?")`:

1. **Search** — FTS5 + optional vector search, top-50 candidates
2. **Route** — Classify query type (temporal, frequency, associative, direct) and adjust scoring weights
3. **Activate** — Batch-fetch ACT-R activation states
4. **Spread** — BFS spreading activation through the graph (2 hops, fan-based dampening)
5. **Enrich** — Compute similarity for entities discovered via spreading
6. **Score** — Composite: `0.40 × semantic + 0.25 × activation + 0.15 × spreading + 0.15 × edge_proximity + exploration`
7. **Rerank** — Optional cross-encoder (Cohere) + MMR diversity filter
8. **Return** — Top-10 results with score breakdowns

### Memory Consolidation

Engram runs offline consolidation cycles inspired by biological memory consolidation during sleep. Eight phases execute sequentially:

| Phase | What It Does |
|-------|-------------|
| **Triage** | Score QUEUED episodes by heuristics (length, keyword density, novelty), filter system meta-commentary, extract top ~35%, skip the rest |
| **Replay** | Re-extract recent episodes with Claude Haiku to recover missed entities |
| **Merge** | Fuzzy-match duplicate entities (thefuzz + union-find for transitive chains) |
| **Infer** | Create edges for co-occurring entities (PMI scoring, optional LLM validation) |
| **Prune** | Soft-delete dead entities (no relationships, no access, old enough) |
| **Compact** | Logarithmic bucketing of access history + consolidated strength preservation |
| **Reindex** | Re-embed entities affected by earlier phases |
| **Dream** | Offline spreading activation to strengthen associative pathways |

Consolidation is opt-in, controlled by profiles:

| Profile | Behavior |
|---------|----------|
| `off` | Default. No consolidation, no triage, no background worker. |
| `observe` | Consolidation enabled in dry-run mode. Triage + worker active. Good for monitoring. |
| `conservative` | Live consolidation with stricter thresholds. Triage extracts top 25%. |
| `standard` | Full consolidation with all features. Triage extracts top 35%. Pressure-triggered. |

Set via env var: `ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard`

### Background Worker

When enabled (any profile except `off`), the `EpisodeWorker` runs as a background task that processes `observe`d content in near-real-time:

1. Subscribes to `episode.queued` events on the EventBus
2. Filters system meta-commentary via discourse classifier (activation scores, pipeline terms, entity IDs)
3. Scores each episode using lightweight heuristics (no LLM call)
4. High-scoring episodes get immediate extraction (same as `remember`)
5. Low-scoring episodes are stored but not extracted (still searchable via FTS)

This means you don't have to wait for a consolidation cycle — observed content that scores above threshold is extracted within seconds.

## MCP Integration

Engram exposes 10 MCP tools for AI agents:

| Tool | Purpose |
|------|---------|
| `observe` | Store raw text cheaply for background processing (default for most content) |
| `remember` | Store a memory with immediate entity extraction (for high-signal content) |
| `recall` | Retrieve relevant memories using activation-aware search |
| `search_entities` | Search entities by name or type |
| `search_facts` | Search relationships in the knowledge graph |
| `forget` | Soft-delete an entity or fact |
| `get_context` | Pre-assembled context string of most activated memories |
| `get_graph_state` | Graph statistics and top-activated nodes |
| `trigger_consolidation` | Run a memory consolidation cycle |
| `get_consolidation_status` | Check consolidation status |

Plus 3 resources (`engram://graph/stats`, `engram://entity/{id}`, `engram://entity/{id}/neighbors`) and 2 prompts (`engram_system`, `engram_context_loader`).

### Automatic Memory Behavior

Engram ships with built-in MCP instructions that teach compatible AI agents (Claude, Cursor, Windsurf, etc.) to use memory proactively — no user prompting required:

- **Session start**: The agent calls `get_context()` before its first response to load relevant memories
- **Auto-observe**: For general conversation context and uncertain-value content, the agent calls `observe()` (cheap, no LLM)
- **Auto-remember**: For high-signal content (identity facts, explicit preferences, key decisions), the agent calls `remember()` (full extraction)
- **Auto-recall**: When you reference past conversations or ask "do you remember...", the agent calls `recall()` or `search_facts()`
- **Corrections**: When you correct a previously stored fact, the agent calls `forget()` on the old information then `remember()` with the correction

The system prompt biases toward `observe` by default — "if uncertain whether something is worth remembering, use observe." This reduces LLM extraction costs while the background worker and triage phase ensure high-value content still gets fully extracted.

This behavior is powered by the `instructions` parameter on the MCP server, so it works out of the box with any MCP-compatible client.

#### Claude Code: Enhanced Setup

For Claude Code users, you can add two optional layers for stronger enforcement:

**1. Project CLAUDE.md** — Add memory directives to your project's `.claude/CLAUDE.md`:

```markdown
## Engram Memory

- At conversation start, call `get_context()` to load relevant memory before your first response.
- Default to `observe()` for general conversation context and uncertain-value content.
- Use `remember()` only for high-signal items: identity facts, explicit preferences, key decisions, corrections.
- Use `recall()` or `search_facts()` when the user references past conversations or when context would help.
- When the user corrects a memory, call `forget()` on the old fact then `remember()` the correction.
- Do not announce memory operations. Integrate recalled context naturally.
```

**2. SessionStart hook** — Inject memory context via the REST API as a fallback (requires the full Docker stack):

Create `~/.claude/hooks/engram-context.sh`:

```bash
#!/bin/bash
# Fails silently with 3s timeout if server is down.
SNAPSHOT=$(curl -sf --max-time 3 "http://localhost:8100/api/activation/snapshot?limit=10" 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$SNAPSHOT" ]; then
  exit 0
fi
echo "=== Engram Memory Context ==="
echo "$SNAPSHOT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for e in data.get('topActivated', [])[:10]:
        name, etype = e.get('name', '?'), e.get('entityType', '')
        summary = (e.get('summary') or '')[:120]
        act = e.get('activation', 0)
        print(f'- {name} ({etype}, act={act:.2f}): {summary}')
except Exception:
    pass
"
echo "==========================="
```

Then register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/engram-context.sh"
          }
        ]
      }
    ]
  }
}
```

This fires once per session, injecting the top-10 most activated entities into the conversation context. If Docker is down, the hook exits silently with no delay.

### Example Usage

```
User: "We had a long planning meeting about the Q3 roadmap today"

Agent: [calls observe] → Episode stored as QUEUED (~5ms, no LLM)
  Background worker scores it, decides to extract or skip.

User: "Remember that Sarah joined the ML team last week and is working on the recommendation engine"

Agent: [calls remember] → Immediate extraction:
  Sarah [Person] ──MEMBER_OF──▶ ML Team [Organization]
  Sarah [Person] ──WORKS_ON──▶ Recommendation Engine [Project]

User: "Who's working on recommendations?"

Agent: [calls recall("recommendation engine team")] → Returns:
  1. Recommendation Engine (Project) — score: 0.89
     → Sarah WORKS_ON, started last week
  2. Sarah (Person) — score: 0.74
     → MEMBER_OF ML Team, WORKS_ON Recommendation Engine
```

## Dashboard

The real-time dashboard provides a 3D neural brain visualization of your knowledge graph, plus 6 views for exploring and monitoring memory:

| View | Description |
|------|-------------|
| **Graph** | 3D force-directed neural brain — nodes pulse on activation, edges glow on recall, entity types are color-coded |
| **Timeline** | Temporal navigation — view the graph at any historical point |
| **Feed** | Episode ingestion history with extraction details |
| **Activation** | ACT-R leaderboard with decay curve visualization |
| **Stats** | Entity counts, type distribution, growth timeline |
| **Consolidation** | Cycle history, phase timeline, pressure gauge, trigger controls |

When paired with the MCP server (Option 3), the dashboard updates live via a Redis pub/sub bridge — store a memory in Claude, watch the new entities appear in the 3D graph instantly.

```bash
# Development
cd dashboard
pnpm install && pnpm dev    # http://localhost:5173

# Production (via Docker)
docker compose up -d         # http://localhost:3000
```

Built with React 19, TypeScript, Tailwind CSS 4, Three.js (3D graph), Recharts, and Zustand.

## API Reference

### REST Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check (version, mode, services) |
| GET | `/api/graph/neighborhood` | Subgraph centered on entity |
| GET | `/api/entities/search` | Search entities by name/type |
| GET | `/api/entities/{id}` | Entity detail with facts |
| GET | `/api/entities/{id}/neighbors` | Entity neighborhood |
| PATCH | `/api/entities/{id}` | Update entity |
| DELETE | `/api/entities/{id}` | Soft-delete entity |
| GET | `/api/episodes` | List episodes (paginated) |
| GET | `/api/stats` | Graph statistics |
| GET | `/api/activation/snapshot` | Top activated entities |
| GET | `/api/activation/{id}/curve` | ACT-R decay curve |
| POST | `/api/consolidation/trigger` | Trigger consolidation cycle |
| GET | `/api/consolidation/status` | Consolidation status + pressure |
| GET | `/api/consolidation/history` | Cycle history |
| GET | `/api/consolidation/cycle/{id}` | Cycle detail with audit records |
| WS | `/ws/dashboard` | Real-time events (episodes, graph deltas, activation) |

### WebSocket

Connect to `/ws/dashboard` for real-time updates. Events include episode lifecycle, graph mutations, activation snapshots, and consolidation progress. Supports resync via sequence numbers, ping/pong keepalive, and activation monitor subscriptions.

## Benchmarks

Engram includes a deterministic benchmark framework with 1,000 entities, 2,500+ relationships, and 80 ground-truth queries across 8 categories.

```bash
cd server

# Run benchmarks
uv run python scripts/benchmark_ab.py --verbose --seed 42

# With Voyage AI embeddings
uv run python scripts/benchmark_ab.py --embeddings --verbose --seed 42

# Scale test
uv run python scripts/benchmark_ab.py --entities 5000

# Echo chamber test
uv run python scripts/benchmark_echo_chamber.py --queries 200
```

### Results (1K entities, with embeddings)

| Method | P@5 | MRR | Best Category |
|--------|-----|-----|---------------|
| Full Stack | 0.395 | 0.773 | Frequency (0.94) |
| Multi-Pool | 0.388 | 0.751 | Frequency (0.94) |
| Pure Search | 0.307 | 0.801 | Direct (0.33) |

Full pipeline with spreading activation shows +28% P@5 improvement over pure search. Frequency queries (identifying most-accessed entities) are the standout at 94% precision — this is where ACT-R activation shines.

**Query categories**: direct lookup, recency, frequency, associative, temporal, semantic, graph traversal, cross-cluster. **Metrics**: P@5, R@10, MRR, nDCG@5, latency percentiles, bootstrap CI (1,000 resamples).

## Configuration

Engram uses Pydantic Settings with env var support. Copy `.env.example` to `.env`:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...          # Claude Haiku for entity extraction

# Optional
VOYAGE_API_KEY=pa-...                  # Voyage AI embeddings (optional, get key at dash.voyageai.com)
ENGRAM_GROUP_ID=my-project             # Tenant isolation
ENGRAM_MODE=auto                       # auto | lite | full

# Consolidation (off by default)
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard   # off | observe | conservative | standard
ENGRAM_ACTIVATION__WORKER_ENABLED=true              # Background episode processor

# Auth (disabled by default)
ENGRAM_AUTH__ENABLED=true
ENGRAM_AUTH__BEARER_TOKEN=<token>

# Encryption (disabled by default)
ENGRAM_ENCRYPTION__ENABLED=true
ENGRAM_ENCRYPTION__MASTER_KEY=<64-hex-chars>
```

All activation engine parameters (40+ fields) are configurable via `ENGRAM_ACTIVATION__*` env vars. See `server/engram/config.py` for the full schema.

## Security

- **Tenant isolation**: Every query filters by `group_id` (SQLite, FalkorDB, Redis)
- **Authentication**: Optional bearer token auth on all endpoints
- **Encryption**: AES-256-GCM with per-tenant HKDF-SHA256 key derivation
- **WebSocket**: Authenticates before accept (close 4001 on failure)
- **Column injection prevention**: Frozenset validation on updatable fields
- **Docker**: Non-root users, secrets via env vars only, `.env` excluded from images

## Development

### Prerequisites

- Python 3.10+ with [uv](https://docs.astral.sh/uv/)
- Node.js 22+ with pnpm (for dashboard)
- Docker (optional, for full mode)

### Commands

```bash
# Backend
cd server
uv run pytest -m "not requires_docker" -v    # 1,182 tests
uv run ruff check .                           # Lint
uv run python -m engram.mcp.server            # MCP server (stdio)
uv run uvicorn engram.main:app --port 8100    # REST API

# Frontend
cd dashboard
pnpm install && pnpm dev                      # Dev server (port 5173)
pnpm test                                     # 87 Vitest tests
pnpm build                                    # Production build

# Docker (full mode)
docker compose up -d                          # FalkorDB + Redis + Server + Dashboard
docker compose up -d --build                  # Rebuild after code changes
docker compose down                           # Stop everything
docker compose logs -f server                 # Tail server logs
```

### Project Structure

```
server/engram/
  activation/       # ACT-R engine (BFS, PPR, strategy pattern)
  retrieval/        # Pipeline, scorer, router, reranker, MMR
  consolidation/    # 8-phase engine, scheduler, pressure accumulator
  worker.py         # Background episode processor (EventBus-driven)
  extraction/       # Entity extraction (Claude Haiku), canonicalization, discourse classifier
  storage/          # SQLite, FalkorDB, Redis implementations
  mcp/              # MCP server (9 tools, 3 resources, 2 prompts)
  api/              # REST endpoints + WebSocket
  security/         # Auth middleware, AES-256-GCM encryption

dashboard/src/
  components/       # 17 React components (3D graph, panels, controls)
  store/            # 9 Zustand slices
  hooks/            # WebSocket with exponential backoff
  api/              # HTTP client

```

## Troubleshooting

**`remember` returns success but no entities are extracted**
Your `ANTHROPIC_API_KEY` is missing or invalid. The server starts fine without it, but extraction silently returns empty results. Check `docker compose logs server` for `Entity extraction failed` errors.

**Dashboard doesn't update when I use MCP tools**
The Redis event bridge connects the MCP server to the dashboard. Verify:
1. MCP server is configured with `ENGRAM_MODE=full` and the correct `ENGRAM_REDIS__URL` (including password)
2. Docker stack is running (`docker compose ps` — all services should be `healthy`)
3. Redis URL includes the password: `redis://:engram_dev@localhost:6381/0`

**Mode auto-detects as "lite" when I have Docker running**
Redis requires authentication. If `ENGRAM_FALKORDB__PASSWORD` or `ENGRAM_REDIS__URL` aren't set with the correct password, the 2s probe fails and Engram falls back to lite mode. Set the env vars in `.env` (see `.env.example`).

**Docker compose up fails or services are unhealthy**
```bash
docker compose down -v   # Stop and remove volumes (WARNING: deletes data)
docker compose up -d --build
docker compose ps        # Check health status
docker compose logs -f   # Watch for errors
```

**"Connection refused" when MCP connects to Docker services**
The MCP server runs on your host machine, connecting to Docker via mapped ports. FalkorDB maps `6380→6379`, Redis maps `6381→6379`. Make sure your MCP env uses `localhost:6380` (FalkorDB) and `localhost:6381` (Redis), not the Docker-internal port 6379.

## License

Apache 2.0 — see [LICENSE](LICENSE)
