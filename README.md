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
  <img src="https://img.shields.io/badge/tests-1145_passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License">
</p>

---

AI agents forget everything between sessions. Engram fixes that.

Engram is an open-source memory layer that gives AI agents persistent, searchable, activation-aware memory. It builds a temporal knowledge graph from conversations, uses cognitive science (ACT-R) to determine what's relevant, and runs offline consolidation cycles inspired by how biological memory works during sleep.

**Key capabilities:**

- **Remember** conversations, facts, and relationships as a knowledge graph
- **Recall** with activation-aware retrieval that prioritizes recent, frequent, and contextually relevant memories
- **Consolidate** memory offline: merge duplicates, infer missing relationships, prune stale entities, strengthen associative pathways
- **Visualize** the knowledge graph in real-time with a 3D dashboard

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
<summary><strong>Claude Desktop / Claude Code config</strong></summary>

Add to `~/.claude/settings.json` or Claude Desktop config:

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
# Edit .env: set ANTHROPIC_API_KEY (required)
docker compose up -d
```

Opens: **Dashboard** at `http://localhost:3000`, **API** at `http://localhost:8100`

Full mode runs FalkorDB (graph database), Redis (activation store + vector search), the FastAPI server, and the React dashboard.

### Option 3: REST API Only

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

**What it does**: When you `remember` something, Engram sends the text to Claude Haiku (`claude-haiku-4-5-20251001`) to extract entities, relationships, and temporal markers. This is what turns unstructured text into a knowledge graph. Without it, Engram can't ingest new memories.

**Cost**: Claude Haiku is Anthropic's fastest, cheapest model. A typical `remember` call uses ~500-1,500 input tokens and ~200-800 output tokens. At Haiku's pricing (~$0.80/1M input, ~$4/1M output), that's roughly **$0.001-0.005 per memory stored**. Consolidation phases (replay, LLM validation) also use Haiku when enabled.

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
| **When called** | Every `remember`, consolidation replay/validation | Entity creation + reindex |
| **Cost per call** | ~$0.001-0.005 | ~$0.00001 |
| **Without it** | Engram can't ingest memories | Falls back to keyword search |

### Data Flow

```
remember("Alice works at Acme Corp on the Quantum project")
    │
    ▼
  Episode created (QUEUED, ~10ms return)
    │
    ▼
  Entity Extraction (Claude Haiku)
    │  → Alice [Person]
    │  → Acme Corp [Organization]
    │  → Quantum [Project]
    │  → Alice ──WORKS_AT──▶ Acme Corp
    │  → Alice ──WORKS_ON──▶ Quantum
    ▼
  Graph Write + Activation Update
    │
    ▼
  Embedding (Voyage AI, optional)
    │
    ▼
  WebSocket notify → Dashboard updates
```

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

Engram runs offline consolidation cycles inspired by biological memory consolidation during sleep. Seven phases execute sequentially:

| Phase | What It Does |
|-------|-------------|
| **Replay** | Re-extract recent episodes with Claude Haiku to recover missed entities |
| **Merge** | Fuzzy-match duplicate entities (thefuzz + union-find for transitive chains) |
| **Infer** | Create edges for co-occurring entities (PMI scoring, optional LLM validation) |
| **Prune** | Soft-delete dead entities (no relationships, no access, old enough) |
| **Compact** | Logarithmic bucketing of access history + consolidated strength preservation |
| **Reindex** | Re-embed entities affected by earlier phases |
| **Dream** | Offline spreading activation to strengthen associative pathways |

Consolidation is opt-in (`consolidation_enabled=False` by default), defaults to dry-run mode, and can be triggered manually, on a schedule, by pressure accumulation, or at shutdown.

## MCP Integration

Engram exposes 9 MCP tools for AI agents:

| Tool | Purpose |
|------|---------|
| `remember` | Store a memory with automatic entity extraction |
| `recall` | Retrieve relevant memories using activation-aware search |
| `search_entities` | Search entities by name or type |
| `search_facts` | Search relationships in the knowledge graph |
| `forget` | Soft-delete an entity or fact |
| `get_context` | Pre-assembled context string of most activated memories |
| `get_graph_state` | Graph statistics and top-activated nodes |
| `trigger_consolidation` | Run a memory consolidation cycle |
| `get_consolidation_status` | Check consolidation status |

Plus 3 resources (`engram://graph/stats`, `engram://entity/{id}`, `engram://entity/{id}/neighbors`) and 2 prompts (`engram_system`, `engram_context_loader`).

### Example Usage

```
User: "Remember that Sarah joined the ML team last week and is working on the recommendation engine"

Agent: [calls remember] → Episode created, extracts:
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

The real-time dashboard provides 6 views for exploring and monitoring the knowledge graph:

| View | Description |
|------|-------------|
| **Graph** | 3D/2D force-directed graph with activation heatmap and entity type coloring |
| **Timeline** | Temporal navigation — view the graph at any historical point |
| **Feed** | Episode ingestion history with extraction details |
| **Activation** | ACT-R leaderboard with decay curve visualization |
| **Stats** | Entity counts, type distribution, growth timeline |
| **Consolidation** | Cycle history, phase timeline, pressure gauge, trigger controls |

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

# Auth (disabled by default)
ENGRAM_AUTH__ENABLED=true
ENGRAM_AUTH__BEARER_TOKEN=<token>

# Encryption (disabled by default)
ENGRAM_ENCRYPTION__ENABLED=true
ENGRAM_ENCRYPTION__MASTER_KEY=<64-hex-chars>
```

All activation engine parameters (35+ fields) are configurable via `ENGRAM_ACTIVATION__*` env vars or `config.yaml`. See `server/engram/config.py` for the full schema.

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
uv run pytest -m "not requires_docker" -v    # 1,064 tests
uv run ruff check .                           # Lint
uv run python -m engram.mcp.server            # MCP server (stdio)
uv run uvicorn engram.main:app --port 8100    # REST API

# Frontend
cd dashboard
pnpm install && pnpm dev                      # Dev server (port 5173)
pnpm test                                     # 81 Vitest tests
pnpm build                                    # Production build

# Docker (full mode)
docker compose up -d                          # FalkorDB + Redis + Server + Dashboard
```

### Project Structure

```
server/engram/
  activation/       # ACT-R engine (BFS, PPR, strategy pattern)
  retrieval/        # Pipeline, scorer, router, reranker, MMR
  consolidation/    # 7-phase engine, scheduler, pressure accumulator
  extraction/       # Entity extraction (Claude Haiku), canonicalization
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

## License

Apache 2.0 — see [LICENSE](LICENSE)
