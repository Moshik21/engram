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
  <img src="https://img.shields.io/badge/tests-1325_passing-brightgreen" alt="Tests">
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
- **Understand** personal, health, and emotional content with an expanded 17-type entity taxonomy and rich predicate vocabulary
- **Visualize** the knowledge graph in real-time with a 3D neural brain dashboard

## Quickstart

### Setup Wizard (recommended)

The interactive setup wizard configures Engram globally (`~/.engram/.env`) so it works across all your projects:

```bash
cd server
uv sync
uv run python -m engram setup
```

The wizard walks you through:
1. API keys (Anthropic required, Voyage AI optional)
2. Engine mode (lite/full/auto)
3. Consolidation profile
4. Security settings (auth token, encryption)
5. AutoCapture hooks (Claude Code — auto-captures prompts and responses)
6. Generates `~/.engram/.env` with your config
7. Prints ready-to-paste MCP client config for Claude Desktop and Claude Code

Config loads globally from `~/.engram/.env`, with optional per-project overrides via a local `.env`.

### Option 1: MCP Server (recommended for Claude Code / Cursor / Windsurf)

```bash
cd server
uv sync
uv run python -m engram.mcp.server
```

This starts Engram in **lite mode** (zero dependencies beyond Python) using SQLite for storage. If you ran the setup wizard, your API keys are already configured. Otherwise, set them manually and add Engram to your MCP client config:

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
make up                  # or: docker compose up -d --build
```

Opens: **Dashboard** at `http://localhost:3000`, **API** at `http://localhost:8100`

Full mode runs FalkorDB (graph database), Redis (activation store + vector search), the FastAPI server, and the React dashboard. The docker-compose defaults to `standard` profile with all features enabled (LLM triage, PMI inference, dream associations, Sonnet escalation, background worker).

**Verify it's working:**

```bash
make status              # or: make health
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
| Search | FTS5 + numpy cosine | Redis Search HNSW (1024d) |
| Embeddings | Optional (Voyage AI) | Optional (Voyage AI) |

Mode is auto-detected: probes Redis + FalkorDB with a 2s timeout, falls back to lite.

### API Keys

Engram uses external APIs for two things: extracting structure from text and (optionally) embedding entities for semantic search.

#### Anthropic API (required)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Get a key at console.anthropic.com
```

**What it does**: When you `remember` something (or when the background worker promotes an `observe`d episode), Engram sends the text to Claude Haiku (`claude-haiku-4-5-20251001`) to extract entities, relationships, temporal markers, structured attributes, and polarity. The extraction prompt recognizes 17 entity types (including health, emotional, and personal domains), ~100 predicate synonyms that canonicalize to ~40 semantic groups, and handles negation/uncertainty ("stopped using X" invalidates existing edges). This is what turns unstructured text into a knowledge graph. Without it, Engram can't extract structure from memories.

**Cost**: Claude Haiku is Anthropic's fastest, cheapest model. A typical extraction uses ~500-1,500 input tokens and ~200-800 output tokens. At Haiku's pricing (~$0.80/1M input, ~$4/1M output), that's roughly **$0.001-0.005 per memory extracted**. With the `observe` + triage flow, only ~35% of stored content triggers extraction, reducing costs significantly. Consolidation phases (replay, LLM validation) also use Haiku when enabled, with optional Sonnet 4.6 escalation for uncertain verdicts. With prompt caching enabled, static system prompts are cached at $0.10/M vs $1.00/M, reducing input costs by ~80-90% on repeated calls.

#### Voyage AI (optional)

```bash
export VOYAGE_API_KEY=pa-...   # Get a key at dash.voyageai.com
```

**What it does**: Embeds each entity into a 1024-dimensional vector (`voyage-4-lite` model) for semantic search. Engram fuses FTS5 keyword results with vector cosine similarity via RRF, improving retrieval for associative and semantic queries.

**Without it**: Engram still works — retrieval uses FTS5 keyword matching, ACT-R activation, and spreading activation. You lose semantic similarity but keep everything else.

**Cost**: ~$0.01 per 1M tokens embedded. Embeddings are computed once per entity (re-computed only during consolidation reindex).

| | Anthropic (required) | Voyage AI (optional) |
|---|---|---|
| **Purpose** | Entity extraction from text | Semantic vector search |
| **Model** | Claude Haiku | voyage-4-lite (1024d) |
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

### Semantic Compilation

Engram's extraction pipeline acts as a semantic compiler — turning unstructured conversation into a structured, queryable knowledge graph. Five subsystems work together:

**Expanded Entity Taxonomy (17 types):** Beyond the standard `Person`, `Organization`, `Technology` types, Engram recognizes `HealthCondition`, `BodyPart`, `Emotion`, `Goal`, `Preference`, and `Habit`. This prevents personal and health content from being misclassified as generic `Other` entities (which contributed to a 30-40% triage scoring bias against personal content).

**Rich Predicate Vocabulary (~100 synonyms, ~40 canonical forms):** Predicates like `ENJOYS`, `LOVES`, `APPRECIATES` all canonicalize to `LIKES`. Groups cover health (`RECOVERING_FROM`, `HAS_CONDITION`, `TREATS`), sentiment (`LIKES`, `DISLIKES`, `PREFERS`), goals (`AIMS_FOR`), causation (`LED_TO`, `CAUSED_BY`, `REQUIRES`), hierarchy (`HAS_PART`, `PARENT_OF`, `CHILD_OF`), and learning (`STUDYING`). Contradictory pairs (`LIKES`/`DISLIKES`, `AIMS_FOR`/`AVOIDS`) are detected automatically.

**Structured Entity Attributes:** Instead of accumulating facts as appended summary strings that degrade over time, entities now carry a structured `attributes` dict with key-value pairs (e.g., `{"status": "recovering", "duration": "3 weeks", "severity": "mild"}`). New values overwrite old for the same key, keeping entity state current.

**Negation & Uncertainty Handling:** Every relationship carries a `polarity` field: `positive` (default), `negative`, or `uncertain`. "I stopped using React" creates a `negative` polarity `USES` edge that invalidates any existing positive `USES React` edge. Uncertain statements ("might switch to Svelte") are stored with halved weight. Negative-polarity edges are excluded from spreading activation at the SQL level, preventing invalidated relationships from influencing retrieval.

**Variable-Resolution Context Rendering:** `get_context()` renders entities at different detail levels based on their relevance. Seed entities (direct match) and identity core get **full** detail (summary + attributes + 5 facts). Hop-1 neighbors get **summary** detail (summary + 2 facts). Hop-2+ discoveries get **mention** only (name + type). This uses token budget more efficiently — peripheral context doesn't crowd out important details.

### Meta-Commentary Filtering

Engram includes a 5-layer defense against meta-contamination — when debugging discussions or system telemetry get extracted as real-world facts (e.g., "Kallon has activation score 0.91" polluting Kallon's entity summary).

| Layer | Where | What It Does |
|-------|-------|-------------|
| **Discourse classifier** | Worker, Triage, `project_episode()` | Regex-based gate classifies content as `world`, `hybrid`, or `system`. Pure system-discourse episodes are skipped before extraction. |
| **Extraction prompt** | LLM extraction | Instructions tell Claude to ignore system metrics and return empty results for meta-commentary. |
| **Epistemic mode tagging** | LLM extraction + entity loop | Entities tagged `"meta"` by the LLM are skipped during graph writes. |
| **Summary merge guard** | `_merge_entity_attributes()` | Meta-contaminated summaries are rejected for protected entity types (Person, CreativeWork, Location, Event, Organization, Emotion, Goal, Preference) but allowed for technical types (Technology, Concept, Software). |
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
2. **Route** — Classify query type (temporal, frequency, creation, associative, direct) and adjust scoring weights
3. **Activate** — Batch-fetch ACT-R activation states
4. **Spread** — BFS or PPR spreading activation through the graph (2 hops, fan-based dampening; `DREAM_ASSOCIATED` edges exempt from cross-domain penalty)
5. **Enrich** — Compute similarity for entities discovered via spreading
6. **Score** — Composite: `0.40 × semantic + 0.25 × activation + 0.15 × spreading + 0.15 × edge_proximity + exploration` (with hop_distance tracked per result)
7. **Rerank** — Optional cross-encoder (Cohere) + MMR diversity filter
8. **Return** — Top-10 results with score breakdowns including hop_distance

### Memory Consolidation

Engram runs offline consolidation cycles inspired by biological memory consolidation during sleep. Eight phases execute sequentially:

| Phase | What It Does |
|-------|-------------|
| **Triage** | Score QUEUED episodes by heuristics (or LLM judge when enabled), filter system meta-commentary, extract top ~35%, skip the rest |
| **Replay** | Re-extract recent episodes with Claude Haiku to recover missed entities |
| **Merge** | Fuzzy-match duplicate entities (thefuzz + union-find), with optional LLM-assisted borderline resolution |
| **Infer** | Create edges for co-occurring entities (PMI scoring, optional LLM validation with Sonnet 4.6 escalation for uncertain verdicts) |
| **Prune** | Soft-delete dead entities (no relationships, no access, old enough) |
| **Compact** | Logarithmic bucketing of access history + consolidated strength preservation |
| **Reindex** | Re-embed entities affected by earlier phases |
| **Dream** | Offline spreading activation to strengthen associative pathways + discover cross-domain creative connections via dream associations |

Consolidation is opt-in, controlled by profiles:

| Profile | Behavior |
|---------|----------|
| `off` | Default. No consolidation, no triage, no background worker. Extraction only (Haiku). |
| `observe` | Consolidation enabled in dry-run mode. Triage heuristics + worker active. Dream spreading, PMI inference, dream associations active (in dry-run). Good for monitoring. |
| `conservative` | Live consolidation with stricter thresholds. Dream spreading, replay enabled. Merge threshold 0.92, prune min age 60 days. Triage heuristics, extracts top 25%. |
| `standard` | Full consolidation with all features. LLM triage judge, PMI inference, transitivity, dream associations. Infer validation + Sonnet escalation, merge LLM + Sonnet escalation. Pressure-triggered. |

Set via env var: `ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard`

### Background Worker

When enabled (any profile except `off`), the `EpisodeWorker` runs as a background task that processes `observe`d content in near-real-time:

1. Subscribes to `episode.queued` events on the EventBus
2. Filters system meta-commentary via discourse classifier (activation scores, pipeline terms, entity IDs)
3. Scores each episode using heuristics or LLM judge (when `triage_llm_judge_enabled=true`)
4. High-scoring episodes get immediate extraction (same as `remember`)
5. Low-scoring episodes are stored but not extracted (still searchable via FTS)

This means you don't have to wait for a consolidation cycle — observed content that scores above threshold is extracted within seconds.

## MCP Integration

Engram exposes 11 MCP tools for AI agents:

| Tool | Purpose |
|------|---------|
| `observe` | Store raw text cheaply for background processing (default for most content) |
| `remember` | Store a memory with immediate entity extraction (for high-signal content) |
| `recall` | Retrieve relevant memories using activation-aware search |
| `search_entities` | Search entities by name or type |
| `search_facts` | Search relationships in the knowledge graph |
| `forget` | Soft-delete an entity or fact |
| `get_context` | Tiered context with identity/project/recency layers; supports briefing format |
| `get_graph_state` | Graph statistics and top-activated nodes |
| `mark_identity_core` | Mark/unmark an entity as identity core (protected from pruning) |
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

### AutoCapture Hooks (Claude Code)

While MCP instructions encourage the AI to call `observe()`/`remember()`, agents can still forget — especially during long coding sessions. **AutoCapture** solves this with Claude Code hooks that deterministically capture every user prompt and assistant response, feeding them into the existing `store_episode()` → background worker → triage pipeline with zero AI reliance.

#### How It Works

Four shell hooks run async (never blocking Claude) and POST to the Engram REST API:

| Hook | Event | What It Captures |
|------|-------|-----------------|
| `capture-prompt.sh` | `UserPromptSubmit` | User prompts (tagged `[user\|project]`) |
| `capture-response.sh` | `Stop` | Assistant responses, truncated to 2000 chars (tagged `[assistant\|project]`) |
| `session-start.sh` | `SessionStart` | Replays offline queue + posts session marker |
| `session-end.sh` | `SessionEnd` | Posts session marker + triggers consolidation |

The capture flow:

```
User sends prompt → capture-prompt.sh fires (async)
    → POST /api/knowledge/auto-observe
    → Server dedup check (5-min TTL, content hash)
    → store_episode() (~5ms, QUEUED)
    → Background worker buffers adjacent turns (30s window)
    → Merged prompt+response scored by triage
    → High-scoring content extracted (same as remember)
```

**Resilience**: If the REST server is down, hooks append to `~/.engram/capture-queue.jsonl`. On next session start, queued entries are replayed automatically.

**Dedup**: The `/api/knowledge/auto-observe` endpoint hashes content and skips duplicates within 5 minutes, preventing double capture if both MCP `observe` and the hook fire for the same content.

**Batching**: The background worker buffers auto-captured `prompt` + `response` episodes for 30 seconds. Adjacent turns within the window are merged into a single rich episode before scoring — the triage scorer sees the full exchange, not isolated turns.

#### Install AutoCapture

```bash
cd server
uv run python -m engram hooks
```

This:
1. Creates hook scripts in `~/.engram/hooks/` (4 shell scripts, `chmod +x`)
2. Merges hook config into `~/.claude/settings.json` (preserves existing hooks)
3. Prints a summary of what was configured

**Requirements**: The Engram REST server must be running at `http://localhost:8100` (or set `ENGRAM_URL`). Start it with:

```bash
cd server && uv run uvicorn engram.main:app --port 8100
```

Or via Docker (`docker compose up -d`).

#### Verify AutoCapture

After installing hooks, start a new Claude Code session and check:

1. **Session start** — `session-start.sh` fires, replays any queued entries
2. **User message** — appears in dashboard feed with `source: auto:prompt`
3. **Assistant response** — appears with `source: auto:response`
4. **Session end** — consolidation triggers automatically

You can also check the dashboard's episode feed or query the API:

```bash
curl -s http://localhost:8100/api/episodes | python3 -m json.tool
# Look for episodes with source "auto:prompt" / "auto:response"
```

#### Claude Code Setup

No additional setup needed beyond the MCP server and (optionally) AutoCapture hooks. The built-in system prompt instructs the AI to call `get_context()` at session start.

Optionally, add memory directives to your project's `.claude/CLAUDE.md` for stronger enforcement:

```markdown
## Engram Memory

- At conversation start, call `get_context()` to load relevant memory before your first response.
- Default to `observe()` for general conversation context and uncertain-value content.
- Use `remember()` only for high-signal items: identity facts, explicit preferences, key decisions, corrections.
- Use `recall()` or `search_facts()` when the user references past conversations or when context would help.
- When the user corrects a memory, call `forget()` on the old fact then `remember()` the correction.
- Do not announce memory operations. Integrate recalled context naturally.
```

#### Context Loading

`get_context()` assembles memory context in three prioritized tiers with **variable-resolution rendering** — seed entities and identity core get full detail (summary + attributes + 5 facts), hop-1 neighbors get summary detail (2 facts), and hop-2+ discoveries render as mentions only (name + type):

1. **Identity Core** (~200 tokens) — Always-included personal identity entities at full detail
2. **Project Context** (~400 tokens) — Entities relevant to current project, resolution varies by hop distance from query match
3. **Recent Activity** (~400 tokens) — Top-activated entities at summary detail

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 2000 | Total token budget |
| `topic_hint` | None | Topic to bias entity selection |
| `project_path` | None | Project directory; name used as topic hint |
| `format` | "structured" | `"structured"` (markdown) or `"briefing"` (LLM narrative) |

The **briefing format** calls Claude Haiku to synthesize a 2-3 sentence narrative summary, cached for 5 minutes. Example:

> You're talking with Konner, who is building Engram — a persistent memory system for AI agents. He's also writing "The Wound Between Worlds." Recently he's been focused on memory consolidation and dream associations.

### Example Usage

```
User: "We had a long planning meeting about the Q3 roadmap today"

Agent: [calls observe] → Episode stored as QUEUED (~5ms, no LLM)
  Background worker scores it, decides to extract or skip.

User: "Remember that Sarah joined the ML team last week and is working on the recommendation engine"

Agent: [calls remember] → Immediate extraction:
  Sarah [Person] ──MEMBER_OF──▶ ML Team [Organization]
  Sarah [Person] ──WORKS_ON──▶ Recommendation Engine [Project]

User: "I tweaked my Achilles tendon so I'm focusing on upper body hypertrophy now"

Agent: [calls remember] → Extraction with new entity types:
  Achilles Tendon Injury [HealthCondition, attributes: {status: "tweaked"}]
  Achilles Tendon [BodyPart]
  Upper Body Hypertrophy [Goal, attributes: {focus: "upper body"}]
  User ──RECOVERING_FROM──▶ Achilles Tendon Injury
  User ──AIMS_FOR──▶ Upper Body Hypertrophy

User: "I stopped using React and switched to Svelte"

Agent: [calls remember] → Negation handling:
  USES React edge invalidated (polarity: negative)
  User ──USES──▶ Svelte (new positive edge)

User: "Who's working on recommendations?"

Agent: [calls recall("recommendation engine team")] → Returns:
  1. Recommendation Engine (Project) — score: 0.89, hop: 0
     → Sarah WORKS_ON, started last week
  2. Sarah (Person) — score: 0.74, hop: 1
     → MEMBER_OF ML Team, WORKS_ON Recommendation Engine
```

## Dashboard

The real-time dashboard provides a 3D neural brain visualization of your knowledge graph, plus 7 views for exploring and monitoring memory:

| View | Description |
|------|-------------|
| **Graph** | 3D force-directed neural brain — nodes pulse on activation, edges glow on recall, entity types are color-coded |
| **Timeline** | Temporal navigation — view the graph at any historical point |
| **Feed** | Episode ingestion history with extraction details |
| **Activation** | ACT-R leaderboard with decay curve visualization |
| **Stats** | Entity counts, type distribution, growth timeline |
| **Consolidation** | Cycle history, phase timeline, pressure gauge, trigger controls |
| **Knowledge** | Chat interface with memory recall, entity browsing, search overlay, streaming responses |

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
| POST | `/api/knowledge/observe` | Store content without extraction (fast path) |
| POST | `/api/knowledge/auto-observe` | Auto-observe with classification (dedup, tagging) |
| POST | `/api/knowledge/remember` | Ingest with full extraction |
| GET | `/api/knowledge/recall` | Activation-aware memory search |
| GET | `/api/knowledge/facts` | Search facts/relationships |
| GET | `/api/knowledge/context` | Assembled memory context (structured or briefing) |
| POST | `/api/knowledge/forget` | Forget entity or fact |
| POST | `/api/knowledge/chat` | SSE streaming chat with memory context |
| GET | `/api/graph/at` | Temporal subgraph at a point in time |
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

Engram uses Pydantic Settings with env var support. Config is loaded in order (later sources override earlier):

1. `~/.engram/.env` — global config (created by `python -m engram setup`)
2. `./.env` — local per-project overrides
3. Environment variables — always take precedence

For first-time setup, run the wizard: `cd server && uv run python -m engram setup`

Key environment variables (or copy `.env.example` to `.env`):

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

# LLM Model Configuration (all default to OFF, enabled by 'standard' profile)
ENGRAM_ACTIVATION__TRIAGE_LLM_JUDGE_ENABLED=true              # Use Haiku as triage judge (replaces heuristics)
ENGRAM_ACTIVATION__CONSOLIDATION_INFER_LLM_ENABLED=true       # Haiku validates inferred edges
ENGRAM_ACTIVATION__CONSOLIDATION_INFER_ESCALATION_ENABLED=true # Sonnet 4.6 re-validates uncertain edges
ENGRAM_ACTIVATION__CONSOLIDATION_MERGE_LLM_ENABLED=true       # Haiku judges borderline entity merges
ENGRAM_ACTIVATION__CONSOLIDATION_MERGE_ESCALATION_ENABLED=true # Sonnet 4.6 resolves uncertain merges

# Auth (disabled by default)
ENGRAM_AUTH__ENABLED=true
ENGRAM_AUTH__BEARER_TOKEN=<token>

# Encryption (disabled by default)
ENGRAM_ENCRYPTION__ENABLED=true
ENGRAM_ENCRYPTION__MASTER_KEY=<64-hex-chars>
```

All activation engine parameters (150+ fields) are configurable via `ENGRAM_ACTIVATION__*` env vars. See `server/engram/config.py` for the full schema.

### LLM Models

Engram uses a 2-model architecture: Haiku for all high-volume work, Sonnet for escalation of uncertain decisions.

| Role | Model | Config Field | When Used |
|------|-------|-------------|-----------|
| **Extraction** | Claude Haiku 4.5 | `EntityExtractor(model=...)` | `remember`, promoted `observe`, replay |
| **Triage Judge** | Claude Haiku 4.5 | `triage_llm_judge_model` | Scoring queued episodes (replaces heuristics) |
| **Infer Validation** | Claude Haiku 4.5 | `consolidation_infer_llm_model` | Validating inferred edges |
| **Merge Judge** | Claude Haiku 4.5 | `consolidation_merge_llm_model` | Judging borderline entity merges |
| **Infer Escalation** | Claude Sonnet 4.6 | `consolidation_infer_escalation_model` | Re-validating uncertain edge verdicts |
| **Merge Escalation** | Claude Sonnet 4.6 | `consolidation_merge_escalation_model` | Re-validating uncertain merge verdicts |
| **Briefing** | Claude Haiku 4.5 | `briefing_model` | Synthesizing `get_context(format="briefing")` narrative |

All LLM features beyond basic extraction default to OFF. The `standard` consolidation profile enables all of them. Prompt caching is always active — static system prompts are cached via Anthropic's ephemeral cache, reducing input costs by ~80-90%.

## Security

- **Tenant isolation**: Every query filters by `group_id` (SQLite, FalkorDB, Redis)
- **Authentication**: Optional bearer token auth on all endpoints
- **Encryption**: AES-256-GCM with per-tenant HKDF-SHA256 key derivation
- **PII detection**: Entity extraction flags PII (names, emails, phones) with `pii_detected` + `pii_categories` fields
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
uv run pytest -m "not requires_docker" -v    # 1,325 tests
uv run ruff check .                           # Lint
uv run python -m engram.mcp.server            # MCP server (stdio)
uv run uvicorn engram.main:app --port 8100    # REST API
uv run python -m engram setup                 # Interactive setup wizard
uv run python -m engram hooks                 # Install AutoCapture hooks
uv run python -m engram config                # Edit configuration

# Frontend
cd dashboard
pnpm install && pnpm dev                      # Dev server (port 5173)
pnpm test                                     # 158 Vitest tests
pnpm build                                    # Production build

# Docker (full mode) — via Makefile
make up                                       # Build + start full stack (standard profile, all features)
make down                                     # Stop everything
make restart                                  # Stop + rebuild + start
make logs                                     # Tail all logs (make logs-server for server only)
make status                                   # Container status + health check
make clean                                    # Stop + delete volumes (WARNING: deletes data)
```

### Project Structure

```
server/engram/
  activation/       # ACT-R engine (BFS, PPR, strategy pattern)
  api/              # REST endpoints + WebSocket
  benchmark/        # Deterministic benchmark framework
  consolidation/    # 8-phase engine, scheduler, pressure accumulator
  embeddings/       # Voyage AI embedding provider
  events/           # EventBus + Redis pub/sub bridge
  extraction/       # Entity extraction (Claude Haiku), predicate canonicalization, discourse classifier
  ingestion/        # CQRS ingestion paths
  mcp/              # MCP server (11 tools, 3 resources, 2 prompts)
  models/           # Pydantic data models
  retrieval/        # Pipeline, scorer, router, reranker, MMR
  security/         # Auth middleware, AES-256-GCM encryption
  storage/          # SQLite, FalkorDB, Redis implementations
  worker.py         # Background episode processor (EventBus-driven)

dashboard/src/
  components/       # 38 React components (3D graph, panels, knowledge chat)
  store/            # 10 Zustand slices
  hooks/            # WebSocket with exponential backoff
  api/              # HTTP client
  lib/              # Utilities
  test/             # Vitest test suite

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
