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
  <img src="https://img.shields.io/badge/tests-1965_passing-brightgreen" alt="Tests">
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
- **Intend** with prospective memory — graph-embedded intentions that fire automatically via spreading activation when related topics come up ("remind me when...")
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
4. Recall profile (auto-recall, conversation awareness, proactive intelligence, prospective memory)
5. Security settings (auth token, encryption)
6. Generates `~/.engram/.env` with your config
7. Prints ready-to-paste MCP client config for Claude Desktop and Claude Code

Config loads globally from `~/.engram/.env`, with optional local `.env` overrides for API keys or engine mode.

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
        "ANTHROPIC_API_KEY": "sk-ant-..."
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
      "ANTHROPIC_API_KEY": "sk-ant-..."
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

Full mode runs FalkorDB (graph database), Redis (activation store + vector search), the FastAPI server, and the React dashboard. The docker-compose defaults to `standard` profile with all features enabled (multi-signal triage, PMI inference, dream associations, background worker). All triage, merge, and infer scoring is deterministic by default — no LLM API calls for consolidation decisions.

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

### Data Persistence

All data persists across server restarts in both modes:

| Data | Lite Mode | Full Mode |
|------|-----------|-----------|
| Knowledge graph | SQLite (`~/.engram/engram.db`) | FalkorDB (Docker volume `engram_falkordb_data`) |
| Activation store | In-memory (rebuilt from access history) | Redis (Docker volume `engram_redis_data`) |
| Search index | SQLite FTS5 (same db file) | Redis Search (same Redis volume) |
| Consolidation history | SQLite (same db file) | SQLite (Docker volume `engram_server_data`) |
| Episode content | SQLite (same db file) | SQLite (Docker volume `engram_server_data`) |

In full mode, consolidation audit records and episode content are stored in a SQLite sidecar database at `/home/engram/.engram/engram.db` inside the server container. This is persisted via the `engram_server_data` Docker volume — cycle history, triage records, and audit trails survive container rebuilds.

To **reset all data** (both modes):

```bash
# Lite mode
rm ~/.engram/engram.db

# Full mode (Docker) — WARNING: deletes everything
docker compose down -v
```

### API Keys

Engram uses external APIs for two things: extracting structure from text and (optionally) embedding entities for semantic search.

#### Anthropic API (required)

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Get a key at console.anthropic.com
```

**What it does**: When you `remember` something (or when the background worker promotes an `observe`d episode), Engram sends the text to Claude Haiku (`claude-haiku-4-5-20251001`) to extract entities, relationships, temporal markers, structured attributes, and polarity. The extraction prompt recognizes 17 entity types (including health, emotional, and personal domains), ~73 predicate synonyms that canonicalize to ~25 semantic groups, and handles negation/uncertainty ("stopped using X" invalidates existing edges). This is what turns unstructured text into a knowledge graph. Without it, Engram can't extract structure from memories.

**Cost**: Claude Haiku is Anthropic's fastest, cheapest model. A typical extraction uses ~500-1,500 input tokens and ~200-800 output tokens. At Haiku's pricing (~$0.80/1M input, ~$4/1M output), that's roughly **$0.001-0.005 per memory extracted**. Engram is designed to minimize LLM usage:

- **Triage + merge + infer** use deterministic multi-signal scorers (zero API cost) — LLM judges are available as opt-in fallback but disabled by default in the `standard` profile
- **Observe + triage flow** means only ~35% of stored content triggers extraction
- **Background worker** uses three-tier confidence routing (extract/defer/skip) with zero LLM calls
- **Prompt caching** on all extraction/validation prompts — static system prompts cached at $0.10/M vs $1.00/M, reducing input costs by ~80-90% on repeated calls
- **Remaining LLM usage**: entity extraction (`remember` + promoted `observe`), consolidation replay (re-extraction), knowledge chat (optional), and briefing synthesis (cached)

#### Embeddings (optional)

Engram supports two embedding providers for semantic vector search. If neither is configured, it falls back to keyword-only search.

**Option A: Voyage AI (cloud)**

```bash
export VOYAGE_API_KEY=pa-...   # Get a key at dash.voyageai.com
```

Embeds each entity into a 1024-dimensional vector (`voyage-4-lite` model). Cost: ~$0.01 per 1M tokens embedded.

**Option B: Local embeddings (offline, private)**

```bash
pip install engram[local]     # or: uv sync --extra local
export ENGRAM_EMBEDDING__PROVIDER=local
```

Uses [Nomic Embed v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768d, 137M params) via fastembed (ONNX runtime, CPU). The model (~130MB) is downloaded on first use and cached locally. No API key required.

**Auto-fallback**: If `provider=voyage` (default) but no `VOYAGE_API_KEY` is set, Engram automatically falls back to local embeddings when fastembed is installed. If neither is available, vector search is disabled (keyword search still works).

**Without embeddings**: Engram still works — retrieval uses FTS5 keyword matching, ACT-R activation, and spreading activation. You lose semantic similarity but keep everything else.

| | Anthropic (required) | Voyage AI (optional) | Local (optional) |
|---|---|---|---|
| **Purpose** | Entity extraction from text | Semantic vector search | Semantic vector search |
| **Model** | Claude Haiku | voyage-4-lite (1024d) | Nomic Embed v1.5 (768d) |
| **When called** | `remember`, promoted `observe`, consolidation replay, knowledge chat | Entity creation + reindex | Entity creation + reindex |
| **Cost per call** | ~$0.001-0.005 | ~$0.00001 | Free (CPU) |
| **Without it** | Engram can't ingest memories | Falls back to keyword search | Falls back to keyword search |

### Data Flow

Engram has two ingestion paths — a cheap fast path and a full extraction path:

```
observe("talked about the Quantum project today")     remember("Alice works at Acme Corp")
    │                                                      │
    ▼                                                      ▼
  QUEUED episode (~5ms, no LLM)                         QUEUED → immediate extraction
    │                                                      │
    ▼                                                      ▼
  Background Worker (multi-signal scorer, ~2ms)         Entity Extraction (Claude Haiku)
    │  8 signals: entity candidates, embedding              │  → Alice [Person]
    │  surprise, structural extractability,                  │  → Acme Corp [Organization]
    │  knowledge gaps, emotional salience...                 │  → Alice ──WORKS_AT──▶ Acme Corp
    │                                                        ▼
    ├─ high confidence (>0.70) ──▶ extract now           Graph Write + Activation + Embedding
    │     (same extraction as remember)                      │
    │                                                        ▼
    ├─ mid confidence ──▶ defer to Triage phase          WebSocket → Dashboard updates
    │     (batch-scored next consolidation cycle)
    │
    └─ low confidence (<0.15) ──▶ stored, not extracted
                                   (searchable via FTS)
```

The system prompt biases the LLM toward `observe` for most content, reserving `remember` for high-signal items (identity facts, explicit preferences, corrections). The worker's three-tier confidence routing means obvious decisions are made immediately (no LLM), while uncertain episodes are deferred to the triage phase for batch scoring with optional LLM escalation for borderline cases (~5%).

### Semantic Compilation

Engram's extraction pipeline acts as a semantic compiler — turning unstructured conversation into a structured, queryable knowledge graph. Five subsystems work together:

**Expanded Entity Taxonomy (17 types):** Beyond the standard `Person`, `Organization`, `Technology` types, Engram recognizes `HealthCondition`, `BodyPart`, `Emotion`, `Goal`, `Preference`, and `Habit`. This prevents personal and health content from being misclassified as generic `Other` entities (which contributed to a 30-40% triage scoring bias against personal content).

**Rich Predicate Vocabulary (~73 synonyms, ~25 canonical forms):** Predicates like `ENJOYS`, `LOVES`, `APPRECIATES` all canonicalize to `LIKES`. Groups cover health (`RECOVERING_FROM`, `HAS_CONDITION`, `TREATS`), sentiment (`LIKES`, `DISLIKES`, `PREFERS`), goals (`AIMS_FOR`), causation (`LED_TO`, `CAUSED_BY`, `REQUIRES`), hierarchy (`HAS_PART`, `PARENT_OF`, `CHILD_OF`), and learning (`STUDYING`). Contradictory pairs (`LIKES`/`DISLIKES`, `AIMS_FOR`/`AVOIDS`) are detected and invalidated automatically during graph writes.

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
| **Summary merge guard** | `_merge_entity_attributes()` | Meta-contaminated summaries are rejected universally for all entity types, preventing system telemetry from polluting any entity's summary. |
| **MCP prompt warning** | System prompt | Instructs AI agents not to store debugging output, activation scores, or system telemetry as memories. |

This prevents every debugging session from degrading the knowledge graph. The `observe` path is especially protected since meta-commentary tends to be keyword-dense and would otherwise score high on triage heuristics.

### One Brain Per Person

Engram is a brain, not a database. Each person gets **one** Engram instance — all their projects, personal life, health, goals, and conversations live in a single knowledge graph. Projects aren't separate partitions; they're natural entity clusters connected by topology.

This means:
- Memories from work **can** inform personal context (and vice versa)
- Dream associations **can** discover cross-domain creative connections
- Spreading activation reaches across project boundaries through shared concepts
- Identity core entities (name, preferences, health) are always available, regardless of which project you're in

**`group_id`** provides hard isolation between different people — like Row Level Security. It is not for separating projects. If you're the only user, the default (`"default"`) is all you need.

**Federated learning (roadmap):** While each brain is fully private, anonymized aggregate signals (schema patterns, activation curves, triage calibration, graph topology) can be shared across Engram instances to improve the system for everyone — like neuroscience studying many brains without reading anyone's thoughts.

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
6. **Score** — Composite: `0.40 × semantic + 0.25 × activation + 0.15 × spreading + 0.15 × edge_proximity + graph_structural + exploration` (with hop_distance tracked per result)
7. **Rerank** — Optional cross-encoder (Cohere) + MMR diversity filter
8. **Return** — Top-10 results with score breakdowns including hop_distance

### Memory Consolidation

Engram runs offline consolidation cycles inspired by biological memory consolidation during sleep. Nine phases execute sequentially:

| Phase | What It Does |
|-------|-------------|
| **Triage** | Score QUEUED episodes with 8-signal multi-signal scorer (~2ms/ep, zero LLM). Filter system meta-commentary. Extract top ~35%, skip the rest. Optional LLM escalation for ~5% borderline episodes. |
| **Merge** | Fuzzy-match duplicate entities (thefuzz + union-find + embedding ANN). Multi-signal scorer (name analysis + embeddings + neighbor Jaccard + summary Dice) replaces LLM judge — handles acronyms, numeronyms, tech suffixes, and canonical aliases deterministically |
| **Infer** | Create edges for co-occurring entities (PMI scoring). Multi-signal auto-validation (embedding coherence + type compatibility + ubiquity penalty + structural plausibility) replaces LLM judge — self-correcting via Dream LTD decay |
| **Replay** | Re-extract recent episodes with Claude Haiku to recover missed entities (selective — only runs when upstream phases changed the graph) |
| **Prune** | Soft-delete dead entities (no relationships, low access, old enough). Activation safety net prevents pruning warm entities. |
| **Compact** | Logarithmic bucketing of access history + consolidated strength preservation |
| **Reindex** | Re-embed entities affected by earlier phases |
| **Graph Embed** | Train structural graph embeddings (Node2Vec, TransE, GNN) for topology-aware retrieval. Incremental retraining with 5% change threshold; staggered TransE (every 3rd cycle) and GNN (every 5th). |
| **Dream** | Offline spreading activation to strengthen associative pathways + discover cross-domain creative connections via dream associations. LTD decay for unboosted edges. |

#### Three-Tier Scheduling

Phases run at different frequencies based on urgency:

| Tier | Phases | Default Interval |
|------|--------|-----------------|
| **Hot** | triage | 15 minutes |
| **Warm** | merge, infer, compact, reindex | 2 hours |
| **Cold** | replay, prune, graph_embed, dream | 6 hours |

Tiered cycles only run the phases that are due. Optimizations (incremental graph_embed, selective replay) only apply during tiered scheduling — manual triggers, pressure triggers, and scheduled flat cycles always run all phases fully.

#### Profiles

Consolidation is opt-in, controlled by profiles:

| Profile | Behavior |
|---------|----------|
| `off` | No consolidation, no triage, no background worker. Extraction only (Haiku). |
| `observe` | Consolidation enabled in dry-run mode. Triage heuristics + worker active. Dream spreading, PMI inference, dream associations active (in dry-run). Good for monitoring. |
| `conservative` | Live consolidation with stricter thresholds. Dream spreading, replay enabled. Merge threshold 0.92, prune min age 30 days. Triage heuristics, extracts top 25%. |
| `standard` | Full consolidation with all features. Multi-signal deterministic scorers for triage, merge, and infer — zero LLM cost for all consolidation decisions. PMI inference, transitivity, dream associations. Pressure-triggered. Three-tier scheduling. LLM judges available as opt-in fallback. |

Set via env var: `ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard`

### Multi-Signal Scoring Architecture

All consolidation decisions — triage, merge, and infer — use **deterministic multi-signal scorers** instead of LLM API calls. These scorers use strictly more information than an LLM judge (which only sees text) — combining embeddings, graph structure, entity candidates, and statistics. Zero API cost, <5ms latency, deterministic results, with self-improving calibration.

#### Triage Scorer (8 signals)

The triage scorer evaluates whether an episode is worth extracting. It replaces the LLM judge with signals that directly measure what extraction will produce:

| Signal | Weight | What It Measures |
|--------|--------|-----------------|
| Embedding surprise | 0.25 | Cosine distance from EMA corpus centroid (z-score normalized). Novel topics score high; redundant content scores low. |
| Structural extractability | 0.20 | Regex: proper names, relationship verbs, dates, quoted strings, URLs, numbers-in-context. Directly predicts entity yield. |
| Entity candidate count | 0.15 | FTS5 probe against entity index — how many names in the text already exist in the graph? More matches = richer extraction. |
| Knowledge gap | 0.10 | Names found in text that DON'T match existing entities = new knowledge the graph doesn't have yet. |
| Yield prediction | 0.10 | Online logistic regression calibrated from actual extraction outcomes. Cold-starts at 0.5, self-improves over ~200 samples. |
| Emotional salience | 0.10 | Arousal (state-change verbs), self-reference (pronouns), social density (roles), narrative tension (uncertainty markers). |
| Novelty (FTS5) | 0.05 | Episode search for similar existing content — high similarity = redundant, low = novel. |
| Goal boost | 0.05 | Keyword overlap with active goal entities from the graph. |

**Self-improving calibration:** After each extraction, the triage scorer records whether entities were actually extracted. An online logistic regression (sufficient statistics accumulator with exponential decay) learns which signal combinations predict successful extraction. Cold start → blending (30 samples) → fully calibrated (200+ samples). Per-group calibration, automatically adapts to each user's content patterns.

**LLM escalation (optional):** Episodes scoring in the borderline band (0.35–0.55) can be escalated to Claude Haiku for a second opinion. Capped at 5 per cycle. This means ~95% of triage decisions are deterministic, with LLM reserved for genuinely ambiguous cases.

#### Merge and Infer Scorers

**Merge scorer** (6 signals, weighted ensemble):

| Signal | Weight | What It Catches |
|--------|--------|----------------|
| Name analysis (fuzzy + acronym + numeronym + suffix strip + alias table) | 0.40 | JS↔JavaScript, K8s↔Kubernetes, React↔React.js |
| Embedding cosine similarity | 0.30 | Semantic equivalence beyond surface names |
| Neighbor Jaccard overlap | 0.15 | Same graph context = same entity |
| Summary Dice coefficient | 0.15 | Shared description terms |
| Type compatibility | gate | Person↔Technology = never merge |
| Booster rules | override | High-confidence signal combos → auto-merge |

**Infer scorer** (6 signals, weighted ensemble):

| Signal | Weight | What It Catches |
|--------|--------|----------------|
| Embedding coherence | 0.30 | Semantically related entities |
| Type compatibility | 0.20 | Domain-aware noise filtering |
| Statistical confidence (PMI) | 0.20 | Non-random association strength |
| Ubiquity penalty | 0.15 | Filters entities that appear everywhere |
| Structural plausibility (triangle closure) | 0.10 | Shared neighbors = plausible edge |
| Graph embedding similarity | 0.05 | Structural proximity (Node2Vec/TransE) |

**Tiered architecture** — decisions flow through tiers until resolved:

| Tier | Method | Coverage | Latency | Cost |
|------|--------|----------|---------|------|
| **Tier 0** | Multi-signal rules (above) | ~85% | <10ms | $0 |
| **Tier 1** | Cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, already loaded) | ~10% | ~50ms | $0 |
| **Tier 2** | Numpy classifier (future) | — | — | $0 |
| **Tier 3** | LLM fallback (opt-in) | ~2% | ~500ms | API cost |

Uncertain cases from Tier 0 are refined by the cross-encoder (Tier 1), which blends its score with the multi-signal score. Remaining uncertain infer edges self-correct via Dream LTD decay (unused edges weaken) and Hebbian reinforcement (useful edges strengthen). LLM judges remain available as opt-in fallback via `consolidation_merge_llm_enabled` and `consolidation_infer_llm_enabled`.

### Graph Embeddings

Engram learns structural embeddings from the knowledge graph topology during the **Graph Embed** consolidation phase. These capture patterns that text embeddings cannot: structural position, relational geometry, and multi-hop neighborhood identity.

**Why this matters:** Text embeddings know that "parenting routines" and "code architecture" are semantically distant. Graph embeddings can discover they're *structurally* similar — both are hub entities with many `PART_OF` children, both connect to goal-oriented clusters. This is how cross-domain insights surface: not through word similarity, but through shared structural patterns in how you think about different topics.

Three methods are available, each capturing different structural signals:

| Method | What It Learns | Algorithm | Dependencies | Min Threshold |
|--------|---------------|-----------|-------------|---------------|
| **Node2Vec** | Structural position — entities with similar graph neighborhoods cluster together (hubs with hubs, bridges with bridges) | Biased random walks + Skip-gram | Pure numpy | 50 entities |
| **TransE** | Relational geometry — learns `h + r ≈ t` so entities connected by the same predicate get geometrically consistent embeddings. `PARENT_OF` becomes a consistent vector direction. | Margin-based ranking loss | Pure numpy | 100 triples |
| **GNN (GraphSAGE)** | Semantic + structural fusion — initialized from text embeddings, then reshaped by 2-layer neighborhood aggregation with contrastive learning. The only method that combines *what entities mean* with *how they're connected*. | BPR contrastive loss | [PyTorch](https://pytorch.org/) | 200 entities |

#### Progressive Unlock

All three methods are enabled by default and train during consolidation cycles. Training is incremental — a 5% entity change threshold determines whether to do a full retrain or warm-start from existing embeddings. TransE and GNN are staggered across cycles to reduce compute load. Each method has a minimum threshold — methods automatically skip training until the graph is large enough, then activate:

```
50 entities   → Node2Vec activates (structural position)
100 triples   → TransE activates (relational geometry)
200 entities  → GNN activates (semantic + structural fusion)
```

As your graph grows, you progressively unlock richer structural understanding. Early on, Node2Vec provides basic topology awareness. Once relationships accumulate, TransE adds relational consistency. At scale, GNN produces the richest embeddings by fusing text meaning with graph structure.

#### Integration

Graph embeddings are stored in a separate `graph_embeddings` table (not concatenated with text embeddings) and integrated at two levels:

1. **Retrieval scoring** — `weight_graph_structural` (default 0.1) adds a topology-aware signal alongside semantic, activation, spreading, and edge proximity scores. The retrieval pipeline uses the first available method with stored embeddings (priority: node2vec > transe > gnn).

2. **Dream associations** — The dream phase blends graph embeddings (30%) with text embeddings (70%) when discovering cross-domain entity pairs. This means dream associations are based on both semantic similarity *and* structural similarity, producing more meaningful creative connections.

#### Install & Configure

Node2Vec and TransE require no extra dependencies. GNN requires PyTorch:

```bash
# Install with GNN support
pip install engram[gnn]
# or install everything
pip install engram[full]

# Run consolidation to train embeddings
cd server && uv run python -m engram.consolidation --profile standard
```

| Config | Default | Description |
|--------|---------|-------------|
| `graph_embedding_node2vec_enabled` | `true` | Enable Node2Vec random walk embeddings |
| `graph_embedding_node2vec_dimensions` | `64` | Embedding dimensions (16-256) |
| `graph_embedding_node2vec_min_entities` | `50` | Minimum entities to train |
| `graph_embedding_transe_enabled` | `true` | Enable TransE relational embeddings |
| `graph_embedding_transe_dimensions` | `64` | Embedding dimensions (16-256) |
| `graph_embedding_transe_min_triples` | `100` | Minimum relationship triples to train |
| `graph_embedding_gnn_enabled` | `true` | Enable GNN (requires PyTorch) |
| `graph_embedding_gnn_min_entities` | `200` | Minimum entities to train |
| `weight_graph_structural` | `0.1` | Retrieval weight for graph structural similarity (0.0-1.0) |

### Recall Profiles

Engram's retrieval intelligence is organized into four cumulative waves, controlled by a single `recall_profile` setting:

| Profile | What It Enables |
|---------|----------------|
| `off` | Basic `recall()` only — no automatic or proactive retrieval. |
| `wave1` | **AutoRecall** — Piggybacks on `observe`/`remember` to automatically surface related memories. Primes session context on first call. |
| `wave2` | + **Conversation Awareness** — Rolling topic fingerprint, multi-query decomposition, session entity seeds for spreading activation, near-miss detection. |
| `wave3` | + **Proactive Intelligence** — Topic shift detection triggers recall bursts, surprise connection detection (dormant but strongly-linked entities), retrieval priming (1-hop neighbor boosts), graph-connected MMR re-ranking for diversity. |
| `all` | + **Prospective Memory** — Graph-embedded intentions that fire via spreading activation when related entities light up. Create with `intend()`, monitor warmth in `get_context()`. |

Each wave includes all previous waves. Set via env var: `ENGRAM_ACTIVATION__RECALL_PROFILE=all`

### Prospective Memory

Prospective memory lets you set intentions that fire automatically based on context — not explicit recall. Intentions are stored as Entity nodes (type `"Intention"`) in the knowledge graph with `TRIGGERED_BY` edges to related entities. Triggering uses ACT-R spreading activation instead of brute-force embedding comparison.

**How it works:**

```
intend("auth module", "Check XSS fix before deploying", entity_names=["Auth Module"])
    │
    ▼
  Creates Intention entity + TRIGGERED_BY edge to "Auth Module"
    │
    ... later ...
    │
remember("Working on the auth module today")
    │
    ▼
  Extraction finds "Auth Module" entity
    │
  Mini spreading pass: Auth Module lights up → spreads to TRIGGERED_BY → Intention activates
    │
    ▼
  Intention fires → "Check XSS fix before deploying" surfaces in response
```

**Key capabilities:**

- **Transitive triggers** — An intention linked to "Auth Module" can fire when you discuss "JWT tokens" or "login flow" if those entities are graph-connected
- **Warmth monitoring** — `list_intentions` and `get_context()` show how close each intention is to firing (dormant / cool / warming / warm / HOT)
- **Cooldown + exhaustion** — Configurable cooldown (default 5 min) and max fires (default 5) prevent spam
- **Priority levels** — critical / high / normal / low; higher priority surfaces first

**Enable:** Set `ENGRAM_ACTIVATION__RECALL_PROFILE=all` (includes all previous waves). The v2 graph-embedded path is used by default (`prospective_graph_embedded=True`).

### Background Worker

When enabled (any profile except `off`), the `EpisodeWorker` runs as a background task that processes `observe`d content in near-real-time:

1. Subscribes to `episode.queued` events on the EventBus
2. Filters system meta-commentary via discourse classifier (activation scores, pipeline terms, entity IDs)
3. Scores each episode using heuristics or LLM judge (when `triage_llm_judge_enabled=true`)
4. High-scoring episodes get immediate extraction (same as `remember`)
5. Low-scoring episodes are stored but not extracted (still searchable via FTS)

This means you don't have to wait for a consolidation cycle — observed content that scores above threshold is extracted within seconds.

## MCP Integration

Engram exposes 15 MCP tools for AI agents:

| Tool | Purpose |
|------|---------|
| `observe` | Store raw text cheaply for background processing (default for most content) |
| `remember` | Store a memory with immediate entity extraction (for high-signal content) |
| `recall` | Retrieve relevant memories using activation-aware search |
| `search_entities` | Search entities by name or type |
| `search_facts` | Search relationships in the knowledge graph |
| `forget` | Soft-delete an entity or fact |
| `get_context` | Tiered context with identity/project/recency/intentions layers; supports briefing format |
| `get_graph_state` | Graph statistics and top-activated nodes |
| `mark_identity_core` | Mark/unmark an entity as identity core (protected from pruning) |
| `intend` | Create a graph-embedded intention with trigger entities and priority level |
| `dismiss_intention` | Disable or permanently delete an intention |
| `list_intentions` | List active intentions with warmth info (how close to firing) |
| `trigger_consolidation` | Run a memory consolidation cycle |
| `get_consolidation_status` | Check consolidation status |
| `bootstrap_project` | Auto-observe key project files and create a Project entity (idempotent) |

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

### Offline Queue

If the REST server is temporarily unreachable, clients can append entries to `~/.engram/capture-queue.jsonl`. On the next session (or via `POST /api/knowledge/replay-queue`), queued entries are drained atomically and ingested through the normal `store_episode()` → background worker → triage pipeline.

```python
from engram.utils.offline_queue import append_to_queue, drain_queue

# Client-side: queue when server is down
append_to_queue({"content": "...", "source": "offline:prompt"})

# Server-side: replay on reconnect
entries = drain_queue()  # atomically drains and returns all entries
```

The replay endpoint deduplicates against recently seen content (5-min TTL, SHA-256 hash) to prevent double ingestion.

#### Claude Code Setup

No additional setup needed beyond the MCP server. The built-in system prompt instructs the AI to call `get_context()` at session start.

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

`get_context()` assembles memory context in four prioritized tiers with **variable-resolution rendering** — seed entities and identity core get full detail (summary + attributes + 5 facts), hop-1 neighbors get summary detail (2 facts), and hop-2+ discoveries render as mentions only (name + type):

1. **Identity Core** (~200 tokens) — Always-included personal identity entities at full detail
2. **Project Context** (~400 tokens) — Entities relevant to current project, resolution varies by hop distance from query match
3. **Recent Activity** (~400 tokens) — Top-activated entities at summary detail
4. **Active Intentions** (~100 tokens) — Prospective memory intentions with warmth labels (cool / warming / warm / HOT)

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

User: "Remind me to check the XSS fix before we deploy the auth module"

Agent: [calls intend("auth module", "Check XSS fix before deploying",
        entity_names=["Auth Module"])]
  → Creates Intention entity + TRIGGERED_BY edge to Auth Module

... days later ...

User: "I'm working on the login flow today"

Agent: [calls remember] → Extraction finds Login Flow entity
  → Spreading activation: Login Flow → Auth Module → Intention fires
  → Agent surfaces: "Reminder: Check XSS fix before deploying the auth module"
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
| POST | `/api/knowledge/bootstrap` | Bootstrap project: create entity + observe key files (idempotent) |
| POST | `/api/knowledge/intentions` | Create a graph-embedded intention |
| GET | `/api/knowledge/intentions` | List intentions with warmth ratios |
| DELETE | `/api/knowledge/intentions/{id}` | Dismiss (soft/hard delete) an intention |
| POST | `/api/knowledge/chat` | SSE streaming chat with memory context |
| GET | `/api/graph/at` | Temporal subgraph at a point in time |
| POST | `/api/knowledge/replay-queue` | Replay offline capture queue (~/.engram/capture-queue.jsonl) |
| POST | `/api/consolidation/trigger` | Trigger consolidation cycle |
| GET | `/api/consolidation/status` | Consolidation status + pressure |
| GET | `/api/consolidation/history` | Cycle history |
| GET | `/api/consolidation/cycle/{id}` | Cycle detail with audit records |
| GET | `/api/conversations/` | List conversations (paginated) |
| POST | `/api/conversations/` | Create conversation |
| GET | `/api/conversations/{id}/messages` | Fetch conversation messages |
| POST | `/api/conversations/{id}/messages` | Append messages to conversation |
| PATCH | `/api/conversations/{id}` | Update conversation title |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| WS | `/ws/dashboard` | Real-time events (episodes, graph deltas, activation) |

### WebSocket

Connect to `/ws/dashboard` for real-time updates. Events include episode lifecycle, graph mutations, activation snapshots, and consolidation progress.

**Commands**: Send JSON messages to control subscriptions:
- `{"type": "ping"}` — Keepalive (server responds with pong). Auto ping/pong every 25s.
- `{"type": "resync", "since_seq": N}` — Replay missed events since sequence number N
- `{"type": "subscribe.activation_monitor"}` — Subscribe to periodic activation snapshots (top entities + decay curves)

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
2. `./.env` — local overrides (API keys, engine mode)
3. Environment variables — always take precedence

For first-time setup, run the wizard: `cd server && uv run python -m engram setup`

Key environment variables (or copy `.env.example` to `.env`):

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...          # Claude Haiku for entity extraction

# Optional
VOYAGE_API_KEY=pa-...                  # Voyage AI embeddings (optional, get key at dash.voyageai.com)
ENGRAM_EMBEDDING__PROVIDER=local       # local | voyage | noop (auto-fallback if no Voyage key)
ENGRAM_EMBEDDING__LOCAL_MODEL=nomic-ai/nomic-embed-text-v1.5  # fastembed model name
ENGRAM_GROUP_ID=default                # Your brain ID (one per person, not per project)
ENGRAM_MODE=auto                       # auto | lite | full

# Consolidation (standard by default)
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard   # off | observe | conservative | standard
ENGRAM_ACTIVATION__WORKER_ENABLED=true              # Background episode processor

# Recall intelligence (all waves by default)
ENGRAM_ACTIVATION__RECALL_PROFILE=all               # off | wave1 | wave2 | wave3 | all

# Prospective memory tuning (enabled by recall_profile=all)
ENGRAM_ACTIVATION__PROSPECTIVE_ACTIVATION_THRESHOLD=0.5  # Activation level to trigger an intention
ENGRAM_ACTIVATION__PROSPECTIVE_COOLDOWN_SECONDS=300      # Min seconds between fires (default 5 min)
ENGRAM_ACTIVATION__PROSPECTIVE_GRAPH_EMBEDDED=true       # Use v2 graph-embedded intentions (default)

# Graph embeddings (all ON by default, train during consolidation)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_NODE2VEC_ENABLED=true   # Node2Vec random walks (pure numpy)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_TRANSE_ENABLED=true     # TransE relational geometry (pure numpy)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_GNN_ENABLED=true        # GNN/GraphSAGE (requires torch)
ENGRAM_ACTIVATION__WEIGHT_GRAPH_STRUCTURAL=0.1             # Retrieval weight (0.0 = disabled)

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
| **Infer Validation** | Multi-signal scorer (default) or Claude Haiku 4.5 (fallback) | `consolidation_infer_auto_validation_enabled` | Validating inferred edges |
| **Merge Judge** | Multi-signal scorer (default) or Claude Haiku 4.5 (fallback) | `consolidation_merge_multi_signal_enabled` | Judging borderline entity merges |
| **Infer Escalation** | Claude Sonnet 4.6 | `consolidation_infer_escalation_model` | Re-validating uncertain edge verdicts (LLM fallback only) |
| **Merge Escalation** | Claude Sonnet 4.6 | `consolidation_merge_escalation_model` | Re-validating uncertain merge verdicts (LLM fallback only) |
| **Briefing** | Claude Haiku 4.5 | `briefing_model` | Synthesizing `get_context(format="briefing")` narrative |

All LLM features beyond basic extraction default to OFF. The `standard` consolidation profile enables all of them. Prompt caching is always active — static system prompts are cached via Anthropic's ephemeral cache, reducing input costs by ~80-90%.

## Security

- **Brain isolation**: Every query filters by `group_id` — one brain per person, hard-partitioned like RLS
- **Authentication**: Optional bearer token auth on all endpoints, with OIDC JWT support (Clerk-compatible, JWKS caching)
- **Encryption**: AES-256-GCM with per-tenant HKDF-SHA256 key derivation
- **Rate limiting**: Redis-backed sliding window per-tenant per-route (observe: 100/min, remember: 20/min, recall: 60/min, trigger: 2/hour); graceful fallback to unlimited when Redis unavailable
- **Usage metering**: Per-tenant API call and LLM token tracking with daily aggregation (90-day retention)
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
uv run pytest -m "not requires_docker" -v    # 1,354 tests
uv run ruff check .                           # Lint
uv run python -m engram.mcp.server            # MCP server (stdio)
uv run uvicorn engram.main:app --port 8100    # REST API
uv run python -m engram setup                 # Interactive setup wizard
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
  consolidation/    # 9-phase engine, scheduler, pressure accumulator
  embeddings/       # Embedding providers (Voyage AI cloud, fastembed local, noop)
  events/           # EventBus + Redis pub/sub bridge
  extraction/       # Entity extraction (Claude Haiku), predicate canonicalization, discourse classifier
  ingestion/        # CQRS ingestion paths
  mcp/              # MCP server (15 tools, 3 resources, 2 prompts)
  models/           # Pydantic data models
  retrieval/        # Pipeline, scorer, router, reranker, MMR
  security/         # Auth middleware, AES-256-GCM encryption, OIDC, rate limiting
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
