<p align="center">
  <h1 align="center">Engram</h1>
  <p align="center">
    <strong>Long-term memory for AI agents.</strong><br>
    Stores what happened, organizes it, and brings back the right context later.
  </p>
</p>

<p align="center">
  <a href="https://engram-roan.vercel.app">Website</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#helixdb">HelixDB</a> &middot;
  <a href="#multimodal-memory">Multimodal</a> &middot;
  <a href="#mcp-integration">MCP Integration</a> &middot;
  <a href="#one-click-openclaw-install">OpenClaw</a> &middot;
  <a href="#dashboard">Dashboard</a> &middot;
  <a href="#api-reference">API</a> &middot;
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <a href="https://engram-roan.vercel.app"><img src="https://img.shields.io/badge/website-engram-8b5cf6" alt="Website"></a>
  <img src="https://github.com/Moshik21/engram/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/react-19-61dafb" alt="React 19">
  <img src="https://img.shields.io/badge/tests-pytest%20%2B%20vitest-blue" alt="Pytest + Vitest">
  <img src="https://img.shields.io/badge/license-Apache_2.0-green" alt="License">
</p>

---

AI agents are good at reasoning in the moment, but bad at remembering across sessions. Engram gives them long-term memory.

Engram stores conversations as episodes, turns important people, facts, and relationships into a temporal knowledge graph, ranks what matters with ACT-R activation, and runs background consolidation to merge duplicates, promote stable memories, and fade stale noise.

**In one sentence:** Engram gives AI agents a memory that persists across sessions, organizes itself, and retrieves the right context when needed.

**In 30 seconds:** Use `observe()` to save what happened quickly; when the cue layer is enabled, it immediately generates a deterministic latent memory trace that can be recalled before full extraction. Use `remember()` when something is important enough to extract immediately. Later, `recall()` brings back compact memory packets, cue-backed latent episodes, and raw supporting results instead of dumping an entire chat log. In the background, consolidation scores queued memories, merges duplicates, promotes stable patterns, and prunes low-value clutter.

**Key capabilities:**

- **Observe** conversations cheaply with optional cue-first latent memory and background processing that selectively extracts high-value content
- **Remember** critical facts with evidence-first extraction — high-confidence facts commit immediately, ambiguous structure is stored as deferred evidence for later promotion
- **Recall** with activation-aware retrieval, cue-backed latent episodes, planner-driven query bundles, and packetized memory shaped for the current turn
- **Consolidate** memory offline: triage queued episodes, merge duplicates, infer missing relationships, prune stale entities, strengthen associative pathways
- **Project progressively** by turning `observe`d text into `EpisodeCue` records first, then promoting hot episodes into targeted projection when recall demand justifies the cost
- **Intend** with prospective memory — graph-embedded intentions that fire automatically via spreading activation when related topics come up ("remind me when...")
- **Understand** personal, health, and emotional content with an expanded 17-type entity taxonomy and rich predicate vocabulary
- **Visualize** the knowledge graph and memory pipeline in real-time with a 3D neural brain dashboard plus cue/projection observability
- **Atlas** multi-scale graph exploration — zoom from high-level region clusters down to individual entity neighborhoods

## Key Concepts

| Term | Definition |
|------|-----------|
| **ACT-R** | Cognitive architecture model that scores memory relevance by recency and frequency of access |
| **Spreading Activation** | When you recall an entity, related entities also "light up" based on graph proximity |
| **Consolidation** | Background process that merges duplicates, discovers patterns, and prunes stale content (inspired by memory consolidation during sleep) |
| **Activation** | Real-time relevance score for a memory, computed from access history — not stored, always recomputed |
| **Episode** | A raw text input (conversation snippet) stored in the system |
| **Entity** | A person, place, concept, or thing extracted from episodes and stored in the knowledge graph |
| **Cue Layer** | Lightweight, deterministic memory trace stored alongside full extraction — enables latent recall without LLM |
| **CQRS Split** | Ingestion pattern: `observe` (fast, no LLM) vs `remember` (full extraction with LLM) |
| **Memory Tier** | Entities graduate from episodic → transitional → semantic as they mature (like biological memory) |
| **Labile Window** | 5-minute reconsolidation window after recall where entity summaries can be updated with new info |
| **Dream Phase** | Offline spreading activation that strengthens important connections and discovers cross-domain associations |

## Quickstart

### One-Click Install

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

The installer prompts you to choose a mode:

- **Helix native** (recommended) — HelixDB in-process via PyO3, no Docker, best quality + speed
- **Lite** — SQLite backend, no Docker needed, all features included
- **Helix HTTP** — HelixDB via Docker, graph+vector in one
- **Full** — FalkorDB + Redis via Docker, legacy high throughput

Skip the prompt with `bash -s -- lite` or `bash -s -- full`.

Lifecycle commands (both modes):

```bash
engramctl start
engramctl status
engramctl logs
engramctl stop
engramctl update
engramctl uninstall
```

Upgrade from lite to full when you outgrow SQLite: `engramctl upgrade` (starts
fresh graph store — SQLite data preserved on disk for reference).

Details: [`docs/install/lite.md`](docs/install/lite.md) | [`docs/install/full-docker.md`](docs/install/full-docker.md) | [`docs/install/helix.md`](docs/install/helix.md)

### One-Click OpenClaw Install

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw
```

Or install lite and say **yes** to "Install OpenClaw skill?" during setup.

Details: [`docs/install/openclaw.md`](docs/install/openclaw.md)

### Developer / Source Install

If you want to build from local source, use the repo-based workflow instead of
the public installer:

```bash
bash scripts/dev-install.sh
```

Or manually:

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram/server
uv sync
uv run engram setup
```

### Option 1: Native Mode (recommended — best quality, no Docker)

```bash
cd server
uv sync
make build-native         # Build PyO3 extension (one-time)
make mcp-native           # MCP server (stdio) with native HelixDB
# or: make up-native       # REST API with native HelixDB
```

This starts Engram with HelixDB running **in-process** via a Rust PyO3 binding
(`helix_native`). Zero network overhead, ~97ms search latency, no Docker
required. Same 167 compiled queries, same retrieval quality. Best nDCG@10 of all
modes (0.448). The setup wizard defaults to `consolidation_profile=standard`,
`recall_profile=all`, and `integration_profile=rework`.

### Option 2: Lite Mode (zero setup)

```bash
cd server
uv sync
uv run engram mcp        # MCP server (stdio)
# or: uv run engram serve  # REST API
```

This starts Engram in **lite mode** (zero infrastructure beyond Python) using
SQLite for storage. The setup wizard still defaults to the same recall-ready
posture: `consolidation_profile=standard`, `recall_profile=all`, and
`integration_profile=rework`.

### Option 3: Helix Mode (production, multi-service)

```bash
# Option A: Helix CLI
helix push dev

# Option B: Docker Compose
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required), optionally GEMINI_API_KEY or VOYAGE_API_KEY
make up-helix            # or: docker compose -f docker-compose.helix.yml up -d --build
```

This starts HelixDB (single service) with graph+vector+BM25 unified. Same dashboard
and API. Best retrieval quality. Details: [`docs/install/helix.md`](docs/install/helix.md).

### Option 4: Full Mode (FalkorDB + Redis)

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required), optionally GEMINI_API_KEY or VOYAGE_API_KEY
make up                  # or: docker compose up -d --build
```

This is the existing source-built compose flow for developers. The public
installer does **not** use this path; it downloads a release bundle and pulls
prebuilt images instead.

### Option 5: REST API Only

```bash
cd server
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run engram serve
```

## How It Works

### Storage Modes

Engram runs in four modes with identical APIs:

| Mode | Backend | Docker | Best For |
|------|---------|--------|----------|
| **Lite** | SQLite | None | Zero setup, single user |
| **Helix (native)** | HelixDB in-process (PyO3) | **None** | **Best quality + speed, recommended** |
| **Helix (HTTP)** | HelixDB Docker | 1 container | Production, multi-service |
| **Full** | FalkorDB + Redis | 2 containers | Legacy, high throughput |

Mode is auto-detected (Helix → FalkorDB+Redis → SQLite) or set explicitly via `ENGRAM_MODE=helix|full|lite`. Helix transport is selected via `ENGRAM_HELIX__TRANSPORT=native|http|grpc` (default: `http`).

### Architecture

| Layer | Lite Mode | Helix Native (PyO3) | Helix HTTP (Docker) | Full Mode (Docker) |
|-------|-----------|---------------------|--------------------|--------------------|
| Graph | SQLite (WAL, FTS5) | HelixDB in-process (HelixQL) | HelixDB (HelixQL) | FalkorDB (Cypher) |
| Activation | In-memory dict | In-memory dict | In-memory dict | Redis hashes (7-day TTL) |
| Search | FTS5 + numpy cosine | HelixDB native HNSW + BM25 | HelixDB native HNSW + BM25 | Redis Search HNSW (1024d) |
| Embeddings | Optional (Gemini / Voyage / local) | Optional (Gemini / Voyage / local) | Optional (Gemini / Voyage / local) | Optional (Gemini / Voyage / local) |
| Infrastructure | Zero | Zero | 1 Docker service | 2 Docker services |

### Data Persistence

All data persists across server restarts in all modes:

| Data | Lite Mode | Helix Native (PyO3) | Helix HTTP (Docker) | Full Mode |
|------|-----------|---------------------|---------------------|-----------|
| Knowledge graph | SQLite (`~/.engram/engram.db`) | HelixDB (`~/.engram/helix/`) | HelixDB (Docker volume `engram_helix_data`) | FalkorDB (Docker volume `engram_falkordb_data`) |
| Activation store | In-memory (rebuilt from access history) | In-memory (rebuilt from access history) | In-memory (rebuilt from access history) | Redis (Docker volume `engram_redis_data`) |
| Search index | SQLite FTS5 (same db file) | HelixDB native (same directory) | HelixDB native (same volume) | Redis Search (same Redis volume) |
| Consolidation history | SQLite (same db file) | SQLite (same db file) | SQLite (Docker volume `engram_server_data`) | SQLite (Docker volume `engram_server_data`) |
| Episode content | SQLite (same db file) | SQLite (same db file) | SQLite (Docker volume `engram_server_data`) | SQLite (Docker volume `engram_server_data`) |

In native mode, all data lives on the local filesystem with no Docker volumes. In full and helix HTTP modes, consolidation audit records and episode content are stored in a SQLite sidecar database at `/home/engram/.engram/engram.db` inside the server container. This is persisted via the `engram_server_data` Docker volume — cycle history, triage records, and audit trails survive container rebuilds.

To **reset all data**:

```bash
# Lite mode
rm ~/.engram/engram.db

# Helix mode (Docker) — WARNING: deletes everything
docker compose -f docker-compose.helix.yml down -v

# Full mode (Docker) — WARNING: deletes everything
docker compose down -v
```

### HelixDB

HelixDB is the recommended backend, unifying graph, vector, and full-text search in one engine. It is available in two transport modes:

- **Native mode (PyO3)**: HelixDB engine runs in-process via a Rust->Python binding (`helix_native`). Zero network overhead, ~97ms search latency. Same 167 compiled queries. No Docker required. This is the recommended transport. Build with `make build-native`.
- **HTTP mode**: Standard client-server over localhost. Good for production multi-service deployments. One Docker container replaces two (FalkorDB + Redis).

**Why HelixDB:**
- **Native graph algorithms** — BFS, Dijkstra shortest path, and spreading activation run server-side in HelixQL
- **Server-side vector search** — HNSW index with post-filtering, no separate vector store needed
- **Field-level BM25 indexing** — custom enhancement: only fields annotated with the `BM25` schema prefix are full-text indexed (e.g., `Episode.content`, `Entity.name`, `Entity.summary`), preventing metadata fields from diluting search relevance
- **Unified BM25 + vector** — full-text and semantic search in the same engine with RRF fusion
- **HelixQL query language** — 167 compiled queries in the schema (`server/engram/storage/helix/schema.hx`, ~1400 lines)
- **Best retrieval quality** — nDCG@10 of 0.448 (native) / 0.412 (HTTP) vs SQLite 0.390 and FalkorDB 0.406 (see [Benchmarks](#benchmarks))

**Getting started:**

```bash
# Option A: Native mode (recommended — no Docker)
make build-native         # Build PyO3 extension (one-time)
make mcp-native           # MCP server with native HelixDB
# or: make up-native       # REST API with native HelixDB

# Option B: Helix CLI (HTTP mode)
helix push dev

# Option C: Docker Compose (HTTP mode)
make up-helix

# Set mode explicitly (or let auto-detection find it)
ENGRAM_MODE=helix uv run engram serve
```

#### HDB Optimization Journey

HelixDB transport performance has been progressively optimized through four iterations:

| Iteration | Optimization | Impact |
|-----------|-------------|--------|
| **HDB-1** | Batch endpoint | 5-10x for multi-query ops |
| **HDB-2** | HTTP/2 support | Connection multiplexing |
| **HDB-3** | gRPC transport | Binary protocol, lower overhead |
| **HDB-4** | PyO3 native binding (`helix_native`) | ~97ms latency, zero Docker |

Configuration:

| Env Var | Default | Description |
|---------|---------|-------------|
| `ENGRAM_HELIX__HOST` | `localhost` | HelixDB host (HTTP/gRPC modes) |
| `ENGRAM_HELIX__PORT` | `6969` | HelixDB port (HTTP/gRPC modes) |
| `ENGRAM_HELIX__TRANSPORT` | `http` | Transport: `native` \| `http` \| `grpc` |

Details: [`docs/install/helix.md`](docs/install/helix.md)

### API Keys

Engram uses external APIs for two things: extracting structure from text and (optionally) embedding entities for semantic search.

#### Extraction Provider (auto-detected)

Engram auto-detects the best available extraction provider in priority order: **Anthropic** (Claude Haiku) -> **Ollama** (local LLM) -> **Narrow** (deterministic, zero LLM). The narrow pipeline uses staged regex-based extractors (`IdentityFactExtractor`, `AffiliationExtractor`, etc.) that require no API key and no LLM at all. Set explicitly via `ENGRAM_ACTIVATION__EXTRACTION_PROVIDER=auto|anthropic|ollama|narrow`.

```bash
# Option A: Anthropic (best quality)
export ANTHROPIC_API_KEY=sk-ant-...   # Get a key at console.anthropic.com

# Option B: Ollama (local LLM, no API key)
# Requires ollama running locally with a model pulled

# Option C: Narrow (deterministic, zero LLM, default fallback)
# No setup needed — works out of the box
```

**With Anthropic**: When you `remember` something (or when the background worker promotes an `observe`d episode), Engram sends the text to Claude Haiku (`claude-haiku-4-5-20251001`) to extract entities, relationships, temporal markers, structured attributes, and polarity. The extraction prompt recognizes 17 entity types (including health, emotional, and personal domains), ~73 predicate synonyms that canonicalize to ~25 semantic groups, and handles negation/uncertainty ("stopped using X" invalidates existing edges). This produces the highest-quality knowledge graph.

**Without Anthropic**: The narrow deterministic pipeline handles extraction using pattern matching and heuristics. Quality is lower than LLM extraction but sufficient for basic operation with zero API cost.

**Cost**: Claude Haiku is Anthropic's fastest, cheapest model. A typical extraction uses ~500-1,500 input tokens and ~200-800 output tokens. At Haiku's pricing (~$0.80/1M input, ~$4/1M output), that's roughly **$0.001-0.005 per memory extracted**. Engram is designed to minimize LLM usage:

- **Triage + merge + infer** use deterministic multi-signal scorers (zero API cost) — LLM judges are available as opt-in fallback but disabled by default in the `standard` profile
- **Observe + triage flow** means only ~35% of stored content triggers extraction
- **Background worker** uses three-tier confidence routing (extract/defer/skip) with zero LLM calls
- **Prompt caching** on all extraction/validation prompts — static system prompts cached at $0.10/M vs $1.00/M, reducing input costs by ~80-90% on repeated calls
- **Remaining LLM usage**: entity extraction (`remember` + promoted `observe`), consolidation replay (re-extraction), knowledge chat (optional), and briefing synthesis (cached)

#### Embeddings (optional)

Engram supports three embedding providers for semantic vector search, auto-detected in priority order: Gemini → Voyage → FastEmbed (local) → Noop. If none is configured, it falls back to keyword-only search.

**Option A: Gemini Embedding 2 (cloud, multimodal)**

```bash
export GEMINI_API_KEY=...   # Get a key at aistudio.google.com
```

Uses Google's [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/embeddings) (`gemini-embedding-2-preview`) — the first multimodal embedding model. Embeds text, images, audio, video, and PDFs into a unified 3072-dimensional vector space. Task-aware prefixing (`RETRIEVAL_DOCUMENT` for indexing, `RETRIEVAL_QUERY` for search) improves retrieval quality. Supports Matryoshka (MRL) — prefix-slice vectors to 256d for fast approximate comparisons without retraining. Free tier available.

Setting `GEMINI_API_KEY` automatically enables Gemini as the embedding provider. This also unlocks multimodal memory (see [Multimodal Memory](#multimodal-memory) below).

**Option B: Voyage AI (cloud)**

```bash
export VOYAGE_API_KEY=pa-...   # Get a key at dash.voyageai.com
```

Embeds each entity into a 1024-dimensional vector (`voyage-4-lite` model). Cost: ~$0.01 per 1M tokens embedded.

**Option C: Local embeddings (offline, private)**

```bash
pip install engram[local]     # or: uv sync --extra local
export ENGRAM_EMBEDDING__PROVIDER=local
```

Uses [Nomic Embed v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (768d, 137M params) via fastembed (ONNX runtime, CPU). The model (~130MB) is downloaded on first use and cached locally. No API key required.

**Auto-fallback**: Provider priority is Gemini → Voyage → FastEmbed (local) → Noop. If your configured provider's API key is missing, Engram falls to the next available provider. If none is available, vector search is disabled (keyword search still works).

**Without embeddings**: Engram still works — retrieval uses FTS5 keyword matching, ACT-R activation, and spreading activation. You lose semantic similarity but keep everything else.

| | Anthropic (recommended) | Gemini (optional) | Voyage AI (optional) | Local (optional) |
|---|---|---|---|---|
| **Purpose** | Entity extraction from text | Multimodal vector search | Semantic vector search | Semantic vector search |
| **Model** | Claude Haiku | gemini-embedding-2-preview (3072d) | voyage-4-lite (1024d) | Nomic Embed v1.5 (768d) |
| **When called** | `remember`, promoted `observe`, consolidation replay, knowledge chat | Entity creation + reindex + multimodal ingest | Entity creation + reindex | Entity creation + reindex |
| **Cost per call** | ~$0.001-0.005 | Free tier available | ~$0.00001 | Free (CPU) |
| **Multimodal** | No | Yes (text, images, audio, video, PDFs) | No | No |
| **Without it** | Falls back to Ollama or narrow deterministic extraction | Falls back to Voyage/local/keyword | Falls back to local/keyword | Falls back to keyword search |

#### Where LLM Is Used (and Where It Isn't)

Engram is designed to minimize LLM dependency. The only operation that *requires* an LLM is turning text into knowledge graph structure. Everything else — scoring, routing, merging, inferring, pruning, consolidating — is deterministic.

| Operation | Uses LLM? | What Happens |
|-----------|-----------|--------------|
| **`observe`** (store episode) | No | Episode stored in ~5ms. If `cue_layer_enabled`, Engram also builds an `EpisodeCue`, indexes cue text, and can route the episode to `cue_only` or `scheduled` before any LLM call. |
| **`remember`** (extract + store) | Depends | Extraction provider auto-detected: Anthropic (Claude Haiku) -> Ollama -> Narrow (deterministic, zero LLM). Extracts entities, relationships, attributes, temporal markers, and polarity through the projector/apply pipeline. |
| **`recall`** (retrieve memories) | No | Pure DB: FTS5/vector search, ACT-R activation, planner support, cue recall, spreading activation, re-ranking, packet assembly. Zero API calls. |
| **Triage** (episode scoring) | No* | 8-signal multi-signal scorer (~2ms/ep). *Optional LLM escalation remains available as an explicit opt-in fallback. |
| **Merge** (entity dedup) | No | 7-signal deterministic scorer + cross-encoder refinement + structural candidate discovery. |
| **Infer** (edge creation) | No | 6-signal deterministic scorer. Self-corrects via Dream LTD decay. |
| **Replay** (deferred extraction) | No | Runs deferred extraction on triage-skipped episodes, then links known entity names found in episode text. Skips already-extracted episodes. Zero LLM calls. |
| **Knowledge chat** | Yes | Agentic Haiku loop with tool calls (recall, search_entities, search_facts). Rate-limited. |
| **Briefing synthesis** | Yes | Haiku summarizes memory context into 2-3 sentences. Cached with prompt caching. |
| **Dream** (offline consolidation) | No | Spreading activation + embedding similarity. Pure math. |
| **Graph embeddings** | No | Node2Vec, TransE, GNN — all trained with numpy/torch locally. |

**Cost in practice**: With the `standard` profile, a typical active user generates ~$0.50-2.00/day in Haiku costs, primarily from entity extraction. Consolidation scoring (triage, merge, infer) adds $0 — it's all deterministic. Knowledge chat adds ~$0.10-0.50/day depending on usage (rate-limited to 10 requests/minute).

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

The system prompt biases the LLM toward `observe` for most content, reserving `remember` for high-signal items (identity facts, explicit preferences, corrections). The worker's three-tier confidence routing means obvious decisions are made immediately (no LLM), while uncertain episodes are deferred to the triage phase for batch scoring by the deterministic multi-signal scorer.

### Progressive Projection

> **Status**: The cue layer and progressive projection are behind feature flags (`cue_layer_enabled`, `projector_v2_enabled`). When `integration_profile=rework` is set, all flags are enabled together. The design is documented in [`docs/design/extraction-rework.md`](docs/design/extraction-rework.md).

When enabled, the extraction pipeline becomes **cue-first** rather than binary:

1. `store_episode()` writes the raw episode without an LLM call.
2. Engram generates an `EpisodeCue` immediately: a deterministic retrieval tag with cue text, salient spans, mention candidates, contradiction hints, and projection priority. No LLM needed.
3. Recall can surface that cue-backed latent memory right away (`result_type="cue_episode"`) even if the episode has never been fully projected.
4. Repeated cue hits, selected/used feedback, or high-priority routing can promote the episode to `scheduled`.
5. `project_episode()` then uses a deterministic span planner plus a typed projector/apply pipeline to project only the most relevant spans first.

This creates three usable memory layers instead of the original binary (raw text vs. full graph):

| Layer | What It Is | Cost | Recallable? |
|-------|-----------|------|-------------|
| **Raw Episode** | Stored text with timestamps | ~5ms | FTS only |
| **Cue Memory** | Deterministic retrieval cues, embeddings, salience, mention spans | ~10ms | Yes (cue search) |
| **Graph Memory** | Extracted entities, relationships, consolidated structure | LLM call | Yes (full pipeline) |

The middle layer is the key architectural addition — it makes `observe()` materially more useful without paying full projection cost. Episodes that matter surface through recall demand, not just ingestion-time scoring.

### Evidence-First Extraction

> **Status**: The evidence pipeline and commit policy are behind `remember_v2_enabled` and related flags. Design: [`docs/design/remember-and-extractor-v2.md`](docs/design/remember-and-extractor-v2.md).

When `remember()` is called, extraction no longer treats LLM output as immediate graph truth. Instead, it produces **evidence candidates** that pass through a commit policy:

| Confidence | Action | Example |
|-----------|--------|---------|
| **>= 0.85** | Commit immediately | "Alice works at Acme Corp" → durable graph fact |
| **0.50 – 0.84** | Defer as unresolved evidence | "She reminded me about the dentist" → stored, not committed |
| **< 0.50** | Reject | Ambiguous pronouns, hedged future states |

Deferred evidence is not lost — it remains attached to the episode and can be promoted later by:
- Explicit user restatement
- Corroborating episodes
- Consolidation replay
- Optional offline LLM adjudication (batch, never on the hot path)

**Staged narrow extractors** handle specific domains with high precision instead of one monolithic LLM call:
- `IdentityFactExtractor` — "my name is X", "I work at Y"
- `AffiliationExtractor` — "X works at Y", "X is part of Y"
- `PreferenceGoalHabitExtractor` — "I like X", "my goal is Y"
- `CorrectionExtractor` — "actually...", "no longer...", "moved to..."
- `TemporalSignalExtractor` — explicit dates, "since X", "last month"

**Client proposals**: Capable callers (e.g., Claude Code) can pass `proposed_entities` and `proposed_relationships` alongside `remember()`. These are validated and scored — never blindly trusted — but can bypass redundant extraction when precise enough.

The north star: **LLM-not-required memory quality**. The system commits only high-confidence facts immediately, lets latent memory (cues + deferred evidence) handle everything else, and refines over time through consolidation.

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

### Multimodal Memory

Engram supports multimodal episodes — images, audio, video, and PDF attachments alongside text. When the Gemini Embedding 2 provider is active, all modalities are embedded into a unified 3072-dimensional vector space, enabling cross-modal search (text queries find image episodes and vice versa).

**MCP tools:**

| Tool | Purpose |
|------|---------|
| `observe_image` | Store an image with optional text description; embeds via Gemini for cross-modal recall |
| `observe_file` | Store a file (PDF, audio, video) with optional text description |
| `remember` | Supports `image` parameter for image-augmented memory extraction |

**REST API:**

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/knowledge/observe-image` | Store image episode with multimodal embedding |
| POST | `/api/knowledge/observe-file` | Store file episode (PDF, audio, video) with multimodal embedding |

**How it works:**
- Attachments are embedded alongside text using Gemini Embedding 2's unified vector space
- Text queries find image/audio/video episodes through shared embedding geometry
- Image queries find related text episodes (and vice versa)
- Standard retrieval pipeline (ACT-R activation, spreading activation, reranking) applies to multimodal results
- Requires `GEMINI_API_KEY` to be set (Gemini Embedding 2 is the only provider supporting multimodal inputs)

### One Brain Per Person

Engram is a brain, not a database. Each person gets **one** Engram instance — all their projects, personal life, health, goals, and conversations live in a single knowledge graph. Projects aren't separate partitions; they're natural entity clusters connected by topology.

This means:
- Memories from work **can** inform personal context (and vice versa)
- Dream associations **can** discover cross-domain creative connections
- Spreading activation reaches across project boundaries through shared concepts
- Identity core entities (name, preferences, health) are always available, regardless of which project you're in

**`group_id`** provides hard isolation between different people — like Row Level Security. It is not for separating projects. If you're the only user, the default (`"default"`) is all you need.

**Why one brain?** Context windows are expensive, flat, and temporary. Memory is selective, layered, and durable. The future of AI assistants depends on building continuity over weeks, months, and years — and that requires a single persistent substrate where work knowledge can inform personal context and vice versa. See [`docs/vision/one-brain-per-person.md`](docs/vision/one-brain-per-person.md) and [`docs/vision/science-of-engram.md`](docs/vision/science-of-engram.md) for the scientific and strategic rationale.

**Federated learning (roadmap):** While each brain is fully private, anonymized aggregate signals (schema patterns, activation curves, triage calibration, graph topology) can be shared across Engram instances to improve the system for everyone — like neuroscience studying many brains without reading anyone's thoughts.

### ACT-R Activation

Memory retrieval is based on the [ACT-R cognitive architecture](https://en.wikipedia.org/wiki/ACT-R). Each entity has an activation level computed lazily from its access history:

```
B_i(t) = ln(consolidated_strength + Σ (t - t_j)^(-0.5))
```

Entities accessed recently and frequently have higher activation. This is normalized via sigmoid to [0, 1] and combined with other signals for retrieval scoring.

### Retrieval Pipeline

The retrieval scorer still does the heavy lifting once a query exists. When Engram executes the underlying retrieval pipeline:

1. **Search** — FTS5 + optional vector search, top-50 candidates
2. **Route** — Classify query type (temporal, frequency, creation, associative, direct) and adjust scoring weights
3. **Activate** — Batch-fetch ACT-R activation states
4. **Spread** — BFS or PPR spreading activation through the graph (2 hops, fan-based dampening; `DREAM_ASSOCIATED` edges exempt from cross-domain penalty)
5. **Enrich** — Compute similarity for entities discovered via spreading
6. **Score** — Composite: `0.40 × semantic + 0.25 × activation + 0.15 × spreading + 0.15 × edge_proximity + graph_structural + exploration` (with hop_distance tracked per result)
7. **Rerank** — Optional cross-encoder (Cohere) + MMR diversity filter
8. **Return** — Top-10 results with score breakdowns including hop_distance

### Memory Consolidation

Engram runs offline consolidation cycles inspired by biological memory consolidation during sleep. The consolidation pipeline is an active area of improvement — see [`docs/design/consolidation-rework.md`](docs/design/consolidation-rework.md) for the full rework plan covering correctness fixes, store parity, and the neuro-symbolic decision stack.

**Sleep stage mapping**: Phases map to biological sleep stages — triage parallels pre-sleep encoding, replay mirrors NREM3 slow-wave replay, merge maps to NREM2 spindle-driven binding, and dream corresponds to REM creative association.

Twelve phases execute sequentially:

| Phase | What It Does |
|-------|-------------|
| **Triage** | Score QUEUED episodes with 8-signal multi-signal scorer (~2ms/ep, zero LLM). Filter system meta-commentary. Extract top ~35%, skip the rest. Optional LLM escalation remains available only as an explicit fallback. |
| **Merge** | Fuzzy-match duplicate entities (thefuzz + union-find + embedding ANN + structural candidate discovery). Multi-signal scorer with 7 signals (name analysis + embeddings + neighbor Jaccard + summary Dice + referential exclusivity) — handles acronyms, numeronyms, tech suffixes, canonical aliases, and structural equivalents (entities sharing neighbors but with zero name overlap). Summary dedup via token-set Jaccard prevents bloat during merge. |
| **Infer** | Create edges for co-occurring entities (PMI scoring). Multi-signal auto-validation (embedding coherence + type compatibility + ubiquity penalty + structural plausibility) replaces LLM judge — self-correcting via Dream LTD decay |
| **Replay** | Run deferred extraction on triage-skipped episodes (CUE_ONLY/QUEUED), then link known entity names found in episode text via exact substring matching. Skips already-extracted (PROJECTED) episodes — deterministic re-extraction is waste. Zero LLM calls. Selective: only runs when upstream phases changed the graph during tiered scheduling. |
| **Prune** | Soft-delete dead entities (no relationships, low access, old enough). Activation safety net prevents pruning warm entities. |
| **Compact** | Logarithmic bucketing of access history + consolidated strength preservation |
| **Mature** | Promote entities through three memory tiers (episodic → transitional → semantic) based on source diversity, temporal span, relationship richness, and access regularity. Each tier has differential ACT-R decay (0.5 exponent for episodic, 0.3 for semantic) and different pruning resistance (14 days episodic, 180 days semantic). Identity-core entities auto-promote. **Reconsolidation**: recently recalled entities enter a 5-minute labile window where new information can update their summary (max 3 modifications; window does not extend on re-recall). Reconsolidation count feeds maturation bonus. |
| **Semanticize** | Promote episodes from episodic → transitional → semantic tiers based on mature-entity coverage and consolidation cycle count. Same-cycle maturations are visible immediately, so `semanticize` does not need to wait for a later consolidation pass to see newly mature entities. |
| **Schema** | Detect recurring structural motifs, create `Schema` entities for them, and connect matching instances with `INSTANCE_OF` edges. Fingerprints are canonicalized, candidate motifs are biased toward mature/stable support, and promoted schemas record support summaries and reasons. |
| **Reindex** | Re-embed entities affected by earlier phases |
| **Graph Embed** | Train structural graph embeddings (Node2Vec, TransE, GNN) for topology-aware retrieval. Node2Vec supports a true dirty-subgraph incremental path with warm-started vectors. TransE and GNN are staggered, but currently retrain the full graph when they run. |
| **Dream** | Offline spreading activation to strengthen associative pathways + discover cross-domain creative connections. **LTP/LTD**: Boosted edges strengthen (predicate-aware); unboosted edges decay 0.005/cycle (floor 0.1). **Dream associations**: Embedding similarity (70% text + 30% graph) discovers cross-domain entity pairs, creates temporary `DREAM_ASSOCIATED` edges (weight 0.1, 30-day TTL, excluded from Hebbian boosting). Accessed dream edges extend TTL by 30 days. Repeated validation can graduate a dream edge to permanent `RELATED_TO`. |

#### Three-Tier Scheduling

Phases run at different frequencies based on urgency:

| Tier | Phases | Default Interval |
|------|--------|-----------------|
| **Hot** | triage | 15 minutes |
| **Warm** | merge, infer, compact, mature, semanticize, reindex | 2 hours |
| **Cold** | replay, prune, schema, graph_embed, dream | 6 hours |

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

#### Known Limitations

The consolidation rework ([`docs/design/consolidation-rework.md`](docs/design/consolidation-rework.md)) identified issues being addressed:

- **Incremental graph embedding** is implemented for Node2Vec only; TransE and GNN currently retrain the full graph when they run
- **FalkorDB parity** — late-phase methods (mature, semanticize) have incomplete FalkorDB implementations; episode tier fields may not fully round-trip in full mode
- **Merge negative cache** (`keep_separate`) is in-memory without TTL — stale decisions can persist until restart
- **Replay vocab linking** — uses exact substring matching only; fuzzy/partial entity name matches are not detected
- **Schema formation** does not yet prefer mature entities over episodic ones when selecting candidate motifs

### Multi-Signal Scoring Architecture

All consolidation decisions — triage, merge, and infer — use **deterministic multi-signal scorers** instead of LLM API calls. These scorers use strictly more information than an LLM judge (which only sees text) — combining embeddings, graph structure, entity candidates, and statistics. Zero API cost, <5ms latency, deterministic results, with self-improving calibration.

Mutable consolidation phases also emit structured `DecisionTrace` records plus outcome labels. Each cycle persists distillation examples and rolling calibration snapshots, and the cycle-detail API surfaces those artifacts for debugging, offline evaluation, and future scorer retraining.

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

**Merge scorer** (7 signals, weighted ensemble):

| Signal | Weight | What It Catches |
|--------|--------|----------------|
| Name analysis (fuzzy + acronym + numeronym + suffix strip + alias table) | 0.35 | JS↔JavaScript, K8s↔Kubernetes, React↔React.js |
| Embedding cosine similarity | 0.25 | Semantic equivalence beyond surface names |
| Neighbor Jaccard overlap | 0.15 | Same graph context = same entity |
| Summary Dice coefficient | 0.10 | Shared description terms |
| Referential exclusivity | 0.15 | Never co-occur in episodes + shared neighbors = same entity. Frequent co-occurrence = anti-merge penalty (prevents merging siblings). |
| Type compatibility | gate | Person↔Technology = never merge |
| Booster rules | override | Structural equivalence (shared neighbors ≥ 0.40 + never co-occur + embedding ≥ 0.50 → auto-merge), Person name variants, high-confidence combos |

**Three candidate discovery paths** feed into the scorer:
1. **Name-based** — fuzzy string matching with type-blocking and prefix sub-blocks
2. **Embedding ANN** — cosine similarity pre-filtering on entity embeddings
3. **Structural** — inverted neighbor index finds entity pairs sharing ≥3 graph neighbors, regardless of name similarity. Catches semantic duplicates like "Fourth Son" ↔ "Benjamin" that share parent/sibling relationships.

**Summary dedup** — during merge, incoming sentences are compared to existing sentences via token-set Jaccard similarity (≥0.6 = duplicate). Prevents summary bloat from repeated observations.

**Infer scorer** (6 signals, weighted ensemble):

| Signal | Weight | What It Catches |
|--------|--------|----------------|
| Embedding coherence | 0.30 | Semantically related entities |
| Type compatibility | 0.20 | Domain-aware noise filtering |
| Statistical confidence (PMI) | 0.20 | Non-random association strength |
| Ubiquity penalty | 0.15 | Filters entities that appear everywhere |
| Structural plausibility (triangle closure) | 0.10 | Shared neighbors = plausible edge |
| Graph embedding similarity | 0.05 | Structural proximity (Node2Vec/TransE) |

**Tiered architecture** — all consolidation decisions (triage, merge, infer) flow through tiers until resolved:

| Tier | Method | Coverage | Latency | Cost |
|------|--------|----------|---------|------|
| **Tier 0** | Multi-signal rules (triage 8 signals, merge 7 signals, infer 6 signals) | ~85% | <5ms | $0 |
| **Tier 1** | Cross-encoder refinement (`Xenova/ms-marco-MiniLM-L-6-v2`, already loaded) | ~10% | ~50ms | $0 |
| **Tier 2** | Self-improving calibration (online logistic regression, triage only) | built-in | <1ms | $0 |
| **Tier 3** | LLM escalation (opt-in, borderline cases only) | ~5% | ~500ms | API cost |

Uncertain cases from Tier 0 are refined by the cross-encoder (Tier 1), which blends its score with the multi-signal score. The triage scorer additionally feeds extraction outcomes back to an online calibrator (Tier 2) that learns which signal combinations predict successful extraction. Remaining uncertain infer edges self-correct via Dream LTD decay (unused edges weaken) and Hebbian reinforcement (useful edges strengthen). LLM judges remain available as opt-in fallback via `triage_llm_judge_enabled`, `consolidation_merge_llm_enabled`, and `consolidation_infer_llm_enabled`.

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

All three methods are enabled by default and train during consolidation cycles. A 5% entity change threshold decides whether the cycle can stay on its incremental path. Today that incremental path is implemented for Node2Vec only: it expands the dirty entity set into a bounded local subgraph and warm-starts from existing vectors. TransE and GNN are still staggered across cycles to reduce compute load, but when they run they currently retrain the full graph. Each method has a minimum threshold — methods automatically skip training until the graph is large enough, then activate:

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
| `off` | Basic explicit `recall()` only — no automatic or proactive retrieval. Explicit recall still returns packets plus raw results. |
| `wave1` | **AutoRecall + Natural Need Analysis** — Piggybacks on `observe`/`remember`, runs the need analyzer before firing, enables pragmatic + structural signals, primes session context on first call, and records surfaced-vs-used recall semantics. |
| `wave2` | + **Graph Grounding + Recall Planning** — Adds graph resonance for borderline turns, planner-driven multi-intent recall, session entity seeds for spreading activation, rolling topic fingerprinting, and near-miss detection. |
| `wave3` | + **Shift / Impoverishment + Proactive Intelligence** — Shift detection and impoverishment modeling enter live gating, while proactive features add topic-shift bursts, surprise connection detection, retrieval priming, and graph-connected MMR re-ranking. |
| `wave4` | + **Prospective Memory** — Graph-embedded intentions that fire via spreading activation when related entities light up. Create with `intend()`, monitor warmth in `get_context()`. |
| `all` | Alias for `wave4` today — enable every recall wave. |

Each wave includes all previous waves. Set via env var: `ENGRAM_ACTIVATION__RECALL_PROFILE=wave4` or `ENGRAM_ACTIVATION__RECALL_PROFILE=all`

Packet assembly is enabled on recall surfaces by default (`recall_packets_enabled=True`). Recall profiles only control the recall side of the system. They do not turn on cue generation, cue policy learning, or projection promotion by themselves.

### Integration Profile

Use `integration_profile` when you want the three reworks to behave as one loop instead of enabling subsystems independently:

| Profile | What It Enables |
|---------|----------------|
| `off` | Leave consolidation, recall, and cue/projection rollout flags independent. Useful for partial rollouts and subsystem testing. |
| `rework` | Normalize to `consolidation_profile=standard` and `recall_profile=all`, then enable `cue_layer`, `cue_recall`, `cue_policy_learning`, the projector v2/planner path, maturation/episode transitions, and the full live natural recall stack (structural, graph grounding, shift, impoverishment). This is the coherent recall-ready preset. |

Recommended full-loop config: `ENGRAM_ACTIVATION__INTEGRATION_PROFILE=rework`
If you only set `recall_profile=all` or `consolidation_profile=standard`, that is still a partial rollout. The cue layer, cue-policy loop, and projection-promotion path stay off unless you enable them individually or use `integration_profile=rework`.

### Epistemic Routing and Answer Contracts

> **Status**: Enabled by `integration_profile=rework`. Design: [`docs/design/epistemic-routing.md`](docs/design/epistemic-routing.md), [`docs/design/answer-contract-resolver.md`](docs/design/answer-contract-resolver.md).

When a question could depend on stored memory, project artifacts, or current runtime state, Engram routes it through an epistemic resolver before answering. The `route_question` tool (or `POST /api/knowledge/route`) returns:

- **Routing mode**: `remember` (use stored memory), `inspect` (check project artifacts), or `reconcile` (compare both)
- **Answer contract**: How to shape the response — one of six operators:

| Operator | When Used | Behavior |
|----------|----------|----------|
| `direct_answer` | Single clear source | Answer directly from the strongest evidence |
| `compare` | Multiple scopes may disagree | Show raw defaults vs. install defaults vs. repo posture vs. runtime state |
| `reconcile` | Memory and artifacts may conflict | Preserve temporal distinction: "we discussed X, but the repo currently reflects Y" |
| `timeline` | Decision evolved over time | Show the progression: discussed → decided → documented → implemented |
| `recommend` | User wants advice | State evidence first, then give advice |
| `plan` | User wants next steps | State evidence, then outline a plan |

- **Claim states**: Facts are annotated as `discussed`, `tentative`, `decided`, `documented`, `implemented`, or `effective` — so the system can distinguish "we talked about using Redis" from "Redis is currently deployed and running."
- **Decision graph edges**: `DECIDED_IN`, `DOCUMENTED_IN`, `IMPLEMENTED_BY` track how decisions externalize from conversation to code.
- **Required next sources**: The router tells agents which sources to consult (`memory`, `artifacts`, `runtime`) before forming a final answer.

Artifact bootstrap (`bootstrap_project`) indexes key project files (README, config, design docs) as `Artifact` entities so they're available for routed answers.

### Recall Control Loop

With `wave1+`, Engram's recall path becomes need-first rather than query-first:

1. **Analyze need** — Decide whether the current turn likely needs memory using five signal families (details below). Outputs a `need_type` and confidence score.
2. **Plan recall** (`wave2+`) — Add graph grounding, seed detected entities, and build a small recall plan with bounded intents such as `direct`, `topic`, and `session_entity`.
3. **Retrieve evidence** — Run the normal activation-aware pipeline (search, activation, spreading, scoring, reranking).
4. **Assemble packets** — Shape the result into `fact`, `state`, `timeline`, `open_loop`, `intention`, `episode`, or cue-backed packets.
5. **Deliver by surface** — Auto-recall returns compact packets, explicit recall returns packets plus raw results, and knowledge chat tools return packet summaries to the model.
6. **Record usage** — Distinguish `surfaced`, `selected`, `used`, `dismissed`, and `corrected` interactions so passive surfacing does not reinforce memory like real use.

`wave3+` also lets shift detection and impoverishment modeling participate in live gating, so recall can fire for natural follow-ups even when the user never says "remember" or "what did we say".

#### Five Signal Families

The memory-need analyzer uses five orthogonal signal families organized in two layers. Design: [`docs/design/pragmatic-recall-signals.md`](docs/design/pragmatic-recall-signals.md), [`docs/design/natural-need-signals-build.md`](docs/design/natural-need-signals-build.md).

**Layer 1 — Linguistic signals (<2ms, always runs):**

| Family | What It Detects | Examples | Wave |
|--------|----------------|---------|------|
| **Pragmatic** | Cross-session anaphora, bare names, possessive+relational nouns, hedged asides | "my son's school called", "she loved the gift", "still dealing with that bug" | wave1 |
| **Temporal-Relational** | 18 structural patterns: callbacks, life updates, corrections, status checks, memory gaps, milestones, etc. | "he had a great game today", "back to the drawing board on auth" | wave1 |
| **Shift** | 5-channel domain boundary detection: lexical field, register, discourse markers, pronoun ratio, structural changes | "anyway, about my garden..." (topic pivot triggers memory load for new domain) | wave3 |
| **Impoverishment** | Predicts whether a response would be generic without memory: conversational move type, affect with personal stakes, template test | "sorry, kid stuff" (casual aside presupposes shared knowledge) | wave3 |

**Layer 2 — Graph resonance (<8ms, conditional on Layer 1 score > 0.15):**

| Family | What It Does | Wave |
|--------|-------------|------|
| **Graph Probe** | Token-to-entity index lookup, relational noun resolution ("son" → CHILD_OF), batch property query, per-entity resonance scoring (density + activation + urgency + tier bonus) | wave2 |

**Decision rule**: `linguistic_score >= 0.30` → recall. `linguistic >= 0.15 AND resonance >= 0.45` → recall. Otherwise skip. Cost asymmetry is 7:1 (false negatives hurt trust more than false positives waste tokens).

**Key insight — politeness inversion**: "Sorry, kid stuff" presupposes MORE shared knowledge than a full explanation. The less someone explains, the more they assume you know. Casual asides with referents get a confidence boost.

Only true-usage interactions (`used`, `confirmed`, or explicit recall that records access) reinforce ranking feedback. `surfaced`, `selected`, and `dismissed` are telemetry/selection signals, not retrieval reinforcement events.

### Episode And Cue State Debugging

When `integration_profile=rework` is enabled, each observed episode moves through a small projection-state machine. This is the simplest way to reason about how extraction, recall, and consolidation mesh.

| Episode `projection_state` | What it means | Cue behavior |
|---------|----------------|--------------|
| `queued` | Stored and awaiting worker/triage handling. | Cue may not exist yet if cue generation is off. |
| `cued` | A latent cue exists and can be recalled, but the episode is not scheduled for extraction yet. | Cue is searchable and accumulates surfaced/selected/used telemetry. |
| `cue_only` | The episode stays latent unless later feedback or routing promotes it. Common for low-priority or system-like content. | Cue remains searchable but is intentionally not on the immediate projection path. |
| `scheduled` | The worker or cue policy decided the episode should be projected next. | Cue stays searchable until projection completes. |
| `projecting` | Extraction/projection is currently running. | Cue metadata remains attached to the episode. |
| `projected` | Entities/relationships were extracted, linked, and indexed; raw episode recall is now available too. | Cue remains as provenance/debug state and reflects projection completion. |
| `failed` | Projection attempted and failed. | Cue remains available for retry or later promotion. |
| `merged` | The episode was retired into another episode during adjacent-turn batching. | Cue is retired/suppressed and should not recall. |

Cue promotion rules in the integrated loop:

1. `observe()` stores the episode immediately and, if the cue layer is enabled, creates a deterministic cue right away.
2. `surfaced` means the cue was shown; it increments cue telemetry only.
3. `selected` means the caller chose the cue; it can update cue policy, but it still does not count as true memory usage.
4. `used` or `confirmed` means the memory actually informed the answer; this can promote a latent cue to `scheduled` and it records true usage.
5. `dismissed` is neutral for retrieval reinforcement and lowers cue promotion pressure.
6. `corrected` applies negative retrieval feedback without pretending the memory was used successfully.

Where to inspect rollout state:

- `/api/episodes` exposes episode `status`, `projectionState`, `lastProjectionReason`, `lastProjectedAt`, plus cue counters/policy timestamps when a cue exists.
- `/api/knowledge/recall` exposes cue `projectionState`, `routeReason`, `hitCount`, `policyScore`, and `lastFeedbackAt` on cue-backed recall results.
- `/api/stats` aggregates cue coverage, cue-to-projection conversion, projection state counts, and projection yield.

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

**Enable:** Set `ENGRAM_ACTIVATION__RECALL_PROFILE=wave4` or `ENGRAM_ACTIVATION__RECALL_PROFILE=all`. The v2 graph-embedded path is used by default (`prospective_graph_embedded=True`).

### Background Worker

When enabled (any profile except `off`), the `EpisodeWorker` runs as a background task that processes `observe`d content in near-real-time:

1. Subscribes to `episode.queued` events on the EventBus
2. Also processes `episode.projection_scheduled` events emitted by cue-hit promotion and policy feedback
3. Filters system meta-commentary via discourse classifier (activation scores, pipeline terms, entity IDs)
4. Scores each episode with the multi-signal scorer (8 signals, ~2ms, zero LLM calls)
5. **Three-tier confidence routing:**
   - **High confidence** (>0.70): Extract immediately (same pipeline as `remember`)
   - **Mid confidence** (0.15–0.70): Defer to triage phase for batch scoring next cycle
   - **Low confidence** (<0.15): Store without extraction (still searchable via FTS)

This means obvious high-value content is extracted within seconds, obvious noise is skipped instantly, uncertain content gets a more thorough evaluation during the next triage cycle, and cue-backed episodes can be promoted later when recall pressure proves they matter — all without a single LLM API call.

## MCP Integration

Engram exposes 19 MCP tools for AI agents:

| Tool | Purpose |
|------|---------|
| `observe` | Store raw text cheaply; optionally generate a cue-backed latent memory for background processing and later recall |
| `remember` | Store a memory with immediate entity extraction (for high-signal content) |
| `recall` | Retrieve relevant memories using activation-aware search; returns packets plus raw scored results |
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
| `route_question` | Epistemic routing — decides whether a question needs memory, artifact inspection, or runtime state, and returns an answer contract |
| `bootstrap_project` | Auto-observe key project files and create a Project entity (idempotent) |
| `adjudicate_evidence` | Resolve ambiguous entity or relationship evidence |
| `search_artifacts` | Search bootstrapped project artifacts (README, design docs, config) |
| `get_runtime_state` | Check effective mode, active profiles, and enabled flags |

Plus 3 resources (`engram://graph/stats`, `engram://entity/{id}`, `engram://entity/{id}/neighbors`) and 2 prompts (`engram_system`, `engram_context_loader`).

### Automatic Memory Behavior

Engram ships with built-in MCP instructions that teach compatible AI agents (Claude, Cursor, Windsurf, etc.) to use memory proactively — no user prompting required:

- **Session start**: The agent calls `get_context()` before its first response to load relevant memories
- **Auto-observe**: For general conversation context and uncertain-value content, the agent calls `observe()` (cheap, no LLM; cue-backed when the cue layer is enabled)
- **Auto-remember**: For high-signal content (identity facts, explicit preferences, key decisions), the agent calls `remember()` (evidence-first extraction by default)
- **Auto-recall**: With `wave1+`, piggyback recall first runs a memory-need analyzer, then surfaces compact packets only when prior context is likely useful
- **Corrections**: When you correct a previously stored fact, the agent calls `forget()` on the old information then `remember()` with the correction

The system prompt biases toward `observe` by default — "if uncertain whether something is worth remembering, use observe." This reduces extraction cost while the background worker and triage phase ensure high-value content still gets fully extracted.

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
- Use `recall()` when the user references past conversations or when context would help.
- Use `search_facts()` for user-facing relationship lookups. Internal epistemic decision/artifact edges are hidden unless you explicitly opt into debug behavior.
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

> You're talking with Alex, who is building Engram — a persistent memory system for AI agents. Recently they've been focused on memory consolidation and dream associations.

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

The real-time dashboard provides a 3D neural brain visualization of your knowledge graph, plus 8 views for exploring and monitoring memory:

| View | Description |
|------|-------------|
| **Atlas** | Multi-scale graph exploration — zoom from high-level region clusters to individual neighborhoods. Regions form automatically from graph topology. |
| **Graph** | 3D force-directed neural brain — nodes pulse on activation, edges glow on recall, entity types are color-coded |
| **Timeline** | Temporal navigation — view the graph at any historical point |
| **Feed** | Episode ingestion history with extraction details |
| **Activation** | ACT-R leaderboard with decay curve visualization |
| **Stats** | Entity counts, type distribution, growth timeline, cue coverage, projection health, and extraction yield observability |
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
| GET | `/api/stats` | Graph statistics plus cue/projection observability metrics |
| GET | `/api/activation/snapshot` | Top activated entities |
| GET | `/api/activation/{id}/curve` | ACT-R decay curve |
| POST | `/api/knowledge/observe` | Store content without extraction (fast path) |
| POST | `/api/knowledge/auto-observe` | Auto-observe with classification (dedup, tagging) |
| POST | `/api/knowledge/remember` | Ingest with full extraction |
| GET | `/api/knowledge/recall` | Activation-aware memory search (packets + raw results) |
| GET | `/api/knowledge/facts` | Search user-facing facts/relationships (`include_epistemic=true` for debug graph edges) |
| GET | `/api/knowledge/context` | Assembled memory context (structured or briefing) |
| POST | `/api/knowledge/route` | Deterministic epistemic routing plus `answerContract`, `requiredNextSources`, and source-query metadata |
| GET | `/api/knowledge/artifacts/search` | Search bootstrapped project artifacts with supporting claims |
| GET | `/api/knowledge/runtime` | Effective mode/profile/feature state plus artifact freshness |
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
- `{"type": "command", "command": "resync", "lastSeq": N}` — Replay missed events since sequence number `N`
- `{"type": "command", "command": "subscribe.activation_monitor", "interval_ms": 2000}` — Subscribe to periodic activation snapshots
- `{"type": "command", "command": "unsubscribe.activation_monitor"}` — Stop activation snapshots

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

### Backend Comparison

| Metric | Lite (SQLite) | Native (PyO3) | Helix (HTTP) | Full (FalkorDB) |
|--------|:---:|:---:|:---:|:---:|
| **nDCG@10** | 0.390 | **0.448** | 0.412 | 0.406 |
| **MRR** | 0.762 | **0.832** | 0.815 | 0.665 |
| **Precision@10** | 0.310 | **0.372** | 0.356 | 0.396 |
| **Search avg (ms)** | 161 | **238** | 425 | 421 |
| **Infrastructure** | 0 services | **0 services** | 1 service | 2 services |

Native PyO3 mode achieves the best retrieval quality across nDCG@10 and MRR while requiring zero infrastructure — HelixDB runs in-process with no Docker. The 238ms search latency is nearly 2x faster than HTTP mode (425ms) since it eliminates all network overhead. Lite mode remains the fastest per-query but with lower retrieval quality.

### Results (1K entities, with embeddings)

| Method | P@5 | MRR | Best Category |
|--------|-----|-----|---------------|
| Full Stack | 0.395 | 0.773 | Frequency (0.94) |
| Multi-Pool | 0.388 | 0.751 | Frequency (0.94) |
| Pure Search | 0.307 | 0.801 | Direct (0.33) |

Full pipeline with spreading activation shows +28% P@5 improvement over pure search. Frequency queries (identifying most-accessed entities) are the standout at 94% precision — this is where ACT-R activation shines.

**Query categories**: direct lookup, recency, frequency, associative, temporal, semantic, graph traversal, cross-cluster. **Core script metrics**: P@5, R@10, MRR, nDCG@5, latency percentiles, bootstrap CI (1,000 resamples).

### LongMemEval (ICLR 2025)

Engram is benchmarked against [LongMemEval](https://arxiv.org/abs/2407.05045), the standard long-term memory evaluation suite from ICLR 2025. The benchmark tests whether a memory system can answer questions about a user's past conversations across 6 categories: single-session (user, assistant, preference), multi-session, temporal reasoning, and knowledge update.

**Methodology**: Claude Code connects to Engram via MCP, calls `recall` to retrieve evidence, then answers. The baseline stuffs all conversation history into context with no retrieval. Both use Claude Sonnet 4.6 on the same 30 stratified questions (5 per type). Evaluation uses hybrid embedding containment + token overlap judging. No LLM judge calls.

| | Baseline (context stuffing) | Engram (MCP recall) |
|---|---|---|
| **Accuracy** | 96.7% (29/30) | **100% (30/30)** |
| **Tokens per question** | ~8,400 | **~2,200** |
| **Total tokens** | ~257K | **~66K** |
| **Token reduction** | -- | **74%** |

The baseline failed one single-session-preference question — Claude had the answer in context but couldn't find it buried in 40K+ chars. Engram's retrieval found it instantly.

**Scaling**: Context stuffing is O(N) — every query pays the cost of the entire history. Engram is O(1) — retrieval cost is constant regardless of memory size. At 1,000 conversations, stuffing requires ~2M tokens per query (exceeds any context window). Engram still returns in ~2,200 tokens.

| System | Accuracy |
|---|---|
| **Engram + Sonnet 4.6** | **100.0%** |
| Baseline (Sonnet 4.6 + full context) | 96.7% |
| Observational Memory (gpt-5-mini) | 94.9% |
| EmergenceMem Internal | 86.0% |
| Observational Memory (gpt-4o) | 84.2% |
| Oracle GPT-4o | 82.4% |
| Supermemory | 81.6% |
| Zep/Graphiti | 71.2% |
| Full-context GPT-4o | 60.2% |
| Naive RAG | 52.0% |

> **Note**: Results are from a 30-question stratified sample (5 per type). Full 500-question benchmark run pending. Published baselines use the LongMemEval_S variant with GPT-4o reader; Engram uses the oracle variant with Sonnet 4.6.

**Raw results**: [`server/results/longmemeval_agent_sdk_cal.json`](server/results/longmemeval_agent_sdk_cal.json) (Engram) and [`server/results/longmemeval_baseline_v2.json`](server/results/longmemeval_baseline_v2.json) (baseline).

**Run it yourself**:

```bash
cd server

# Baseline (Claude Code CLI, Max subscription)
uv run python scripts/benchmark_baseline.py run \
    --dataset data/longmemeval/longmemeval_oracle.json \
    --n-per-type 5 --output results/baseline.json --verbose

# Engram (Claude Code CLI + Engram MCP, Max subscription)
uv run python scripts/benchmark_agent_sdk.py run \
    --dataset data/longmemeval/longmemeval_oracle.json \
    --n-per-type 5 --output results/engram.json --verbose
```

### Recall Behavior Metrics

The benchmark module also includes Phase 6 evaluation primitives for recall behavior:

- `memory_need_precision` — of turns that triggered recall, how often memory actually helped
- `useful_packet_rate` and `false_recall_rate` — how often surfaced packets were used vs misleading
- `session_continuity_lift` — score lift on multi-turn tasks that require prior context
- `open_loop_recovery_rate` and `temporal_correctness` — whether Engram surfaces unresolved work and newer facts at the right moment
- Echo chamber surfaced-vs-used tracking — the echo chamber benchmark now records surfaced count, used count, and surfaced-to-used ratio to catch passive reinforcement loops

## Configuration

Engram uses Pydantic Settings with env var support. Config is loaded in order (later sources override earlier):

1. `~/.engram/.env` — global config (created by `python -m engram setup`)
2. repo-root `.env` — useful when launching from `server/` inside this repository
3. `./.env` — current-working-directory overrides
4. Environment variables — always take precedence

For first-time setup, run the wizard: `cd server && uv run python -m engram setup`

Key environment variables (or copy `.env.example` to `.env`):

```bash
# Optional — Extraction (auto-detects: Anthropic → Ollama → Narrow deterministic)
ANTHROPIC_API_KEY=sk-ant-...          # Claude Haiku for entity extraction (best quality; narrow fallback works without it)

# Optional — Embeddings (priority: Gemini → Voyage → local → noop)
GEMINI_API_KEY=...                     # Google Gemini API key for multimodal embeddings (aistudio.google.com)
VOYAGE_API_KEY=pa-...                  # Voyage AI embeddings (optional, get key at dash.voyageai.com)
ENGRAM_EMBEDDING__PROVIDER=auto        # auto | gemini | voyage | local | noop (auto-detects by API key priority)
ENGRAM_EMBEDDING__LOCAL_MODEL=nomic-ai/nomic-embed-text-v1.5  # fastembed model name

# Optional — General
ENGRAM_GROUP_ID=default                # Your brain ID (one per person, not per project)
ENGRAM_MODE=auto                       # auto | lite | helix | full

# Optional — HelixDB connection (for helix mode)
ENGRAM_HELIX__HOST=localhost           # HelixDB host (default: localhost)
ENGRAM_HELIX__PORT=6969                # HelixDB port (default: 6969)
ENGRAM_HELIX__TRANSPORT=http           # Transport: native | http | grpc (default: http)

# Consolidation (off by default in code; Docker Compose defaults to standard)
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard   # Set to enable: off | observe | conservative | standard
ENGRAM_ACTIVATION__WORKER_ENABLED=true              # Optional override; non-off profiles enable worker automatically

# Full rework integration (recommended for MCP and Docker)
ENGRAM_ACTIVATION__INTEGRATION_PROFILE=rework       # Normalizes to consolidation_profile=standard + recall_profile=all, enables cue/projection rollout flags, the live natural recall stack, and epistemic routing/artifact bootstrap

# Recall intelligence (advanced / partial-rollout overrides)
ENGRAM_ACTIVATION__RECALL_PROFILE=all               # off | wave1 | wave2 | wave3 | wave4 | all
ENGRAM_ACTIVATION__RECALL_NEED_ANALYZER_ENABLED=true   # Advanced override; enabled by wave1+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_NEED_STRUCTURAL_ENABLED=true # Advanced override; enabled by wave1+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_NEED_GRAPH_PROBE_ENABLED=true # Advanced override; enabled by wave2+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_NEED_SHIFT_ENABLED=true      # Advanced override; enabled by wave3+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_NEED_IMPOVERISHMENT_ENABLED=true # Advanced override; enabled by wave3+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_NEED_SHIFT_SHADOW_ONLY=false    # Defaults true when shift is manually enabled
ENGRAM_ACTIVATION__RECALL_NEED_IMPOVERISHMENT_SHADOW_ONLY=false # Defaults true when manually enabled
ENGRAM_ACTIVATION__RECALL_PLANNER_ENABLED=true         # Advanced override; enabled by wave2+ or integration_profile=rework
ENGRAM_ACTIVATION__RECALL_PACKETS_ENABLED=true      # Return packets on recall surfaces (default true)
ENGRAM_ACTIVATION__RECALL_PACKET_AUTO_LIMIT=2       # Auto-recall packet cap
ENGRAM_ACTIVATION__RECALL_PACKET_EXPLICIT_LIMIT=3   # Explicit recall packet cap
ENGRAM_ACTIVATION__RECALL_PACKET_CHAT_LIMIT=2       # Chat tool packet cap
ENGRAM_ACTIVATION__RECALL_USAGE_FEEDBACK_ENABLED=true  # surfaced/selected/used semantics (enabled by wave1+ or integration_profile=rework)
ENGRAM_ACTIVATION__RECALL_NEED_ADAPTIVE_THRESHOLDS_ENABLED=false  # Optional bounded runtime tuning; off by default
ENGRAM_ACTIVATION__RECALL_NEED_GRAPH_OVERRIDE_ENABLED=false       # Optional graph-only recall override; off by default
ENGRAM_ACTIVATION__RECALL_NEED_POST_RESPONSE_SAFETY_NET_ENABLED=false # Optional one-shot knowledge-chat retry; off by default

# Epistemic routing / artifact substrate (enabled by integration_profile=rework)
ENGRAM_ACTIVATION__EPISTEMIC_ROUTING_ENABLED=true        # Route questions as remember | inspect | reconcile
ENGRAM_ACTIVATION__ARTIFACT_BOOTSTRAP_ENABLED=true       # Bootstrap key project docs/config into Artifact entities
ENGRAM_ACTIVATION__ARTIFACT_RECALL_ENABLED=true          # Let artifacts participate in routed answers
ENGRAM_ACTIVATION__EPISTEMIC_RUNTIME_EXECUTOR_ENABLED=true  # Surface effective runtime/config state
ENGRAM_ACTIVATION__DECISION_GRAPH_ENABLED=true           # Track discussed -> documented -> implemented decision edges
ENGRAM_ACTIVATION__EPISTEMIC_RECONCILE_ENABLED=true      # Reconcile memory with artifacts/runtime
ENGRAM_ACTIVATION__ANSWER_CONTRACT_ENABLED=true          # Shape answers as direct_answer | compare | reconcile | timeline | recommend | plan
ENGRAM_ACTIVATION__CLAIM_STATE_MODELING_ENABLED=true     # Annotate claims as discussed | tentative | decided | documented | implemented | effective
ENGRAM_ACTIVATION__ARTIFACT_BOOTSTRAP_STALE_SECONDS=86400  # Artifact refresh cadence

# Routed answers distinguish multiple truth scopes when needed:
# - raw config defaults      (code/config defaults)
# - shipped install defaults (setup wizard, README, .env.example)
# - repo current posture     (bootstrapped artifacts)
# - effective runtime        (current running mode/profile/flags)

# Progressive projection / cue layer (advanced / partial-rollout overrides)
ENGRAM_ACTIVATION__CUE_LAYER_ENABLED=true              # Generate EpisodeCue records on observe/store
ENGRAM_ACTIVATION__CUE_VECTOR_INDEX_ENABLED=true       # Index cue text for vector search when cue layer is on
ENGRAM_ACTIVATION__CUE_RECALL_ENABLED=true             # Let cue-backed latent episodes participate in recall
ENGRAM_ACTIVATION__CUE_POLICY_LEARNING_ENABLED=true    # Use surfaced/selected/used/near-miss feedback
ENGRAM_ACTIVATION__TARGETED_PROJECTION_ENABLED=true    # Allow long episodes to use span-selected projection
ENGRAM_ACTIVATION__PROJECTOR_V2_ENABLED=true           # Enable the progressive planner/projector path
ENGRAM_ACTIVATION__PROJECTION_PLANNER_ENABLED=true     # Deterministic span planner before extractor calls

# Prospective memory tuning (enabled by recall_profile=wave4 or all)
ENGRAM_ACTIVATION__PROSPECTIVE_ACTIVATION_THRESHOLD=0.5  # Activation level to trigger an intention
ENGRAM_ACTIVATION__PROSPECTIVE_COOLDOWN_SECONDS=300      # Min seconds between fires (default 5 min)
ENGRAM_ACTIVATION__PROSPECTIVE_GRAPH_EMBEDDED=true       # Use v2 graph-embedded intentions (default)

# Graph embeddings (enabled by default; train when consolidation runs)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_NODE2VEC_ENABLED=true   # Node2Vec random walks (pure numpy)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_TRANSE_ENABLED=true     # TransE relational geometry (pure numpy)
ENGRAM_ACTIVATION__GRAPH_EMBEDDING_GNN_ENABLED=true        # GNN/GraphSAGE (requires torch)
ENGRAM_ACTIVATION__WEIGHT_GRAPH_STRUCTURAL=0.1             # Retrieval weight (0.0 = disabled)

# LLM fallback configuration (all default to OFF; enable explicitly if desired)
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
| **Extraction** | Auto: Anthropic (Haiku 4.5) -> Ollama -> Narrow (deterministic) | `extraction_provider` | `remember`, promoted `observe`, replay |
| **Triage Judge** | Claude Haiku 4.5 | `triage_llm_judge_model` | Scoring queued episodes (replaces heuristics) |
| **Infer Validation** | Multi-signal scorer (default) or Claude Haiku 4.5 (fallback) | `consolidation_infer_auto_validation_enabled` | Validating inferred edges |
| **Merge Judge** | Multi-signal scorer (default) or Claude Haiku 4.5 (fallback) | `consolidation_merge_multi_signal_enabled` | Judging borderline entity merges |
| **Infer Escalation** | Claude Sonnet 4.6 | `consolidation_infer_escalation_model` | Re-validating uncertain edge verdicts (LLM fallback only) |
| **Merge Escalation** | Claude Sonnet 4.6 | `consolidation_merge_escalation_model` | Re-validating uncertain merge verdicts (LLM fallback only) |
| **Briefing** | Claude Haiku 4.5 | `briefing_model` | Synthesizing `get_context(format="briefing")` narrative |

All LLM features beyond basic extraction default to OFF. The `standard` consolidation profile keeps deterministic multi-signal scoring on by default; turn on the `...LLM...` flags above only if you want LLM fallback or escalation. Prompt caching is always active — static system prompts are cached via Anthropic's ephemeral cache, reducing input costs by ~80-90%.

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
- Docker (optional, for helix or full mode)

### Commands

```bash
# Public local install
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash     # Full Docker product
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw
engramctl status
engramctl update

# Backend
cd server
uv run pytest -m "not requires_docker" -v    # Backend test suite
uv run ruff check .                           # Lint
uv run python -m engram.mcp.server            # MCP server (stdio)
uv run uvicorn engram.main:app --port 8100    # REST API
uv run python -m engram setup                 # Interactive setup wizard
uv run python -m engram config                # Edit configuration

# Frontend
cd dashboard
pnpm install && pnpm dev                      # Dev server (port 5173)
pnpm test                                     # Frontend test suite
pnpm build                                    # Production build

# Docker (developer/source full mode) — via Makefile
make up                                       # Build + start full stack (standard consolidation + rework integration)
make down                                     # Stop everything
make restart                                  # Stop + rebuild + start
make logs                                     # Tail all logs (make logs-server for server only)
make status                                   # Container status + health check
make clean                                    # Stop + delete volumes (WARNING: deletes data)

# Native mode (PyO3 — no Docker, recommended)
make build-native                             # Build PyO3 extension (one-time, requires Rust toolchain)
make up-native                                # Start server with native HelixDB
make mcp-native                               # Start MCP with native HelixDB
make patch-helix                              # Re-apply HDB fork changes after helix push

# Docker (HelixDB HTTP mode) — single-service graph+vector backend
make up-helix                                 # Build + start Helix stack (HelixDB + server + dashboard)
make down-helix                               # Stop Helix stack
make restart-helix                            # Rebuild and restart
make logs-helix                               # Tail Helix stack logs
make mcp-helix                                # MCP server (streamable HTTP, connects to Docker Helix)
```

### Project Structure

```
server/engram/
  activation/       # ACT-R engine (BFS, PPR, strategy pattern)
  api/              # REST endpoints + WebSocket
  benchmark/        # Deterministic benchmark framework
  consolidation/    # 15-phase engine, scheduler, pressure accumulator
  embeddings/       # Embedding providers (Gemini multimodal, Voyage AI cloud, fastembed local, noop)
  events/           # EventBus + Redis pub/sub bridge
  extraction/       # Entity extraction (Claude Haiku), predicate canonicalization, discourse classifier
  ingestion/        # CQRS ingestion paths
  mcp/              # MCP server (19 tools, 3 resources, 2 prompts)
  models/           # Pydantic data models
  retrieval/        # Pipeline, scorer, router, reranker, MMR
  security/         # Auth middleware, AES-256-GCM encryption, OIDC, rate limiting
  storage/          # HelixDB, SQLite, FalkorDB, Redis implementations
  worker.py         # Background episode processor (EventBus-driven)

dashboard/src/
  components/       # Graph renderers, dashboard panels, and knowledge chat UI
  store/            # Zustand slices and selectors
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

**Public installer prerequisites fail**
The one-click installer requires Docker and Docker Compose. If Docker is not
installed, the installer exits with exact next-step instructions instead of
trying to provision it automatically. After Docker is running, retry:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

**"Connection refused" when MCP connects to Docker services**
The MCP server runs on your host machine, connecting to Docker via mapped ports. FalkorDB maps `6380→6379`, Redis maps `6381→6379`. Make sure your MCP env uses `localhost:6380` (FalkorDB) and `localhost:6381` (Redis), not the Docker-internal port 6379.

## License

Engram is licensed under **Apache 2.0** — see [LICENSE](LICENSE).

### Third-Party License Notice: HelixDB

HelixDB is licensed under **AGPL-3.0**. How this affects you depends on which transport you use:

| Transport | License impact | Default? |
|-----------|---------------|----------|
| **HTTP / gRPC** (Docker) | **No impact** — HelixDB runs as a separate service. Engram stays Apache 2.0. | Yes (default) |
| **Native (PyO3)** | **AGPL applies** — HelixDB is linked into the Python process. Your deployment must comply with AGPL-3.0 terms. | No (opt-in) |

**If you use the default HTTP or gRPC transport**, HelixDB is a separate process communicating over the network. This is the same as using PostgreSQL or Redis — no license contamination. Your code and any extensions you build remain under whatever license you choose.

**If you use native mode** (`make build-native`), the HelixDB engine is compiled into a Python extension module and runs in-process. Under AGPL-3.0, this may require you to make your source code available if you provide the software as a network service. This is fine for personal use, self-hosted deployments, and open-source projects. For commercial/proprietary use with native mode, contact the [HelixDB team](https://github.com/HelixDB/helix-db) about commercial licensing options.

The Lite (SQLite) and Full (FalkorDB + Redis) backends have no AGPL concerns.
