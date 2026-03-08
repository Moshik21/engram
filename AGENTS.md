# Engram — Development Guide

## Project Overview

Engram is a persistent memory layer for AI agents. It builds temporal knowledge graphs from conversations, uses ACT-R cognitive architecture for activation-aware retrieval, and runs offline consolidation cycles inspired by biological memory.

## Architecture

- **Dual mode**: SQLite (lite) / FalkorDB+Redis (full), auto-detected
- **Backend**: `server/engram/` (~110 Python files), `server/tests/` (~108 test files)
- **Frontend**: `dashboard/` (React 19 + TypeScript + Tailwind v4 + Zustand + Recharts + Three.js)
- **MCP server**: stdio transport, 15 tools, 3 resources, 2 prompts

### CQRS Split

Episode ingestion uses a CQRS pattern:

- `store_episode()` — Fast path. Creates QUEUED episode, no LLM call. Used by `observe` MCP tool.
- `project_episode()` — Slow path. Runs LLM extraction + entity resolution + embedding. Used by `remember` or background worker.
- `ingest_episode()` — Legacy wrapper that calls both sequentially.

### Memory Consolidation (12 phases)

```
triage → merge → infer → replay → prune → compact → mature → semanticize → schema → reindex → graph_embed → dream
```

- **Triage** (phase 0): Scores QUEUED episodes by heuristics, promotes top ~35% for extraction, skips the rest
- **Merge**: Fuzzy-matches duplicate entities (thefuzz + union-find + embedding ANN). Tiered scoring: Tier 0 multi-signal (name analysis + embeddings + neighbor Jaccard + summary Dice), Tier 1 cross-encoder refinement for uncertain cases, Tier 3 LLM fallback (opt-in). Handles acronyms, numeronyms, tech suffixes, canonical aliases deterministically
- **Infer**: Creates edges for co-occurring entities (PMI + tiered validation). Tier 0 multi-signal (embedding coherence, type compatibility, ubiquity penalty, structural plausibility), Tier 1 cross-encoder for uncertain, Tier 3 LLM opt-in. Self-correcting via Dream LTD
- **Replay**: Re-extracts recent episodes to find missed entities (selective — skips during tiered scheduling if no upstream graph changes)
- **Prune**: Soft-deletes dead entities (no relationships, low access count ≤ 2, older than 14 days). Activation safety net: entities with activation > 0.05 survive. Memory-tier-aware: semantic entities survive 180 days, transitional 2x episodic.
- **Compact**: Logarithmic bucketing of access history
- **Mature**: Graduates entities from episodic → transitional → semantic tier based on maturity score (source diversity, temporal span, relationship richness, access regularity). Identity core entities auto-promote. Reconsolidation count bonus.
- **Semanticize**: Promotes episodes from episodic → transitional → semantic tier based on entity coverage (fraction of linked entities that are mature) and consolidation cycle count.
- **Schema**: Detects recurring structural motifs (predicate-type fingerprints) and promotes them to first-class Schema entities. Instances connect via `INSTANCE_OF` edges. Reinforces existing schemas on re-detection. Enabled in standard profile.
- **Reindex**: Re-embeds entities affected by earlier phases
- **Graph Embed**: Trains structural graph embeddings (Node2Vec, TransE, GNN). Incremental retraining with 5% change threshold. TransE staggered every 3rd cycle, GNN every 5th.
- **Dream**: Offline spreading activation to strengthen pathways + dream associations (cross-domain creative connections). LTD decay for unboosted edges.

**Three-tier scheduling**: Hot (triage, 15min) → Warm (merge/infer/compact/mature/semanticize/reindex, 2hr) → Cold (replay/prune/schema/graph_embed/dream, 6hr). Manual/pressure/scheduled triggers bypass tiering and run all phases.

Controlled by `consolidation_profile`: off (default), observe (dry-run), conservative, standard.

### Background Worker

`EpisodeWorker` subscribes to `episode.queued` EventBus events and processes episodes in near-real-time:
- Scores content using same heuristics as triage phase
- High-score episodes get immediate extraction via `project_episode()`
- Low-score episodes are marked completed (stored but not extracted)
- Enabled via `worker_enabled=True` (set by consolidation profiles)

### System Prompt Strategy

The MCP system prompt biases toward `observe` (cheap store) by default:
- `observe` — Default for most content capture (general context, uncertain value)
- `remember` — Reserved for high-signal items (identity facts, explicit preferences, corrections)
- "If uncertain, observe it" — not "remember it"

## Key Technical Decisions

- Using `anthropic` Python SDK directly for extraction (not Vercel AI SDK) — all-Python backend
- Codex Haiku (`Codex-haiku-4-5-20251001`) for entity extraction
- Spreading activation does NOT record access (prevents phantom reinforcement loops)
- Activation is lazy — computed from access_history on read, never stored as decaying float
- Consolidation phases return `tuple[PhaseResult, list[AuditRecord]]` for engine to persist
- Entity resolution uses `resolve_entity_fast()` with indexed `find_entity_candidates()` (FTS5/CONTAINS)
- Dream spreading boosts edge weights (not spreading_bonus) — persistent and immediately effective
- Dream associations discover cross-domain entity pairs via embedding similarity, create temporary `DREAM_ASSOCIATED` edges with TTL
- `DREAM_ASSOCIATED` edges are excluded from Hebbian boosting (prevents dream drift) and exempt from cross-domain penalty in BFS/PPR
- `valid_to` temporal semantics: `(valid_to IS NULL OR datetime(valid_to) > datetime('now'))` — TTL edges visible until expiry
- Hybrid search: fts_weight=0.3, vec_weight=0.7 (RRF fusion default)
- Memory maturation: entities graduate episodic → transitional → semantic with differential ACT-R decay (0.5 → 0.3 exponent). Semantic entities survive pruning 180 days vs 14 days for episodic.
- Reconsolidation: recently recalled entities enter a 5-minute labile window where new information can update their summary (max 3 modifications). Window does NOT extend on re-recall.

## Commands

```bash
# Backend tests (lite mode, no Docker required)
cd server && uv run pytest -m "not requires_docker" -v

# Lint
cd server && uv run ruff check .

# Frontend tests
cd dashboard && pnpm test

# MCP server (stdio)
cd server && uv run python -m engram.mcp.server

# REST API (lite mode)
cd server && uv run uvicorn engram.main:app --port 8100

# Full stack (Docker)
docker compose up -d --build

# One-shot consolidation
cd server && uv run python -m engram.consolidation --profile observe
```

## Config

All activation/consolidation parameters are in `server/engram/config.py` (`ActivationConfig`). Key fields:

- `consolidation_profile`: **off** (default) | observe | conservative | standard
- `recall_profile`: **off** (default) | wave1 | wave2 | wave3 | wave4 | all
- Standard profile enables: triage, worker, dream associations, graph embeddings (Node2Vec), LLM merge/infer judges, pressure-triggered, three-tier scheduling
- Graph embeddings: `weight_graph_structural` (0.1), Node2Vec/TransE train with pure numpy, GNN requires PyTorch (`pip install engram[gnn]`)
- GNN and TransE are on standby — enabled but skip training until minimum thresholds are met (50 entities for Node2Vec, 100 triples for TransE, 200 entities for GNN)
- Prune defaults: `min_age_days=14`, `min_access_count=2`, `activation_floor=0.05`. Conservative profile: 30 days.
- Merge defaults: `threshold=0.88`, `use_embeddings=True`, `ann_llm_max=20`. ANN→LLM pipeline routes semantic duplicates to Haiku judge.

Environment variables: `ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard`, etc.
