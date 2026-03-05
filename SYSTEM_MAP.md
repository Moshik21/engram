# Engram System Map -- Integration Analysis for Brain-Like Enhancements

**Date:** 2026-03-05
**Purpose:** Foundation document for implementing 6 brain-like features:
1. Inhibitory spreading activation
2. Memory systems taxonomy (episodic->semantic transition)
3. Emotional salience scoring
4. State-dependent retrieval
5. Reconsolidation on recall
6. Schema formation

---

## 1. Config Architecture

### Current Structure

`EngramConfig` (Pydantic BaseSettings) at `/server/engram/config.py` (847 lines):

- **Root level:** `mode` (lite/full/auto), `default_group_id`
- **Sub-configs (separate BaseModel classes):**
  - `ServerConfig` (3 fields)
  - `SQLiteConfig` (2 fields)
  - `FalkorDBConfig` (6 fields)
  - `RedisConfig` (3 fields)
  - `PostgreSQLConfig` (3 fields)
  - `EmbeddingConfig` (11 fields)
  - `AuthConfig` (7 fields)
  - `RateLimitConfig` (5 fields)
  - `EncryptionConfig` (2 fields)
  - `CORSConfig` (2 fields)
  - **`ActivationConfig` (~200 fields)** -- THE critical config class

### ActivationConfig Field Groups (~200 fields total)

| Group | Approx Fields | Key Examples |
|-------|--------------|--------------|
| ACT-R base | 10 | `decay_exponent`, `B_mid`, `B_scale`, `min_age_seconds` |
| Score weights | 6 | `weight_semantic` (0.40), `weight_activation` (0.25), `weight_spreading` (0.15), `weight_edge_proximity` (0.15), `weight_graph_structural` (0.1) |
| Spreading activation | 6 | `spread_max_hops`, `spread_decay_per_hop`, `spread_energy_budget` |
| Spreading strategy | 1 | `spreading_strategy` (bfs/ppr/actr) |
| PPR params | 4 | `ppr_alpha`, `ppr_max_iterations` |
| Thompson Sampling | 4 | `ts_enabled`, `ts_weight` |
| Fan-based spreading | 2 | `fan_s_max`, `fan_s_min` |
| ACT-R spreading | 2 | `actr_total_w`, `actr_max_sources` |
| RRF fusion | 2 | `rrf_k`, `use_rrf` |
| Community spreading | 4 | `community_spreading_enabled`, `community_bridge_boost` |
| Multi-pool candidates | 8 | `multi_pool_enabled`, pool sizes per source |
| Re-ranker | 4 | `reranker_enabled`, `reranker_provider` (cohere/local/noop) |
| MMR diversity | 2 | `mmr_enabled`, `mmr_lambda` |
| Implicit feedback | 2 | `feedback_enabled`, `feedback_ttl_days` |
| Context gating | 2 | `context_gating_enabled`, `context_gate_floor` |
| Structure-aware embeddings | 3 | `structure_aware_embeddings`, `predicate_natural_names` |
| Working memory | 4 | `working_memory_enabled`, capacity, TTL, seed energy |
| Episode retrieval | 3 | `episode_retrieval_enabled`, weight, max |
| Typed edge weighting | 2 | `predicate_weights` (dict), `predicate_weight_default` |
| Consolidation core | 6 | `consolidation_profile`, enabled, interval, dry_run |
| Three-tier scheduling | 4 | `consolidation_tiered_enabled`, hot/warm/cold seconds |
| Merge phase | 8+4 | threshold, max, block_size, ANN embedding params, LLM |
| Prune phase | 4 | floor, min_age, min_access, max_per_cycle |
| Infer phase | 8+4+3 | co-occurrence, PMI, LLM validation, escalation |
| Compact phase | 3 | horizon, keep_min, logarithmic |
| Replay phase | 4 | enabled, max, window, min_age |
| Reindex phase | 1 | max_per_cycle |
| Dream phase | 10 | enabled, seeds, floor/ceiling, weight increment |
| Dream LTD | 3 | enabled, decay, min_weight |
| Dream associations | 11 | enabled, surprise, TTL, domain pair cap |
| Pressure triggering | 6 | enabled, threshold, weights per signal |
| Triage phase | 7 | enabled, ratio, min_score, personal boost, LLM judge |
| Identity core | 2 | enabled, predicates list |
| Cross-domain penalty | 3 | enabled, factor, `domain_groups` dict |
| Context tiers | 3 | identity/project/recency budgets |
| Briefing | 4 | enabled, model, cache TTL, max tokens |
| Recall profile | 1 | `recall_profile` (off/wave1-4/all) |
| Worker | 1 | `worker_enabled` |
| AutoRecall | 9 | enabled, limit, score, cooldown, rate limit, session prime |
| Conversation awareness (Wave 2) | 10 | fingerprint, multi-query, session entity seeds, near-miss |
| Topic shift (Wave 3) | 3 | enabled, threshold, recall boost |
| Surprise detection (Wave 3) | 5 | enabled, floors, dormancy, cache TTL |
| Retrieval priming (Wave 3) | 5 | enabled, top_n, boost, TTL, max_neighbors |
| Prospective memory (Wave 4) | 9 | enabled, thresholds, TTL, fires, warmth levels |
| Graph embeddings | 12 | Node2Vec/TransE/GNN params, stagger intervals |
| GC-MMR (Wave 3) | 4 | enabled, lambda weights |

### Profile Mechanism

Two profile fields use `model_post_init` to set field bundles:

1. **`consolidation_profile`** (off/observe/conservative/standard): Sets consolidation-related flags in bulk. Each profile enables progressively more features. The `standard` profile enables LLM features (guarded by `ANTHROPIC_API_KEY` check).

2. **`recall_profile`** (off/wave1/wave2/wave3/wave4/all): Cumulative -- each wave enables all previous waves' features.

### Pattern for Adding New Config Fields

1. Add field to `ActivationConfig` with `Field(default=..., ge=..., le=...)` validation
2. Use `bool` for feature gates (e.g., `emotional_salience_enabled: bool = Field(default=False)`)
3. Group related fields with comment headers (e.g., `# --- Emotional Salience ---`)
4. Add to profile presets in `model_post_init` if appropriate
5. Environment variable pattern: `ENGRAM_ACTIVATION__FIELD_NAME=value`
6. All fields have `extra = "forbid"` -- no unknown fields allowed

### CRITICAL CONSTRAINT: Score Weight Budget

The 5 main scoring weights in `score_candidates()` are:
```
weight_semantic (0.40) + weight_activation (0.25) + weight_spreading (0.15)
+ weight_edge_proximity (0.15) + weight_graph_structural (0.1) = 1.05
```
Plus additive bonuses: exploration, rediscovery, context boost, priming boost. Adding emotional_salience or state_dependent signals requires either: (a) adding as new additive bonus, or (b) rebalancing existing weights. Option (a) is safer for backwards compatibility.

---

## 2. Schema Architecture

### Current Tables (`/server/engram/storage/sqlite/schema.sql`)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `entities` | Knowledge graph nodes | id, name, entity_type, summary, attributes (JSON), group_id, activation_base/current, access_count, last_accessed, deleted_at |
| `relationships` | Typed edges | id, source_id, target_id, predicate, weight, valid_from, valid_to, source_episode, group_id |
| `episodes` | Raw text inputs | id, content, source, status, group_id, session_id, created_at |
| `episode_entities` | Episode-entity junction | episode_id, entity_id (PK) |
| `entities_fts` | FTS5 virtual table | name, summary, entity_type (content-synced) |
| `episodes_fts` | FTS5 virtual table | content (content-synced) |
| `intentions` | Prospective memory (Wave 4) | id, trigger_text, action_text, trigger_type, entity_name, threshold, max_fires, fire_count, enabled, group_id |
| `conversations` | Knowledge tab chat | id, group_id, title, session_date |
| `conversation_messages` | Chat messages | id, conversation_id, role, content, parts_json |
| `conversation_entities` | Conversation-entity junction | conversation_id, entity_id |
| `graph_embeddings` | Structural embeddings | id, group_id, embedding (BLOB), dimensions, method, model_version |

### Indexes

- `entities`: name, type, group_id
- `relationships`: source_id, target_id, predicate, group_id
- `episodes`: group_id
- `intentions`: group_id, entity_name
- `conversations`: group_id + session_date
- `conversation_messages`: conversation_id
- `conversation_entities`: entity_id
- `graph_embeddings`: group_id, method

### Schema Change Pattern

The codebase uses **schema.sql with `IF NOT EXISTS`** -- no formal migration framework. New tables/columns are added by:
1. Adding `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` to schema.sql
2. For new columns on existing tables: **ALTER TABLE** in `initialize()` with try/except (used by consolidation_store.py pattern)
3. The `attributes` JSON blob on entities is the escape hatch for entity-level metadata without schema changes

### Schema Changes Needed Per Feature

| Feature | Schema Changes |
|---------|---------------|
| 1. Inhibitory spreading | **None** -- purely algorithmic change in `spreading.py` |
| 2. Memory taxonomy | New `entity_maturity` column on entities (or use attributes JSON). Possible new `semantic_facts` table for graduated facts. New `memory_type` column on episodes (episodic/semantic/procedural). |
| 3. Emotional salience | New `emotional_valence` REAL + `emotional_arousal` REAL on entities and/or episodes. Possibly `emotional_tags TEXT` (JSON array). |
| 4. State-dependent retrieval | New `context_state` TEXT on episodes (JSON: mood, time_of_day, project, etc.). New `state_embeddings` table or column for context vectors. |
| 5. Reconsolidation | New `reconsolidation_count` INTEGER + `last_reconsolidated` TEXT on entities. New `reconsolidation_records` audit table. |
| 6. Schema formation | New `schemas` table (id, name, description, member_entity_ids, prototype_embedding, group_id). New `entity_schema_membership` junction table. |

---

## 3. Protocol Constraints

### Current Protocols (`/server/engram/storage/protocols.py`)

**GraphStore** -- 32+ methods:
- Entity CRUD: create, get, update, delete, find, find_entity_candidates
- Relationship CRUD: create, get (by entity, by predicate, at time), invalidate, find_conflicting, find_existing, get_all_edges, get_expired
- Episode CRUD: create, update, get, get_by_id, get_paginated
- Episode-entity: get_episode_entities, link_episode_entity
- Stats: get_stats, get_top_connected, get_growth_timeline, get_entity_type_counts
- Consolidation: get_co_occurring_entity_pairs, get_entity_episode_counts, get_dead_entities, merge_entities, update_relationship_weight
- Graph traversal: get_neighbors, get_active_neighbors_with_weights, path_exists_within_hops
- Identity: get_identity_core_entities
- Prospective: create/get/list/update/delete/increment intention

**ActivationStore** -- 7 methods:
- get/set_activation, batch_get, batch_set, record_access, clear_activation, get_top_activated

**SearchIndex** -- 9 methods:
- initialize, close, index_entity, index_episode, search, batch_index_entities, remove, compute_similarity, search_episodes, get_entity_embeddings, get_graph_embeddings

**ConsolidationStore** -- 18 methods:
- Cycle lifecycle: save, update, get, get_recent
- Audit records: save/get for merge, inferred_edge, prune, reindex, replay, dream, dream_association, triage, graph_embed
- cleanup

### Dual Backend Constraint

Every protocol method must be implemented in BOTH:
- `server/engram/storage/sqlite/graph.py` (SQLiteGraphStore)
- `server/engram/storage/falkordb/graph.py` (FalkorDBGraphStore)

Plus activation stores:
- `server/engram/storage/memory/activation.py` (MemoryActivationStore -- in-process)
- `server/engram/storage/redis/activation.py` (RedisActivationStore -- full mode)

Plus search indexes:
- `server/engram/storage/sqlite/search.py` (FTS5SearchIndex + sqlite-vec)
- `server/engram/storage/vector/redis_search.py` (RedisSearchIndex)

### Protocol Changes Needed Per Feature

| Feature | Protocol Changes |
|---------|-----------------|
| 1. Inhibitory | None -- uses existing `get_active_neighbors_with_weights` |
| 2. Taxonomy | `GraphStore.get_entities_by_maturity()`, `update_entity_maturity()`. Or use existing `update_entity` with attributes. |
| 3. Emotional | `GraphStore.get_entities_by_emotion()` or filter in retrieval. Possibly `SearchIndex.search()` with emotional filter. |
| 4. State-dependent | `SearchIndex.search_with_state()` or state similarity computed in pipeline. Episode metadata already extensible. |
| 5. Reconsolidation | `GraphStore.update_entity()` already supports arbitrary dict updates. New audit table needs ConsolidationStore methods. |
| 6. Schemas | `GraphStore.get_schema_members()`, `create_schema()`, etc. Or model schemas as entities with `entity_type="Schema"`. |

**Key insight:** Many features can avoid protocol changes by using the existing `attributes` JSON blob on entities and filtering in the retrieval pipeline rather than the storage layer. This avoids dual-backend implementation cost.

---

## 4. Test Patterns

### Test Infrastructure

- **Framework:** pytest + pytest-asyncio
- **136 Python files** in `server/engram/`, **128 test files** in `server/tests/`
- **1568+ backend tests passing**, 176 frontend tests

### Fixture Pattern (consolidation tests)

```python
@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx

@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())

@pytest_asyncio.fixture
async def consol_store(store):
    s = SQLiteConsolidationStore(store._db_path)
    await s.initialize(db=store._db)
    return s

@pytest_asyncio.fixture
async def engine(store, activation, search, consol_store):
    return ConsolidationEngine(store, activation, search, cfg=ActivationConfig(),
                                consolidation_store=consol_store, event_bus=EventBus())
```

### Mock Pattern (phase tests)

Phases are tested with `AsyncMock` for all stores:
```python
graph_store = AsyncMock()
graph_store.get_episodes = AsyncMock(return_value=[ep])
graph_store.find_entities = AsyncMock(return_value=[])
```

### Retrieval Pipeline Tests

Use lightweight `_FakeEntity`, `_FakeSearchIndex`, `_FakeActivationStore`, `_FakeGraphStore` stub classes (not mocks) -- more predictable, less brittle. Each stub has minimal implementations of needed protocol methods.

### Phase Execute Signature

All consolidation phases must implement:
```python
async def execute(
    self, group_id, graph_store, activation_store, search_index,
    cfg: ActivationConfig, cycle_id: str, dry_run: bool = False,
    context: CycleContext | None = None,
) -> tuple[PhaseResult, list[AuditRecord]]
```

### Testing Strategy for New Features

1. **Config tests**: Verify new fields have correct defaults, profiles set them properly
2. **Unit tests**: Test scoring/algorithm functions in isolation with controlled inputs
3. **Phase tests**: Use AsyncMock stores, test execute() returns correct PhaseResult
4. **Pipeline integration**: Use Fake stores (not mocks), verify end-to-end retrieval behavior
5. **Engine integration**: Use real SQLiteGraphStore with tmp_path, run full cycles
6. **MCP tests**: Verify tool output format includes new fields
7. **Marker**: `@pytest.mark.requires_docker` for full-mode tests

---

## 5. Event Flow

### Current Event Types

| Event Type | Publisher | Consumers |
|-----------|----------|-----------|
| `episode.queued` | GraphManager.store_episode() | EpisodeWorker, PressureAccumulator |
| `episode.completed` | GraphManager.project_episode() | PressureAccumulator |
| `episode.failed` | GraphManager.project_episode() | (none) |
| `graph.nodes_added` | GraphManager.project_episode() | PressureAccumulator, Dashboard |
| `graph.delta` | ConsolidationEngine (prune/merge) | Dashboard |
| `consolidation.started` | ConsolidationEngine | Dashboard |
| `consolidation.completed` | ConsolidationEngine | Dashboard |
| `consolidation.phase.{name}.started` | ConsolidationEngine | Dashboard |
| `consolidation.phase.{name}.completed` | ConsolidationEngine | Dashboard |
| `consolidation.phase.{name}.failed` | ConsolidationEngine | Dashboard |

### Event Architecture

- **EventBus**: In-process pub/sub with per-group queues, ring buffer history, on-publish hooks
- **Redis bridge**: In full mode, hooks into EventBus to forward events via Redis pub/sub
- **WebSocket**: Dashboard subscribes to EventBus events via WebSocket connection
- Events are simple dicts: `{seq, type, timestamp, group_id, payload}`

### New Events Needed Per Feature

| Feature | New Events |
|---------|-----------|
| 3. Emotional | `episode.emotional_tags` (after extraction with emotional metadata) |
| 5. Reconsolidation | `entity.reconsolidated` (when recall triggers entity update) |
| 6. Schema | `schema.formed`, `schema.updated` (when schema detection runs) |

---

## 6. MCP Tool Surface

### Current Tools (15)

| Tool | Parameters | Notes |
|------|-----------|-------|
| `remember` | content, source | Full extraction, auto-recall, intentions |
| `observe` | content, source | Store-only (QUEUED), auto-recall, intentions |
| `recall` | query, limit | Full retrieval pipeline |
| `search_entities` | name, entity_type, limit | Name/type lookup |
| `search_facts` | query, subject, predicate, include_expired, limit | Relationship search |
| `forget` | entity_name OR fact, reason | Soft delete |
| `get_context` | max_tokens, topic_hint, project_path, format | Pre-assembled context |
| `bootstrap_project` | project_path | Auto-observe project files |
| `get_graph_state` | top_n, include_edges, entity_types | Stats + top nodes |
| `mark_identity_core` | entity_name, identity_core | Protect from pruning |
| `trigger_consolidation` | dry_run | Manual consolidation |
| `get_consolidation_status` | (none) | Check running state |
| `intend` | trigger_text, action_text, trigger_type, entity_names, threshold, priority, context, see_also | Prospective memory |
| `dismiss_intention` | intention_id, hard | Disable/delete intention |
| `list_intentions` | enabled_only | List with warmth info |

### Resources (3)

- `engram://graph/stats`
- `engram://entity/{entity_id}`
- `engram://entity/{entity_id}/neighbors`

### Prompts (2)

- `engram_system` (persona, auto_remember)
- `engram_context_loader` (topic)

### MCP Changes Needed Per Feature

| Feature | Tool Changes |
|---------|-------------|
| 1. Inhibitory | None -- internal algorithm change. May want new scoring weight visible in `recall` response. |
| 2. Taxonomy | `recall` response could include `memory_type` field. New tool `mature_memory` or automatic via consolidation. |
| 3. Emotional | `remember`/`observe` could accept `emotional_context` parameter. `recall` could accept `mood` filter. `recall` response includes emotional scores. |
| 4. State-dependent | `recall` could accept `context_state` dict. `observe`/`remember` could accept state metadata. |
| 5. Reconsolidation | `recall` auto-triggers reconsolidation -- no new tools. Reconsolidation metadata in recall response. |
| 6. Schema | New `get_schemas` tool. Schema info in `recall` and `get_context` responses. |

---

## 7. Dependency Matrix

### Feature Dependencies

```
         1.Inhib  2.Tax  3.Emot  4.State  5.Recon  6.Schema
1.Inhib    --      no     no      no       no       no
2.Tax      no      --     weak    no       STRONG   STRONG
3.Emot     no      weak   --      MEDIUM   weak     no
4.State    no      no     MEDIUM  --       no       no
5.Recon    no      STRONG weak    no       --       weak
6.Schema   no      STRONG no      no       weak     --
```

**Key Dependencies:**

- **2 <-> 5 (STRONG)**: Reconsolidation updates entity summaries/attributes, which is the mechanism by which episodic memories become semantic. Taxonomy needs reconsolidation to drive the transition.
- **2 <-> 6 (STRONG)**: Schema formation requires semantic-level entities (mature memories) to cluster. Taxonomy provides the maturity signal schemas need.
- **3 <-> 4 (MEDIUM)**: Emotional salience IS a form of state. Emotional state during encoding affects state-dependent retrieval. These share "state at encoding time" infrastructure.
- **3 <-> 2 (weak)**: Emotional memories may resist semantic graduation (flashbulb memory effect).
- **5 <-> 3 (weak)**: Reconsolidation could update emotional tags as memories are re-experienced.

### Independent Features

- **Feature 1 (Inhibitory spreading)**: Fully independent. Pure algorithm change in `spreading.py` and `strategy.py`. No schema, no protocol, no event changes.
- **Features 3+4 (Emotional + State)**: Share infrastructure (encoding context) but can be built independently and composed.

### Shared Infrastructure Needed

1. **Entity metadata extension**: All of {taxonomy, emotional, state, schema} need richer entity metadata. Use the `attributes` JSON blob pattern -- avoids schema migration for each feature.

2. **Episode metadata extension**: Features {emotional, state} need episode-level context at encoding time. Add a `context` JSON column to episodes (or use existing `attributes` pattern).

3. **Scoring signal extension**: Features {inhibitory, emotional, state, taxonomy} all want to influence retrieval scoring. Need a clean "signal plugin" pattern in `scorer.py` rather than ad-hoc additions.

4. **Consolidation phase extension**: Features {taxonomy, reconsolidation, schema} need new consolidation phases. The engine already supports N phases -- just register new ones.

---

## 8. Risk Register

### Performance Risks

| Risk | Severity | Feature | Mitigation |
|------|----------|---------|------------|
| Inhibitory spreading doubles BFS work | MEDIUM | 1 | Budget system already exists (`spread_energy_budget`). Inhibitory edges just reduce energy, don't add traversals. |
| Emotional extraction adds latency to remember/observe | HIGH | 3 | Use triage pattern: extract emotion only on remember (expensive path), not observe. Or extract asynchronously in worker. |
| State-dependent retrieval adds dimension to search | MEDIUM | 4 | State similarity can be computed lazily as a reranking step, not during initial candidate generation. |
| Reconsolidation on every recall is expensive | HIGH | 5 | Rate-limit: only reconsolidate if entity hasn't been reconsolidated in N hours. Use lightweight update (summary append, not full re-extraction). |
| Schema detection over full graph is O(N^2) | HIGH | 6 | Run as consolidation phase (offline). Use incremental detection (only affected entities). Cap max entities per cycle. |
| Memory taxonomy graduation queries all entities | MEDIUM | 2 | Run as consolidation phase. Filter by access_count and age thresholds first. |

### Migration Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Schema changes break existing databases | HIGH | Use `IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN` with try/except. Never drop columns. |
| New config fields break existing .env files | LOW | Pydantic defaults handle missing fields. `extra="forbid"` only on ActivationConfig. EngramConfig uses `extra="ignore"`. |
| New protocol methods break FalkorDB backend | MEDIUM | Add methods to both backends simultaneously. Use `attributes` JSON where possible to avoid protocol changes. |
| Score weight rebalancing changes retrieval behavior | HIGH | Make new signals additive bonuses (like exploration_bonus) rather than rebalancing existing weights. Gate behind feature flags. |

### Backwards Compatibility Concerns

1. **MCP tool responses**: New fields in recall response are additive -- won't break existing consumers. But changing field names or removing fields would break.
2. **Event types**: New event types are safe. Changing payload structure of existing events affects dashboard.
3. **Consolidation phase order**: Adding phases at the end is safe. Reordering existing phases requires careful testing.
4. **Database**: Adding tables/columns is safe. Never remove columns from existing tables.

---

## 9. Phased Implementation Order

### Phase 0: Shared Infrastructure (PREREQUISITE)

**Duration:** ~2 days
**Dependencies:** None

1. **Scoring signal plugin pattern**: Refactor `score_candidates()` to accept an extensible `signals: dict[str, float]` per candidate, with corresponding `weights: dict[str, float]` from config. This avoids modifying the scorer for every new feature.

2. **Episode context column**: Add `encoding_context TEXT` (JSON) to episodes table. This is the shared infrastructure for features 3 (emotional) and 4 (state-dependent).

3. **Entity attributes conventions**: Document that `attributes` JSON blob keys for new features use namespaced prefixes: `emo_*` for emotional, `tax_*` for taxonomy, `schema_*` for schemas, `recon_*` for reconsolidation.

### Phase 1: Independent Features (PARALLEL)

**Duration:** ~1 week each
**Can run in parallel -- no dependencies between them**

#### 1a. Inhibitory Spreading Activation

- **Files changed**: `activation/spreading.py`, `activation/strategy.py`, `config.py` (3-4 new fields)
- **Schema changes**: None
- **Protocol changes**: None (uses existing `get_active_neighbors_with_weights` -- needs predicate info for inhibition, which 4th tuple element already provides)
- **Tests**: Pure unit tests on spreading functions
- **Risk**: LOW

#### 1b. Emotional Salience Scoring

- **Files changed**: `extraction/extractor.py` or new `extraction/emotional.py`, `retrieval/scorer.py`, `config.py` (~8 new fields)
- **Schema changes**: Episode `encoding_context` (Phase 0). Entity attributes: `emo_valence`, `emo_arousal`, `emo_tags`.
- **Protocol changes**: None if using attributes JSON
- **Tests**: Extraction tests (mock LLM), scorer tests, pipeline integration
- **Risk**: MEDIUM (LLM call latency)

#### 1c. State-Dependent Retrieval

- **Files changed**: New `retrieval/state_matching.py`, `retrieval/pipeline.py`, `config.py` (~6 new fields)
- **Schema changes**: Episode `encoding_context` (Phase 0). State at recall time is transient (not stored).
- **Protocol changes**: None
- **Tests**: State matching unit tests, pipeline integration with state context
- **Risk**: MEDIUM (needs clear API for providing state)

### Phase 2: Memory Taxonomy + Reconsolidation (SEQUENTIAL)

**Duration:** ~1.5 weeks
**Must be built together -- reconsolidation IS the mechanism for taxonomy graduation**

#### 2a. Memory Taxonomy (Episodic -> Semantic)

- **Files changed**: New `consolidation/phases/graduate.py`, `models/entity.py` or attributes, `config.py` (~10 new fields)
- **Schema changes**: None if using attributes (preferred). Otherwise `memory_maturity` column on entities.
- **New consolidation phase**: `GraduatePhase` -- runs after dream, evaluates entity maturity based on access_count, age, reconsolidation_count, relationship density
- **Tests**: Phase tests, maturity scoring unit tests
- **Risk**: MEDIUM

#### 2b. Reconsolidation on Recall

- **Files changed**: `graph_manager.py` (recall method), new `retrieval/reconsolidate.py`, `config.py` (~6 new fields)
- **Schema changes**: None if using attributes. Track `recon_count`, `last_reconsolidated` in entity attributes.
- **Protocol changes**: None (uses existing `update_entity`)
- **New event**: `entity.reconsolidated`
- **Tests**: Unit tests for reconsolidation logic, integration tests for recall-triggered updates
- **Risk**: HIGH (modifying the recall hot path)

### Phase 3: Schema Formation (DEPENDS ON Phase 2)

**Duration:** ~1 week
**Depends on taxonomy maturity signal**

- **Files changed**: New `consolidation/phases/schema.py`, new `models/schema.py`, `config.py` (~8 new fields)
- **Schema changes**: New `schemas` table, `schema_members` junction table
- **New consolidation phase**: `SchemaFormationPhase` -- clusters mature semantic entities by embedding similarity + graph proximity
- **Protocol changes**: Add `get_schemas`, `create_schema`, `get_schema_members` to GraphStore (must implement for both backends)
- **New MCP tool**: `get_schemas` (optional, could just expose in get_context)
- **Tests**: Phase tests with entity clustering, schema detection accuracy
- **Risk**: HIGH (graph-wide operation, dual-backend protocol changes)

### Execution Timeline

```
Week 1:  [Phase 0: Infrastructure] + [Phase 1a: Inhibitory -- start]
Week 2:  [Phase 1a: complete] + [Phase 1b: Emotional -- start] + [Phase 1c: State -- start]
Week 3:  [Phase 1b+1c: complete] + [Phase 2a: Taxonomy -- start]
Week 4:  [Phase 2a+2b: Taxonomy + Reconsolidation]
Week 5:  [Phase 3: Schema Formation]
```

### Parallelization Opportunities

- Phase 1a, 1b, 1c can ALL run in parallel (different agents/developers)
- Phase 2a and 2b should be same developer (tight coupling)
- Phase 3 waits for Phase 2 completion

### Test Budget Per Phase

| Phase | Expected New Tests | Test Strategy |
|-------|-------------------|---------------|
| 0 (Infrastructure) | ~10 | Config tests, scorer refactor tests |
| 1a (Inhibitory) | ~15 | Spreading unit tests, strategy tests |
| 1b (Emotional) | ~20 | Extraction mock tests, scorer tests, pipeline tests |
| 1c (State-dependent) | ~15 | State matching tests, pipeline tests |
| 2a (Taxonomy) | ~20 | Phase tests, maturity scoring, graduation logic |
| 2b (Reconsolidation) | ~25 | Recall-path tests, update logic, rate limiting |
| 3 (Schema) | ~25 | Phase tests, clustering accuracy, dual-backend |
| **Total** | **~130** | |

---

## Key Architectural Insights

1. **The `attributes` JSON blob is your best friend.** Most new per-entity metadata (emotional tags, maturity level, reconsolidation count, schema membership) should go in `attributes` rather than new columns. This avoids schema migrations and protocol changes for both backends.

2. **The scoring pipeline is the integration point.** All 6 features ultimately affect retrieval. The scorer (`score_candidates`) is where they converge. A signal plugin pattern prevents it from becoming a 20-parameter monster.

3. **Consolidation phases are the offline workhorse.** Features 2, 5, 6 are primarily offline operations (taxonomy graduation, reconsolidation batching, schema detection). The consolidation engine's phase system is designed for exactly this -- just add new phases.

4. **The CQRS split protects the hot path.** `observe` (cheap) vs `remember` (expensive) is the pattern to follow. Emotional extraction should piggyback on `remember`'s extraction, not add latency to `observe`.

5. **Feature flags are mandatory.** Every feature must be gatable (`feature_enabled: bool = False`). The profile system should gain a new level or these should be added to existing profiles progressively.

6. **The 4th element of `get_active_neighbors_with_weights` already returns entity_type.** This is critical for inhibitory spreading (need predicate to determine inhibitory edges) and cross-domain penalty (need entity_type). The tuple is `(neighbor_id, weight, predicate, entity_type)`.

7. **The `CycleContext` dataclass is the inter-phase communication bus.** New phases should add their own set fields (e.g., `graduated_entity_ids`, `reconsolidated_entity_ids`) to pass information forward to downstream phases.
