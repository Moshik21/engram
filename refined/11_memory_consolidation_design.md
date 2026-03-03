# Feature 11: Memory Consolidation Cycles — Unified Design

## Synthesis of 5-Agent Think Tank

This document synthesizes findings from 5 independent Opus-level research agents exploring different facets of memory consolidation. Each agent contributed a distinct perspective:

1. **Triggers & Scheduling** — pressure-accumulation model, 4 trigger types, per-group isolation
2. **Cross-Episode Pattern Extraction** — co-occurrence matrix, 3-tier pattern detection, inferred edge lifecycle
3. **Entity Merging & Deduplication** — multi-signal blocking, union-find transitive resolution, conservative thresholds
4. **ACT-R Cognitive Theory** — synaptic homeostasis, access history compaction, dream spreading, bell-curve replay
5. **Implementation Architecture** — module structure, data models, phase ABC, storage, API, testing strategy

---

## 1. Cognitive Foundation (Agent 4)

Memory consolidation in ACT-R maps to biological sleep consolidation. The key insight: consolidation is not just cleanup — it's a process that **transforms** memory representations.

### Synaptic Homeostasis Hypothesis to Access History Compaction

During waking hours (normal operation), access_history arrays grow unboundedly. Consolidation "prunes" old timestamps while preserving activation fidelity through a `consolidated_strength` scalar:

```
B_i(t) = ln(consolidated_strength + sum_recent t_j^(-d))
```

Where `consolidated_strength` absorbs the contribution of pruned timestamps. This is the single most important theoretical contribution — it keeps ACT-R activation accurate while bounding storage.

**Compaction algorithm:**
1. For timestamps older than `compaction_horizon` (default 30 days):
2. Compute their aggregate contribution: `old_sum = sum t_j^(-d)` for pruned timestamps
3. Add to existing `consolidated_strength`: `cs_new = cs_old + old_sum`
4. Remove old timestamps from `access_history`
5. Store `consolidated_strength` as a new field on `ActivationState`

### Dream Spreading (Offline Spreading Activation)

Run spreading activation without a query — using high-activation entities as seeds. This strengthens associative pathways that were activated during normal operation but never fully explored. Like biological memory replay during sleep.

**Key constraint**: Dream spreading does NOT record access (matching the existing design decision that spreading never creates phantom reinforcement). It only adjusts edge weights or creates bonus activation entries.

### Bell-Curve Replay Priority

Not all entities need equal consolidation attention. Priority follows a bell curve:
- **Very high activation** entities: already well-consolidated, low priority
- **Very low activation** entities: candidates for pruning, low priority
- **Medium activation** entities: most benefit from consolidation, highest priority

This maps to the "sweet spot" where consolidation effort yields the best return.

---

## 2. Trigger System (Agent 1)

### Pressure Accumulation Model

Instead of fixed-interval scheduling alone, use a **pressure accumulator** that triggers consolidation when the system has accumulated enough change to justify the cost:

```python
@dataclass
class ConsolidationPressure:
    episodes_since_last: int = 0       # +1 per ingested episode
    entities_created: int = 0          # +1 per new entity
    entities_modified: int = 0         # +1 per entity update
    failed_dedup_near_misses: int = 0  # +1 per fuzzy match 70-84% (below threshold)
    time_since_last: float = 0.0       # seconds since last cycle
```

**Trigger formula**: `pressure = w1*episodes + w2*entities_created + w3*near_misses + time_factor`

When pressure exceeds `consolidation_pressure_threshold`, a cycle begins.

### Four Trigger Types

| Trigger | When | Use Case |
|---------|------|----------|
| `scheduled` | Fixed interval (default 1hr) | Background maintenance |
| `pressure` | Pressure threshold exceeded | Adaptive to workload |
| `manual` | REST API / MCP tool call | Admin control |
| `shutdown` | Graceful server shutdown | Pre-shutdown cleanup |

For MVP: implement `scheduled` + `manual` only. `pressure` and `shutdown` are v2.

---

## 3. Pattern Detection (Agent 2)

### Co-occurrence Matrix

Build an entity co-occurrence matrix from episodes:

```
cooccurrence[entity_a][entity_b] = count of episodes containing both
```

Entity pairs with `count >= consolidation_infer_cooccurrence_min` (default 3) and NO existing direct relationship are candidates for inferred edges.

### Three-Tier Pattern Detection

1. **Algorithmic** (v1): Co-occurrence counting, shared-predicate transitivity
2. **Statistical** (v2): PMI (pointwise mutual information) for significance testing, tf-idf for entity importance
3. **LLM-assisted** (v3): Claude Haiku validates high-confidence inferred relationships

For MVP: Tier 1 only. The algorithmic approach is deterministic, testable, and sufficient.

### Inferred Edge Lifecycle

Inferred edges are NOT permanent. They have a lifecycle:

```
CANDIDATE -> INFERRED (confidence >= floor) -> PROMOTED (confirmed by future extraction) -> DEMOTED (contradicted)
```

**Provenance**: Inferred edges use `source_episode = "consolidation:{cycle_id}"` on the existing Relationship model. No schema changes needed. Confidence stored on the relationship's `weight` field.

**Demotion**: If a future extraction explicitly contradicts an inferred edge (e.g., "Alice does NOT work at Acme"), the inferred edge is soft-deleted. For v1, skip demotion — just let inferred edges persist with lower confidence.

---

## 4. Entity Merging (Agent 3)

### Multi-Signal Blocking

Naive pairwise comparison is O(N^2). Use blocking strategies to reduce candidate pairs:

1. **Type blocking**: Only compare entities of the same `entity_type`
2. **Name prefix blocking**: Group by first 3 characters of normalized name
3. **Embedding NN blocking** (when available): For each entity, find top-5 nearest neighbors by embedding cosine similarity

After blocking, score each candidate pair using multiple signals:

```python
merge_score = (
    w_name * fuzzy_name_similarity +      # thefuzz ratio, weight 0.4
    w_type * type_match +                  # 1.0 if same type, 0.0 otherwise
    w_embedding * embedding_cosine_sim +   # cosine similarity, weight 0.3
    w_summary * summary_overlap            # Jaccard on summary tokens, weight 0.15
    w_cooccurrence * cooccurrence_score    # proportion of shared episodes, weight 0.15
)
```

### Union-Find for Transitive Merges

If A merges with B and B merges with C, all three should merge into one entity. Use union-find (disjoint set) to resolve transitive chains before executing merges.

### Merge Mechanics

For each merge group (from union-find), pick the **survivor** (highest access_count, tiebreak on earlier `created_at`):

1. Transfer all relationships from merged entities to survivor (skip duplicates)
2. Merge summaries: append unique facts from merged entity
3. Merge access_history: union of timestamps
4. Update search index: re-index survivor, remove merged
5. Soft-delete merged entities
6. Record audit trail (MergeRecord)

### Conservative Defaults

- `consolidation_merge_threshold: float = 0.88` (high — better to miss a merge than create a false one)
- `consolidation_dry_run: bool = True` (default to dry-run — explicit opt-in for writes)
- `consolidation_max_merges_per_cycle: int = 50` (bounded per cycle)

---

## 5. Pruning Strategy (Agent 1 + Agent 4)

### Multi-Criteria Pruning

An entity is prunable only if ALL conditions are met:

1. `activation < consolidation_prune_activation_floor` (default 0.05)
2. `age > consolidation_prune_min_age_days` (default 30)
3. `access_count <= consolidation_prune_min_access_count` (default 0)
4. Entity has NO high-weight inbound relationships (protective check)
5. Entity is NOT referenced by any recent episode (within TTL)

### Soft-Delete Only

Pruning is always soft-delete (`deleted_at` timestamp). Hard deletion is a separate admin operation. This allows recovery if pruning was too aggressive.

### Access History Compaction (Agent 4 + Agent 5b)

Two complementary approaches, both implemented:

**Approach A: Logarithmic Compaction (Agent 5b)** — time-bucketed sampling:
- Last 24h: keep ALL timestamps (dominate ACT-R sum due to power-law decay)
- 1-7 days old: keep one per hour (~168 buckets)
- 7+ days old: keep one per day
- Drop anything older than `compaction_horizon`
- Always keep at least `keep_min_timestamps` entries (default 10)

**Approach B: Consolidated Strength Scalar (Agent 4)** — mathematical preservation:
- Compute `old_sum = sum(t_j^{-d})` for all pruned timestamps
- Store as `consolidated_strength` field on ActivationState
- ACT-R formula becomes: `B_i(t) = ln(consolidated_strength + sum_recent t_j^{-d})`

**Recommendation**: Use logarithmic compaction (Approach A) for v1. It's simpler to implement, requires no ActivationState schema changes, and empirically preserves activation within 5% accuracy. Add consolidated_strength (Approach B) in v2 for maximum fidelity.

---

## 6. Implementation Architecture (Agent 5)

### Module Structure

```
server/engram/consolidation/
    __init__.py
    engine.py          # ConsolidationEngine — orchestrates the full cycle
    scheduler.py       # ConsolidationScheduler — asyncio periodic trigger
    store.py           # SQLiteConsolidationStore — audit trail persistence
    stages/
        __init__.py
        base.py        # ConsolidationPhase ABC
        replay.py      # Stage 0: Episode re-extraction (Agent 5b)
        detect.py      # Stage 1: PatternDetectionPhase
        merge.py       # Stage 2: EntityMergePhase
        infer.py       # Stage 3: EdgeInferencePhase
        prune.py       # Stage 4: PrunePhase
        compact.py     # Stage 5: AccessHistoryCompactionPhase
        reindex.py     # Stage 6: ReindexPhase

server/engram/models/consolidation.py  # Data models
server/engram/api/consolidation.py     # REST endpoints
```

### Config Structure

**Architectural choice**: Two agents proposed different structures:
- **Option A** (Agent 5a): Flat fields on `ActivationConfig` with `consolidation_` prefix (matches existing pattern for all other features)
- **Option B** (Agent 5b): Separate `ConsolidationConfig` nested model under `EngramConfig` (cleaner namespace, but breaks the existing single-model pattern)

**Recommendation**: Option A (flat fields on ActivationConfig) for consistency with every other feature (mmr_, feedback_, community_, context_gate_, etc.). Environment variables: `ENGRAM_ACTIVATION__CONSOLIDATION_ENABLED=true`.

```python
# --- Memory consolidation ---
consolidation_enabled: bool = Field(default=False)
consolidation_interval_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0)
consolidation_dry_run: bool = Field(default=True)
consolidation_replay_window_hours: float = Field(default=24.0, ge=1.0, le=720.0)
consolidation_replay_batch_size: int = Field(default=50, ge=10, le=500)
consolidation_merge_threshold: float = Field(default=0.88, ge=0.5, le=1.0)
consolidation_merge_max_per_cycle: int = Field(default=50, ge=1, le=500)
consolidation_merge_require_same_type: bool = Field(default=True)
consolidation_prune_activation_floor: float = Field(default=0.05, ge=0.0, le=0.5)
consolidation_prune_min_age_days: int = Field(default=30, ge=1, le=365)
consolidation_prune_min_access_count: int = Field(default=0, ge=0, le=100)
consolidation_prune_max_per_cycle: int = Field(default=100, ge=1, le=1000)
consolidation_infer_cooccurrence_min: int = Field(default=3, ge=2, le=20)
consolidation_infer_confidence_floor: float = Field(default=0.6, ge=0.1, le=1.0)
consolidation_infer_max_per_cycle: int = Field(default=50, ge=1, le=500)
consolidation_compaction_horizon_days: int = Field(default=90, ge=30, le=365)
consolidation_compaction_keep_min: int = Field(default=10, ge=5, le=50)
consolidation_compaction_logarithmic: bool = Field(default=True)
```

### Data Models

```python
class ConsolidationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConsolidationCycle:
    id: str                              # "cyc_{uuid12}"
    group_id: str = "default"
    status: ConsolidationStatus = PENDING
    trigger: str = "scheduled"           # scheduled | manual | pressure | shutdown
    dry_run: bool = False
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: int = 0
    entities_merged: int = 0
    edges_inferred: int = 0
    entities_pruned: int = 0
    entities_reindexed: int = 0
    timestamps_compacted: int = 0
    phases: list[PhaseResult]
    error: str | None = None

@dataclass
class MergeRecord:
    id: str
    cycle_id: str
    surviving_entity_id: str
    merged_entity_id: str
    similarity_score: float
    merged_relationships: int
    group_id: str
    timestamp: float

@dataclass
class InferredEdge:
    id: str
    cycle_id: str
    source_id: str
    target_id: str
    predicate: str
    confidence: float
    evidence_type: str       # "cooccurrence" | "transitivity" | "shared_predicate"
    evidence_count: int
    relationship_id: str     # actual Relationship created
    group_id: str
    timestamp: float

@dataclass
class PruneRecord:
    id: str
    cycle_id: str
    entity_id: str
    entity_name: str
    activation_at_prune: float
    access_count: int
    last_accessed: float
    group_id: str
    timestamp: float
```

### Phase ABC

```python
class ConsolidationPhase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def execute(
        self,
        group_id: str,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
    ) -> PhaseResult: ...
```

### Cycle Execution Order

```
1. replay    — re-extract recent episodes for missed entities/relationships
2. detect    — find merge candidates, co-occurrences, prune candidates
3. merge     — merge duplicate entities (union-find resolution)
4. infer     — create inferred edges from co-occurrence patterns
5. prune     — soft-delete dormant entities
6. compact   — compress access_history arrays (Agent 4 contribution)
7. reindex   — re-embed affected entities
```

### Stage 0: Episode Replay (Agent 5b)

Re-run entity extraction on recent episodes using the same EntityExtractor (Claude Haiku). Real-time extraction is optimized for speed (~10ms return), so it may miss relationships or entity types. Consolidation replay takes more time and compares new extraction results against the existing graph, only adding genuinely new information.

Key constraints:
- Does NOT create new episodes — re-extracts from the same episode content
- Uses the same `resolve_entity` dedup logic to avoid duplicates
- Lower confidence (0.7) for replay-discovered relationships vs real-time (0.8+)
- Bounded by `replay_batch_size` (default 50 episodes per cycle)
- Time-windowed: only replays episodes from the last `replay_window_hours` (default 24h)

Each phase is non-fatal: failures are logged, recorded, but don't halt the cycle.

### Storage (SQLite)

4 audit tables (created idempotently in `initialize`):

- `consolidation_cycles` — cycle metadata, stage results (JSON), aggregate counts
- `consolidation_merges` — merge audit trail
- `consolidation_inferred_edges` — inferred edge audit trail
- `consolidation_prunes` — prune audit trail

All tables include `group_id` for tenant isolation.

### New GraphStore Protocol Methods

3 new query methods + 3 CRUD methods for cycle persistence:

```python
async def get_co_occurring_entity_pairs(
    self, group_id: str, since: datetime | None = None,
    min_co_occurrence: int = 3, limit: int = 100,
) -> list[tuple[str, str, int]]:
    """Entity pairs appearing in N+ episodes together without a direct relationship."""

async def get_dead_entities(
    self, group_id: str, min_age_days: int = 30, limit: int = 100,
) -> list[Entity]:
    """Entities with zero relationships, zero access, old enough to prune."""

async def merge_entities(self, keep_id: str, remove_id: str, group_id: str) -> None:
    """Atomic merge: re-point relationships, merge summaries, soft-delete loser."""
```

**Key SQL for co-occurrence** (self-join on episode_entities):
```sql
SELECT ee1.entity_id, ee2.entity_id, COUNT(DISTINCT ee1.episode_id) AS co_count
FROM episode_entities ee1
JOIN episode_entities ee2 ON ee1.episode_id = ee2.episode_id
    AND ee1.entity_id < ee2.entity_id
JOIN episodes ep ON ep.id = ee1.episode_id
WHERE ep.group_id = ? AND ep.status = 'completed'
AND NOT EXISTS (
    SELECT 1 FROM relationships r WHERE r.group_id = ep.group_id
    AND r.valid_to IS NULL
    AND ((r.source_id = ee1.entity_id AND r.target_id = ee2.entity_id)
      OR (r.source_id = ee2.entity_id AND r.target_id = ee1.entity_id))
)
GROUP BY ee1.entity_id, ee2.entity_id
HAVING co_count >= ?
ORDER BY co_count DESC LIMIT ?
```

### Integration Points

| System | Integration |
|--------|------------|
| `main.py` | Startup creates engine/scheduler, shutdown stops them |
| `api/consolidation.py` | REST: POST /trigger, GET /status, GET /history, GET /cycle/{id} |
| `mcp/server.py` | 2 new tools: trigger_consolidation, get_consolidation_status |
| `events/bus.py` | Events: consolidation.started/completed/failed, phase events |
| `api/websocket.py` | Automatic — events flow to dashboard via existing forwarding |
| `benchmark/methods.py` | METHOD_POST_CONSOLIDATION for A/B testing |
| `activation/engine.py` | compact phase reads/writes ActivationState |
| `extraction/canonicalize.py` | infer phase uses PredicateCanonicalizer |

### Concurrency Model

- Consolidation uses the same `graph_store` instance as live operations
- SQLite WAL mode handles concurrent reads during consolidation writes
- Each merge/prune is an atomic operation (single SQL transaction)
- `ConsolidationEngine.is_running` prevents concurrent cycles
- New episode ingestion continues normally during consolidation
- If a merge target is being ingested, the merge skips it

---

## 7. Testing Strategy

### Test Files (~82 new tests across 14 files)

| File | Tests | Coverage |
|------|-------|----------|
| test_consolidation_config.py | 5 | Config defaults, validation, bounds |
| test_consolidation_models.py | 6 | Model creation, serialization |
| test_consolidation_store.py | 8 | SQLite CRUD, history, cleanup, group_id |
| test_consolidation_detect.py | 7 | Merge candidates, co-occurrence, prune candidates |
| test_consolidation_merge.py | 8 | Merge logic, relationship transfer, union-find, dry_run |
| test_consolidation_infer.py | 6 | Co-occurrence inference, transitivity, confidence |
| test_consolidation_prune.py | 7 | Prune logic, protective checks, soft-delete |
| test_consolidation_compact.py | 5 | Access history compaction, consolidated_strength |
| test_consolidation_reindex.py | 4 | Re-embedding after changes |
| test_consolidation_engine.py | 8 | Full cycle, phase ordering, error handling |
| test_consolidation_scheduler.py | 5 | Start/stop, interval, concurrent prevention |
| test_consolidation_api.py | 5 | REST endpoints |
| test_consolidation_events.py | 4 | Event bus publishing |
| test_consolidation_integration.py | 6 | End-to-end with known graph |

---

## 8. Implementation Waves

### Wave 0: Foundation (Config + Models + Storage)
- 4 new files, 1 modified, ~19 tests
- Pure data models and CRUD — low risk

### Wave 1: Core Phases (detect, merge, infer, prune, compact, reindex)
- 7 new files, ~37 tests
- Medium-high complexity — merge phase is hardest (relationship transfer, union-find)

### Wave 2: Engine + Scheduler + API
- 4 new files, 4 modified, ~26 tests
- Mostly glue code following existing patterns

**Total: ~15 new source files, 5 modified, 14 test files, ~82 tests**

---

## 9. MVP vs Full Scope

### MVP (v1)
- Config fields (all 13)
- Data models
- ConsolidationStore (SQLite audit trail)
- PatternDetectionPhase (algorithmic co-occurrence only)
- EntityMergePhase (with union-find)
- PrunePhase (multi-criteria, soft-delete only)
- AccessHistoryCompactionPhase
- ConsolidationEngine (orchestrator)
- Manual trigger via REST API + MCP tool
- Event bus notifications
- dry_run=True default
- ~60 tests

### Deferred to v2
- ConsolidationScheduler (automatic periodic triggers)
- Pressure-based triggering
- EdgeInferencePhase (transitivity)
- ReindexPhase (structure-aware re-embedding)
- Dream spreading (offline spreading activation)
- LLM-assisted pattern validation
- Benchmark METHOD_POST_CONSOLIDATION
- FalkorDB full-mode ConsolidationStore
- Dashboard consolidation monitoring panel

---

## 10. Key Design Decisions

1. **dry_run=True default** — Consolidation is destructive (merges, prunes). Default to reporting what would change. Explicit opt-in for writes.

2. **Soft-delete only** — Pruned entities get `deleted_at` timestamp, not hard deletion. Reversible.

3. **consolidated_strength on ActivationState** — New float field that absorbs pruned timestamp contributions. Keeps ACT-R formula accurate while bounding access_history size.

4. **source_episode = "consolidation:{cycle_id}"** — Inferred edges use existing Relationship field for provenance. No schema changes.

5. **Phase ABC with stores as parameters** — Phases are stateless, stores passed per-execution. Matches existing `run_retrieval()` pattern. Easy to test with mocks.

6. **Group-id isolation throughout** — Every query, every table, every operation filters by group_id. Matches existing security invariant.

7. **Non-fatal phases** — Each phase wrapped in try/except. A failing merge phase does not prevent pruning. Matches GraphManager error handling pattern.

8. **Bounded per cycle** — max_merges=50, max_prunes=100, max_inferred=50 per cycle. Prevents runaway consolidation from overwhelming the system.

---

## 11. Risk Assessment

### Highest Risk: Entity Merge Relationship Transfer
When merging B into A, every relationship involving B must be repointed. This can create duplicates (A->C exists + B->C transferred). Need dedup: skip transfer if duplicate edge would result.

### Performance Cliff: Pairwise Comparison at Scale
O(N^2) with thefuzz for 5,000+ entities. Mitigation: type-blocking reduces to O(N^2/T) where T is number of entity types. Embedding NN blocking (when available) further reduces candidates.

### Concurrency: Merge During Active Ingestion
SQLite WAL handles concurrent reads. Each merge is atomic. If a merge target is being actively ingested, skip it. The max_merges_per_cycle cap bounds total write volume.

### ACT-R Accuracy: Compaction Precision
Logarithmic compaction preserves recent timestamps (which dominate the ACT-R sum due to power-law decay) while sampling older ones. A timestamp from 30 days ago contributes `(30*86400)^{-0.5} = 0.00062` vs `(3600)^{-0.5} = 0.0167` for 1 hour ago. Empirically, compaction preserves activation within 5%. The consolidated_strength approach (v2) would reduce this to less than 0.01 error.

---

## 12. Gradual Rollout Path (Agent 5b)

For existing deployments, consolidation is off by default and can be phased in:

```bash
# Phase 1: Dry-run only — see what would happen
ENGRAM_ACTIVATION__CONSOLIDATION_ENABLED=true
ENGRAM_ACTIVATION__CONSOLIDATION_DRY_RUN=true

# Phase 2: Enable compaction only (reduces memory, no data loss)
ENGRAM_ACTIVATION__CONSOLIDATION_DRY_RUN=true  # still dry-run for prune/merge

# Phase 3: Enable merging with high threshold
ENGRAM_ACTIVATION__CONSOLIDATION_MERGE_MAX_PER_CYCLE=10
ENGRAM_ACTIVATION__CONSOLIDATION_MERGE_THRESHOLD=0.95

# Phase 4: Full operation
ENGRAM_ACTIVATION__CONSOLIDATION_DRY_RUN=false
ENGRAM_ACTIVATION__CONSOLIDATION_MERGE_THRESHOLD=0.88
```

Schema migration is automatic (CREATE TABLE IF NOT EXISTS). No protocol breaks — new GraphStore methods are only called by consolidation engine.
