# Dream Associations — Cross-Domain Creative Linking

## Status: Design (Reviewed)

*Original draft reviewed by 5-agent deep analysis (March 2026). This version
incorporates findings from creative architecture (Opus), implementation
feasibility, performance analysis, cognitive science validation, and systems
integration analysis.*

## Concept

During the Dream consolidation phase (phase 7), detect entities from different
topic domains that share surprising semantic similarity and create weak
`DREAM_ASSOCIATED` edges between them.

This mirrors how biological REM sleep creates novel cross-domain associations —
the cognitive source of creative insight and unexpected connections.

Engram's Dream phase already boosts edge weights between co-activated entities.
Dream Associations extends this by creating entirely new edges between entities
that are semantically similar but structurally distant.

## Cognitive Science Parallel

During REM sleep, the hippocampus replays recent memories while the neocortex
makes loose, associative connections between them. Unlike waking cognition
(which follows existing pathways), dreaming creates novel bridges between
unrelated memory traces. Research suggests this process underlies:

- Creative problem-solving (e.g., Kekulé's benzene ring dream)
- Emotional integration of new experiences with existing knowledge
- Schema formation across disparate domains

### Sleep Stage Mapping

Engram's consolidation phases map to biological sleep stages:

| Engram Phase | Sleep Stage Analog | Rationale |
|---|---|---|
| triage | Pre-sleep encoding tagging | Hippocampal tagging of memories for consolidation (synaptic tagging hypothesis) |
| replay | NREM3 (SWR replay) | Sharp-wave ripples replay recent episodic traces |
| merge | NREM2 (spindle transfer) | Sleep spindles mediate neocortical integration of replayed traces |
| infer | NREM3 (gist extraction) | SWS extracts statistical regularities from episodes |
| prune | Synaptic homeostasis (SHY) | Downscale weak synapses (Tononi & Cirelli, 2006) |
| compact | Synaptic homeostasis (SHY) | Remove redundant trace information |
| reindex | Transition to REM | Update representations before creative processing |
| **dream** | **REM** | **Associative, cross-domain, emotional** |

### ACT-R Integration

`DREAM_ASSOCIATED` edges interact with ACT-R's spreading activation:

- **Fan effect**: Entities with many dream associations naturally attenuate
  each one via `S_ji = max(fan_s_min, fan_s_max - ln(fan+1))`.
- **ACT-R strategy** (`actr.py`) is 1-hop only — dream associations only
  affect retrieval when one endpoint is directly in working memory. This is
  neurologically apt: dream-formed associations require a direct cue, not
  deep chains of inference.
- Dream edges need an explicit predicate weight (~0.1) to avoid using the
  default 0.5 (see Configuration).

---

## Prerequisite: `valid_to` Temporal Semantics Fix

**Blocking issue.** The entire codebase equates "active relationship" with
`valid_to IS NULL`. A `DREAM_ASSOCIATED` edge with `valid_to = now + 30 days`
is invisible to spreading activation, neighbor lookups, and prune queries.

### Affected Locations

All must change from `valid_to IS NULL` to
`(valid_to IS NULL OR valid_to > datetime('now'))`:

**SQLite** (`storage/sqlite/graph.py`):
- `get_relationships()` — `active_only` filter
- `get_active_neighbors_with_weights()` — neighbor traversal
- `get_neighbors()` — recursive CTE (2 locations)
- `update_relationship_weight()` — weight updates
- `get_co_occurring_entity_pairs()` — co-occurrence detection
- `get_dead_entities()` — prune candidate detection

**FalkorDB** (`storage/falkordb/graph.py`):
- Same set of methods, Cypher equivalents

This fix benefits the entire system beyond dream associations — it enables any
future feature that needs time-bounded edges (fact versioning, provisional
inferences, conflict resolution).

**Recommendation: Ship as a standalone PR before dream associations.**

---

## Algorithm: Surprise Scoring

The original design proposed pairwise cosine similarity with a hard threshold.
Analysis revealed three weaknesses:

1. **Finds the expected, not the surprising.** "Python" and "JavaScript" score
   high but connecting them is tautological, not creative.
2. **Ignores graph structure** beyond a binary 3-hop filter. A pair 2 hops
   apart is far less interesting than a disconnected pair.
3. **Top-N by activation is biased.** Highest-activation entities are already
   well-connected. Dream associations are most valuable for medium-activation
   entities.

### Surprise Score

Replace raw cosine similarity with a composite score:

```
surprise(A, B) = embedding_similarity(A, B)
                 × structural_distance_bonus(A, B)
                 × activation_balance(A, B)
```

```python
def compute_surprise_score(
    sim: float,                   # cosine similarity [0, 1]
    structural_proximity: float,  # PPR proximity [0, 1], 0 = disconnected
    act_a: float,                 # activation of entity A [0, 1]
    act_b: float,                 # activation of entity B [0, 1]
) -> float:
    # Structural bonus: disconnected = max surprise
    structural_bonus = 1.0 - structural_proximity

    # Activation asymmetry penalty
    act_diff = abs(act_a - act_b)
    asymmetry_penalty = 1.0 - (act_diff * 0.7)

    # Medium-activation bonus (bell curve at 0.40)
    act_mean = (act_a + act_b) / 2.0
    mid_bonus = 1.0 - min(1.0, abs(act_mean - 0.40) / 0.35)

    return sim * structural_bonus * asymmetry_penalty * (0.7 + 0.3 * mid_bonus)
```

### Structural Proximity via PPR

Use the existing PPR infrastructure to measure structural reachability as a
continuous value instead of a binary 3-hop check:

```python
async def compute_structural_proximity(
    entity_id: str,
    candidate_ids: set[str],
    graph_store,
    cfg: ActivationConfig,
    group_id: str,
) -> dict[str, float]:
    """PPR from entity_id — returns {candidate_id: proximity [0,1]}."""
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread(
        seed_nodes=[(entity_id, 1.0)],
        neighbor_provider=DreamFilteredProvider(graph_store),  # excludes dream edges
        cfg=cfg,
        group_id=group_id,
    )
    return {cid: min(1.0, bonuses.get(cid, 0.0)) for cid in candidate_ids}
```

**Important:** The proximity computation must exclude existing `DREAM_ASSOCIATED`
edges to prevent dream chains (A→B dream + B→C dream creating a phantom path
that suppresses A→C). Use a `DreamFilteredProvider` wrapper that filters
`DREAM_ASSOCIATED` from `get_active_neighbors_with_weights()`.

For initial implementation, BFS with a 3-hop depth limit is acceptable as a
simpler alternative, upgrading to PPR-based proximity later.

### Full Algorithm

```python
async def _find_dream_associations(self, group_id, graph_store,
                                    search_index, activation_store, cfg):
    now = time.time()

    # 1. Get entities with activation, group by domain
    #    Select medium-activation range (0.10–0.80), sort by midpoint proximity
    domain_buckets = await self._partition_by_domain(
        activation_store, graph_store, group_id, cfg, now
    )

    # 2. Batch-retrieve embeddings for all candidates
    all_ids = [eid for bucket in domain_buckets.values() for eid, _ in bucket]
    embeddings = await search_index.get_entity_embeddings(all_ids, group_id)

    # 3. Cross-domain pairwise similarity (matrix multiplication)
    high_sim_pairs = self._compute_cross_domain_similarities(
        domain_buckets, embeddings, cfg
    )

    # 4. Compute structural proximity for high-similarity pairs
    proximity = await self._batch_structural_proximity(
        high_sim_pairs, graph_store, cfg, group_id
    )

    # 5. Score by surprise, enforce per-domain-pair quotas
    scored = []
    domain_pair_counts = {}
    for src, tgt, sim in high_sim_pairs:
        prox = proximity.get(src, {}).get(tgt, 0.0)
        act_a, act_b = candidate_activations[src], candidate_activations[tgt]
        surprise = compute_surprise_score(sim, prox, act_a, act_b)
        if surprise < cfg.dream_associations_surprise_threshold:
            continue

        pair_key = tuple(sorted([entity_domains[src], entity_domains[tgt]]))
        if domain_pair_counts.get(pair_key, 0) >= cfg.dream_associations_max_per_domain_pair:
            continue
        domain_pair_counts[pair_key] = domain_pair_counts.get(pair_key, 0) + 1
        scored.append((src, tgt, sim, surprise))

    # 6. Top-M by surprise
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:cfg.dream_associations_max_per_cycle]
```

### Candidate Selection

- Activation range: 0.10–0.80 (medium activation, matching dream seed
  selection philosophy)
- Sort by proximity to midpoint (0.40), take top-N per domain
- **Summary gating**: Skip entities with no summary or summary < 20 chars —
  prevents name-only entities from producing spurious matches (e.g., "React"
  the framework vs. "React" the emotion)

---

## Edge Properties

```python
Relationship(
    predicate="DREAM_ASSOCIATED",
    weight=0.1,                    # Very weak
    confidence=similarity_score,   # Embedding similarity
    valid_from=now,
    valid_to=now + timedelta(days=30),  # 30-day TTL
)
```

---

## Lifecycle

### TTL and Access-Based Extension

- Dream associations have a 30-day TTL (`valid_to`).
- If accessed during recall, TTL extends by 30 days — validating the connection.
- Unaccessed associations expire and are cleaned up by the prune phase.
- "Use it or lose it" — matches biological memory consolidation.

### Predicate Graduation

After repeated validation (N accesses or confidence exceeding a threshold), a
dream association should graduate from `DREAM_ASSOCIATED` to `RELATED_TO` with
no TTL. This mirrors systems consolidation theory: initially fragile
hippocampal-dependent associations become stable cortically-consolidated
memories.

### Reconsolidation

When a dream association is accessed during retrieval:
1. **Extend TTL** (reinforcement)
2. **Update weight** based on context — if the query relates to both endpoints,
   increase weight slightly; if only one, leave unchanged
3. **Track co-retrieval frequency** for graduation decisions

---

## Safety Guards

### Dream Drift Prevention (Critical)

The existing Hebbian spreading in the dream phase boosts edge weights for
co-activated entities. Without protection, a dream edge at weight 0.1
gradually grows to 1.0+ over ~50 cycles, becoming indistinguishable from
real edges.

**Fix:** Exclude `DREAM_ASSOCIATED` from Hebbian weight boosting in
`DreamSpreadingPhase._accumulate_edge_boosts()`:

```python
for neighbor_id, _weight, predicate, _etype in neighbors:
    if predicate == "DREAM_ASSOCIATED":
        continue  # Never boost dream edges via Hebbian spreading
```

### Cross-Domain Penalty Exemption (Critical)

Feature 3 (topic-aware spreading) applies a 0.3x penalty when crossing domain
boundaries. Dream edges cross domains by definition. Combined with edge weight
0.1 and predicate weight 0.1, the effective spread is `0.1 × 0.1 × 0.3 × 0.5
= 0.0015` — far below the firing threshold of 0.01.

**Fix:** Exempt `DREAM_ASSOCIATED` from the cross-domain penalty, or apply a
gentler factor:

```python
cross_domain_exempt_predicates: list[str] = ["DREAM_ASSOCIATED"]
# Or: dream_associations_spreading_factor: float = 0.6
```

### Dream Chain Prevention

Dream edges must be excluded when computing structural proximity for new
candidates. Otherwise A→B (dream) + B→C (dream) creates a phantom path that
suppresses A→C, or worse, enables dream-of-dream chains.

### Per-Domain-Pair Quotas

Limit associations per domain combination (default 3) to prevent any single
domain pair from dominating:

```python
dream_associations_max_per_domain_pair: int = 3
```

### Summary Gating

Only consider entities with summaries of 20+ characters. Name-only entities
produce spurious embedding matches.

---

## Incubation Queue

*Novel feature inspired by the incubation effect (Sio & Ormerod, 2009; Cai
et al., 2009) — sleeping on unsolved problems produces solutions.*

When a recall query returns zero results or results below a relevance
threshold, save the query embedding to a `dream_incubation_queue`. During
dream associations, use incubation queries as additional similarity anchors:

- Entity A is similar to incubation query Q
- Entity B is also similar to Q
- A and B are in different domains with no existing path
- → High-priority dream association candidate

This turns "I don't know" into "let me sleep on it."

```python
dream_incubation_enabled: bool = False
dream_incubation_max_queries: int = 5
dream_incubation_window_hours: float = 72.0
dream_incubation_similarity_bonus: float = 0.15
```

Incubation queries expire after one dream cycle or when the user successfully
retrieves relevant information (problem solved).

---

## Integration

### Phase Placement

Added as a sub-step in `DreamSpreadingPhase` (phase 7), after existing
weight-boosting logic:

```python
class DreamSpreadingPhase:
    async def execute(self, ...):
        # Existing: Hebbian weight boosting for co-activated entities
        ...

        # New: create dream associations
        if cfg.dream_associations_enabled:
            associations = await self._find_dream_associations(
                group_id, graph_store, search_index, activation_store, cfg
            )
            for src_id, tgt_id, sim, surprise in associations:
                if not dry_run:
                    rel = Relationship(
                        predicate="DREAM_ASSOCIATED",
                        weight=cfg.dream_associations_weight,
                        confidence=sim,
                        valid_from=datetime.utcnow(),
                        valid_to=datetime.utcnow() + timedelta(
                            days=cfg.dream_associations_ttl_days
                        ),
                        ...
                    )
                    await graph_store.create_relationship(rel)
                    context.dream_association_ids.add(rel.id)
                    context.affected_entity_ids.update({src_id, tgt_id})
```

### CycleContext Interaction

The dream phase should use existing `CycleContext` fields:
- `pruned_entity_ids` — skip pruned entities as targets
- `merge_survivor_ids` — prefer merged survivors
- `inferred_edge_entity_ids` — avoid redundant connections with infer phase

Add new field: `dream_association_ids: set[str]`

### Audit Trail

New `DreamAssociationRecord` dataclass with fields: `cycle_id`, `group_id`,
`source/target_entity_id`, `source/target_entity_name`, `source/target_domain`,
`embedding_similarity`, `structural_proximity`, `surprise_score`,
`explanation`, `relationship_id`.

New consolidation store table: `consolidation_dream_associations`.

---

## Configuration

```python
# ActivationConfig additions

# Core
dream_associations_enabled: bool = False
dream_associations_similarity_threshold: float = 0.65  # Pre-filter (surprise does final scoring)
dream_associations_surprise_threshold: float = 0.25
dream_associations_max_per_cycle: int = 10
dream_associations_max_per_domain_pair: int = 3
dream_associations_ttl_days: int = 30
dream_associations_top_n_per_domain: int = 20
dream_associations_weight: float = 0.1
dream_associations_require_summary: bool = True
dream_associations_min_summary_length: int = 20
dream_associations_max_duration_ms: int = 5000
dream_associations_exclude_from_dream_boost: bool = True

# Spreading
dream_associations_spreading_factor: float = 0.6  # Gentler than cross_domain_penalty (0.3)

# Predicate weights (add to defaults)
# "DREAM_ASSOCIATED": 0.1

# Predicate natural names (add to defaults)
# "DREAM_ASSOCIATED": "dream-associated with"

# Incubation (Tier 2)
dream_incubation_enabled: bool = False
dream_incubation_max_queries: int = 5
dream_incubation_window_hours: float = 72.0
dream_incubation_similarity_bonus: float = 0.15
```

### Profile Presets

```python
# conservative: dream_associations_enabled = False
# standard:     dream_associations_enabled = True, max_per_cycle = 5
# observe:      not applicable (requires embeddings)
```

---

## Performance

At default parameters (N=20, D=4):

| Operation | Cost |
|-----------|------|
| Embedding retrieval (batch SQL) | ~5ms |
| Domain partitioning | ~10ms |
| Pairwise similarity (matrix multiplication) | <1ms |
| Structural proximity (pre-computed, 80 entities) | ~160ms |
| Edge creation (up to 10) | ~10ms |
| **Total** | **~200ms** |

### Requirements

- **Must use matrix multiplication** for similarity — not scalar
  `cosine_similarity()`. At N=1000, scalar takes 30s vs 120ms with matrix ops.
- **Must use batch embedding retrieval** — add `get_entity_embeddings()` to
  `SearchIndex` protocol. Single query with `WHERE id IN (...)`.
- **Pre-compute reachability** before structural distance filtering. Amortize
  80 BFS/PPR queries (~160ms) instead of 2,400 per-pair queries (~4.8s).
- **Add `max_duration_ms` safety valve** (default 5000ms). Abort gracefully if
  exceeded.

### Steady-State Edge Count

With `max_per_cycle=10` and `ttl_days=30`, daily cycles produce ~300 edges at
steady state. In practice, similarity threshold + structural filter + quotas
produce 0-5 per cycle, giving a realistic steady state of ~50-200 edges.

### Memory Footprint

80 vectors × 1024 dims × 4 bytes = 320KB. Negligible at all scales up to
N=200 (3.2MB).

---

## New Infrastructure Required

### GraphStore Protocol Additions

```python
async def path_exists_within_hops(
    self, src_id: str, tgt_id: str, max_hops: int, group_id: str,
) -> bool

async def get_expired_relationships(
    self, group_id: str, predicate: str | None = None, limit: int = 100,
) -> list[Relationship]
```

### SearchIndex Protocol Addition

```python
async def get_entity_embeddings(
    self, entity_ids: list[str], group_id: str,
) -> dict[str, list[float]]
```

Implementations needed in `HybridSearchIndex` (SQLite), `RedisSearchIndex`
(full mode), and `FTS5SearchIndex` (stub returning `{}`).

### Mode Gating

Gate on `embeddings_enabled` (not "full mode"). Lite mode with Voyage AI
embeddings can support this feature. If `get_entity_embeddings()` returns
empty, the feature gracefully does nothing.

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| `valid_to IS NULL` makes dream edges invisible | **Blocking** | Fix temporal semantics globally (standalone PR) |
| Cross-domain penalty kills dream edge spreading | **Critical** | Predicate exemption or separate spreading factor |
| Dream drift via Hebbian weight boosting | **Critical** | Exclude `DREAM_ASSOCIATED` from boost logic |
| Dream chains create phantom connectivity | High | Filter dream edges from proximity computation |
| Spurious matches from name-only entities | High | Summary gating (min 20 chars) |
| Domain collapse via excessive same-pair associations | Medium | Per-domain-pair quota (max 3) |
| Noise amplification if threshold too low | Medium | Surprise scoring (not raw similarity), TTL expiry |
| Prune race — entity pruned same cycle as association | Low | Check `context.pruned_entity_ids` |
| Merge creates duplicate dream edges | Low | Duplicates are low-weight, auto-expire — acceptable |
| Embedding dimension mismatch (model upgrades) | Low | Dream edges are weak and auto-expire — minimal harm |

---

## Example

Given a graph with:
- "The Wound Between Worlds" (CreativeWork) — a novel about interconnected
  realities built from fragments of memory
- "Engram knowledge graph" (Technology) — a system that builds worlds from
  interconnected pieces of conversation

These share high embedding similarity (~0.75) around concepts of
"interconnection", "building from pieces", and "worlds from fragments" — but
have no structural path between them.

Surprise score: `0.75 (sim) × 1.0 (disconnected) × 0.95 (balanced activation)
= 0.71` — well above the surprise threshold of 0.25.

A `DREAM_ASSOCIATED` edge surfaces this connection when querying either entity.

---

## Estimated Effort

~615 lines (revised from original 300 estimate):

| Component | Lines |
|-----------|-------|
| `valid_to` semantics fix (both backends, 6+ locations) | ~30 |
| New protocol methods (3 methods, 2 backends) | ~90 |
| `batch_get_embeddings` (3 implementations) | ~50 |
| Config fields + profile presets | ~30 |
| `DreamAssociationRecord` + consolidation store | ~60 |
| Core algorithm (`_find_dream_associations` + helpers) | ~120 |
| Dream phase integration + safety guards | ~50 |
| Engine dispatch + prune cleanup | ~25 |
| Tests (new feature + `valid_to` regression) | ~200 |

---

## Implementation Order

### Tier 1 — Ship with feature

1. `valid_to` temporal semantics fix (standalone PR, benefits whole system)
2. `batch_get_embeddings` + `DREAM_ASSOCIATED` predicate weight config
3. Core algorithm with surprise scoring + structural proximity
4. Safety guards (drift exclusion, domain-pair quotas, summary gating,
   cross-domain exemption)

### Tier 2 — Next iteration

5. Incubation queue (log unsatisfied queries as dream seeds)
6. Predicate graduation (`DREAM_ASSOCIATED` → `RELATED_TO` after N accesses)
7. Dream journal (non-LLM explanation generation from entity summaries)
8. Adaptive threshold (self-tuning via access feedback loop)

### Tier 3 — Future (implement when data supports it)

9. Negative associations (anti-edges from LLM-rejected similar pairs)
10. Lucid dreaming MCP tool (user-guided association hints)
11. Emotional valence threading (feed `_PERSONAL_PATTERNS` into seed selection)
12. Multi-pass consolidation cycling (later cycles emphasize dream phase)
13. Reconsolidation on access (update weight based on retrieval context)
14. Cross-session dreaming (dream across `group_id` boundaries)

---

## Dependencies

- **Embeddings (Voyage AI)**: Required. Gate on `embeddings_enabled`.
- **Domain groups config**: Reuses `domain_groups` from cross-domain penalty.
- **`_resolve_domain()` helper**: Already in `activation/bfs.py`.
- **PPR infrastructure**: Already in `activation/ppr.py` (for structural
  proximity, Tier 1 can use BFS instead).
- **CycleContext**: Already has `pruned_entity_ids`, `inferred_edge_entity_ids`,
  `merge_survivor_ids` — add `dream_association_ids`.
- **Consolidation store**: Needs new table + save/get methods.

## References

- Stickgold & Walker (2013) — Sleep-dependent memory consolidation
- Wilhelm et al. (2011) — Moderate-activation traces preferentially reactivated
- Tononi & Cirelli (2006) — Synaptic homeostasis hypothesis
- Nader & Hardt (2009) — Memory reconsolidation
- Sio & Ormerod (2009) — Incubation effect meta-analysis
- Cai et al. (2009) — REM sleep and creative problem-solving
- McClelland et al. (1995) — Complementary learning systems
- Anderson & Schooler (1991) — ACT-R decay exponent
