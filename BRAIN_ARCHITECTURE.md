# Engram Brain Architecture: Unified Design

**Date:** 2026-03-05
**Source:** 5-agent deep dive (2 Opus-level, 3 engineering-focused)
**Goal:** Make Engram a true "one brain per person" — everything exists, not everything is activated at once.

---

## Executive Summary

Six brain-like mechanisms, organized into 4 implementation phases:

| # | Mechanism | Brain Analog | Primary Fix |
|---|-----------|-------------|-------------|
| 1 | Inhibitory Spreading | GABAergic lateral inhibition | Suppress competing/ambiguous retrievals |
| 2 | Emotional Salience | Amygdala pre-conscious appraisal | Fix 30-40% personal content triage bias |
| 3 | State-Dependent Retrieval | Encoding specificity / hippocampal context | Context-aware retrieval (mood, time, domain) |
| 4 | Memory Maturation | Hippocampal → neocortical transfer | Episodic → semantic transition with differential decay |
| 5 | Reconsolidation | Memory lability after reactivation | Recalled entities become updatable |
| 6 | Schema Formation | Cortical schema extraction | Detect recurring structural patterns |

**Estimated new tests:** ~130. **New config fields:** ~45. **New files:** ~6. **Schema changes:** 2 new tables, 3 new columns.

---

## Phase 0: Shared Infrastructure

### 0A. Episode Encoding Context Column

All emotional and state features need per-episode context at encoding time.

```sql
ALTER TABLE episodes ADD COLUMN encoding_context TEXT;
-- JSON: {"arousal": 0.65, "self_ref": 0.87, "social": 0.5, "tension": 0.3,
--         "domain": "personal", "time_bucket": "evening", "cognitive_mode": "reflective"}
```

### 0B. Entity Attributes Namespace Convention

Use existing `attributes` JSON blob with namespaced keys to avoid schema migrations:

| Prefix | Feature | Keys |
|--------|---------|------|
| `emo_` | Emotional salience | `emo_arousal`, `emo_self_ref`, `emo_social`, `emo_composite` |
| `mat_` | Maturation | `mat_score`, `mat_system` (episodic/transitional/semantic) |
| `recon_` | Reconsolidation | `recon_count`, `recon_last` |

### 0C. Scoring Signal Extension

New retrieval signals are **additive bonuses** (like existing `exploration_bonus`, `prime_boost`), not rebalancing of the 5 core weights. This preserves backwards compatibility.

```
score = base_weights(semantic + activation + spreading + edge_prox + graph_structural)
      + exploration_bonus      # existing
      + prime_boost            # existing
      + emotional_boost        # NEW (Phase 1b)
      + state_bias             # NEW (Phase 1c)
      + goal_prime_boost       # NEW (Phase 1b)
      - inhibition_penalty     # NEW (Phase 1a)
```

---

## Phase 1: Independent Features (Parallel)

### 1A. Inhibitory Spreading Activation

**Biological basis:** GABAergic interneurons suppress neighboring neurons competing for the same representational slot. Without inhibition, retrieval is noisy — the brain would hallucinate blended memories.

**Two-layer inhibition:**

#### Layer 1: Predicate-Based Suppression (explicit)

After spreading activation, scan for contradictory predicates among activated entities. Contradictory pairs already defined in extraction (`LIKES/DISLIKES`, `AIMS_FOR/AVOIDS`). Winner-take-all: the stronger predicate group suppresses the weaker.

```python
# After excitatory spreading computes bonuses:
for each contradictory pair (pred_A, pred_B) reachable from same seed:
    group_A_energy = sum(bonuses[n] for n in pred_A_targets)
    group_B_energy = sum(bonuses[n] for n in pred_B_targets)
    losers = pred_B_targets if group_A_energy >= group_B_energy else pred_A_targets
    for n in losers:
        bonuses[n] *= (1.0 - inhibit_strength)  # default 0.3
```

#### Layer 2: Lateral Inhibition via Embedding Proximity (implicit)

Entities semantically similar to seeds but graph-disconnected are suppressed. This resolves polysemy ("Python programming" vs "Python snake").

```python
# Formula:
inhibition(n) = inhibit_strength * seed_similarity(n) * (1.0 - graph_proximity(n))
# Where graph_proximity = 0.5^hop_distance (0 if unreachable)
# Net: bonuses[n] = max(0, excitatory_bonus - inhibition)
```

**Performance:** ~15-30ms. Batch embedding lookup (existing table), O(seeds x candidates) cosine similarity.

**Key design choice:** Dream spreading does NOT apply inhibition — in the brain, dreaming is characterized by *reduced* inhibition (disinhibited replay). This is biologically accurate and prevents dream associations from being prematurely suppressed.

**Config:**
```python
inhibitory_spreading_enabled: bool = False
inhibit_strength: float = 0.3             # 0-1
inhibit_similarity_threshold: float = 0.6  # min cosine sim to trigger
inhibit_max_seed_anchors: int = 5
inhibition_predicate_suppression: bool = True
```

**Files:** `retrieval/pipeline.py` (integration point after step 4), `retrieval/scorer.py` (penalty field), `config.py`
**New files:** `retrieval/inhibition.py`
**Tests:** ~15

---

### 1B. Emotional Salience Scoring

**Biological basis:** The amygdala performs rapid pre-conscious appraisal along arousal (intensity) and self-reference (about me) dimensions in ~120ms. This tag modulates consolidation priority: high-arousal events are replayed more during sleep, resist pruning, and surface more easily during retrieval.

**Core insight:** The problem isn't sentiment analysis — it's that keyword density heuristics reward technical writing markers (capitalization, numbers, quotes) that personal narrative lacks. The fix is a multi-dimensional arousal detector that mirrors the amygdala's appraisal.

#### EmotionalSalience (4 dimensions, all regex, no LLM)

```python
@dataclass
class EmotionalSalience:
    arousal: float = 0.0          # Intensity regardless of valence (state-change verbs, intensifiers)
    self_reference: float = 0.0   # First-person + possessive relations ("my mom", "my project")
    social_density: float = 0.0   # Social role terms + named actors
    narrative_tension: float = 0.0 # Open loops, uncertainty markers (Zeigarnik effect)

    @property
    def composite(self) -> float:
        return 0.35*arousal + 0.30*self_reference + 0.20*social_density + 0.15*narrative_tension
```

**Detection heuristics (all regex, ~0.1ms):**

| Dimension | Detects | Examples |
|-----------|---------|---------|
| Arousal | State-change verbs, intensifiers, punctuation energy | "diagnosed", "realized", "completely", "!!!" |
| Self-reference | I/me/my ratio + possessive relations | "my mom", "my project", "I've been trying" |
| Social density | Role terms + named actors | "mom", "boss", "Sarah" |
| Narrative tension | Uncertainty markers, open loops | "not sure", "working on", "haven't decided" |

#### Revised Triage Formula

```
score = length(0-0.25) + keyword_density(0-0.20) + novelty(0-0.30)
      + emotional_salience(0-0.25)
```

Keyword density ceiling reduced from 0.30 to 0.20. Personal boost removed (subsumed by emotional salience).

**Personal floor guarantee:** If `emotional_salience.composite >= 0.15`, score is floored at 0.45 (guaranteed extraction).

#### Worked Example

"My mom was diagnosed with breast cancer last month. I'm terrified."

| Signal | Old Score | New Score |
|--------|-----------|-----------|
| Length | 0.08 | 0.07 |
| Keywords | 0.09 | 0.06 |
| Novelty | 0.20 | 0.20 |
| Personal boost | 0.15 | — |
| Emotional salience | — | 0.16 (arousal 0.65, self-ref 0.87, social 0.5, tension 0.3) |
| **Total** | **0.52** | **0.49** (floor: 0.49 > 0.45, guaranteed) |

vs. "React v19 introduces Pydantic v2 serialization. See RFC 7519."

| Signal | Old Score | New Score |
|--------|-----------|-----------|
| Length | 0.05 | 0.04 |
| Keywords | 0.24 | 0.16 |
| Novelty | 0.20 | 0.20 |
| Emotional salience | — | 0.00 |
| **Total** | **0.49** | **0.40** |

**Gap widens from 0.03 to 0.09.** Personal content now consistently wins over neutral technical content.

#### Carry-Through Effects

- **Pruning:** Entities with `emo_composite > 0.5` get activation floor boost of 0.15 (resist pruning)
- **Retrieval:** Additive `emotional_retrieval_boost * emo_composite` (default 0.08)
- **Dream spreading:** Emotionally salient entities preferentially selected as dream seeds
- **Extraction:** Salience computed at extraction time, stored in entity attributes

**Config:**
```python
emotional_salience_enabled: bool = True
emotional_triage_weight: float = 0.25
emotional_prune_resistance: float = 0.15
emotional_retrieval_boost: float = 0.08
triage_personal_floor: float = 0.45
triage_personal_floor_threshold: float = 0.15
```

**Files:** `consolidation/phases/triage.py`, `retrieval/scorer.py`, `consolidation/phases/prune.py`, `config.py`
**New files:** `extraction/salience.py`
**Tests:** ~20

---

### 1C. State-Dependent Retrieval

**Biological basis:** Encoding specificity (Tulving, 1973) — retrieval is most effective when internal state at retrieval matches state at encoding. Divers who learned underwater recalled better underwater. The hippocampus binds content + full context envelope.

#### CognitiveState (inferred, no explicit user input)

```python
@dataclass
class CognitiveState:
    arousal_level: float = 0.3        # EMA of emotional salience in recent episodes
    mode: str = "neutral"             # "task" | "reflective" | "exploratory" | "neutral"
    domain_weights: dict[str, float]  # {"personal": 0.6, "technical": 0.3}
    time_of_day_bucket: str           # "morning" | "afternoon" | "evening" | "night"
    session_duration_minutes: float
```

**Signals (all inferred from usage patterns):**

| Signal | Source | Update |
|--------|--------|--------|
| Arousal trajectory | EMA of salience.composite on each observe/remember | Per episode |
| Cognitive mode | Query pattern (short factual → task; "tell me about" → exploratory) | Per recall |
| Domain distribution | Entity type distribution in last N episodes | Per episode |
| Time of day | `datetime.now().hour` bucketed | Per session |

**Retrieval bias (additive, ~0.5ms):**

```python
state_boost = state_domain_weight * domain_affinity
            + state_arousal_match_weight * (1.0 - abs(current_arousal - entity_arousal))
```

**Key insight:** This pairs with emotional salience — the arousal level at encoding time (stored in entity attributes) is matched against current arousal level. High-arousal state → retrieve high-arousal memories.

**Config:**
```python
state_dependent_retrieval_enabled: bool = True
state_domain_weight: float = 0.06
state_arousal_match_weight: float = 0.04
state_arousal_ema_alpha: float = 0.3
```

**Files:** `retrieval/pipeline.py`, `retrieval/context.py`, `config.py`
**New files:** `retrieval/state.py`
**Tests:** ~15

---

### 1D. Goal-Relevance Gating

**Biological basis:** The prefrontal cortex maintains active goals in working memory and biases what the hippocampus encodes. Goal-relevant information gets preferential consolidation and retrieval. The Zeigarnik effect: incomplete goals maintain their attentional hold.

#### Active Goal Identification

```python
active_goals = entities where:
    entity_type in ("Goal", "Intention")
    AND activation >= goal_activation_threshold (0.15)
    AND NOT completed/abandoned (no COMPLETED/ABANDONED edges)
    AND (for Intentions) enabled=True and not expired
```

#### Dual influence:

**1. Triage boost:** Episodes mentioning goal-adjacent entities get +0.10 triage score.

**2. Retrieval priming:** Before spreading activation, active goals are injected as low-energy seeds. Their 1-hop neighbors get micro-boosts.

```python
# Goal priming (cached with 60s TTL):
for each active_goal:
    inject goal as seed with energy = goal_priming_boost * goal_strength
    for each 1-hop neighbor:
        inject with energy * 0.3
```

**3. Prune protection:** Entities within 1-hop of active goals are exempt from pruning.

**4. Natural decay:** Goals decay via ACT-R activation (not mentioned → drops below threshold). Completion detected via COMPLETED/ABANDONED predicates in extraction.

**Config:**
```python
goal_priming_enabled: bool = False
goal_priming_boost: float = 0.10
goal_priming_activation_floor: float = 0.15
goal_priming_max_goals: int = 5
goal_priming_max_neighbors: int = 10
goal_priming_cache_ttl_seconds: float = 60.0
goal_triage_weight: float = 0.10
goal_prune_protection: bool = True
```

**Files:** `retrieval/pipeline.py`, `consolidation/phases/triage.py`, `consolidation/phases/prune.py`, `config.py`
**New files:** `retrieval/goals.py`
**Tests:** ~15

---

## Phase 2: Memory Maturation + Reconsolidation (Sequential)

### 2A. Memory Maturation (Episodic → Semantic Transition)

**Biological basis:** The hippocampus holds episodic memories (time-stamped, contextually rich). Through repeated reactivation during sleep (sharp-wave ripples), traces transfer to the neocortex as semantic knowledge (decontextualized facts). Episodic memories decay faster but are vivid. Semantic memories are durable but schematic.

#### Dual-Layer Transition

**Entity maturation** (what facts become durable):

```
episodic → transitional → semantic
```

Entities graduate based on a maturation score:

```python
maturation_score = (
    0.30 * source_diversity       # mentioned in N distinct episodes
    + 0.25 * temporal_span        # referenced across multiple days
    + 0.25 * relationship_richness # connected to other entities
    + 0.20 * access_regularity    # accessed at regular intervals (not bursts)
)
```

Thresholds: `transitional >= 0.42` (60% of full), `semantic >= 0.70`
Minimum age: `transitional: 7 days`, `semantic: 5 consolidation cycles`

**Episode transition** (when raw content can be compressed):

```
episodic → transitional → semantic
```

Episodes transition when their linked entities are well-established (access_count >= threshold):
- `transitional`: 50% entity coverage after 2 cycles
- `semantic`: 85% entity coverage after 5 cycles

On semantic transition: content is summarized, original content hash preserved for audit.

#### Differential Decay

```python
# ACT-R B_i(t) = ln(consolidated_strength + sum((t - t_j)^(-d)))
# d varies by memory system:
episodic_decay_exponent: 0.5    # fast decay (standard ACT-R)
transitional_decay_exponent: 0.4
semantic_decay_exponent: 0.3    # slow decay

# After 30 days: episodic retains ~18%, semantic retains ~31%
# After 90 days: episodic ~10%, semantic ~22%
```

#### Decontextualization

When an entity graduates to semantic, its summary is stripped of temporal/spatial specifics:

```
"Discussed Python migration at standup on March 1" →
"Interested in Python migration"
```

Episodic context is preserved in `mat_episodic_context` attribute for audit.

#### Prune Phase Enhancement

- Entities linked to `episodic` episodes: protected from pruning (source still "vivid")
- `semantic` entities: pruned at 180 days (vs 30 for episodic)
- `identity_core` entities: auto-graduated to semantic regardless of maturation score

#### Schema Changes

```sql
-- Episodes
ALTER TABLE episodes ADD COLUMN memory_tier TEXT NOT NULL DEFAULT 'episodic';
ALTER TABLE episodes ADD COLUMN consolidation_cycles INTEGER NOT NULL DEFAULT 0;
ALTER TABLE episodes ADD COLUMN entity_coverage REAL NOT NULL DEFAULT 0.0;
CREATE INDEX IF NOT EXISTS idx_episodes_tier ON episodes(memory_tier);
```

Entity maturation lives in `attributes` JSON (no column changes).

#### New Consolidation Phases

**MaturationPhase** (entities): After `compact`, before `reindex`.
**SemanticTransitionPhase** (episodes): After `MaturationPhase`.

```
triage → merge → infer → replay → prune → compact → mature → semanticize → reindex → graph_embed → dream
```

Both phases run on the **cold tier** (every 6 hours).

**Config:**
```python
memory_systems_enabled: bool = False
episodic_decay_exponent: float = 0.5
semantic_decay_exponent: float = 0.3
maturation_threshold: float = 0.7
maturation_min_sources: int = 3
maturation_min_age_days: int = 7
episodic_prune_age_days: int = 14
semantic_prune_age_days: int = 180
semantic_compress_enabled: bool = True
semantic_entity_established_threshold: int = 5
```

**Files:** `activation/engine.py` (differential decay), `consolidation/phases/prune.py`, `config.py`
**New files:** `consolidation/phases/mature.py`, `consolidation/phases/semanticize.py`
**Tests:** ~20

---

### 2B. Reconsolidation on Recall

**Biological basis:** When a memory is reactivated, it enters a labile state (1-6 hours) where it can be modified by new information. Demonstrated by Nader et al. (2000). Not a bug — it's how the brain keeps memories current without explicit correction.

#### LabileWindowTracker (in-memory TTL cache)

```python
class LabileWindowTracker:
    _entries: dict[str, LabileEntry]  # entity_id → entry
    _ttl: float = 300.0              # 5 minute window
    _max_entries: int = 50

    def mark_labile(entity_id, name, type, summary, query): ...
    def get_labile(entity_id) -> LabileEntry | None: ...
    def record_modification(entity_id): ...
```

#### Flow

```
recall("What does Sarah work on?")
  → Returns Sarah entity (activation recorded)
  → Sarah enters labile buffer (5 min TTL)

observe("Sarah just told me she moved to Anthropic")
  → Extraction finds Sarah entity
  → Check labile buffer: Sarah IS labile
  → Reconsolidation: summary enriched, WORKS_AT updated
  → Modification count incremented (max 3 per window)
```

#### Scoping Rules (prevent drift)

| Field | Labile Update | Protected |
|-------|--------------|-----------|
| Summary | Enrichment (append + cap at 500 chars) | — |
| Attributes | New keys added, existing keys updated | — |
| Entity type | — | Always protected |
| Entity name | — | Always protected |
| identity_core entities | Summary enrichment only | Attributes frozen |

#### Safety Mechanisms

- **Window not extended:** Recalling the same entity again doesn't reset the timer
- **Modification budget:** Max 3 updates per window (prevents telephone-game drift)
- **Content overlap check:** New info must have >10% Jaccard token overlap with existing summary (prevents unrelated updates)
- **Activation boost:** Reconsolidated entities get a bonus access record (reconsolidation-enhancement effect)
- **Full mode:** Replace dict with Redis `SETEX` + `GET` for multi-process support

#### Interaction with Maturation

Reconsolidation accelerates maturation: entities that are recalled and then updated demonstrate active use and relevance. Reconsolidation count (`recon_count` in attributes) feeds into the maturation score.

**Config:**
```python
reconsolidation_enabled: bool = False
reconsolidation_window_seconds: float = 300.0
reconsolidation_max_modifications: int = 3
reconsolidation_max_entries: int = 50
reconsolidation_overlap_threshold: float = 0.1
```

**New event:** `entity.reconsolidated`
**Files:** `graph_manager.py` (recall path + ingest path), `config.py`
**New files:** `retrieval/reconsolidation.py`
**Tests:** ~25

---

## Phase 3: Schema Formation (Depends on Phase 2)

**Biological basis:** Over time, the neocortex forms abstract schemas — "restaurant script" (enter, sit, order, eat, pay), "debugging workflow" (reproduce, isolate, fix, verify). These are meta-patterns extracted from many similar episodic experiences.

### Motif Detection (Practical, Not NP-Hard)

Exact subgraph isomorphism is O(NP-hard). Instead, detect motifs via **predicate-type fingerprints** — the structural "shape" of local neighborhoods, reduced to hash-based frequency counting.

```python
# For each entity, compute its structural fingerprint:
fingerprint = frozenset of (source_type, predicate, target_type) triples
# e.g., {(Person, EXPERT_IN, Technology), (Person, MEMBER_OF, Organization)}

# Count fingerprint occurrences across all entities:
motif_counter: dict[frozenset, list[entity_id]]

# Promote to Schema when occurrences >= schema_min_instances (default 5)
```

### Schema Entities

Schemas are first-class entities with `entity_type="Schema"`. Instance entities are connected via `INSTANCE_OF` edges.

```sql
CREATE TABLE IF NOT EXISTS schema_members (
    schema_entity_id TEXT NOT NULL REFERENCES entities(id),
    role_label       TEXT NOT NULL,
    member_type      TEXT NOT NULL,
    member_predicate TEXT NOT NULL,
    group_id         TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (schema_entity_id, role_label, group_id)
);
```

### Phase Ordering

```
... → semanticize → schema → reindex → graph_embed → dream
```

After `semanticize` (needs mature entities for clean motifs), before `reindex` (new Schema entities need indexing).

### Reinforcement

On subsequent cycles, existing schemas get activation recorded (reinforced) rather than recreated. New instances are linked via `INSTANCE_OF`.

**Config:**
```python
schema_formation_enabled: bool = False
schema_min_instances: int = 5
schema_min_edges: int = 2
schema_max_per_cycle: int = 5
```

**Files:** `consolidation/engine.py` (register phase), `config.py`
**New files:** `consolidation/phases/schema.py`
**Tests:** ~25

---

## Dependency Matrix

```
         1.Inhib  2.Emot  3.State  4.Mature  5.Recon  6.Schema  7.Goals
1.Inhib    --      no      no       no        no       no        no
2.Emot     no      --      MEDIUM   weak      weak     no        no
3.State    no      MEDIUM  --       no        no       no        no
4.Mature   no      weak    no       --        STRONG   STRONG    no
5.Recon    no      weak    no       STRONG    --       weak      no
6.Schema   no      no      no       STRONG    weak     --        no
7.Goals    no      weak    no       no        no       no        --
```

**Independent:** Inhibitory (1), Goals (7)
**Paired:** Emotional + State (2+3 share arousal encoding)
**Tightly coupled:** Maturation + Reconsolidation (4+5 — reconsolidation IS the graduation mechanism)
**Dependent:** Schema (6) requires Maturation (4)

---

## Implementation Timeline

```
Phase 0 (2 days):   Infrastructure (encoding_context column, attributes conventions)
Phase 1 (1 week):   [1A Inhibitory] + [1B Emotional + Goals] + [1C State] — PARALLEL
Phase 2 (1.5 weeks): [2A Maturation] + [2B Reconsolidation] — SEQUENTIAL (same developer)
Phase 3 (1 week):   [3A Schema Formation] — DEPENDS on Phase 2
```

### New Pipeline Order (13 phases)

```
triage → merge → infer → replay → prune → compact → mature → semanticize → schema → reindex → graph_embed → dream
```

### Tiered Scheduling

| Tier | Interval | Phases |
|------|----------|--------|
| Hot  | 15 min   | triage, compact |
| Warm | 2 hours  | merge, infer, reindex |
| Cold | 6 hours  | replay, prune, mature, semanticize, schema, graph_embed, dream |

---

## Files Changed Summary

| File | Changes |
|------|---------|
| `server/engram/config.py` | ~45 new config fields across all features |
| `server/engram/consolidation/engine.py` | Register 3 new phases, handle new audit record types |
| `server/engram/consolidation/phases/triage.py` | New scoring formula with emotional salience + goal relevance |
| `server/engram/consolidation/phases/prune.py` | Memory-tier-aware + emotional + goal prune resistance |
| `server/engram/retrieval/pipeline.py` | Inhibitory spreading, goal priming, state bias integration |
| `server/engram/retrieval/scorer.py` | New additive signals (emotional, state, goal, inhibition penalty) |
| `server/engram/retrieval/context.py` | CognitiveState tracking in ConversationContext |
| `server/engram/graph_manager.py` | Reconsolidation labile tracker, goal priming cache |
| `server/engram/activation/engine.py` | Differential decay by memory system |
| `server/engram/worker.py` | Updated scoring to match new triage formula |
| `server/engram/storage/sqlite/schema.sql` | episodes columns, schema_members table |
| `server/engram/extraction/prompts.py` | New predicates (COMPLETED, ABANDONED, WORKS_TOWARD, BLOCKS) |

### New Files

| File | Purpose |
|------|---------|
| `server/engram/extraction/salience.py` | EmotionalSalience computation (regex-only) |
| `server/engram/retrieval/inhibition.py` | Predicate + lateral inhibition |
| `server/engram/retrieval/state.py` | CognitiveState inference + state-dependent bias |
| `server/engram/retrieval/goals.py` | Active goal identification + goal priming cache |
| `server/engram/retrieval/reconsolidation.py` | LabileWindowTracker + reconsolidation logic |
| `server/engram/consolidation/phases/mature.py` | Entity maturation (episodic → semantic) |
| `server/engram/consolidation/phases/semanticize.py` | Episode tier transition + compression |
| `server/engram/consolidation/phases/schema.py` | Motif detection + schema entity creation |

---

## Profile Integration

### Consolidation Profile Updates

| Feature | off | observe | conservative | standard |
|---------|-----|---------|-------------|----------|
| emotional_salience | on | on | on | on |
| inhibitory_spreading | off | off | on | on |
| memory_systems | off | dry-run | on | on |
| reconsolidation | off | off | off | on |
| schema_formation | off | off | off | on |
| goal_priming | off | off | on | on |
| state_dependent | off | off | on | on |

Emotional salience is always-on because it's pure regex (~0.1ms) and fixes the critical triage bias.

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Inhibition false-positives (suppress related entities) | MEDIUM | High cosine threshold (0.6), exempt graph-connected entities |
| Emotional detection too crude (regex vs NLU) | LOW | Designed for precision not recall — complements LLM judge in standard profile |
| Reconsolidation drift (telephone game) | MEDIUM | 3-modification budget, content overlap check, window not extended |
| Schema detection O(N^2) | MEDIUM | Offline consolidation only, cold tier (6hr), cap entities per cycle |
| Score bloat from additive bonuses | LOW | All new signals gated by feature flags, conservative defaults |
| Premature semantic graduation | MEDIUM | Multi-signal maturation score, min age 7 days, transitional buffer |
