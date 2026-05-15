# Natural Memory Need Signals - Rollout Status

## Status: Phases 0-5 shipped; optional controls remain gated

Research background:
[natural-need-signals-research.md](natural-need-signals-research.md),
[pragmatic-recall-signals.md](pragmatic-recall-signals.md),
[temporal-relational-need-patterns.md](temporal-relational-need-patterns.md)

This document records the shipped analyzer rollout for one part of the
recall rework:
`server/engram/retrieval/need.py` and the hot-path logic in
`analyze_memory_need()`.

The recall rework already defines the larger architecture:

1. detect whether memory is needed
2. build a recall plan
3. retrieve evidence
4. assemble packets
5. deliver packets
6. track usage feedback

This document focuses only on step 1 so recall can trigger on natural
conversational cues instead of explicit keywords. It now serves as a
status document for what shipped, what is live by profile, and which
controls remain optional.

---

## 1. Problem

The current analyzer is explicit-request-biased. It catches "what changed",
"did we decide", and "how's X going", but misses the more important class
of turns where the user assumes shared context.

| Input | Current result | Desired result |
|-------|----------------|----------------|
| "Emma scored two goals!" | `none` | recall |
| "my son had a great game" | `none` | recall |
| "btw Marcus says hi" | `none` | recall |
| "still dealing with that bug" | often `none` | recall |
| "sorry, kid stuff" | `none` | recall |
| "we finally shipped v2" | `none` | recall |
| "talked to Sarah about it" | `none` | recall |

The current question: "Is the user requesting memory?"

The better question: **"Would a person who knows this user respond
differently here?"**

---

## 2. Relationship to recall rework

This doc is an analyzer-focused follow-on to `docs/design/recall-rework.md`.
It should not redefine planner, packet, or feedback architecture.

Key alignment decisions:

1. Keep the recall rework as source of truth for planner, packet assembly,
   delivery policy, and surfaced-vs-used feedback.
2. Keep `need_type` coarse in v1 so the packet layer does not need a
   parallel taxonomy rewrite.
3. Put richer trigger detail into metadata: `trigger_family`,
   `trigger_kind`, `signal_scores`, `detected_entities`.
4. Treat graph grounding as a boost and planner-seeding mechanism first,
   not a separate recall policy.

Coarse `need_type` values stay:

`identity`, `project_state`, `temporal_update`, `open_loop`, `prospective`,
`fact_lookup`, `broad_context`, `none`

Richer trigger metadata travels alongside:

```
trigger_family="pragmatic", trigger_kind="possessive_relational"
trigger_family="pragmatic", trigger_kind="bare_name"
trigger_family="pragmatic", trigger_kind="cross_session_anaphora"
trigger_family="structural", trigger_kind="callback"
trigger_family="structural", trigger_kind="correction"
```

This lets the analyzer become smarter without forcing packet routing to
change at the same pace.

---

## 3. Design constraints

1. **Deterministic hot path.** No model calls.
2. **Highest-leverage misses first.** Natural-personal cases before the
   long tail.
3. **Low-latency default.** Text-only signals run every turn in <2ms.
4. **Storage-agnostic graph grounding.** Cannot be SQLite-only.
5. **Telemetry before tuning.** Shadow metrics before committing to
   thresholds.
6. **Avoid performative recall.** Improve timing, not make the assistant
   look like it is trying too hard.

---

## 4. Signal families and phase assignment

| Family | Purpose | Phase |
|--------|---------|-------|
| Pragmatic presupposition | Detects omitted background that assumes shared knowledge | Shipped in Phase 1 |
| Graph grounding | Confirms the turn touches stored entities or relationships | Shipped in Phase 2 |
| Tier 1 structural patterns | Callback, correction, life update, identity claim, memory gap | Shipped in Phase 3 |
| Shift detection | Domain transitions where entity context should refresh | Shipped in Phase 4 |
| Impoverishment modeling | Predicts whether a memory-free response would be hollow | Shipped in Phase 4 |
| Tier 2-3 pattern catalog | Expanded coverage after telemetry exists | Largely shipped in Phase 5 |
| Optional controls | Adaptive thresholds, graph-only override, chat retry safety net | Implemented, off by default |

This was the rollout order:

- Build presupposition detection first. It catches the highest-value misses
  with the lowest false-positive risk.
- Add graph grounding second. It confirms borderline cases and seeds the
  planner.
- Expand the pattern catalog and dampening after the earlier phases are
  stable.

Current live posture:

- `wave1`: analyzer + structural live
- `wave2`: add graph grounding and planner seeding
- `wave3`: add shift + impoverishment live
- `wave4` / `all`: add prospective memory
- `integration_profile=rework`: normalize to `standard` + `all` and turn on
  cue/projection plus the full natural recall stack

---

## 5. Target architecture

```
Turn arrives
  |
  +-- Layer 0: Short-circuit
  |     Empty turn -> none
  |     Acknowledgement -> none
  |
  +-- Layer 1: Linguistic signals (<2ms, every turn)
  |     +-- Pragmatic signals (shipped)
  |     +-- Structural patterns (shipped)
  |     +-- Shift signals (shipped; can be shadowed by config)
  |     +-- Impoverishment signals (shipped; can be shadowed by config)
  |     +-- Existing keyword patterns (preserved)
  |     |
  |     = linguistic_score: 0.0 - 1.0
  |
  +-- Layer 2: Graph grounding probe (shipped, <8ms, conditional)
  |     Runs for borderline turns and for bounded anchored-graph checks
  |     when graph-only override is explicitly enabled
  |     |
  |     = resonance_score: 0.0 - 1.0
  |     = detected_entities: list[str]
  |
  +-- Decision
  |     linguistic_score >= 0.30                     -> recall
  |     linguistic >= 0.15 AND resonance >= 0.45     -> recall
  |     otherwise                                     -> skip
  |
  +-- Output
        MemoryNeed with:
          - coarse need_type
          - trigger_family + trigger_kind
          - signal_scores
          - detected_entities (for planner seeding)
          - query_hint (best referent for retrieval)
```

Layer 1 runs every turn. Layer 2 is conditional and bounded.

The analyzer output does three things:

1. Decides `should_recall`
2. Produces a coarse `need_type`
3. Attaches metadata for planner seeding and telemetry

---

## 6. Phase 0: Telemetry and scaffolding

**Goal:** Observe richer signals without changing recall behavior.

### Work

1. Extend `MemoryNeed` with analyzer metadata:

```python
@dataclass
class MemoryNeed:
    need_type: str
    should_recall: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)
    query_hint: str | None = None
    urgency: float = 0.0
    packet_budget: int = 1
    entity_budget: int = 3
    # New fields
    signal_scores: dict[str, float] | None = None
    trigger_family: str | None = None
    trigger_kind: str | None = None
    detected_entities: list[str] | None = None
    detected_referents: list[str] | None = None
    resonance_score: float = 0.0
```

2. Create `server/engram/retrieval/signals.py` with signal extraction stubs
   that return typed result objects.
3. Create `server/engram/retrieval/graph_probe.py` with a storage-agnostic
   interface and a no-op fallback.
4. Wire signal extraction into `analyze_memory_need()` to populate
   `signal_scores` on every call, but DO NOT change the return decision.
5. Emit `signal_scores` in `publish_memory_need_analysis()` telemetry.

### Files

- `server/engram/models/recall.py`
- `server/engram/retrieval/need.py`
- `server/engram/retrieval/signals.py`
- `server/engram/retrieval/graph_probe.py`
- `server/tests/test_recall_need.py`

### Acceptance

- Existing keyword-driven behavior is unchanged.
- `MemoryNeed` payloads include `signal_scores` when available.
- We can inspect which turns would have triggered pragmatic signals even
  when recall was skipped.

---

## 7. Phase 1: Pragmatic presupposition detector

**Goal:** Catch the highest-value natural recall moments that the current
keyword cascade misses.

### Signals

Five pragmatic signals, each producing a score in [0, 1]. Combined via
noisy-OR into a single pragmatic composite.

#### P1: Possessive + relational noun

"My son", "our CEO", "his sister" — presupposes the listener knows about
this relationship.

**Detection:** `\b(my|our|his|her|their)\s+RELATIONAL_NOUN\b`

**Relational noun lexicon (4 tiers):**

| Tier | Score | Terms |
|------|-------|-------|
| Family | 0.40 | son, daughter, wife, husband, partner, mom, dad, mother, father, brother, sister, kid, kids, child, children, baby, uncle, aunt, cousin, grandma, grandpa, niece, nephew, fiancee, ex, stepson, stepdaughter |
| Professional | 0.35 | boss, manager, coworker, colleague, teammate, mentor, mentee, intern, lead, director, client, investor, advisor |
| Care | 0.35 | doctor, therapist, dentist, coach, trainer, teacher, professor, tutor, vet |
| Social | 0.30 | friend, roommate, neighbor, classmate, landlord |

**Exclusion: non-relational possessives** (~20 fixed phrases):
my bad, my pleasure, my point, my guess, my take, my turn, my fault,
my way, my god, my goodness, my apologies, my concern, my understanding,
my question, my issue, my opinion

**Exclusion: technical possessives** (~30 terms):
my PR, my branch, my repo, my deployment, my build, my pipeline,
my environment, my terminal, my editor, my config, my package,
my codebase, my function, my endpoint, my server, my API, my stack,
my database, my schema, my query, my container, my cluster, my instance

**Boost:** If a proper name follows ("my son Marcus"), extract it as
`query_hint` and boost score to 0.50.

#### P2: Bare proper name (unintroduced referent)

"Emma scored two goals" — assumes the listener knows who Emma is.

**Detection:** Capitalized name sequence not in `session_entity_names`,
not in communal-ground set, not preceded by an introduction marker.

Reuse existing `_extract_named_terms()` from `need.py`, then filter.

**Communal-ground exclusion** (~200 terms): programming languages,
frameworks, companies, countries, major cities. Expanded from existing
`_NON_ENTITY_TITLE_WORDS`. Only applies to bare mentions — "our React
migration" still fires via P1.

**Introduction markers** (reduce score to 0 — name is being introduced,
not presupposed):

```
(?:named|called)\s+NAME
NAME,?\s+(?:who|which)\s+(?:is|was)
(?:a|this)\s+(?:guy|person|friend|colleague)\s+(?:named|called)\s+NAME
```

**Ambiguous-name set** (~50 common-word names): Will, Grace, Ruby, Mark,
Crystal, Chase, Faith, Hope, June, May, Dawn, Bill, Rich, Art, Pat, etc.
Require secondary signal (relational framing, mid-sentence capitalization,
person-verb like "said"/"thinks") before treating as name.

**Score:** 0.30. Reduced to 0.15 for ambiguous names without secondary
signal.

#### P3: Cross-session anaphora

"She loved it" when no "she" was introduced this session.

**Detection:** Pronouns (she, he, they, it, that, this) with no antecedent
in `session_entity_names` or `recent_turns`.

**Expletive "it" filter** — suppress when it appears in:
- Weather/time: "it's cold", "it's late", "it's raining"
- Pleonastic: "it seems like", "it turns out", "it doesn't matter"
- Discourse: "it's fine", "it's okay", "it depends"

Pattern:
`^it\s+(?:is|'s|was|seems?|looks?|appears?)\s+(?:okay|fine|cold|hot|late|early|raining|obvious|clear|that|like|as)`

**Antecedent check:** Scan `recent_turns` for plausible referents.
For she/he: names or gendered relational nouns. For it/that/this: noun
phrases. If partial antecedent found, halve the score.

**Score:** 0.35 (highest single pragmatic signal).

#### P4: Continuation markers

"Still broken", "finally shipped", "back to the drawing board" —
presupposes a prior state.

| Marker | Pattern |
|--------|---------|
| still (continuative) | `\bstill\s+(?:\w+ing\b\|\w+ed\b\|not\b\|the same)` |
| again (non-imperative) | `\bagain\b` excluding `(try\|run\|say\|do) .* again` |
| finally | `\bfinally\b` |
| back to | `\bback to\b` |
| yet (negated) | `\b(?:hasn't\|haven't\|not) .* \byet\b` |
| already | `\balready\b` |
| no longer | `\bno longer\b` |
| turns out | `\bturns out\b` |

**Session check:** If `recent_turns` explains the continuation
(state mentioned 2-3 turns ago), reduce score by 50%.

**Score:** 0.18 (cross-session), 0.09 (intra-session).

#### P5: Hedged aside with personal content

"Btw Marcus says hi", "sorry, kid stuff" — casual asides that presuppose
shared knowledge. The less explained, the more assumed (the Politeness
Inversion).

**Hedge markers:** oh btw, btw, by the way, anyway, oh and, on another
note, random but, sorry, side note, speaking of, that reminds me, fwiw,
fyi, oh right

**Detection:** Hedge marker + personal content (bare name, relational noun,
or personal-domain term after the hedge).

**Politeness inversion heuristic** — after hedge marker:
- total_words < 8 AND referent_count >= 1: boost 1.3x
- total_words > 20 AND has introduction markers: dampen 0.4x

**Score:** 0.22. With politeness inversion boost: up to 0.29.

### Pragmatic composite

```
pragmatic_score = 1 - (1 - P1)(1 - P2)(1 - P3)(1 - P4)(1 - P5)
```

Noisy-OR — same formula as `plan.py:merge_support()`. Single strong signal
dominates; multiple weak signals compound; saturates toward 1.0.

### Need type mapping

Route pragmatic signals to existing coarse types:

| Trigger kind | `need_type` |
|-------------|-------------|
| possessive_relational | `fact_lookup` |
| bare_name | `fact_lookup` |
| cross_session_anaphora | `fact_lookup` |
| continuation_marker | `open_loop` or `temporal_update` |
| hedged_aside | `fact_lookup` |
| relational + project term | `project_state` |

### Decision rule

```python
# Existing keyword patterns still take priority
if keyword_match:
    return existing_behavior()

# New: pragmatic signals
if pragmatic_score >= 0.25:
    return MemoryNeed(
        need_type=mapped_type,
        should_recall=True,
        confidence=min(0.85, 0.5 + pragmatic_score),
        trigger_family="pragmatic",
        trigger_kind=dominant_signal_kind,
        query_hint=best_referent,
        detected_referents=all_referents,
        signal_scores={"pragmatic": pragmatic_score, ...},
        urgency=0.6,
        packet_budget=1,
        entity_budget=3,
    )

# Existing fallback (confidence lowered when pragmatic signals are present)
none_confidence = 0.82 - (pragmatic_score * 0.3) if pragmatic_score > 0 else 0.82
return MemoryNeed(need_type="none", should_recall=False,
                  confidence=none_confidence, ...)
```

Sub-threshold pragmatic signals (0 < score < 0.25) still lower the
confidence of the "none" verdict. This lets downstream logic make
borderline decisions more carefully.

### Files

- `server/engram/retrieval/need.py`
- `server/engram/retrieval/signals.py`
- `server/tests/test_recall_need.py`

### Acceptance

- `"my son had a great game"` triggers recall (`possessive_relational`)
- `"Emma scored two goals"` triggers recall (`bare_name`)
- `"talked to Sarah about it"` triggers recall (`bare_name`)
- `"still dealing with that bug"` triggers recall (`continuation_marker`)
- `"btw Marcus says hi"` triggers recall (`hedged_aside`)
- `"my PR needs review"` does NOT trigger relational recall
- `"my bad"` does NOT trigger relational recall
- `"Can you write a Python function?"` does NOT trigger recall
- Existing patterns like `"did we decide"` and `"what changed"` behave
  the same

---

## 8. Phase 2: Lightweight graph grounding

**Goal:** Use stored graph structure to confirm borderline linguistic cases
and seed the recall planner.

### Important constraint

In v1, graph grounding should NOT trigger recall from a zero-signal turn.
It lifts or focuses turns that already look somewhat memory-relevant.

This prevents "Can you write a for loop in Python?" from firing recall
just because a `Python` entity exists.

### Mention detection (in-memory, <1ms)

Three-layer detector, cheapest first:

**Layer A: Token-to-entity index.**

In-memory dict mapping lowercased name tokens to entity references. Built
from one query at startup:

```sql
SELECT id, name, entity_type FROM entities WHERE group_id = ?
```

For ~500 entities at ~2 tokens each: ~1000 entries, ~40KB. Invalidated by
a version counter incremented on store/merge. Rebuild on next probe if
stale (<1ms).

For multi-token entities ("Golden Gate Bridge"), index both full name and
individual tokens. Require >= 2/3 token overlap for multi-token match.

**Layer B: Relational noun resolution.**

Map relational nouns to relationship predicates, then query the graph:

```
son/daughter/child  -> CHILD_OF (inverse)
wife/husband/partner -> MARRIED_TO
boss/manager        -> REPORTS_TO
mom/dad             -> CHILD_OF (forward)
friend              -> FRIEND_OF
coworker/colleague  -> WORKS_WITH
brother/sister      -> SIBLING_OF
... (~25 mappings)
```

"My son" -> find entities where `(identity_entity) -[CHILD_OF]-> (?)`.
At most 1 graph query per relational noun (typically 0-1 per turn).

**Layer C: FTS5/CONTAINS fallback.**

If token index misses, fall back to existing
`find_entity_candidates(token, group_id)` for tokens 4+ chars, not
stopwords. Bounded to 3 fallback queries per turn. Works on both SQLite
(FTS5) and FalkorDB (CONTAINS).

### Graph property probe (batch, <2ms)

Once candidate entities are identified, fetch properties in one batch:

```sql
SELECT
    e.id, e.name, e.entity_type, e.memory_tier, e.access_count,
    (SELECT MAX(timestamp) FROM entity_access_log
     WHERE entity_id = e.id) AS last_access,
    (SELECT COUNT(*) FROM relationships
     WHERE source_id = e.id OR target_id = e.id) AS degree,
    (SELECT COUNT(DISTINCT predicate) FROM relationships
     WHERE source_id = e.id OR target_id = e.id) AS predicate_diversity,
    (SELECT COUNT(DISTINCT episode_id) FROM episode_entities
     WHERE entity_id = e.id) AS episode_count
FROM entities e
WHERE e.id IN (?, ?, ...)
```

Two queries total regardless of candidate count. Must work on both storage
backends via `GraphStore` protocol.

### Per-entity resonance scoring

```
density = 0.40 * min(degree / 10, 1.0)
        + 0.30 * min(distinct_predicates / 5, 1.0)
        + 0.30 * min(episode_count / 8, 1.0)

activation_est:
    last_access < 30min  -> 0.9
    last_access < 2hr    -> 0.7
    last_access < 24hr   -> 0.4
    last_access < 7d     -> 0.2
    older / none         -> 0.1

urgency = max tier across relationship types:
    Personal  (CHILD_OF, MARRIED_TO, FRIEND_OF, SIBLING_OF) -> 0.9
    Identity  (PREFERS, IDENTITY_CORE)                       -> 0.9
    Experiential (PLAYS, ATTENDS, WORKS_AT)                  -> 0.7
    Technical (USES, DEPENDS_ON, IMPLEMENTS)                  -> 0.5
    Weak      (MENTIONED_IN, CO_OCCURS_WITH)                 -> 0.2

tier_bonus:
    semantic      -> 0.20
    transitional  -> 0.10
    episodic      -> 0.00

entity_resonance = 0.35 * density
                 + 0.25 * activation_est
                 + 0.25 * urgency
                 + 0.15 * tier_bonus
```

### Match quality modifier

| Match type | Quality |
|-----------|---------|
| Full name exact | 1.00 |
| Full name case-insensitive | 0.95 |
| Multi-token, >= 2/3 tokens | 0.80 |
| Relational noun resolution | 0.90 |
| Single token of multi-token name | 0.40 |
| FTS5/CONTAINS fallback | 0.60 |

### Turn-level aggregation

```
resonance_score = 1 - product(1 - entity_resonance_i * match_quality_i)
```

Noisy-OR across all matched entities. Bonuses (additive, capped at 1.0):

| Bonus | Value | Condition |
|-------|-------|-----------|
| Cross-entity connectivity | +0.15 | Two matched entities share a direct edge |
| Identity proximity | +0.10 | Matched entity within 2 hops of identity core |

### Decision rule

Graph grounding lifts borderline turns. Does not replace linguistic signals:

```python
if linguistic_score >= 0.30:
    recall = True
elif linguistic_score >= 0.15 and resonance_score >= 0.45:
    recall = True
else:
    recall = False
```

When recall is approved and `detected_entities` is non-empty, pass them
into the recall planner as seed entities to focus retrieval on the relevant
subgraph.

### Performance budget

| Step | Time |
|------|------|
| Tokenize + token index lookup | <0.2ms |
| Relational noun resolution + graph query | <2ms |
| FTS5/CONTAINS fallback (0-3 queries) | <3ms |
| Batch property query | <2ms |
| Score computation | <0.1ms |
| **Total** | **<8ms** |

### Files

- `server/engram/retrieval/need.py`
- `server/engram/retrieval/graph_probe.py`
- `server/engram/retrieval/plan.py`
- `server/engram/storage/protocols.py` (if probe needs new graph methods)
- `server/tests/test_recall_need.py`
- `server/tests/test_recall_planner.py`

### Acceptance

- `"my son had a great game"` resolves the child entity even without
  naming the child
- Borderline linguistic turns improve when stored personal entities exist
- `"Can you write a for loop in Python?"` does NOT trigger recall despite
  Python entity existing (linguistic_score too low for graph lift)
- Probe latency stays bounded for both SQLite and full mode
- Planner receives `detected_entities` as seeds when recall is approved

---

## 9. Optional controls and ongoing tuning

Phases 3-5 are no longer future work; they are in the analyzer. The
remaining work is tuning, evaluation, and deciding which optional controls
should ever become default-on.

### Shipped in later phases

1. Tier 1 structural patterns are live.
2. Shift detection and impoverishment modeling are implemented and can run
   live or shadow-only depending on config/profile.
3. Expanded Tier 2-3 coverage and anti-composition dampening are in the
   analyzer and benchmark fixtures.
4. Runtime recall metrics are exposed through `/api/stats`,
   `get_graph_state`, and `engram://graph/stats`.

### Still optional

1. Adaptive thresholds via bounded hit-rate tracking
2. Graph-only override for high-confidence anchored matches
3. Knowledge-chat post-response safety net for generic replies

These controls are implemented but off by default. They should stay gated
until real usage telemetry justifies broader rollout.

---

## 10. File map

### Shipped core files

- `server/engram/models/recall.py` — extend MemoryNeed
- `server/engram/retrieval/need.py` — analyzer changes
- `server/engram/retrieval/signals.py` — signal extraction
- `server/engram/retrieval/graph_probe.py` — graph resonance
- `server/engram/retrieval/plan.py` — planner seeding
- `server/engram/retrieval/context.py` — cached turn features for shift scoring
- `server/engram/retrieval/control.py` — runtime metrics and optional thresholds
- `server/engram/graph_manager.py` — metrics wiring and stats exposure
- `server/engram/mcp/server.py` — auto-recall and stats resource wiring
- `server/engram/api/knowledge.py` — chat analyzer wiring and optional retry
- `server/engram/benchmark/memory_need.py` — analyzer benchmark summaries
- `server/tests/test_recall_need.py`
- `server/tests/test_recall_planner.py`
- `server/tests/test_knowledge_api.py`
- `server/tests/test_api_endpoints.py`
- `server/tests/benchmark/test_memory_need_eval.py`

### Optional-control surfaces

- `server/engram/retrieval/control.py`
- `server/engram/retrieval/need.py`
- `server/engram/api/knowledge.py`

---

## 11. Metrics

### Detection quality

| Metric | Definition | Target |
|--------|-----------|--------|
| Need precision | Of triggered recalls, how many benefited | >55% |
| Need recall | Of beneficial turns, how many triggered | >75% |
| Pragmatic hit rate | Phase 1 signals match real memory moments | track first |
| Graph lift rate | Phase 2 turns borderline into useful recall | track first |

### Behavioral quality

| Metric | Definition | Target |
|--------|-----------|--------|
| Surfaced-to-used ratio | Recalled vs actually used in response | <2.5:1 |
| False recall rate | Irrelevant auto-surfaced packets | <30% |
| Missed personal-connection rate | Personal turns with no recall | <15% |

### Performance

| Metric | Definition | Target |
|--------|-----------|--------|
| Layer 1 p99 | Signal extraction latency | <3ms |
| Graph probe p99 | Layer 2 latency when triggered | <10ms |
| Probe trigger rate | Fraction of turns reaching Layer 2 | 30-50% |

---

## 12. Current rollout posture

1. `recall_profile=wave1` enables analyzer + structural gating.
2. `recall_profile=wave2` adds graph grounding and planner seeding.
3. `recall_profile=wave3` adds shift/impoverishment to live gating.
4. `recall_profile=wave4` and `all` add prospective memory.
5. `integration_profile=rework` makes the whole recall/consolidation/cue loop
   coherent and recall-ready for MCP installs.

Adaptive thresholds, graph-only override, and the knowledge-chat retry
safety net are implemented but remain off by default.

---

## 13. Failure modes

### Common-word entity collisions (Spring, May, Will)

- Ambiguous-name set (~50 entries) requires secondary signal
- Single-token common-word entities with degree < 2: match quality 0.10
- Surrounding-token type checks in graph probe

### Technical possessives (my PR, my branch, my repo)

- Explicit exclusion set in pragmatic detector (~30 terms)
- "my project" defaults to technical unless emotional/identity signals
  co-occur

### Over-eager graph grounding

- Graph cannot trigger recall from a zero-signal turn in the normal path
- Requires linguistic_score >= 0.15 before graph lift
- Graph-only override is gated behind a separate off-by-default flag and
  requires an anchored high-confidence match

### Performative recall

- Keep thresholds explicit and measurable
- Pair with surfaced-vs-used telemetry from recall rework
- Delivery-policy cooldowns deferred to later phase if needed

---

## 14. Summary

Engram should not wait for users to speak in recall keywords. It should
detect when a knowledgeable person would naturally use memory.

The shipped rollout followed the intended path:

1. Add telemetry and scaffolding
2. Ship pragmatic presupposition signals
3. Add graph grounding and planner seeding
4. Bring structural, shift, impoverishment, and expanded catalog coverage
   into the live analyzer
5. Keep adaptive thresholds, graph-only override, and chat retry gated

That gives Engram natural recall in practice without making every
experimental control default-on.
