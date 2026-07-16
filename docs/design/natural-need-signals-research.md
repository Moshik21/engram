# Natural Memory Need Signals - Unified Design

> Status: implemented — design realized in signals.py/need.py; signal controls gated off by default


## Status: Proposed Build Path

Synthesized from five independent research threads: semantic shift detection,
entity-graph resonance, conversational pragmatics, absence-aware response
modeling, and temporal-relational pattern analysis.

**Target file:** `server/engram/retrieval/need.py` - `analyze_memory_need()`

---

## 1. Problem

The current `analyze_memory_need()` uses a keyword-regex cascade to detect
when recall is needed. It catches explicit requests - "what changed", "do you
remember", "how's X going" - and returns `need_type="none"` for everything
else.

This misses the vast majority of turns where memory matters:

| Input | Current result | Correct result |
|-------|---------------|----------------|
| "Emma scored two goals!" | `none` | recall (presupposed referent) |
| "my son had a great game" | `none` | recall (relational presupposition) |
| "btw Marcus says hi" | `none` | recall (hedged aside, bare name) |
| "still dealing with that bug" | `none` (no recent_turns) | recall (continuation + anaphora) |
| "sorry, kid stuff" | `none` | recall (politeness inversion) |
| "we finally shipped v2" | `none` | recall (milestone, entity) |
| "talked to Sarah about it" | `none` | recall (bare name, anaphora) |

The system asks: "Is the user requesting memory?"

The right question: **"Would a person who knows this user respond differently here?"**

---

## 2. Design Thesis

Memory need is detectable from five orthogonal signal families. No single
family is sufficient. Combined via noisy-OR, they cover ~85% of natural
recall-worthy turns that the keyword cascade misses.

The five families:

| Family | What it detects | Cost |
|--------|----------------|------|
| **Pragmatic** | What the speaker doesn't say - presupposed shared knowledge | <1ms |
| **Temporal-Relational** | Structural patterns that predict recall regardless of domain | <0.5ms |
| **Shift** | Domain boundary crossings where the entity set needs refreshing | <1ms |
| **Impoverishment** | Turns where the memory-free response would be hollow | <0.5ms |
| **Graph Resonance** | Whether mentioned concepts have stored graph structure | <8ms |

The first four are pure text analysis (Layer 1). Graph resonance requires a
graph probe (Layer 2) and only runs when Layer 1 detects something worth
checking.

---

## 3. Architecture

```
Turn arrives
  |
  +-- Layer 0: Short-circuit
  |     Empty turn -> none
  |     Acknowledgement -> none
  |
  +-- Layer 1: Linguistic Signal Extraction (~2ms, every turn)
  |     +-- Pragmatic signals
  |     +-- Temporal-relational signals
  |     +-- Shift signals (vs rolling window)
  |     +-- Impoverishment signals
  |     +-- Existing keyword patterns (preserved)
  |     |
  |     +-- Signal composition (noisy-OR across categories)
  |     +-- Anti-composition dampening
  |     +-- Session position modifier
  |     |
  |     = linguistic_score: 0.0-1.0
  |
  +-- Layer 2: Graph Resonance Probe (~5-8ms, conditional)
  |     Only runs when linguistic_score > 0.15
  |     +-- Token-to-entity index lookup
  |     +-- Relational noun resolution
  |     +-- Batch graph property query
  |     +-- Resonance scoring
  |     |
  |     = resonance_score: 0.0-1.0
  |
  +-- Ensemble
  |     combined = max(
  |       linguistic_score,
  |       resonance_score * 0.85,     # graph can override linguistics
  |       keyword_legacy_score * 0.70 # existing patterns still work
  |     )
  |
  +-- Threshold (0.30)
        combined >= 0.30 -> recall
        combined <  0.30 -> skip
```

### Why Two Layers

Layer 1 runs on every turn. It must be <2ms. No graph queries, no model
calls. Pure regex + small set lookups + arithmetic.

Layer 2 checks the graph. It's the only signal grounded in actual stored
knowledge (not just linguistic surface). But it costs 5-8ms, so it only runs
when Layer 1 says "there might be something here." The threshold (0.15) is
deliberately low - it's cheaper to probe the graph unnecessarily than to miss
a recall-worthy turn.

### Why Graph Resonance Can Override

A turn like "my son had a great game" may score moderate on linguistics (0.40
from pragmatic signals) but very high on graph resonance (0.85 if the son
entity has rich stored relationships). The override path
(`resonance_score * 0.85`) lets the graph drive the recall decision even when
linguistic signals are modest. This is the architectural decision that solves
the "my son" problem.

---

## 4. Layer 1: Linguistic Signal Families

### 4.1 Pragmatic Signals

Based on what the speaker presupposes the listener already knows. The core
insight from Grice's Cooperative Principle: speakers omit background they
assume is shared. The omission is the signal.

#### Signal P1: Possessive + Relational Noun

**Detection:** `\b(my|our|his|her|their)\s+RELATIONAL_NOUN\b`

The single highest-value detector the current system misses. "My son" is not
introducing a son - it's assuming you know about the son.

**Relational noun lexicon (~70 terms, 4 tiers):**

| Tier | Weight | Terms |
|------|--------|-------|
| Family | 0.40 | son, daughter, wife, husband, partner, mom, dad, mother, father, brother, sister, kid, kids, child, children, baby, uncle, aunt, cousin, grandma, grandpa, niece, nephew, fiancee, ex, stepson, stepdaughter |
| Professional | 0.35 | boss, manager, coworker, colleague, teammate, mentor, mentee, intern, lead, director, client, investor, advisor |
| Medical/Care | 0.35 | doctor, therapist, dentist, coach, trainer, teacher, professor, tutor, vet |
| Social | 0.30 | friend, roommate, neighbor, classmate, landlord |

**Exclusion set** (non-relational possessives, ~25 terms): "my bad", "my
pleasure", "my point", "my guess", "my take", "my turn", "my fault", "my
way", "my god", "my goodness", "my apologies", "my concern", "my
understanding", "my question", "my issue", "my opinion"

**Technical possessive exclusion** (~35 terms): "my PR", "my branch", "my
repo", "my deployment", "my build", "my pipeline", "my environment", "my
terminal", "my editor", "my config", "my package", "my codebase", "my
function", "my endpoint", "my server", "my API", "my stack", "my database",
"my schema", "my query", "my container", "my cluster", "my instance"

**Score:** 0.25-0.40 depending on tier. If a proper name follows ("my son
Marcus"), extract it as query_hint and boost to 0.50.

#### Signal P2: Bare Proper Name (Unintroduced Referent)

**Detection:** Capitalized name sequence not in `session_entity_names`, not
in communal-ground set, not preceded by an introduction marker.

When someone says "Emma's got a tournament this weekend," they assume you
know who Emma is. A stranger would hear "my daughter Emma, who plays soccer."

**Communal-ground exclusion set** (~200 terms): major programming languages,
frameworks, companies, countries, cities. Expanded from existing
`_NON_ENTITY_TITLE_WORDS`. Only applies to bare mentions - "our React
migration" still fires via possessive framing.

**Introduction markers** (reduce score to 0 - the name is being introduced,
not presupposed):
- "my friend X", "X, who is...", "a guy named X", "this person X"
- Pattern: `(?:named|called)\s+NAME` or `NAME,?\s+(?:who|which)\s+(?:is|was)`

**Ambiguous-name set** (~50 terms that are also common words): Will, Grace,
Ruby, Mark, Crystal, Chase, Faith, Hope, June, May, Dawn, Bill, Rich, Art,
Pat, etc. Require secondary signal (relational framing, mid-sentence
capitalization, person-verb like "said/thinks") before treating as name.

**Score:** 0.30. Reduced to 0.15 for ambiguous names without secondary signal.

#### Signal P3: Cross-Session Anaphora

**Detection:** Pronouns (she, he, they, it, that, this) with no antecedent
in `session_entity_names` or `recent_turns`.

"She loved it" when no "she" was introduced this session - the speaker is
referencing across sessions. This is the strongest presupposition signal
because the utterance is literally uninterpretable without memory.

**Expletive "it" filter:** Suppress when `it` appears in:
- Weather/time: "it's cold", "it's late", "it's raining"
- Pleonastic: "it seems like", "it turns out", "it doesn't matter"
- Discourse: "it's fine", "it's okay", "it depends"
- Pattern: `^it\s+(?:is|'s|was|seems?|looks?|appears?)\s+(?:okay|fine|cold|hot|late|early|raining|obvious|clear|that|like|as)`

**Antecedent check** (cheap scan of recent_turns):
- For `she/he`: look for proper names or gendered relational nouns
- For `it/that/this`: look for noun phrases (capitalized sequences, quoted)
- If partial antecedent found, halve the score

**Score:** 0.35 (highest single pragmatic signal).

#### Signal P4: Definite Reference to Unintroduced Entity

**Detection:** `the + NOUN_PHRASE` where the noun phrase hasn't appeared in
`recent_turns` or `session_entity_names`.

"The migration is almost done" presupposes shared knowledge of a specific
migration. "The proposal looks good" presupposes a known proposal.

**Discourse-structural filter:** Exclude "the thing is", "the point is",
"the problem is", "the question is", "the fact that", "the way I see it",
"the truth is", "the issue is" when followed by a clause.

Pattern: `\bthe\s+(?:thing|point|problem|question|fact|way|truth|issue|idea|reason|matter)\s+(?:is|was|being)\b`

**Communal-generic filter:** Exclude "the internet", "the web", "the cloud",
"the market", "the government", "the world", "the weekend", "the office"
(without specific context).

**Score:** 0.20.

#### Signal P5: Hedged Aside with Personal Content

**Detection:** Aside/hedge marker + personal content (bare name, relational
noun, or personal-domain term).

**Hedge markers:** "oh btw", "btw", "by the way", "anyway", "oh and",
"also" (turn-initial), "on another note", "random but", "sorry", "side
note", "speaking of", "that reminds me", "incidentally", "fwiw", "fyi",
"oh right"

**The Politeness Inversion:** The less someone explains, the more they assume
you know. "Sorry, kid stuff" presupposes more shared knowledge than "I need
to leave because my daughter has a dance recital."

Detection heuristic: after hedge marker, count words and referent-nouns.
- total_words < 8 AND referent_count >= 1: high presupposition (score * 1.3)
- total_words > 20 AND contains introduction markers: low presupposition (score * 0.4)

**Score:** 0.22. With politeness inversion boost: up to 0.29.

#### Signal P6: Continuation Presupposition

**Detection:** Markers that presuppose a prior state the listener should know.

| Marker | Presupposition | Pattern |
|--------|---------------|---------|
| still | State persists | `\bstill\s+(?:\w+ing\b|\w+ed\b|not\b|the same)` |
| again | Recurrence | `\bagain\b` (exclude imperative: "try again") |
| finally | Long-awaited resolution | `\bfinally\b` |
| anymore | Cessation | `\b(?:not|don't|doesn't).*anymore\b` |
| back to | Return to prior state | `\bback to\b` |
| yet | Expected completion not reached | `\b(?:hasn't|haven't|not)\s+.*\byet\b` |
| already | Completed before expected | `\balready\b` |
| no longer | Cessation | `\bno longer\b` |
| turns out | Expectation revision | `\bturns out\b` |

**Session-context check:** If `recent_turns` provides context for the
continuation (the prior state was mentioned 2-3 turns ago), this is
intra-session - reduce score by 50%.

**"still" disambiguation:** `still` as adjective ("still water") vs adverb
of continuation. Only match continuative pattern (still + verb/adjective
predicate).

**Score:** 0.18 (cross-session), 0.09 (intra-session).

#### Pragmatic Composite

```
P = 1 - (1 - P1)(1 - P2)(1 - P3)(1 - P4)(1 - P5)(1 - P6)
```

Noisy-OR. Properties: single strong signal dominates; multiple weak signals
compound; saturates toward 1.0.

### 4.2 Temporal-Relational Pattern Signals

Based on recurring conversational shapes that reliably predict recall value
regardless of domain. These are structural - detected from how the utterance
is shaped, not what keywords appear.

#### 18 Pattern Types

Each pattern has a detection regex, a confidence score, and maps to a
`need_type`. Patterns are organized into 3 implementation tiers:

**Tier 1 - Build First (highest impact, lowest false positive risk):**

| Pattern | Confidence | Detection Core |
|---------|-----------|----------------|
| **Callback** | 0.92 | `(remember when|going back to|you (said|mentioned|suggested)|we (talked|discussed|decided)|like (we|you) said)` |
| **Memory Gap** | 0.90 | `(i (can't|don't) remember|i forget|not sure (if|what|whether)|can't recall|there was (a|some) reason)` |
| **Correction** | 0.88 | `(actually|correction|to clarify|that's (changed|wrong|not right)|scratch that|i was wrong|not anymore)` + contrastive: `not X but Y`, `switched from X to Y` |
| **Life Update** | 0.85 | Possessive/relational noun + state-change verb (started, stopped, quit, joined, moved, married, graduated, retired, promoted, adopted, born, died, enrolled, switched, launched, sold, bought) |
| **Identity Claim** | 0.85 | `(i'm (more of|kind of|the type|really|basically)|i consider myself|i've always been|i see myself as|i'm a .* (person|type|developer))` |
| **Status Check** | 0.84 | Question word + entity/project reference + progress verb. Already partially covered by `_TEMPORAL_PATTERNS` - strengthen with entity co-occurrence requirement |

**Tier 2 - Build Second:**

| Pattern | Confidence | Detection Core |
|---------|-----------|----------------|
| **Social Graph Update** | 0.84 | Known name + role/relationship change verb (`got promoted|is leaving|is now|became|aren't .* anymore`) |
| **Recurring Problem** | 0.84 | `(again|same .* (issue|problem|bug)|keeps (happening|breaking|failing)|third time|yet again)` |
| **Comparison** | 0.83 | `(like (the|that|when)|same (as|issue)|similar to|reminds me of|unlike|opposite of|better than (last|before))` |
| **Temporal Narrative** | 0.82 | 3+ sequence markers in one turn: `(first|initially) .* (then|after|next) .* (now|finally)` |
| **Milestone** | 0.81 | `(finally (shipped|launched|finished|completed|passed)|all .* (passing|working|done)|went live|in production)` |
| **Continuation** | 0.80 | `(so i (tried|did|went|ended up)|took your (advice|suggestion)|following up|went ahead and|gave .* a try)` + anaphoric reference boost |

**Tier 3 - Build Third:**

| Pattern | Confidence | Detection Core |
|---------|-----------|----------------|
| **Causal Context** | 0.79 | `(the reason (is|was|we)|because of|that's (why|because)|since .* (we|i) (decided|chose))` |
| **Introduction** | 0.78 | Possessive + role + proper name: "my friend Sarah", "our CEO Mark", "this guy Jake" |
| **Delegation** | 0.78 | Delegation verb + person: `(told|asked|assigned) NAME to (do|handle)` or receipt: `NAME (sent|gave|shared) me` |
| **Planning** | 0.77 | `(thinking about|planning to|considering|might|hoping to|goal is to)` + entity reference |
| **Emotional Anchor** | 0.76 | Emotion word + entity: `(excited|frustrated|worried|proud|relieved) (about|with|by)` + first-person framing. "Again"/"still"/"finally" modifier boosts to 0.85 |
| **Implicit Preference** | 0.74 | `(i always|i never|i (keep|tend to|usually) (use|choose|prefer)|whenever i .* i)` |

#### Temporal Signal Subcategories

Temporal references compound with patterns. Extracted once, scored independently:

| Category | Examples | Base Score |
|----------|----------|-----------|
| **Time anchors** | yesterday, last week, a few months ago, when we started | 0.20 |
| **Sequence markers** | first...then...after that...finally | 0.15 (per marker, need 2+) |
| **Duration markers** | for months, since January, it's been ages | 0.20 |
| **Frequency markers** | again, always, every time, keeps happening | 0.25 |
| **Temporal contrast** | used to...now, switched from...to, before/after | 0.30 |

Time anchors alone are weak. Combined with an entity reference, they jump
to 0.55+. Combined with a relational noun, 0.70+.

#### Relational Signal Subcategories

People-detection extracted once, used by multiple patterns:

| Signal | Detection | Base Score |
|--------|-----------|-----------|
| **Possessive + relational noun** | (shared with P1) | 0.25-0.40 |
| **Possessive + relational + name** | "my coworker Jake" | 0.50+ |
| **Bare name match** | Name in entity store (via graph probe) | 0.40-0.75 |
| **Role reference** | "the new hire", "the former CTO" | 0.30 |

#### Pattern Composition

Individual signals compose within categories, then categories compose via
noisy-OR with a cross-category bonus:

**Within-category:** max(pattern scores) per category. A single turn rarely
matches multiple patterns from the same category.

**Across categories** (6 categories: pragmatic, temporal, relational,
emotional, structural-pattern, entity):

```
composite = 1 - product(1 - category_score_i)
```

**Cross-category bonus:** When 3+ categories fire with score > 0.15, apply
1.15x multiplier (capped at 0.95).

**Worked example: "My son finally scored yesterday"**

1. Pragmatic: possessive+relational "my son" = 0.40
2. Temporal: "yesterday" = 0.20
3. Emotional: "finally" = 0.25
4. Structural: life_update (state-change "scored" + relational) = 0.35

Noisy-OR: `1 - (0.60)(0.80)(0.75)(0.65) = 1 - 0.234 = 0.77`
Cross-category bonus (4 categories): `min(0.95, 0.77 * 1.15) = 0.88`

This triggers recall with high confidence. The keyword cascade returns `none`.

#### Anti-Composition (Signal Dampening)

Certain combinations should reduce confidence:

| Condition | Dampening | Rationale |
|-----------|-----------|-----------|
| Temporal signals fire but no relational/emotional/entity signals | 0.6x on temporal | Likely technical process: "after the build completes" |
| "my" + technical noun (PR, branch, repo, etc.) | Zero relational signal | Not about people |
| Emotion word without entity reference | 0.5x on emotional | Generic mood: "I'm frustrated" vs "frustrated with the auth migration" |
| Imperative mood + temporal | 0.5x on temporal | Instructions: "first run X, then deploy Y" |

### 4.3 Shift Signals

Detects when the conversation crosses domain boundaries. The moment where
vocabulary, register, and pronoun profile change is a high-value recall
moment because the new domain's entities aren't primed in the session.

#### Five Channels

Operate against a **rolling window** of the last 5-8 turns. Each turn's
feature snapshot is cached (content-word set, pronoun counts, register marker
counts). The shift score measures change from the window, not absolute
properties.

**Channel 1: Lexical Field Departure (weight 0.30)**

Content-word pseudo-stemming (suffix strip, stopword removal) + Jaccard
distance between current turn's stems and rolling window's stems.

Optional: ~10 domain-lexicon buckets (50-100 words each: software, family,
work, health, food, travel, finance, sports, education, entertainment).
Domain bucket crossover adds 0.15 bonus.

Score: `1.0 - jaccard_similarity(current, window)`

**Channel 2: Register Shift (weight 0.25)**

Interpersonal register markers: hedging ("sorry", "just", "actually"),
temporal narrative ("had to", "ended up"), relational nouns, exclamatory
markers ("oh!", "honestly"), first-person narrative density.

Technical register markers: imperatives, code tokens (backticks, camelCase,
dotted paths, ALL_CAPS), precision language ("specifically", version numbers).

Score: angular distance between register vectors of window vs current turn.

**Channel 3: Discourse Shift Markers (weight 0.20)**

High-confidence shift markers (0.7-1.0): "anyway", "by the way", "oh btw",
"speaking of", "that reminds me", "on a different note"

Medium-confidence (0.3-0.5): "actually" (turn-initial), "also" (turn-initial),
"hey", "random but"

Continuation markers (negative, -0.2): "yeah so", "right", "exactly",
"because", "and then"

Score: max of detected markers, minus continuation penalty, clamped [0, 1].
Turn-initial position gets 1.2x.

**Channel 4: Pronoun Shift (weight 0.15)**

Track technical/collective pronouns (it, we, they, that) vs
personal/narrative pronouns (I, my, he, she, his, her).

Score: absolute change in personal-pronoun ratio vs window. Delta of 0.4+
maps to 1.0.

Bonus: proper noun introduction not in rolling window adds 0.10.

**Channel 5: Structural Shift (weight 0.10)**

Turn length ratio change, question density change, punctuation profile shift.
Weakest individual signal, useful as confirmation.

#### Shift Composite

```
shift = 0.30 * lexical + 0.25 * register + 0.20 * discourse +
        0.15 * pronoun + 0.10 * structural
```

**Directional asymmetry:** Technical-to-personal shifts get 1.2x multiplier
(personal-domain recall failures are socially costlier).

**Suppression rules:**
- First turn after >30min gap: suppress (no window to compare against)
- Code block in prior turn followed by prose explanation: 0.6x dampening
- Greeting patterns ("hey", "hi", "how are you"): exempt from shift scoring

**Score range interpretation:**

| Range | Meaning |
|-------|---------|
| 0.00-0.15 | Normal topic progression |
| 0.15-0.30 | Mild drift, same domain |
| 0.30-0.50 | Moderate shift, adjacent domains |
| 0.50-0.70 | Strong shift, different domains |
| 0.70-1.00 | Radical shift + explicit markers |

### 4.4 Impoverishment Signals

Inverts the question: instead of "does this turn need memory?", asks "would
my response be noticeably worse without memory?"

#### Signal I1: Conversational Move Type

Classify the turn's pragmatic function. Different moves have different
memory value:

| Move Type | Memory Value | Detection |
|-----------|-------------|-----------|
| Life update | 0.95 | Declarative + personal entity + temporal marker |
| Resumption | 0.90 | "so about", "going back to", "update on" |
| Sharing | 0.80 | Declarative + first person + personal content |
| Check-in/greeting | 0.75 | Opening turn + greeting formula |
| Musing | 0.70 | "been thinking about", hedging + topic |
| Opinion | 0.50 | Evaluative language, "I think", "I prefer" |
| Asking-personal | 0.45 | Question + self-reference |
| Asking-general | 0.15 | Question + no personal reference |
| Commanding | 0.10 | Imperative, "please do", "can you" |

Score: max of matched move types.

#### Signal I2: Affect with Personal Stakes

Emotional content anchored to the user's world (not task frustration).

Score: `affect_intensity * personal_anchor_strength`

- `affect_intensity`: marker density (exclamation, intensifiers, sentiment
  verbs). 0.0 (neutral) to 1.0 (strong).
- `personal_anchor_strength`: 0.0 (task-directed: "this function is broken")
  to 1.0 (personal: "Emma got into Stanford"). Keyed on whether the subject
  is a person/personal entity vs a technical artifact.

#### Signal I3: Template Test

The most novel signal. If the best memory-free response is a generic
acknowledgment ("That's great!", "I'm sorry to hear that", "Tell me more"),
that's the strongest evidence that memory would transform the response.

| Turn Classification | Template Score |
|--------------------|---------------|
| Personal sharing/update | 0.85 |
| Greeting/check-in | 0.80 |
| Opinion/musing | 0.40 |
| Factual question | 0.10 |
| Task command | 0.05 |

Score: classification-based lookup.

#### Impoverishment Composite

```
I = 0.30 * move_type + 0.30 * template_test + 0.25 * affect_personal +
    0.15 * entity_novelty_to_session
```

Where `entity_novelty_to_session` = fraction of detected entity-like terms
not in `session_entity_names`. Captures "the turn introduces things we
haven't discussed yet this session."

**Convergence bonus:** When 3+ sub-signals fire above 0.5, apply 1.3x
(capped at 1.0).

### 4.5 Layer 1 Ensemble

The four signal families produce four scores: pragmatic P, temporal-
relational T (pattern composite), shift S, impoverishment I. Plus the
existing keyword cascade K.

```
linguistic_score = 1 - (1 - P)(1 - T)(1 - S)(1 - I)(1 - K)
```

Noisy-OR across families. Each family independently contributes evidence.

**Session position modifier:**

| Position | Modifier | Rationale |
|----------|----------|-----------|
| Turn 1-2 | 1.3x | Cross-session reconnection |
| Turn 3-5 | 1.1x | Early-session mild bonus |
| Turn 6-20 | 1.0x | Normal |
| Turn 21+ | 1.05x | Long session, more context at risk |

Applied after noisy-OR, capped at 0.95.

---

## 5. Layer 2: Graph Resonance Probe

The only signal grounded in actual stored knowledge. Runs when
`linguistic_score > 0.15` (low bar - it's cheaper to probe than to miss).

### 5.1 Mention Detection (In-Memory)

Three-layer detector, no graph queries:

**Layer A: Token-to-entity index.**

In-memory dict mapping lowercased name tokens to entity IDs. Built from
`SELECT id, name, entity_type FROM entities WHERE group_id = ?` at startup
(or first probe). ~500 entities * ~2 tokens = ~1000 entries, ~40KB.

For multi-token entities ("Golden Gate Bridge"), index both full name and
individual tokens. Require >=2/3 token overlap for multi-token match.

Invalidation: version counter incremented by store/merge operations. Rebuild
on next probe if stale (<1ms rebuild for 500 entities).

**Layer B: Relational noun resolution.**

Map relational nouns to relationship predicates, then query graph:

```
"son/daughter/child" -> CHILD_OF (inverse)
"wife/husband/partner" -> MARRIED_TO
"boss/manager"        -> REPORTS_TO
"mom/dad"             -> CHILD_OF (forward)
"friend"              -> FRIEND_OF
"coworker/colleague"  -> WORKS_WITH
... (~25 mappings)
```

When "my son" is detected, find entities where
`(identity_entity) -[CHILD_OF]-> (?)`. At most 1 graph query per relational
noun detected (typically 0-1 per turn).

**Layer C: FTS5 fallback.**

If the token index misses (staleness, fuzzy spelling), fall back to
`find_entity_candidates(token, group_id)` for tokens 4+ chars, not
stopwords. Bounded to 3 fallback queries per turn.

### 5.2 Graph Property Probe (Batch Query)

Once candidate entity IDs are identified, fetch properties in a single
batch:

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

Plus highest-urgency relationship type per entity (mapped in application
code from predicate to urgency tier).

Two queries total, regardless of candidate count. <2ms on SQLite with
indexes.

### 5.3 Per-Entity Resonance Scoring

For each matched entity:

```
density = 0.40 * min(degree/10, 1)
        + 0.30 * min(distinct_predicates/5, 1)
        + 0.30 * min(episode_count/8, 1)

activation_est =
    0.9  if last_access < 30min ago
    0.7  if last_access < 2hr ago
    0.4  if last_access < 24hr ago
    0.2  if last_access < 7 days ago
    0.1  otherwise

urgency = max tier across relationship types:
    Identity (IS, IDENTITY_CORE, PREFERS):       1.0
    Personal (CHILD_OF, MARRIED_TO, FRIEND_OF):  0.9
    Experiential (PLAYS, ATTENDS, WORKS_AT):     0.7
    Technical (USES, DEPENDS_ON, IMPLEMENTS):    0.5
    Weak (MENTIONED_IN, CO_OCCURS_WITH):          0.2

tier_bonus =
    semantic:      0.30
    transitional:  0.15
    episodic:      0.00

entity_resonance = 0.35 * density
                 + 0.25 * activation_est
                 + 0.25 * urgency
                 + 0.15 * tier_bonus
```

### 5.4 Match Quality Modifier

| Match Type | Quality |
|-----------|---------|
| Full name exact | 1.00 |
| Full name case-insensitive | 0.95 |
| Multi-token entity, >=2/3 tokens | 0.80 |
| Relational noun resolution | 0.90 |
| Single token of multi-token name | 0.40 |
| FTS5 fallback | 0.60 |

```
adjusted_resonance = entity_resonance * match_quality
```

### 5.5 Turn-Level Aggregation

```
resonance_score = 1 - product(1 - adjusted_resonance(e_i))
```

Noisy-OR across all matched entities. Properties:
- 0 entities: 0.0
- 1 entity at 0.5: 0.5
- 2 entities at 0.5 and 0.6: 0.80
- 3 entities at 0.5 each: 0.875

**Bonus signals** (additive, capped at 1.0):

| Bonus | Value | Condition |
|-------|-------|-----------|
| Cross-entity connectivity | +0.15 | Two matched entities share a direct edge |
| Novel extension | +0.10 | Unrecognized term connects to recognized entity via relational pattern |
| Identity proximity | +0.10 | Any matched entity within 2 hops of identity core |

### 5.6 Worked Examples

**"My son had a great game yesterday"**

- Layer B: "son" -> resolves to entity "Benjamin" (CHILD_OF user)
- Benjamin: degree=5, 4 distinct predicates, 6 episodes
- density = 0.665, activation = 0.4 (18hr ago), urgency = 0.9 (Personal), tier = 0.15 (transitional)
- entity_resonance = 0.581, match_quality = 0.9 -> adjusted = 0.523
- Token "game" -> weak match to "Soccer" entity (0.145 adjusted)
- Cross-entity: Benjamin -[PLAYS]-> Soccer = +0.15
- Identity proximity: Benjamin 1 hop from user = +0.10
- **resonance_score = 0.84**

**"Can you write a for loop in Python?"**

- Token "Python" -> exact match (quality 1.0)
- Python: high density, recent activation, but Technical urgency (0.5)
- entity_resonance = 0.695
- Single entity, no bonuses
- **resonance_score = 0.695**
- But linguistic_score is ~0.05 (command, no personal content)
- Combined: max(0.05, 0.695 * 0.85, 0.0) = 0.59
- Above threshold but the recall plan would be shaped for technical context,
  not personal. Delivery policy limits to entity lookup, not full packet
  assembly. This is acceptable - surfacing the user's Python-related
  project context might actually help.

**"What's the weather like today?"**

- No token matches in entity index.
- **resonance_score = 0.0**
- linguistic_score ~0.0 (asking-general)
- **Combined = 0.0. No recall.**

### 5.7 Cold Start

When total entity count < 20, lower the recall threshold proportionally:

```
effective_threshold = base_threshold * min(entity_count / 20, 1.0)
```

At 5 entities, threshold drops to 25% of normal. Early in the relationship,
the system should be eager to recall what little it knows - this builds
trust.

### 5.8 Performance Budget

| Step | Time |
|------|------|
| Tokenize + token index lookup | <0.2ms |
| Relational noun check + graph query (0-1) | <2ms |
| FTS5 fallback (0-3 queries) | <3ms |
| Batch property query | <2ms |
| Score computation | <0.1ms |
| **Total** | **<8ms** |

In-memory footprint: ~50KB for 500 entities. Scales linearly.

---

## 6. The Cost Asymmetry

This is the most important design constraint and it appears independently
in three of the five research threads.

### False Negative vs False Positive

**False negative** (missed recall): The user says something that assumes
shared knowledge. The agent responds generically. The user perceives the
agent as not knowing them. Trust erodes. Repeated misses teach the user to
stop sharing personal context.

**Cost: HIGH (0.7/1.0)**

**False positive** (unnecessary recall): The system probes the graph and
retrieves context that isn't relevant. The retrieval pipeline returns results;
the response generation model ignores them because they don't fit.

**Cost: LOW (0.1/1.0)**

The false positive cost is further reducible because recall and usage are
decoupled. Surfacing irrelevant memories costs latency but not response
quality - the generation model filters them.

### Threshold Implications

The asymmetry ratio (~7:1) mathematically suggests a threshold of:

```
optimal = C_FP / (C_FP + C_FN) = 0.1 / 0.8 = 0.125
```

In practice, latency budget pushes this up. **Recommended threshold: 0.30.**
This catches the top ~45% of turns, which aligns with the estimate that ~40%
of conversational turns genuinely benefit from memory.

### Threshold Modulation

| Condition | Adjustment | Rationale |
|-----------|-----------|-----------|
| Session start (turns 1-2) | -0.10 | Reconnection is prime recall territory |
| >24hr since last session | -0.05 | More likely stale context needs refreshing |
| Previous turn was sharing/update | -0.05 | User is in disclosure mode |
| Last recall returned nothing | +0.10 | Graph may be sparse; reduce wasted probes |
| Cold start (<20 entities) | proportional reduction | Be eager with what little you know |

---

## 7. Integration with Existing Code

### 7.1 Changes to `analyze_memory_need()`

The existing function signature is preserved:

```python
def analyze_memory_need(
    current_turn: str,
    *,
    recent_turns: list[str] | None = None,
    session_entity_names: list[str] | None = None,
    mode: str = "auto_recall",
) -> MemoryNeed:
```

New optional parameters:

```python
    rolling_window: RollingWindow | None = None,  # for shift detection
    graph_probe: GraphProbe | None = None,         # for Layer 2
```

When `graph_probe` is None, Layer 2 is skipped (backward compatible with
tests). When `rolling_window` is None, shift signals are skipped.

### 7.2 New Need Types

Add to the `need_type` vocabulary (existing types preserved):

```
presupposed_reference    # bare name, cross-session anaphora
presupposed_relationship # possessive + relational noun
presupposed_entity       # definite unintroduced "the X"
presupposed_context      # hedged aside with personal content
life_update              # state change with personal entity
correction               # contradicts or updates stored fact
continuation             # follows up on prior discussion
milestone                # achievement closes open loop
memory_gap               # user explicitly can't remember
identity_claim           # self-description
social_graph_update      # relationship network change
recurring_problem        # same issue happening again
graph_resonance          # graph probe found rich stored context
```

### 7.3 Extended MemoryNeed Model

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
    signal_scores: dict[str, float] | None = None  # per-family breakdown
    detected_entities: list[str] | None = None      # from graph probe
    detected_referents: list[str] | None = None     # bare names, relational nouns
    resonance_score: float = 0.0
```

### 7.4 New Modules

```
server/engram/retrieval/need.py           # Modified (main analyzer)
server/engram/retrieval/signals.py        # NEW: signal extraction functions
server/engram/retrieval/graph_probe.py    # NEW: Layer 2 graph resonance
server/engram/retrieval/rolling_window.py # NEW: shift detection state
```

### 7.5 Processing Flow in `analyze_memory_need()`

```python
def analyze_memory_need(current_turn, *, recent_turns=None,
                        session_entity_names=None, mode="auto_recall",
                        rolling_window=None, graph_probe=None):
    text = normalize(current_turn)

    # Layer 0: short-circuit
    if not text:
        return MemoryNeed(need_type="none", should_recall=False, ...)
    if is_acknowledgement(text):
        return MemoryNeed(need_type="none", should_recall=False, ...)

    # Layer 1: linguistic signals
    pragmatic    = extract_pragmatic_signals(text, session_entity_names, recent_turns)
    temporal_rel = extract_temporal_relational_signals(text, session_entity_names, recent_turns)
    shift        = extract_shift_signals(text, rolling_window) if rolling_window else 0.0
    impoverish   = extract_impoverishment_signals(text, session_entity_names)
    keyword      = extract_keyword_signals(text, recent_turns, session_entity_names, mode)

    linguistic = noisy_or([pragmatic.composite, temporal_rel.composite,
                           shift, impoverish.composite, keyword])
    linguistic *= session_position_modifier(recent_turns)
    linguistic = min(0.95, linguistic)

    # Layer 2: graph probe (conditional)
    resonance = 0.0
    detected_entities = []
    if graph_probe is not None and linguistic > 0.15:
        probe_result = graph_probe.probe(text, session_entity_names)
        resonance = probe_result.score
        detected_entities = probe_result.entity_ids

    # Ensemble
    combined = max(linguistic, resonance * 0.85, keyword * 0.70)

    # Threshold
    threshold = compute_threshold(mode, recent_turns, graph_probe)
    if combined < threshold:
        return MemoryNeed(
            need_type="none", should_recall=False,
            confidence=1.0 - combined,  # lower confidence when signals present
            signal_scores={...}, resonance_score=resonance,
        )

    # Classify need type from dominant signal
    need_type = classify_need_type(pragmatic, temporal_rel, shift,
                                   impoverish, keyword, resonance)
    return MemoryNeed(
        need_type=need_type, should_recall=True,
        confidence=min(0.95, combined),
        query_hint=best_query_hint(pragmatic, temporal_rel, detected_entities),
        urgency=max_urgency(pragmatic, temporal_rel, impoverish),
        packet_budget=compute_packet_budget(combined, mode),
        entity_budget=compute_entity_budget(combined, mode),
        signal_scores={...},
        detected_entities=detected_entities,
        resonance_score=resonance,
    )
```

### 7.6 Interaction with Existing Systems

**Recall planner** (`plan.py`): When `detected_entities` is non-empty, pass
them as seed entities into the recall plan. This focuses retrieval on the
relevant subgraph instead of doing a blind text search.

**Packet assembler** (`packets.py`): Use `need_type` to select packet type
(already implemented - `_entity_packet_type` routes based on need_type).

**Feedback** (`feedback.py`): Tag recall events with trigger source
(linguistic vs graph_resonance vs keyword_legacy) for hit-rate tracking.

**MCP server** (`mcp/server.py`): Wire `graph_probe` and `rolling_window`
into `observe()` and `remember()` auto-recall paths.

**Triage policy** (`triage_policy.py`): Share signal extraction with the
need analyzer. A turn that matches multiple need patterns is also likely to
produce extractable entities if stored.

---

## 8. Failure Modes

### 8.1 Common Word Entity Collision

**Problem:** Entity "Spring" (Java framework) matches "spring is my favorite
season."

**Mitigations:**
- Type-context check: if surrounding tokens have no technology associations,
  downweight Technology-type matches
- Single-token common-word entities with degree < 2: match quality = 0.10
- Entity name stop-list (~50 common English words that happen to be entity
  names)

### 8.2 Stale Entity Over-Recall

**Problem:** Entity discussed months ago has high degree but low current
relevance.

**Mitigations:**
- `activation_est` inherently dampens old entities (0.1-0.2)
- Retrieval pipeline applies temporal decay in ranking
- Over-recall is cheaper than under-recall (cost asymmetry)

### 8.3 The "Tries Too Hard" Agent

**Problem:** Agent surfaces memory on every turn, making it feel performative.

**Mitigations:**
- Cooldown: after recall is used in response, add +0.15 bias against next recall
- Diversity cap: don't recall the same entity cluster more than twice per session
- These are delivery-policy constraints, not need-analyzer constraints

### 8.4 Technical Temporal Language

**Problem:** "After the build completes, deploy to staging" has temporal
structure but no personal memory value.

**Mitigations:**
- Anti-composition: temporal + no relational/emotional/entity = 0.6x dampening
- Imperative mood detection: commands with temporal markers get 0.5x
- Subject check: technical noun as subject (not "I", "we", person name) = dampen

### 8.5 Non-Relational "My"

**Problem:** "My PR needs review" triggers possessive detection.

**Mitigation:** Technical possessive exclusion set (~35 terms). When "my"
precedes a word in this set, zero the relational signal.

### 8.6 Ambiguous Names

**Problem:** "Will you help?" vs "my friend Will"

**Mitigation:** Ambiguous-name set (~50 entries). Require secondary signal
(relational framing, mid-sentence capitalization, person-verb) before
treating as a name reference.

### 8.7 Over-Triggering on Casual Chat

**Problem:** "I had pizza for lunch" is personal but has no recall value.

**Mitigation:** The composition function handles this naturally. A single
weak signal without entity match, temporal anchor, or emotional marker stays
below threshold. The graph probe returns nothing (no "pizza" entity with rich
structure), so resonance = 0.

---

## 9. Build Phases

### Phase 0: Signal Infrastructure

**Goal:** Extraction framework without changing recall behavior.

**Work:**
1. Create `signals.py` with signal extraction functions (all return typed
   result objects with per-signal scores)
2. Create `rolling_window.py` with turn feature caching
3. Create `graph_probe.py` with token-to-entity index + batch query
4. Add `signal_scores`, `detected_entities`, `detected_referents`,
   `resonance_score` fields to `MemoryNeed`
5. Wire extraction into `analyze_memory_need()` but DO NOT change the
   return value yet - only populate `signal_scores` for telemetry

**Acceptance:**
- All new signal extraction runs on every turn
- `signal_scores` is populated in `MemoryNeed` for observability
- Zero behavioral change - existing keyword cascade still drives decisions
- Telemetry events emitted via `publish_memory_need_analysis()`

### Phase 1: Pragmatic + Relational Signals

**Goal:** Catch the "my son" class of misses.

**Work:**
1. Implement P1 (possessive+relational) and P2 (bare proper name) in
   `signals.py`
2. Add relational noun lexicon (4 tiers), communal-ground set, technical
   possessive exclusion set, ambiguous-name set
3. Implement P6 (continuation markers) with session-context check
4. Wire pragmatic composite into `analyze_memory_need()` decision path
5. New need types: `presupposed_relationship`, `presupposed_reference`

**Acceptance:**
- "my son had a great game" returns `should_recall=True`
- "Emma scored two goals" returns `should_recall=True` (bare name)
- "my PR needs review" returns `should_recall=False` (technical exclusion)
- "still dealing with that bug" returns `should_recall=True` (continuation)
- Existing keyword-triggered turns still work

### Phase 2: Graph Resonance Probe

**Goal:** Ground linguistic signals in actual stored knowledge.

**Work:**
1. Implement token-to-entity index with version-based invalidation
2. Implement relational noun resolution (predicate mapping + graph query)
3. Implement batch property probe (degree, predicates, activation, tier)
4. Implement resonance scoring (density, activation, urgency, tier bonus)
5. Wire into `analyze_memory_need()` with the override path
6. New need type: `graph_resonance`

**Acceptance:**
- "my son" resolves to stored child entity without naming the child
- Resonance score reflects graph neighborhood richness
- Graph probe only runs when linguistic_score > 0.15
- Probe completes in <8ms for typical graphs (~500 entities)
- Override: high resonance triggers recall even when linguistic score is low

### Phase 3: Structural Patterns (Tier 1)

**Goal:** Catch corrections, callbacks, life updates, identity claims.

**Work:**
1. Implement Tier 1 patterns (callback, memory_gap, correction, life_update,
   identity_claim, status_check) in `signals.py`
2. Wire pattern composite into the Layer 1 ensemble
3. New need types for each pattern

**Acceptance:**
- "actually, we switched to MySQL" returns `correction`
- "I can't remember if we decided on Postgres" returns `memory_gap`
- "we moved to Austin" returns `life_update`
- "I'm more of a backend person" returns `identity_claim`

### Phase 4: Shift + Impoverishment

**Goal:** Detect domain transitions and response-quality prediction.

**Work:**
1. Implement rolling window with 5-channel shift detector
2. Implement impoverishment signals (move type, affect, template test)
3. Wire both into Layer 1 ensemble
4. Add session position modifier

**Acceptance:**
- Technical-to-personal conversation shift boosts recall probability
- "Emma scored two goals!" (high affect + personal + life update) scores >0.90
- "Can you write a for loop?" (command, generic) scores <0.10
- Shift signal suppressed on first turn and after code blocks

### Phase 5: Structural Patterns (Tiers 2+3) + Tuning

**Goal:** Complete pattern coverage and calibrate thresholds.

**Work:**
1. Implement remaining 12 patterns
2. Implement anti-composition dampening rules
3. Threshold calibration against conversation logs
4. Add hit-rate tracking (fraction of recalls that surface useful context)
5. Add self-adjusting threshold based on rolling hit rate

**Acceptance:**
- Full 18-pattern catalog active
- Anti-composition prevents over-triggering on technical temporal language
- Hit rate tracked and threshold adapts
- Latency budget met: <2ms (no probe), <10ms (with probe)

### Phase 6: Post-Hoc Impoverishment Detector (Optional)

**Goal:** Safety net for turns that slip through all signals.

**Work:**
1. After generating response but before sending, check for template-match
   impoverishment
2. If response is generic acknowledgment AND turn had linguistic_score > 0.20,
   trigger recall-and-regenerate
3. Log all post-hoc triggers as RIP blind spots for threshold tuning

**Acceptance:**
- Post-hoc fires on <8% of turns
- Each post-hoc trigger improves the response (validated by usage feedback)
- Post-hoc trigger data feeds back into Phase 5 calibration

---

## 10. Metrics

### Recall Quality

| Metric | Definition | Target |
|--------|-----------|--------|
| **Recall hit rate** | Fraction of recalls that surface >=1 relevant memory | >60% |
| **Usage rate** | Fraction of surfaced memories actually used in response | >40% |
| **False recall rate** | Fraction of auto-surfaced packets that were irrelevant | <30% |
| **Surfaced-to-used ratio** | How much auto-recall is noise vs value | <2.5:1 |

### Need Detection Quality

| Metric | Definition | Target |
|--------|-----------|--------|
| **Need precision** | Of turns that triggered recall, how many benefited | >55% |
| **Need recall** | Of turns that would have benefited, how many triggered | >75% |
| **Post-hoc fire rate** | How often the safety net catches misses | <8% |

### Performance

| Metric | Definition | Target |
|--------|-----------|--------|
| **Layer 1 latency** | p99 signal extraction time | <3ms |
| **Layer 2 latency** | p99 graph probe time | <10ms |
| **Probe trigger rate** | Fraction of turns that trigger Layer 2 | 30-50% |

### Behavioral

| Metric | Definition | Target |
|--------|-----------|--------|
| **Missed connection rate** | Turns with personal content that got no recall | <15% |
| **Over-recall rate** | Back-to-back recalls with low usage | <10% |
| **Signal family coverage** | Which family triggered each recall | balanced |

---

## 11. Open Questions

1. **Graph probe in the hot path.** The probe adds 5-8ms to ~40% of turns.
   Is this acceptable in the MCP auto-recall path? Could the probe be async
   (fire-and-await with timeout)?

2. **Shared signal extraction.** The triage policy (`triage_policy.py`)
   already detects some of the same signals for storage-side scoring. Factor
   into a common `signal_extractor.py`?

3. **LLM-assisted need detection.** The design is entirely heuristic. Should
   there be an optional LLM escalation path for borderline turns (0.25-0.35
   combined score), similar to how merge/infer use tiered LLM judges?

4. **Graph probe caching.** If the same entity is mentioned multiple turns in
   a row, should the probe cache results? Session-level entity resonance
   cache with TTL?

5. **Calibration data.** The thresholds and weights are theoretically
   motivated but not empirically tuned. What's the plan for collecting
   labeled conversation data (turn + "did memory help?" annotations)?

6. **Template test implementation.** The post-hoc impoverishment detector
   requires inspecting the generated response before sending. This needs
   a hook in the MCP server's response path. Is that architecturally clean?

---

## 12. Summary

The current `analyze_memory_need()` answers "is the user asking for memory?"
with regex keyword matching. This catches ~15% of turns where memory matters.

The proposed system answers "would a person who knows this user respond
differently?" using five signal families:

1. **Pragmatic:** "my son" presupposes shared knowledge of the son
2. **Temporal-Relational:** "finally scored yesterday" has a structural shape
   that predicts recall value
3. **Shift:** Technical-to-personal transition means the entity set needs
   refreshing
4. **Impoverishment:** The best memory-free response to "Emma scored two
   goals!" is "That's great!" - memory transforms it
5. **Graph Resonance:** The graph itself says "I know about this" before any
   retrieval runs

Combined via noisy-OR with a 0.30 threshold biased by the 7:1 cost
asymmetry (missed connections are far worse than unnecessary recalls), the
system catches an estimated ~85% of natural recall-worthy turns.

Total latency: <2ms without graph probe, <10ms with. No model calls. All
deterministic heuristics with pre-compiled regexes and small lookup tables.

The goal is not perfect recall detection. The goal is that using Engram
feels like talking to someone who actually knows you.
