# Pragmatic Recall Signals: Detecting Unstated Memory Need

> Status: implemented — pragmatic detectors in need.py/signals.py; flag-gated off by default


Design document for conversational pragmatics layer in `analyze_memory_need()`.

**Problem**: The current memory-need analyzer (`server/engram/retrieval/need.py`) uses keyword patterns that detect *explicit* recall requests ("catch me up", "what changed", "do you remember"). Real conversation doesn't work this way. When someone says "picking up Emma from practice," they aren't requesting memory — they're *assuming* the listener already knows who Emma is and what sport she plays. The current system returns `need_type="none"` for this input. That is a fundamental failure.

**Thesis**: The most important recall signals are pragmatic — they live in what the speaker *doesn't say*. Shared knowledge is not requested; it is presupposed. A pragmatic layer can detect these presuppositions deterministically, without model calls, in under 5ms.

---

## 1. Theoretical Foundation

### 1.1 Grice's Cooperative Principle

Grice's maxims predict that speakers say only what is necessary. When someone says "Emma's got a tournament this weekend," they are obeying the Maxim of Quantity — they provide the new information (tournament, this weekend) and omit what they assume is shared (who Emma is, what sport). The *omission itself* is the signal. If the speaker thought the listener didn't know Emma, they would say "my daughter Emma, who plays soccer, has a tournament."

The detector's job: identify where the speaker has omitted background that would be necessary for a stranger, but not for someone with shared history.

### 1.2 Clark's Common Ground

Herbert Clark's theory of common ground distinguishes:
- **Communal common ground**: shared cultural knowledge ("the President")
- **Personal common ground**: knowledge specific to the conversational dyad ("the migration," "Emma")

The memory system cares only about personal common ground. Definite references to entities not established in communal ground or in the current session are *presuppositions of personal common ground* — direct evidence that the speaker expects cross-session memory.

### 1.3 Relevance Theory (Sperber & Wilson)

Relevance Theory states that every utterance carries an implicit guarantee: "this is worth your processing effort because it connects to something you already know." When a speaker drops an aside — "sorry, kid stuff" — the low processing effort they invest signals high assumed relevance of the background. The listener is expected to bridge the gap using existing knowledge. The *less explanation given, the more memory is assumed*.

This creates the Politeness Inversion (Section 7): casual, throwaway mentions presuppose MORE shared knowledge than formal introductions.

### 1.4 Presupposition Triggers (Karttunen, Stalnaker)

Linguistic presupposition triggers are lexical or syntactic constructions that require certain propositions to be already in the common ground. Key triggers relevant to memory:

| Trigger | Example | Presupposition |
|---------|---------|---------------|
| Definite description | "the migration" | There is a specific migration known to both |
| Possessive + relational | "my son" | Speaker has a son; listener may know this |
| Factive verb | "I forgot that..." | The embedded proposition is true and known |
| Change-of-state verb | "stopped using React" | Was previously using React |
| Iterative | "went back to Python" | Had used Python before |
| Temporal clause | "before we switched" | There was a switch; listener knows about it |
| Cleft | "it was Emma who..." | Emma exists in common ground |

### 1.5 Anaphora Resolution (Centering Theory)

Centering Theory (Grosz, Joshi, Weinstein) defines a "backward-looking center" — the most salient entity carried forward between utterances. When a pronoun or demonstrative appears with no antecedent in the current session, the speaker is reaching across session boundaries. This is *cross-session anaphora* — the strongest possible signal that memory is needed, because the utterance is literally uninterpretable without it.

---

## 2. Signal Catalog

Six pragmatic signals, ordered by priority. Each is designed for regex/heuristic detection in <1ms.

### Signal 1: Cross-Session Anaphora (Unresolved Reference)

**Phenomenon**: Pronouns, demonstratives, or bare ellipsis with no session-local antecedent.

**Examples**:
- "she loved it" (no "she" introduced this session)
- "that thing we discussed" (no prior discussion this session)
- "still broken" (no prior mention of what's broken)
- "it finally shipped" (no antecedent for "it")

**Detection heuristic**:
1. Identify pronouns/demonstratives in current turn: `she`, `he`, `they`, `it`, `that`, `this`, `those`, `these` (excluding expletive "it" — see failure modes).
2. Check `session_entity_names` and `recent_turns` for plausible antecedents.
3. If the pronoun has no antecedent in session context, flag as cross-session anaphora.

**Antecedent check** (cheap):
- For `she`/`he`: scan `recent_turns` for proper names or gendered relational nouns.
- For `it`/`that`/`this`: scan `recent_turns` for noun phrases (capitalized sequences, quoted terms).
- For `they`/`those`/`these`: scan for plural nouns or group references.
- If `session_entity_names` is non-empty and pronoun is present, partial antecedent assumed (reduce score but don't zero it).

**Expletive "it" filter**: Exclude `it` when it appears in:
- Sentence-initial position before `is/was/seems/looks/appears` ("it is raining")
- Fixed phrases: "it's okay", "it doesn't matter", "it's fine", "it depends"
- Weather/time: "it's cold", "it's late"

Pattern: `^it\s+(?:is|was|'s|seems?|looks?|appears?|doesn't|does not)\s+(?:okay|fine|cold|hot|late|early|raining|snowing|clear|obvious)` — if matched, suppress.

**Score**: 0.35 (highest single signal)
**Urgency**: 0.80
**Recall type**: `"presupposed_reference"`

### Signal 2: Bare Proper Name (Unintroduced Referent)

**Phenomenon**: A proper name dropped without introduction, appositive, or role marker. The speaker assumes the listener knows who/what this is.

**Examples**:
- "Emma's got a tournament this weekend" (who is Emma?)
- "talked to Marcus about it" (who is Marcus?)
- "deploying to Kubernetes" (communal ground — NOT a recall signal)

**Detection heuristic**:
1. Extract capitalized name sequences using existing `_has_named_terms()` / `_extract_named_terms()`.
2. Filter out names present in `session_entity_names` (already established this session).
3. Filter out communal-ground terms: a static set of ~200 well-known proper nouns (programming languages, major frameworks, companies, countries, cities). This set already partially exists in `_NON_ENTITY_TITLE_WORDS`.
4. Check if the name is accompanied by an introduction marker: "my friend X", "X, who is", "X from Y", "a guy named X". If introduced, it is NEW information, not a presupposition. Reduce score.
5. Remaining bare names = presupposed personal referents.

**Introduction marker pattern**:
```
(?:my\s+\w+\s+)?(?:named|called)\s+{NAME}
{NAME},?\s+(?:who|which|that)\s+(?:is|was|works|lives)
(?:a|an|this)\s+(?:guy|person|friend|colleague|tool|app|service)\s+(?:named|called)\s+{NAME}
```
If matched, the name is being INTRODUCED, not presupposed. Score = 0.

**Score**: 0.30
**Urgency**: 0.75
**Recall type**: `"presupposed_referent"`
**Query hint**: The bare name itself — ideal for entity lookup.

### Signal 3: Possessive + Relational Noun (Assumed Relationship Knowledge)

**Phenomenon**: "my {relational_noun}" presupposes the listener knows about this relationship. "My son" is not introducing a son — it's assuming you know about the son.

**Relational noun lexicon** (~60 terms):

*Family*: son, daughter, wife, husband, partner, mom, dad, mother, father, brother, sister, uncle, aunt, cousin, grandma, grandpa, grandmother, grandfather, kid, kids, child, children, baby, fiancee, fiance, nephew, niece, stepson, stepdaughter, ex, ex-wife, ex-husband

*Professional*: boss, manager, team, lead, coworker, colleague, mentor, mentee, intern, client, customer, investor, advisor, therapist, doctor, lawyer, accountant

*Social*: friend, roommate, neighbor, landlord, coach, trainer, teacher, professor, classmate

*Possessions with identity*: dog, cat, pet, car (when named: "my car, Betsy")

**Detection heuristic**:
1. Match `\b(?:my|our|his|her|their)\s+(\w+)\b` against the relational noun lexicon.
2. Exclude non-relational possessives via negative set: "my bad", "my pleasure", "my point", "my guess", "my take", "my turn", "my fault", "my way", "my god", "my goodness", "my apologies", "my concern", "my understanding", "my question", "my issue".
3. If a proper name follows the relational noun ("my son Marcus"), extract Marcus as a query hint.

**Score**: 0.25
**Urgency**: 0.65
**Recall type**: `"presupposed_relationship"`
**Query hint**: If name follows ("my son Marcus") -> "Marcus"; else -> the relational noun category ("family", "professional").

### Signal 4: Definite Reference to Unintroduced Entity

**Phenomenon**: "the {noun}" where the noun has not been introduced in this session. "The migration" presupposes shared knowledge of a specific migration. "The meeting" presupposes a known meeting.

**Examples**:
- "the migration is almost done" (which migration?)
- "the bug from yesterday" (which bug?)
- "any news on the proposal?" (which proposal?)

**Detection heuristic**:
1. Extract `the\s+(\w+(?:\s+\w+)?)` patterns from current turn.
2. Filter out discourse-structural "the": "the thing is", "the point is", "the problem is", "the question is", "the fact that", "the way I see it", "the truth is", "the issue is" (when followed by a clause, not a specific entity).
3. Filter out generic/communal "the": "the internet", "the web", "the cloud", "the market", "the government", "the world", "the team" (when no specific team is contextually salient), "the office", "the weekend".
4. Check whether the noun phrase appears in `recent_turns` or `session_entity_names`. If yes, it was introduced this session — not a cross-session presupposition.
5. Remaining definite descriptions = presupposed shared knowledge.

**Discourse "the" filter pattern**:
```
\bthe\s+(?:thing|point|problem|question|fact|way|truth|issue|idea|reason|matter)\s+(?:is|was|being)\b
```

**Score**: 0.20
**Urgency**: 0.60
**Recall type**: `"presupposed_entity"`
**Query hint**: The noun phrase after "the".

### Signal 5: Continuation Presupposition (State Persistence Markers)

**Phenomenon**: Words that presuppose a prior state that should be known to the listener.

**Lexicon with presupposition type**:

| Marker | Presupposition | Example |
|--------|---------------|---------|
| still | State persists from known prior | "still using Vim" |
| again | Event recurrence | "broke again" |
| finally | Long-awaited resolution | "finally got the offer" |
| anymore | State cessation | "don't use it anymore" |
| back to | Return to prior state | "back to Python" |
| keep + gerund | Ongoing repeated action | "keeps crashing" |
| yet | Expected completion not reached | "hasn't shipped yet" |
| already | Completed before expected | "already migrated" |
| no longer | State cessation | "no longer blocked" |
| turns out | Expectation revision | "turns out it was a bug" |

**Detection heuristic**:
1. Match continuation markers in current turn.
2. If `recent_turns` provides context for the continuation (e.g., "still" where the prior state was mentioned 2 turns ago), this is intra-session continuation — score reduced by 50%.
3. If no session context explains the continuation, it is a cross-session presupposition.

**"Still" disambiguation**: "still" as adjective ("still water", "still life", "sit still") vs. adverb of continuation. Heuristic: if "still" is followed by a verb or adjective predicate ("still running", "still broken", "still waiting"), it is continuative. If followed by a noun without a verb ("still water"), it is adjectival. Pattern for continuative: `\bstill\s+(?:\w+ing\b|\w+ed\b|not\b|haven't\b|hasn't\b|can't\b|won't\b|don't\b)` plus `\bstill\s+(?:the same|there|here|available|broken|open|blocked|pending|waiting|working|alive|active|down|up)\b`.

**Score**: 0.18
**Urgency**: 0.55
**Recall type**: `"presupposed_prior_state"`

### Signal 6: Hedged Aside with Personal Content (The Politeness Inversion)

**Phenomenon**: When someone casually drops personal information as a parenthetical or aside, the low conversational weight signals HIGH assumed shared knowledge. "Oh btw, Emma's recital is Friday" packs two presuppositions: you know Emma, and you know she does recitals.

**Hedge/aside markers**: "oh btw", "btw", "by the way", "anyway", "oh and", "also", "on another note", "random but", "sorry", "side note", "quick thing", "oh right", "speaking of", "that reminds me", "incidentally", "fwiw", "fyi"

**Detection heuristic**:
1. Check if the turn starts with or contains a hedge/aside marker.
2. Check if personal content follows (reuse `_PERSONAL_PATTERNS` from `triage_policy.py`, or check for bare proper names / possessive + relational nouns after the hedge).
3. The combination of hedge + personal content = high-confidence pragmatic presupposition.

**Why this matters**: These asides are where people are MOST likely to mention things they assume you know. "Sorry, dealing with kid stuff" assumes you know they have kids. "Btw Marcus says hi" assumes you know Marcus. The casual register is itself evidence of assumed intimacy/shared knowledge.

**Score**: 0.22
**Urgency**: 0.70
**Recall type**: `"presupposed_shared_context"`

---

## 3. Priority Ranking

Ranked by the product of three factors:

| Rank | Signal | Recall Urgency | Detection Reliability | Compute Cost | Combined |
|------|--------|---------------|----------------------|-------------|----------|
| 1 | Cross-session anaphora | 0.95 | 0.70 | ~0.3ms | **0.665** |
| 2 | Bare proper name | 0.90 | 0.80 | ~0.2ms | **0.720** |
| 3 | Hedged aside + personal | 0.85 | 0.75 | ~0.3ms | **0.638** |
| 4 | Possessive + relational | 0.80 | 0.85 | ~0.1ms | **0.680** |
| 5 | Definite unintroduced entity | 0.70 | 0.60 | ~0.3ms | **0.420** |
| 6 | Continuation markers | 0.65 | 0.65 | ~0.2ms | **0.423** |

**Recall urgency** = how embarrassing/costly is it if the agent doesn't recall? Cross-session anaphora is worst because the agent literally cannot parse the utterance ("she loved it" — who?). Bare names are nearly as bad ("tell Emma I said hi" — who is Emma?). Continuation markers are lower because the agent can often respond usefully even without knowing the prior state.

**Detection reliability** = true positive rate. Bare proper names are most reliable because capitalization is a strong signal. Cross-session anaphora is least reliable because expletive "it" and generic "they" create false positives. Definite descriptions have the lowest reliability due to discourse-structural uses.

---

## 4. Signal Combination: Scoring Function

### 4.1 Design Principles

- **Noisy-OR combination**: Multiple weak signals should produce a strong recall decision, but signals should not double-count.
- **Floor guarantee**: Any single signal above its threshold should trigger recall.
- **Ceiling**: Combined score capped at 1.0.

### 4.2 Formula

Individual signal scores `s_i` are already in [0, 1] based on detection confidence (binary for most, but reduced for partial matches like intra-session continuation).

**Pragmatic composite**:
```
P = 1 - product(1 - w_i * s_i) for all signals i
```

This is the noisy-OR formula already used in `plan.py:merge_support()`. It has the right properties:
- Single strong signal (e.g., bare name at 0.30) produces P = 0.30
- Two medium signals (0.20 + 0.18) produce P = 0.344 (not 0.38 — diminishing returns)
- Saturates toward 1.0 but never exceeds it

**Weights `w_i`** (signal-specific, tuned to normalize contribution):

| Signal | Weight |
|--------|--------|
| Cross-session anaphora | 0.35 |
| Bare proper name | 0.30 |
| Possessive + relational | 0.25 |
| Hedged aside + personal | 0.22 |
| Definite unintroduced entity | 0.20 |
| Continuation markers | 0.18 |

### 4.3 Integration with Existing `analyze_memory_need()`

The pragmatic composite P integrates as a new branch in the existing cascade, positioned AFTER acknowledgement filtering but BEFORE the existing explicit-pattern checks:

```
1. Empty turn -> none
2. Acknowledgement -> none
3. [NEW] Pragmatic presupposition (P >= 0.20) -> presupposed_reference
4. Broad context patterns -> broad_context
5. Prospective patterns -> prospective
6. ... (rest of existing cascade)
7. [MODIFIED] Final fallback: if P > 0.0 but < 0.20, add to reasons list
   and lower the confidence of "none" verdict proportionally
```

When pragmatic signals fire (P >= 0.20):
```python
MemoryNeed(
    need_type="presupposed_reference",  # or most specific sub-type
    should_recall=True,
    confidence=min(0.85, 0.5 + P),  # scales with signal strength
    reasons=[list of firing signals],
    query_hint=best_query_hint,  # bare name > relational noun > "the X"
    urgency=max(urgency of firing signals),
    packet_budget=1,
    entity_budget=3,
)
```

When pragmatic signals are sub-threshold (0 < P < 0.20), they still influence the final "none" verdict:
```python
# In the final fallback:
confidence = 0.82 - (P * 0.5)  # e.g., P=0.15 -> confidence drops to 0.745
# This makes downstream "maybe recall anyway" logic more likely to trigger
```

### 4.4 Interaction with Existing Signals

Pragmatic signals are ADDITIVE with existing pattern matches, not replacements:

- "How's the migration going?" fires both `_TEMPORAL_PATTERNS` (existing, "how's X going") AND Signal 4 (definite unintroduced "the migration"). The existing path handles it. Pragmatic signals add reasons but don't override.
- "Still waiting on Marcus" fires both `_FOLLOWUP_MARKERS` (existing, "still") AND Signal 2 (bare name "Marcus") AND Signal 5 (continuation "still"). The pragmatic layer adds `presupposed_referent` with query hint "Marcus", which is strictly better than the generic followup handling.
- "Emma's recital is Friday" fires NOTHING in the existing system. The pragmatic layer catches it via Signal 2 (bare name "Emma") with score 0.30. This is the critical gap the pragmatic layer fills.

---

## 5. Failure Modes and Mitigations

### 5.1 False Positive: Discourse Markers Mimicking Definite Reference

**Problem**: "The thing is, I need more time" — "the thing" is a discourse marker, not a reference to a known entity.

**Mitigation**: The discourse-structural filter (Signal 4, step 2) catches the most common cases. The pattern `the\s+(?:thing|point|problem|question|fact|way|truth|issue|idea|reason|matter)\s+(?:is|was|being)` covers the top ~15 discourse uses of "the". Additional filter: if "the X" is followed by a clause-introducing complementizer ("that", "is that"), it is likely discourse-structural.

**Residual risk**: Low. Discourse "the + NP + is" is a highly regular construction.

### 5.2 False Positive: Non-Relational Possessives

**Problem**: "My bad", "my point is", "my pleasure" — possessive but no relational noun.

**Mitigation**: Negative set of ~20 fixed phrases (Signal 3, step 2). These are closed-class expressions that don't evolve rapidly.

**Residual risk**: Very low. The negative set is small and stable.

### 5.3 False Positive: Communal Ground Proper Names

**Problem**: "React is great" — bare proper name but communal knowledge, not personal.

**Mitigation**: Static communal-ground set (~200 terms: programming languages, major frameworks, OS names, major companies, countries). This is the same concept as `_NON_ENTITY_TITLE_WORDS` but expanded.

**Important nuance**: Some communal-ground terms ARE personal when contextualized. "Python" is communal; "our Python migration" is personal. The possessive "our" + communal term should STILL fire (via Signal 3 or the possessive+communal combination). The communal filter only applies to bare mentions without possessive/demonstrative framing.

**Residual risk**: Medium. The communal set will never be complete. Mitigation: if a bare name matches an entity in `session_entity_names`, skip communal filtering — it was already established as personally relevant.

### 5.4 False Positive: Generic "Still" and "Again"

**Problem**: "I still think React is better" — "still" as opinion persistence, not state-change presupposition. "Try again" — generic retry instruction.

**Mitigation**: The continuative-"still" pattern (Signal 5) requires "still" + verb/adjective predicate. "Still think" will match, but this is arguably correct — the speaker IS presupposing a prior discussion about React. The question is whether this is worth recalling. Solution: weight "still + opinion verb" (think, believe, feel, prefer) at 50% of the normal continuation score.

"Again" filter: exclude imperative "again" ("try again", "say that again", "run it again") via pattern `\b(?:try|say|run|do|check|test|start|click|open|close)\s+(?:it\s+)?again\b`.

### 5.5 False Negative: New Information Disguised as Presupposition

**Problem**: "My friend Sarah just got a new job" — this LOOKS like a presupposition ("my friend Sarah") but might be introducing Sarah for the first time.

**Mitigation**: This is actually correct behavior. Even if Sarah is new, the speaker is presenting her as if she should be known. The right response is to recall, find nothing, and proceed normally. A failed recall is cheap; a missed recall is expensive. **Bias toward recall is the correct default for pragmatic signals.** The system already handles "recall found nothing" gracefully — it just means no context is injected.

### 5.6 False Positive: Expletive "It"

**Problem**: "It's a nice day" — "it" is expletive (weather), not anaphoric. "It seems like a good plan" — "it" is pleonastic.

**Mitigation**: The expletive-it filter (Signal 1) catches weather, time, pleonastic constructions. Additional filter: if "it" is the subject of `seems`, `appears`, `looks like`, `turns out`, and the complement is a full clause (contains a verb), it is pleonastic.

Pattern: `\bit\s+(?:is|'s|was|seems?|appears?|looks?\s+like|turns?\s+out)\s+(?:that|like|as\s+if|a\s+\w+\s+\w+)` -> suppress cross-session anaphora signal.

**Residual risk**: Low-medium. Expletive "it" is well-studied and the patterns are finite.

---

## 6. The Politeness Inversion (Detailed)

This deserves special treatment because it is counterintuitive and has no analog in the existing system.

### The Principle

In pragmatics, **face-threatening acts** (Brown & Levinson) are managed through hedging, indirection, and minimization. When someone says "sorry, kid stuff" instead of "I'm sorry, I need to step away because my daughter has a dance recital and I need to pick her up," they are:

1. **Minimizing the imposition** on the conversation (politeness)
2. **Maximizing the assumed shared knowledge** (if you knew about the dance recital, "kid stuff" is sufficient)

The result: **the less someone explains, the more they assume you know.** This is the inverse of what a naive keyword system would predict. A keyword system scores "my daughter has a dance recital" higher than "kid stuff" because it has more extractable content. But pragmatically, "kid stuff" is a STRONGER signal that memory is needed, because the speaker is relying on shared context to fill the gap.

### Detection Strategy

The aside/hedge markers (Signal 6) are the entry point. Once a hedge is detected, the scoring inverts:

- **Hedge + full explanation** ("btw, my daughter Emma, who plays soccer, has a tournament"): This is an introduction. The speaker is NOT assuming shared knowledge. Pragmatic score = LOW (the hedge is just a topic-shift marker).
- **Hedge + minimal reference** ("btw, Emma's tournament"): Moderate presupposition. Score = MEDIUM.
- **Hedge + bare minimum** ("kid stuff", "Marcus says hi", "the usual"): Maximum presupposition. Score = HIGH.

**Heuristic for explanation density**:
```
explanation_density = word_count_after_hedge / noun_count_after_hedge
```
- High density (>4 words per noun): speaker is explaining -> low presupposition
- Low density (<=2 words per noun): speaker is assuming -> high presupposition
- This is a rough proxy but captures the key asymmetry

**Alternative heuristic** (simpler, probably sufficient):
- Count proper names + relational nouns after the hedge marker.
- Count total words after the hedge marker.
- If total_words < 8 AND referent_count >= 1: high presupposition (score * 1.3)
- If total_words > 20 AND contains introduction markers: low presupposition (score * 0.4)

---

## 7. Computational Budget

All six signals are regex-based with small fixed lexicons. Estimated per-signal timings on a single core:

| Signal | Operations | Estimated Time |
|--------|-----------|---------------|
| Cross-session anaphora | 2 regex + session scan | 0.3ms |
| Bare proper name | 1 regex + set lookup + session check | 0.2ms |
| Possessive + relational | 1 regex + lexicon lookup | 0.1ms |
| Definite unintroduced entity | 2 regex + session check | 0.3ms |
| Continuation markers | 1 regex + session check | 0.2ms |
| Hedged aside + personal | 2 regex + word count | 0.3ms |
| **Noisy-OR combination** | 6 multiplies + 1 product | **<0.01ms** |
| **Total** | | **<1.5ms** |

This is well within the <5ms budget. The session context checks (`session_entity_names`, `recent_turns`) are O(N) string scans over small lists (typically <20 items).

All regex patterns should be pre-compiled at module level (as the existing code already does with `_FACT_QUERY_PREFIX`, `_FOLLOWUP_MARKERS`, etc.).

---

## 8. Data Structures

### 8.1 PragmaticSignals (returned alongside MemoryNeed)

```
PragmaticSignals:
    cross_session_anaphora: float      # 0.0-1.0
    bare_proper_name: float            # 0.0-1.0
    possessive_relational: float       # 0.0-1.0
    definite_unintroduced: float       # 0.0-1.0
    continuation_marker: float         # 0.0-1.0
    hedged_aside: float                # 0.0-1.0
    composite: float                   # noisy-OR combination
    detected_referents: list[str]      # extracted names/nouns for query hints
    presupposition_types: list[str]    # which signal types fired
```

### 8.2 Integration with MemoryNeed

Add optional field to existing `MemoryNeed`:
```
pragmatic_signals: PragmaticSignals | None = None
```

This preserves backward compatibility. The pragmatic analysis is computed alongside the existing keyword analysis, and the composite score from pragmatic signals feeds into the recall decision as described in Section 4.3.

---

## 9. What This Catches That the Current System Misses

| Input | Current System | With Pragmatic Layer |
|-------|---------------|---------------------|
| "picking up Emma from practice" | `none` (no keywords) | `presupposed_referent` (bare name "Emma") |
| "she loved the gift" | `none` | `presupposed_reference` (cross-session "she") |
| "still dealing with that bug" | `open_loop` (if recent_turns) / `none` | `presupposed_prior_state` + `presupposed_entity` ("that bug") |
| "my son's school called" | `none` | `presupposed_relationship` ("my son") |
| "btw Marcus says hi" | `none` | `presupposed_shared_context` (hedge + bare name) |
| "the proposal looks good" | `none` | `presupposed_entity` ("the proposal") |
| "finally shipped it" | `none` | `presupposed_prior_state` ("finally") |
| "back to the drawing board on auth" | `project_state` (has "auth") | `project_state` + `presupposed_prior_state` ("back to") — richer reasons |
| "sorry, kid stuff" | `none` | `presupposed_shared_context` (hedge + personal, high politeness inversion) |
| "Emma nailed her recital!" | `none` | `presupposed_referent` ("Emma") — urgency HIGH because the speaker clearly expects you to care |

The critical column is the transitions from `none` to a recall-positive verdict. These are conversations where the current system stays silent while the user expects the agent to know their world.

---

## 10. Non-Goals and Boundaries

- **This layer does NOT attempt discourse parsing.** It uses shallow pattern matching with lexicon lookups, not constituency or dependency parsing.
- **This layer does NOT model the speaker's mental model.** It detects linguistic presupposition triggers, not Theory of Mind inference.
- **This layer does NOT replace the existing keyword patterns.** It adds a parallel signal path that fires on the implicit cases the keyword path misses.
- **This layer does NOT require LLM calls.** Every detector is regex + set membership + arithmetic.
- **This layer is session-context-aware but NOT stateful across calls.** It uses `session_entity_names` and `recent_turns` (already passed to `analyze_memory_need()`) to distinguish intra-session from cross-session presuppositions. It does not maintain its own state between invocations.
