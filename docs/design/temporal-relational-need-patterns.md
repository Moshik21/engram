# Temporal-Relational Need Patterns

## Design Document: Structural Memory Need Detection

**Date:** 2026-03-06
**Status:** Design proposal
**Target:** `server/engram/retrieval/need.py` — `analyze_memory_need()`

---

## 1. Problem Statement

The current `analyze_memory_need()` in `server/engram/retrieval/need.py` uses keyword-pattern matching organized into flat buckets (acknowledgement, broad context, identity, open loop, temporal, project, fact query, followup). This catches explicit requests for memory ("remind me where we are", "what changed") but misses the vast majority of natural conversational moments where memory would be valuable.

The gap is structural: the analyzer looks for *what* words appear, not *how* the utterance is shaped. A human friend hearing "my son finally scored yesterday" would immediately connect it to everything they know about the child, the sport, and the parent's emotions. The current analyzer sees no keyword match and returns `need_type="none"`.

This document catalogs recurring conversational *shapes* — structural patterns that reliably predict recall value regardless of topic domain — and proposes detection heuristics fast enough to run on every turn.

---

## 2. Pattern Catalog

### Pattern 1: Life Update

**Description:** The user reports a change in their life or the life of someone they know. These are factual state transitions — something that was one way is now another.

**Examples:**
- "Emma started kindergarten last month"
- "We moved to Austin"
- "Got promoted to senior engineer"
- "My dad retired"
- "Sarah and I broke up"
- "We adopted a dog"

**Why memory is needed:** The system should recall what it knows about Emma (age, previous school mentions), the user's previous location, their previous job title, the dad's career, Sarah's role in the user's life, or any previous pet discussions. The update may contradict stored facts (lived in Denver → now Austin) requiring reconsolidation.

**Detection heuristic:**
- Possessive or relational noun + past-tense state-change verb
- State-change verbs: started, stopped, quit, joined, moved, married, divorced, graduated, retired, hired, fired, promoted, adopted, born, died, enrolled, transferred, switched, launched, sold, bought, closed, opened, diagnosed, recovered
- Pattern: `(my|our|his|her|their|RELATIONAL_NOUN) + PROPER_NAME? + STATE_CHANGE_VERB`
- Also: bare subject + state-change when session entities exist ("She started kindergarten" — who is "she"?)

**False positive risk:** LOW. State-change verbs with possessives or relational nouns are almost always personally relevant. Technical state changes ("the server started") are filtered by absence of relational framing.

**Confidence:** 0.85
**Need type:** `life_update`

---

### Pattern 2: Callback / Explicit Reference

**Description:** The user explicitly references a previous conversation or something the AI said.

**Examples:**
- "Remember when we talked about the Redis migration?"
- "Going back to what you said about testing"
- "Like we discussed last time"
- "You mentioned something about a caching layer"
- "That thing you brought up about error handling"

**Why memory is needed:** Direct request to surface prior context. High-confidence signal.

**Detection heuristic:**
- Already partially covered by `_TEMPORAL_PATTERNS` ("last time") and `_OPEN_LOOP_PATTERNS` ("follow up")
- Expand with: `(remember|recall|you (said|mentioned|suggested|recommended|brought up)|we (talked|discussed|decided|agreed)|going back to|as we discussed|like (we|you) said|that (thing|idea|approach) (you|we))`
- Second-person past tense is the strongest signal: "you said X" almost always needs recall

**False positive risk:** VERY LOW. These are explicit recall requests.

**Confidence:** 0.92
**Need type:** `callback`

---

### Pattern 3: Status Check

**Description:** The user asks about the current state of an ongoing project, task, or situation.

**Examples:**
- "How's the auth migration going?"
- "Any progress on the Redis thing?"
- "Where are we with the API redesign?"
- "What's the status of the hiring?"
- "Did the deploy go through?"

**Why memory is needed:** The system should recall the project/task entity, its last known state, recent related episodes, and any stored intentions or blockers.

**Detection heuristic:**
- Already partially covered by `_TEMPORAL_PATTERNS` ("status", "how's .* going") and `_PROJECT_PATTERNS`
- Strengthen with composition: question word + named entity or project reference + progress/state verb
- Pattern: `(how's|what's the status|any (progress|update|news)|where are we with|did .* (work|go through|land|ship|deploy|merge))`
- The presence of a proper noun or known entity name dramatically increases confidence

**False positive risk:** LOW when combined with entity detection. "How's it going?" as a greeting is filtered by lack of entity reference.

**Confidence:** 0.84
**Need type:** `status_check`

---

### Pattern 4: Correction / Contradiction

**Description:** The user corrects a previously stated fact, updates outdated information, or contradicts something in memory.

**Examples:**
- "Actually, it's not PostgreSQL, it's MySQL"
- "I was wrong about the deadline — it's next Friday"
- "That's changed — we're using Go now"
- "Not anymore — she left the company"
- "I should clarify — I meant the staging environment"
- "Scratch that, we decided to go with option B"

**Why memory is needed:** The system must find the contradicted fact to update or supersede it. Without recall, the correction is stored but the old fact persists, creating a contradiction in the knowledge graph.

**Detection heuristic:**
- Correction markers: `(actually|correction|to clarify|i meant|not anymore|that's (changed|wrong|not right|incorrect|outdated)|scratch that|never mind that|i was wrong|i misspoke|update:|wait no|on second thought)`
- Negation + past reference: `(it's not X, it's Y|no longer|doesn't .* anymore|stopped .* ing)`
- Contrastive structure: `not X but Y`, `X instead of Y`, `switched from X to Y`

**False positive risk:** LOW. Correction language is distinctive and almost always implies a prior stored fact that needs updating.

**Confidence:** 0.88
**Need type:** `correction`

---

### Pattern 5: Continuation / Follow-Through

**Description:** The user reports following up on something previously discussed — they tried the suggestion, implemented the plan, or took the next step.

**Examples:**
- "So I tried what you suggested with the caching"
- "I went ahead and refactored the auth module"
- "Ended up going with Postgres after all"
- "I finally got around to setting up the tests"
- "Took your advice and talked to the team"
- "Following up on that — I ran the benchmarks"

**Why memory is needed:** The system should recall the original suggestion, plan, or discussion to provide contextual follow-up. "I tried what you suggested" is meaningless without knowing what was suggested.

**Detection heuristic:**
- Follow-through markers: `(so i (tried|did|went|ended up|decided)|i (finally|actually|ended up) (did|tried|went|set up|implemented|built|wrote|fixed|deployed)|took your (advice|suggestion)|following up|as (planned|discussed|agreed)|went ahead and|gave .* a (try|shot))`
- Anaphoric reference + past tense action: "did that thing", "tried it", "went with that approach"
- Outcome reporting: `(it worked|it didn't work|it broke|it fixed|it helped|no luck|success|failed)`

**False positive risk:** MODERATE. "So I tried" could be about something new. Mitigated by checking for anaphoric references ("it", "that", "what you said") or session entity overlap.

**Confidence:** 0.80 (0.88 with anaphoric reference)
**Need type:** `continuation`

---

### Pattern 6: Comparison / Analogy

**Description:** The user draws a parallel between the current situation and a past one, or compares two things the system should know about.

**Examples:**
- "This is like the other project where we had the same issue"
- "Same problem as last time with the database"
- "Reminds me of when we dealt with the auth bug"
- "Similar to what happened with the Redis cluster"
- "Unlike the previous approach, this one actually works"
- "It's the opposite of what we did before"

**Why memory is needed:** The comparison is only meaningful if the system can retrieve the compared-to entity. "Same issue as last time" requires knowing what happened last time.

**Detection heuristic:**
- Comparison markers: `(like (the|that|when)|same (as|issue|problem|thing)|similar to|reminds me of|just like|unlike|opposite of|different from|better than (last|before|the old)|worse than|compared to|versus what we)`
- Temporal comparison: `(last time|before|previously|the old way|the first time|back when)`

**False positive risk:** LOW. Comparative structures with temporal or entity references are strong recall signals. Generic comparisons without entity anchors ("this is like a tree") are filtered by the entity/temporal co-occurrence requirement.

**Confidence:** 0.83
**Need type:** `comparison`

---

### Pattern 7: Introduction / New Entity

**Description:** The user introduces a new person, tool, concept, or project into the conversation for the first time.

**Examples:**
- "So my coworker Jake has been working on this"
- "There's this new framework called Remix"
- "I started talking to a recruiter"
- "My therapist suggested I try journaling"
- "The new CTO wants to rewrite everything"
- "Have you heard of Bun?"

**Why memory is needed:** Paradoxically, introductions need recall to check whether the entity is actually *new*. "My coworker Jake" — do we already know Jake? Is he the same Jake from a previous conversation? The system should check for existing entity matches and surface any prior context.

**Detection heuristic:**
- Introduction frames: `(my (coworker|colleague|friend|boss|manager|therapist|doctor|teacher|mentor|partner|wife|husband) NAME|there's this (new )?(tool|framework|library|service|app|person|thing)|started (using|working with|talking to)|have you heard of|do you know (about )?NAME|met (someone|a guy|a woman|this person))`
- Possessive + role + proper name: "my friend Sarah", "our CEO Mark"
- Demonstrative + descriptor + name: "this guy Jake", "that framework Remix"

**False positive risk:** LOW for possessive+role+name. MODERATE for bare introductions without names ("started using a new tool" — too vague for entity lookup).

**Confidence:** 0.78
**Need type:** `introduction`

---

### Pattern 8: Emotional Anchor

**Description:** The user expresses emotion tied to a specific entity, situation, or outcome. Emotional framing signals personally significant memory content.

**Examples:**
- "I'm really excited about the new role"
- "Frustrated with the deployment process again"
- "Super proud of how the team handled it"
- "Worried about the reorg"
- "Finally feeling good about the architecture"
- "I dread these sprint planning meetings"

**Why memory is needed:** Emotional anchors create strong retrieval cues in human memory. The system should recall context about the entity to provide emotionally appropriate responses. "Frustrated with X again" implies a history of frustration the system should acknowledge.

**Detection heuristic:**
- Emotion word + entity/situation reference: `(excited|frustrated|worried|proud|grateful|anxious|stressed|happy|sad|angry|disappointed|overwhelmed|relieved|thrilled|dreading|loving|hating|annoyed|confused) (about|with|by|that|over) .*`
- Intensifiers amplify: "really", "so", "super", "incredibly", "absolutely"
- "Again" or "still" + emotion = recurring pattern (higher value)
- "Finally" + positive emotion = resolution of prior negative state

**False positive risk:** MODERATE. Emotional language appears in non-personal contexts ("excited about the new API" in a technical review). Mitigated by checking for first-person framing ("I'm", "I feel", "makes me").

**Confidence:** 0.76 (0.85 with "again"/"still"/"finally" modifier)
**Need type:** `emotional_anchor`

---

### Pattern 9: Temporal Narrative Thread

**Description:** The user tells a multi-step story anchored in time — a sequence of events that memory should contextualize.

**Examples:**
- "So first I tried the cache approach, then switched to direct queries, and now I'm looking at materialized views"
- "Last week I set up the CI pipeline, yesterday I added the tests, today I'm doing the deploy"
- "It started with the memory leak in January, got worse in February, and we finally traced it to the connection pool last week"

**Why memory is needed:** Each step in the narrative may reference stored entities and events. The system should fill in gaps — was the cache approach discussed previously? Is the memory leak a known entity?

**Detection heuristic:**
- Sequence markers in a single turn: 3+ of (first, then, after that, next, now, finally, eventually, later, meanwhile, at that point)
- Temporal progression: mix of past and present tense with temporal anchors
- Pattern: `(first|initially|started) .* (then|after|next) .* (now|finally|currently|today)`

**False positive risk:** LOW. Multi-step temporal narratives almost always contain recallable entities. Technical sequences ("first compile, then link, then run") are lower value but still harmless to recall against.

**Confidence:** 0.82
**Need type:** `temporal_narrative`

---

### Pattern 10: Implicit Question About Stored Knowledge

**Description:** The user makes a statement that implicitly asks whether the system knows something, without using question syntax.

**Examples:**
- "I can't remember if we decided on Postgres or MySQL"
- "Not sure what the team agreed on for the API versioning"
- "I forget whether Sarah is in the London or Berlin office"
- "Something about the rate limiting — I can't recall the details"
- "There was a reason we chose that approach but I forgot"

**Why memory is needed:** The user is explicitly acknowledging a memory gap and implicitly requesting the system to fill it. These are high-value recall moments disguised as statements.

**Detection heuristic:**
- Memory-gap language: `(i (can't|cannot|don't) remember|i forget|not sure (if|what|whether|which|who|where|when)|can't recall|what was it|there was (a|some) (reason|decision|plan|thing)|something about .* (can't|don't) (remember|recall))`
- Hedged references: "I think we said...", "wasn't it something like...", "if I recall correctly..."

**False positive risk:** VERY LOW. Explicit memory-gap language is one of the strongest recall signals possible.

**Confidence:** 0.90
**Need type:** `memory_gap`

---

### Pattern 11: Preference Revelation Through Action

**Description:** The user reveals a preference not by stating it directly ("I prefer X") but by describing behavior that implies it.

**Examples:**
- "I always end up using vim for quick edits"
- "I keep coming back to Python for these scripts"
- "Whenever I have to choose, I go with the simpler option"
- "I never use the GUI — always the CLI"
- "For some reason I always start projects in TypeScript"

**Why memory is needed:** These behavioral preferences should be cross-referenced with stored preferences and past choices. "I always end up using vim" — has the user previously mentioned an editor preference? Is this consistent or a change?

**Detection heuristic:**
- Habitual markers + action: `(i always|i never|i (keep|tend to|usually|normally|typically|generally) (use|choose|pick|go with|prefer|start with|end up|default to|gravitate toward)|whenever i .* i (use|choose|go with)|for some reason i)`
- Frequency + action: "every time I...", "9 times out of 10 I..."

**False positive risk:** MODERATE. "I always use git commit" is a preference but may be too generic. Mitigated by checking for entity specificity — preferences about named tools/approaches are more valuable.

**Confidence:** 0.74
**Need type:** `implicit_preference`

---

### Pattern 12: Anticipatory / Planning Statement

**Description:** The user describes future plans, goals, or intentions that relate to stored context.

**Examples:**
- "I'm thinking about switching to Rust for the backend"
- "We're planning to hire two more engineers next quarter"
- "I want to eventually move to a microservices architecture"
- "Next sprint we should tackle the auth refactor"
- "If this works out, I might leave my job"

**Why memory is needed:** Plans should be checked against stored goals/intentions (prospective memory), prior discussions of the same topic, and potentially contradictory stored facts ("didn't you say last month you wanted to stay with Python?").

**Detection heuristic:**
- Planning language + entity: `(thinking about|planning to|want to|going to|considering|might|hoping to|aiming to|need to eventually|should probably|next (sprint|quarter|month|week|year) .* (we|i)|goal is to|end goal|long term)`
- Conditional planning: "if X works out, then Y"
- Aspiration language: "dream of", "eventually want to", "working toward"

**False positive risk:** LOW-MODERATE. "Planning to" with a named entity is high-signal. "Thinking about it" without specifics is too vague.

**Confidence:** 0.77
**Need type:** `planning`

---

### Pattern 13: Identity Claim / Self-Description

**Description:** The user defines or redefines who they are — their role, characteristics, beliefs, or identity.

**Examples:**
- "I'm more of a backend person"
- "I consider myself a generalist"
- "I'm the kind of person who needs a plan"
- "I've always been a morning person"
- "At heart, I'm really a designer who codes"
- "I identify as someone who..."

**Why memory is needed:** Identity statements are among the most durable and valuable facts to store and recall. They should be cross-referenced with stored identity facts for consistency and to detect evolution ("used to be a backend person, now more full-stack").

**Detection heuristic:**
- Self-description frames: `(i('m| am) (more of|kind of|the type|the kind|really|basically|fundamentally|at heart)|i consider myself|i identify as|i've always been|i see myself as|people (say|think|know) i'm|my (strength|weakness|style) is|i'm (a|an) .* (person|type|developer|engineer|designer|thinker))`

**False positive risk:** LOW. Self-description frames are distinctive and almost always identity-relevant.

**Confidence:** 0.85
**Need type:** `identity_claim`

---

### Pattern 14: Social Graph Update

**Description:** The user describes a change in their relationship network — someone new, a role change, or a relationship status update.

**Examples:**
- "Jake got moved to a different team"
- "My manager is leaving next month"
- "Sarah is now the tech lead"
- "We have a new PM — her name is Priya"
- "My brother and his wife are expecting"
- "Tom and I aren't working together anymore"

**Why memory is needed:** The system should recall the mentioned person's entity, their current stored role/relationship, and update accordingly. "My manager is leaving" requires knowing who the manager is.

**Detection heuristic:**
- Known entity name + role/relationship change verb: `NAME + (got (promoted|moved|transferred|fired|hired)|is (leaving|joining|now|starting)|became|was (promoted|transferred)|aren't .* anymore)`
- Relational noun + change: `my (manager|boss|lead|partner|coworker) + (is|was|got) + CHANGE_VERB`
- New relationship introduction: "we have a new ...", "there's a new ..."

**False positive risk:** LOW. Relationship change language with proper names or relational nouns is high-signal.

**Confidence:** 0.84
**Need type:** `social_graph_update`

---

### Pattern 15: Explanatory Context ("The Reason Is...")

**Description:** The user provides the *reason* behind a previously observed fact or decision. This enriches stored knowledge with causal links.

**Examples:**
- "The reason we chose FalkorDB is because of the Cypher support"
- "I picked TypeScript because of the type safety"
- "The whole Redis migration happened because of the latency issues"
- "We went with microservices since the team was growing"
- "That's why I always test locally first"

**Why memory is needed:** Causal explanations should be linked to the entity they explain. "The reason we chose FalkorDB" — the system should recall the FalkorDB entity and attach this causal context to the existing `USES` relationship.

**Detection heuristic:**
- Causal frames: `(the reason (is|was|we|i)|because of|that's (why|because)|due to|since .* (we|i) (decided|chose|went|picked|switched)|this is why|the whole .* happened because)`
- "Picked/chose/went with X because Y" structure

**False positive risk:** LOW. Causal frames with entity references are almost always high-value enrichment of existing knowledge.

**Confidence:** 0.79
**Need type:** `causal_context`

---

### Pattern 16: Recurring Problem

**Description:** The user describes something happening again — a bug, an annoyance, a failure mode. The recurrence itself is the signal.

**Examples:**
- "The builds are failing again"
- "Same Redis timeout issue"
- "This keeps happening with the auth service"
- "Third time this week the deploy broke"
- "Still getting that weird error in production"
- "Back to square one with the migration"

**Why memory is needed:** Recurrence implies prior episodes about the same problem. The system should surface previous occurrences, attempted fixes, and any patterns.

**Detection heuristic:**
- Recurrence markers: `(again|same .* (issue|problem|bug|error)|keeps (happening|breaking|failing|crashing)|still (getting|seeing|having)|third time|nth time|back to (square one|the drawing board)|yet again|once more|as usual)`
- "Still" + present progressive + negative outcome

**False positive risk:** LOW. Recurrence language is distinctive and high-value for connecting temporal episodes.

**Confidence:** 0.84
**Need type:** `recurring_problem`

---

### Pattern 17: Delegation / Handoff Reference

**Description:** The user references work handed off to or received from someone else, or delegates something to the AI.

**Examples:**
- "I told Jake to handle the monitoring setup"
- "Sarah sent me the API docs"
- "Can you take over from where we left off on the schema?"
- "The client sent their requirements"
- "I'm passing the frontend work to the new hire"

**Why memory is needed:** Delegation creates dependency chains. "Take over from where we left off" explicitly requires recalling prior state. "I told Jake to handle X" should be linked to Jake's entity and tracked as a pending item.

**Detection heuristic:**
- Delegation verbs with person reference: `(told|asked|assigned|delegated|handed off|passed) + NAME/ROLE + to (do|handle|take|work on|finish|set up)`
- Receipt: `(NAME|ROLE) (sent|gave|shared|forwarded|passed) me`
- Handoff request: `(take over|pick up where|continue from|carry on with)`

**False positive risk:** LOW-MODERATE. "I told the compiler to optimize" is a false positive, but the person/role requirement filters most technical language.

**Confidence:** 0.78
**Need type:** `delegation`

---

### Pattern 18: Milestone / Achievement

**Description:** The user reports reaching a significant checkpoint — something working, a goal achieved, a release shipped.

**Examples:**
- "We finally shipped v2.0"
- "All tests are passing now"
- "Hit 10K users this week"
- "The migration is done"
- "Got the green light from the board"
- "Passed the security audit"

**Why memory is needed:** Milestones close open loops. The system should recall the project/goal entity, mark it as achieved, and surface any related next steps or intentions.

**Detection heuristic:**
- Achievement frames: `(finally (shipped|launched|finished|completed|done|released|deployed|passed|hit)|all .* (passing|working|done|complete|green)|got the (green light|approval|sign-off)|reached|achieved|milestone|shipped|went live|in production)`
- Quantitative achievement: number + growth/scale word ("hit 10K", "reached 99.9% uptime")

**False positive risk:** LOW. Achievement language combined with entity references is high-signal.

**Confidence:** 0.81
**Need type:** `milestone`

---

## 3. Temporal Signal Detection

Time references are the most underused recall signals in the current system. The existing `_TEMPORAL_PATTERNS` catch "latest", "recently", "last time", but miss the deeper temporal structures that imply timeline recall.

### 3.1 Relative Time Anchors

These position events on the user's personal timeline and imply episodic memory lookup.

| Category | Examples | Recall implication |
|----------|----------|-------------------|
| Recent past | "yesterday", "earlier today", "this morning", "last night" | Search episodes within 24-48 hours |
| Near past | "last week", "a few days ago", "the other day" | Search episodes within 7-14 days |
| Medium past | "last month", "a few weeks ago", "recently" | Search episodes within 30-60 days |
| Far past | "a few months ago", "last year", "way back", "ages ago" | Broad entity search, not episode search |
| Relational time | "when we started", "when I joined", "since the reorg", "after the launch" | Named-event temporal anchor — resolve the event entity first, then find temporally adjacent episodes |

**Detection regex:**
```
\b(yesterday|earlier today|this morning|last night|
last (week|month|year|quarter|sprint)|
(a few|couple of?) (days|weeks|months|years) ago|
the other day|recently|way back|ages ago|
when (we|i|you) (started|began|joined|launched|shipped|moved|switched)|
since (the|we|i) \w+|
after (the|we|i) \w+|
before (the|we|i) \w+)\b
```

**Scoring:** Relative time anchors alone are worth 0.25-0.35 confidence. Combined with an entity reference, they jump to 0.70+.

### 3.2 Sequence Markers

Sequence markers indicate narrative structure that benefits from timeline reconstruction.

| Marker | Pattern | Implication |
|--------|---------|-------------|
| Ordering | "first... then... after that... finally" | Multi-event narrative; retrieve all referenced events |
| Consequence | "and then", "so after that", "which led to" | Causal chain; retrieve the prior cause |
| Interruption | "but then", "until", "but before that" | Exception to expected sequence |
| Resumption | "anyway, after that", "back to", "so then" | Return to interrupted narrative |

**Detection:** Count sequence markers per turn. 2+ markers = `temporal_narrative` pattern (Pattern 9). A single "then" is too weak alone but compounds with other signals.

### 3.3 Duration Markers

Duration implies an ongoing state the system should be tracking.

| Marker | Examples | Recall implication |
|--------|----------|-------------------|
| Ongoing since | "for a while now", "since January", "for months", "all year" | Check for stored start-event; track duration |
| Recent start | "just started", "only been a week", "brand new" | Check for related planning episodes |
| Long-running | "it's been months", "going on two years", "forever" | High-maturity entity; deep history expected |

**Detection regex:**
```
\b(for (a while|months|weeks|years|ages)|
since (january|february|...|last|the)|
it's been (months|weeks|years|a while|ages|forever)|
(just|recently|newly) (started|began|joined)|
going on (\d+|two|three|several) (months|years|weeks)|
all (year|month|week|quarter))\b
```

**Scoring:** Duration markers alone are 0.20. With a named entity, 0.55+. With "still" or "keeps" modifier, 0.70+.

### 3.4 Frequency Markers

Frequency implies repeated episodes that should be connected.

| Marker | Examples | Recall implication |
|--------|----------|-------------------|
| Repetition | "again", "once more", "another time" | Find prior occurrences |
| Habitual | "always", "every time", "usually", "tends to" | Check stored behavior patterns |
| Increasing | "more and more", "increasingly", "getting worse" | Track trend across episodes |
| Cessation | "not anymore", "stopped", "used to but..." | State transition; find the prior state |

**Detection regex:**
```
\b(again|once more|another (time|round|attempt)|
always|every (time|day|week|sprint)|
(more and more|increasingly|getting (worse|better|harder|easier))|
(not|never) anymore|stopped|used to|no longer)\b
```

**Scoring:** "Again" alone is 0.30. "Again" + entity name is 0.65. "Always" + action verb is 0.50 (habitual preference detection).

### 3.5 Temporal Contrast

Temporal contrast signals state transitions that may require knowledge graph updates.

| Marker | Examples | Recall implication |
|--------|----------|-------------------|
| Past vs present | "used to X, now Y", "was X, switched to Y" | Find and potentially supersede stored fact |
| Before/after | "before the reorg... after the reorg..." | Retrieve both temporal windows |
| Change point | "ever since X", "once we switched to Y" | Identify the change event as a temporal anchor |

**Detection regex:**
```
\b(used to .* (now|but now|currently)|
was .* (switched|changed|moved|migrated) to|
before .* (now|after|since)|
ever since (we|i|the)|
once (we|i) (switched|changed|moved|started))\b
```

**Scoring:** Temporal contrast is 0.40 alone, 0.80+ with named entities. These are strong correction/update signals.

---

## 4. Relational Signal Detection

### 4.1 Relational Noun Lexicon

Organized by domain for weighted scoring:

**Family (highest memory value — 0.40 base):**
mom, dad, mother, father, son, daughter, brother, sister, wife, husband, partner, spouse, child, children, kids, grandma, grandmother, grandpa, grandfather, aunt, uncle, cousin, niece, nephew, in-laws, stepmother, stepfather, stepsister, stepbrother, stepdaughter, stepson, fiancé, fiancée, ex, ex-wife, ex-husband, baby

**Professional (high — 0.35 base):**
boss, manager, coworker, colleague, teammate, mentor, mentee, intern, lead, director, VP, CEO, CTO, PM, tech lead, skip-level, report, direct report, founder, co-founder, client, contractor, recruiter, HR, the new hire, the new guy/girl/person

**Social (medium — 0.30 base):**
friend, best friend, roommate, neighbor, classmate, acquaintance, buddy, pal, ex-girlfriend, ex-boyfriend

**Medical/Care (high — 0.35 base):**
doctor, therapist, psychiatrist, counselor, dentist, surgeon, specialist, nurse, vet, trainer, coach, tutor, teacher, professor

### 4.2 Detection Patterns

**Possessive + Relational Noun:**
```
(my|our|his|her|their) + RELATIONAL_NOUN
```
Examples: "my boss", "our CEO", "his sister"
Confidence: 0.35-0.40 depending on category

**Possessive + Relational Noun + Proper Name:**
```
(my|our) + RELATIONAL_NOUN + PROPER_NAME
```
Examples: "my coworker Jake", "our CEO Sarah", "my sister Emma"
Confidence: 0.50+ (strong entity resolution signal)

**Relational Noun + Proper Name (no possessive):**
```
RELATIONAL_NOUN + PROPER_NAME
```
Examples: "friend Sarah", "manager Tom"
Confidence: 0.40

**Role Reference (definite article):**
```
(the) + (new|old|former|current|previous) + ROLE_NOUN
```
Examples: "the new hire", "the old manager", "the former CTO"
Confidence: 0.30

### 4.3 Known vs New Entity Discrimination

This is the critical question: when someone mentions a person, should the system recall or just observe?

**Decision algorithm:**

1. **Extract the entity reference** — name, relational noun, or both
2. **Check entity index** — `find_entity_candidates(name, group_id)` for proper names
3. **If match found:** This is a KNOWN entity. Recall is valuable. Return `should_recall=True` with the entity ID as query hint.
4. **If no match found but relational noun present:** Semi-known. The relational role may match even if the name doesn't. Search for the role ("the user's coworker") — there may be a stored entity with that relationship to the user.
5. **If no match at all:** NEW entity. Recall is less critical but still useful to check for partial matches (same first name, same role). Return `should_recall=True` with lower confidence (0.55) and a narrower entity budget (2).

**Bare name heuristic (hardest case):**

When the user says just "Sarah" or "Jake" without relational context:
- Check `session_entity_names` — was this name mentioned earlier in the session? If yes, it's a known-in-session entity; use session context.
- Check entity store for exact or fuzzy match.
- If the name matches a known entity, recall is high-value (0.75+).
- If no match, the name may be new. Still worth a low-confidence recall (0.45) to catch near-matches.
- **Ambiguity flag:** If multiple entities match the bare name, set `entity_budget=3+` and add `ambiguous_entity` to reasons.

**Common-word names (Will, Grace, Ruby, Heather, Mark, etc.):**

Maintain a set of ~50 English first names that are also common words. For these names:
- Require at least one additional signal (relational noun, capitalization at non-sentence-start, possessive framing) before treating as a person reference
- "Will you help me?" — not a person. "My friend Will" — person.
- Detection: check if the token appears at sentence start (likely common word) vs mid-sentence with capitalization (likely name)

---

## 5. Pattern Composition Model

### 5.1 The Problem

Individual signals are often weak. "Yesterday" alone is 0.25. "My son" alone is 0.35. But "my son scored yesterday" together is 0.80+. How do you model this?

### 5.2 Proposed Composition Function: Noisy-OR with Category Boost

The system already uses noisy-OR in `plan.py`'s `merge_support()`. Extend this to signal composition:

```
composite = 1 - product(1 - signal_i for each detected signal_i)
```

This gives diminishing returns for redundant signals but ensures that *any* strong signal dominates.

**Category signals** (each contributes independently):

| Category | Signal source | Weight range |
|----------|--------------|-------------|
| Temporal | Time anchor, sequence, duration, frequency, contrast | 0.15 - 0.40 |
| Relational | Relational noun, possessive+name, bare name match | 0.20 - 0.50 |
| Emotional | Emotion word, intensifier, "again"/"still"/"finally" | 0.15 - 0.35 |
| Structural | Pattern match (correction, continuation, milestone, etc.) | 0.30 - 0.88 |
| Entity | Named entity found in store, session entity overlap | 0.20 - 0.50 |
| Anaphoric | "it", "that", "this" + action verb (unresolved reference) | 0.15 - 0.30 |

**Composition example: "My son finally scored yesterday"**

1. Relational signal: "my son" → 0.40
2. Temporal signal: "yesterday" → 0.25
3. Emotional signal: "finally" → 0.25
4. Structural signal: life_update (state change "scored") → 0.35

Noisy-OR: `1 - (0.60 * 0.75 * 0.75 * 0.65) = 1 - 0.219 = 0.78`

Then apply a **cross-category bonus**: when 3+ categories fire, apply a 1.15x multiplier (capped at 0.95):

`min(0.95, 0.78 * 1.15) = 0.90`

This matches intuition: four independent weak signals pointing toward recall need is strong evidence.

### 5.3 Anti-Composition (Signal Damping)

Some signal combinations should *reduce* confidence:

- **Technical temporal + no relational** = likely process description, not personal timeline
  - "After the build completes, deploy to staging" — temporal but not personal
  - Damp by 0.6x when temporal signals fire but no relational/emotional/entity signals

- **"My" + non-relational noun** = possessive but not about people
  - "My PR", "my branch", "my repo", "my deployment"
  - Maintain a set of ~30 technical possessives that should not trigger relational detection

- **Generic emotion without entity** = mood statement, not recall need
  - "I'm frustrated" alone is weaker than "frustrated with the auth migration"
  - Damp emotional signal by 0.5x when no entity reference is present

---

## 6. Session vs Cross-Session Differentiation

### 6.1 Session-Start Patterns (Cross-Session Recall)

The first 1-3 turns of a new session carry outsized recall value. People naturally pick up threads.

**High-value session-start signals:**

| Pattern | Example | Signal strength |
|---------|---------|----------------|
| Thread resumption | "Where were we?", "Picking up where we left off" | 0.90 |
| Topic re-entry | "So about the migration..." (first turn, known entity) | 0.82 |
| Status request | "Any updates?" (first turn) | 0.85 |
| Time-gap reference | "It's been a while", "Long time no talk" | 0.75 |
| Progress report | "I've been working on X since we last talked" | 0.85 |
| New context for old topic | "So I tried the Postgres approach" (continuation from prior session) | 0.80 |

**Detection:** Check `recent_turns` length and/or `session_turn_index` (if available). When `session_turn_index <= 2`:
- Boost all entity-matching signals by 1.3x (cross-session entity references are higher value)
- Activate thread-resumption detection
- Lower the "none" confidence threshold (more willing to recall on weak signals)

### 6.2 Mid-Session Patterns (Within-Session Recall)

Within a long session, memory need shifts from "what do we know?" to "what did we say earlier?"

**Mid-session recall signals:**

| Pattern | Example | Signal strength |
|---------|---------|----------------|
| Callback to earlier turns | "Like I mentioned earlier", "Going back to..." | 0.85 |
| Topic switch + old entity | "Oh, and about Jake..." (topic change to known entity) | 0.70 |
| Summarization request | "So to recap what we've covered..." | 0.80 |
| Contradiction with earlier turn | User says X now but said Y 20 turns ago | 0.85 |
| Abandoned thread pickup | Returns to topic from 10+ turns ago | 0.75 |

**Detection:** When `len(recent_turns) > 10`:
- Check for topic shift (current turn's entity set differs from last 3 turns)
- Check for temporal backtracking language ("earlier", "before", "when we were talking about")
- These signals should query the session's own episode history, not just the long-term graph

### 6.3 Session Position Modifier

Apply a position-dependent modifier to the composite score:

```
if session_turn_index <= 2:
    modifier = 1.3  # Cross-session bonus
elif session_turn_index <= 5:
    modifier = 1.1  # Early-session mild bonus
elif session_turn_index > 20:
    modifier = 1.05  # Long-session slight bonus (more context to lose)
else:
    modifier = 1.0  # Normal mid-session
```

---

## 7. The Long Tail: Rare but High-Value Patterns

### 7.1 Anniversary / Date-Based Recall

**Pattern:** The user mentions a date or duration that aligns with a stored event.

**Examples:**
- "It's been a year since we launched"
- "Can you believe it's been six months since the reorg?"
- "One year anniversary at the company"
- "This time last year we were still on the old system"

**Detection:** Requires matching temporal expressions against stored entity timestamps. This is expensive — better suited for a background check that runs when temporal markers are detected, not on every turn.

**Implementation:** When a duration or anniversary expression is detected, compute the implied date and query episodes/entities created near that date. This is a post-detection enrichment step, not a primary signal.

**Confidence:** 0.80 when a matching entity is found near the computed date.

### 7.2 Contradiction Detection

**Pattern:** The user states something that conflicts with a stored fact.

**Examples:**
- Stored: "User works at Google" → User says: "My new job at Microsoft is great"
- Stored: "User prefers Python" → User says: "I've been doing everything in Rust lately"
- Stored: "User lives in Denver" → User says: "The Austin weather is nice"

**Detection:** This cannot be a pattern-matching heuristic — it requires comparing the current turn against stored facts. However, the *opportunity* for contradiction can be signaled:
- When the current turn contains entity + relational verb + different value than stored (requires entity lookup)
- Approximate: when the turn mentions a known entity + a state verb ("works at", "lives in", "uses") + a value that differs from the entity's stored attributes

**Implementation:** Flag turns that match the pattern `KNOWN_ENTITY + STATE_VERB + VALUE` and inject a contradiction-check step after initial entity retrieval. This is a two-phase process: first detect the structural pattern, then verify against stored facts.

**Confidence:** 0.70 for structural pattern match, 0.92 if actual contradiction confirmed.

### 7.3 Goal Progress

**Pattern:** The user describes progress toward a stored intention or goal.

**Examples:**
- Stored intention: "Migrate to Postgres" → "The Postgres migration is 80% done"
- Stored intention: "Learn Rust" → "Finished chapter 5 of the Rust book"
- Stored intention: "Hire a senior engineer" → "We have three candidates in the final round"

**Detection:** Requires matching against stored prospective memory (intentions). When the current turn mentions an entity that appears in a stored intention, and the turn contains progress language ("done", "finished", "completed", "halfway", "started", "making progress"), trigger recall.

**Approximate heuristic:** Progress language + known entity name → recall with `intent_type="goal_progress"`.

**Confidence:** 0.75 on pattern match, 0.88 if stored intention match confirmed.

### 7.4 Expertise Signal

**Pattern:** The user demonstrates deep knowledge about a topic they haven't discussed before, or asks about something at a level that implies existing knowledge the system might have stored.

**Examples:**
- "The B-tree index on that column is probably causing the write amplification"
- "Have you considered using CRDTs for the conflict resolution?"
- "The issue is in the connection pool — probably the idle timeout"

**Detection:** Technical depth (specific terminology, causal reasoning) + entity reference. This is hard to detect structurally and may not be worth the false positive cost. Better handled by the entity-matching system than the need analyzer.

**Confidence:** 0.55 (too weak for primary detection; better as a secondary signal).

---

## 8. Failure Modes and Mitigations

### 8.1 Technical Temporal Language

**Problem:** "After the build completes, run the tests" has temporal structure but no personal memory value.

**Mitigation:**
- Check for imperative mood (instructions/commands)
- Check for technical process nouns as subjects ("the build", "the deploy", "the pipeline", "the server")
- Apply 0.6x damping when temporal signals fire without relational, emotional, or personal entity signals
- The absence of first-person or possessive framing is a strong negative indicator

**Concrete rule:** If temporal signals fire AND the sentence subject is a technical noun (not "I", "we", "my", a person name, or a relational noun), dampen temporal contribution by 0.6x.

### 8.2 "My" in Non-Relational Context

**Problem:** "My PR", "my branch", "my environment", "my terminal" use possessive framing but don't indicate relationships worth recalling.

**Mitigation:**
- Maintain a technical possessive exclusion set:
  ```
  PR, branch, repo, fork, commit, deployment, build, pipeline, container,
  pod, cluster, instance, environment, terminal, editor, IDE, workspace,
  config, settings, package, module, codebase, project (when referring to
  code, not human endeavor), test, suite, directory, file, function,
  class, method, variable, database, schema, table, query, endpoint, API,
  server, service, lambda, stack, bucket, queue
  ```
- When "my" precedes a word in this set, do NOT count it as a relational signal
- "My project" is ambiguous — could be personal or technical. Default to technical unless other signals (emotional, identity) co-occur.

### 8.3 Ambiguous Names

**Problem:** "Will", "Grace", "Ruby", "Mark", "Crystal", "Iris", "Sage", "Jasper", "Chase", "Faith", "Hope", "June", "May", "Dawn", "Bill", "Rich", "Art", "Pat"

**Mitigation:**
- Maintain an ambiguous-name set (~50 entries)
- For names in this set, require at least one additional signal:
  - Relational framing: "my friend Will"
  - Mid-sentence capitalization (not at sentence start)
  - Verb following the name that implies a person: "Will said", "Grace thinks"
  - Prior mention in session entities
- Without additional signals, treat as common word and do not trigger entity lookup

### 8.4 Over-Triggering on Casual Conversation

**Problem:** Not every personal statement needs recall. "I had pizza for lunch" is personal but rarely worth recalling unless the system stores dietary preferences.

**Mitigation:**
- The composition function naturally handles this: a single weak signal ("I had") without entity match, temporal anchor, emotional marker, or structural pattern stays below threshold
- Minimum composite threshold for `should_recall=True`: 0.55
- For borderline cases (0.55-0.70), use lower `packet_budget=1` and `entity_budget=2` to minimize recall cost

### 8.5 Command/Instruction Confusion

**Problem:** "Remember to deploy the staging build" — is this a recall request or a prospective memory instruction?

**Mitigation:**
- "Remember to" + infinitive = prospective (future task), not recall request
- "Remember when" + past tense = callback (recall request)
- "Remember that" + fact = could be either; check for question intonation or entity match
- Pattern: `remember to \w+` → route to prospective, not recall

---

## 9. Integration Design

### 9.1 Augmented `MemoryNeed` Model

The existing `MemoryNeed` dataclass is sufficient. Proposed additions to the `need_type` vocabulary:

**New need types:** `life_update`, `callback`, `status_check`, `correction`, `continuation`, `comparison`, `introduction`, `emotional_anchor`, `temporal_narrative`, `memory_gap`, `implicit_preference`, `planning`, `identity_claim`, `social_graph_update`, `causal_context`, `recurring_problem`, `delegation`, `milestone`

**New fields on MemoryNeed (optional):**
- `signals: list[str]` — which individual signals fired (e.g., `["temporal:yesterday", "relational:my_son", "emotional:finally"]`)
- `category_scores: dict[str, float]` — per-category contributions before composition
- `session_position: str` — "start" | "early" | "mid" | "late" (for session-aware scoring)

### 9.2 Processing Flow

The augmented `analyze_memory_need()` should follow this order:

1. **Short-circuit checks** (unchanged): empty turn, acknowledgement
2. **Explicit recall patterns** (unchanged but expanded): broad context, identity, open loop, callback, memory gap
3. **Signal extraction** (new):
   a. Extract temporal signals (anchors, sequences, durations, frequencies, contrasts)
   b. Extract relational signals (nouns, possessives, names)
   c. Extract emotional signals (emotion words, intensifiers, modifiers)
   d. Extract structural pattern matches (all 18 patterns)
   e. Extract entity signals (named terms, session entity overlap)
   f. Extract anaphoric signals (unresolved references)
4. **Apply dampening** (new): technical temporal, non-relational possessive, generic emotion
5. **Compose score** (new): noisy-OR across categories with cross-category bonus
6. **Apply session modifier** (new): position-based adjustment
7. **Threshold and classify**: if composite >= 0.55, find the dominant pattern and set `need_type`

### 9.3 Performance Budget

The current `analyze_memory_need()` is pure regex — sub-millisecond. The augmented version must stay under 2ms per turn to avoid adding latency to every interaction.

**Cost breakdown:**
- Regex scanning (all patterns): ~0.3ms (compile patterns once, reuse)
- Entity store lookup (`find_entity_candidates`): ~1ms (already indexed via FTS5)
- Score composition: ~0.01ms
- Total: ~1.3ms

The entity store lookup is the only potentially expensive operation. It should be optional — only triggered when entity signals are detected. For turns with no named terms and no relational nouns, skip entity lookup entirely.

### 9.4 Interaction with Triage Policy

The need analyzer (`need.py`) operates at recall time — deciding whether to query memory for a given turn. The triage policy (`triage_policy.py`) operates at storage time — deciding whether to extract entities from an episode. These are complementary:

- **Triage policy** already detects corrections, preferences, profiles, tasks, temporal markers for *storage* priority
- **Need analyzer** should detect the same signals (plus more) for *recall* priority
- Shared signal extraction could be factored into a common `signal_extractor.py` module to avoid duplication

The patterns in this document should also feed into the triage scorer's `structural_extractability` signal — a turn that matches multiple need patterns is also likely to produce extractable entities if stored.

---

## 10. Summary: Pattern Priority Matrix

Patterns ranked by (confidence x recall value x frequency):

| Priority | Pattern | Confidence | Recall Value | Frequency | Score |
|----------|---------|-----------|-------------|-----------|-------|
| 1 | Callback | 0.92 | Very High | Medium | 0.92 |
| 2 | Memory Gap | 0.90 | Very High | Low-Medium | 0.90 |
| 3 | Correction | 0.88 | Very High | Medium | 0.88 |
| 4 | Identity Claim | 0.85 | Very High | Low | 0.85 |
| 5 | Life Update | 0.85 | High | Medium | 0.85 |
| 6 | Status Check | 0.84 | High | High | 0.84 |
| 7 | Social Graph Update | 0.84 | High | Medium | 0.84 |
| 8 | Recurring Problem | 0.84 | High | Medium | 0.84 |
| 9 | Comparison | 0.83 | High | Medium | 0.83 |
| 10 | Temporal Narrative | 0.82 | High | Low | 0.82 |
| 11 | Milestone | 0.81 | Medium-High | Low | 0.81 |
| 12 | Continuation | 0.80 | High | High | 0.80 |
| 13 | Causal Context | 0.79 | Medium-High | Medium | 0.79 |
| 14 | Introduction | 0.78 | Medium | High | 0.78 |
| 15 | Delegation | 0.78 | Medium | Medium | 0.78 |
| 16 | Planning | 0.77 | Medium | Medium | 0.77 |
| 17 | Emotional Anchor | 0.76 | Medium | High | 0.76 |
| 18 | Implicit Preference | 0.74 | Medium | Low-Medium | 0.74 |

**Implementation order recommendation:** Start with patterns 1-6 (highest impact, lowest false positive risk), then add 7-12, then the rest. Each batch should be validated against real conversation logs before adding the next.
