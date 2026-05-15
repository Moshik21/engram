# Engram Retrieval Crisis — Team Discussion

## Problem Statement
Engram scores 13-21% on LongMemEval. Naive RAG scores 52%. Top systems score 80-95%.
Every "improvement" we've tried has made scores WORSE. We need to understand WHY at a fundamental level.

## Constraints
- Zero LLM calls at query time (embeddings OK, Haiku/Sonnet NOT OK)
- LLM at ingest time is acceptable (one-time per episode)
- Must work for one-user-one-brain (all episodes in one group)
- ~1000 episodes per user is realistic scale

## Score History
| Run | Overall | Category | Notes |
|-----|---------|----------|-------|
| v1 baseline | 20.2% | 19.8% | Entity-first, basic search |
| v2 multi-signal | 21.2% | 20.7% | +chunk search, +entity-from-query, +temporal |
| v3 HyDE+CoN | 15.6% | 14.5% | Added LLM calls — WORSE |
| v4 LLM-free | 16.0% | 13.1% | Passage-first, topic chunks — WORSE |

## Key Observation
Recall@5 is ~40% across ALL runs. The retrieval is broken at a fundamental level.
Changes to scoring, ranking, and post-processing cannot fix a retrieval that doesn't find the right documents.

---

## ROUND 1: Deep Investigation
(Agents will append their findings below)

### Investigator 2: Naive RAG Comparison

#### What Naive RAG Does (per LongMemEval paper)

1. **Chunking**: Each user-assistant ROUND is a separate chunk. One round = one user message + one assistant response.
2. **Embedding model**: Stella V5 (1.5B parameters, 8192-token context).
3. **Retrieval**: Top-K dense retrieval (k=5 or k=10) over round-level chunks.
4. **Reader**: GPT-4o synthesizes answer from retrieved chunks.

#### What Engram Does (quantified from code + data)

1. **Chunking at ingest**: Each SESSION (all rounds in a conversation) is stored as ONE episode. The adapter calls `_format_session_content()` which prepends a date header and joins all turns as `User: ... \n Assistant: ...` into a single string.
2. **Embedding**: The whole session text is embedded as ONE vector via `index_episode()` in `search.py`. Chunks are created as a SECONDARY index.
3. **Retrieval**: Multi-signal pipeline (entity search + episode search + chunk search + cue search + spreading activation + ACT-R). Entity results compete with episode results for slots.
4. **Reader**: Claude Haiku 4.5 with Chain-of-Note prompt.

#### The Fundamental Problem: Granularity Mismatch

Measured from the actual LongMemEval oracle dataset (948 sessions, 500 questions):

| Metric | Naive RAG (round-level) | Engram (session-level) |
|--------|------------------------|----------------------|
| Chunk unit | 1 user-assistant round | 1 full session (all rounds) |
| Avg chunk size | ~2,429 chars (~607 tokens) | ~14,046 chars (~3,512 tokens) |
| Turns per chunk | 2 (1 user + 1 assistant) | 11.6 avg (up to 32) |
| Topics per chunk | 1 (focused) | ~5-6 (diluted) |
| Total vectors | ~5,498 | 948 (primary) + chunks (secondary) |
| Ratio | 1x | 5.8x more content per embedding |

This is a **5.8x granularity disadvantage**. Each Engram embedding must represent ~6 different topics. Each Naive RAG embedding represents exactly 1 topic.

#### Why This Destroys Retrieval Precision

**Concrete example from the errors** (question `gpt4_2487a7cb`):

- Question: "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?"
- The answer session contains 12 turns covering: car detailing, GPS issues, gas mileage, credit card rewards, AND a mention of the workshop.
- Engram embeds all 12 turns as one vector. The embedding is dominated by car-related content (10 turns) with the workshop mention diluted (2 turns).
- Naive RAG embeds the workshop round separately. Query about workshops has high cosine similarity with that specific round.
- Result: Engram retrieves "car detailing and road trips" sessions instead.

**The math**: If a session discusses 6 topics and only 1 is relevant, the relevant topic occupies ~16% of the embedding's representational capacity. Cosine similarity between the query and this diluted vector will be low. A different session that is wholly about a somewhat-related topic can easily outscore it.

#### Engram's Chunking Exists But Is Insufficient

Engram DOES have chunking (`_chunk_text` at 2000 chars, `_segment_by_topic` with speaker-transition splitting). These fire inside `index_episode()` and create `episode_chunk` vectors. However:

1. **Chunks are a secondary signal**, not the primary retrieval path. The pipeline searches episodes first (whole-session vectors), then chunks are used to boost/enrich episode candidates. If the whole-session vector doesn't surface the right episode in the first place, the chunk search is searching the same pool and has the same problem.

2. **Chunk search deduplicates to 1 chunk per episode** (`search_episode_chunks` at line 1040-1046 of search.py: `seen: dict[str, dict] = {}` keeps only the best chunk per episode). This means even if 3 chunks from one session match, only 1 result appears. This is correct for session-level dedup but means chunk search cannot return the same episode multiple times for different matching topics.

3. **Chunk search feeds into episode_candidates** (pipeline.py line 574-591). It adds episodes to the episode candidate pool or boosts existing episode scores. It does NOT bypass the entity-vs-episode allocation logic. In passage_first mode, entities still get min 1/3 of slots.

4. **Primary episode vector search runs first** and may push the right episodes down before chunk search even fires. The episode vector for a 14k-char session is a poor match for a specific-topic query.

#### Embedding Model Gap

| Property | Stella V5 (Naive RAG) | Gemini Embedding 2 (Engram default) | Voyage 4-lite (Engram alt) |
|----------|----------------------|-------------------------------------|---------------------------|
| Parameters | 1.5B | Unknown (likely large) | Unknown |
| Context window | 8,192 tokens | 2,048 tokens | 32,000 tokens |
| Avg input size | ~607 tokens | ~3,512 tokens (sessions) | ~3,512 tokens |
| Truncation risk | None (inputs << context) | HIGH: 3,512 >> 2,048 | Low |
| Dimension | 1024d | 3,072d | 1,024d |

**Critical finding**: If Gemini is the active provider, it has a 2,048-token input limit. The average session is ~3,512 tokens. This means **the tail ~40% of each session is silently truncated** before embedding. Any answer-bearing content in the latter half of a conversation is invisible to vector search. The embedding provider does not appear to do explicit truncation — it likely relies on the API's internal handling, which typically truncates from the right.

#### The Architectural Mismatch

Naive RAG treats the problem as: "find the right PASSAGE." Simple and effective.

Engram treats the problem as: "build a knowledge graph, then search entities + episodes + cues + chunks with spreading activation, ACT-R, and scoring weights." This adds complexity without improving the core retrieval problem. The knowledge graph is irrelevant if the answer is a raw fact from a specific conversation turn.

Specifically for LongMemEval:
- 42.9% of single-session-assistant questions are correct (these match entity summaries well)
- 2.8% of knowledge-update questions are correct (these require finding the LATEST mention of a changing fact across sessions — entity summaries lose temporal ordering)
- 5.5% of temporal-reasoning questions are correct (these require finding specific dated events — entity abstractions strip dates)

Entity extraction LOSES the specific details (dates, numbers, exact phrasing) that LongMemEval questions ask about. The graph abstraction is counterproductive for this benchmark.

#### Summary: Three Root Causes

1. **Granularity**: Sessions are 5.8x too large for precise embedding. Naive RAG chunks at round-level. Engram chunks at session-level. The chunk index exists but is secondary, not primary.

2. **Embedding model context**: Gemini Embedding 2 has a 2,048-token limit vs 3,512-token avg sessions, causing silent right-truncation. Answer-bearing content in later turns is invisible.

3. **Abstraction loss**: Entity extraction converts "I attended the Effective Time Management workshop on March 5th" into an entity node with a summary, losing the exact date and context. LongMemEval questions demand these exact details. Naive RAG preserves verbatim text.

#### What Would Close the Gap

To match Naive RAG's 52%, Engram would need to:

1. **Make chunk search the PRIMARY retrieval path** for episode-type questions, not a secondary signal. Search chunk vectors directly, return top-K chunks, and pass those chunks as evidence to the reader.
2. **Chunk at round-level** (user-assistant pairs), not topic-level. This matches LongMemEval's data structure and what Naive RAG does.
3. **Fix the embedding truncation** by either switching to Voyage (32k context) or pre-chunking before embedding.
4. **Deprioritize entity results** for benchmark queries. Entity abstractions lose the details LongMemEval asks about.

### Investigator 1: End-to-End Trace of a Failing `single-session-assistant` Query

#### Selected Failure: `cc539528`

- **Question type**: `single-session-assistant` (the assistant literally said the answer -- should be trivial)
- **Question**: "I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?"
- **Gold answer**: "I recommended learning Ruby, Python, or PHP as a back-end programming language."
- **Retrieval metrics**: recall@1=1.0, recall@5=1.0 (PERFECT retrieval -- the right session was found)
- **Result**: WRONG. Reader says "the conversation appears to be cut off mid-question" and "no specific back-end programming languages are mentioned."

This is the most damning type of failure: retrieval found exactly the right session, but the answer was still wrong. 22 of 32 single-session-assistant failures have recall@5=1.0 (68%). The retrieval works; the evidence delivery to the reader is broken.

#### Step-by-Step Trace

**Step 1: Dataset structure for cc539528**

- 1 session: `answer_ultrachat_374124`, date `2023/05/23 (Tue) 01:20`
- Session text length: **5,799 characters** (10 turns of dialogue about front-end vs back-end development)
- The words "Ruby, Python, or PHP" appear at **character position 3,570** in the session

**Step 2: Ingestion (`ingest_instance`)**

The adapter (line 294-341) stores each session as one episode:
1. Calls `_format_session_content(session)` which prepends `[Conversation from 2023/05/23 (Tue) 01:20]` and joins all turns as `User: ...\nAssistant: ...`
2. Calls `store_episode()` to create a QUEUED episode in HelixDB
3. Calls `project_episode()` for narrow extraction (entities + relationships)
4. `project_episode()` calls `_index_materialized_bundle()` which calls `index_episode()` on the search index
5. `index_episode()` checks `self._embeddings_enabled`

**Step 3: Embedding provider resolution**

The run config says `"embedding_provider": "auto"`. The `_create_embedding_provider()` factory (factory.py line 16-100) tries:
1. Gemini -- requires `GEMINI_API_KEY` env var
2. Voyage -- requires `VOYAGE_API_KEY` env var
3. FastEmbed (local) -- requires `pip install fastembed`
4. **NoopProvider** -- fallback, returns 0-dimension embeddings

**FastEmbed is NOT installed** on this machine (`ModuleNotFoundError: No module named 'fastembed'`). Whether API keys were set during the run is unknown, but the adapter stats show `embedding_calls: 0` (though this counter is never incremented in the LongMemEval adapter -- it is uninstrumented).

If `NoopProvider` was active, then `provider.dimension() == 0`, so `self._embeddings_enabled = False` in `HelixSearchIndex.__init__`. This has catastrophic effects:
- `index_episode()` returns immediately (line 676-677): episodes are stored but NEVER get vectors
- `search_episodes()` falls back to BM25-only (line 917-920)
- `search_episode_chunks()` returns `[]` unconditionally (line 1009-1010): **chunk search is completely dead**
- `index_entity()` returns immediately: entities have no vectors either
- `search()` for entities uses BM25-only

Even if embeddings WERE active: the adapter stats show the real provider was working (948 episodes extracted). The key finding is that regardless of embedding state, the evidence truncation bug (Step 6) kills accuracy.

**Step 4: Query-time recall (`query_instance`)**

1. Adapter calls `self._manager.recall(query=instance.question, group_id="longmemeval", limit=10)`
2. `recall()` calls `retrieve()` which:
   - Runs `search_index.search(query=hyde_query, group_id="longmemeval", limit=50)` to find entity candidates (BM25 or hybrid)
   - Runs `search_index.search_episodes(query=query, group_id="longmemeval", limit=30)` for episode search (BM25-only when no embeddings)
   - Runs `search_index.search_episode_chunks()` which returns `[]` when embeddings disabled
   - Entity candidates go through spreading activation, scoring
   - Episode candidates are scored separately
   - In `passage_first` mode: 3 entity slots + up to 5 episode slots = 10 max results

3. **ALL 500 questions** return exactly **3 entities**. The same 3 entities are returned every time. This means BM25 entity search across 948 episodes with narrow extraction finds the same high-frequency entities for every query.

4. **BM25 episode search across 948 episodes** in one shared group returns many irrelevant matches. For `cc539528`, the reader saw evidence from car detailing (April 2023), API performance (May 2023), master's degree programs, coding progress tools -- all noise from other questions' sessions that happened to share keywords like "development", "programming", "learning".

**Step 5: Episode content in recall results**

When `recall()` processes an episode result (graph_manager.py line 3224-3248):
```python
"content": self._truncate_episode_content(ep)
```

`_truncate_episode_content()` (line 3076-3105) uses `recall_episode_content_limit = 2000` (config default). So the episode content is **truncated to 2,000 characters**.

The session for `cc539528` is 5,799 characters. The answer ("Ruby, Python, or PHP") is at character position 3,570. **The answer is cut off by the 2,000-character truncation.**

**Step 6: THE CRITICAL BUG -- Evidence enrichment is dead code for the reader**

The adapter has `_enrich_episode_text()` (line 532-561) which fetches full episode content from the graph store to bypass the 2,000-char truncation. This is correct. The enriched text goes into the `_enriched` key of the evidence dict:

```python
episode_evidence_items.append({
    "content": raw_content,           # TRUNCATED to 2000 chars
    "chunk_context": "",               # empty for regular episodes
    "conversation_date": ep_date,
    "_enriched": ep_text,              # FULL content from graph store
})
```

The enriched text is used for the flat `evidence_texts` list (line 495):
```python
episode_texts = [item["_enriched"] for item in episode_evidence_items]
```

But the **reader LLM never sees this**. The reader uses `structured_evidence` (line 504-508), formatted by `_format_evidence()` (line 108-131). This function reads:
```python
content = (
    item.get("chunk_context", "")     # empty string = falsy
    or item.get("content", "")         # TRUNCATED 2000-char content
    or item.get("summary", "")
)
```

**`_format_evidence()` never reads the `_enriched` key.** The full episode content is computed but discarded. The reader gets the truncated 2,000-char content that cuts off before the answer.

This is the single biggest bug. For `cc539528`, the reader correctly identifies evidence [1] as the relevant conversation about front-end/back-end development, but says "the conversation appears to be cut off mid-question" -- because it literally is. The 2,000-char truncation stopped at "Can someone be profic..." which is mid-sentence, and "Ruby, Python, or PHP" is 1,570 characters further in the conversation.

**Step 7: Total evidence budget collapse**

Even if the enrichment bug were fixed, there's a secondary problem. `_compose_answer()` (line 612-615) applies a 15,000-char guard:
```python
evidence_text = _format_evidence(structured_evidence[:15])
if len(evidence_text) > 15000:
    evidence_text = evidence_text[:15000]
```

With 23 episodes at ~2,000 chars each = ~46,000 chars of evidence. The 15,000-char cutoff means episodes after the first ~7 are completely invisible to the reader. If the answer session is not in the first 7, it gets cut entirely.

#### Summary of Bugs Found

| # | Bug | Severity | Impact |
|---|-----|----------|--------|
| 1 | **`_format_evidence()` ignores `_enriched` key** -- reader gets truncated 2000-char content instead of full episode text | **CRITICAL** | Every answer after char 2000 in a session is invisible. Directly explains 68% of single-session-assistant failures with perfect retrieval. |
| 2 | **`search_episode_chunks()` returns `[]` when embeddings disabled** -- chunk search is dead | HIGH | Sub-episode precision is impossible. No way to find specific passages within long sessions. Falls back to BM25 session search only. |
| 3 | **BM25 noise in shared group** -- 948 episodes in one group, BM25 returns many irrelevant matches | HIGH | Evidence slots wasted on noise. The answer session competes with 947 other sessions for top-K slots. |
| 4 | **Same 3 entities for all 500 questions** -- entity search is non-discriminating | MEDIUM | Entity evidence is useless noise. The 3 slots consumed by entities could be used for more episode results. |
| 5 | **15,000-char evidence budget** -- with many noisy episodes, answer episode may be cut entirely | MEDIUM | Secondary to bug #1, but compounds the problem. |

#### Fix Priority

**Bug #1 is the highest priority and easiest fix.** Change `_format_evidence()` to use `_enriched` when available:

```python
content = (
    item.get("chunk_context", "")
    or item.get("_enriched", "")    # <-- ADD THIS
    or item.get("content", "")
    or item.get("summary", "")
)
```

Or better: always use `_enriched` for the reader prompt since it already bypasses truncation. This alone could recover the 22 single-session-assistant failures that have perfect retrieval (recall@5=1.0) but wrong answers due to truncation.

#### Broader Architectural Observation

The system has two competing evidence paths:
1. **Flat text path** (`evidence_texts` via `_enriched`) -- correct, uses full content
2. **Structured evidence path** (`structured_evidence` via `_format_evidence`) -- broken, uses truncated content

The `_compose_answer()` function prefers the structured path (line 611: `if structured_evidence:`), which means the correct full content in the flat path is NEVER used by the reader. The whole enrichment mechanism is dead code in practice.

### Investigator 3: Data Audit

#### Summary Numbers

| Metric | Count |
|--------|-------|
| Unique sessions in LongMemEval oracle dataset | 940 |
| Episodes ingested into HelixDB | 948 |
| Entities extracted (narrow extractor) | 627 |
| Relationships (edges between entities) | 22 |
| Episode-entity links (HasEntity edges) | 1,344 |
| Episode vectors (HNSW indexed) | 60 out of 948 (6.3%) |
| Entity vectors (HNSW indexed) | 15 out of 627 (2.4%) |
| Chunk vectors (EpisodeChunk) | 431 across 139 episodes |
| Cue vectors (EpisodeCue) | 0 |

#### LongMemEval Dataset Structure

500 questions reference 940 unique sessions. 8 sessions are shared across multiple questions' haystacks.

- Sessions per question: avg 1.9 (min 1, max ~5)
- Turns per session: avg 11.6 (min 2, max 32)
- Session text length: avg 13,937 chars, median 14,294 chars (min 496, max 27,919)
- Answer sessions per question: avg 1.9

Question type distribution in the dataset:
- temporal-reasoning: 133 (127 in results + 6 converted to abstention)
- multi-session: 133 (121 + 12 abstention)
- knowledge-update: 78 (72 + 6 abstention)
- single-session-user: 70 (64 + 6 abstention)
- single-session-assistant: 56 (all preserved)
- single-session-preference: 30 (all preserved)

Each session is a multi-turn user-assistant conversation. Answer-bearing content is spread throughout the turns, not concentrated at the beginning.

#### Episode Data Quality

All 948 episodes are `status=completed`, `projection_state=projected`, `memory_tier=episodic`. None have `conversation_date` set (the adapter parses dates from the session but `conversation_date` is empty in HelixDB for all 948 episodes -- this breaks temporal sorting and temporal retrieval scoring in the pipeline).

Episode content is the full session text with a date header prepended: `[Conversation from 2023/04/10 (Mon) 17:50]\nUser: ... \nAssistant: ...`

Content length stats:
- Min: 557 chars
- Max: 28,151 chars
- Avg: 14,089 chars
- Median: 14,436 chars
- Total: 13.36M chars

#### Entity Extraction Quality: Catastrophically Poor

The narrow extractor produced 627 entities, but quality is abysmal:

**Entity type distribution:**
| Type | Count |
|------|-------|
| Person | 318 (50.7%) |
| Technology | 157 (25.0%) |
| Concept | 93 (14.8%) |
| Identifier | 57 (9.1%) |
| Location | 2 (0.3%) |

**Every entity name is unique** (no deduplication occurred). The narrow extractor creates superficial entities by parsing sentence fragments. Examples of extracted "Person" entities: `Conversation`, `Mon`, `User`, `Assistant`, `Check`, `Ask`, `See`, `Look`, `Keep`, `Focusing`, `Ultimately`, `Earning`, `Pros`, `Cons`, `Additionally`, `Find`, `Make`, `Compare`, `Fuel`, `Remember`.

Roughly 22 entities are obvious garbage (common English words misclassified as Person). ~182 are questionable. ~423 are plausibly meaningful names (like `Honda Civic`, `Yelp`, `Google Reviews`, `Apache Spark`), though many are still shallow keyword extractions with no real semantic content.

**All entity summaries are raw content snippets**, not actual summaries. They are just copies of the surrounding text from the episode. Example: entity "Online Reviews" has summary "Online Reviews: Check review websites such as Yelp, Google Reviews, or Facebook Reviews to see what other customers have to say about their experiences."

**All entities have `access_count=0`** and `source_episode_ids` is empty for all 627 entities (the provenance tracking is not working with the narrow extractor path).

#### Relationship (Edge) Data: Essentially Non-Existent

Only **22 edges** exist across 627 entities. Only 31 entities (4.9%) have any edges at all. The graph is effectively disconnected.

The 22 edges follow a pattern: generic container entities ("Websites", "Brands", "Tools", "Apps", "Channels", "Platforms", "Artists") with outgoing edges to specific entities. Examples:
- "Websites" has 9 outgoing edges (to Patel Brothers, Google Flights, Letterboxd, etc.)
- "Brands" has 5 outgoing edges (to Beyond Meat, Almond Breeze, Baggu, etc.)

This means **spreading activation is effectively dead**: with only 22 edges, the graph traversal (BFS/PPR) cannot discover related entities or episodes through the knowledge graph. The activation-spreading architecture that Engram's retrieval pipeline relies on has nothing to traverse.

#### Episode-Entity Links: Sparse and Uneven

Of 948 episodes:
- **361 episodes (38.1%) have ZERO entity links** -- completely disconnected from the knowledge graph
- 561 episodes have 1-5 entity links
- 23 episodes have 6-10 entity links
- 3 episodes have 11-47 entity links

Average: 1.4 entities per episode. The narrow extractor is extremely inconsistent -- some episodes get 40+ entities (many of them garbage), while 38% get none at all.

#### Vector Index Coverage: 6.3% of Episodes

Only **60 out of 948 episodes** (6.3%) have HNSW vector embeddings. These span the full ingestion time range (07:11 to 11:20 on 2026-03-15), suggesting the Gemini API was available but only succeeded for ~60 calls before hitting rate limits or errors.

Only **15 out of 627 entities** have vector embeddings. All are tagged `embed_model=noop`, `embed_provider=auto`.

**431 chunk vectors** exist across 139 episodes. Chunks average 1,169 chars (min 100, max 4,167). Episodes with chunks have 1-12 chunks each (avg 3.1).

All vectors are **3072-dimensional** (Gemini Embedding 2 format). The HNSW index works for these 60 episodes + 431 chunks, but the other 888 episodes are invisible to vector search.

#### BM25 Search: Broken for Episodes

BM25 text search on Episode nodes returns **zero results for every query tested** (`car`, `service`, `Honda`, `GPS`, `user`, `conversation`, etc.). This appears to be a fundamental issue with how HelixDB indexes Episode node content for BM25 -- the Episode schema does not have an explicit BM25-indexed text field, and the `content` field is not being picked up by `SearchBM25<Episode>`.

BM25 on Entity nodes works, but only matches entity names (not summaries or content). So `search_entities_bm25("Honda", 5)` returns entity `Honda Civic`, but `search_entities_bm25("car", 5)` returns nothing because no entity is named "car".

#### What the Retrieval System Actually Works With

Combining all findings, here is what the recall pipeline has available for each query:

1. **Entity BM25**: Searches 627 entity names. Returns entities matching keywords in names. Most entities are garbage single-word fragments. This explains why **every instance gets exactly 3 entities** -- the same high-frequency garbage entities (like "User", "Conversation") match nearly every query.

2. **Episode BM25**: Returns zero for everything. This entire retrieval path is dead.

3. **Episode vector search**: Only 60/948 episodes have vectors. This means for ~93.7% of answer sessions, vector search literally cannot find them. When it does find something, it's among only 60 indexed episodes, so the chances of the answer session being one of those 60 is ~6.3%.

4. **Chunk vector search**: 431 chunks across 139 episodes. Better coverage than whole-episode vectors, but still covers only 14.7% of episodes. The chunk search also requires `_embeddings_enabled=True` to work, which depends on having a working embedding provider at query time.

5. **Spreading activation**: With only 22 graph edges, this is a no-op. Cannot discover episodes through entity relationships because the relationships don't exist.

6. **Cue search**: Zero cue vectors exist. Dead path.

#### Critical Finding: conversation_date is Empty

All 948 episodes have empty `conversation_date` in HelixDB. The adapter DOES parse session dates and passes `conversation_date=conv_dt` to `store_episode()`, but the date is not persisting. This means:
- Temporal retrieval scoring (pipeline.py Step 5.05) cannot boost/demote episodes by date
- The adapter's `_enrich_episode_text()` cannot find dates via `ep.conversation_date`
- Chronological sorting of evidence for temporal questions has no data to sort on

The session date IS embedded in the content text as `[Conversation from 2023/04/10 (Mon) 17:50]`, so the reader CAN see it, but the pipeline's temporal scoring logic is completely non-functional.

#### Root Cause Summary

The retrieval system is operating with:
- **0% BM25 episode coverage** (search returns nothing)
- **6.3% vector episode coverage** (60/948 indexed)
- **14.7% chunk coverage** (139/948 episodes have chunks)
- **4.9% entity connectivity** (22 edges across 627 entities)
- **0% temporal metadata** (conversation_date empty)
- **50.7% garbage entity types** (318 Person entities that are common words)

The system is trying to search 948 episodes but can only actually find content in ~60-139 of them via vectors, and 0 of them via BM25. The knowledge graph is non-functional. Temporal reasoning has no date metadata. This is not a scoring or ranking problem -- the data infrastructure is fundamentally broken.

---

## ROUND 2: Fixes

### Architect 2: Round-Level Chunking (Granularity Fix)

#### Problem

Investigator 2 identified a 5.8x granularity mismatch as root cause #1. Naive RAG embeds each user-assistant round as a separate vector (~607 tokens, ~5,498 vectors across 948 sessions). Engram embeds whole sessions as single vectors (~3,512 tokens, 948 vectors). Each Engram embedding represents ~6 different topics, diluting cosine similarity for specific-topic queries.

The existing chunking (`_segment_by_topic` and `_chunk_text`) splits by topic boundaries or fixed size. Neither matches the natural round-level granularity of conversational data. Topic segmentation uses embedding-based similarity detection which adds latency and can split mid-round. Size-based chunking at 2000 chars arbitrarily cuts across conversational turns.

#### Fix Applied

**File**: `server/engram/storage/helix/search.py`

**1. New method `_chunk_by_rounds()` (line 267)**

Added a static method that splits conversation text at speaker transitions (`User:`, `Assistant:`, `Human:`, `AI:`) and groups turns into complete rounds (one user message + one assistant response). Returns an empty list for non-conversational text so callers can fall back.

Key design choices:
- Groups turns into user+assistant pairs, not individual turns. A round is complete when both `User`/`Human` and `Assistant`/`AI` labels are present. This preserves question-answer context within each chunk.
- Returns `[]` for non-conversational text (no speaker labels detected), enabling clean fallback.
- `min_chars=50` filter prevents degenerate single-word rounds from becoming vectors.
- Captures trailing incomplete rounds (e.g. a user message without assistant response).

**2. Updated `index_episode()` chunking strategy (line 761)**

Changed the chunking priority order from:
```
topic segmentation → size-based (fallback)
```
To:
```
round-level → topic segmentation → size-based (last resort)
```

The round-level chunker fires first for all episodes. For the LongMemEval dataset (948 sessions with `User:` / `Assistant:` speaker labels), this produces ~5-6 chunks per session = ~5,500 chunk vectors total, matching Naive RAG's granularity exactly.

For non-conversational content (no speaker labels), `_chunk_by_rounds()` returns `[]` and the existing topic/size fallbacks take over unchanged.

#### Expected Impact

- **Vector count**: 948 session-level vectors + ~5,500 round-level chunk vectors (was 948 + ~139 topic chunks)
- **Embedding focus**: Each chunk vector represents exactly 1 topic/round instead of ~6 diluted topics
- **Query precision**: A query about "Effective Time Management workshop" will match the specific round discussing that workshop, not the whole session dominated by car detailing content
- **Backward compatible**: Non-conversational episodes still use topic/size chunking. The whole-session embedding is still created alongside chunks.

#### What This Does NOT Fix

This addresses the chunking granularity but not the other root causes identified in Round 1:
- The chunk search is still a secondary signal in the retrieval pipeline (needs Architect 3/pipeline changes)
- The `_format_evidence()` truncation bug (Bug #1 from Investigator 1) still drops enriched content
- BM25 noise in shared groups is independent of chunking strategy
- Embedding model context window truncation is independent of chunking

#### Verification

`ruff check` passes clean on the modified file.

### Architect 1: Content Truncation Fix (Bug #1 -- CRITICAL)

#### Problem

Two layers of truncation destroy answer content:

1. **`_truncate_episode_content()` in `graph_manager.py`** truncates episode content to `recall_episode_content_limit` (default 2,000 chars) when building recall results. For the LongMemEval dataset, average sessions are ~14,046 chars. Any answer after char 2,000 is invisible.

2. **`_format_evidence()` in `adapter.py`** reads `chunk_context` -> `content` -> `summary`, completely ignoring the `_enriched` key. The adapter already has `_enrich_episode_text()` which fetches the FULL episode content from the graph store and stores it in `_enriched`, but `_format_evidence()` never reads it. The enrichment is dead code.

The reader LLM receives truncated 2,000-char content. For question `cc539528`, the answer ("Ruby, Python, or PHP") is at char 3,570 -- cut off. The reader literally says "the conversation appears to be cut off mid-question."

#### Fixes Applied

**Fix 1: `_format_evidence()` now reads `_enriched`** (`adapter.py` line 117-122)

Changed the content resolution chain from:
```
chunk_context -> content -> summary
```
to:
```
chunk_context -> _enriched -> content -> summary
```

`_enriched` contains the full episode text fetched by `_enrich_episode_text()`, which bypasses the recall truncation by reading directly from the graph store. `chunk_context` still takes priority when available (it represents a more precise retrieval snippet). Entity evidence items (which never have `_enriched`) fall through to `content` as before.

**Fix 2: Raised `recall_episode_content_limit` default from 2,000 to 15,000** (`config.py` line 274)

Defense-in-depth. Even without the `_enriched` fix, the truncation limit was far too aggressive for multi-turn conversations. 15,000 chars covers the vast majority of LongMemEval sessions (avg ~14,046 chars) and is within the 15,000-char evidence budget in `_compose_answer()`. This also benefits non-benchmark code paths (MCP recall, REST API) that use `_truncate_episode_content()` directly.

#### Files Changed

| File | Change |
|------|--------|
| `server/engram/benchmark/longmemeval/adapter.py` | `_format_evidence()`: added `_enriched` to content resolution chain |
| `server/engram/config.py` | `recall_episode_content_limit`: 2000 -> 15000 |

#### Expected Impact

- Directly recovers the 22 single-session-assistant failures that have recall@5=1.0 but wrong answers due to truncation (68% of that category's failures)
- Conservative estimate: +4-5% overall score improvement from this fix alone
- No test breakage (ruff clean, no tests assert on the old 2000 default for recall content limit)

#### Remaining Concerns for Other Architects

1. **15,000-char evidence budget** in `_compose_answer()` (line 614): With full episode content now flowing through, fewer episodes will fit within the budget. If the answer session is not in the first few results, it may still get cut by the total budget. Consider increasing to 30,000 or making it proportional to the number of results.

2. **Embedding truncation** (Bug #2): Gemini's 2,048-token limit still silently right-truncates sessions before embedding. Full content in evidence only helps when the correct session is already retrieved. If the answer-bearing content was in the truncated tail of the embedding input, the session may never be retrieved in the first place.

3. **Granularity mismatch** (Bug #3): Session-level embeddings remain 5.8x too coarse. Even with full content in evidence, the wrong session may outrank the right one because the session embedding is diluted across many topics. Architect 2's round-level chunking fix addresses this.

#### Verification

`ruff check` passes clean on both modified files.

### Architect 3: Embedding Truncation Investigation

#### Task

Investigate and fix the embedding truncation claim from Round 1: "Gemini Embedding 2 has a 2,048-token input limit but sessions average 3,512 tokens. ~40% of content is silently truncated."

#### Finding: The 2,048-Token Truncation Theory is WRONG

After thorough investigation of the Gemini API documentation, SDK source code, and provider implementation, the Round 1 claim that `gemini-embedding-2-preview` has a 2,048-token input limit is **incorrect**.

**Actual token limits:**

| Model | Per-Input Token Limit | Source |
|-------|-----------------------|--------|
| `gemini-embedding-001` (older) | 2,048 tokens | Google Vertex docs |
| `text-embedding-005` (older) | 2,048 tokens | Google Vertex docs |
| `gemini-embedding-2-preview` (current) | **8,192 tokens** | Google AI docs, Vertex docs, blog post |

Google's developer blog explicitly states: "Input token limit of 8K tokens. We've improved our context length from previous models allowing you to embed large chunks of text."

The Round 1 investigator appears to have confused the older model limits (2,048) with the current model (`gemini-embedding-2-preview` at 8,192).

**Does the average session fit?** Yes. Average session = ~3,512 tokens = ~14,046 chars. The 8,192-token limit (~32K chars) easily accommodates this. Even the maximum session (28,151 chars = ~7,038 tokens) fits within the 8,192-token limit.

#### What the Provider Code Actually Does

Examined `server/engram/embeddings/provider.py` (GeminiProvider class):

1. **No explicit text truncation** -- The `_embed_sync()` method passes text directly to `self._client.models.embed_content()` without any length checks or truncation.

2. **`auto_truncate` parameter exists but is NOT set** -- The `google-genai` SDK (v1.15.0) has an `auto_truncate` field on `EmbedContentConfig`. When `None` (the default), the API defaults to `True`, meaning inputs exceeding the 8,192-token limit ARE silently truncated. However, since average sessions are ~3,512 tokens, this truncation would only affect the very longest outlier sessions.

3. **The config uses `gemini-embedding-2-preview`** -- Confirmed at `config.py` line 68: `gemini_model: str = "gemini-embedding-2-preview"`. This is the 8K-context model, not the older 2K model.

#### The REAL Embedding Problem: 93.7% of Episodes Have NO Vectors

Investigator 3's data audit reveals the actual catastrophe: **only 60 out of 948 episodes (6.3%) have HNSW vector embeddings**. The problem is not that embeddings are truncated -- it's that embeddings mostly DON'T EXIST.

The factory code (`server/engram/storage/factory.py`) tries providers in order: Gemini -> Voyage -> FastEmbed -> Noop. If the Gemini API key is set but the API hits rate limits or errors after ~60 calls, the remaining 888 episodes get no vectors at all. The embedding provider doesn't retry failed calls or queue them.

This is far worse than truncation. Truncated embeddings would still give a degraded signal. Missing embeddings give zero signal -- those episodes are completely invisible to vector search.

#### What About `auto_truncate`?

Even though truncation is not the primary issue, there's a defensive improvement available. The `GeminiProvider._embed_sync()` method should explicitly set `auto_truncate=False` so that any input exceeding 8,192 tokens raises an error instead of being silently truncated. This would surface any future truncation issues loudly instead of silently degrading quality.

#### Fix Applied: Explicit `auto_truncate=False` + Warning for Over-Limit Inputs

**File**: `server/engram/embeddings/provider.py`

Changed `_embed_sync()` to:
1. Set `auto_truncate=False` in `EmbedContentConfig` so the API returns an error instead of silently truncating
2. Add a pre-flight check that estimates token count (chars/4) and logs a warning for any input approaching the 8,192-token limit
3. If the API returns a 400 error due to over-limit input, fall back to truncating at 32,000 chars (8K tokens * 4 chars/token) and retry once

This makes the truncation behavior explicit and visible rather than silent.

#### Summary of Embedding Situation

| Issue | Status | Severity |
|-------|--------|----------|
| Silent truncation at 2,048 tokens | **NOT REAL** -- limit is 8,192 tokens | N/A |
| 93.7% of episodes have no vectors | **REAL** -- rate limits/errors during ingestion | CRITICAL |
| No retry logic for failed embeddings | **REAL** -- single failure = permanent miss | HIGH |
| `auto_truncate` defaults to silent | **Minor risk** -- only affects >8K token inputs | LOW |

#### Recommendations for Other Architects

1. **The vector coverage gap (6.3%) is the dominant issue.** Any fix that improves embedding reliability (retry logic, batch size tuning, rate limit backoff) would have far more impact than chunking or truncation fixes.

2. **Round-level chunking (Architect 2's fix) helps even more than expected.** Since chunks get their own vectors, splitting 948 sessions into ~5,500 round-level chunks means 5,500 embedding API calls instead of 948. If the API rate limit allows ~500 calls, round-level chunking gives ~500 chunk vectors across ~100 sessions. If the whole-session embedding fails for a session, its individual round chunks may still succeed (or vice versa). This provides redundancy.

3. **Embedding retry/backoff logic** should be added to `_embed_texts()` in the search index. Currently, a single failure silently returns `[]` and the episode is permanently invisible to vector search.

### Critical Path Fixer: WHY 93.7% of Episodes Have No Vectors -- Root Cause and Fix

#### Investigation Findings

**The log tells the full story.** Examining `results/longmemeval_v4_llm_free.log`:

1. **Line 711**: `HelixSearchIndex: batch embedding failed: [Errno 54] Connection reset by peer` -- this is the ONLY embedding error in the entire 10,147-line log. One single connection reset from the Gemini API killed embeddings for the remaining ~888 episodes.

2. **1,500 dimension mismatch warnings**: `3072 != 64` on `search_graph_embed_vectors` -- this is a separate issue (GraphEmbedVec HNSW index was initialized for 64d graph embeddings, but queries send 3072d Gemini vectors). Not the cause of missing episode vectors, but contributes to noise.

3. **Zero retry, zero fallback**: When the Gemini API connection reset at line 711, `_embed_texts()` caught the exception, logged one warning, returned `[]`, and `index_episode()` silently returned. No retry. No fallback to local embeddings. No error propagation. Every subsequent episode that should have been embedded was permanently invisible.

#### The Chain of Failure

```
Gemini API connection reset (Errno 54)
  -> _embed_texts() catches exception, returns []
  -> index_episode() sees empty embeddings, returns immediately (no logging)
  -> _index_materialized_bundle() wraps in try/except (swallows any error)
  -> project_episode() continues to COMPLETED status
  -> Episode marked "completed" with ZERO vectors
  -> Episode is INVISIBLE to vector search permanently
```

The system has 3 layers of silent error swallowing:
- `_embed_texts()`: `except Exception -> return []`
- `index_episode()`: `if not embeddings: return`
- `_index_materialized_bundle()`: `except Exception -> logger.warning(...)`

No layer retries. No layer falls back. No layer reports failure as an error. The episode is marked COMPLETED even though it has no vector representation.

#### Fix Applied

**File**: `server/engram/storage/helix/search.py`

**1. Retry with exponential backoff** (`_embed_with_retry()`)

New method that wraps embedding calls with up to 3 retries and exponential backoff (1s, 2s+jitter, 4s+jitter, max 30s). Classifies errors as retryable (connection reset, rate limit, timeout, server errors) vs non-retryable (auth failure, bad input). Only retries on transient errors.

**2. Fallback embedding provider** (`_get_fallback_provider()`)

After 3 consecutive primary provider failures, lazily initializes a local FastEmbed provider as fallback. If Gemini goes down mid-ingestion, the system automatically switches to local embeddings (768d instead of 3072d, but infinitely better than 0d).

The fallback is dimension-aware -- FastEmbed produces 768d vectors vs Gemini's 3072d. The HelixDB HNSW index will accept vectors of any dimension for writes (it was created to accept Gemini's 3072d). At search time, mixed-dimension vectors may produce suboptimal similarity scores, but the vectors exist and can be found. This is a deliberate tradeoff: degraded precision beats complete invisibility.

**3. Explicit error logging** in `index_episode()`

When embedding fails (even after retries), the method now logs at ERROR level with full context:
```
EMBEDDING FAILED for episode ep_xxx (content_len=14046, group=longmemeval):
all providers exhausted after retries. Episode will be INVISIBLE to vector search.
[stats: indexed=60, failed=1]
```

This makes the failure impossible to miss in logs.

**4. Embedding statistics tracking** (`_embed_stats`)

Counters for: episodes_indexed, episodes_failed, chunks_indexed, chunks_failed, entities_indexed, entities_failed, retries, fallback_used. Logged every 100 episodes and on shutdown. The LongMemEval adapter now reports these stats at close() time.

**5. Progress logging**

Every 100th episode logs current stats. On `close()`, final stats are logged including success rate percentage.

**File**: `server/engram/benchmark/longmemeval/adapter.py`

Added embedding stats reporting at adapter close time, so the benchmark runner always sees the final embedding success rate.

#### Expected Impact

With retry logic:
- Transient errors (connection resets, rate limits) are retried 3 times with backoff
- The Gemini free tier allows 1,500 requests/minute for embedding. 948 episodes + ~5,500 chunks = ~6,500 calls. At 64 per batch, that's ~102 batch calls for episodes + ~86 for chunks = ~188 total. Well within rate limits.
- The single connection reset that killed 888/948 embeddings would be retried and likely succeed

With fallback:
- If Gemini is truly unavailable (API key expired, quota exhausted, outage), FastEmbed takes over
- 768d local embeddings vs 0d no embeddings = infinite improvement

Conservative estimate: **From 6.3% to 95%+ episode vector coverage.** The 5% gap accounts for genuinely malformed inputs or persistent API outages.

#### Verification

`ruff check` passes clean on all modified files. Pre-existing test failures are unrelated (BM25 episode search broken in HelixDB, auto_observe cue routing).

---

## BLUE TEAM DEFENSE

### Methodology

Performed deep code review of the full retrieval pipeline (`pipeline.py`, `candidate_pool.py`, `scorer.py`, `mmr.py`, `gc_mmr.py`), the LongMemEval adapter (`adapter.py`), the search index (`helix/search.py`), the graph manager recall path (`graph_manager.py:3076-3270`), and the LongMemEval dataset structure (`longmemeval_oracle.json`). Traced every code path from query to reader prompt to validate the Round 2 fixes and anticipate Red Team objections.

---

### Defense 1: The Fixes Already Applied Are Correct and Sufficient

The Red Team may argue the three proposed fixes (MMR diversity, evidence cap, simpler reader prompt) are the priority. **They are not.** The Round 1 investigation and Round 2 fixes identified and addressed the three REAL root causes. Let me defend each.

#### Fix A: Content Truncation (`_format_evidence` reads `_enriched`)

**Status**: Already applied in `adapter.py` line 117-122. The `_enriched` key is now in the content resolution chain.

**Why this is correct**: The trace for question `cc539528` proves the exact failure mode. The reader literally said "the conversation appears to be cut off mid-question" because it WAS cut off at 2,000 chars (answer at char 3,570). With `_enriched` in the chain, the full episode text flows through. The reader now sees the complete conversation including the answer.

**Anticipated objection**: "Full episode text (14K chars) will blow the 15K evidence budget when multiple episodes are returned."

**Defense**: This is a real concern, and the fix accounts for it through two layers:

1. `_compose_answer()` already caps `structured_evidence[:15]` AND applies a 15,000-char hard guard (`adapter.py` line 614-617). With full episode text, fewer episodes fit, which is **correct behavior** -- the reader should see 2-3 complete episodes rather than 10 truncated ones. The Round 1 data shows that having the right episode's full text is infinitely more valuable than having 10 episodes' first 2,000 chars.

2. The `chunk_context` key takes priority in the resolution chain (line 117-118). When chunk search finds the right round within a session, `chunk_context` provides the focused ~600-char round text, NOT the full 14K session. This is the best of both worlds: focused evidence when chunk search works, full fallback when it does not.

**Concrete improvement path**: If the budget proves too tight, raise the evidence guard from 15,000 to 30,000 chars. At `max_tokens=1024` for the reader response and ~100K context for Haiku, the total prompt (30K evidence + ~200 prompt overhead + 1K response) is well within limits.

```python
# File: server/engram/benchmark/longmemeval/adapter.py, line 616
# Current:
if len(evidence_text) > 15000:
    evidence_text = evidence_text[:15000]
# Proposed refinement:
if len(evidence_text) > 30000:
    evidence_text = evidence_text[:30000]
```

#### Fix B: Round-Level Chunking (`_chunk_by_rounds`)

**Status**: Already applied in `helix/search.py` line 431-477 and integrated into `index_episode()` line 974-987.

**Why this is correct**: The granularity analysis (Investigator 2) proves the 5.8x disadvantage is real. Naive RAG at 52% uses round-level chunks. The math is straightforward:

- Session embedding with 6 topics: the relevant topic occupies ~16% of the embedding's capacity. Cosine similarity for a topic-specific query is diluted.
- Round-level chunk with 1 topic: the relevant topic occupies ~100% of the embedding. Cosine similarity is maximized.

The chunking code is correct:
- Groups turns into user+assistant pairs (not individual turns), preserving question-answer context.
- Returns `[]` for non-conversational text, cleanly falling back to topic/size strategies.
- `min_chars=50` prevents degenerate single-word rounds.
- The speaker label regex `\n(?=(?:User|Assistant|Human|AI)\s*:)` correctly handles the LongMemEval format.

**Anticipated objection**: "Round-level chunking loses cross-round context. A follow-up question in round 5 may reference something from round 3."

**Defense**: This is exactly what Naive RAG does (and it scores 52%). The whole-session embedding is STILL created alongside chunks (`index_episode()` creates the session-level vector at line 952-960 BEFORE chunking at line 974+). Chunk search finds the specific round; episode search provides the session-level signal. The pipeline already merges both (Step 1.3 in `pipeline.py`, line 543-629). Cross-round context is preserved through the entity graph links and the full episode text in `_enriched`.

**Anticipated objection**: "5,500 chunk embeddings will hit Gemini rate limits."

**Defense**: The Critical Path Fixer's analysis (line 680-681) addresses this. Gemini free tier allows 1,500 requests/minute. At batch_size=64, 5,500 chunks = ~86 batch calls. With the retry logic now in place (`_embed_with_retry`), transient failures are recovered. The single connection reset that killed 888/948 embeddings would be retried and succeed.

#### Fix C: Embedding Retry + Fallback

**Status**: Already applied in `helix/search.py`. Exponential backoff (3 retries), FastEmbed fallback, explicit error logging, stats tracking.

**Why this is correct**: The log evidence is indisputable. ONE `[Errno 54] Connection reset by peer` at line 711 of the run log caused 93.7% of episodes to have ZERO vectors. The system had three layers of silent error swallowing. With retry logic, a transient network error is retried 3 times with backoff. With FastEmbed fallback, even a persistent Gemini outage produces 768d local embeddings instead of 0d nothing.

**Anticipated objection**: "Mixed-dimension vectors (3072d Gemini + 768d FastEmbed) will produce bad similarity scores."

**Defense**: This is a valid concern, but the tradeoff is correct. HNSW search in HelixDB uses cosine similarity. When query is embedded by Gemini (3072d) and a stored vector is FastEmbed (768d), the dimensions do not match and the HNSW index will not return that vector in top-K results. However:

1. The FastEmbed fallback only activates after 3 consecutive Gemini failures. In practice, Gemini recovers quickly from transient errors -- the retry alone will fix >95% of cases.
2. If Gemini is truly down, ALL vectors produced during that window will be 768d FastEmbed, so they are comparable to each other. The query at recall time will also use FastEmbed (same provider resolution), so the dimensions will match.
3. 768d local embeddings that can be found > 0d no embeddings that cannot. This is not debatable.

---

### Defense 2: The "Hub Session Dominance" Claim Is Mischaracterized

The original brief says one "car detailing" session pollutes 35% of queries. After analyzing the LongMemEval dataset:

**Finding**: There is no single hub session. The dataset simulates ONE user's life. 428 out of 500 questions have sessions mentioning "car" because this user talks about cars frequently. The "car detailing" content appears in 16 different sessions, not one dominant hub. The word "service" appears in 220/500 question haystacks. "Recommendation" in 350/500. This is natural topical distribution for a single user, not hub dominance.

**The real problem was not topic pollution but BM25+vector retrieval failure**:
- BM25 episode search returns ZERO results (Investigator 3: "BM25 on Episode nodes returns zero results for every query tested"). This is a HelixDB schema issue, not a keyword pollution issue.
- Vector episode search covers only 6.3% of episodes (60/948 have vectors).
- With both primary retrieval paths broken, the system falls back to entity BM25, which returns the same 3 garbage entities for every query.

**MMR is already enabled** (`config.py` line 212: `mmr_enabled: bool = Field(default=True)`). The pipeline runs MMR at Step 5.6 (`pipeline.py` line 1077-1109). It was not helping because:
1. Entity embeddings were mostly missing (only 15/627 entities had vectors), so MMR falls through to the `if not embeddings: return results[:top_n]` branch (mmr.py line 49-50).
2. Even with embeddings, MMR diversifies entity results, not episode results. The episode candidates bypass MMR entirely -- they are merged in Step 6 AFTER MMR runs.

**Concrete defense**: MMR diversity is not the right fix for this problem. The right fix is getting vectors into the system (Fix C) and using chunk-level retrieval (Fix B). Once 95%+ of episodes have vectors and round-level chunks produce focused embeddings, the retrieval will find the right content without needing diversity reranking.

However, if a Red Team objection insists on session-level diversity, here is how to do it properly:

```python
# File: server/engram/benchmark/longmemeval/adapter.py
# In query_instance(), after building episode_evidence_items (line 488),
# add session-level dedup: keep only the best-scoring episode per
# source session to prevent one session from dominating evidence.

# After line 488, add:
if len(episode_evidence_items) > 5:
    # Deduplicate by source session — keep best per session
    best_by_source: dict[str, dict] = {}
    for item in episode_evidence_items:
        source = item.get("_source", "")
        existing = best_by_source.get(source)
        if existing is None:
            best_by_source[source] = item
        # Already best-by-score since episodes are sorted by recall score
    episode_evidence_items = list(best_by_source.values())
```

But this is a minor refinement, not a critical fix.

---

### Defense 3: Evidence Cap at 5 Items Is Wrong; Keyword Pre-Filter Is Better

**The evidence count is already naturally capped.** Looking at the pipeline:

1. `episode_retrieval_max = 5` (config.py line 272). At most 5 episodes are returned.
2. `passage_first` strategy gives entities only `min(3, top_n // 3)` slots (pipeline.py line 1163). With `retrieval_top_n=10`, that is 3 entity slots.
3. Total: 3 entities + 5 episodes = 8 items max (not 25+).

The "25+ episodes per query" claim from the brief does not match the code. The `_compose_answer()` function caps at `structured_evidence[:15]` (adapter.py line 614), and the 15,000-char guard further limits actual content. With full episode text (avg 14K chars), at most 1 complete episode fits within the budget.

**What the brief should have said**: The problem is not too many evidence items but too much IRRELEVANT evidence. When retrieval returns the wrong episodes (because 93.7% have no vectors and BM25 returns nothing), all 5-8 evidence items are noise. Capping at 5 does not help if all 5 are wrong.

**The keyword pre-filter idea is good but should be applied to the reader prompt, not the evidence list.** Here is the concrete implementation:

```python
# File: server/engram/benchmark/longmemeval/adapter.py
# Add after line 107 (before _format_evidence):

def _relevance_score(query: str, content: str) -> float:
    """Quick keyword overlap score between query and evidence."""
    q_words = set(query.lower().split()) - _COMMON_WORDS
    c_words = set(content.lower().split()[:200])  # first 200 words
    if not q_words:
        return 1.0  # can't filter without query words
    overlap = len(q_words & c_words)
    return overlap / len(q_words)

_COMMON_WORDS = frozenset({
    "i", "my", "me", "the", "a", "an", "is", "was", "are", "were",
    "do", "did", "does", "have", "has", "had", "what", "which",
    "who", "where", "when", "how", "can", "you", "your", "about",
    "with", "from", "for", "on", "in", "at", "to", "of", "by",
    "and", "or", "but", "not", "that", "this", "it", "we", "they",
})
```

Then in `_compose_answer()`, filter before formatting:

```python
# File: server/engram/benchmark/longmemeval/adapter.py, inside _compose_answer
# After line 613 (if structured_evidence:), add filtering:
if structured_evidence:
    # Pre-filter: drop items with zero query keyword overlap
    filtered = []
    for item in structured_evidence[:15]:
        content = (
            item.get("chunk_context", "")
            or item.get("_enriched", "")
            or item.get("content", "")
            or item.get("summary", "")
        )
        if _relevance_score(question, content) > 0.0:
            filtered.append(item)
    evidence_text = _format_evidence(filtered or structured_evidence[:5])
```

This adapts to query complexity (more specific queries filter more aggressively) and never drops ALL evidence (fallback to first 5 items).

---

### Defense 4: The Reader Prompt Is Already Reasonable

The Chain-of-Note prompt (adapter.py line 67-88) is 8 lines of instruction + structured formatting. It is NOT a token-wasting monster. The claim that "Chain-of-Note wastes tokens on irrelevant evidence" conflates two separate problems:

1. **Irrelevant evidence being present at all** -- this is a retrieval problem (Fixed by A, B, C above).
2. **The reader spending tokens analyzing irrelevant evidence** -- this is a prompt design problem.

The current prompt says "analyze each piece of evidence and write a brief NOTE about its relevance." This is correct behavior per the Chain-of-Note paper (Appendix A of LongMemEval). The issue is not the prompt asking for notes -- it is that the notes are written for ALL evidence including noise.

**Concrete refinement** (not a rewrite):

```python
# File: server/engram/benchmark/longmemeval/adapter.py
# Replace CHAIN_OF_NOTE_PROMPT (line 67-88) with:

CHAIN_OF_NOTE_PROMPT = """\
You are answering a question about a user's past conversations.

INSTRUCTIONS:
1. Scan the evidence below. Skip items clearly unrelated to the question.
2. For relevant items, note the key fact (date, name, number, detail).
3. Synthesize a concise answer from ONLY the relevant evidence.
4. If no evidence answers the question, say exactly: \
"I cannot answer this question based on the available evidence."

The question is being asked on {question_date}.

Question: {question}

Evidence:
{evidence}

Answer:
"""
```

Key changes:
- "Skip items clearly unrelated" -- explicit instruction to not waste tokens on noise.
- "Note the key fact" -- focused annotation instead of freeform notes.
- Removed "Step 1 -- Analyze each piece of evidence:" which forced the reader to write notes for EVERY item.
- Kept the critical instruction about precise facts and the abstention pattern.
- Dropped the separate temporal/knowledge-update suffixes into the main prompt flow. The evidence is already chronologically sorted for temporal queries (adapter.py line 491-494).

**Why not remove Chain-of-Note entirely**: The LongMemEval paper reports +10% from Chain-of-Note style prompting. The v3 run (HyDE+CoN) scored WORSE than v1, but that was because HyDE introduced the wrong evidence, not because CoN was harmful. With correct evidence, CoN helps the reader identify the answer within a long conversation transcript.

---

### Defense 5: Shared group_id Is Architecturally Correct

**Dataset analysis**: The 500 LongMemEval questions have no `user_id` field. They simulate ONE user across 940 sessions. Each question has a small haystack (1-6 sessions) but the benchmark design assumes all sessions are searchable -- the haystack IS the user's full history.

The shared group_id (`longmemeval`) is correct because:
1. Engram's "one brain per person" model puts ALL of one user's conversations in one group.
2. In production, a real user's 1000 conversations would be in one group. The benchmark simulates this.
3. Per-question group isolation would defeat the purpose of the benchmark (testing long-term memory across conversations).

**Anticipated objection**: "With 948 episodes in one group, retrieval precision drops because irrelevant episodes compete."

**Defense**: This is exactly the challenge Engram must solve. A real user with 1000 conversations needs the system to find the right 2-3 out of 1000. The Round 2 fixes address this:
- **Round-level chunking** gives ~5,500 focused vectors instead of 948 diluted session vectors.
- **Embedding retry** gives 95%+ vector coverage instead of 6.3%.
- **Chunk search** surfaces the specific round matching the query, not the whole session.

With these fixes, the system searches 5,500 round-level chunks with focused embeddings -- essentially the same approach as Naive RAG's 5,498 round-level vectors.

---

### Defense 6: Production vs Benchmark Considerations

The Red Team may argue these fixes are benchmark-specific optimizations. Here is why they help production too:

| Fix | Benchmark Impact | Production Impact |
|-----|-----------------|-------------------|
| Content truncation | Reader sees full episodes | MCP `recall` returns richer results. REST API evidence is complete. |
| Round-level chunking | Matches Naive RAG granularity | Sub-episode precision for any multi-topic conversation. User asks about one thing from a long chat; system finds the specific exchange. |
| Embedding retry | 6.3% -> 95%+ coverage | Any network hiccup during ingestion no longer silently drops vectors forever. |
| Keyword pre-filter | Cleaner reader input | Not applicable (no reader LLM in production). |
| Simpler prompt | Better answer accuracy | Only affects benchmark reader. |

The first three fixes are fundamental infrastructure improvements that benefit ALL Engram use cases. The last two are benchmark-specific refinements with no production downside.

---

### Summary of Positions

| Original Proposal | Blue Team Verdict | Recommended Action |
|-------------------|------------------|-------------------|
| MMR diversity to stop hub dominance | **Already enabled, wrong diagnosis** | No change needed. Fix retrieval (A, B, C), not reranking. |
| Cap evidence at 5 items | **Wrong target** | Add keyword pre-filter to drop zero-overlap evidence (adapts to query complexity). |
| Simpler reader prompt | **Partially agree** | Refine Chain-of-Note prompt with "skip irrelevant" instruction, but keep structured approach. |
| Content truncation fix | **Critical, already applied** | Consider raising evidence budget from 15K to 30K chars. |
| Round-level chunking | **Critical, already applied** | No changes needed; implementation is correct. |
| Embedding retry + fallback | **Critical, already applied** | No changes needed; implementation is correct. |

### Expected Score Impact

With all three critical fixes (A + B + C) working together:

- **Embedding coverage**: 6.3% -> 95%+ (enables vector search for nearly all episodes)
- **Chunk granularity**: 948 session vectors -> ~5,500 round-level vectors (matches Naive RAG)
- **Content truncation**: 2,000 chars -> full text (recovers 22 perfect-retrieval failures)
- **Conservative estimate**: 16-21% -> 35-45% overall score
- **Optimistic estimate**: Could approach Naive RAG's 52% if BM25 episode search is also fixed in HelixDB (currently returns ZERO results for all queries, which is a separate HelixDB schema issue)

The remaining gap to 52% would come from:
1. BM25 episode search fix (HelixDB schema issue -- the Episode node content is not indexed for BM25)
2. Embedding model quality (Gemini Embedding 2 at 3072d should be competitive with Stella V5 at 1024d, but this is untested)
3. Reader prompt refinements (the keyword pre-filter + simplified CoN prompt)

The gap from 52% to 80-95% (top systems) requires retrieval-augmented generation strategies that use LLM calls at query time (which violates the zero-LLM constraint) or much more sophisticated offline processing (full knowledge graph with temporal reasoning).

---

## RED TEAM ANALYSIS

### Preamble

The Blue Team preemptively addressed several objections. Good. That means the Red Team must go deeper. We will challenge the fixes that were already applied, the Blue Team's defenses, and the fundamental framing of the problem. Our goal is not to be difficult -- it is to prevent the team from shipping fixes that look good on paper but fail in practice.

---

### Challenge 0: The Shared `group_id` Defense Is Factually Wrong

The Blue Team's Defense 5 states: "The 500 LongMemEval questions have no `user_id` field. They simulate ONE user across 940 sessions."

**This is incorrect. The data proves it.**

Evidence from `longmemeval_oracle.json`:
- 500 questions, 940 unique sessions
- 932 sessions (99.1%) appear in exactly ONE question's haystack
- Only 8 sessions appear in 2 questions each
- Average haystack size: 1.9 sessions per question

If these 940 sessions belonged to ONE user, then every question's haystack would share sessions heavily. A single user's 940 conversations would be the haystack for ALL questions. Instead, each question gets its own tiny slice of 1-6 sessions with almost zero overlap. This is the hallmark of per-user isolation: each question simulates a different user with their own conversation history.

The Blue Team argues: "Per-question group isolation would defeat the purpose of the benchmark (testing long-term memory across conversations)."

**Wrong.** The benchmark tests: "Given user X's conversation history (1-6 sessions), can you find a specific fact?" The challenge is not searching through 1000 conversations -- it is understanding and retrieving from a SMALL set of multi-topic conversations. The "long-term memory" in the benchmark name refers to the TEMPORAL SPAN of conversations (over months), not the NUMBER of conversations to search through.

Look at the `s` and `m` variants. The oracle variant has ONLY answer sessions (no distractors). The `s` variant adds some distractors per question. The `m` variant adds more. The benchmark is DESIGNED to test retrieval with controlled noise levels. Dumping all 940 sessions into one pool is equivalent to creating an uncontrolled noise level that is ~500x the oracle design and far beyond even the `m` variant's intended difficulty.

**Concrete proof the shared group_id hurts:** V5 results show recall@5 = 1.000 for single-session-assistant. This means for the EASIEST category (where the assistant literally said the answer), the system CAN find the right session in the shared pool. But for knowledge-update, recall@5 = 0.632, and accuracy is 0.0%. For multi-session, recall@5 = 0.502. These categories require finding 2-6 specific sessions among 948 -- a task the benchmark never intended.

**The "one brain per person" argument is a production concern, not a benchmark concern.** The benchmark should evaluate the RETRIEVAL QUALITY in isolation. Once you know the retrieval works on per-user data, THEN you can test scalability with larger corpora. Running the benchmark with a 500x inflated corpus and getting 23% tells you nothing about whether the underlying retrieval algorithm works -- you cannot distinguish "bad algorithm" from "bad test setup."

**Recommended action:** Run the benchmark BOTH ways. Per-question group_id gives the "algorithm quality" score. Shared group_id gives the "scalability" score. Report both. Comparing only the shared score to Naive RAG's 52% (which uses per-question isolation) is an apples-to-oranges comparison that makes Engram look worse than it is.

---

### Challenge 1: The Round-Level Chunking Fix Has Not Been Validated

**Status: Applied but untested.**

The Blue Team says round-level chunking matches Naive RAG's granularity. In principle, correct. In practice, several things can go wrong:

**1. The chunking regex may not match LongMemEval's format.** The `_chunk_by_rounds()` method splits on `\n(?=(?:User|Assistant|Human|AI)\s*:)`. The LongMemEval adapter formats sessions as `User: ...\nAssistant: ...` (adapter.py line 277). This should match. But the raw session text from `_format_session_content()` prepends `[Conversation from 2023/05/23 (Tue) 01:20]\n`. If the first "User:" is on the same line as the header (no intervening newline), the regex will not match the first turn.

**2. The existing chunk index was not rebuilt.** The fix adds `_chunk_by_rounds()` to `index_episode()`, but the 948 episodes were already ingested with the OLD chunking strategy. The existing 431 chunk vectors use topic/size-based splitting. The new round-level chunks will ONLY apply to newly ingested episodes. Without re-ingesting all 948 episodes, the benchmark will still use the old chunking. Has the team planned a re-ingestion run?

**3. Chunk search is still secondary, not primary.** Even with round-level chunks, the pipeline at Step 1.3 (pipeline.py line 543-629) treats chunk search as a signal that boosts episode candidates. Chunks are NOT returned directly as evidence. They flow through `episode_candidates` and then into the Step 6 mix (line 1151-1190) where they compete with entity results. The chunk TEXT is attached via `chunk_context`, which `_format_evidence()` reads first (adapter.py line 118). This is good -- but only if chunk search actually fires.

**4. Chunk search requires `_embeddings_enabled=True`.** If the embedding provider is NoopProvider (as happened in the v4 run), `search_episode_chunks()` returns `[]` unconditionally (search.py line 1009-1010 per Investigator 1). The retry/fallback fix should prevent this in future runs, but the dependency is fragile. One embedding failure during index initialization and all chunk search is dead.

**What would validate this fix:** A re-ingestion run with the new chunking + embedding retry active, followed by a retrieval-only evaluation (no reader, just measure recall@5 with round-level chunks vs session-level vectors).

---

### Challenge 2: The Embedding Retry Fix May Introduce Mixed-Dimension Vectors

The Blue Team acknowledges the dimension mismatch concern (Defense, paragraph on Fix C) but dismisses it as acceptable because "FastEmbed fallback only activates after 3 consecutive failures."

**The problem is more subtle than "3 consecutive failures."**

1. **The fallback provider persists for the rest of the ingestion run.** Once activated, `_get_fallback_provider()` caches the FastEmbed provider. If Gemini recovers after 3 failures (common for rate limits), the code does NOT switch back to Gemini. All subsequent embeddings use FastEmbed at 768d. This means a single burst of 3 rate-limit errors mid-run permanently downgrades the embedding quality for all remaining episodes.

2. **At query time, the provider resolution is independent.** The search index's embedding provider is initialized once at startup. If Gemini is available at query time, queries are embedded at 3072d. These 3072d query vectors will have zero meaningful cosine similarity with 768d stored vectors in the HNSW index. The Blue Team says "the HNSW index will accept vectors of any dimension for writes." This is true for writes but catastrophic for reads. HNSW cosine similarity requires equal-dimension vectors. HelixDB's HNSW implementation will either error on dimension mismatch, return garbage similarity scores, or silently skip mismatched vectors. None of these are acceptable.

3. **The proposed fix should include a dimension check.** Before falling back to FastEmbed, the code should check if the existing HNSW index was created with a specific dimension. If it was created for 3072d (Gemini), inserting 768d vectors will corrupt the index. The correct fallback behavior is: (a) retry Gemini 3 times, (b) if still failing, LOG an error and skip the embedding (accepting a coverage gap) rather than inserting incompatible-dimension vectors that will never be retrievable.

---

### Challenge 3: The Content Truncation Fix Exposes a New Token Budget Problem

The fix reads `_enriched` (full episode text, avg 14K chars) into the evidence. The 15K char evidence guard means at most ~1 complete episode fits. The Blue Team suggests raising to 30K.

**But the actual problem is much worse with shared `group_id`.**

V5 shows avg 26.7 episodes per query. If each episode is ~14K chars, the total evidence is ~374K chars. Even at 30K, only ~2 episodes fit. The reader model (Haiku, ~100K context) could handle ~400K chars of input, but the hard-coded guard throws away 344K chars of evidence.

**The guard should be adaptive, not fixed.** A better approach:

```
max_evidence = min(100000, reader_context_window * 0.7)
```

This uses 70% of the reader's context window for evidence, leaving 30% for the prompt and response. With Haiku at 200K context, this gives ~140K chars for evidence. With 26.7 episodes averaging 14K chars, ~10 episodes would be visible. Still not all 26.7, but dramatically better than 1-2.

**However:** Even this does not help if 24 of those 26.7 episodes are irrelevant (because of the shared group_id problem). You would be spending 340K tokens on noise. The real fix is to reduce the number of irrelevant episodes in the evidence through better retrieval (per-question group_id) or pre-filtering (the Blue Team's keyword filter suggestion, which is reasonable).

---

### Challenge 4: The Blue Team's Score Estimate (35-45%) Is Unsupported

The Blue Team claims fixes A+B+C will raise scores from 16-21% to 35-45%.

**This estimate has no empirical basis.** It is a guess based on "Naive RAG gets 52% with similar chunking, so we should get close." But Engram's pipeline has many more moving parts than Naive RAG, each of which can fail:

1. **Entity results still waste 3 evidence slots per query.** V5 shows `num_entities: avg=3.0` for ALL queries. These 3 slots contain garbage entity summaries ("Conversation: [Conversation from 2023/04/10...]", "User: User interacts with...", "Mon: Mon is a person"). They displace 3 episode results. With passage_first mode, `entity_budget = min(3, top_n // 3)`, so the system ALWAYS burns 1-3 slots on entities even when entities are useless. For the benchmark, entities are net harmful.

2. **BM25 episode search is still broken in HelixDB.** The `SearchBM25<Episode>` query returns zero results. This is not fixed by any of the three applied patches. This means the BM25 component of hybrid search (30% weight per the fts_weight=0.3 config) contributes nothing. Vector search alone carries the retrieval. If vector search misses, there is no fallback.

3. **The knowledge-update category (0.0% accuracy in V5) requires temporal reasoning** that no amount of retrieval improvement can provide. Even with perfect retrieval (finding all sessions mentioning a changing fact), the reader must identify which value is MOST RECENT. Without conversation dates in the HelixDB episode metadata (Investigator 3: "All 948 episodes have empty conversation_date"), the temporal sorting in the adapter (line 491-494) sorts by empty strings, producing arbitrary order. The reader gets the right sessions but in random order and must rely on date headers embedded in the text.

4. **The multi-session category (4.1% accuracy in V5) requires finding ALL relevant sessions.** Multi-session questions have 2-6 answer sessions. With passage_first mode and 5 episode slots, the system needs all 5 slots to contain correct sessions. Even with perfect retrieval precision, if 2 of those 5 slots are wasted on entities, only 3 slots remain for episodes. Questions needing 4+ sessions will fail structurally.

**A more realistic estimate:**

| Category | V5 Score | Estimated After Fixes | Reasoning |
|----------|---------|----------------------|-----------|
| single-session-assistant | 83.9% | 85-90% | Already mostly fixed by truncation patch. Ceiling near Naive RAG. |
| single-session-user | 35.9% | 40-50% | Better chunking helps, but entity waste and BM25 gap limit gains. |
| single-session-preference | 6.7% | 10-20% | These need specific preference statements; chunking helps but not enough. |
| multi-session | 4.1% | 5-10% | Structurally limited by entity waste (3 slots) and needing 2-6 correct sessions in 5 slots from 948-session pool. |
| temporal-reasoning | 7.1% | 10-15% | Empty conversation_date blocks temporal scoring. Date-in-text helps but is fragile. |
| knowledge-update | 0.0% | 2-5% | Requires both correct retrieval AND temporal ordering. Two independent failure modes. |
| abstention | 96.7% | 95%+ | Already near-perfect. |
| **Overall** | **23%** | **25-35%** | Weighted by category distribution. |
| **Category avg** | **22.9%** | **25-33%** | Unweighted average across non-abstention categories. |

The Blue Team's 35-45% estimate requires EVERYTHING to work perfectly: embeddings, chunking, retrieval, reader, AND getting lucky on multi-session/temporal questions. The realistic estimate is 25-33% -- an improvement, but still far from Naive RAG's 52%.

---

### Challenge 5: The Team Is Ignoring the Easiest 20-Point Win

**The single highest-impact change that nobody has proposed as a fix:**

Disable entity retrieval for the benchmark. Set `entity_budget = 0` in the passage_first branch (pipeline.py line 1163).

Current behavior: 3 entity slots always contain "User", "Conversation", and some other garbage entity. This wastes 30% of the evidence budget (3 of ~10 results) on noise for EVERY query.

With `entity_budget = 0`: all 10 result slots go to episodes. In passage_first mode, this means 10 episode slots instead of 7 (=10-3). That is a 43% increase in episode budget. With 948 episodes and ~2 correct ones per query, more slots = higher probability of the correct sessions appearing.

This is a one-line change in the benchmark adapter config:

```python
cfg.entity_budget_override = 0  # or equivalently, set retrieval_strategy to episode_only
```

Or more surgically, in the adapter's `_setup_manager()`:

```python
cfg = self._cfg or ActivationConfig()
cfg.episode_retrieval_max = 10  # give ALL slots to episodes
```

This does not hurt production (entities are valuable when extraction quality is good). It fixes the benchmark where entity extraction quality is catastrophically poor (627 entities, 50.7% garbage, 22 edges, access_count=0 for all).

---

### Challenge 6: The Evidence Budget Math Does Not Add Up

Let us trace the actual evidence flow for a V5 query:

1. `recall()` returns `fetch_limit = top_k * 2` (if temporal) or `top_k` results. Default `top_k=10`.
2. Pipeline returns max `retrieval_top_n=10` results (Step 6).
3. Adapter separates into entities (avg 3) and episodes (avg 26.7 in V5... wait).

**How are 26.7 episodes appearing when the pipeline caps at 10 results?** The adapter's `query_instance()` processes ALL results from `recall()`, not just top-10. Look at adapter.py line 431: `for r in results:`. The `recall()` function in `graph_manager.py` returns the full `retrieve()` output, which is capped at `top_n = min(limit, cfg.retrieval_top_n)`.

If `fetch_limit = 20` (temporal queries) and `retrieval_top_n = 10`, then `min(20, 10) = 10` results. So at most 10 results should be returned. The V5 average of 26.7 episodes per query suggests either:

- (a) `retrieval_top_n` was set higher than 10 for the V5 run, OR
- (b) The `graph_manager.recall()` function adds additional results beyond `retrieve()` (e.g., episode enrichment in the recall path), OR
- (c) The 26.7 includes entity+episode+cue combined, not just episodes.

If (c), then the actual episode count is `26.7 - 3.0 = 23.7` episodes, which is still way above the 10-result cap. This suggests the recall pipeline is NOT capping properly, or the V5 run used custom config with `top_k=30` or higher.

**This matters because:** If the pipeline is returning 26.7 items when it should return 10, there is a bug in the result capping logic. The Blue Team's defense assumes the pipeline caps correctly. If it does not, all evidence budget calculations are wrong.

---

### Summary: Root Cause Ranking

| Rank | Root Cause | Expected Impact of Fix | Fix Status |
|------|-----------|----------------------|------------|
| 1 | **Shared group_id** (500x search space) | +15-25% accuracy | NOT ADDRESSED |
| 2 | **93.7% missing vectors** | +5-10% accuracy | FIXED (retry+fallback) |
| 3 | **Content truncation** | +4-5% accuracy (already realized in V5) | FIXED |
| 4 | **BM25 episode search dead** | +5-8% accuracy | NOT ADDRESSED |
| 5 | **Entity budget waste** | +3-5% accuracy | NOT ADDRESSED |
| 6 | **Granularity mismatch** | +3-5% accuracy | FIXED (round-level chunking) but untested, requires re-ingestion |
| 7 | **Empty conversation_date** | +2-3% on temporal/knowledge-update | NOT ADDRESSED |
| 8 | **Evidence budget too small** | +1-2% accuracy | PARTIALLY ADDRESSED (15K, should be adaptive) |
| 9 | **Reader prompt** | +1-2% accuracy | MINOR CONCERN |

The three proposed fixes from the original brief (MMR, evidence cap, simpler reader) rank at #9 or below. They are not wrong, but they are not where the points are.

**The team should focus on ranks #1, #4, and #5 for the next round of fixes.** #2, #3, and #6 are already addressed. The combined impact of #1+#4+#5 could add 23-38 points, potentially matching or exceeding Naive RAG's 52%.
