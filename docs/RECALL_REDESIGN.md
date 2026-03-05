# Engram Recall Redesign — Synthesis

## The Problem

Engram's ingestion pipeline is deep: autocapture + observe/remember + triage + extraction + consolidation + dream associations. But recall is fundamentally **pull-based** — the AI has to choose to call `recall()` at the right time with the right query. It often doesn't.

### 8 Failure Points Identified (Pipeline Analyst)

| # | Failure | Layer | Severity |
|---|---------|-------|----------|
| 1 | **AI never calls recall** — no proactive trigger | AI Decision | CRITICAL |
| 2 | **Hybrid search misses** — FTS5 prefix-OR has no synonym awareness, seed_threshold=0.3 excludes relevant entities | Candidate Gen | HIGH |
| 3 | **Episodes filtered from MCP output** — `result_type="episode"` silently dropped | Rendering | MEDIUM |
| 4 | **Working memory is in-process, session-scoped** — cold starts have no short-term context, 300s TTL | Cross-cutting | HIGH |
| 5 | **Activation reinforcement loop** — get_context records access for all included entities, entrenching frequently-accessed ones | Scoring | MEDIUM |
| 6 | **Variable resolution too coarse** — hop-2+ entities render as just a name | Rendering | LOW |
| 7 | **search_facts fragile for non-exact names** — FTS5 fallback uses exact string comparison | Structured Query | MEDIUM |
| 8 | **5-relationship cap per entity** — no pagination | Rendering | LOW |

### 3 Missing ACT-R Components (Cognitive Researcher)

Engram implements `A_i = B_i + S_i` but the full ACT-R retrieval equation is `A_i = B_i + S_i + P_i + ε_i`:

- **P_i (Partial Matching)** — Missing. No structured matching on entity attributes/type beyond embedding cosine similarity. Querying "that Python framework" can't partially match Django via `entity_type=Technology + BUILT_WITH→Python`.
- **ε_i (Activation Noise)** — Missing. Retrieval is deterministic. Same query always returns same results. No exploration/serendipity.
- **τ (Retrieval Threshold)** — Missing. Engram returns whatever scores highest regardless of absolute quality. Can return stale, barely-relevant memories with confidence.

---

## The Solution: Four Complementary Systems

The 5 agents converged on a unified architecture with four interlocking systems. Each is independently valuable, but together they close the recall gap completely.

### System 1: AutoRecall (Inline Recall-on-Ingest)

**Core insight**: Every `observe`/`remember` call reveals what the AI is thinking about. Piggyback recall on ingestion.

**How it works**:
1. AI calls `observe("User is debugging their React app")`
2. Extract recall query from content (fast heuristic: proper nouns + first sentence, ~0.5ms)
3. Run lightweight retrieval (FTS5 + activation scoring, skip spreading, ~25-50ms)
4. Filter against working memory (don't echo back what AI already knows)
5. Return results in the tool response as `recalled_context` field

```json
{
  "status": "stored",
  "episode_id": "ep_abc",
  "recalled_context": {
    "entities": [
      {"name": "React", "type": "Technology", "summary": "Frontend framework user is expert in",
       "top_facts": ["User EXPERT_IN React", "Project X BUILT_WITH React"]}
    ]
  }
}
```

**Session priming**: First tool call of a session auto-runs `get_context()` and includes it in the response. No more relying on the AI to remember to call it.

**Throttling**: Max 3/minute, 60s topic cooldown, skip if explicit `recall` was called in last 30s.

**Cognitive basis**: Maps to ACT-R's spreading activation from goal buffer + recognition-triggered recall (familiarity check during ingestion triggers recollection).

### System 2: Proactive Context Injection (3 Layers)

**Layer 1 — Inline Hints** (LOW complexity, HIGH impact):
- Augment observe/remember responses with 0-3 related entity summaries
- Cheap path: semantic search only, no spreading activation
- "Dinner party" principle: share one relevant tidbit, not your life story
- Token budget: 300 tokens max for hints

**Layer 2 — Conversation Tracker** (MEDIUM complexity):
- Evolve `WorkingMemoryBuffer` into `ConversationTracker` that maintains:
  - Entity mention buffer (from ingestion content)
  - Rolling topic embedding (EMA of recent turns)
  - Topic shift detection (cosine distance between windows)
- Topic shift triggers richer proactive recall (5 hints instead of 3)

**Layer 3 — Surprise Detection** (HIGHER complexity):
- After `project_episode` extracts entities, check for surprise connections
- Surprise = strong graph connection + dormant activation ("you just mentioned X, and you told me Y about X 2 weeks ago")
- Runs in background worker, results cached for next tool call

**Cognitive basis**: Maps to priming effects (recently activated memories prime neighbors), encoding specificity (match retrieval context to encoding context), and spontaneous retrieval (context overlap triggers recall without explicit cue).

### System 3: Conversation-Shaped Retrieval

**Problem**: Current recall takes a single query string. Conversations have multiple topics, implicit references, temporal flow, and emotional context.

**Multi-query decomposition** (no LLM, ~2ms):
- Topic query: concatenate last 3 user turns
- Entity query: top-5 session entities by mention weight
- Temporal query: time expressions from recent turns
- Intent query: the actual user query
- Weights shift: early turns are query-dominant (0.6), later turns are context-dominant (0.35 topic + 0.3 entity)

**Conversation fingerprinting**:
- Rolling EMA embedding of recent turns (decay=0.85 per turn)
- Used as additional retrieval vector for re-ranking
- Captures "what this conversation is about" as a dense vector

**Session memory graph**:
- In-session co-occurrence tracking: which entities appeared together in this conversation
- Session entities become additional spreading seeds with energy proportional to session score
- Creates virtuous cycle: retrieval → session tracking → better future retrieval

**Graph-Connected MMR (GC-MMR)**:
- Standard MMR maximizes diversity. GC-MMR adds **coherence**: prefer results that are graph-connected to already-selected results
- `score = λ₁*relevance - λ₂*max_sim(selected) + λ₃*connectivity(selected, graph)`
- Results form a connected narrative, not just individually relevant fragments

**Cognitive basis**: Maps to context-dependent memory (encoding specificity principle), cue-dependent retrieval (multiple cues > single cue), and narrative coherence (memories are organized in stories, not random access).

### System 4: Cognitive Architecture Upgrades

Direct imports from ACT-R that Engram is currently missing:

**Retrieval Threshold (τ)**:
- When no entity exceeds threshold, return empty or "low confidence" flag
- Prevents returning garbage with apparent confidence
- Default: 0.12 on Engram's 0-1 scale (maps to ACT-R's τ=-2.0)

**Near-Miss Detection**:
- Track entities scoring just below threshold or just outside top-N
- Return as `near_misses` with partial info: "I think there's something about X but can't place it"
- Feed near-misses into consolidation pressure (frequent near-misses → strengthen that memory)

**Retrieval Priming Buffer**:
- After each retrieval, temporarily boost 1-hop neighbors of result entities
- Creates "warm zone" that makes follow-up queries on related topics faster
- Decays in seconds (not minutes) to avoid polluting unrelated queries

**Prospective Memory**:
- Store intentions with semantic trigger conditions
- "Tell user about Python 3.12 migration when they mention Python upgrades"
- During ingestion, match content against triggers → surface associated memory
- Maps directly to event-based prospective memory research

**Activation Noise (ε)**:
- Optional logistic noise on final retrieval score
- Makes retrieval probabilistic → enables serendipitous discovery
- Configurable: `retrieval_noise_s` (default 0.0 to preserve current behavior)

---

## Implementation Roadmap

### Wave 1: AutoRecall Core (Highest impact, lowest risk)

**Goal**: Memory surfaces automatically during normal conversation flow.

| Component | Files | Effort |
|-----------|-------|--------|
| `_extract_recall_query()` heuristic | `mcp/server.py` | ~50 LOC |
| `RecallCooldown` throttling | `mcp/server.py` | ~40 LOC |
| `_auto_recall()` piggybacking | `mcp/server.py` | ~60 LOC |
| Session priming on first call | `mcp/server.py` | ~30 LOC |
| Config fields + profile integration | `config.py` | ~20 LOC |
| System prompt update | `mcp/prompts.py` | ~10 LOC |
| Retrieval threshold (τ) | `retrieval/scorer.py` | ~15 LOC |
| Tests | `tests/test_autorecall.py` | ~200 LOC |

**Estimated**: ~425 LOC, 1 session

### Wave 2: Conversation Awareness (Medium impact, medium complexity)

**Goal**: Retrieval understands the conversation, not just individual queries.

| Component | Files | Effort |
|-----------|-------|--------|
| `ConversationContext` data structure | `retrieval/context.py` (new) | ~100 LOC |
| `ConversationFingerprinter` | `retrieval/context.py` | ~50 LOC |
| Multi-query decomposition | `retrieval/pipeline.py` | ~80 LOC |
| Session entity seed injection | `retrieval/pipeline.py` | ~20 LOC |
| Contextual re-ranking | `retrieval/scorer.py` | ~40 LOC |
| Near-miss detection | `retrieval/scorer.py` | ~30 LOC |
| Config fields | `config.py` | ~20 LOC |
| Tests | `tests/test_conversation_retrieval.py` | ~250 LOC |

**Estimated**: ~590 LOC, 1-2 sessions

### Wave 3: Proactive Intelligence (High impact, higher complexity)

**Goal**: System detects when memories are relevant and surfaces them without being asked.

| Component | Files | Effort |
|-----------|-------|--------|
| `ConversationTracker` (evolve WM buffer) | `retrieval/context.py` | ~120 LOC |
| Topic shift detection | `retrieval/context.py` | ~60 LOC |
| Surprise detection | `graph_manager.py` | ~80 LOC |
| Worker integration for surprise | `worker.py` | ~30 LOC |
| Retrieval priming buffer | `retrieval/pipeline.py` | ~40 LOC |
| GC-MMR (coherence-aware diversity) | `retrieval/scorer.py` | ~60 LOC |
| Tests | `tests/test_proactive_recall.py` | ~300 LOC |

**Estimated**: ~690 LOC, 2 sessions

### Wave 4: Prospective Memory (Novel capability)

**Goal**: Engram can hold intentions and trigger them when conditions match.

| Component | Files | Effort |
|-----------|-------|--------|
| Prospective memory model | `models/prospective.py` (new) | ~40 LOC |
| Storage (SQLite table) | `storage/sqlite/graph.py` | ~60 LOC |
| Trigger matching during ingestion | `graph_manager.py` | ~80 LOC |
| MCP tool: `set_intention` | `mcp/server.py` | ~40 LOC |
| Tests | `tests/test_prospective.py` | ~200 LOC |

**Estimated**: ~420 LOC, 1 session

---

## Design Principles

1. **Piggyback, don't poll** — AutoRecall runs during existing tool calls. Zero additional round-trips.
2. **Dinner party rule** — Share one relevant tidbit, not your life story. Hard token caps, aggressive filtering.
3. **Progressive enhancement** — Each layer works independently. Feature flags on everything.
4. **Cognitive fidelity** — Follow ACT-R's architecture where it maps cleanly (retrieval threshold, goal buffer, partial matching, noise).
5. **Conversation over query** — The retrieval unit should be the conversation, not the query string. Multi-query decomposition + fingerprinting + session graphs.
6. **Recognition before recall** — Fast familiarity detection during ingestion triggers targeted recall. Cheaper than blind retrieval.

---

## Key Metrics to Track

- **Recall rate**: % of conversations where relevant memories surface (vs. being silently lost)
- **AutoRecall precision**: % of auto-recalled entities that the AI actually uses in its response
- **Latency delta**: ms added to observe/remember by autorecall (target: <50ms)
- **Near-miss rate**: % of retrievals that produce near-misses (indicates memories needing strengthening)
- **Topic shift accuracy**: % of detected topic shifts that correspond to actual conversation turns
- **Surprise relevance**: % of surprise connections the AI mentions to the user
