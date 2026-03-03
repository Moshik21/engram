# 03 - Async Ingestion Pipeline

## Problem Statement

The original spec describes a synchronous ingestion path:

```
remember() -> Claude API extraction -> entity resolution -> graph writes -> activation updates
```

This is fragile and latency-blocking. A single `remember` call requires:
- Claude API call for extraction (~1-3s, variable, can timeout)
- Entity deduplication with fuzzy matching + embedding similarity (~200ms)
- Graph writes to FalkorDB (~50ms per entity/edge)
- Activation state updates in Redis (~10ms)
- Embedding generation for new entities (~500ms)

**Total synchronous latency: 2-5 seconds per episode.** The MCP client (Claude Desktop / Claude Code) blocks during this entire time, degrading the user experience.

Additionally, Claude API failures (rate limits, timeouts, 500s) would bubble up as MCP tool errors, with no retry mechanism.

---

## Design: Async Pipeline with Immediate Acknowledgment

### Core Principle

The `remember` MCP tool returns **immediately** (~10ms) with an `episode_id`. All heavy processing happens asynchronously in a background worker pipeline. Clients can poll status or subscribe via WebSocket for completion notifications.

---

## Sequence Diagram

```
 MCP Client          MCP Server            Redis Stream        Worker Pool          Claude API       FalkorDB/Redis     Voyage AI
 (Claude)            (FastAPI)             (Queue)             (Background)         (Extraction)     (Storage)          (Embeddings)
    |                    |                    |                    |                    |                |                   |
    |  remember(text)    |                    |                    |                    |                |                   |
    |------------------->|                    |                    |                    |                |                   |
    |                    | generate episode_id|                    |                    |                |                   |
    |                    | store episode stub |                    |                    |                |                   |
    |                    |-------------------------------------->  |                    |                |                   |
    |                    | XADD episode_stream|                    |                    |                |                   |
    |                    |------------------->|                    |                    |                |                   |
    |  {episode_id,      |                    |                    |                    |                |                   |
    |   status:"queued"} |                    |                    |                    |                |                   |
    |<-------------------|                    |                    |                    |                |                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    | XREADGROUP         |                    |                |                   |
    |                    |                    |------------------->|                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "extracting"       |                |                   |
    |                    |                    |                    |------------------->|                |                   |
    |                    |                    |                    | Claude extraction  |                |                   |
    |                    |                    |                    |<-------------------|                |                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "resolving"        |                |                   |
    |                    |                    |                    | entity resolution  |                |                   |
    |                    |                    |                    |---------------------------------------->|                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "writing"          |                |                   |
    |                    |                    |                    | graph writes       |                |                   |
    |                    |                    |                    | (track changed     |                |                   |
    |                    |                    |                    |  entity summaries) |                |                   |
    |                    |                    |                    |---------------------------------------->|                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "embedding"        |                |                   |
    |                    |                    |                    | embed_ingestion_   |                |                   |
    |                    |                    |                    | batch(): episode + |                |                   |
    |                    |                    |                    | changed entities   |                |                   |
    |                    |                    |                    |---------------------------------------------------------->|
    |                    |                    |                    | store vectors      |                |                   |
    |                    |                    |                    |---------------------------------------->|                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "activating"       |                |                   |
    |                    |                    |                    | activation updates |                |                   |
    |                    |                    |                    |---------------------------------------->|                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |                    | set status:        |                |                   |
    |                    |                    |                    | "completed"        |                |                   |
    |                    |                    |                    |                    |                |                   |
    |                    |                    |  WebSocket push: episode_completed     |                |                   |
    |                    |<---------------------------------------|                    |                |                   |
    |                    |                    | XACK               |                    |                |                   |
    |                    |                    |<-------------------|                    |                |                   |
```

**Note on pipeline ordering:** Embedding (step 4) now runs **before** activation (step 5), aligned with the embedding strategy in `04_embedding_strategy.md`. This ensures vectors are available for retrieval before activation state is updated, since activation updates may trigger spreading activation queries that benefit from having the new entity embeddings indexed.

---

## Episode Lifecycle State Machine

```
                         +-----------+
                         |  QUEUED   |  <-- remember() creates episode, enqueues
                         +-----+-----+
                               |
                       worker picks up
                               |
                         +-----v-----+
                    +--->| EXTRACTING |  <-- Claude API call for entity/relationship extraction
                    |    +-----+-----+
                    |          |
                    |    success / failure
                    |          |
                    |    +-----v-----+
                    |    | RESOLVING  |  <-- Entity dedup, conflict detection, merge
                    |    +-----+-----+
                    |          |
                    |    +-----v-----+
                    |    |  WRITING   |  <-- Graph writes to FalkorDB (tracks changed entity summaries)
                    |    +-----+-----+
                    |          |
                    |    +-----v-----+
                    |    | EMBEDDING  |  <-- embed_ingestion_batch(): episode + changed entities
                    |    +-----+-----+     (via Voyage AI / configured provider)
                    |          |
                    |    embedding fails?──> queue to pending_embeddings, skip to ACTIVATING
                    |          |
                    |    +-----v-----+
                    |    | ACTIVATING |  <-- Activation state updates in Redis
                    |    +-----+-----+
                    |          |
                    |    +-----v-----+
                    |    | COMPLETED  |  <-- Terminal success state
                    |    +-----------+
                    |
                retry (exponential backoff)
                    |
              +-----+-----+
              |  RETRYING  |  <-- Transient failure, will re-enter EXTRACTING
              +-----+-----+
                    |
              max retries exceeded
                    |
              +-----v------+
              | DEAD_LETTER |  <-- Terminal failure state, needs manual intervention
              +-----------+
```

**Pipeline step ordering (aligned with `04_embedding_strategy.md`):**

```
Step 1: Store raw episode in FalkorDB (via WRITING stage)
Step 2: Claude entity extraction (EXTRACTING stage)
Step 3: Entity resolution + graph operations (RESOLVING + WRITING stages)
Step 4: Embed episode content (EMBEDDING stage)
Step 5: Embed new/updated entities (EMBEDDING stage, only entities whose summaries changed)
Step 6: Update activation state (ACTIVATING stage)
Step 7: WebSocket notify (on COMPLETED)
```

Steps 4 and 5 are batched into a single Voyage AI API call via `embed_ingestion_batch()` -- episode content and all changed entity summaries go in one request. If the embedding API fails, the pipeline **continues** (skips to step 6). Failed embeddings are queued in `engram:{group_id}:pending_embeddings` and retried by a background task every 60 seconds.

### State Descriptions

| State | Description | Redis Key | TTL |
|-------|-------------|-----------|-----|
| `queued` | Episode accepted, waiting for worker pickup | `episode:{id}:status` | 24h |
| `extracting` | Claude API call in progress | `episode:{id}:status` | 24h |
| `resolving` | Entity dedup and conflict resolution in progress | `episode:{id}:status` | 24h |
| `writing` | Graph mutations being committed to FalkorDB; tracks which entity summaries changed | `episode:{id}:status` | 24h |
| `embedding` | Generating vector embeddings via `embed_ingestion_batch()` (episode content + changed entity summaries in one API call to Voyage AI) | `episode:{id}:status` | 24h |
| `activating` | Activation state being updated in Redis | `episode:{id}:status` | 24h |
| `completed` | All processing finished successfully | `episode:{id}:status` | 7d |
| `retrying` | Transient failure, scheduled for retry | `episode:{id}:status` | 24h |
| `dead_letter` | Permanently failed after max retries | `episode:{id}:status` | 30d |

### State Transition Rules

1. Only forward transitions allowed (no going backwards except via retry).
2. `retrying` always re-enters at `extracting` (idempotent from start of pipeline).
3. `completed` and `dead_letter` are terminal states.
4. Each transition updates `episode:{id}:status` and `episode:{id}:updated_at` in Redis.
5. Transitions to `completed` or `dead_letter` trigger WebSocket notification.

---

## Queue Architecture: Redis Streams

### Why Redis Streams (not Celery, not asyncio.Queue)

- **Already in the stack.** Redis is used for activation state; no new dependency.
- **Consumer groups.** Multiple workers can process in parallel with exactly-once delivery.
- **Persistence.** Survives server restarts. asyncio.Queue does not.
- **Backpressure.** XREADGROUP with COUNT limits prevents worker overload.
- **Visibility.** XPENDING shows stuck messages. XINFO shows queue depth. Built-in observability.
- **Simpler than Celery.** No broker config, no result backend, no extra processes.

### Stream Layout

```
Stream:   engram:episode_stream
Group:    engram_workers
Consumer: worker-{instance_id}

Message payload:
{
    "episode_id": "ep_01HXYZ...",
    "group_id": "grp_default",
    "content": "<raw episode text>",
    "source": "claude_desktop",
    "metadata": "{\"conversation_id\": \"...\"}",
    "created_at": "2026-02-27T10:30:00Z",
    "priority": "normal"     // "normal" | "high" | "batch"
}
```

### Dead Letter Stream

```
Stream:   engram:dead_letter_stream

Message payload:
{
    "episode_id": "ep_01HXYZ...",
    "original_message_id": "1234567890-0",
    "error": "Claude API: 429 rate limited after 3 retries",
    "retry_count": 3,
    "last_attempt_at": "2026-02-27T10:35:00Z",
    "group_id": "grp_default"
}
```

---

## Error Handling Matrix

| Stage | Error Type | Example | Retry? | Max Retries | Backoff | Recovery |
|-------|-----------|---------|--------|-------------|---------|----------|
| **Extracting** | Rate limit (429) | Claude API rate limited | Yes | 5 | Exponential: 2s, 4s, 8s, 16s, 32s | Back off, retry |
| **Extracting** | Timeout | Claude API >30s | Yes | 3 | Exponential: 5s, 10s, 20s | Retry with shorter content if >4K tokens |
| **Extracting** | Server error (500/502/503) | Claude API outage | Yes | 5 | Exponential: 5s, 10s, 20s, 40s, 80s | Retry, then dead-letter |
| **Extracting** | Auth error (401) | Invalid API key | No | 0 | N/A | Dead-letter immediately, alert admin |
| **Extracting** | Malformed response | Claude returns invalid JSON | Yes | 2 | Fixed: 1s | Retry with stricter prompt, then dead-letter |
| **Resolving** | Duplicate conflict | Ambiguous entity merge | No | 0 | N/A | Accept both, flag for manual review |
| **Resolving** | Embedding service down | Cannot compute similarity | Yes | 3 | Exponential: 2s, 4s, 8s | Skip dedup (accept as new entity), flag |
| **Writing** | FalkorDB connection lost | TCP reset | Yes | 5 | Exponential: 1s, 2s, 4s, 8s, 16s | Reconnect and retry |
| **Writing** | FalkorDB write error | Constraint violation | No | 0 | N/A | Dead-letter with graph state dump |
| **Embedding** | Voyage AI rate limit (429) | Free tier 300 RPM | Yes | 3 | Exponential: 2s, 4s, 8s | Retry; if exhausted, skip to activating + queue to `pending_embeddings` |
| **Embedding** | Embedding provider down | API timeout/500 | No (skip) | 0 | N/A | Skip to activating stage; queue episode + entity IDs to `engram:{group_id}:pending_embeddings`; background task retries every 60s |
| **Embedding** | Local model OOM | nomic-embed-text on CPU | No (skip) | 0 | N/A | Same skip-and-queue behavior |
| **Activating** | Redis connection lost | TCP reset | Yes | 5 | Exponential: 1s, 2s, 4s, 8s, 16s | Reconnect and retry |
| **Any** | Worker crash | OOM, segfault | Yes | Auto | Auto (pending message reclaim) | XPENDING reclaim after visibility timeout |

### Retry Strategy: Exponential Backoff with Jitter

```
delay = min(base_delay * (2 ^ attempt) + random_jitter(0, 1s), max_delay)

Defaults:
  base_delay  = 2 seconds
  max_delay   = 120 seconds
  max_retries = 5 (configurable per error type)
  jitter      = uniform random [0, 1000ms]
```

### Dead Letter Handling

Episodes that exhaust retries move to `engram:dead_letter_stream` with:
- Original episode payload
- Error trace (last error message + type)
- Retry count and timestamps of each attempt
- Worker ID that last attempted processing

**Recovery options:**
1. **Manual re-queue:** Admin API endpoint `POST /admin/episodes/{id}/requeue` moves from dead letter back to main stream.
2. **Bulk retry:** Admin endpoint `POST /admin/dead-letter/retry-all` re-queues all dead-lettered episodes.
3. **Dashboard visibility:** Dead-lettered episodes appear in Memory Feed with error badge. User can click to re-queue or discard.

---

## Data Models

### Episode Status (Redis Hash)

```
Key: episode:{episode_id}:status

Fields:
  status:       "queued" | "extracting" | "resolving" | "writing" | "activating" | "embedding" | "completed" | "retrying" | "dead_letter"
  created_at:   ISO 8601 timestamp
  updated_at:   ISO 8601 timestamp
  group_id:     tenant group ID
  source:       "claude_desktop" | "claude_code" | "api" | ...
  retry_count:  integer (0 on first attempt)
  error:        last error message (empty string if none)
  worker_id:    ID of worker currently processing
  entities:     JSON array of extracted entity IDs (populated after resolving stage)
  facts:        integer count of extracted facts (populated after writing stage)
```

### Episode Record (FalkorDB - persisted after writing stage)

```cypher
(:Episode {
    episode_id: string,        // ULID - sortable, unique
    content: string,
    source: string,
    group_id: string,
    created_at: datetime,
    entity_count: int,
    relationship_count: int,
    processing_duration_ms: int
})
```

---

## `remember` Tool Response Shape

The `remember` MCP tool returns immediately with minimal data:

```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "status": "queued",
    "message": "Memory received and queued for processing.",
    "status_url": "/api/episodes/ep_01HXYZ9ABCDEF1234567890/status"
}
```

This is the contract the MCP server exposes to Claude clients. The `episode_id` is a ULID (Universally Unique Lexicographically Sortable Identifier) providing both uniqueness and temporal ordering.

---

## Status Endpoint

### `GET /api/episodes/{episode_id}/status`

Returns current processing status for an episode.

**Response (in-progress):**
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "status": "resolving",
    "created_at": "2026-02-27T10:30:00Z",
    "updated_at": "2026-02-27T10:30:02Z",
    "retry_count": 0,
    "error": null,
    "entities": null,
    "facts": null
}
```

**Response (completed):**
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "status": "completed",
    "created_at": "2026-02-27T10:30:00Z",
    "updated_at": "2026-02-27T10:30:04Z",
    "retry_count": 0,
    "error": null,
    "entities": ["ent_abc123", "ent_def456"],
    "facts": 3,
    "processing_duration_ms": 4012
}
```

**Response (dead-lettered):**
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "status": "dead_letter",
    "created_at": "2026-02-27T10:30:00Z",
    "updated_at": "2026-02-27T10:35:00Z",
    "retry_count": 5,
    "error": "Claude API: 429 Too Many Requests - rate limit exceeded",
    "entities": null,
    "facts": null
}
```

---

## WebSocket Events for Ingestion Status

> **Canonical reference:** `07_frontend_architecture.md` Section 4 defines the full WebSocket protocol,
> including framing, reconnection, and client commands. This section documents the **worker's publishing
> contract** -- which events the ingestion pipeline emits and in what order.

### Connection

```
ws://localhost:8080/ws/dashboard
```

Unified WebSocket endpoint for all dashboard events. Auth middleware extracts `group_id` from the token (or uses the default group when auth is disabled). See `07_frontend_architecture.md` Section 4.1.

### Framing Envelope

All events are wrapped in the standard envelope with a monotonically increasing `seq` number per connection (managed by the `WebSocketManager`, not the worker):

```json
{
    "seq": 7,
    "type": "graph.nodes_added",
    "timestamp": "2026-02-27T10:30:03Z",
    "payload": { ... }
}
```

The worker calls `ws.publish(group_id, type, payload)` and the `WebSocketManager` handles `seq` assignment and envelope wrapping.

### Event Types Emitted by Ingestion Pipeline

#### Episode Lifecycle Events

**`episode.queued`** -- emitted by `handle_remember()` after enqueuing:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "source": "claude_desktop",
    "created_at": "2026-02-27T10:30:00Z"
}
```

**`episode.status_changed`** -- emitted on every pipeline stage transition:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "previous_status": "extracting",
    "status": "resolving",
    "updated_at": "2026-02-27T10:30:02Z"
}
```

**`episode.completed`** -- emitted as the terminal success event:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "entities": [
        {"id": "ent_abc123", "name": "ReadyCheck", "type": "Project"},
        {"id": "ent_def456", "name": "Stripe", "type": "Tool"}
    ],
    "relationships": [
        {"subject": "User", "predicate": "WORKS_ON", "object": "ReadyCheck"},
        {"subject": "ReadyCheck", "predicate": "INTEGRATES", "object": "Stripe"}
    ],
    "facts_count": 3,
    "processing_duration_ms": 4012,
    "updated_at": "2026-02-27T10:30:04Z"
}
```

**`episode.failed`** -- emitted when episode is dead-lettered:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "error": "Claude API: 429 Too Many Requests",
    "retry_count": 5,
    "updated_at": "2026-02-27T10:35:00Z"
}
```

#### Granular Graph Mutation Events

These replace the previous aggregate `graph.updated` event. They enable surgical Zustand store updates in the dashboard without full graph re-fetches.

**`graph.nodes_added`** -- emitted during WRITING stage for new entities:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "nodes": [
        {
            "id": "ent_new789",
            "name": "retatrutide",
            "entity_type": "Concept",
            "summary": "GLP-1/GIP/glucagon tri-agonist peptide",
            "activation_base": 0.1,
            "activation_current": 0.45,
            "access_count": 1,
            "last_accessed": "2026-02-27T10:30:03Z",
            "created_at": "2026-02-27T10:30:03Z",
            "updated_at": "2026-02-27T10:30:03Z"
        }
    ]
}
```

**`graph.edges_added`** -- emitted during WRITING stage for new relationships:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "edges": [
        {
            "id": "rel_new001",
            "source": "ent_user",
            "target": "ent_new789",
            "predicate": "RESEARCHING",
            "weight": 0.6,
            "valid_from": "2026-02-27T10:30:03Z",
            "valid_to": null,
            "created_at": "2026-02-27T10:30:03Z"
        }
    ]
}
```

**`graph.edges_invalidated`** -- emitted during WRITING stage when old facts are superseded:
```json
{
    "episode_id": "ep_01HXYZ9ABCDEF1234567890",
    "edge_ids": ["rel_old002"],
    "valid_to": "2026-02-27T10:30:03Z"
}
```

**`graph.nodes_updated`** -- emitted during WRITING stage when entity summaries change:
```json
{
    "nodes": [
        {
            "id": "ent_abc123",
            "summary": "Meeting prep SaaS product with Stripe integration. 8-week launch target.",
            "updated_at": "2026-02-27T10:30:03Z"
        }
    ]
}
```

**`activation.updated`** -- emitted during ACTIVATING stage after activation state changes:
```json
{
    "changes": [
        {"entity_id": "ent_abc123", "name": "ReadyCheck", "activation": 0.87},
        {"entity_id": "ent_def456", "name": "Stripe", "activation": 0.72},
        {"entity_id": "ent_new789", "name": "retatrutide", "activation": 0.45}
    ],
    "trigger": "ingestion",
    "source_episode_id": "ep_01HXYZ9ABCDEF1234567890"
}
```

### Required Event Publishing Order

The worker **must** publish events in this order to keep the dashboard Zustand store consistent (see `07_frontend_architecture.md` Section 4.5):

```
 1. episode.queued              (remember() called)
 2. episode.status_changed      (extracting)
 3. episode.status_changed      (resolving)
 4. episode.status_changed      (writing)
 5. graph.nodes_added           (new entities from write, if any)
 6. graph.edges_added           (new relationships from write, if any)
 7. graph.edges_invalidated     (superseded facts, if any)
 8. graph.nodes_updated         (changed entity summaries, if any)
 9. episode.status_changed      (embedding)
10. episode.status_changed      (activating)
11. activation.updated          (activation state changes)
12. episode.completed           (terminal success)
```

Not all events fire for every episode. Events 5-8 only fire if the corresponding graph mutations occurred during the write stage. Event 11 only fires if activation changes are non-empty.

---

## Batch Extraction Support

The spec risk table mentions batch extraction to reduce Claude API costs. The pipeline supports this via a batch coalescing mechanism.

### How Batching Works

1. When multiple episodes arrive within a short window (~2 seconds), the worker can coalesce them into a single Claude extraction call.
2. A batch coordinator checks the queue depth before each extraction.
3. If >1 message is pending AND all are from the same `group_id`, coalesce up to 5 episodes per batch call.

### Batch Flow

```
                  Queue depth check
                        |
            +-----------+-----------+
            |                       |
       depth == 1              depth > 1
            |                       |
    single extraction       coalesce up to 5
            |                  same group_id
            |                       |
    normal pipeline          batch extraction
                                    |
                          split results back to
                          individual episodes
                                    |
                          each episode continues
                          through resolve -> write -> activate -> embed
```

### Batch Extraction Prompt Adjustment

For batch calls, the Claude extraction prompt wraps multiple episodes:

```
Extract entities and relationships from each of the following episodes.
Return results keyed by episode_id.

Episode ep_01A: "I've been working on ReadyCheck..."
Episode ep_01B: "Had a meeting with Sarah about the launch..."
Episode ep_01C: "Switched from Jest to Vitest for testing..."
```

Claude returns structured output keyed by episode ID. The worker splits results and feeds each episode individually through the remainder of the pipeline (resolve, write, activate, embed).

### Cost Savings

- Single episodes: 1 Claude API call per episode.
- Batched: 1 Claude API call per 5 episodes = ~80% reduction in API calls during burst ingestion.
- Haiku tier: Episodes under 500 tokens (simple statements like "I like coffee") route to Claude Haiku instead of Sonnet/Opus, further reducing cost.

### Model Routing Logic

```python
def select_extraction_model(content: str) -> str:
    token_count = estimate_tokens(content)
    if token_count < 500:
        return "claude-haiku-4-5-20251001"  # Simple extraction
    else:
        return "claude-sonnet-4-6"          # Complex extraction
```

---

## Worker Architecture

### Worker Pool

```
                  ┌─────────────────────────────────┐
                  │         Worker Manager           │
                  │  (asyncio event loop, main.py)   │
                  └───────────┬─────────────────────┘
                              |
              ┌───────────────┼───────────────┐
              |               |               |
        ┌─────v─────┐  ┌─────v─────┐  ┌─────v─────┐
        │  Worker 1  │  │  Worker 2  │  │  Worker 3  │
        │  (asyncio  │  │  (asyncio  │  │  (asyncio  │
        │   task)    │  │   task)    │  │   task)    │
        └───────────┘  └───────────┘  └───────────┘
              |               |               |
              └───────────────┼───────────────┘
                              |
                    Redis Stream Consumer Group
                    (XREADGROUP, COUNT=1 per worker)
```

Workers run as asyncio tasks within the same FastAPI process. No separate worker process needed for the initial release. Each worker:

1. Calls `XREADGROUP GROUP engram_workers worker-{n} COUNT 1 BLOCK 5000 STREAMS engram:episode_stream >`
2. Processes one episode through the full pipeline.
3. Calls `XACK` on success.
4. On failure: increments retry count, re-adds to stream with delay (via a delayed re-queue mechanism), or moves to dead letter.

### Concurrency Configuration

```yaml
# config.yaml
ingestion:
  worker_count: 3              # Number of concurrent workers
  batch_window_ms: 2000        # Wait time for batch coalescing
  batch_max_size: 5            # Max episodes per batch extraction
  stream_name: "engram:episode_stream"
  consumer_group: "engram_workers"
  dead_letter_stream: "engram:dead_letter_stream"

retry:
  max_retries: 5
  base_delay_seconds: 2
  max_delay_seconds: 120
  jitter_max_ms: 1000

extraction:
  timeout_seconds: 30
  haiku_token_threshold: 500
  default_model: "claude-sonnet-4-6"
  haiku_model: "claude-haiku-4-5-20251001"
```

### Scaling Path

- **v1 (Week 1-4):** In-process asyncio workers. 3 concurrent workers handle typical personal use (a few episodes per minute).
- **v2 (Post-launch):** Separate worker process for horizontal scaling. Same Redis Stream consumer group, just more consumers.
- **v3 (Hosted SaaS):** Worker pool per tenant with rate limiting. Kubernetes job workers pulling from per-tenant streams.

---

## Python Pseudocode

### `remember` MCP Tool Handler

```python
from ulid import ULID
import redis.asyncio as redis
import json
from datetime import datetime, timezone

async def handle_remember(
    content: str,
    source: str,
    group_id: str,
    metadata: dict | None = None,
    redis_client: redis.Redis = Depends(get_redis),
) -> dict:
    """MCP remember tool - returns immediately with episode_id."""
    episode_id = f"ep_{ULID()}"
    now = datetime.now(timezone.utc).isoformat()

    # Store episode status in Redis hash
    await redis_client.hset(
        f"episode:{episode_id}:status",
        mapping={
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "group_id": group_id,
            "source": source,
            "retry_count": "0",
            "error": "",
            "worker_id": "",
            "entities": "[]",
            "facts": "0",
        },
    )
    await redis_client.expire(f"episode:{episode_id}:status", 86400)  # 24h TTL

    # Enqueue to Redis Stream
    await redis_client.xadd(
        "engram:episode_stream",
        {
            "episode_id": episode_id,
            "group_id": group_id,
            "content": content,
            "source": source,
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
            "priority": "normal",
        },
    )

    # Notify WebSocket subscribers
    await publish_ws_event(group_id, "episode.queued", {
        "episode_id": episode_id,
        "source": source,
        "created_at": now,
    })

    return {
        "episode_id": episode_id,
        "status": "queued",
        "message": "Memory received and queued for processing.",
        "status_url": f"/api/episodes/{episode_id}/status",
    }
```

### Worker: Pipeline Processor

```python
import asyncio
import traceback
from enum import Enum

class EpisodeStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    WRITING = "writing"
    EMBEDDING = "embedding"      # Before activation (see 04_embedding_strategy.md)
    ACTIVATING = "activating"
    COMPLETED = "completed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class IngestionWorker:
    def __init__(
        self,
        worker_id: str,
        redis_client: redis.Redis,
        extractor: ClaudeExtractor,
        resolver: EntityResolver,
        graph_store: GraphStore,
        activation_engine: ActivationEngine,
        embedding_service: EmbeddingService,
        ws_manager: WebSocketManager,
        config: IngestionConfig,
    ):
        self.worker_id = worker_id
        self.redis = redis_client
        self.extractor = extractor
        self.resolver = resolver
        self.graph_store = graph_store
        self.activation = activation_engine
        self.embeddings = embedding_service
        self.ws = ws_manager
        self.config = config

    async def run(self):
        """Main worker loop. Reads from stream, processes episodes."""
        # Ensure consumer group exists
        try:
            await self.redis.xgroup_create(
                self.config.stream_name,
                self.config.consumer_group,
                id="0",
                mkstream=True,
            )
        except redis.ResponseError:
            pass  # Group already exists

        while True:
            messages = await self.redis.xreadgroup(
                groupname=self.config.consumer_group,
                consumername=self.worker_id,
                streams={self.config.stream_name: ">"},
                count=1,
                block=5000,  # Block 5s waiting for messages
            )

            if not messages:
                continue

            for stream_name, entries in messages:
                for message_id, data in entries:
                    await self._process_message(message_id, data)

    async def _process_message(self, message_id: str, data: dict):
        """Process a single episode through the full pipeline."""
        episode_id = data["episode_id"]
        group_id = data["group_id"]
        content = data["content"]
        source = data["source"]
        start_time = asyncio.get_event_loop().time()

        try:
            # Stage 1: Extraction
            await self._set_status(episode_id, EpisodeStatus.EXTRACTING)
            extraction_result = await self.extractor.extract(
                content=content,
                source=source,
                model=self._select_model(content),
            )

            # Stage 2: Entity Resolution
            await self._set_status(episode_id, EpisodeStatus.RESOLVING)
            resolved = await self.resolver.resolve(
                entities=extraction_result.entities,
                relationships=extraction_result.relationships,
                group_id=group_id,
            )

            # Stage 3: Graph Writes
            # write_episode returns a WriteResult with:
            #   - changed_entities: dict[str, str] (entity_id -> new summary)
            #   - new_nodes: list[GraphNode] (full node objects for newly created entities)
            #   - updated_nodes: list[dict] (partial updates for existing entities)
            #   - new_edges: list[GraphEdge] (full edge objects for new relationships)
            #   - invalidated_edge_ids: list[str] (edges whose valid_to was set)
            await self._set_status(episode_id, EpisodeStatus.WRITING)
            write_result = await self.graph_store.write_episode(
                episode_id=episode_id,
                entities=resolved.entities,
                relationships=resolved.relationships,
                facts=extraction_result.facts,
                group_id=group_id,
                source=source,
            )
            changed_entity_summaries: dict[str, str] = write_result.changed_entities

            # Publish granular graph mutation events (order matters -- see 07_frontend_architecture.md 4.5)
            # Events 5-8 in the ingestion sequence. Each is only emitted if there is data.
            now = datetime.now(timezone.utc).isoformat()

            if write_result.new_nodes:
                await self.ws.publish(group_id, "graph.nodes_added", {
                    "episode_id": episode_id,
                    "nodes": [n.to_ws_dict() for n in write_result.new_nodes],
                })

            if write_result.new_edges:
                await self.ws.publish(group_id, "graph.edges_added", {
                    "episode_id": episode_id,
                    "edges": [e.to_ws_dict() for e in write_result.new_edges],
                })

            if write_result.invalidated_edge_ids:
                await self.ws.publish(group_id, "graph.edges_invalidated", {
                    "episode_id": episode_id,
                    "edge_ids": write_result.invalidated_edge_ids,
                    "valid_to": now,
                })

            if write_result.updated_nodes:
                await self.ws.publish(group_id, "graph.nodes_updated", {
                    "nodes": write_result.updated_nodes,
                })

            # Stage 4: Embedding Generation (before activation -- see 04_embedding_strategy.md)
            # Uses embed_ingestion_batch(): episode content + changed entity summaries
            # go in a single API call to the configured embedding provider (default: Voyage AI).
            # If embedding fails, we skip to activation and queue for background retry.
            await self._set_status(episode_id, EpisodeStatus.EMBEDDING)
            try:
                if changed_entity_summaries or True:  # Always embed episode content
                    vectors = await self.embeddings.embed_ingestion_batch(
                        episode_content=content,
                        entity_summaries=changed_entity_summaries,
                    )
                    await self.embeddings.store_vectors(
                        vectors=vectors,
                        episode_id=episode_id,
                        group_id=group_id,
                    )
            except Exception as embed_err:
                # Embedding failure is non-fatal: skip to activation, queue for retry.
                # Episode + entity IDs are added to a Redis set for background retry (every 60s).
                await self.redis.sadd(
                    f"engram:{group_id}:pending_embeddings",
                    json.dumps({
                        "episode_id": episode_id,
                        "episode_content": content,
                        "entity_summaries": changed_entity_summaries,
                    }),
                )
                await self._set_status(
                    episode_id, EpisodeStatus.EMBEDDING,
                    extra={"embedding_error": str(embed_err)},
                )

            # Stage 5: Activation Updates
            await self._set_status(episode_id, EpisodeStatus.ACTIVATING)
            activation_changes = await self.activation.update_on_ingestion(
                entity_ids=[e.id for e in resolved.entities],
                group_id=group_id,
            )

            # Publish activation.updated (event 11 in ingestion sequence)
            if activation_changes:
                await self.ws.publish(group_id, "activation.updated", {
                    "changes": [
                        {"entity_id": eid, "name": n, "activation": a}
                        for eid, n, a in activation_changes
                    ],
                    "trigger": "ingestion",
                    "source_episode_id": episode_id,
                })

            # Mark completed
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            entity_ids = [e.id for e in resolved.entities]
            await self._set_status(
                episode_id,
                EpisodeStatus.COMPLETED,
                extra={
                    "entities": json.dumps(entity_ids),
                    "facts": str(len(extraction_result.facts)),
                },
            )

            # ACK the message
            await self.redis.xack(
                self.config.stream_name,
                self.config.consumer_group,
                message_id,
            )

            # Publish episode.completed (event 12 -- terminal, must be last)
            await self.ws.publish(group_id, "episode.completed", {
                "episode_id": episode_id,
                "entities": [
                    {"id": e.id, "name": e.name, "type": e.entity_type}
                    for e in resolved.entities
                ],
                "relationships": [
                    {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                    for r in resolved.relationships
                ],
                "facts_count": len(extraction_result.facts),
                "processing_duration_ms": duration_ms,
                "updated_at": now,
            })

        except RetryableError as e:
            await self._handle_retry(episode_id, message_id, data, e)
        except NonRetryableError as e:
            await self._send_to_dead_letter(episode_id, message_id, data, e)
        except Exception as e:
            # Unexpected errors are retryable by default
            await self._handle_retry(
                episode_id, message_id, data,
                RetryableError(f"Unexpected: {e}", original=e),
            )

    async def _handle_retry(
        self,
        episode_id: str,
        message_id: str,
        data: dict,
        error: RetryableError,
    ):
        """Handle retryable failure with exponential backoff."""
        retry_count = int(
            await self.redis.hget(f"episode:{episode_id}:status", "retry_count") or 0
        )
        retry_count += 1

        max_retries = error.max_retries or self.config.max_retries

        if retry_count > max_retries:
            await self._send_to_dead_letter(episode_id, message_id, data, error)
            return

        # Calculate backoff delay with jitter
        delay = min(
            self.config.base_delay * (2 ** (retry_count - 1))
            + random.uniform(0, self.config.jitter_max_ms / 1000),
            self.config.max_delay,
        )

        await self._set_status(
            episode_id,
            EpisodeStatus.RETRYING,
            extra={
                "retry_count": str(retry_count),
                "error": str(error),
            },
        )

        # ACK current message (we will re-enqueue)
        await self.redis.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message_id,
        )

        # Re-enqueue after delay
        await asyncio.sleep(delay)
        await self.redis.xadd(self.config.stream_name, data)

    async def _send_to_dead_letter(
        self,
        episode_id: str,
        message_id: str,
        data: dict,
        error: Exception,
    ):
        """Move failed episode to dead letter stream."""
        retry_count = int(
            await self.redis.hget(f"episode:{episode_id}:status", "retry_count") or 0
        )

        await self.redis.xadd(
            self.config.dead_letter_stream,
            {
                "episode_id": episode_id,
                "original_message_id": message_id,
                "error": str(error),
                "error_type": type(error).__name__,
                "retry_count": str(retry_count),
                "last_attempt_at": datetime.now(timezone.utc).isoformat(),
                "group_id": data.get("group_id", ""),
                "content": data.get("content", ""),
                "source": data.get("source", ""),
            },
        )

        await self._set_status(
            episode_id,
            EpisodeStatus.DEAD_LETTER,
            extra={"error": str(error), "retry_count": str(retry_count)},
        )

        # Extend TTL for dead-lettered episodes (30 days for investigation)
        await self.redis.expire(f"episode:{episode_id}:status", 2592000)

        # ACK the original message
        await self.redis.xack(
            self.config.stream_name,
            self.config.consumer_group,
            message_id,
        )

        # WebSocket: episode failed
        await self.ws.publish(data.get("group_id", ""), "episode.failed", {
            "episode_id": episode_id,
            "error": str(error),
            "retry_count": retry_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    async def _set_status(
        self,
        episode_id: str,
        status: EpisodeStatus,
        extra: dict | None = None,
    ):
        """Update episode status in Redis and notify via WebSocket."""
        now = datetime.now(timezone.utc).isoformat()
        previous = await self.redis.hget(f"episode:{episode_id}:status", "status")

        fields = {
            "status": status.value,
            "updated_at": now,
            "worker_id": self.worker_id,
        }
        if extra:
            fields.update(extra)

        await self.redis.hset(f"episode:{episode_id}:status", mapping=fields)

        group_id = await self.redis.hget(f"episode:{episode_id}:status", "group_id")
        if group_id:
            await self.ws.publish(group_id, "episode.status_changed", {
                "episode_id": episode_id,
                "previous_status": previous,
                "status": status.value,
                "updated_at": now,
            })

    def _select_model(self, content: str) -> str:
        """Route to Haiku for simple episodes, Sonnet for complex ones."""
        # Rough token estimate: ~4 chars per token
        estimated_tokens = len(content) / 4
        if estimated_tokens < self.config.haiku_token_threshold:
            return self.config.haiku_model
        return self.config.default_model
```

### Worker Manager (Lifecycle)

```python
class WorkerManager:
    """Manages the pool of ingestion workers and background tasks within the FastAPI lifecycle."""

    def __init__(self, config: IngestionConfig, dependencies: Dependencies):
        self.config = config
        self.deps = dependencies
        self.workers: list[asyncio.Task] = []
        self._embedding_retry_task: asyncio.Task | None = None

    async def start(self):
        """Start worker pool and background tasks. Called from FastAPI lifespan."""
        for i in range(self.config.worker_count):
            worker = IngestionWorker(
                worker_id=f"worker-{i}",
                redis_client=self.deps.redis,
                extractor=self.deps.extractor,
                resolver=self.deps.resolver,
                graph_store=self.deps.graph_store,
                activation_engine=self.deps.activation,
                embedding_service=self.deps.embeddings,
                ws_manager=self.deps.ws_manager,
                config=self.config,
            )
            task = asyncio.create_task(worker.run())
            self.workers.append(task)

        # Start the pending embeddings retry task.
        # This runs in the same process as ingestion workers so it shares
        # the EmbeddingProvider instance and its TokenBucket rate limiter.
        self._embedding_retry_task = asyncio.create_task(
            self._retry_pending_embeddings()
        )

    async def stop(self):
        """Graceful shutdown. Cancel workers and background tasks."""
        if self._embedding_retry_task:
            self._embedding_retry_task.cancel()
        for task in self.workers:
            task.cancel()
        all_tasks = self.workers + (
            [self._embedding_retry_task] if self._embedding_retry_task else []
        )
        await asyncio.gather(*all_tasks, return_exceptions=True)

    async def _retry_pending_embeddings(self):
        """Retry failed embeddings every 60s.

        Polls the engram:{group_id}:pending_embeddings Redis set for items
        that failed during ingestion. Retries each individually (not batched)
        since different items may have failed for different reasons. Shares
        the same EmbeddingProvider and rate limiter as the ingestion workers.

        See 04_embedding_strategy.md section 12 for the pending_embeddings contract.
        """
        while True:
            await asyncio.sleep(60)
            # Scan all group pending sets (pattern: engram:*:pending_embeddings)
            cursor = 0
            while True:
                cursor, keys = await self.deps.redis.scan(
                    cursor=cursor, match="engram:*:pending_embeddings", count=100
                )
                for set_key in keys:
                    # Extract group_id from key: engram:{group_id}:pending_embeddings
                    parts = set_key.split(":")
                    group_id = parts[1] if len(parts) >= 3 else "unknown"

                    pending = await self.deps.redis.smembers(set_key)
                    if not pending:
                        continue

                    for item_json in pending:
                        try:
                            item = json.loads(item_json)
                            episode_id = item["episode_id"]
                            episode_content = item["episode_content"]
                            entity_summaries = item.get("entity_summaries", {})

                            # Re-attempt embedding via the shared provider
                            vectors = await self.deps.embeddings.embed_ingestion_batch(
                                episode_content=episode_content,
                                entity_summaries=entity_summaries,
                            )
                            await self.deps.embeddings.store_vectors(
                                vectors=vectors,
                                episode_id=episode_id,
                                group_id=group_id,
                            )

                            # Success: remove from pending set
                            await self.deps.redis.srem(set_key, item_json)
                            # Metric: engram.embedding.retry_success
                        except Exception:
                            # Leave in set for next retry cycle
                            # Metric: engram.embedding.retry_failure
                            continue

                if cursor == 0:
                    break


# FastAPI lifespan integration
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_manager = WorkerManager(config, dependencies)
    await worker_manager.start()
    yield
    await worker_manager.stop()

app = FastAPI(lifespan=lifespan)
```

### Status Endpoint

```python
@router.get("/api/episodes/{episode_id}/status")
async def get_episode_status(
    episode_id: str,
    redis_client: redis.Redis = Depends(get_redis),
):
    status_data = await redis_client.hgetall(f"episode:{episode_id}:status")
    if not status_data:
        raise HTTPException(status_code=404, detail="Episode not found")

    entities = json.loads(status_data.get("entities", "[]"))
    return {
        "episode_id": episode_id,
        "status": status_data["status"],
        "created_at": status_data["created_at"],
        "updated_at": status_data["updated_at"],
        "retry_count": int(status_data.get("retry_count", 0)),
        "error": status_data.get("error") or None,
        "entities": entities if entities else None,
        "facts": int(status_data.get("facts", 0)) or None,
    }
```

### Admin: Re-queue Dead-Lettered Episode

```python
@router.post("/admin/episodes/{episode_id}/requeue")
async def requeue_episode(
    episode_id: str,
    redis_client: redis.Redis = Depends(get_redis),
):
    """Move a dead-lettered episode back to the main processing stream."""
    status_data = await redis_client.hgetall(f"episode:{episode_id}:status")
    if not status_data:
        raise HTTPException(status_code=404, detail="Episode not found")
    if status_data["status"] != "dead_letter":
        raise HTTPException(status_code=400, detail="Episode is not dead-lettered")

    # Find the episode in the dead letter stream
    # In practice, we store enough data in the DLQ message to re-enqueue
    entries = await redis_client.xrange(
        "engram:dead_letter_stream",
        min="-",
        max="+",
        count=1000,
    )
    for entry_id, entry_data in entries:
        if entry_data.get("episode_id") == episode_id:
            # Re-enqueue to main stream
            await redis_client.xadd(
                "engram:episode_stream",
                {
                    "episode_id": episode_id,
                    "group_id": entry_data.get("group_id", ""),
                    "content": entry_data.get("content", ""),
                    "source": entry_data.get("source", ""),
                    "metadata": "{}",
                    "created_at": entry_data.get("last_attempt_at", ""),
                    "priority": "normal",
                },
            )
            # Reset status
            await redis_client.hset(
                f"episode:{episode_id}:status",
                mapping={
                    "status": "queued",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "retry_count": "0",
                    "error": "",
                },
            )
            return {"episode_id": episode_id, "status": "queued", "message": "Re-queued successfully"}

    raise HTTPException(status_code=404, detail="Episode not found in dead letter queue")
```

---

## Idempotency Guarantees

Each pipeline stage must be idempotent to support retries safely.

| Stage | Idempotency Strategy |
|-------|---------------------|
| **Extracting** | Pure function (same input = same output). Safe to re-run. |
| **Resolving** | Entity dedup uses deterministic matching. Re-resolving the same entities produces the same merge result. |
| **Writing** | Graph writes use MERGE (upsert) semantics in Cypher. Re-writing the same entities/edges is a no-op. Episode node uses `episode_id` as unique key. |
| **Embedding** | `embed_ingestion_batch()` is deterministic for same inputs. Redis HSET on vector keys is idempotent (overwrite with same data). If already present, re-write is a no-op at the semantic level. |
| **Activating** | Activation updates are additive. Re-applying the same access boost is acceptable (slightly inflates activation, but within tolerance). |

---

## Monitoring and Observability

### Metrics to Track

| Metric | Type | Description |
|--------|------|-------------|
| `engram.episodes.queued` | Counter | Episodes enqueued to stream |
| `engram.episodes.completed` | Counter | Episodes successfully processed |
| `engram.episodes.failed` | Counter | Episodes sent to dead letter |
| `engram.episodes.retried` | Counter | Retry attempts |
| `engram.episodes.processing_duration_ms` | Histogram | End-to-end processing time |
| `engram.episodes.stage_duration_ms` | Histogram (per stage) | Time spent in each pipeline stage |
| `engram.queue.depth` | Gauge | Current stream length (XLEN) |
| `engram.queue.pending` | Gauge | Messages claimed but not ACKed (XPENDING) |
| `engram.dead_letter.depth` | Gauge | Dead letter stream length |
| `engram.extraction.model_calls` | Counter (per model) | Claude API calls by model tier |
| `engram.embedding.api_calls` | Counter (per provider) | Embedding API calls (Voyage, OpenAI, local) |
| `engram.embedding.pending` | Gauge | Size of `pending_embeddings` set (failed embeddings awaiting retry) |
| `engram.embedding.batch_size` | Histogram | Number of texts per `embed_ingestion_batch()` call |
| `engram.embedding.retry_success` | Counter | Pending embeddings successfully retried by background task |
| `engram.embedding.retry_failure` | Counter | Pending embedding retry attempts that failed |

### Health Check

```python
@router.get("/health")
async def health_check(redis_client: redis.Redis = Depends(get_redis)):
    stream_len = await redis_client.xlen("engram:episode_stream")
    pending = await redis_client.xpending_range(
        "engram:episode_stream", "engram_workers", "-", "+", 100
    )
    dead_letter_len = await redis_client.xlen("engram:dead_letter_stream")

    return {
        "status": "healthy",
        "queue_depth": stream_len,
        "pending_messages": len(pending),
        "dead_letter_depth": dead_letter_len,
    }
```

---

## Integration Points

### With MCP Server (Task #6)

The `remember` tool response shape is defined above. The MCP server calls `handle_remember()` which enqueues and returns immediately. The MCP tool definition should document that `remember` is async and results are not immediately available.

Key contract:
- `remember` returns `{episode_id, status: "queued", message, status_url}`
- MCP server does NOT need to expose the status endpoint as an MCP tool (it is a REST endpoint for the dashboard)

### With Frontend/Dashboard (Task #7)

> **Canonical reference:** `07_frontend_architecture.md` Section 4.

WebSocket endpoint: `ws://localhost:8080/ws/dashboard` (unified, auth-scoped by `group_id` from token).

All events are wrapped in a `{seq, type, timestamp, payload}` envelope by the `WebSocketManager`. The worker publishes `(group_id, event_type, payload)` and the manager assigns monotonically increasing `seq` numbers per connection for gap detection.

The ingestion pipeline emits events in this strict order per episode (see Section 4.5 of `07_frontend_architecture.md`):

1. `episode.queued` - Add skeleton to Memory Feed
2. `episode.status_changed` (extracting, resolving, writing, embedding, activating) - Update progress indicator
3. `graph.nodes_added` - Surgical Zustand `mergeGraphDelta` for new entities
4. `graph.edges_added` - Add new relationships to graph canvas
5. `graph.edges_invalidated` - Mark superseded facts with `valid_to`
6. `graph.nodes_updated` - Update changed entity summaries
7. `activation.updated` - Update node activation values + Activation Monitor chart
8. `episode.completed` - Terminal success, update Memory Feed with full entity/fact data
9. `episode.failed` - Show error badge with re-queue action (calls `POST /admin/episodes/{id}/requeue`)

The `graph_store.write_episode()` return type (`WriteResult`) must include `new_nodes`, `new_edges`, `invalidated_edge_ids`, `updated_nodes`, and `changed_entities` so the worker can emit all granular graph events.

### With Activation Engine (Task #2)

The `activating` stage calls `activation_engine.update_on_ingestion()` which:
1. Boosts `current_activation` for all entities mentioned in the episode.
2. Increments `access_count`.
3. Runs spreading activation (2 hops) from touched entities.
4. Returns list of activation changes for WebSocket notification.

### With Embedding Strategy (Task #4)

Per `04_embedding_strategy.md`, the `embedding` stage runs **before** activation (steps 4-5 of 7). Key integration points:

1. **`embed_ingestion_batch()`:** The pipeline calls this with the episode content + a dict of `{entity_id: summary}` for entities whose summaries changed during the writing stage. All texts go in a single API call to the configured provider (default: Voyage AI `voyage-3-lite`, 512 dimensions).

2. **Tracking changed summaries:** `graph_store.write_episode()` must return `changed_entities: dict[str, str]` -- a mapping of entity IDs to their new/updated summaries. Only these entities get re-embedded, not all touched entities.

3. **Graceful failure:** If the embedding API fails (rate limit, timeout, provider down), the pipeline **continues** to the activation stage. The failed episode + entity IDs are queued in `engram:{group_id}:pending_embeddings` (Redis set). A background task retries these every 60 seconds with exponential backoff.

4. **Provider configurability:** The embedding provider is set in config (`embedding.provider`: `"voyage"` | `"openai"` | `"local"`). The ingestion pipeline does not need to know which provider is used -- it calls the abstract `EmbeddingProvider.embed()` interface.

5. **Vector storage keys:** Vectors are stored in Redis at `engram:{group_id}:vec:entity:{entity_id}` and `engram:{group_id}:vec:episode:{episode_id}`, indexed by a single RediSearch HNSW index (`engram_vectors`).
