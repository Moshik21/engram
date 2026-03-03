# 04 -- Embedding Strategy & Vector Storage

## Overview

The tech spec's retrieval flow starts with "Semantic search (top-K candidate nodes)" but never specifies the embedding model, vector index, or storage backend. This document fills that gap with concrete, implementation-ready decisions.

**Design constraints from the spec:**
- Redis is already in the stack (activation state storage)
- FalkorDB runs as a Redis module (same Redis instance or cluster)
- Retrieval is hot-path: semantic search is the first step of every `recall` and `get_context` call
- Multi-tenant via `group_id` from day one
- Claude-first ecosystem (but embedding model is independent of LLM choice)

---

## 1. Embedding Model Selection

### Recommendation: Voyage AI `voyage-3-lite`

| Model | Dimensions | Max Tokens | Quality (MTEB avg) | Latency (p50) | Cost per 1M tokens | Notes |
|-------|-----------|------------|---------------------|---------------|---------------------|-------|
| `voyage-3` | 1024 | 32K | ~68.3 | ~80ms | $0.06 | Top quality, higher cost |
| **`voyage-3-lite`** | **512** | **32K** | **~65.2** | **~35ms** | **$0.02** | **Best cost/quality tradeoff** |
| `text-embedding-3-small` (OpenAI) | 1536 | 8K | ~62.3 | ~50ms | $0.02 | Higher dimensions, lower quality |
| `text-embedding-3-large` (OpenAI) | 3072 | 8K | ~64.6 | ~100ms | $0.13 | Expensive, large vectors |
| `nomic-embed-text` (local) | 768 | 8K | ~62.5 | ~15ms* | Free | Requires GPU or slow on CPU |

*Local latency varies by hardware.

### Rationale

- **`voyage-3-lite` is the default.** 512 dimensions keeps vector storage compact (2KB per vector in float32, 1KB in float16). Quality is sufficient for entity/fact retrieval where we are not relying on similarity alone -- the activation engine re-ranks results.
- **`voyage-3` is the upgrade path.** Users who want higher recall quality can switch via config. Same API, just 1024 dimensions and 3x the cost.
- **Local model (`nomic-embed-text`) is the zero-infra option.** For users who do not want external API calls for embeddings. Supported via config toggle. Runs through `sentence-transformers` or Ollama.
- **OpenAI models are supported but not default.** Users already in the OpenAI ecosystem can configure `text-embedding-3-small`. We normalize the interface.

### Provider Abstraction

```python
# server/engram/retrieval/embeddings.py

from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    """Abstract embedding provider. All providers return list[float]."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        ...

class VoyageProvider(EmbeddingProvider):
    """Voyage AI embeddings (voyage-3, voyage-3-lite)."""

    def __init__(self, model: str = "voyage-3-lite", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        self._dim = 512 if "lite" in model else 1024

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import voyageai
        client = voyageai.AsyncClient(api_key=self.api_key)
        result = await client.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    def dimension(self) -> int:
        return self._dim

class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small, text-embedding-3-large)."""
    # Similar pattern, uses openai SDK

class LocalProvider(EmbeddingProvider):
    """Local embeddings via sentence-transformers or Ollama."""
    # Loads model on first call, caches in memory
```

---

## 2. What Gets Embedded

### Two embedding categories, both stored in the same vector index:

| Content Type | Source | When Embedded | Update Frequency | Key Field |
|-------------|--------|---------------|------------------|-----------|
| **Entity summary** | `Entity.summary` field from FalkorDB | On entity creation and summary update | When extraction updates the summary | `entity:{entity_id}` |
| **Episode content** | `Episode.content` (raw conversation text) | On episode ingestion | Never (episodes are immutable) | `episode:{episode_id}` |

### Why both?

- **Entity summaries** capture distilled knowledge: "Konner is a software engineer building ReadyCheck, a meeting prep SaaS." Searching against summaries finds the right entities.
- **Episode content** captures raw context: the actual conversation where something was mentioned. This is needed for fact retrieval (`search_facts`) where the user's exact phrasing matters.
- The retrieval flow uses entity embeddings first (for `recall` and `get_context`), then optionally falls back to episode embeddings for `search_facts`.

### What does NOT get embedded:

- **Relationship predicates** -- too short, low information density. Relationships are traversed via the graph after entities are found.
- **Individual facts** -- facts are stored as relationship properties. They are retrieved by traversing from activated entities, not via vector search.
- **Activation state** -- purely numerical, no semantic content.

---

## 3. Vector Storage: Redis Search with HNSW

### Why Redis Search?

Redis is already in the stack for activation state. FalkorDB itself runs as a Redis module. Redis Search (RediSearch) supports HNSW vector indexing natively. Using it means:

- No additional infrastructure (no Pinecone, Qdrant, Weaviate, pgvector)
- Sub-millisecond KNN queries for personal-scale graphs (< 100K vectors)
- Tenant isolation via key prefix + query filter
- Atomic operations with activation state (same Redis instance)

### Index Schema

```
# Redis key pattern for vector documents
engram:{group_id}:vec:{content_type}:{id}

# Examples:
engram:user_abc:vec:entity:ent_01HXYZ
engram:user_abc:vec:episode:ep_01HXYZ
```

### Index Creation (RediSearch FT.CREATE)

```python
VECTOR_INDEX_SCHEMA = {
    "index_name": "engram_vectors",       # Single global index
    "prefix": "engram:",                   # Matches all tenant keys
    "schema": {
        "group_id":      {"type": "TAG"},                    # Tenant filter
        "content_type":  {"type": "TAG"},                    # "entity" or "episode"
        "source_id":     {"type": "TAG"},                    # Entity or episode ID
        "text":          {"type": "TEXT"},                    # Original text (for BM25 hybrid)
        "entity_type":   {"type": "TAG"},                    # Entity type (Person, Project, etc.)
        "created_at":    {"type": "NUMERIC", "SORTABLE": True},
        "embedding":     {
            "type": "VECTOR",
            "algorithm": "HNSW",
            "data_type": "FLOAT32",
            "dim": 512,                                     # Matches voyage-3-lite
            "distance_metric": "COSINE",
            "initial_cap": 10000,
            "m": 16,                                         # HNSW connectivity
            "ef_construction": 200,                          # Build-time accuracy
        }
    }
}
```

### HNSW Parameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dim` | 512 | Matches `voyage-3-lite`. Configurable per provider. |
| `distance_metric` | COSINE | Standard for text embeddings. Normalized dot product. |
| `m` | 16 | Default connectivity. Good balance of speed and recall for < 100K vectors. |
| `ef_construction` | 200 | Higher build accuracy. Ingestion is async so build time is not critical. |
| `ef_runtime` | 50 | Query-time search depth. Set per query, not at index creation. Default 50 gives > 95% recall. |
| `data_type` | FLOAT32 | Full precision. At 512 dims, each vector is 2KB. 100K vectors = ~200MB. Acceptable. |

### Dimension Reconfiguration

When the user switches embedding models (e.g., `voyage-3-lite` to `voyage-3`), the dimension changes. This requires:

1. Drop the existing index: `FT.DROPINDEX engram_vectors`
2. Delete all vector keys: `DEL engram:*:vec:*`
3. Recreate index with new dimension
4. Re-embed all entities and episodes (background job)

This is a destructive but infrequent operation. The system logs a warning and requires explicit confirmation via config flag `embedding.allow_reindex: true`.

---

## 4. Embedding Generation Timing

### During Async Ingestion Pipeline

Embedding generation is a step in the ingestion pipeline, not a lazy operation. Here is where it fits:

```
Episode arrives (MCP `remember` call)
  |
  v
Step 1: Store raw episode in FalkorDB
  |
  v
Step 2: Claude entity extraction (async)
  |
  v
Step 3: Graph operations (create/update entities, edges)
  |
  v
Step 4: Embed episode content          <-- NEW
  |       (voyage-3-lite API call)
  |       Store in Redis: engram:{group_id}:vec:episode:{episode_id}
  |
  v
Step 5: Embed new/updated entities     <-- NEW
  |       (batch: all entities whose summary changed in step 3)
  |       Store in Redis: engram:{group_id}:vec:entity:{entity_id}
  |
  v
Step 6: Update activation state (Redis)
  |
  v
Step 7: WebSocket notify (dashboard)
```

### Why eager (not lazy)?

- **Retrieval must be fast.** If embeddings are generated lazily on first query, the first `recall` call after ingestion would block on an API call. Unacceptable for MCP tool response times.
- **Batching is efficient.** Voyage AI supports batch embedding (up to 128 texts per request). A single episode typically creates 2-5 entities. One batch API call covers the episode + all its entities.
- **Cost is bounded.** Each episode produces ~1 embedding for the episode text + ~3 embeddings for new/updated entities. At $0.02/1M tokens with `voyage-3-lite`, a 500-token episode costs ~$0.00001.

### Batch Embedding Strategy

```python
async def embed_ingestion_batch(
    episode_content: str,
    entity_summaries: dict[str, str],  # {entity_id: summary}
    provider: EmbeddingProvider,
) -> dict[str, list[float]]:
    """Embed episode and entities in a single batch API call."""
    texts = [episode_content] + list(entity_summaries.values())
    ids = ["episode"] + list(entity_summaries.keys())

    vectors = await provider.embed(texts)

    return dict(zip(ids, vectors))
```

---

## 5. Tenant Isolation

### Single Index, Tag-Filtered Queries

Rather than one index per `group_id` (which would require dynamic index creation), we use a single global index with `group_id` as a TAG field. Every query includes a mandatory `group_id` filter.

```python
# Query pattern: KNN search filtered by tenant
VECTOR_SEARCH_QUERY = """
FT.SEARCH engram_vectors
  "(@group_id:{{$group_id}} @content_type:{{$content_type}})
   =>[KNN $k @embedding $blob AS score]"
  PARAMS 6
    group_id {group_id}
    content_type {content_type}
    k {k}
    blob {embedding_bytes}
  SORTBY score
  LIMIT 0 {k}
  RETURN 4 source_id text score entity_type
  DIALECT 2
"""
```

### Why single index?

- Simpler operations: one index to create, monitor, and maintain.
- Redis Search TAG filters are applied before the HNSW traversal, so tenant data never leaks.
- At personal scale (< 10K vectors per tenant), a single index with filtering performs identically to separate indices.
- If hosted service grows beyond 1M total vectors, we can shard by `group_id` range.

---

## 6. Retrieval Query Patterns

### Pattern 1: Entity Retrieval (for `recall` and `get_context`)

This is the primary retrieval path used in the activation-aware retrieval flow from the spec.

```python
async def semantic_search_entities(
    query: str,
    group_id: str,
    top_k: int = 20,
    provider: EmbeddingProvider,
    redis: Redis,
) -> list[ScoredEntity]:
    """
    Step 1 of the retrieval flow: find candidate entities by semantic similarity.
    Returns entity IDs with similarity scores for the activation engine.
    """
    query_vector = await provider.embed([query])
    query_bytes = np.array(query_vector[0], dtype=np.float32).tobytes()

    results = await redis.execute_command(
        "FT.SEARCH", "engram_vectors",
        f"(@group_id:{{{group_id}}} @content_type:{{entity}})"
        f"=>[KNN {top_k} @embedding $blob AS score]",
        "PARAMS", "2", "blob", query_bytes,
        "SORTBY", "score",
        "LIMIT", "0", str(top_k),
        "RETURN", "3", "source_id", "score", "entity_type",
        "DIALECT", "2",
    )

    return parse_search_results(results)
```

### Pattern 2: Fact/Episode Retrieval (for `search_facts`)

```python
async def semantic_search_episodes(
    query: str,
    group_id: str,
    top_k: int = 10,
    provider: EmbeddingProvider,
    redis: Redis,
) -> list[ScoredEpisode]:
    """Search episode content for fact retrieval."""
    query_vector = await provider.embed([query])
    query_bytes = np.array(query_vector[0], dtype=np.float32).tobytes()

    results = await redis.execute_command(
        "FT.SEARCH", "engram_vectors",
        f"(@group_id:{{{group_id}}} @content_type:{{episode}})"
        f"=>[KNN {top_k} @embedding $blob AS score]",
        "PARAMS", "2", "blob", query_bytes,
        "SORTBY", "score",
        "LIMIT", "0", str(top_k),
        "RETURN", "3", "source_id", "text", "score",
        "DIALECT", "2",
    )

    return parse_search_results(results)
```

### Pattern 3: Hybrid Search (Semantic + BM25)

For queries that contain specific names or terms, pure semantic search can miss exact matches. The `text` field in the index supports BM25 keyword search.

```python
async def hybrid_search_entities(
    query: str,
    group_id: str,
    top_k: int = 20,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    provider: EmbeddingProvider,
    redis: Redis,
) -> list[ScoredEntity]:
    """
    Combine semantic similarity with BM25 keyword matching.
    Used when the query contains specific entity names.
    """
    # Run both searches
    semantic_results = await semantic_search_entities(
        query, group_id, top_k * 2, provider, redis
    )

    keyword_results = await redis.execute_command(
        "FT.SEARCH", "engram_vectors",
        f"@group_id:{{{group_id}}} @content_type:{{entity}} "
        f"@text:({escape_query(query)})",
        "LIMIT", "0", str(top_k * 2),
        "RETURN", "2", "source_id", "entity_type",
        "DIALECT", "2",
    )

    # Reciprocal rank fusion
    return reciprocal_rank_fusion(
        semantic_results, keyword_results,
        semantic_weight, keyword_weight, top_k
    )
```

---

## 7. Update Strategy When Summaries Change

Entity summaries change when new information is extracted. The embedding must stay in sync.

### Strategy: Re-embed on Summary Update

```python
async def update_entity_embedding(
    entity_id: str,
    new_summary: str,
    group_id: str,
    provider: EmbeddingProvider,
    redis: Redis,
) -> None:
    """Re-embed an entity when its summary changes."""
    vector = await provider.embed([new_summary])
    vector_bytes = np.array(vector[0], dtype=np.float32).tobytes()

    key = f"engram:{group_id}:vec:entity:{entity_id}"
    await redis.hset(key, mapping={
        "group_id": group_id,
        "content_type": "entity",
        "source_id": entity_id,
        "text": new_summary,
        "embedding": vector_bytes,
    })
```

### When does a summary change?

During ingestion step 3, the graph manager may update an entity's summary when new facts are extracted. The ingestion pipeline tracks which entities had their summaries modified and passes that set to step 5 (embedding generation).

```python
# In the ingestion pipeline
changed_entities = graph_manager.upsert_entities(extracted_entities)
# changed_entities = {"ent_01": "Updated summary...", "ent_03": "New summary..."}

# Only re-embed entities whose summaries actually changed
if changed_entities:
    vectors = await embed_ingestion_batch(
        episode_content, changed_entities, provider
    )
    await store_vectors(vectors, group_id, redis)
```

### Stale embedding window

Between the moment a summary changes and the new embedding is stored, there is a brief window where the old embedding is served. This is acceptable because:

1. The window is typically < 1 second (embedding API call latency).
2. The activation engine compensates: a freshly-updated entity has high activation, so even with a slightly stale embedding it will rank highly.
3. Ingestion is async -- the user is not waiting for a response during this window.

---

## 8. Integration with Composite Scoring

The spec defines a composite retrieval score:

```
Score = semantic_similarity * 0.4
      + current_activation * 0.3
      + recency_score * 0.2
      + frequency_score * 0.1
```

The embedding system provides the `semantic_similarity` component. Here is how it integrates:

```python
async def retrieve_memories(
    query: str,
    group_id: str,
    top_k_candidates: int = 20,
    top_n_results: int = 5,
    provider: EmbeddingProvider,
    redis: Redis,
    activation_engine: ActivationEngine,
) -> list[ScoredMemory]:
    """Full retrieval flow from the spec."""

    # Step 1: Semantic search -- top-K candidates from vector index
    candidates = await semantic_search_entities(
        query, group_id, top_k_candidates, provider, redis
    )

    # Step 2: Contextual boost on candidates
    entity_ids = [c.source_id for c in candidates]
    activation_engine.contextual_boost(entity_ids)

    # Step 3: Spreading activation from boosted nodes (2 hops)
    activation_engine.spread(entity_ids, hops=2)

    # Step 4: Composite scoring
    scored = []
    for candidate in candidates:
        state = await activation_engine.get_state(candidate.source_id)
        score = (
            candidate.similarity_score * 0.4
            + state.current_activation * 0.3
            + state.recency_score * 0.2
            + state.frequency_score * 0.1
        )
        scored.append(ScoredMemory(
            entity_id=candidate.source_id,
            score=score,
            state=state,
        ))

    # Step 5: Return top-N, update activation
    scored.sort(key=lambda x: x.score, reverse=True)
    top_results = scored[:top_n_results]

    for result in top_results:
        await activation_engine.record_access(result.entity_id)

    return top_results
```

---

## 9. Configuration Fields

The following fields should be added to the Engram config schema:

```yaml
embedding:
  # Provider selection
  provider: "voyage"                    # "voyage" | "openai" | "local"
  model: "voyage-3-lite"               # Model name passed to provider

  # API credentials (provider-specific)
  api_key: null                         # Set via env var ENGRAM_EMBEDDING_API_KEY

  # Vector index parameters
  dimensions: 512                       # Auto-set from model, but overridable
  distance_metric: "COSINE"             # "COSINE" | "IP" | "L2"
  hnsw_m: 16                            # HNSW connectivity parameter
  hnsw_ef_construction: 200             # Build-time search depth
  hnsw_ef_runtime: 50                   # Query-time search depth

  # Retrieval parameters
  candidate_top_k: 20                   # Number of candidates from vector search
  hybrid_search: false                  # Enable BM25 + semantic hybrid
  semantic_weight: 0.7                  # Weight for semantic score in hybrid
  keyword_weight: 0.3                   # Weight for keyword score in hybrid

  # Operational
  batch_size: 64                        # Max texts per embedding API call
  allow_reindex: false                  # Must be true to allow dimension changes

  # Local provider settings
  local_model_name: "nomic-ai/nomic-embed-text-v1.5"
  local_device: "cpu"                   # "cpu" | "cuda" | "mps"
```

---

## 10. Benchmark Baseline: Vector Search

For the Week 3 benchmark comparing activation-based retrieval vs pure semantic search, the vector search baseline is:

### Baseline: Pure Semantic Search

```python
async def baseline_semantic_retrieval(query, group_id, top_n=5):
    """Baseline: rank by cosine similarity only."""
    candidates = await semantic_search_entities(
        query, group_id, top_k=top_n
    )
    return candidates  # No activation scoring, no spreading, just similarity
```

### Benchmark Setup

1. **Seed data:** 50 episodes, ~150 entities, ~300 relationships
2. **Query set:** 20 queries with human-labeled relevant entity sets
3. **Access simulation:** Before benchmark queries, simulate 2 weeks of access patterns (some entities accessed frequently, some recently, some dormant)
4. **Metrics:**
   - Precision@5: fraction of top-5 results that are relevant
   - Recall@10: fraction of relevant results found in top-10
   - MRR: mean reciprocal rank of first relevant result
   - Latency p50/p95 for both approaches

### Expected outcome

Pure semantic search will perform well on explicit queries ("What do I know about Python?") but poorly on associative queries ("What was I working on last week?") where activation and recency matter. The activation engine should show significant gains on the associative query subset.

---

## 11. Memory & Cost Estimates

### Storage

| Scale | Entities | Episodes | Total Vectors | Redis Memory | Monthly Embedding Cost |
|-------|----------|----------|---------------|-------------|----------------------|
| Light (1 month) | 200 | 500 | 700 | ~1.5 MB | ~$0.01 |
| Medium (6 months) | 2,000 | 5,000 | 7,000 | ~15 MB | ~$0.05 |
| Heavy (1 year) | 10,000 | 25,000 | 35,000 | ~75 MB | ~$0.20 |
| Hosted (100 users) | 200,000 | 500,000 | 700,000 | ~1.5 GB | ~$2.00 |

*Assumes 512-dim FLOAT32 vectors (~2KB each) plus index overhead (~2x raw size).*

Embedding costs are negligible compared to Claude API costs for entity extraction.

### Latency

| Operation | Expected Latency |
|-----------|-----------------|
| Embedding API call (single text, voyage-3-lite) | 30-50ms |
| Embedding API call (batch of 10, voyage-3-lite) | 40-70ms |
| Redis FT.SEARCH KNN (10K vectors, top-20) | 1-3ms |
| Redis FT.SEARCH KNN (100K vectors, top-20) | 3-8ms |
| Full retrieval flow (embed query + search + score) | 35-60ms |

---

## 12. Error Handling & Resilience

### Embedding API Failures

If the embedding provider is unavailable during ingestion:

1. Store the episode/entity without an embedding (skip steps 4-5 of the pipeline).
2. Mark the record as `embedding_pending` in a Redis set: `engram:{group_id}:pending_embeddings`.
3. A background task retries pending embeddings every 60 seconds with exponential backoff.
4. Retrieval still works for records that have embeddings; pending records are invisible to vector search but visible via graph traversal.

### Rate Limiting

Voyage AI rate limits at 300 RPM on the free tier. For batch ingestion:

- Use batch embedding (up to 128 texts per call) to minimize request count.
- Implement a token bucket rate limiter in the embedding provider.
- If rate limited, queue embeddings and process them as capacity frees up.

```python
class RateLimitedProvider(EmbeddingProvider):
    """Wraps any provider with rate limiting and retry logic."""

    def __init__(self, inner: EmbeddingProvider, max_rpm: int = 300):
        self.inner = inner
        self.limiter = TokenBucket(max_rpm, per_seconds=60)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await self.limiter.acquire()
        try:
            return await self.inner.embed(texts)
        except RateLimitError:
            await asyncio.sleep(self.limiter.retry_after())
            return await self.inner.embed(texts)
```

---

## 13. SearchIndex Protocol Conformance (Lite Mode Compatibility)

The lite mode design (see `09_lite_mode.md`) defines a `SearchIndex` protocol that both the full vector search backend and the lite FTS5 backend must implement:

```python
class SearchIndex(Protocol):
    async def index_entity(self, entity: Entity) -> None: ...
    async def index_relationship(self, rel: Relationship) -> None: ...
    async def index_episode(self, episode: Episode) -> None: ...
    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Return (entity_id, relevance_score) with score normalized to 0.0-1.0."""
        ...
    async def remove(self, entity_id: str) -> None: ...
```

### VectorSearchIndex Implementation

The full-mode vector search backend wraps the Redis Search HNSW index and conforms to this protocol:

```python
class VectorSearchIndex:
    """Full-mode search using Redis Search HNSW vectors."""

    def __init__(self, provider: EmbeddingProvider, redis: Redis):
        self.provider = provider
        self.redis = redis

    async def index_entity(self, entity: Entity) -> None:
        vector = await self.provider.embed([entity.summary])
        vector_bytes = np.array(vector[0], dtype=np.float32).tobytes()
        key = f"engram:{entity.group_id}:vec:entity:{entity.id}"
        await self.redis.hset(key, mapping={
            "group_id": entity.group_id,
            "content_type": "entity",
            "source_id": entity.id,
            "text": entity.summary,
            "entity_type": entity.entity_type,
            "created_at": entity.created_at.timestamp(),
            "embedding": vector_bytes,
        })

    async def index_episode(self, episode: Episode) -> None:
        vector = await self.provider.embed([episode.content])
        vector_bytes = np.array(vector[0], dtype=np.float32).tobytes()
        key = f"engram:{episode.group_id}:vec:episode:{episode.id}"
        await self.redis.hset(key, mapping={
            "group_id": episode.group_id,
            "content_type": "episode",
            "source_id": episode.id,
            "text": episode.content,
            "entity_type": "",
            "created_at": episode.created_at.timestamp(),
            "embedding": vector_bytes,
        })

    async def index_relationship(self, rel: Relationship) -> None:
        pass  # Relationships are not embedded (see section 2)

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        query_vector = await self.provider.embed([query])
        query_bytes = np.array(query_vector[0], dtype=np.float32).tobytes()

        # Build filter: group_id is required, entity_types is optional
        type_filter = ""
        if entity_types:
            types_str = "|".join(entity_types)
            type_filter = f" @entity_type:{{{types_str}}}"

        results = await self.redis.execute_command(
            "FT.SEARCH", "engram_vectors",
            f"(@group_id:{{{group_id}}} @content_type:{{entity}}{type_filter})"
            f"=>[KNN {limit} @embedding $blob AS score]",
            "PARAMS", "2", "blob", query_bytes,
            "SORTBY", "score",
            "LIMIT", "0", str(limit),
            "RETURN", "2", "source_id", "score",
            "DIALECT", "2",
        )

        # Redis cosine distance is 0 (identical) to 2 (opposite).
        # Normalize to 0.0-1.0 similarity: similarity = 1 - (distance / 2)
        return [
            (doc["source_id"], 1.0 - (float(doc["score"]) / 2.0))
            for doc in parse_ft_search(results)
        ]

    async def remove(self, entity_id: str) -> None:
        # Scan for keys matching this entity across all group_ids
        async for key in self.redis.scan_iter(
            match=f"engram:*:vec:entity:{entity_id}"
        ):
            await self.redis.delete(key)
```

### Score Normalization

Redis Search HNSW with COSINE distance returns values in `[0, 2]` where 0 = identical. The protocol requires `[0.0, 1.0]` where 1.0 = most relevant. The conversion is:

```
similarity = 1.0 - (cosine_distance / 2.0)
```

This aligns with the lite mode's BM25 normalization (which divides by max BM25 score in the result set), ensuring the composite scorer receives comparable values from either backend.

### Retrieval Weight Adjustment

When lite mode is active (FTS5 backend), the composite scoring weights shift to compensate for lower-quality text matching:

| Weight | Full Mode (Vector) | Lite Mode (FTS5) |
|--------|-------------------|------------------|
| Text match | 0.40 | 0.25 |
| Activation | 0.30 | 0.35 |
| Recency | 0.20 | 0.25 |
| Frequency | 0.10 | 0.15 |

These weights are configured in `retrieval.scoring_weights` (see config schema).

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default embedding model | `voyage-3-lite` (512d) | Best quality/cost/size tradeoff; activation engine compensates for slightly lower similarity quality |
| Vector storage | Redis Search HNSW | Already in the stack, no new infra, sub-millisecond KNN |
| What gets embedded | Entity summaries + episode content | Entities for recall, episodes for fact search |
| Embedding timing | Eager, during async ingestion pipeline (steps 4-5) | Retrieval must be fast; no lazy embedding on query path |
| Tenant isolation | Single index, TAG-filtered queries on `group_id` | Simpler ops, same performance at personal scale |
| Similarity metric | Cosine | Standard for text embeddings |
| Update strategy | Re-embed on summary change, tracked by ingestion pipeline | Brief staleness window is acceptable |
| Provider abstraction | `EmbeddingProvider` ABC with Voyage, OpenAI, local implementations | Swap providers via config without code changes |
| Lite mode compatibility | `VectorSearchIndex` implements `SearchIndex` protocol | Retrieval layer swaps vector/FTS5 backends transparently |
