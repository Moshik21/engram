# Engram Benchmark Methodology -- Week 3 Validation

## Purpose

The activation engine is Engram's core hypothesis: that spreading activation retrieval surfaces more relevant memories than pure vector similarity, keyword search, or graph traversal alone. This benchmark validates that hypothesis **before** building Weeks 4-8. If activation does not meaningfully outperform baselines, we pivot early.

This document defines the dataset, scoring rubric, comparison methodology, go/no-go framework, and cost model.

---

## 1. Dataset Design

### 1.1 Overview

| Parameter | Value |
|-----------|-------|
| Total episodes | 50 |
| Conversation threads | 10 |
| Episodes per thread | 5 (avg) |
| Total entities expected | ~80-120 |
| Total relationships expected | ~150-250 |
| Evaluation queries | 20 |
| Ground truth annotations | Human-annotated, 3-5 relevant results per query |

### 1.2 Conversation Threads

Each thread simulates a realistic user-agent interaction pattern. Threads are designed to create the graph structures that test each retrieval method's strengths and weaknesses.

**Thread 1: Career Evolution (6 episodes)**
Tests entity evolution and temporal contradiction.

```
Episode 1 (Day 1):  "I work at Acme Corp as a senior engineer. Been there 3 years.
                      Working mostly in Python and React."
Episode 2 (Day 3):  "Had a great 1:1 with my manager Sarah. She mentioned the
                      tech lead opening on the Platform team."
Episode 3 (Day 10): "Started interviewing at Vercel. Really excited about their
                      edge runtime work."
Episode 4 (Day 15): "Got the Vercel offer! $230K total comp. Going to negotiate."
Episode 5 (Day 18): "Accepted the Vercel offer. Putting in my notice at Acme next week.
                      Sarah was disappointed but supportive."
Episode 6 (Day 25): "First week at Vercel. Onboarding is great. Working on the
                      deployment pipeline team. Using TypeScript and Go now."
```

Entities created: User, Acme Corp, Sarah, Platform team, Vercel, deployment pipeline team
Temporal contradictions: works_at (Acme -> Vercel), primary languages (Python/React -> TypeScript/Go)

**Thread 2: Side Project Arc (5 episodes)**
Tests project lifecycle and technology associations.

```
Episode 1 (Day 2):  "Starting a new SaaS project called ReadyCheck. It's a
                      meeting prep tool. Using Next.js and Supabase."
Episode 2 (Day 8):  "ReadyCheck progress: built the auth flow and basic dashboard.
                      Need to integrate Stripe for billing."
Episode 3 (Day 14): "Stripe integration is done. ReadyCheck can now handle
                      subscriptions. Planning to launch on Product Hunt next month."
Episode 4 (Day 22): "Had to pivot ReadyCheck's pricing model. Free tier was too
                      generous. Switching to usage-based with Stripe metered billing."
Episode 5 (Day 30): "ReadyCheck soft launch went well. 15 signups on day one.
                      Got feedback to add Google Calendar integration."
```

Entities: ReadyCheck, Next.js, Supabase, Stripe, Product Hunt, Google Calendar
Relationships: builds, uses_tech, launched_on, integrates

**Thread 3: Health & Fitness (5 episodes)**
Tests concept associations and preference tracking.

```
Episode 1 (Day 1):  "Started researching peptide protocols. Interested in BPC-157
                      for tendon recovery."
Episode 2 (Day 5):  "Been doing zone 2 cardio 4x/week. Heart rate stays around
                      135 bpm. Also looking into retatrutide for body composition."
Episode 3 (Day 12): "Ordered BPC-157 from Amino Asylum. Starting 250mcg 2x daily
                      subcutaneous."
Episode 4 (Day 20): "BPC-157 seems to be helping with my elbow tendinopathy.
                      Also started creatine 5g/day."
Episode 5 (Day 28): "Body comp update: down 4lbs, strength maintained. Retatrutide
                      protocol working well alongside zone 2."
```

Entities: BPC-157, retatrutide, Amino Asylum, zone 2 cardio, creatine
Associations: peptides -> body_composition -> zone 2 -> heart rate

**Thread 4: Book & Learning (5 episodes)**
Tests concept graph density and cross-domain connections.

```
Episode 1 (Day 3):  "Reading 'Designing Data-Intensive Applications' by Kleppmann.
                      Chapter on replication is excellent."
Episode 2 (Day 9):  "Finished DDIA chapter on partitioning. Reminds me of how
                      FalkorDB handles graph sharding."
Episode 3 (Day 16): "Started 'The Mom Test' for ReadyCheck customer discovery.
                      Need to validate the meeting prep pain point."
Episode 4 (Day 23): "Interesting connection: DDIA's chapter on stream processing
                      is relevant to the event-driven architecture I'm planning
                      for ReadyCheck notifications."
Episode 5 (Day 29): "Watching Andrej Karpathy's neural net lectures. Want to
                      understand transformer attention for the Engram activation model."
```

Entities: DDIA, Kleppmann, The Mom Test, FalkorDB, Karpathy
Cross-domain links: DDIA -> ReadyCheck architecture, Karpathy -> Engram

**Thread 5: Social/People (5 episodes)**
Tests person entity resolution and relationship mapping.

```
Episode 1 (Day 4):  "Caught up with my friend Marcus. He's building a dev tool
                      startup too, called Stackblitz competitor."
Episode 2 (Day 11): "Marcus introduced me to Elena, a YC partner. She's interested
                      in dev tools space."
Episode 3 (Day 17): "Coffee with Elena went well. She suggested applying to YC
                      with ReadyCheck for W26 batch."
Episode 4 (Day 24): "Marcus's startup just raised a seed round. $2.5M from
                      Andreessen. Happy for him."
Episode 5 (Day 30): "Elena sent over YC application link. Deadline is March 15.
                      Need to prep the application with ReadyCheck metrics."
```

Entities: Marcus, Elena, YC, Andreessen, Stackblitz
Network: Marcus -> Elena -> YC -> ReadyCheck funding path

**Thread 6: Home & Location (4 episodes)**
Tests location entities and life event tracking.

```
Episode 1 (Day 2):  "Looking at apartments in SF now that the Vercel role is
                      remote-friendly from their SF office."
Episode 2 (Day 13): "Found a place in SOMA, 1BR for $3200. Close to the
                      Caltrain station."
Episode 3 (Day 19): "Signed the lease. Moving from Mesa to SF on March 1st."
Episode 4 (Day 27): "Started packing. Need to find a good gym in SOMA.
                      Currently at EOS Fitness in Mesa."
```

Temporal contradictions: lives_in (Mesa -> SF), gym (EOS Fitness -> TBD)

**Thread 7: Finance & Goals (4 episodes)**
Tests numeric fact tracking and goal entities.

```
Episode 1 (Day 6):  "Setting Q1 financial goals: save $15K, max out 401K
                      contributions, get ReadyCheck to $1K MRR."
Episode 2 (Day 15): "Vercel 401K match is 4%. Setting contribution to 10%
                      of base salary."
Episode 3 (Day 22): "ReadyCheck revenue: $340 MRR after soft launch.
                      Need to 3x to hit Q1 goal."
Episode 4 (Day 29): "Monthly savings: $4,200. On track for $15K Q1 target.
                      ReadyCheck at $520 MRR now."
```

Entities: Q1 goals, 401K, ReadyCheck MRR
Evolving facts: MRR ($0 -> $340 -> $520)

**Thread 8: AI/Tech Research (5 episodes)**
Tests technical concept clustering.

```
Episode 1 (Day 7):  "Exploring Anthropic's MCP protocol for Engram integration.
                      HTTP/SSE transport looks cleanest."
Episode 2 (Day 12): "Tested FalkorDB with 10K nodes. Query latency under 5ms
                      for 2-hop traversals. Good enough."
Episode 3 (Day 18): "Reading about spreading activation models in cognitive science.
                      Collins & Loftus 1975 paper is the foundation."
Episode 4 (Day 24): "Mem0 just shipped graph memory. Their approach: vector search
                      + keyword extraction. No activation dynamics."
Episode 5 (Day 28): "Graphiti's temporal model is solid. Their bi-temporal edges
                      (valid_time, transaction_time) is what we should use."
```

Entities: MCP, FalkorDB, Collins & Loftus, Mem0, Graphiti
Domain cluster: memory systems, graph databases, cognitive science

**Thread 9: Travel & Events (4 episodes)**
Tests event entities with dates and participants.

```
Episode 1 (Day 10): "Planning to attend AI Engineer World's Fair in June.
                      Want to demo Engram there."
Episode 2 (Day 16): "Marcus might join for the conference. Could share a booth."
Episode 3 (Day 21): "Booked flights to SF for the move. March 1 one-way.
                      Also booked hotel in Austin for SXSW in March."
Episode 4 (Day 26): "SXSW panel accepted: 'Building Memory for AI Agents'.
                      Co-presenting with the Mem0 founder."
```

Cross-links: Marcus, SF move, Engram demo, Mem0 founder

**Thread 10: Miscellaneous & Preferences (7 episodes)**
Tests preference tracking and low-frequency recall.

```
Episode 1 (Day 1):  "Prefer dark mode in all my tools. Use VS Code with the
                      GitHub Dark theme."
Episode 2 (Day 4):  "Allergic to shellfish. Important when eating out."
Episode 3 (Day 8):  "My dog's name is Pixel. She's a 2-year-old Australian Shepherd."
Episode 4 (Day 14): "Coffee preference: oat milk latte, no sugar. Usually from
                      Blue Bottle or Verve."
Episode 5 (Day 19): "Favorite podcast: Lex Fridman. Currently listening to the
                      episode with Ilya Sutskever."
Episode 6 (Day 23): "Birthday is October 15th. Turning 29 this year."
Episode 7 (Day 27): "Switched from VS Code to Cursor. The AI features are
                      significantly better."
```

Preferences: dark mode, shellfish allergy, coffee order, podcast
Temporal: IDE (VS Code -> Cursor)

### 1.3 Dataset Properties

The dataset intentionally creates these structures that stress-test retrieval:

| Property | Count | Purpose |
|----------|-------|---------|
| Temporal contradictions | 6+ | Test freshness-aware retrieval |
| Cross-thread entity references | 8+ | Test associative paths |
| Entity evolution (same entity, changed attributes) | 5+ | Test temporal resolution |
| High-frequency entities (mentioned 5+ times) | 4-5 | Test frequency weighting |
| Low-frequency but important entities | 3-4 | Test that rare facts are still retrievable |
| Multi-hop association paths | 5+ | Test spreading activation |
| Hub nodes (high-degree, e.g. "User") | 1-2 | Test hub dampening (sqrt-degree normalization) |
| Dense concept clusters (e.g. tech nodes) | 1-2 | Test energy budget enforcement during spread |

---

## 2. Evaluation Query Set

### 2.1 Query Categories

Queries are grouped into categories that test different retrieval capabilities. Each query has human-annotated ground truth: a ranked list of relevant entities/facts with relevance scores (3=highly relevant, 2=relevant, 1=marginally relevant).

#### Category A: Direct Recall (5 queries)
Baseline queries that any method should handle. These establish a floor.

| ID | Query | Ground Truth (ranked) | Notes |
|----|-------|----------------------|-------|
| Q1 | "Where do I work?" | Vercel (3), deployment pipeline team (2), Acme Corp (1, temporal: past) | Must return Vercel, not Acme |
| Q2 | "What tech stack is ReadyCheck built on?" | Next.js (3), Supabase (3), Stripe (2) | Direct fact lookup |
| Q3 | "What's my dog's name?" | Pixel (3), Australian Shepherd (2) | Low-frequency, exact recall |
| Q4 | "What books am I reading?" | DDIA (3), The Mom Test (3), Kleppmann (2) | Multiple results needed |
| Q5 | "When am I moving to SF?" | March 1 (3), SOMA apartment (2), Mesa (1, temporal: past) | Temporal fact |

#### Category B: Recency-Sensitive (4 queries)
Queries where the correct answer depends on temporal freshness. Pure vector search returns stale results.

| ID | Query | Ground Truth (ranked) | Why Activation Wins |
|----|-------|----------------------|---------------------|
| Q6 | "What IDE do I use?" | Cursor (3) | NOT VS Code -- must respect temporal override |
| Q7 | "What programming languages am I working with?" | TypeScript (3), Go (3) | NOT Python/React -- job changed |
| Q8 | "What's ReadyCheck's current revenue?" | $520 MRR (3) | NOT $340 -- must return latest |
| Q9 | "Where do I live?" | SF/SOMA (3) | NOT Mesa -- must respect move |

#### Category C: Associative Recall (5 queries)
Queries where the best answer requires following association paths across topics. This is where activation is expected to dominate.

| ID | Query | Ground Truth (ranked) | Association Path |
|----|-------|----------------------|------------------|
| Q10 | "How could my reading help with ReadyCheck?" | DDIA stream processing -> ReadyCheck notifications (3), The Mom Test -> customer discovery (3) | Book -> Project (cross-thread) |
| Q11 | "Who could help me get funding for ReadyCheck?" | Elena/YC (3), Marcus intro (2), YC W26 deadline March 15 (2) | People -> Funding -> Project |
| Q12 | "What's relevant to my Engram project from my research?" | Collins & Loftus spreading activation (3), Karpathy attention mechanisms (3), Graphiti bi-temporal model (2), FalkorDB perf results (2) | Research -> Project (multi-source) |
| Q13 | "What should I prepare for SXSW?" | Panel on AI memory (3), co-presenting with Mem0 founder (3), Mem0 graph memory competitive intel (2), Engram demo (2) | Event -> Project -> Competitor |
| Q14 | "How does my health routine connect to my work goals?" | Zone 2 -> sustained energy -> productivity (2), body comp -> confidence (1) | Weak but valid associations |

#### Category D: Frequency-Weighted (3 queries)
Queries where frequently-referenced entities should rank higher.

| ID | Query | Ground Truth (ranked) | Frequency Signal |
|----|-------|----------------------|-----------------|
| Q15 | "What are my top priorities right now?" | ReadyCheck (3, ~15 mentions), Vercel job (3, ~8 mentions), SF move (2, ~5 mentions), Health (2, ~5 mentions) | High-mention entities should dominate |
| Q16 | "What technologies am I most invested in?" | Next.js (3), Stripe (3), FalkorDB (2), TypeScript (2) | Frequency across multiple threads |
| Q17 | "Who are the most important people in my network?" | Sarah (2, career), Marcus (3, repeated mentions), Elena (3, high-value connection) | Mention frequency + relationship weight |

#### Category E: Contradiction-Aware (3 queries)
Queries that specifically test whether the system handles contradictory information correctly.

| ID | Query | Ground Truth (ranked) | Contradiction |
|----|-------|----------------------|---------------|
| Q18 | "Tell me about my career history" | Vercel (3, current), Acme Corp (2, past), Sarah (2, past manager), TypeScript/Go (2, current), Python/React (1, past) | Must present both with temporal ordering |
| Q19 | "What's my gym situation?" | EOS Fitness Mesa (2, past), looking for SOMA gym (3, current need) | Incomplete transition |
| Q20 | "Summarize ReadyCheck's pricing evolution" | Usage-based/metered (3, current), free tier too generous (2, lesson learned), Stripe metered billing (2, implementation) | Pivot captured |

### 2.2 Difficulty Classification

| Difficulty | Queries | Expected: Vector Wins | Expected: Activation Wins |
|-----------|---------|----------------------|--------------------------|
| Easy (direct recall) | Q1-Q5 | All methods ~equal | Marginal gain from recency on Q1, Q5 |
| Medium (recency/frequency) | Q6-Q9, Q15-Q17 | Fails on Q6-Q9 (stale results) | Wins via temporal decay + frequency |
| Hard (associative) | Q10-Q14 | Cannot follow multi-hop paths | Wins via spreading activation |
| Hard (contradiction) | Q18-Q20 | Returns all versions unranked | Wins via temporal ordering + recency |

---

## 3. Retrieval Methods Under Test

### 3.0 Shared Embedding Configuration

All methods use the same embedding model and vector index for fair comparison. Per the embedding strategy (`04_embedding_strategy.md`):

| Parameter | Value | Source |
|-----------|-------|--------|
| Embedding model | `voyage-3-lite` | Pinned for reproducibility; Voyage API is versioned by model name |
| Dimensions | 512 (float32, 2048 bytes/vector) | Matches model output |
| Asymmetric input types | `input_type="document"` for indexing, `input_type="query"` for retrieval | Voyage AI recommendation for asymmetric search |
| Distance metric | Cosine | Standard for text embeddings |
| Vector storage | Redis Search HNSW index | Single index, TAG-filtered by group_id |
| HNSW `m` | 16 | Default connectivity |
| HNSW `ef_construction` | 200 | Build-time accuracy |
| HNSW `ef_runtime` | 50 | Query-time search depth, > 95% recall |
| What gets embedded | Entity summaries + episode content | Both in same index with `content_type` TAG |
| Benchmark searches against | Entity embeddings only (`content_type=entity`) | Ground truth is entity-level; episode embeddings not used for this benchmark |
| Score normalization | `similarity = 1.0 - (cosine_distance / 2.0)` | Redis cosine distance [0,2] -> similarity [0,1] |
| Top-K candidates | 20 (Methods A, A2, C, D); top-N direct (Method B) | Method B returns top-N directly, no intermediate K |
| Batch ingestion | 2 API calls total: 50 episodes in batch 1, ~120 entities in batch 2 | Voyage supports 128 texts/batch |

**Access simulation**: Before running evaluation queries, simulate 2 weeks of access patterns to build realistic activation state. Entities are categorized as:
- **Frequent**: 10+ accesses over 14 days (ReadyCheck, Vercel, TypeScript)
- **Recent**: 1-3 accesses in last 2 days (Cursor, $520 MRR, SF/SOMA)
- **Dormant**: No access in last 7 days (BPC-157 early episodes, Acme Corp)

This prevents the benchmark from testing a cold-start scenario, which is not representative of real usage.

### 3.1 Method A: Engram Activation-Based Retrieval

The full Engram retrieval pipeline per the refined activation engine spec (`02_activation_engine.md`). Uses the ACT-R base-level learning equation for activation (recency + frequency encoded in a single signal) and three orthogonal scoring signals:

1. Embed query, find top-K candidate nodes via cosine similarity (K=20)
2. Compute base activation for each candidate via ACT-R formula: `B_i(t) = ln(sum(t - t_j)^{-d})`, normalized to [0,1] via sigmoid
3. Identify seed nodes (candidates with semantic_similarity >= 0.3)
4. Run spreading activation from seeds (default 2 hops, 0.5 decay per hop, sqrt(degree) normalization, energy budget 5.0, firing threshold 0.05)
5. Compute composite score with three orthogonal signals:
   - `score = semantic_sim * 0.50 + activation * 0.35 + edge_proximity * 0.15`
   - `activation` = ACT-R base-level activation + spreading bonus (clamped to [0,1])
   - `edge_proximity` = structural closeness to seed nodes (0.5^hops, 1.0 for seeds)
6. Return top-N by composite score

Key differences from original spec: recency and frequency are no longer separate signals -- both are encoded in the ACT-R activation formula. `edge_proximity` replaces them as the third signal, capturing graph structure independently of activation history.

Activation state (access_history timestamps) is maintained across the full episode ingestion sequence. Activation is computed lazily (no background decay sweep).

**Hop configuration**: Default is 2 hops. For queries requiring 3+ hop association paths (e.g., Q11: Marcus -> Elena -> YC -> ReadyCheck), also benchmark at `spread_max_hops=3` to measure the precision/noise tradeoff.

### 3.2 Method A2: Vector + ACT-R Activation, No Spreading (Ablation)

Isolates the value of ACT-R recency/frequency modeling from spreading activation and edge proximity:

1. Embed query, find top-K candidate nodes via cosine similarity (K=20)
2. Compute base activation for each candidate via ACT-R formula (same as Method A)
3. Compute composite score with **NO spreading activation and NO edge proximity**:
   - `score = semantic_sim * 0.50 + activation * 0.50 + edge_proximity * 0.0`
   - Equivalently: `weight_semantic=0.50, weight_activation=0.50, weight_edge_proximity=0.0, spread_energy_budget=0.0`
4. Return top-N by composite score

This ablation answers: "How much of the improvement comes from spreading activation vs. ACT-R decay modeling?" Comparing Method A vs A2 isolates spreading activation + edge proximity contribution. Comparing A2 vs B isolates ACT-R activation's contribution (recency + frequency). This decomposition is critical -- if A2 already captures most of the gain, spreading activation may not justify its complexity and latency cost.

### 3.3 Method B: Pure Vector Similarity (Baseline)

Cosine similarity on entity embeddings only. Uses the `vector_search_baseline()` function from the embedding strategy:

1. Embed query with `voyage-3-lite` using `input_type="query"`
2. KNN search against entity embeddings in Redis Search index (`content_type=entity`)
3. Return top-N directly by cosine similarity (no intermediate K, no re-ranking)
4. No activation state, no temporal awareness, no graph traversal

This is a genuinely strong baseline -- `voyage-3-lite` at 512d with HNSW (m=16, ef_runtime=50) gives >95% recall. There is no handicap. If Engram cannot beat this, the activation engine adds no value.

### 3.4 Method C: Mem0-Style Hybrid

Simulates Mem0's approach -- vector search augmented with keyword matching. Uses the `hybrid_search_entities()` function from the embedding strategy (Section 6, Pattern 3), which combines Redis Search KNN with BM25 on the `text` field:

1. Run vector KNN search against entity embeddings (top-K=40, `input_type="query"`)
2. Run BM25 keyword search against the `text` field in the same Redis Search index
3. Merge results with Reciprocal Rank Fusion (semantic_weight=0.7, keyword_weight=0.3)
4. No temporal awareness beyond timestamp sorting
5. No activation dynamics

### 3.5 Method D: Graph Traversal (Graphiti-Style)

Simulates Graphiti's approach -- graph-aware retrieval without activation dynamics:

1. Embed query, find top-K candidate nodes by cosine similarity
2. Expand candidates by 1-hop graph neighbors
3. Score by: `semantic_sim * 0.6 + graph_centrality * 0.2 + recency * 0.2` (recency here is simple `1/(1+hours_since_access)`, not ACT-R)
4. Temporal filtering: only return currently-valid facts (valid_to IS NULL)
5. No ACT-R activation dynamics, no frequency weighting, no spreading activation

---

## 4. Metrics

### 4.1 Retrieval Quality Metrics

**Precision@5**: Of the top 5 returned results, what fraction are relevant (relevance >= 2)?

```
P@5 = |relevant in top 5| / 5
```

**Recall@10**: Of all relevant results for the query, what fraction appear in the top 10?

```
R@10 = |relevant in top 10| / |total relevant|
```

**Mean Reciprocal Rank (MRR)**: Reciprocal of the rank of the first highly-relevant result (relevance = 3).

```
MRR = mean(1 / rank_of_first_highly_relevant) across all queries
```

**Normalized Discounted Cumulative Gain (nDCG@5)**: Measures ranking quality using graded relevance scores.

```
DCG@5 = sum(relevance_i / log2(i + 1)) for i in 1..5
nDCG@5 = DCG@5 / ideal_DCG@5
```

### 4.2 Latency Metrics

All latency measured end-to-end from query receipt to results returned.

| Metric | Description | Target |
|--------|-------------|--------|
| p50 latency | Median retrieval time | < 100ms |
| p95 latency | 95th percentile | < 250ms |
| p99 latency | 99th percentile | < 500ms |
| Activation overhead | Time added by activation scoring vs pure vector | < 50ms at p95 |

Latency measured with the full 50-episode graph loaded (expected ~100 nodes, ~200 edges).

### 4.3 Activation-Specific Metrics

| Metric | Description |
|--------|-------------|
| Activation spread coverage | How many nodes get non-zero activation from a query |
| Temporal accuracy | % of recency-sensitive queries (Q6-Q9) returning the latest fact |
| Association discovery | % of associative queries (Q10-Q14) returning cross-thread results |
| Frequency correlation | Spearman correlation between entity mention count and retrieval rank |

---

## 5. Scoring Rubric

### 5.1 Per-Query Scoring

Each query is scored against its ground truth annotations:

```python
def score_result(returned_results, ground_truth):
    """
    returned_results: list of (entity_id, score) tuples, ranked
    ground_truth: dict of {entity_id: relevance_score (1-3)}
    """
    precision_at_5 = len([r for r in returned_results[:5]
                          if r.entity_id in ground_truth
                          and ground_truth[r.entity_id] >= 2]) / 5

    recall_at_10 = len([r for r in returned_results[:10]
                        if r.entity_id in ground_truth]) / len(ground_truth)

    mrr = 0
    for i, r in enumerate(returned_results):
        if r.entity_id in ground_truth and ground_truth[r.entity_id] == 3:
            mrr = 1 / (i + 1)
            break

    return precision_at_5, recall_at_10, mrr
```

### 5.2 Aggregate Scoring

Report per-method:

```
Method                                              | P@5   | R@10  | MRR   | nDCG@5 | p50(ms) | p99(ms)
----------------------------------------------------|-------|-------|-------|--------|---------|--------
A: Engram Full (0.50/0.35/0.15, spreading ON)       |       |       |       |        |         |
A2: Vector + ACT-R (0.50/0.50/0.0, spreading OFF)   |       |       |       |        |         |
B: Pure Vector (1.0/0.0/0.0)                         |       |       |       |        |         |
C: Mem0-Style Hybrid (vector + BM25 RRF)             |       |       |       |        |         |
D: Graph Traversal (semantic + centrality + recency)  |       |       |       |        |         |
```

Also report ablation decomposition (the headline result):

```
Component Contribution                      | Delta P@5 vs B | % of Total Gain
--------------------------------------------|----------------|----------------
ACT-R decay modeling (A2 - B)               |                |
Spreading + edge proximity (A - A2)         |                |
Total activation system (A - B)             |                | 100%
```

Additionally, benchmark Method A at both `spread_max_hops=2` (default) and `spread_max_hops=3` to measure the precision/noise tradeoff for deeper association paths:

```
Hop Configuration     | P@5   | P@5 (Cat C only) | Noise (false positives in top-5)
----------------------|-------|-------------------|--------------------------------
A (2 hops, default)   |       |                   |
A (3 hops)            |       |                   |
```

Also report per-category breakdown:

```
Category              | Engram P@5 | Vector P@5 | Delta  | Significant?
----------------------|------------|------------|--------|-------------
A: Direct Recall      |            |            |        |
B: Recency-Sensitive  |            |            |        |
C: Associative Recall |            |            |        |
D: Frequency-Weighted |            |            |        |
E: Contradiction      |            |            |        |
```

### 5.3 Statistical Significance

With 20 queries, use paired bootstrap resampling (1000 iterations) to compute 95% confidence intervals on the P@5 difference between Engram and each baseline. Report whether the improvement is statistically significant at p < 0.05.

---

## 6. Test Cases ONLY Activation Can Solve

These are the critical queries where we expect activation to provide results that no other method can match. If activation does not win these, the hypothesis is wrong.

### 6.1 Associative Leap: "How could my reading help with ReadyCheck?" (Q10)

**Why vector fails**: The embedding for "reading" and "ReadyCheck" are semantically distant. No single episode mentions both DDIA stream processing AND ReadyCheck notifications.

**Why activation wins**: When ReadyCheck is high-frequency (many mentions), its activation is elevated. DDIA was mentioned in connection with "event-driven architecture I'm planning for ReadyCheck notifications" (Thread 4, Episode 4). Spreading activation from ReadyCheck -> architecture -> DDIA creates the bridge.

**Expected results from each method**:
- Pure vector: Returns Thread 4 Episodes about books, maybe Thread 2 about ReadyCheck. Does not connect them.
- Mem0 hybrid: Keyword "ReadyCheck" pulls project episodes. "Reading" pulls book episodes. RRF merges but does not rank the connection.
- Graph traversal: If DDIA and ReadyCheck are connected via intermediate nodes, might find the 1-hop path. But relies on the specific edge existing.
- Engram activation: ReadyCheck's elevated activation spreads to connected nodes including architecture concepts, which connect to DDIA. Returns the cross-domain insight.

### 6.2 Temporal Override: "What IDE do I use?" (Q6)

**Why vector fails**: VS Code appears in 2 episodes, Cursor in 1. By embedding similarity, VS Code is a stronger match.

**Why activation wins**: Cursor episode is more recent. The ACT-R power-law decay means Cursor's single recent access produces a larger `(t - t_j)^{-d}` term than VS Code's older accesses. Additionally, the temporal contradiction (switched from X to Y) sets `valid_to` on the "uses VS Code" edge -- spreading activation won't propagate through expired edges, so VS Code gets no associative boost.

### 6.3 Network Path Discovery: "Who could help me get funding?" (Q11)

**Why vector fails**: "Funding" as a query does not semantically match "Elena" or "Marcus" directly. The word "YC" might match, but the path Marcus -> introduction -> Elena -> YC partner -> suggested applying is a multi-hop reasoning chain.

**Why activation wins**: Elena and YC have been discussed recently (high ACT-R activation from recent accesses). Marcus's connection to Elena has edge_weight reinforced by multiple mentions. Spreading activation from "funding" -> YC -> Elena -> Marcus surfaces the full network path.

**Note**: This path is 3 hops, requiring `spread_max_hops=3`. At the default 2 hops, spreading from YC reaches Elena (1 hop) but not Marcus (2 hops from YC, 3 from the funding seed). This query is a key test case for the 2-hop vs 3-hop comparison.

### 6.4 Frequency-Weighted Priority: "What are my top priorities?" (Q15)

**Why vector fails**: "Priorities" is semantically vague. Returns a mix of everything. Cannot distinguish high-priority from low-priority based on embedding distance alone.

**Why activation wins**: ReadyCheck (~15 mentions across threads) has much higher ACT-R activation because each access adds a `(t - t_j)^{-d}` term to the sum. Fifteen accesses spread across 30 days produce a significantly higher `B_i(t)` than "shellfish allergy" with a single access. The activation signal (weighted 0.35) re-ranks ReadyCheck above low-frequency entities that may be equally semantically relevant to the vague query.

### 6.5 Competitive Intelligence Assembly: "What should I prepare for SXSW?" (Q13)

**Why vector fails**: "SXSW" appears in Thread 9. But the relevant competitive intel about Mem0 (Thread 8, Episode 4) requires the associative leap: SXSW panel -> co-presenting with Mem0 founder -> Mem0 just shipped graph memory.

**Why activation wins**: The Mem0 entity is connected to both the SXSW panel (Thread 9) and competitive research (Thread 8). Spreading activation from SXSW -> panel -> Mem0 -> competitive intel surfaces the prep material.

---

## 7. Go/No-Go Framework

### 7.1 Go Criteria (ALL must be met)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Overall P@5 improvement over pure vector | >= 15% | Minimum to justify activation complexity |
| Overall P@5 improvement over Mem0-style hybrid | >= 10% | Must beat the closest competitor approach |
| Category C (Associative) P@5 | >= 0.60 | Core value proposition must work |
| Category B (Recency) temporal accuracy | >= 75% | Must handle temporal contradictions |
| Activation overhead latency p95 | < 50ms | Cannot sacrifice UX for accuracy |
| At least 3 of 5 "activation-only" test cases won | Yes | Core hypothesis validated |

### 7.2 No-Go: Pivot Plans

If we fail go criteria, we do NOT abandon the project. We pivot the activation approach.

**Pivot A: If activation overhead is too high but quality is good**
- ACT-R computation is O(candidates * max_history_size) = O(50 * 200) = 10K float ops -- unlikely to be the bottleneck
- If spreading is slow on dense subgraphs, reduce `spread_energy_budget` from 5.0 to 2.0 (halves the frontier)
- Cache spreading activation results with TTL for repeated similar queries
- Re-benchmark with reduced budget

**Pivot B: If quality improvement is < 15% overall but associative queries win**
- Simplify to activation-only-on-associative mode
- Use pure vector for direct recall (Categories A, D)
- Use activation only when query is classified as associative
- Reduces complexity while keeping the differentiating feature

**Pivot C: If quality improvement is < 10% everywhere**
- Activation hypothesis is invalidated for this dataset size
- Pivot Engram's differentiator to: visualization + temporal graph (not activation)
- Simplify retrieval to graph-aware vector search (Method D without activation)
- Reposition as "best-in-class memory visualization" rather than "novel retrieval"
- Save 2+ weeks of activation engine development time

**Pivot D: If recency/contradiction handling fails**
- Separate temporal resolution from activation
- Implement explicit temporal filtering as a pre-retrieval step
- Keep activation for frequency and association only

### 7.3 Decision Timeline

```
Day 1-2: Implement dataset ingestion, build graph
Day 3-4: Implement all 5 retrieval methods (A, A2, B, C, D)
Day 5:   Run benchmark, collect results
Day 6:   Analyze results, write report
Day 7:   Go/no-go decision meeting
```

Results documented in `tests/benchmarks/results/` with full reproducibility.

---

## 8. Unit Economics Model

### 8.1 Cost Per Episode Ingestion

Ingestion uses Claude API for entity extraction.

| Component | Tokens (est.) | Cost (Claude Haiku 4.5) | Cost (Claude Sonnet 4.6) |
|-----------|--------------|------------------------|-------------------------|
| System prompt (entity extraction) | ~800 input | $0.0008 | $0.0024 |
| Episode text (avg) | ~200 input | $0.0002 | $0.0006 |
| Structured output (entities + rels) | ~400 output | $0.0004 | $0.0012 |
| **Total per episode** | **~1,400 tokens** | **$0.0014** | **$0.0042** |

Embedding cost (assuming Voyage AI or similar):
| Component | Cost |
|-----------|------|
| Embed entity summary (~100 tokens) | ~$0.00001 per entity |
| Embed episode text (~200 tokens) | ~$0.00002 per episode |
| Avg 3 entities per episode | ~$0.00005 |

**Total ingestion cost per episode: ~$0.0015 (Haiku) / ~$0.0043 (Sonnet)**

### 8.2 Cost Per Retrieval

| Component | Cost | Notes |
|-----------|------|-------|
| Query embedding | ~$0.00002 | Single embedding call |
| Vector search (self-hosted) | $0 | FalkorDB/local vector store |
| Activation computation | $0 | CPU only, no API calls |
| Redis activation state reads | $0 | Self-hosted |
| **Total per retrieval** | **~$0.00002** | Negligible |

### 8.3 Infrastructure Costs (Self-Hosted)

| Component | Resource | Monthly Cost |
|-----------|----------|-------------|
| FalkorDB (Redis module) | 512MB RAM | Included in Redis |
| Redis (activation state) | 256MB RAM | $0 (local Docker) |
| Vector store | Disk-based, <1GB | $0 (local) |
| Total self-hosted | Single machine, 2GB RAM | $0 (user's machine) |

### 8.4 Projected Monthly Cost Per Active User (Hosted SaaS)

Assumptions for an active user:
- 30 conversations/month, avg 10 episodes per conversation = 300 episodes/month
- 50 retrievals/day = 1,500 retrievals/month
- Graph size: ~500 entities, ~1,000 edges after 3 months

| Component | Monthly Cost |
|-----------|-------------|
| Episode ingestion (300 x $0.0015 Haiku) | $0.45 |
| Embeddings (300 episodes + 900 entities) | $0.03 |
| Retrievals (1,500 x $0.00002) | $0.03 |
| FalkorDB hosting (shared, per-user allocation) | $1.50 |
| Redis hosting (shared, per-user allocation) | $0.50 |
| Compute (FastAPI, shared) | $1.00 |
| **Total per user/month** | **~$3.51** |

At $15/month pro tier: **~77% gross margin** (using Haiku for extraction).

### 8.5 Break-Even Analysis

| Metric | Value |
|--------|-------|
| Fixed infrastructure (base cluster) | ~$150/month |
| Variable cost per user | ~$3.51/month |
| Revenue per pro user | $15/month |
| Contribution margin per user | $11.49/month |
| Users to break even on infrastructure | ~14 paying users |
| Free tier cost (if 100 free users, 50 eps/month each) | ~$22.50/month |

---

## 9. Reproducibility

### 9.1 File Structure

```
tests/benchmarks/
  dataset/
    episodes.json          # 50 episodes with metadata
    ground_truth.json      # 20 queries with relevance annotations
    threads.json           # Thread metadata and ordering
    access_patterns.json   # Simulated 2-week access history per entity
  methods/
    engram_retrieval.py    # Method A: full activation pipeline
    vector_retrieval.py    # Method B: pure cosine similarity
    hybrid_retrieval.py    # Method C: Mem0-style RRF
    graph_retrieval.py     # Method D: graph traversal
  run_benchmark.py         # Single-command runner
  analyze_results.py       # Generate tables and charts
  results/                 # Output directory (gitignored except summary)
    summary.md             # Human-readable results
    raw_scores.json        # Per-query, per-method scores
    latency.json           # Timing data
```

### 9.2 Single-Command Runner

```bash
# Run the full benchmark suite
python -m tests.benchmarks.run_benchmark

# Options
python -m tests.benchmarks.run_benchmark --methods engram,vector  # subset
python -m tests.benchmarks.run_benchmark --queries Q10,Q11,Q12    # subset
python -m tests.benchmarks.run_benchmark --iterations 3           # avg over N runs
python -m tests.benchmarks.run_benchmark --output results/run_01  # custom output dir
```

### 9.3 Runner Implementation Outline

```python
"""
run_benchmark.py -- single-command benchmark runner

1. Load dataset (episodes.json, ground_truth.json, access_patterns.json)
2. Initialize graph store (fresh FalkorDB + Redis instance via Docker)
3. Initialize embedding provider (voyage-3-lite, 512d, pinned version)
4. Ingest all 50 episodes in chronological order
   - Entity extraction via Claude API
   - Embed entity summaries + episode content via Voyage AI
   - Store vectors in Redis Search HNSW index
   - Build activation state incrementally
5. Simulate 2 weeks of access patterns (access_patterns.json)
   - Frequent entities: 10+ accesses over 14 days
   - Recent entities: 1-3 accesses in last 2 days
   - Dormant entities: no access in last 7 days
   - This builds realistic activation state before evaluation
6. For each retrieval method (A, A2, B, C, D):
   a. For each query:
      - Record start time
      - Execute retrieval (top 10 results)
      - Record end time
      - Score against ground truth
   b. Compute aggregate metrics
7. Run paired bootstrap (1000 iterations) for significance testing
8. Compute ablation decomposition (A vs A2 vs B)
9. Write results to output directory
10. Print summary table to stdout
"""
```

### 9.4 Environment Requirements

```
- Python 3.11+
- Docker (for FalkorDB + Redis with RediSearch module)
- Claude API key (for entity extraction during ingestion, uses Haiku 4.5)
- Voyage AI API key (for vector embeddings, voyage-3-lite, 512 dimensions)
- ~2GB RAM, ~1GB disk
- Estimated runtime: ~10 minutes (mostly API calls for extraction + embedding)
```

### 9.5 Determinism

- Embedding model and version are pinned in config
- Random seeds set for any stochastic components
- Activation decay computed to a fixed reference timestamp (not wall clock)
- Results include git commit hash of the benchmark code

---

## 10. Benchmark Schedule (Week 3)

| Day | Task |
|-----|------|
| Mon | Write dataset JSON files. Implement episode ingestion into graph. |
| Tue | Implement all 4 retrieval methods against the live graph. |
| Wed | Implement scoring framework and single-command runner. |
| Thu | Run benchmark. Debug any issues. Re-run for clean results. |
| Fri | Analyze results. Write summary. Make go/no-go recommendation. |

---

## 11. Success Criteria Summary

The benchmark answers one question: **Does spreading activation meaningfully improve memory retrieval for personal AI agents?**

"Meaningfully" means:
1. 15%+ improvement on precision@5 over pure vector search
2. Clear wins on the associative and recency query categories
3. Acceptable latency overhead (< 50ms at p95)
4. At least 3 of 5 "activation-only" test cases resolved correctly

If yes: proceed to Weeks 4-8 with confidence.
If no: pivot per Section 7.2 within 48 hours and save weeks of misallocated effort.
