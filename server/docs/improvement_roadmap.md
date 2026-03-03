# Engram Retrieval Improvement Roadmap

Deferred items from round-table analysis. These require significant effort or research and are tracked here for future implementation.

## Deferred Improvements

| Priority | Item | Description | Effort | Dependencies |
|----------|------|-------------|--------|--------------|
| 1 | **LoCoMo Benchmark** | Run Engram against LoCoMo/LongMemEval datasets for industry-standard comparability. Requires adapter layer to translate between benchmark format and Engram's MCP tools. | Large | Benchmark framework (done) |
| 2 | **Echo Chamber Benchmark** | 200+ sequential queries with `record_access()` to measure coverage drift and Gini coefficient over time. Tests whether activation creates filter bubbles that narrow retrieval diversity. | Large | Exploration bonus improvements |
| 3 | **Thompson Sampling Exploration** | Replace UCB1-inspired exploration bonus with posterior-based Thompson Sampling using implicit feedback from retrieval results. Would enable adaptive exploration that learns per-entity exploration rates. | Research | Current exploration bonus stable |
| 4 | **Personalized PageRank** | Replace BFS-based spreading activation with Personalized PageRank (PPR) for smoother graph relevance distribution. PPR naturally handles varying graph densities and provides a principled decay from seed nodes. | Medium | Spreading activation refactor |
| 5 | **Typed Edge Weighting** | Assign different propagation weights per predicate type during spreading activation. E.g., `WORKS_AT` edges propagate more strongly than `MENTIONED_WITH`. Requires calibration data from real usage patterns. | Medium | Edge proximity enabled |

## Implementation Notes

### LoCoMo Benchmark
- Dataset: ~500 multi-turn conversations with ground-truth memory queries
- Metrics: exact match, F1, BERTScore
- Need adapter: LoCoMo's "memory" format → Engram episodes + entities
- Estimated: 2-3 weeks including adapter and analysis

### Echo Chamber Benchmark
- Simulate 200+ sequential user sessions
- After each session, measure:
  - Coverage: fraction of corpus entities retrievable at P@5 > 0
  - Gini coefficient of access counts
  - Top-10 entity stability (Jaccard similarity between consecutive snapshots)
- Go/no-go: Gini < 0.7, coverage > 40%

### Thompson Sampling
- Model each entity's "relevance probability" as Beta(alpha, beta)
- On retrieval: sample from posterior, use as exploration weight
- On positive feedback (user clicked/used): alpha += 1
- On negative feedback (retrieved but ignored): beta += 1
- Requires defining "feedback signal" — implicit from access patterns?

### Personalized PageRank
- Algorithm: Power iteration with restart probability alpha = 0.15
- Seed vector: current search results (weighted by semantic similarity)
- Convergence: typically 10-20 iterations
- Advantage over BFS: no hard hop cutoff, handles dense subgraphs gracefully
- Library options: networkit, graph-tool, or custom sparse matrix implementation

### Typed Edge Weighting
- Default weight matrix (calibrate from usage data):
  - WORKS_AT: 0.8
  - EXPERT_IN: 0.9
  - USES: 0.6
  - KNOWS: 0.5
  - MENTIONED_WITH: 0.3
  - RELATED_TO: 0.4
- Store as config, allow per-deployment customization
- Requires changes to `get_active_neighbors_with_weights()` in both SQLite and FalkorDB stores
