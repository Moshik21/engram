# LongMemEval Benchmark Report (ORACLE)

## Configuration

| Setting | Value |
|---|---|
| Dataset variant | oracle |
| Extraction mode | narrow |
| Embedding provider | auto |
| Consolidation | no |
| Reader model | claude-haiku-4-5-20251001 |
| Judge model | claude-haiku-4-5-20251001 |
| Total instances | 30 |
| Elapsed time | 603s |

## Overall Results

**Overall accuracy: 0.0%** (0/30)

**Category accuracy (official): 0.0%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 4 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| single-session-assistant | 5 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| single-session-preference | 5 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| multi-session | 5 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| temporal-reasoning | 5 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| knowledge-update | 5 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| abstention | 1 | 0 | 0.0% | 0ms | 0.0% | 0.0% |

## Comparison with Published Baselines

| System | Accuracy |
|---|---|
| Observational Memory (gpt-5-mini) | 94.9% |
| EmergenceMem Internal | 86.0% |
| Observational Memory (gpt-4o) | 84.2% |
| EmergenceMem Simple | 82.4% |
| Oracle GPT-4o | 82.4% |
| Supermemory | 81.6% |
| TiMem (GPT-4o-mini) | 76.9% |
| Zep/Graphiti | 71.2% |
| Full-context GPT-4o | 60.2% |
| Naive RAG | 52.0% |
| Best guess (no context) | 18.8% |
| Engram (narrow/auto) | 0.0% **<--** |

## Adapter Statistics

| Metric | Value |
|---|---|
| Sessions ingested | 0 |
| Episodes stored | 0 |
| Episodes extracted | 0 |
| Extraction calls | 0 |
| Embedding calls | 0 |
| Recall calls | 0 |
| Reader calls | 0 |
| Total ingest time | 0.0s |
| Total query time | 0.0s |

## Error Analysis

- **Best category**: single-session-user (0.0%)
- **Worst category**: single-session-user (0.0%)

### Sample Errors (5 of 30)

**4d6b87c8** (knowledge-update)
- Q: How many titles are currently on my to-watch list?
- Gold: 25
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**852ce960** (knowledge-update)
- Q: What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?
- Gold: $400,000
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**1cea1afa** (knowledge-update)
- Q: How many Instagram followers do I currently have?
- Gold: 600
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**18bc8abd** (knowledge-update)
- Q: What brand of BBQ sauce am I currently obsessed with?
- Gold: Kansas City Masterpiece
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**01493427** (knowledge-update)
- Q: How many new postcards have I added to my collection since I started collecting again?
- Gold: 25
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes
