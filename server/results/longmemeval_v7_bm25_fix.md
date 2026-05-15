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
| Total instances | 500 |
| Elapsed time | 12s |

## Overall Results

**Overall accuracy: 0.0%** (0/500)

**Category accuracy (official): 0.0%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| single-session-assistant | 56 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| single-session-preference | 30 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| multi-session | 121 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| temporal-reasoning | 127 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| knowledge-update | 72 | 0 | 0.0% | 0ms | 0.0% | 0.0% |
| abstention | 30 | 0 | 0.0% | 0ms | 0.0% | 0.0% |

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

### Sample Errors (5 of 500)

**gpt4_2655b836** (temporal-reasoning)
- Q: What was the first issue I had with my new car after its first service?
- Gold: GPS system not functioning correctly
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**gpt4_76048e76** (temporal-reasoning)
- Q: Which vehicle did I take care of first in February, the bike or the car?
- Gold: bike
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes

**0bb5a684** (temporal-reasoning)
- Q: How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?
- Gold: 7 days. 8 days (including the last day) is also acceptable.
- Predicted: [ERROR]
- Evidence retrieved: 0 items, 0 entities, 0 episodes
