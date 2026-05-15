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
| Elapsed time | 5284s |

## Overall Results

**Overall accuracy: 20.2%** (101/500)

**Category accuracy (official): 19.8%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 13 | 20.3% | 1989ms | 39.1% | 29.8% |
| single-session-assistant | 56 | 32 | 57.1% | 2271ms | 85.7% | 85.7% |
| single-session-preference | 30 | 7 | 23.3% | 3404ms | 43.3% | 26.2% |
| multi-session | 121 | 12 | 9.9% | 2494ms | 45.6% | 43.9% |
| temporal-reasoning | 127 | 7 | 5.5% | 2604ms | 39.8% | 39.8% |
| knowledge-update | 72 | 2 | 2.8% | 2283ms | 49.3% | 49.6% |
| abstention | 30 | 28 | 93.3% | 2077ms | 37.5% | 27.5% |

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
| Engram (narrow/auto) | 19.8% **<--** |
| Best guess (no context) | 18.8% |

## Adapter Statistics

| Metric | Value |
|---|---|
| Sessions ingested | 948 |
| Episodes stored | 948 |
| Episodes extracted | 948 |
| Extraction calls | 948 |
| Embedding calls | 0 |
| Recall calls | 500 |
| Reader calls | 500 |
| Total ingest time | 3675.7s |
| Total query time | 1215.9s |

## Error Analysis

- **Best category**: abstention (93.3%)
- **Worst category**: knowledge-update (2.8%)

### Sample Errors (5 of 399)

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: Based on the retrieved memories, there is no evidence of you attending either an 'Effective Time Management' workshop or a 'Data Analysis using Python' webinar.

The conversations show:
- Discussion a
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: Based on the evidence, the **Dell XPS 13 came first**.

**Temporal reasoning:**
- Dell XPS 13: Pre-ordered on January 28th, arrived on February 25th
- Samsung Galaxy S22: Described as "new" in the con
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**0bb5a684** (temporal-reasoning)
- Q: How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?
- Gold: 7 days. 8 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories provided.

The evidence does not contain any information about:
- A team meeting you were preparing for
- A workshop on 'Effective Commun
- Evidence retrieved: 11 items, 5 entities, 6 episodes

**08f4fc43** (temporal-reasoning)
- Q: How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?
- Gold: 30 days. 31 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories. The evidence shows that the user attended an Ash Wednesday service at the cathedral on February 1st, but there is no mention of a Sunday
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**2c63a862** (temporal-reasoning)
- Q: How many days did it take for me to find a house I loved after starting to work with Rachel?
- Gold: 14 days. 15 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories provided. The conversations in the evidence do not contain any information about working with someone named Rachel to find a house, or any
- Evidence retrieved: 11 items, 5 entities, 6 episodes
