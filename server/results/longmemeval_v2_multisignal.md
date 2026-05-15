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
| Elapsed time | 6663s |

## Overall Results

**Overall accuracy: 21.2%** (106/500)

**Category accuracy (official): 20.7%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 12 | 18.8% | 2484ms | 40.6% | 30.5% |
| single-session-assistant | 56 | 32 | 57.1% | 2778ms | 85.7% | 85.7% |
| single-session-preference | 30 | 8 | 26.7% | 4051ms | 46.7% | 29.7% |
| multi-session | 121 | 15 | 12.4% | 2990ms | 44.5% | 42.8% |
| temporal-reasoning | 127 | 8 | 6.3% | 3145ms | 39.6% | 39.2% |
| knowledge-update | 72 | 2 | 2.8% | 2890ms | 49.3% | 49.6% |
| abstention | 30 | 29 | 96.7% | 2697ms | 35.8% | 26.8% |

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
| Engram (narrow/auto) | 20.7% **<--** |
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
| Total ingest time | 4796.6s |
| Total query time | 1486.3s |

## Error Analysis

- **Best category**: abstention (96.7%)
- **Worst category**: knowledge-update (2.8%)

### Sample Errors (5 of 394)

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: I cannot answer this question based on the retrieved memories.

The evidence provided contains conversations about car detailing, road trip planning, task management apps, and data visualization reso
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: The Dell XPS 13 came first.

**Temporal reasoning:**
- Dell XPS 13: Pre-ordered on January 28th, arrived on February 25th
- Samsung Galaxy S22: Described as "new" in the conversation from March 15th,
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**0bb5a684** (temporal-reasoning)
- Q: How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?
- Gold: 7 days. 8 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories provided.

The evidence contains conversations about car detailing, road trips, task management, and data visualization, but there is no
- Evidence retrieved: 11 items, 5 entities, 6 episodes

**08f4fc43** (temporal-reasoning)
- Q: How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?
- Gold: 30 days. 31 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories.

The evidence shows that the user attended an Ash Wednesday service at the cathedral on **February 1st** (from conversation [5]). However
- Evidence retrieved: 10 items, 5 entities, 5 episodes

**2c63a862** (temporal-reasoning)
- Q: How many days did it take for me to find a house I loved after starting to work with Rachel?
- Gold: 14 days. 15 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories. None of the conversations provided contain any information about you working with someone named Rachel or searching for a house. The memo
- Evidence retrieved: 11 items, 5 entities, 6 episodes
