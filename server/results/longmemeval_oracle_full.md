# LongMemEval Benchmark Report (ORACLE)

## Configuration

| Setting | Value |
|---|---|
| Dataset variant | oracle |
| Extraction mode | auto |
| Embedding provider | local |
| Consolidation | no |
| Reader model | claude-haiku-4-5-20251001 |
| Judge model | claude-haiku-4-5-20251001 |
| Total instances | 500 |
| Elapsed time | 6346s |

## Overall Results

**Overall accuracy: 16.2%** (81/500)

**Category accuracy (official): 9.0%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 1 | 1.6% | 6813ms | 1.6% | 1.6% |
| single-session-assistant | 56 | 1 | 1.8% | 4443ms | 1.8% | 1.8% |
| single-session-preference | 30 | 1 | 3.3% | 7752ms | 3.3% | 3.3% |
| multi-session | 121 | 21 | 17.4% | 8177ms | 33.9% | 36.4% |
| temporal-reasoning | 127 | 15 | 11.8% | 8013ms | 25.0% | 28.8% |
| knowledge-update | 72 | 13 | 18.1% | 9642ms | 26.4% | 27.0% |
| abstention | 30 | 29 | 96.7% | 16059ms | 30.8% | 32.2% |

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
| Engram (auto/local) | 9.0% **<--** |

## Adapter Statistics

| Metric | Value |
|---|---|
| Sessions ingested | 948 |
| Episodes stored | 948 |
| Episodes extracted | 390 |
| Extraction calls | 390 |
| Embedding calls | 0 |
| Recall calls | 500 |
| Reader calls | 177 |
| Total ingest time | 1876.2s |
| Total query time | 4100.3s |

## Error Analysis

- **Best category**: abstention (96.7%)
- **Worst category**: single-session-user (1.6%)

### Sample Errors (5 of 419)

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: Based on the evidence provided, I can only find information about one event with a specific date:

- **"Effective Time Management" workshop**: Attended last Saturday (which would be 2023/05/27, the da
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**gpt4_76048e76** (temporal-reasoning)
- Q: Which vehicle did I take care of first in February, the bike or the car?
- Gold: bike
- Predicted: Based on the evidence provided:

**The car came first.**

Temporal reasoning:
- **Bike repair**: Mid-February (approximate date mentioned in [1] and [2])
- **Car wash**: Monday, February 27th (specifi
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: Based on the evidence provided, the **Dell XPS 13 came first**.

**Temporal reasoning:** The Dell XPS 13 laptop arrived on **February 25th, 2023**. However, the evidence does not specify when you rece
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**08f4fc43** (temporal-reasoning)
- Q: How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?
- Gold: 30 days. 31 days (including the last day) is also acceptable.
- Predicted: I cannot answer this question based on the retrieved memories.

The evidence shows that the user attended an Ash Wednesday service at the cathedral on **February 1st**, but there is **no mention of a
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**2c63a862** (temporal-reasoning)
- Q: How many days did it take for me to find a house I loved after starting to work with Rachel?
- Gold: 14 days. 15 days (including the last day) is also acceptable.
- Predicted: I don't have enough information to answer this question. While the evidence shows you started working with Rachel on 2/15 (February 15, 2022), there is no date mentioned in the retrieved memories indi
- Evidence retrieved: 10 items, 7 entities, 3 episodes
