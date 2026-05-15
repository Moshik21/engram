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
| Elapsed time | 3957s |

## Overall Results

**Overall accuracy: 23.0%** (115/500)

**Category accuracy (official): 23.0%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 23 | 35.9% | 4146ms | 60.9% | 46.6% |
| single-session-assistant | 56 | 47 | 83.9% | 4878ms | 100.0% | 99.1% |
| single-session-preference | 30 | 2 | 6.7% | 4898ms | 20.0% | 14.0% |
| multi-session | 121 | 5 | 4.1% | 4626ms | 50.2% | 45.1% |
| temporal-reasoning | 127 | 9 | 7.1% | 4216ms | 58.0% | 55.5% |
| knowledge-update | 72 | 0 | 0.0% | 3519ms | 63.2% | 60.1% |
| abstention | 30 | 29 | 96.7% | 4276ms | 37.2% | 22.5% |

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
| Engram (narrow/auto) | 23.0% **<--** |
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
| Total ingest time | 785.7s |
| Total query time | 2162.3s |

## Error Analysis

- **Best category**: abstention (96.7%)
- **Worst category**: knowledge-update (0.0%)

### Sample Errors (5 of 385)

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: ## Analysis of Evidence

**NOTE on Evidence [1]:** This evidence contains a conversation from 2023/04/10 about car detailing, gas mileage, rewards programs, and credit card redemptions. However, it do
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_76048e76** (temporal-reasoning)
- Q: Which vehicle did I take care of first in February, the bike or the car?
- Gold: bike
- Predicted: NOTE ON EVIDENCE RELEVANCE:

[1] This evidence is from a conversation dated 2023/04/10, which is AFTER the question's reference date of 2023/03/10. The conversation discusses car service on March 15th
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: **NOTE ANALYSIS:**

[1] This evidence discusses car maintenance, gas mileage, rewards programs, and credit card redemptions. It contains conversations about car detailing, waxing, gas station rewards,
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**08f4fc43** (temporal-reasoning)
- Q: How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?
- Gold: 30 days. 31 days (including the last day) is also acceptable.
- Predicted: **NOTE on Evidence [1]:**
This conversation from 2023/04/10 discusses car detailing, gas mileage, rewards programs, and credit card redemptions. It mentions a car service on March 15th and a GPS syste
- Evidence retrieved: 9 items, 3 entities, 6 episodes

**2c63a862** (temporal-reasoning)
- Q: How many days did it take for me to find a house I loved after starting to work with Rachel?
- Gold: 14 days. 15 days (including the last day) is also acceptable.
- Predicted: **NOTE on Evidence [1]:**
This evidence discusses car detailing, gas mileage, rewards programs, and credit card redemptions. It contains a conversation dated 2023/04/10 about car maintenance and acces
- Evidence retrieved: 9 items, 3 entities, 6 episodes
