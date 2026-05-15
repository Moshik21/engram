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
| Elapsed time | 9281s |

## Overall Results

**Overall accuracy: 15.6%** (78/500)

**Category accuracy (official): 14.5%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 5 | 7.8% | 7620ms | 39.1% | 30.1% |
| single-session-assistant | 56 | 25 | 44.6% | 7429ms | 82.1% | 82.1% |
| single-session-preference | 30 | 7 | 23.3% | 8924ms | 36.7% | 25.0% |
| multi-session | 121 | 7 | 5.8% | 8232ms | 44.3% | 42.4% |
| temporal-reasoning | 127 | 5 | 3.9% | 10661ms | 38.4% | 38.9% |
| knowledge-update | 72 | 1 | 1.4% | 7255ms | 49.3% | 49.3% |
| abstention | 30 | 28 | 93.3% | 7028ms | 32.5% | 25.2% |

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
| Engram (narrow/auto) | 14.5% **<--** |

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
| Total ingest time | 4644.2s |
| Total query time | 4254.6s |

## Error Analysis

- **Best category**: abstention (93.3%)
- **Worst category**: knowledge-update (1.4%)

### Sample Errors (5 of 422)

**gpt4_2655b836** (temporal-reasoning)
- Q: What was the first issue I had with my new car after its first service?
- Gold: GPS system not functioning correctly
- Predicted: # Analysis of Evidence

**NOTE on each piece of evidence:**

[1] Date: 2026-03-15 (conversation from 2023/04/10 17:50) - User mentions getting car serviced for the first time on March 15th and discuss
- Evidence retrieved: 6 items, 3 entities, 3 episodes

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: # Analysis of Evidence:

**NOTE [1]**: Discusses car detailing and mentions car service on March 15th. Not relevant to the question about workshops/webinars.

**NOTE [2]**: Discusses car detailing tip
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_76048e76** (temporal-reasoning)
- Q: Which vehicle did I take care of first in February, the bike or the car?
- Gold: bike
- Predicted: ## NOTE on Evidence Relevance

- **Evidence [1], [2], [3], [6], [7], [8]**: Discuss car detailing, waxing, and car service in April 2023, or data visualization resources—not relevant to February vehic
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: # Analysis of Evidence

**NOTE on Evidence [1]**: Car detailing conversation from April 10, 2023 — not relevant to the question about Samsung Galaxy S22 or Dell XPS 13.

**NOTE on Evidence [2]**: Car
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**0bb5a684** (temporal-reasoning)
- Q: How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?
- Gold: 7 days. 8 days (including the last day) is also acceptable.
- Predicted: # Analysis of Evidence

**NOTE on Evidence [1], [2], [3], [7], [8], [9]:** These discuss car detailing, car service, road trips, and car purchases. They are NOT relevant to the question about a worksh
- Evidence retrieved: 9 items, 3 entities, 6 episodes
