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
| Elapsed time | 13518s |

## Overall Results

**Overall accuracy: 16.0%** (80/500)

**Category accuracy (official): 13.1%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 64 | 7 | 10.9% | 5918ms | 39.1% | 29.8% |
| single-session-assistant | 56 | 24 | 42.9% | 5961ms | 82.1% | 82.1% |
| single-session-preference | 30 | 3 | 10.0% | 6697ms | 33.3% | 22.5% |
| multi-session | 121 | 8 | 6.6% | 6229ms | 43.4% | 41.9% |
| temporal-reasoning | 127 | 7 | 5.5% | 5937ms | 39.8% | 39.8% |
| knowledge-update | 72 | 2 | 2.8% | 5286ms | 50.0% | 50.5% |
| abstention | 30 | 29 | 96.7% | 5560ms | 32.5% | 25.2% |

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
| Engram (narrow/auto) | 13.1% **<--** |

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
| Total ingest time | 10176.2s |
| Total query time | 2968.5s |

## Error Analysis

- **Best category**: abstention (96.7%)
- **Worst category**: knowledge-update (2.8%)

### Sample Errors (5 of 420)

**gpt4_2655b836** (temporal-reasoning)
- Q: What was the first issue I had with my new car after its first service?
- Gold: GPS system not functioning correctly
- Predicted: # Analysis of Evidence

**NOTE on Evidence [1]:** User mentions getting car serviced for the first time on March 15th during a conversation on 2023/04/10. However, this evidence does NOT contain infor
- Evidence retrieved: 6 items, 3 entities, 3 episodes

**gpt4_2487a7cb** (temporal-reasoning)
- Q: Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?
- Gold: 'Data Analysis using Python' webinar
- Predicted: # Analysis of Evidence

## NOTE on Relevance:

**Evidence [1], [2], [3]**: Discussions about car detailing, waxing, and road trip planning from April 2023. Not relevant to the question about workshops
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_76048e76** (temporal-reasoning)
- Q: Which vehicle did I take care of first in February, the bike or the car?
- Gold: bike
- Predicted: **NOTE on each piece of evidence:**

[1] Discusses car service on March 15th - NOT relevant (March, not February)
[2] General detailing advice - NOT relevant (no February dates mentioned)
[3] Mentions
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**gpt4_2312f94c** (temporal-reasoning)
- Q: Which device did I got first, the Samsung Galaxy S22 or the Dell XPS 13?
- Gold: Samsung Galaxy S22
- Predicted: # Analysis of Evidence

**NOTE ON RELEVANCE:**

Evidence [5] is highly relevant. It contains a conversation from 2023/03/15 (Wed) 10:31 where the user discusses their new Dell XPS 13 laptop and Samsun
- Evidence retrieved: 8 items, 3 entities, 5 episodes

**0bb5a684** (temporal-reasoning)
- Q: How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?
- Gold: 7 days. 8 days (including the last day) is also acceptable.
- Predicted: # Step 1 — Analysis of Evidence Relevance

**Evidence [1], [2], [3]**: About car detailing and road trips - NOT RELEVANT to the question about a workshop and team meeting.

**Evidence [4]**: About tas
- Evidence retrieved: 9 items, 3 entities, 6 episodes
