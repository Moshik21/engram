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
| Total instances | 6 |
| Elapsed time | 44s |

## Overall Results

**Overall accuracy: 33.3%** (2/6)

**Category accuracy (official): 20.0%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 1 | 1 | 100.0% | 2960ms | 100.0% | 100.0% |
| single-session-assistant | 1 | 0 | 0.0% | 2654ms | 100.0% | 100.0% |
| single-session-preference | 1 | 0 | 0.0% | 6243ms | 100.0% | 100.0% |
| multi-session | 1 | 0 | 0.0% | 2336ms | 50.0% | 63.7% |
| knowledge-update | 1 | 0 | 0.0% | 6585ms | 0.0% | 0.0% |
| abstention | 1 | 1 | 100.0% | 2516ms | 100.0% | 100.0% |

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
| Engram (auto/local) | 20.0% **<--** |
| Best guess (no context) | 18.8% |

## Adapter Statistics

| Metric | Value |
|---|---|
| Sessions ingested | 11 |
| Episodes stored | 11 |
| Episodes extracted | 11 |
| Extraction calls | 11 |
| Embedding calls | 0 |
| Recall calls | 6 |
| Reader calls | 6 |
| Total ingest time | 15.4s |
| Total query time | 23.3s |

## Error Analysis

- **Best category**: single-session-user (100.0%)
- **Worst category**: single-session-assistant (0.0%)

### Sample Errors (4 of 4)

**4d6b87c8** (knowledge-update)
- Q: How many titles are currently on my to-watch list?
- Gold: 25
- Predicted: I don't know.

The retrieved memories contain recommendations for shows like Poldark, Outlander, and references to Victoria and a show about the Wars of the Roses, but they don't specify what is actua
- Evidence retrieved: 7 items, 7 entities, 0 episodes

**gpt4_d84a3211** (multi-session)
- Q: How much total money have I spent on bike-related expenses since the start of the year?
- Gold: $185
- Predicted: I don't know.

The retrieved memories don't contain any information about your bike-related expenses or spending history. To find this information, you would need to check your receipts, bank statemen
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**28bcfaac** (single-session-assistant)
- Q: I'm going back to our previous conversation about music theory. You mentioned some online resources for learning music theory. Can you remind me of the website you recommended for free lessons and exercises?
- Gold: MusicTheory.net
- Predicted: I don't know.

While the retrieved memories do mention online platforms like Udemy and Coursera, as well as YouTube channels (Rick Beato and Adam Neely), none of these are described as offering free
- Evidence retrieved: 9 items, 7 entities, 2 episodes

**caf03d32** (single-session-preference)
- Q: I've been struggling with my slow cooker recipes. Any advice on getting better results?
- Gold: The user would prefer responses that provide tips and advice specifically tailored to their slow cooker experiences, utilizing their recent success with beef stew and interest in making yogurt in the slow cooker. They might not prefer general slow cooker recipes or advice unrelated to their specific experiences and interests.
- Predicted: Based on your interests in slow cooker cooking, here's some practical advice for better results:

**Key Tips:**

1. **Use Low Heat**: Set your slow cooker to a low heat setting to prevent scorching or
- Evidence retrieved: 9 items, 7 entities, 2 episodes
