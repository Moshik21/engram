# LongMemEval Benchmark Report (ORACLE)

## Configuration

| Setting | Value |
|---|---|
| Dataset variant | oracle |
| Extraction mode | narrow |
| Embedding provider | none |
| Consolidation | no |
| Reader model | claude-haiku-4-5-20251001 |
| Judge model | claude-haiku-4-5-20251001 |
| Total instances | 18 |
| Elapsed time | 15s |

## Overall Results

**Overall accuracy: 16.7%** (3/18)

**Category accuracy (official): 19.4%** (unweighted avg across question types)

## Per-Type Breakdown

| Question Type | Count | Correct | Accuracy | Avg Latency | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| single-session-user | 3 | 0 | 0.0% | 49ms | 100.0% | 100.0% |
| single-session-assistant | 3 | 2 | 66.7% | 34ms | 100.0% | 100.0% |
| single-session-preference | 3 | 0 | 0.0% | 50ms | 100.0% | 100.0% |
| multi-session | 2 | 1 | 50.0% | 28ms | 100.0% | 100.0% |
| temporal-reasoning | 2 | 0 | 0.0% | 63ms | 100.0% | 100.0% |
| knowledge-update | 3 | 0 | 0.0% | 436ms | 33.3% | 33.3% |
| abstention | 2 | 0 | 0.0% | 70ms | 50.0% | 50.0% |

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
| Engram (narrow/none) | 19.4% **<--** |
| Best guess (no context) | 18.8% |

## Adapter Statistics

| Metric | Value |
|---|---|
| Sessions ingested | 28 |
| Episodes stored | 28 |
| Episodes extracted | 28 |
| Extraction calls | 28 |
| Embedding calls | 0 |
| Recall calls | 18 |
| Reader calls | 18 |
| Total ingest time | 12.1s |
| Total query time | 2.0s |

## Error Analysis

- **Best category**: single-session-assistant (66.7%)
- **Worst category**: single-session-user (0.0%)

### Sample Errors (5 of 15)

**4d6b87c8** (knowledge-update)
- Q: How many titles are currently on my to-watch list?
- Gold: 25
- Predicted: British: It's a British period drama that follows the reign of Queen Victoria, from her ascension to the throne at 18 to her marriage to Prince Albert and beyond.; Here are a few more recommendations
- Evidence retrieved: 7 items, 7 entities, 0 episodes

**852ce960** (knowledge-update)
- Q: What was the amount I was pre-approved for when I got my mortgage from Wells Fargo?
- Gold: $400,000
- Predicted: Wells Fargo: User: I'm actually buying a $325,000 house, and I got pre-approved for $350,000 from Wells Fargo. Assuming: Assuming you're getting a conventional loan from Wells Fargo, here's a breakdow
- Evidence retrieved: 9 items, 7 entities, 2 episodes

**1cea1afa** (knowledge-update)
- Q: How many Instagram followers do I currently have?
- Gold: 600
- Predicted: Instagram Live: **Instagram Live**: Host an Instagram Live session to discuss the results, answer questions, and provide additional insights into your content strategy. Instagram Stories: User: I like
- Evidence retrieved: 7 items, 7 entities, 0 episodes

**80ec1f4f_abs** (multi-session)
- Q: How many different museums or galleries did I visit in December?
- Gold: 0. You did not mention visitng any museum in December
- Predicted: Many: **Local art schools and universities**: Many art schools and universities have galleries and exhibition spaces that are open to the public. Local: **Local art organizations and museums**: Look u
- Evidence retrieved: 10 items, 7 entities, 3 episodes

**60bf93ed** (multi-session)
- Q: How many days did it take for my laptop backpack to arrive after I bought it?
- Gold: 5 days. 6 days (including the last day) is also acceptable.
- Predicted: Congratulations: Assistant: Congratulations on taking the first step towards digital organization! 1/15: I bought it from Amazon on 1/15. Amazon: I bought it from Amazon on 1/15.
- Evidence retrieved: 9 items, 7 entities, 2 episodes
