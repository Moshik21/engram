# Engram Benchmark Suite

- Track: `all`
- Mode: `quick`
- Seeds: `7`
- Generated: `2026-03-08T05:52:09+00:00`
- Output: `/Users/konnermoshier/Engram/server/.benchmarks/showcase/20260308-055209`
## Fairness Contract

- Track: `all`
- Strict fairness: `True`
- Transcript invariant: `True`
- Vector provider family: `local`
- Answer model: `deterministic`
- Answer provider: `deterministic`

| Scenario | Top-k | Evidence Tokens | Answer Tokens |
|---|---:|---:|---:|
| cue_delayed_relevance | 5 | 120 | 90 |
| temporal_override | 5 | 80 | 70 |
| prospective_trigger | 5 | 120 | 80 |
| cross_cluster_association | 5 | 120 | 70 |

## Executive Table

| Baseline | Available | Scenario Pass | False Recall | Temporal | Negation | Open Loop | Prospective | Answer Pass | Answer Score | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.750 | 0.750 | 4.299 | 4.688 |
| Context + Summary | yes | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.750 | 0.750 | 0.041 | 0.051 |
| Markdown Canonical | yes | 0.250 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.060 | 0.075 |
| Hybrid RAG Temporal | yes | 0.250 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 11.720 | 18.355 |

## Capability Scorecard

| Baseline | Association | Continuity | Cue | Graph | Prospective | Temporal |
|---|---:|---:|---:|---:|---:|---:|
| Engram Full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Context + Summary | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Markdown Canonical | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| Hybrid RAG Temporal | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Cue Delayed Relevance | Engram Full | Latent memory stayed available without carrying raw history in prompt. |
| Temporal Override | Engram Full | Current-state memory won without leaking stale or negated facts. |
| Prospective Trigger | Engram Full | Prospective retrieval surfaced the right intention from related entity activity. |
| Cross Cluster Association | Engram Full | Associative retrieval connected lexically distant but linked entities. |

## Scenario Matrix

### Cue Delayed Relevance

- Why it matters: Observed content can stay cheap until a later query makes it useful.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | 0.000 | 0.000 | 63.000 | 0.048 |
| Engram Full | 1.000 | 1.000 | 1.000 | 34.000 | 2.951 |
| Hybrid RAG Temporal | 1.000 | 1.000 | 1.000 | 96.000 | 11.720 |
| Markdown Canonical | 1.000 | 1.000 | 1.000 | 120.000 | 0.076 |

### Temporal Override

- Why it matters: Engram should preserve the latest fact instead of surfacing both versions.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | 1.000 | 1.000 | 63.000 | 0.025 |
| Engram Full | 1.000 | 1.000 | 1.000 | 59.000 | 2.754 |
| Hybrid RAG Temporal | 0.000 | 1.000 | 1.000 | 71.000 | 8.758 |
| Markdown Canonical | 0.000 | 1.000 | 1.000 | 80.000 | 0.038 |

### Prospective Trigger

- Why it matters: Intentions should fire from related entity activity rather than raw lexical overlap alone.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | 1.000 | 1.000 | 114.500 | 0.045 |
| Engram Full | 1.000 | 0.000 | 0.000 | 120.000 | 4.550 |
| Hybrid RAG Temporal | 0.000 | 1.000 | 1.000 | 120.000 | 8.194 |
| Markdown Canonical | 0.000 | 1.000 | 1.000 | 120.000 | 0.066 |

### Cross Cluster Association

- Why it matters: Graph-aware retrieval should outperform flat lexical retrieval on associative queries.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | 1.000 | 1.000 | 50.000 | 0.041 |
| Engram Full | 1.000 | 1.000 | 1.000 | 96.000 | 4.299 |
| Hybrid RAG Temporal | 0.000 | 1.000 | 1.000 | 91.000 | 18.904 |
| Markdown Canonical | 0.000 | 1.000 | 1.000 | 97.000 | 0.054 |

## Cost And Error Summary

| Baseline | False Recall | Token Efficiency | Tokens / Success | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Engram Full | 0.000 | 0.236 | 107.250 | 4.299 | 4.688 |
| Context + Summary | 0.600 | 0.272 | 0.000 | 0.041 | 0.051 |
| Markdown Canonical | 0.600 | 0.038 | 120.000 | 0.060 | 0.075 |
| Hybrid RAG Temporal | 0.600 | 0.111 | 96.000 | 11.720 | 18.355 |

## Ablation Attribution

| Ablation | Available | Scenario Pass | False Recall | Cue/Planning Signal |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 1.000 | 0.000 | 1.000 |
| Engram Search Only | yes | 0.750 | 0.000 | 0.000 |

## Appendix Baselines

| Baseline | Available | Scenario Pass | False Recall | p50 ms |
|---|---:|---:|---:|---:|
| Context Window | yes | 0.250 | 0.400 | 0.035 |
| Markdown Memory | yes | 0.250 | 0.400 | 0.023 |
| Vector RAG | yes | 0.250 | 0.400 | 14.507 |

## External And Supporting Tracks

| Track | Available | Executed | Summary | Recommended Command |
|---|---:|---:|---|---|
| retrieval_ab | yes | no | purpose=Controlled retrieval A/B comparison on the synthetic corpus. | `uv run python scripts/benchmark_ab.py --verbose --seed 42` |
| working_memory | yes | no | purpose=Working-memory continuity and bridge recall. | `uv run python scripts/benchmark_working_memory.py --verbose` |
| echo_chamber | yes | no | purpose=Long-run drift, coverage, and surfaced-vs-used behavior. | `uv run python scripts/benchmark_echo_chamber.py --queries 200` |
| locomo | no (LoCoMo dataset path not provided or missing) | no | purpose=External multi-turn memory evaluation. | `uv run python scripts/benchmark_locomo.py --dataset-path <path>` |

## Where Engram Wins

- Headline showcase pass rate: `1.000` for Engram Full.
- Lower or equal false recall versus: Context + Summary, Markdown Canonical, Hybrid RAG Temporal.
- Higher scenario pass rate versus: Context + Summary, Markdown Canonical, Hybrid RAG Temporal.
- Primary scenario wins: Cue Delayed Relevance, Temporal Override, Prospective Trigger, Cross Cluster Association.

## Where Competitors Stay Competitive

- No primary competitor won a showcase scenario in this run.

## Supporting Artifacts

- `benchmark_ab`: `/Users/konnermoshier/Engram/server/scripts/benchmark_ab.py`
- `benchmark_echo_chamber`: `/Users/konnermoshier/Engram/server/scripts/benchmark_echo_chamber.py`
- `benchmark_locomo`: `/Users/konnermoshier/Engram/server/scripts/benchmark_locomo.py`
- `benchmark_working_memory`: `/Users/konnermoshier/Engram/server/scripts/benchmark_working_memory.py`

## README Snippet

Benchmark results (quick, measured against equal retrieval budgets): `engram_full` passed 1.000 of showcase scenarios with false recall 0.000, versus context summary 0.000, markdown canonical 0.250, hybrid rag temporal 0.250.
