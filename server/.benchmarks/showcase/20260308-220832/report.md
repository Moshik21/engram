# Engram Benchmark Suite

- Track: `showcase`
- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T22:08:32+00:00`
- Output: `/Users/konnermoshier/Engram/server/.benchmarks/showcase/20260308-220832`
## Fairness Contract

- Track: `showcase`
- Strict fairness: `True`
- Transcript invariant: `True`
- Vector provider family: `local`
- Answer model: `not configured`
- Answer provider: `not configured`

| Scenario | Top-k | Evidence Tokens | Answer Tokens |
|---|---:|---:|---:|
| cue_delayed_relevance | 5 | 120 | 90 |
| temporal_override | 5 | 80 | 70 |
| negation_correction | 5 | 120 | 70 |
| open_loop_recovery | 5 | 110 | 90 |
| prospective_trigger | 5 | 120 | 80 |
| cross_cluster_association | 5 | 120 | 70 |
| multi_session_continuity | 5 | 70 | 60 |
| context_budget_compression | 5 | 60 | 80 |
| meta_contamination_resistance | 5 | 80 | 70 |
| selective_extraction_efficiency | 5 | 70 | 70 |
| correction_chain | 5 | 70 | 60 |
| summary_drift_resistance | 5 | 80 | 70 |

## Executive Table

| Baseline | Available | Scenario Pass | False Recall | Temporal | Negation | Open Loop | Prospective | Answer Pass | Answer Score | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 3.415 | 4.994 |
| Context + Summary | yes | 0.417 | 0.462 | 0.500 | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | 0.038 | 0.054 |
| Markdown Canonical | yes | 0.083 | 0.538 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.054 | 0.088 |
| Hybrid RAG Temporal | yes | 0.083 | 0.385 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 12.548 | 25.989 |

## Capability Scorecard

| Baseline | Association | Compression | Continuity | Cue | Efficiency | Graph | Meta | Negation | Open Loop | Prospective | Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Context + Summary | 0.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | 1.000 | 0.000 | 0.500 |
| Markdown Canonical | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| Hybrid RAG Temporal | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Cue Delayed Relevance | Engram Full | Latent memory stayed available without carrying raw history in prompt. |
| Temporal Override | Engram Full | Current-state memory won without leaking stale or negated facts. |
| Negation And Correction | Engram Full | Current-state memory won without leaking stale or negated facts. |
| Open Loop Recovery | Engram Full | Latent memory stayed available without carrying raw history in prompt. |
| Prospective Trigger | Engram Full | Prospective retrieval surfaced the right intention from related entity activity. |
| Cross Cluster Association | Engram Full | Associative retrieval connected lexically distant but linked entities. |
| Multi Session Continuity | Engram Full | Latent memory stayed available without carrying raw history in prompt. |
| Context Budget Compression | Context + Summary | Rolling summaries kept enough durable state to stay competitive. |
| Meta Contamination Resistance | Engram Full | Canonical memory excluded system chatter and paraphrase drift. |
| Selective Extraction Efficiency | Engram Full | Engram should answer later questions without projecting every observed turn. |
| Correction Chain | Context + Summary | Rolling summaries kept enough durable state to stay competitive. |
| Summary Drift Resistance | Engram Full | Current-state memory won without leaking stale or negated facts. |

## Scenario Matrix

### Cue Delayed Relevance

- Why it matters: Observed content can stay cheap until a later query makes it useful.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 63.000 | 0.044 |
| Engram Full | 1.000 | n/a | n/a | 90.000 | 3.848 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 98.667 | 21.518 |
| Markdown Canonical | 0.000 | n/a | n/a | 120.000 | 0.090 |

### Temporal Override

- Why it matters: Engram should preserve the latest fact instead of surfacing both versions.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 63.000 | 0.024 |
| Engram Full | 1.000 | n/a | n/a | 59.000 | 3.325 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 71.000 | 11.041 |
| Markdown Canonical | 0.000 | n/a | n/a | 80.000 | 0.036 |

### Negation And Correction

- Why it matters: Negative polarity should suppress stale relationships instead of appending noise.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 57.000 | 0.042 |
| Engram Full | 1.000 | n/a | n/a | 63.000 | 4.480 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 120.000 | 17.476 |
| Markdown Canonical | 0.000 | n/a | n/a | 112.000 | 0.057 |

### Open Loop Recovery

- Why it matters: Latent unresolved work should return later without keeping full history in prompt.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 1.000 | n/a | n/a | 70.667 | 0.041 |
| Engram Full | 1.000 | n/a | n/a | 51.000 | 2.939 |
| Hybrid RAG Temporal | 1.000 | n/a | n/a | 90.000 | 16.332 |
| Markdown Canonical | 1.000 | n/a | n/a | 97.333 | 0.063 |

### Prospective Trigger

- Why it matters: Intentions should fire from related entity activity rather than raw lexical overlap alone.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 114.500 | 0.047 |
| Engram Full | 1.000 | n/a | n/a | 120.000 | 4.792 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 120.000 | 12.566 |
| Markdown Canonical | 0.000 | n/a | n/a | 120.000 | 0.068 |

### Cross Cluster Association

- Why it matters: Graph-aware retrieval should outperform flat lexical retrieval on associative queries.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 50.000 | 0.043 |
| Engram Full | 1.000 | n/a | n/a | 96.000 | 4.966 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 91.000 | 21.395 |
| Markdown Canonical | 0.000 | n/a | n/a | 97.000 | 0.054 |

### Multi Session Continuity

- Why it matters: Durable project state should survive beyond the immediate conversation window.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 1.000 | n/a | n/a | 70.000 | 0.027 |
| Engram Full | 1.000 | n/a | n/a | 60.000 | 3.031 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 68.667 | 15.157 |
| Markdown Canonical | 0.000 | n/a | n/a | 70.000 | 0.041 |

### Context Budget Compression

- Why it matters: Structured memory should keep the key facts even when raw notes get truncated.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 1.000 | n/a | n/a | 60.000 | 0.043 |
| Engram Full | 1.000 | n/a | n/a | 60.000 | 3.176 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 60.000 | 9.757 |
| Markdown Canonical | 0.000 | n/a | n/a | 60.000 | 0.071 |

### Meta Contamination Resistance

- Why it matters: System telemetry must not be mistaken for user memory.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 66.000 | 0.022 |
| Engram Full | 1.000 | n/a | n/a | 55.000 | 2.588 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 68.000 | 11.754 |
| Markdown Canonical | 0.000 | n/a | n/a | 80.000 | 0.028 |

### Selective Extraction Efficiency

- Why it matters: Engram should answer later questions without projecting every observed turn.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 0.000 | n/a | n/a | 70.000 | 0.087 |
| Engram Full | 1.000 | n/a | n/a | 62.000 | 3.229 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 70.000 | 9.072 |
| Markdown Canonical | 0.000 | n/a | n/a | 70.000 | 0.052 |

### Correction Chain

- Why it matters: Repeated corrections should converge to a single current truth.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 1.000 | n/a | n/a | 44.000 | 0.021 |
| Engram Full | 1.000 | n/a | n/a | 56.000 | 2.799 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 48.000 | 13.195 |
| Markdown Canonical | 0.000 | n/a | n/a | 70.000 | 0.029 |

### Summary Drift Resistance

- Why it matters: Repeated paraphrases and exploratory chatter should not rewrite current truth.

| Baseline | Evidence Pass | Answer Pass | Answer Score | Avg Tokens | Avg Latency ms |
|---|---:|---:|---:|---:|---:|
| Context + Summary | 1.000 | n/a | n/a | 78.333 | 0.026 |
| Engram Full | 1.000 | n/a | n/a | 61.000 | 3.472 |
| Hybrid RAG Temporal | 0.000 | n/a | n/a | 70.667 | 11.576 |
| Markdown Canonical | 0.000 | n/a | n/a | 80.000 | 0.042 |

## Cost And Error Summary

| Baseline | False Recall | Token Efficiency | Tokens / Success | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| Engram Full | 0.000 | 0.210 | 79.417 | 3.415 | 4.994 |
| Context + Summary | 0.462 | 0.216 | 64.600 | 0.038 | 0.054 |
| Markdown Canonical | 0.538 | 0.029 | 97.333 | 0.054 | 0.088 |
| Hybrid RAG Temporal | 0.385 | 0.101 | 90.000 | 12.548 | 25.989 |

## Ablation Attribution

| Ablation | Available | Scenario Pass | False Recall | Cue/Planning Signal |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 0.917 | 0.000 | 1.000 |
| Engram Search Only | yes | 0.833 | 0.000 | 0.000 |

## Appendix Baselines

| Baseline | Available | Scenario Pass | False Recall | p50 ms |
|---|---:|---:|---:|---:|
| Context Window | yes | 0.083 | 0.308 | 0.043 |
| Markdown Memory | yes | 0.083 | 0.308 | 0.024 |
| Vector RAG | yes | 0.083 | 0.308 | 12.874 |

## Where Engram Wins

- Headline showcase pass rate: `1.000` for Engram Full.
- Lower or equal false recall versus: Context + Summary, Markdown Canonical, Hybrid RAG Temporal.
- Higher scenario pass rate versus: Context + Summary, Markdown Canonical, Hybrid RAG Temporal.
- Primary scenario wins: Cue Delayed Relevance, Temporal Override, Negation And Correction, Open Loop Recovery, Prospective Trigger, Cross Cluster Association, Multi Session Continuity, Meta Contamination Resistance, Selective Extraction Efficiency, Summary Drift Resistance.

## Where Competitors Stay Competitive

- Context Budget Compression (Context + Summary).
- Correction Chain (Context + Summary).

## Supporting Artifacts

- `benchmark_ab`: `/Users/konnermoshier/Engram/server/scripts/benchmark_ab.py`
- `benchmark_echo_chamber`: `/Users/konnermoshier/Engram/server/scripts/benchmark_echo_chamber.py`
- `benchmark_locomo`: `/Users/konnermoshier/Engram/server/scripts/benchmark_locomo.py`
- `benchmark_working_memory`: `/Users/konnermoshier/Engram/server/scripts/benchmark_working_memory.py`

## README Snippet

Benchmark results (full, measured against equal retrieval budgets): `engram_full` passed 1.000 of showcase scenarios with false recall 0.000, versus context summary 0.417, markdown canonical 0.083, hybrid rag temporal 0.083.
