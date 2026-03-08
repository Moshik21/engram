# Engram Showcase Benchmark

- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T03:43:44+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 0.900 | 0.500 | 1.000 | 1.000 | 0.091 | 3.236 | 4.863 | 0.444 |
| Context Window | yes | 0.200 | 0.000 | 1.000 | 0.000 | 0.273 | 0.031 | 0.060 | 0.000 |
| Markdown Memory | yes | 0.200 | 0.000 | 1.000 | 0.000 | 0.273 | 0.021 | 0.046 | 0.000 |
| Vector RAG | yes | 0.200 | 0.000 | 1.000 | 0.000 | 0.273 | 14.755 | 25.124 | 0.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Cue Delayed Relevance | Engram Full | Observed-only content remained retrievable without full projection. |
| Temporal Override | Engram Full | Structured state kept the current fact while stale evidence stayed out. |
| Negation And Correction | Context Window | The task stayed local enough that recent history was sufficient. |
| Open Loop Recovery | Engram Full | Observed-only content remained retrievable without full projection. |
| Prospective Trigger | Engram Full | Graph-linked intentions surfaced from related entity activity. |
| Cross Cluster Association | Engram Full | Spreading activation connected lexically distant but linked entities. |
| Multi Session Continuity | Engram Full | Durable project state should survive beyond the immediate conversation window. |
| Context Budget Compression | Engram Full | Structured context preserved key facts under a tighter token budget. |
| Meta Contamination Resistance | Engram Full | System chatter stayed out of the durable memory surface. |
| Selective Extraction Efficiency | Engram Full | Engram should answer later questions without projecting every observed turn. |

## Scenario Matrix

### Cue Delayed Relevance

- Why it matters: Observed-only content remained retrievable without full projection.
- Context Window: pass_rate=1.000, avg_tokens=101.333, avg_latency_ms=0.059
- Engram Full: pass_rate=1.000, avg_tokens=67.333, avg_latency_ms=3.235
- Markdown Memory: pass_rate=1.000, avg_tokens=111.333, avg_latency_ms=0.045
- Vector RAG: pass_rate=1.000, avg_tokens=101.333, avg_latency_ms=19.836

### Temporal Override

- Why it matters: Structured state kept the current fact while stale evidence stayed out.
- Context Window: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=0.019
- Engram Full: pass_rate=1.000, avg_tokens=59.000, avg_latency_ms=3.088
- Markdown Memory: pass_rate=0.000, avg_tokens=38.000, avg_latency_ms=0.011
- Vector RAG: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=9.516

### Negation And Correction

- Why it matters: The task stayed local enough that recent history was sufficient.
- Context Window: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=0.025
- Engram Full: pass_rate=0.000, avg_tokens=84.000, avg_latency_ms=4.475
- Markdown Memory: pass_rate=0.000, avg_tokens=29.000, avg_latency_ms=0.017
- Vector RAG: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=19.163

### Open Loop Recovery

- Why it matters: Observed-only content remained retrievable without full projection.
- Context Window: pass_rate=1.000, avg_tokens=88.333, avg_latency_ms=0.055
- Engram Full: pass_rate=1.000, avg_tokens=51.000, avg_latency_ms=2.540
- Markdown Memory: pass_rate=1.000, avg_tokens=98.333, avg_latency_ms=0.045
- Vector RAG: pass_rate=1.000, avg_tokens=90.000, avg_latency_ms=18.927

### Prospective Trigger

- Why it matters: Graph-linked intentions surfaced from related entity activity.
- Context Window: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=0.027
- Engram Full: pass_rate=1.000, avg_tokens=197.000, avg_latency_ms=4.490
- Markdown Memory: pass_rate=0.000, avg_tokens=74.000, avg_latency_ms=0.020
- Vector RAG: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=10.771

### Cross Cluster Association

- Why it matters: Spreading activation connected lexically distant but linked entities.
- Context Window: pass_rate=0.000, avg_tokens=20.000, avg_latency_ms=0.025
- Engram Full: pass_rate=1.000, avg_tokens=83.000, avg_latency_ms=4.051
- Markdown Memory: pass_rate=0.000, avg_tokens=24.000, avg_latency_ms=0.017
- Vector RAG: pass_rate=0.000, avg_tokens=20.000, avg_latency_ms=22.395

### Multi Session Continuity

- Why it matters: Durable project state should survive beyond the immediate conversation window.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.032
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=3.142
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.021
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=12.783

### Context Budget Compression

- Why it matters: Structured context preserved key facts under a tighter token budget.
- Context Window: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.037
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=3.078
- Markdown Memory: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.024
- Vector RAG: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=11.004

### Meta Contamination Resistance

- Why it matters: System chatter stayed out of the durable memory surface.
- Context Window: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=0.018
- Engram Full: pass_rate=1.000, avg_tokens=55.000, avg_latency_ms=2.274
- Markdown Memory: pass_rate=0.000, avg_tokens=35.000, avg_latency_ms=0.010
- Vector RAG: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=13.038

### Selective Extraction Efficiency

- Why it matters: Engram should answer later questions without projecting every observed turn.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.034
- Engram Full: pass_rate=1.000, avg_tokens=62.000, avg_latency_ms=2.994
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.024
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=11.367

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 0.900 | 0.091 | 0.444 |
| Engram Search Only | yes | 0.800 | 0.091 | 0.444 |

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
