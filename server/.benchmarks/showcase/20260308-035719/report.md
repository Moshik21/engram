# Engram Showcase Benchmark

- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T03:57:19+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 0.900 | 0.500 | 1.000 | 1.000 | 0.091 | 3.007 | 4.964 | 0.444 |
| Context Window | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 0.032 | 0.057 | 0.000 |
| Markdown Memory | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 0.021 | 0.044 | 0.000 |
| Vector RAG | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 11.731 | 19.208 | 0.000 |

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
- Context Window: pass_rate=0.000, avg_tokens=101.333, avg_latency_ms=0.054
- Engram Full: pass_rate=1.000, avg_tokens=67.333, avg_latency_ms=3.037
- Markdown Memory: pass_rate=0.000, avg_tokens=111.333, avg_latency_ms=0.042
- Vector RAG: pass_rate=0.000, avg_tokens=101.333, avg_latency_ms=13.383

### Temporal Override

- Why it matters: Structured state kept the current fact while stale evidence stayed out.
- Context Window: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=0.017
- Engram Full: pass_rate=1.000, avg_tokens=59.000, avg_latency_ms=3.093
- Markdown Memory: pass_rate=0.000, avg_tokens=38.000, avg_latency_ms=0.010
- Vector RAG: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=14.737

### Negation And Correction

- Why it matters: The task stayed local enough that recent history was sufficient.
- Context Window: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=0.038
- Engram Full: pass_rate=0.000, avg_tokens=84.000, avg_latency_ms=19.698
- Markdown Memory: pass_rate=0.000, avg_tokens=29.000, avg_latency_ms=0.041
- Vector RAG: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=13.021

### Open Loop Recovery

- Why it matters: Observed-only content remained retrievable without full projection.
- Context Window: pass_rate=1.000, avg_tokens=88.333, avg_latency_ms=0.049
- Engram Full: pass_rate=1.000, avg_tokens=51.000, avg_latency_ms=2.280
- Markdown Memory: pass_rate=1.000, avg_tokens=98.333, avg_latency_ms=0.042
- Vector RAG: pass_rate=1.000, avg_tokens=90.000, avg_latency_ms=13.414

### Prospective Trigger

- Why it matters: Graph-linked intentions surfaced from related entity activity.
- Context Window: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=0.024
- Engram Full: pass_rate=1.000, avg_tokens=197.000, avg_latency_ms=4.271
- Markdown Memory: pass_rate=0.000, avg_tokens=74.000, avg_latency_ms=0.019
- Vector RAG: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=7.725

### Cross Cluster Association

- Why it matters: Spreading activation connected lexically distant but linked entities.
- Context Window: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=0.026
- Engram Full: pass_rate=1.000, avg_tokens=96.000, avg_latency_ms=4.165
- Markdown Memory: pass_rate=0.000, avg_tokens=26.000, avg_latency_ms=0.016
- Vector RAG: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=17.821

### Multi Session Continuity

- Why it matters: Durable project state should survive beyond the immediate conversation window.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.030
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=2.998
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.021
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=10.641

### Context Budget Compression

- Why it matters: Structured context preserved key facts under a tighter token budget.
- Context Window: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.033
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=2.806
- Markdown Memory: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.023
- Vector RAG: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=9.222

### Meta Contamination Resistance

- Why it matters: System chatter stayed out of the durable memory surface.
- Context Window: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=0.017
- Engram Full: pass_rate=1.000, avg_tokens=55.000, avg_latency_ms=2.556
- Markdown Memory: pass_rate=0.000, avg_tokens=35.000, avg_latency_ms=0.009
- Vector RAG: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=9.878

### Selective Extraction Efficiency

- Why it matters: Engram should answer later questions without projecting every observed turn.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.032
- Engram Full: pass_rate=1.000, avg_tokens=62.000, avg_latency_ms=2.835
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.023
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=8.611

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 0.800 | 0.091 | 0.444 |
| Engram Search Only | yes | 0.700 | 0.091 | 0.444 |

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
