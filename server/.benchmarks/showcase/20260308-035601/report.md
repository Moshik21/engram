# Engram Showcase Benchmark

- Mode: `quick`
- Seeds: `7`
- Generated: `2026-03-08T03:56:01+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 3.889 | 4.050 | 0.545 |
| Context Window | yes | 0.000 | 0.000 | 0.000 | 0.000 | 0.200 | 0.023 | 0.068 | 0.000 |
| Markdown Memory | yes | 0.000 | 0.000 | 0.000 | 0.000 | 0.200 | 0.017 | 0.041 | 0.000 |
| Vector RAG | yes | 0.000 | 0.000 | 0.000 | 0.000 | 0.200 | 11.980 | 14.251 | 0.000 |
| Engram Full Hybrid | no (VOYAGE_API_KEY not set) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Cue Delayed Relevance | Engram Full | Observed-only content remained retrievable without full projection. |
| Temporal Override | Engram Full | Structured state kept the current fact while stale evidence stayed out. |
| Prospective Trigger | Engram Full | Graph-linked intentions surfaced from related entity activity. |
| Cross Cluster Association | Engram Full | Spreading activation connected lexically distant but linked entities. |

## Scenario Matrix

### Cue Delayed Relevance

- Why it matters: Observed-only content remained retrievable without full projection.
- Context Window: pass_rate=0.000, avg_tokens=102.000, avg_latency_ms=0.076
- Engram Full: pass_rate=1.000, avg_tokens=67.000, avg_latency_ms=3.120
- Markdown Memory: pass_rate=0.000, avg_tokens=112.000, avg_latency_ms=0.045
- Vector RAG: pass_rate=0.000, avg_tokens=102.000, avg_latency_ms=11.980

### Temporal Override

- Why it matters: Structured state kept the current fact while stale evidence stayed out.
- Context Window: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=0.017
- Engram Full: pass_rate=1.000, avg_tokens=59.000, avg_latency_ms=2.904
- Markdown Memory: pass_rate=0.000, avg_tokens=38.000, avg_latency_ms=0.011
- Vector RAG: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=8.366

### Prospective Trigger

- Why it matters: Graph-linked intentions surfaced from related entity activity.
- Context Window: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=0.026
- Engram Full: pass_rate=1.000, avg_tokens=197.000, avg_latency_ms=3.977
- Markdown Memory: pass_rate=0.000, avg_tokens=74.000, avg_latency_ms=0.020
- Vector RAG: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=7.212

### Cross Cluster Association

- Why it matters: Spreading activation connected lexically distant but linked entities.
- Context Window: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=0.023
- Engram Full: pass_rate=1.000, avg_tokens=96.000, avg_latency_ms=3.989
- Markdown Memory: pass_rate=0.000, avg_tokens=26.000, avg_latency_ms=0.017
- Vector RAG: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=13.820

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 0.750 | 0.000 | 0.545 |
| Engram Search Only | yes | 0.500 | 0.000 | 0.545 |

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
