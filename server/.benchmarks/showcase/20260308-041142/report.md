# Engram Showcase Benchmark

- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T04:11:42+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.299 | 4.486 | 0.444 |
| Context Window | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 0.031 | 0.061 | 0.000 |
| Markdown Memory | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 0.020 | 0.058 | 0.000 |
| Vector RAG | yes | 0.100 | 0.000 | 1.000 | 0.000 | 0.273 | 13.582 | 22.393 | 0.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Cue Delayed Relevance | Engram Full | Observed-only content remained retrievable without full projection. |
| Temporal Override | Engram Full | Structured state kept the current fact while stale evidence stayed out. |
| Negation And Correction | Engram Full | Structured state kept the current fact while stale evidence stayed out. |
| Open Loop Recovery | Engram Full | Observed-only content remained retrievable without full projection. |
| Prospective Trigger | Engram Full | Graph-linked intentions surfaced from related entity activity. |
| Cross Cluster Association | Engram Full | Spreading activation connected lexically distant but linked entities. |
| Multi Session Continuity | Engram Full | Durable project state should survive beyond the immediate conversation window. |
| Context Budget Compression | Engram Full | Structured context preserved key facts under a tighter token budget. |
| Meta Contamination Resistance | Engram Full | System chatter stayed out of the durable memory surface. |
| Selective Extraction Efficiency | Engram Full | Engram should answer later questions without projecting every observed turn. |

## Scenario Matrix

### Cue Delayed Relevance

- Why it matters: Observed content can stay cheap until a later query makes it useful.
- Context Window: pass_rate=0.000, avg_tokens=101.333, avg_latency_ms=0.062
- Engram Full: pass_rate=1.000, avg_tokens=67.333, avg_latency_ms=3.078
- Markdown Memory: pass_rate=0.000, avg_tokens=111.333, avg_latency_ms=0.055
- Vector RAG: pass_rate=0.000, avg_tokens=101.333, avg_latency_ms=18.263

### Temporal Override

- Why it matters: Engram should preserve the latest fact instead of surfacing both versions.
- Context Window: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=0.022
- Engram Full: pass_rate=1.000, avg_tokens=59.000, avg_latency_ms=3.070
- Markdown Memory: pass_rate=0.000, avg_tokens=38.000, avg_latency_ms=0.010
- Vector RAG: pass_rate=0.000, avg_tokens=34.000, avg_latency_ms=9.113

### Negation And Correction

- Why it matters: Negative polarity should suppress stale relationships instead of appending noise.
- Context Window: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=0.026
- Engram Full: pass_rate=1.000, avg_tokens=63.000, avg_latency_ms=4.234
- Markdown Memory: pass_rate=0.000, avg_tokens=29.000, avg_latency_ms=0.017
- Vector RAG: pass_rate=0.000, avg_tokens=25.000, avg_latency_ms=17.045

### Open Loop Recovery

- Why it matters: Latent unresolved work should return later without keeping full history in prompt.
- Context Window: pass_rate=1.000, avg_tokens=88.333, avg_latency_ms=0.055
- Engram Full: pass_rate=1.000, avg_tokens=51.000, avg_latency_ms=2.389
- Markdown Memory: pass_rate=1.000, avg_tokens=98.333, avg_latency_ms=0.066
- Vector RAG: pass_rate=1.000, avg_tokens=90.000, avg_latency_ms=17.919

### Prospective Trigger

- Why it matters: Intentions should fire from related entity activity rather than raw lexical overlap alone.
- Context Window: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=0.025
- Engram Full: pass_rate=1.000, avg_tokens=197.000, avg_latency_ms=4.293
- Markdown Memory: pass_rate=0.000, avg_tokens=74.000, avg_latency_ms=0.019
- Vector RAG: pass_rate=0.000, avg_tokens=64.000, avg_latency_ms=9.826

### Cross Cluster Association

- Why it matters: Graph-aware retrieval should outperform flat lexical retrieval on associative queries.
- Context Window: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=0.030
- Engram Full: pass_rate=1.000, avg_tokens=96.000, avg_latency_ms=4.416
- Markdown Memory: pass_rate=0.000, avg_tokens=26.000, avg_latency_ms=0.018
- Vector RAG: pass_rate=0.000, avg_tokens=22.000, avg_latency_ms=17.721

### Multi Session Continuity

- Why it matters: Durable project state should survive beyond the immediate conversation window.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.029
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=3.179
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.020
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=11.438

### Context Budget Compression

- Why it matters: Structured memory should keep the key facts even when raw notes get truncated.
- Context Window: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.034
- Engram Full: pass_rate=1.000, avg_tokens=60.000, avg_latency_ms=2.990
- Markdown Memory: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=0.023
- Vector RAG: pass_rate=0.000, avg_tokens=60.000, avg_latency_ms=11.796

### Meta Contamination Resistance

- Why it matters: System telemetry must not be mistaken for user memory.
- Context Window: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=0.017
- Engram Full: pass_rate=1.000, avg_tokens=55.000, avg_latency_ms=2.167
- Markdown Memory: pass_rate=0.000, avg_tokens=35.000, avg_latency_ms=0.009
- Vector RAG: pass_rate=0.000, avg_tokens=31.000, avg_latency_ms=11.553

### Selective Extraction Efficiency

- Why it matters: Engram should answer later questions without projecting every observed turn.
- Context Window: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.038
- Engram Full: pass_rate=1.000, avg_tokens=62.000, avg_latency_ms=3.174
- Markdown Memory: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=0.025
- Vector RAG: pass_rate=0.000, avg_tokens=70.000, avg_latency_ms=12.186

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|
| Engram No Cues | yes | 0.900 | 0.000 | 0.444 |
| Engram Search Only | yes | 0.800 | 0.000 | 0.444 |

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
