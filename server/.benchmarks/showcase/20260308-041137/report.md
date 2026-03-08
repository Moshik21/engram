# Engram Showcase Benchmark

- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T04:11:37+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.554 | 5.833 | 1.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Negation And Correction | Engram Full | Structured state kept the current fact while stale evidence stayed out. |

## Scenario Matrix

### Negation And Correction

- Why it matters: Negative polarity should suppress stale relationships instead of appending noise.
- Engram Full: pass_rate=1.000, avg_tokens=63.000, avg_latency_ms=4.891

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
