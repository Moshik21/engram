# Engram Showcase Benchmark

- Mode: `full`
- Seeds: `7, 19, 31`
- Generated: `2026-03-08T04:09:24+00:00`

## Executive Table

| Baseline | Available | Pass Rate | Temporal | Open Loop | Prospective | False Recall | p50 ms | p95 ms | Selective Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Engram Full | yes | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 4.467 | 7.220 | 1.000 |

## Scenario Winners

| Scenario | Winner | Why |
|---|---|---|
| Negation And Correction | No Baseline Passed | No primary baseline passed this scenario. |

## Scenario Matrix

### Negation And Correction

- Why it matters: Negative polarity should suppress stale relationships instead of appending noise.
- Engram Full: pass_rate=0.000, avg_tokens=88.000, avg_latency_ms=5.356

## Ablations

| Ablation | Available | Pass Rate | False Recall | Selective Extraction |
|---|---:|---:|---:|---:|

## Supporting Artifacts

- Existing retrieval A/B benchmark: `server/scripts/benchmark_ab.py`
- Existing working-memory benchmark: `server/scripts/benchmark_working_memory.py`
