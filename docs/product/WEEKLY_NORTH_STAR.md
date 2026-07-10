# Weekly north-star metric

**Metric:** Cold Decision hit rate on your **real native brain**.

> Fresh session / other agent surfaces ≥1 high-signal prior Decision without
> opening a handoff doc.

Not LongMemEval. Not open_work_count.

---

## How to measure (weekly dogfood)

```bash
# Disposable smoke (CI / installer) — always run
cd server && uv run engram continuity --smoke

# Real native brain (your data-dir)
export ENGRAM_MODE=helix
export ENGRAM_HELIX__TRANSPORT=native
export ENGRAM_HELIX__DATA_DIR="${ENGRAM_HELIX__DATA_DIR:-$HOME/.helix/engram-native}"

# Cold get_context / recall (new process — no warm cache)
uv run engram axi context --project "$PWD" --budget 800 --timeout 8
uv run engram axi recall "strategy decision" --limit 5 --timeout 8
```

Pass criteria for the week:

| Check | Pass |
|-------|------|
| Cold context shows ≥1 Decision packet | yes |
| Cold recall hits a known Decision by name/theme | yes |
| False Decision scrap in top-5 | 0 |
| open_work | ignore as product KPI |

Log one line in your notes (or later a dashboard sample):

```
date | cold_decision_hit=0|1 | query=... | brain=native | notes=
```

## Dashboard

Stats → **Continuity Scorecard** (Decision / Preference / Person counts).
Evaluation → **Continuity Scorecard** (labeled lift). Open work stays under
Graph Hygiene / Consolidate as secondary.

## Latency note (Phase D)

Process cache (45s TTL) + 1s durable pack budget already ship. Live p95 on a
17G dogfood brain is optional dogfood — do not gate releases on fabricated p95.
