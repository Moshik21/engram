# M4.3 — Thompson-Sampling Arm A/B (does `ts_weight=0.08` earn its keep?)

**Date:** 2026-07-17
**Question:** After M1.2 (deterministic seeding + feedback-flood fix), does the
TS exploration term (`ts_enabled=True`, `ts_weight=0.08`) improve retrieval?
Judge-free metrics only (reachability, mean rank, repeat-determinism).

## Verdict

**NO — TS does not earn its weight. Recommend `ts_enabled=False` by default.**

- **Budget 0 (shipped default): exact no-op.** TS-on vs TS-off byte-identical
  on all 36 queries (ids AND scores), and with entity budget 0 the M1.2 guard
  means no feedback is ever recorded — the bandit can never learn. Doubly inert.
- **Budget 3 (rework-profile shape), state-clean: no lift, slight loss.**
  reach@5 20 vs **21**, reach@10 31 vs **32**, mean rank 5.35 vs 5.38
  (TS-on vs TS-off). TS reshuffled the surfaced list on **36/36** queries but
  changed the gold outcome on only 2 (one gained, one lost) — churn, not signal.
- **Determinism: contract holds, practice doesn't.** Fresh-state repeats are
  byte-identical in both arms (12/12, scores included) — M1.2's seed works for
  state-identical recalls. But within a live session, repeat-stability of the
  same query collapses from **33/36 (TS-off) to 0/36 (TS-on)**: in-memory
  session state (priming buffer / working memory, mutated even with
  `record_access=False`) drifts the candidate pool (56 → 61 candidates on
  consecutive identical recalls), and TS assigns Beta draws **positionally**
  (`scorer.py:score_candidates_thompson` — one `rng.betavariate` per candidate
  in list order), so any pool drift reassigns every entity's sample and
  reshuffles the whole ranking. TS turns benign pool jitter into ~10x ranking
  churn.

## Setup

Reused the M3.1 oracle brain unchanged (copy of `brain.db`): lite/SQLite +
local FastEmbed, 102 episodes / 72 entities / 36 edges, 36 planted
bridge questions (gold = the episode about topic B; query mentions only
person A). Fully local, no keys, no LLM judge. `limit=10`,
`record_access=False` (⇒ `record_feedback=False`, arms are read-only).

Arms (all on the same brain):

| arm | ts_enabled | passage_first_entity_budget |
|-----|-----------|------------------------------|
| A3 | True (shipped default) | 3 |
| B3 | False | 3 |
| A0 | True | 0 (shipped default) |
| B0 | False | 0 |

Two harness variants: **session-shaped** (one manager runs all 36 queries,
each query recalled twice for the repeat-determinism probe) and
**state-clean** (fresh manager per query — deterministic, so the A/B is
exactly the TS term).

## Results

State-clean A/B at budget 3 (fresh manager per query; n=36):

| arm | reach@5 | reach@10 | found | mean rank (found) |
|-----|---------|----------|-------|-------------------|
| A3 TS-on | 20 | 31 | 31 | 5.35 |
| B3 TS-off | **21** | **32** | 32 | 5.38 |

Gold-rank moves: q7 (TS-on **missed**, TS-off rank 5), q17 (8 vs 9). All 36
surfaced lists differ between arms; only these 2 outcomes moved.

Budget-0 no-op check (n=36): A0 == B0 on every query, ids and scores
(reach@5 2, reach@10 2 both — the §1.3 starvation baseline, as in M3.1).

Repeat-determinism (same query recalled twice in the same session):

| arm | identical top-10 (of 36) | note |
|-----|--------------------------|------|
| A0 / B0 (budget 0) | 36 / 36 | pure episode path, stable |
| B3 TS-off (budget 3) | 33 | 3 flips from time/state-drifted near-ties |
| A3 TS-on (budget 3) | **0** | positional Beta draws amplify pool drift |

Fresh-state repeats (12 queries, 2 passes, fresh manager each): 12/12
byte-identical for BOTH arms — the M1.2 seed itself is correct.

Session-shaped runs show the same no-lift picture at lower absolute reach
(A3 17 vs B3 15 @10, A3 7 vs B3 8 @5 — inside the 0/36-repeat-stability noise
of the TS arm itself, so not evidence of lift).

## Incidental findings

1. **Positional sample assignment defect.** The rng seed is
   `blake2b(group + query)` (pipeline.py step 5) but draws are consumed in
   candidate-list order, so sample→entity assignment is only as stable as the
   candidate pool. If TS is ever kept, draw per-entity instead, e.g. seed each
   sample from `blake2b(group, query, entity_id)` — that makes the exploration
   term invariant to pool composition/order.
2. **Session state mutates on read.** Two identical recalls in one manager see
   different candidate pools (56 vs 61) even with `record_access=False` —
   priming buffer / working memory / conv context update on recall. This also
   showed up as session runs reaching far below state-clean runs
   (reach@10 15-17 vs 31-32 on the same brain and queries): cross-persona
   priming pollution. Worth its own ticket.
3. `ts_enabled=False` also disables TS-posterior usage feedback
   (`retrieval/feedback.py` confirmed/corrected recording and pipeline step 7)
   — that is the intended pairing; at budget 0 the M1.2 guard already blocks
   all of it.

## Recommendation

Flip the default to **`ts_enabled=False`**. At the shipped budget-0 config it
is provably dead code on the read path; at budget 3 it costs determinism
(0/36 repeat-stable) and buys no reachability (−1@5, −1@10 on the state-clean
A/B). Re-open only after (a) per-entity seeded draws (finding 1) and (b) a
config where usage feedback actually flows, since exploration without feedback
cannot learn.

## Reproduction

```
SCRATCH=<scratchpad>/experiments/m43_ts_arm   # brain.db copied from m31_oracle_surface
HOME=<scratchpad>/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=<scratchpad>/experiments/fastembed-cache \
uv run python $SCRATCH/run_ts_ab.py        # 4 session-shaped arms + repeat probe
uv run python $SCRATCH/run_fresh_ab.py     # state-clean A/B (fresh manager/query)
uv run python $SCRATCH/probe_determinism.py  # candidate-pool drift probe
uv run python $SCRATCH/probe_fresh.py        # fresh-state determinism check
```

Artifacts: `arm_*.json`, `fresh_*.json` in the scratch dir.
