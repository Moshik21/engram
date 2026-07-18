# M4.1 — The Activation Arm (does ACT-R activation help or hurt retrieval?)

**Date:** 2026-07-17
**Question:** Benchmarks pin `record_access=False`
(`benchmark/longmemeval/adapter.py:431-436`), so the north-star number cannot
see ACT-R activation at all. Given a *realistic* access pattern, does the
activation term help or hurt reachability?

## Verdict

**HURT — catastrophically, in every routing regime, once any real access
history exists.** With a one-pass usage history, held-out bridge reachability@10
collapsed **23/36 → 2/36** (temporal-routed queries) and **12/36 → 0/36**
(neutral-routed). Zeroing the activation *term* (arm E) recovers to 21/36 —
the collapse is 100% the score term, not candidate-pool membership. In the
benchmark regime (no history) activation is a bit-identical no-op
(A ≡ A2 ≡ C). **Recommendation (one line): neutralize activation as a ranking
term (`weight_activation` → 0 in all router profiles) until it is
de-saturated and the TEMPORAL router profile is fixed; keep `record_access`
for telemetry only.**

## Setup

- Reused the M3.1 oracle-surface corpus + brain unchanged (lite/SQLite +
  local FastEmbed, 102 episodes, 72 entities, 36 planted bridges; see
  `M3_1_oracle_surface.md`). Fully local, no keys, no LLM judge.
- Base retrieval config for ALL arms = the M3.1 recommendation
  (`entity_episode_traversal_source="candidates"`, K=10) so gold is reachable
  and activation has room to move ranks.
- **Usage-history pass:** replay the 36 original bridge queries once with
  `record_access=True` and `passage_first_entity_budget=3` (depth-tier opt-in —
  see structural finding 1 for why the shipped budget records nothing).
  Result: 27 person entities accrued 108 access events; topic entities 0.
- **Measurement:** 36 held-out paraphrases (different query template per
  bridge), `record_access=False`, budget back to 0. Metric: reachability@5/@10,
  mean rank of gold, per arm. Each arm runs on a fresh copy of the pristine
  brain; arm B's in-memory activation store carries the usage pass (lite-mode
  activation lives only in `MemoryActivationStore`).

## Results (n=36 held-out bridge questions)

| Arm | usage history | activation term | reach@5 | reach@10 | mean rank | via trav |
|-----|---------------|-----------------|---------|----------|-----------|----------|
| A shipped (record_access=False) | none | routed (0.10-0.55) | 18 | 23 | 3.35 | 21 |
| A2 repeat of A (determinism) | none | routed | 18 | 23 | 3.35 | 21 |
| C `weight_activation=0` | none | zero* | 18 | 23 | 3.35 | 21 |
| Ap usage pass, record_access=False | replayed, unrecorded | routed | 18 | 23 | 3.35 | 21 |
| B0 usage pass, record=True, shipped budget=0 | **0 events recorded** | routed | 18 | 23 | 3.35 | 21 |
| **B usage pass, record=True, budget=3** | 27 ents / 108 events | routed | **0** | **2** | 9.5 | 0 |
| E = B + activation term zeroed in router | same as B | **zero** | **20** | **21** | 2.91 | 20 |
| F = B + activation candidate-pool disabled | same as B | routed | 0 | 2 | 9.33 | 1 |
| H neutral (non-temporal) queries, no usage | none | routed | 12 | 12 | 1.92 | 12 |
| G = H after usage pass | same as B | routed | **0** | **0** | 12 | 0 |

\* C's override is cosmetic: the router deep-copies cfg and overwrites
`weight_activation` per query type (`retrieval/router.py:100-108`), so a plain
cfg override never reaches the scorer. Irrelevant for C (no history ⇒
`base_act=0` for every node, scorer.py:92-93), decisive for interpreting any
"ablation" arm — E patches the router's `_WEIGHT_PROFILES` instead.

A ≡ A2 exactly (bit-identical ranks) — the rig is deterministic; deltas are real.

## Mechanism (traced, not inferred)

Entity candidate pool for held-out q0, top scores:

- pre-usage: `Marisol 0.54, Bartholomew 0.47, floor ~0.28` (topics reachable)
- post-usage: `Marisol 1.03, Bartholomew 1.02, Callum 0.96, … floor 0.82` —
  **all 10 slots are recently-accessed Persons**; topic entities (0 accesses)
  are evicted from the K=10 traversal pool, so traversal expands 10 wrong-person
  ep1s at score ≈ parent×0.6 ≈ 0.5-0.6 each, burying gold (gold's own episode
  scores ≈ 0.2-0.4).

Three shipped defaults compound:

1. **Sigmoid saturation** (`B_mid=-4.0, B_scale=1.7`, decay 0.5): ONE access
   minutes ago ⇒ activation 0.91; sixteen accesses ⇒ 0.98. Within-session
   activation is a near-binary "recently touched" bit with no discrimination —
   and agent sessions are exactly where accesses cluster.
2. **TEMPORAL router profile** (`router.py:49-56`): any query containing
   "lately/latest/recently/what's new" gets `weight_activation=0.55`,
   `weight_semantic=0.20` — activation outweighs semantics 2.75:1 on precisely
   the "catch me up" queries a memory product lives on. (All 6 corpus query
   templates route TEMPORAL.)
3. **Rich-get-richer access loop:** recall records access only on *surfaced*
   entities; surfaced entity slots go to whoever is already boosted. Persons
   got 108 events, topics 0, purely from being surfaceable — no user signal
   involved.

Arm G shows this is NOT just the temporal profile: neutral queries
(DIRECT_LOOKUP/DEFAULT routes, act weight 0.10-0.25) still collapse 12→0,
because +0.1-0.25 × 0.9+ on 27 of 72 entities is bigger than the pool's score
spread. Arm F shows the activation *candidate pool* (get_top_activated, ×3 on
temporal) is not the vector — membership without the term is harmless (E), the
term without the extra membership still kills (F).

## Structural findings (silent-inert family)

1. **`record_access=True` is a no-op under shipped defaults.** Recall records
   access only for entity *results* (`primary_results.py:285`), and the shipped
   `passage_first_entity_budget=0` means entity results never surface — arm B0
   recorded **0 events in 36 recalls with `record_access=True`**. Episodes
   never record access at all. So today there is no code path by which organic
   recall feeds ACT-R activation in the core tier — activation is live only via
   the depth tier (budget ≥ 1) or MCP flows that surface entities.
2. **The benchmark blindspot is real but currently moot:** with a fresh
   process (in-memory activation store starts empty) `base_act=0` for every
   candidate, so `weight_activation` provably cannot change any benchmark
   ranking (A ≡ C). The north-star number neither credits nor debits
   activation — and per this experiment, that accidentally *flatters* it.
3. **Config ablations of routed weights are illusory.** `apply_route()`
   deep-copies the config and overwrites the four core weights, so
   `weight_activation=0` set on the live config silently does nothing for
   routed queries (this experiment's first D arm was void because of it).
   Anyone "turning off activation" via config today is not turning it off.

## Recommendation

Neutralize, don't tune: set the activation component to 0 in all
`_WEIGHT_PROFILES` rows (and default `weight_activation=0`) for the shipped
depth tier; keep `record_access` plumbing for telemetry/consolidation
(prune/mature use access counts productively — those are unaffected here).
Re-admit activation as a ranking term only with, at minimum:

1. de-saturated normalization (rank-relative or z-scored activation, not a
   0.91-0.98 saturated recency bit),
2. a fixed TEMPORAL profile (0.55 on a saturated signal is a distractor
   amplifier; episode recency for temporal queries is already handled by the
   Step-5.05 recency multiplier),
3. an access-recording rule that breaks the rich-get-richer loop (record on
   user-confirmed use, not on mere surfacing),

then rerun this exact rig (arms A/B/E) as the gate.

## Reproduction

```
SCRATCH=<scratchpad>  # rig reuses ../m31_oracle_surface corpus + brain copy
HOME=$SCRATCH/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=$SCRATCH/experiments/fastembed-cache \
uv run python $SCRATCH/experiments/m41_activation/run_m41.py    # arms A,A2,C,Ap,B0,B,D
uv run python $SCRATCH/experiments/m41_activation/run_m41b.py   # arms E,F,H,G
uv run python $SCRATCH/experiments/m41_activation/diagnose3.py  # pool dump pre/post usage
```

Artifacts: `final_summaries.json`, `final_summaries_b.json`, per-arm
`arm_*.json`, `diag3.log` in `<scratchpad>/experiments/m41_activation/`.
