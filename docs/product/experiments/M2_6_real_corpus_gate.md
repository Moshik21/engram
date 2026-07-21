# M2.6 — Real-corpus flip gate for `usage_ranking_enabled` (RF goal, gates G1/G2/G6)

**Date:** 2026-07-21. **HEAD:** 9428c03 (final RF stack: compute_u, u-tiebreaker + usage-view novelty, episode u via cues, all behind `usage_ranking_enabled=False`). **G7 precondition:** PASS (RF_G7_citation_scan.md, all four gates clear).

## Protocol / safety

- **Brain:** APFS clone (`cp -Rc`, instant, space-shared) of the read-only 17GB
  live-copy snapshot → `<scratchpad>/experiments/live-work`. The live data dirs
  (`~/.helix/engram-native-dogfood-axi`, `~/.engram`) were never opened; the
  live shell was touched only via read-only HTTP GETs on :8100. One process at
  a time on the clone, `ENGRAM_MODE=helix` + explicit
  `ENGRAM_HELIX__DATA_DIR=<live-work>`, `ENGRAM_HOME` → scratch dir.
- **Clone sanity:** clone 8,876 episodes / 686 entities / 274 relationships /
  2,806 cues vs live shell 8,917 / 689 / 275 / 2,810 — matches within one day
  of organic drift; not corrupt. (The briefed expectation of ~1.5k
  relationships / ~8.7k cues was stale — the real brain has 275 relationships.)
- **Rig:** `<scratchpad>/experiments/m26_real_corpus/run_m26.py` — one process,
  one native-engine open, shared graph/search handles, per-arm
  `ActivationConfig` + fresh in-memory activation store, `record_access=False`
  on every recall (no cross-arm or clone contamination), `time.time` pinned
  during recall (Step 5.05 + compute_u read wall clock; unpinned drift breaks
  byte-identity). Raw arm outputs: `arm_*.json` / `arm_*R.json` /
  `probe_deep_b2.json` in the rig dir.

## Query provenance

`~/.engram/axi-hook-runs.jsonl` (374 rows) is **telemetry-only** — operation /
status / durations, **zero query text**. The 42-query set (`queries.json`) was
therefore reconstructed from the brain's own real content (entity dump +
episode samples from the clone): 8 Decision-probe-class queries (incl. the
continuity gate's literal `"Decision strategy continuity"` and real organic
Decision names), 30 substantive project-thread queries (Engram /
MachineShopScheduler / writeide-work), 4 temporal/frequency queries (t4 routes
FREQUENCY and exercises the M2.4 top-used retriever).

## Arms

| arm | flag | usage store |
|---|---|---|
| A | OFF (shipped) | empty |
| E | ON | empty — **the flip-day state** (G1 on real data) |
| B1 | ON | spec-literal proxy: entities with live hygiene `access_count>=2` → ONE used-tier event at `last_accessed` (**1 entity qualifies**) |
| B2 | ON | mention-derived proxy: entities linked to ≥2 episodes → ONE used-tier event at last-mention time (**191 entities**, 28% of graph) |

**PROXY LABEL (Judge-3 caveat):** B1/B2 events are synthesized from
surfacing/mention-biased hygiene history. They estimate the accumulated-usage
*future*, not current truth.

**Organic usage state at flip time:** the ranking-eligible signal is **empty**.
Live activation snapshot: 3 entities with any hygiene history, exactly **1**
with `access_count>=2` ("server", 3 accesses, 2026-07-17); `cue_used_count=0`
across 2,806 cues. There is nothing for the flip to act on today.

## Results

### Lane 1 — shipped budgets (`recall_budget_explicit_search_ms=1500`)

- Reach: only 4–7 of 42 queries non-empty per arm; **all 8 Decision probes and
  both temporal probes n=0 in every arm**.
- **E vs A: 36/42 byte-identical.** All 6 mismatches (q09 q23 q24 q25 q26 t4)
  are appear/disappear of *episode* results with disjoint id sets — five are
  n=0↔n≈3–5, q09 is n=5 vs n=5 with fully disjoint sets — search-stage
  timeout jitter, not ranking. Proof it is instrument noise: the
  flag-identical pairs mismatch on the same qids (B1 vs E: q23 q24 q25 t4;
  B2 vs E: q09 q24 q25 t4), and a 1-entity u-plant cannot swap disjoint
  episode-id sets.
- **B churn:** B1 37/42, B2 38/42 identical to A; jaccard@10 mean 0.90/0.90
  (min 0.0 driven by the n=0↔n=5 flips); top-1 changes 5/4 — all inside the
  jitter set.
- **Theorem check: zero within-common-top-10 inversions in B1 or B2 → zero
  violations of the 1.30× band** (vacuous: usage never reordered anything; all
  churn was membership jitter equally present between flag-identical arms).

### Lane 2 — relaxed budgets (30s wall / 20s search; isolates the deep pipeline)

- **Arm A: 0/42 queries non-empty.** With the timeout-degrade lane disabled,
  the deep retrieval pipeline yields **zero results on the entire query set**
  — every non-empty result in Lane 1 came from the timeout-degraded
  fast-fallback path, not the primary pipeline.
- **E vs A: 42/42 byte-identical** (both all-empty, deterministic) — **G1 on
  real corpus: PASS.**
- Focused deep probe (flag ON + all 191 B2 plants; t1/t4/q09/q24, each run
  twice): **n=0 on all, rep1==rep2 byte-identical** — a populated usage store,
  including the M2.4 FREQUENCY top-used retriever at t4, cannot change
  deep-lane output on this brain.
- Diagnostic captured: `get_entity_embeddings` recovers only 3/10 and 33/40
  entity vectors ("stale vector metadata or incomplete indexing") — the known
  Jul-13 FastEmbed-outage vector-less backlog; a prime suspect for the
  deep-lane emptiness and separately for the continuity failure below.

### Continuity (`engram continuity --against-live --organic`)

| run | flag | result | recall_ms | hits |
|---|---|---|---|---|
| LIVE shell :8100 (shipped) | OFF | **FAIL** | 6508.7 | 0 (organic target exists: `GOLDEN_DECISION_1783643390`) |
| clone :8199 | OFF | **FAIL** | 6507.7 | 0 |
| clone :8199 | ON (empty store) | **FAIL** | 6506.5 | 0 |

- Clone OFF vs ON payloads: **identical in every substantive field** (passed,
  organic_target, scrap=0, counts, hit sets); only wall-clock fields differ —
  the G1 assertion holds on the continuity path.
- The FAIL is **pre-existing and flag-independent**: identical ~6.5s
  recall-timeout structure with 0 Decision hits in the shipped state, on both
  the live brain and the untouched clone, loaded or unloaded.

## Flip recommendation

Framing per the gate:

- **(a) Is E==A byte-identical (flip is a no-op today)?** **YES** — 42/42 in
  the deterministic deep lane, substantive-identical on the continuity path,
  and Lane-1 mismatches are proven instrument noise (present equally between
  flag-identical arms). Doubly a no-op: the organic usage store is empty AND
  the primary pipeline currently produces nothing to rank.
- **(b) Is B's churn bounded + tie-break-shaped?** **Vacuously yes** — zero
  usage-driven inversions anywhere, zero 1.30×-band violations, top-used
  retriever inert. But this is weak evidence about the accumulated future: the
  corpus produced no non-degraded results to reorder, so the theorem was never
  given a live counterexample opportunity. Re-measure churn after recall is
  healthy.
- **(c) Continuity PASS both?** **NO — FAIL/FAIL, identically.** No regression
  from the flag, but G6 requires *no regression AND live usage-event yield >
  floor*: yield is **zero** (no organic used/confirmed/corrected events
  exist), and the gate itself is failing in the shipped state.
- **(d) G2 (usage never degrades: arm-A parity reach@10 ≥ 23)?** **UNMET —
  and unmeasurable as specified.** Arm-A reach is 4/42 shipped-lane and 0/42
  in the deep lane; the ≥23 parity floor cannot be adjudicated on a recall
  stack that returns almost nothing. G2 must be re-adjudicated explicitly at
  the post-repair rerun.

**RECOMMENDATION: DO NOT FLIP.** (The flip is mechanically safe — a proven
no-op today — but three of the gate's conditions objectively fail: G6's
yield floor at zero, the organic continuity PASS, and G2's parity floor. A
default flip inside a degraded recall stack would also contaminate later
attribution. Revisit after the prerequisites below.)

1. Safety is proven (G1 real-corpus PASS; flip cannot change today's output),
   so nothing blocks the flip *mechanically*.
2. But G6's flip condition is objectively unmet: organic usage-event yield is
   zero (below any floor). Flipping now activates a ranking channel with no
   input and would timestamp the default change inside a degraded recall
   stack, contaminating later attribution.
3. Prerequisites before the flip (all outside M2.6's scope):
   - repair the deep-recall emptiness — warm-window reindex of the vector-less
     backlog (missing entity vectors measured above) and re-establish the
     organic continuity PASS;
   - let the G7-cleared citation scan accumulate a real capture window of
     used-tier events, then re-run this gate's arm-B churn measurement on a
     recall stack that returns results (expected: tie-break-shaped, ≤1.30×);
   - adjudicate G2's arm-A parity floor (reach@10 ≥ 23) explicitly at that
     rerun.
4. When (3) holds and G6's yield floor is met, flip `usage_ranking_enabled`
   per M2.5 — this report satisfies the "report committed under experiments/"
   DoD; the parent owns the config edit.

## Verify pass (2026-07-21, adversarial)

Independently re-verified from raw artifacts: safety protocol held (live
dirs untouched, clone-only data dirs, one process); E==A byte-identity
asserted on ids AND `repr(scores)`, all pairwise numbers reproduce exactly
from `arm_*.json` including the flag-identical-pair jitter proof; the live
continuity FAIL independently reproduced (6503.6 ms, 0 hits, same organic
target). Verdict concurred and sharpened to **DO NOT FLIP** (recorded
above). Known instrument caveats from the verify pass: the archived
`run_m26.py` contains an A2 instrument-determinism arm that was not executed
(no `arm_A2.json`); the `arm_*_run1.json` files are same-minute copies, not
independent reruns — jitter attribution rests on the flag-identical-pair
argument, which the verifier confirmed holds; no raw artifact was saved for
the live-shell continuity run (numbers corroborated by the verifier's
independent rerun).

## Artifacts

- Rig + raw outputs: `<scratchpad>/experiments/m26_real_corpus/`
  (`run_m26.py`, `queries.json`, `arm_{A,E,B1,B2}.json` shipped lane,
  `arm_{A,E}R.json` relaxed lane, `probe_deep_b2.py(.json)`,
  `analyze_inversions.py`, `m26_report.json`, `m26_reportR-partial` via logs,
  `continuity_clone_{OFF,ON}.json`, `entities_dump.json`,
  `mention_counts.json`, `live_activation_snapshot.json`).
- The relaxed lane's B1R/B2R/A2R full arms were cut short (deep lane proven
  all-empty and deterministic; the focused probe covers the flag-ON populated
  case) — noted for protocol honesty.
