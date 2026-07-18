# Experiment Decisions — 2026-07-17

Decision record for the M3/M4 experiment arms. Source reports (same dir):
`M3_1_oracle_surface.md`, `M4_1_activation_arm.md`, `M4_3_ts_arm.md`.
Context: `docs/product/COGNITIVE_CORE_GOAL.md` (rows M3.1, M4.1–M4.4),
`docs/product/COGNITIVE_CORE_REVIEW_2026-07-17.md` (§1.3, §3, §4).

Shared caveats that bound EVERY verdict below:

- **Corpus size:** one synthetic corpus — 102 episodes / 72 entities / 36
  planted bridge questions, 3 personas, lite/SQLite + local FastEmbed. Planted
  graph (client proposals), so extraction quality is out of frame.
- **Metric:** judge-free REACHABILITY@K only (gold episode id in top-K). No
  answer-accuracy claim is made or licensed — the project has established that
  local-judge accuracy is below instrument floor. Reachability is necessary,
  not sufficient, for answer quality.
- Per project policy: small-corpus results do NOT flip shipped defaults; they
  park as INCONCLUSIVE-for-default with a re-run command, unless the finding
  is a mechanism proof (no-op / determinism / silent-inert), which corpus size
  cannot rescue.

---

## M3.1 / M4.2 — Oracle-surface arm (graph→episode traversal from the candidate pool)

**Verdict: POSITIVE (mechanism confirmed).** Feeding
`append_entity_linked_episodes` from the ranked entity *candidate* pool instead
of final results (which the shipped entity budget of 0 leaves empty — review
§1.3 starvation) converts connected-but-not-surfaced bridges into hits.

**Evidence:** reach@5 2/36 → 22/36, reach@10 2/36 → 23/36, mean gold rank 2.74,
21/23 hits via the traversal channel, +2.9 ms mean latency. K=10 ≡ K=20, so
residual 13 misses are pool *scoring* (RRF-vs-cosine scale, review §1.2/§3),
not pool size. Baseline arm independently confirmed §1.3: `via_traversal=0`
under shipped defaults — the feature is default-enabled but never fires.

**Config decision:** production wiring shipped as
`entity_episode_traversal_source: "results" | "candidates"` in
`ActivationConfig`, **default `"results"` (today's behavior) — default NOT
flipped.** Flip to `"candidates"` (keep `entity_episode_max_entities=10`) is
recommended but **EVAL-GATED on a real-corpus run** per the M3.1 row; land
candidate-pool score normalization (review §1.2) in the same stack to attack
the residual misses.

**Caveats:** planted bridges are the best case for this mechanism — 61% is an
oracle ceiling on synthetic structure, not an expected organic lift; no
interference measured on non-bridge queries beyond the 30 filler episodes.
Gate re-run:

```
cd server && uv run engram continuity --against-live --organic   # with the flag flipped
# plus the depth-eval slice; oracle rig re-run:
HOME=<scratchpad>/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=<scratchpad>/experiments/fastembed-cache \
uv run python <scratchpad>/experiments/m31_oracle_surface/run_experiment.py --reuse
```

---

## M4.1 — Activation arm (ACT-R activation as a ranking term)

**Verdict: NEGATIVE — activation as a ranking term HURTS once any real access
history exists; it is a bit-identical no-op with none (benchmark regime,
A ≡ A2 ≡ C).** After a one-pass usage history, held-out reach@10 collapsed
23/36 → 2/36 (temporal-routed) and 12/36 → 0/36 (neutral-routed); zeroing only
the score term (arm E, router `_WEIGHT_PROFILES` patch) recovers to 21/36 —
the collapse is 100% the term. Mechanism traced: saturated sigmoid (1 access ⇒
0.91), TEMPORAL profile weighting activation 2.75:1 over semantics, and a
rich-get-richer access loop (surfaced-entity-only recording).

**Config decision: PARK as ranking term, do not flip weights globally yet.**
The *directional* decision is firm — do not invest in activation-as-ranker in
its current form — but the shipped default is already effectively inert
(`passage_first_entity_budget=0` ⇒ `record_access=True` records 0 events; arm
B0), so there is nothing safe to flip that changes shipped behavior. The
recommended change (zero the activation component in all `_WEIGHT_PROFILES`
rows + default `weight_activation=0`, keep `record_access` for
telemetry/consolidation — prune/mature use counts productively) is
**parked pending a real-corpus confirmation** because the collapse magnitude
comes from a 36-question synthetic rig. Re-admission gate (all three):
de-saturated normalization, fixed TEMPORAL profile, confirmed-use access
recording (see M4.4), then rerun arms A/B/E.

**Evidence line:** B 23→2 @10 (0 @5); E (term zeroed) 21 @10; G neutral 12→0;
B0 proves shipped-config recording is a no-op (0 events / 36 recalls).

**Caveats:** access history was synthetic (one replay pass, 27 person entities,
108 events) — a plausible but extreme session shape; small corpus; the
depth-tier budget=3 opt-in was required to make activation observable at all.
Structural finding stands regardless of corpus: config-level
`weight_activation=0` is illusory for routed queries (router deep-copy
overwrite) — any future ablation must patch the router profiles.
Re-run:

```
HOME=<scratchpad>/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=<scratchpad>/experiments/fastembed-cache \
uv run python <scratchpad>/experiments/m41_activation/run_m41.py && \
uv run python <scratchpad>/experiments/m41_activation/run_m41b.py
```

---

## M4.3 — Thompson-sampling arm (`ts_enabled` / `ts_weight=0.08`)

**Verdict: NEGATIVE — TS does not earn its weight.** At the shipped budget-0
config it is provably dead code on the read path (TS-on ≡ TS-off, ids AND
scores, 36/36; the M1.2 guard also blocks all feedback ⇒ the bandit can never
learn — doubly inert). At budget 3, state-clean A/B shows no lift and a slight
loss (reach@5 20 vs 21, reach@10 31 vs 32), while within-session
repeat-stability collapses 33/36 → 0/36 because Beta draws are assigned
*positionally* and session state drifts the candidate pool.

**Config decision: recommend `ts_enabled=False` default; flip PARKED as
INCONCLUSIVE-on-lift / CONFIRMED-on-mechanism.** The no-op proof and the
positional-draw determinism defect are corpus-independent facts; the −1/−1
reachability delta is inside small-corpus noise, so the flip is justified by
"dead code + determinism cost," not by the reachability numbers. Before any
re-enable: (a) per-entity seeded draws
(`blake2b(group, query, entity_id)` instead of list-order consumption), and
(b) a config where usage feedback actually flows.

**Evidence line:** budget-0 A0 ≡ B0 byte-identical 36/36; budget-3 state-clean
20/31 (on) vs 21/32 (off); repeat-stable top-10 0/36 (on) vs 33/36 (off);
fresh-state determinism 12/12 both arms (M1.2 seed itself correct).

**Caveats:** exploration terms cannot show value in a 36-query read-only rig —
TS's theoretical payoff requires a feedback loop over time, which the current
config structurally prevents; that absence-of-loop is itself the finding.
Re-run:

```
HOME=<scratchpad>/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=<scratchpad>/experiments/fastembed-cache \
uv run python <scratchpad>/experiments/m43_ts_arm/run_ts_ab.py && \
uv run python <scratchpad>/experiments/m43_ts_arm/run_fresh_ab.py
```

---

## M4.4 — Confirmed-use access recording (design note; PARKED)

**Design note (one paragraph):** Access recording should move from
"record on surfacing" to "record on confirmed use," reusing the existing
feedback path (`retrieval/feedback.py:183` — the confirmed/corrected recording
that already feeds TS posteriors). Concretely: recall itself records nothing
into ACT-R access history; instead, when a client/agent signals that a
surfaced memory was actually used (an explicit feedback call, a citation of
the memory id in the agent's answer, or a `remember()`/correction touching the
same entity), that signal — and only that — appends an access event. This
breaks the rich-get-richer loop M4.1 traced (persons accrued 108 events and
topics 0 purely from being surfaceable), makes the recency/frequency prior
reflect *usage* rather than ranker output, and requires no new storage: the
feedback event's entity ids map onto the existing `access_history` append.
Prune/mature keep consuming access counts unchanged; only the writer changes.

**Decision: PARKED pending a real usage-signal source.** Today no shipped
client emits confirmed-use signals (MCP surface has no feedback tool on the
frozen public 9-tool loop, and the harness does not cite memory ids), so
implementing the writer now would record ~0 events and be another
silent-inert. Unpark when a usage signal exists (e.g. operator-surface
feedback tool or harness citation protocol); it is also gate (3) for
re-admitting activation as a ranking term (M4.1).

---

## Summary table

| Arm | Verdict | Default change now | Parked / gated on |
|-----|---------|--------------------|-------------------|
| M3.1/M4.2 oracle surface | POSITIVE (2/36→22/36 @5) | none — flag shipped, default `"results"` | real-corpus eval + §1.2 score normalization |
| M4.1 activation | NEGATIVE as ranking term; no-op in benchmark regime | none (shipped config already inert) | real-corpus rerun of A/B/E before zeroing router profiles |
| M4.3 TS | NEGATIVE (dead at budget 0; churn at budget 3) | recommend `ts_enabled=False`; flip parked | per-entity seeded draws + live feedback loop |
| M4.4 confirmed-use recording | design note written (above) | none | a real usage-signal source |
