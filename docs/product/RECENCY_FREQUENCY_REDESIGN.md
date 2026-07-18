# Recency/Frequency Redesign — usage signals must earn their rank

**Status:** DRAFT pending founder review
**Created:** 2026-07-17
**Sources:** R/F cartography + session-pollution investigation
(`<scratchpad>/experiments/rf_cartography.md`, `rf_session_pollution.md`,
`rf_judge_math.py`), `experiments/M4_1_activation_arm.md`,
`experiments/M4_3_ts_arm.md`, `experiments/M3_1_oracle_surface.md`,
`experiments/DECISIONS_2026-07-17.md`, `COGNITIVE_CORE_REVIEW_2026-07-17.md`
(§1, §4), `COGNITIVE_CORE_GOAL.md` (M3.3, M4.x rows).

**Provenance note (updated 2026-07-18):** the original synthesis was assembled
from reconstructed inputs after an orchestration payload-interpolation failure.
The real design draft and gear-triage list were subsequently recovered intact
from the workflow journal (`<scratchpad>/experiments/rf_target_design.md`,
`rf_gear_triage.md`) and the design section below now embeds them — the
`[RECONSTRUCTED]` qualifier is retired. Cartography (§`rf_cartography.md`) and
the session-pollution bisection (`rf_session_pollution.md`) were always real.
The **judge panel** remains the one degraded input: all three judges were
payload-starved (their numeric scores are void, not assessments); their
design-independent requirements are folded into the acceptance gates below.
A corrected judge pass runs against this doc before founder sign-off unless the
founder waives it (Decision F8).

## Objective (one sentence)

Recency and frequency influence what Engram surfaces **only through signals
that reflect real use, at magnitudes calibrated against the measured collapse
modes** — every access writer, weight, store, and session mutable in the R/F
family becomes deliberate: live-and-verified, eval-gated, or deleted.

## The measured system (facts this design answers)

- **Saturation:** 1 access ⇒ activation 0.76–0.91; 16 in-session ⇒ 0.91;
  within-session spread Δ=0.149 vs a 0.0-access floor — a near-binary
  "recently touched" bit (`rf_judge_math.py`; engine.py:11-47; B_mid=-4.0,
  B_scale=1.7, d=0.5, epoch-seconds).
- **Collapse:** with any real access history, activation as a ranking term
  destroys reachability — 23/36 → 2/36 @10 temporal-routed, 12→0 neutral;
  zeroing only the term recovers 21/36 (M4.1 arms B/E/G).
- **Rich-get-richer loop:** access records only on SURFACED entity results
  (episodes never); persons accrued 108 events, topics 0, with no user signal
  (M4.1). `get_context` has its own closed loop (P9 + layer-3
  `get_top_activated` ordering, context_builder.py:1052,1203-1216).
- **Shipped default is inert:** `passage_first_entity_budget=0` ⇒ recall
  records 0 events in 36 recalls even with `record_access=True` (M4.1 arm
  B0) — one flag couples surfacing, recording, TS learning, priming input,
  and near-miss payloads (cartography §11.1).
- **Session pollution is a single, proven defect:** working memory leaks
  surfaced *episode* IDs into the *entity* candidate pool
  (candidate_pool.py:203-205; pipeline.py:543-548, 1190-1197); the MMR
  top-20 embedding window gives missing-embedding candidates zero diversity
  penalty (pipeline.py:1917-1922; mmr.py:73-80). Filtering only episode-typed
  WM entries recovers session reach@10 **15/36 → 32/36 = state-clean parity**
  (rf_session_pollution arms S_NOEPIS/S_WM_OFF). WM writes ignore
  `record_access` (primary_results.py:187-194, 301-308) — "read-only" recall
  is not read-only.
- **TS:** dead code at budget 0 (byte-identical on/off, and M1.2 guard blocks
  all learning); at budget 3, no lift (20/31 vs 21/32) and repeat-stability
  33/36 → 0/36 from positional Beta draws (M4.3).
- **Two disagreeing stores:** in-RAM `ActivationState` vs graph-row
  `access_count/last_accessed`; the only sync path (`snapshot_to_graph`) has
  zero production callers; snapshot saves on clean shutdown only; brain-side
  accesses are discarded by design (cartography §1, §9).
- **Router overwrite trap:** `apply_route` deep-copies cfg and overwrites the
  four core weights (router.py:100-108) — config-level `weight_*` ablation is
  illusory; TEMPORAL weights activation 0.55 vs semantic 0.20 (2.75:1),
  FREQUENCY 4:1. `_activation_pool` constructs a fresh `ActivationConfig()`
  (candidate_pool.py:140) — sigmoid retunes silently never reach it.
- **Importance prior exists** (M3.3): `consolidated_strength` seeds at commit
  (cap 0.05 ⇒ standalone act ≈ 0.64), default-off
  (`importance_prior_enabled=False`, config.py:1519); ~10 `compute_activation`
  call sites omit the `cs` argument (spreading, goals, dream, microglia,
  candidate pool — cartography §7) — flipping the flag today makes
  scorer/prune honor the prior while seeding/hygiene don't.

---

## Resolved design (from the recovered design draft — full version in `rf_target_design.md`)

**One store, two views.** The single fix that makes the gears mesh: keep one
usage-event store, expose it through two purpose-calibrated functions — an
ACT-R **hygiene** view (prune floor, maturation, dashboard — *unchanged*) and a
new **ranking** view `u`. The root cause being fixed is one formula serving a
768-day prune floor and a 30%-relevance tiebreaker at once.

**D1 — The ranking signal `u = f · r′`** (replaces the saturating sigmoid in
ranking only; §1). Log-compressed frequency × half-life recency with a floor:
`f = min(1, ln(1+n_eff)/ln(1+N_cap))`, `r = 2^(−Δ_last/h)`, `r′ = r_floor +
(1−r_floor)·r`. Defaults pending §6 calibration: `N_cap=50, h=14d,
r_floor=0.25`. Worked numbers: 1 used @10min = 0.067, 5 used last week = 0.225,
50 used @30d = 0.296, 1 confirmed = 0.176 — real discrimination across
count and age, versus the sigmoid's 0.76–0.91 ceiling on a *single* access.
Saturation now requires ~50 confirmed events — earned, not first-touch.
**ACT-R is not deleted**: `compute_base_level`, the sigmoid, differential tier
decay stay exactly as-is for the hygiene consumers (prune's floor still
protects a single access ~768 days).

**D2 — Recording semantics: the loop fix, zero new MCP tools** (§2). Access
becomes tier-weighted: `w_surfaced=0` (ranking), `w_used=0.3`,
`w_confirmed=1.0`, `w_corrected=0.5`. Surfacing is **killed as a ranking
writer, kept as a hygiene event** — surfacing is ranker output, not
environment, and any nonzero surfaced weight reopens the rich-get-richer loop
(discounting changes the rate, not the fixed point). The **agent-used** signal
is the design's keystone and needs no new tool on the frozen 9-tool surface:
an **ingestion-time citation scan** — recall keeps a per-group ring buffer of
surfaced `(entity_id, name, ts)`; `store_episode` scans the *next observed
turn* with the existing `_matches_entity_name` matcher (`feedback.py:700-711`);
a surfaced entity the agent then cites in its answer (which reaches Engram via
the following `observe()`) becomes an agent-used event. Confirmed/corrected
reuse the existing feedback tool + the already-working cue-hit promotion.
Storage: add `usage_events: list[(ts, weight)]` beside `access_history` (which
stays whole for prune/hygiene).

**D3 — Router: a multiplicative tiebreaker with a "usage never beats
semantics" theorem** (§3). Delete the additive `w_act·act` term; apply
`final = composite_sem × (1 + β_route · u)`, `β_max = 0.30`. The invariant is
provable, not tuned: **X overtakes Y only if `sem(X) > sem(Y)/(1+β)`** — usage
flips near-ties inside a ≤30% relevance band and can never rescue a
semantically buried item. Worked on the exact M4.1 collapse (temporal query,
wrong-person sem 0.30 saturated-usage vs gold topic sem 0.50 no-usage): old
router scored person 0.561 vs topic 0.100 (the 23→2 collapse); new scores
person 0.375 vs topic 0.50 → gold wins, while a used entity still beats an
unused one at equal semantics. Router profiles lose the activation column and
gain a `β_route` (FREQUENCY keeps the largest at 0.30 — the one class where
usage *is* relevance). **Delete the activation-based candidate backfill** —
usage is a reranker, not a retriever (arm E vs A: membership alone cost 2
hits); rediscovery belongs to the graph channel M3.1 just proved (2/36→22/36).
The **kill-switch works provably** because `usage_ranking_enabled` lives at the
multiplication site and `apply_route`'s write-set no longer contains any usage
field — the deep-copy trap that voided M4.1's first ablation can't reach it.

**D4 — Episode unification: two recencies, one `u`** (§4). Engram conflates
*environmental* recency (when a memory was created — the Step-5.05
conversation-date boost, benchmark-validated, **unchanged**) with *behavioral*
recency (when it was useful — `u`). Episodes are the core tier yet accrue no
usage signal at all. Fix: `episode_final = rrf × (1 + β·u_episode) × (1 +
temporal_cue_boost)` — same `u`, sourced from the **existing cue substrate**
(add `used_count`, `last_used_at` to the cue record; same citation scan via
`_matches_cue_content`). Episodes without cues get u=0 — no per-episode
`ActivationState` bloat. The two recencies compose multiplicatively and are
separately gated and killable — conflating them is exactly what
`TEMPORAL→weight_activation=0.55` did.

**D5 — Session state becomes typed and gated** (Task B; the biggest live
gear). The bisection isolated the mechanism to a single defect: working memory
leaks *episode* ids into the *entity* candidate pool; MMR's 20-id embedding
window, giving missing-embedding candidates a zero diversity penalty, amplifies
+5 phantom candidates into a 2× reachability loss (state-clean 21/32 vs
session-shaped 8/15, fully recovered by S_NOEPIS). NOT conv-context topic
shaping — that contributes zero in recall-only sessions. Fix: (a) type-filter
episode entries at the 3 WM consumption sites; (b) gate WM writes on
`record_access` to restore the read-only-recall contract; (c) harden MMR to
fetch embeddings for all candidates or penalize missing ones. Entity-typed WM
priming is design and stays. Yields the determinism theorem (G4): same query,
same session, no intervening used/confirmed event ⇒ identical ranking.

**D6 — Interaction contracts and durability** (§5). *Importance ≠ usage*:
`consolidated_strength` (M3.3) is excluded from `n_eff`/`Δ_last` — importance,
usage, recency are three separable, separately-killable multipliers (product
bounded ≤ 2.5×1.30). Prune keeps consuming full `access_history` unchanged.
Snapshot durability is tiered: surfaced loss is free, used-loss acceptable,
**confirmed/corrected loss is not** → a small append-only JSONL journal for
confirmed events, replayed on load, exempt from the 14-day age-out (user
signal doesn't expire). Snapshot format bumps to v2 (v1 loads as legacy/empty
usage). The stale graph-row `access_count` (zero callers) is resolved by
wiring `snapshot_to_graph` into the mop window (shell paused ⇒ single-writer),
ranking never reads it (Decision F3).

**D7 — Thompson sampling: KILL** (gear triage #1; **supersedes** the recorded
"SEEDED-ON, flip parked" — needs founder ack, F4). TS is a bandit whose reward
channel is structurally absent: at budget 0 doubly inert (byte-identical
on/off *and* feedback blocked), at budget >0 it buys −1@5/−1@10 and costs
33→0 in-session repeat-stability via positional Beta draws. Delete
`score_candidates_thompson`, the pipeline branch, `activation/feedback.py`,
`ts_*` knobs, and the `ts_alpha/ts_beta` fields (+snapshot compat). Floor if
ack withheld: the one-line `ts_enabled=False` default (mechanism-justified,
corpus-independent). If ever revived: per-entity seeded draws
`blake2b(group,query,entity_id)`.

**D8 — One temporal vocabulary, eventually.** Router regex, temporal-cue
detector, and decomposer classify independently (cartography §11.5); this
design touches only the router table. Full unification is parked (P5) unless
the founder promotes it.

## Constraints (binding)

- **Live-brain safety:** never open `~/.helix/engram-native-dogfood-axi`;
  read-only HTTP to 127.0.0.1:8100 permitted; experiments use the fakehome
  pattern (`HOME=<scratchpad>/fakehome ENGRAM_MODE=lite`) or the read-only
  live copy at `<scratchpad>/experiments/live-copy/`.
- **Fully-local north star:** no external keys for any mechanism here; all
  rigs are judge-free reachability (local-judge accuracy is below instrument
  floor — DECISIONS shared caveats).
- **Eval-gated flips:** small-corpus results do not flip shipped defaults;
  mechanism proofs (no-op / determinism / silent-inert) may.
- **Byte-identical defaults until gates pass** — with one carve-out: M0
  defect fixes intentionally change *session-shaped* behavior (the WM leak is
  live on every install). The gate for those is the session-pollution rig
  (state-clean arms must stay byte-identical: FRESH 21/32 unchanged;
  session-shaped must recover to parity), not byte-identity.
- **Concurrent-edit hazard:** `retrieval/pipeline.py`, `candidate_pool.py`,
  `context_builder.py` are under sibling-workflow edit as of 2026-07-17. M0
  rows touching them land only after that stack merges; line numbers below
  are snapshot-anchored (use the cartography's step labels).
- Public MCP 9-tool surface FROZEN (any feedback tool lands on the
  operator surface unless F1 decides otherwise).
- Storage silent-swallow contract stays green; new tolerated failures carry
  `# silent-ok: <reason>`.

## Global verification gates

- `cd server && uv run ruff check . && uv run ruff format --check .`
- `env HOME=<fakehome> ENGRAM_MODE=lite uv run pytest -m "not requires_helix" -q` → 0 failed.
- **Regression instruments (all fully local):**
  - M4.1 rig: `uv run python <scratchpad>/experiments/m41_activation/run_m41.py && run_m41b.py`
  - Session rig: `uv run python <scratchpad>/experiments/rf_session_pollution/run_bisect.py`
  - TS rig: `uv run python <scratchpad>/experiments/m43_ts_arm/run_ts_ab.py && run_fresh_ab.py`
  - Oracle rig: `<scratchpad>/experiments/m31_oracle_surface/run_experiment.py --reuse`
- North-star: `cd server && uv run engram continuity --against-live --organic`
  green before any default flip.
- EVAL-GATED rows require their experiment report committed under
  `docs/product/experiments/` before the default flips.

---

---

## Executable plan → `RECENCY_FREQUENCY_GOAL.md`

This document is the **design**. The **work order** — milestones M0–M5 with
per-row files, DoD, the canonical acceptance gates (G1–G7), and the
founder-decision list — lives in `docs/product/RECENCY_FREQUENCY_GOAL.md`,
which folds in the three-lens judge panel's blockers (echo-guarded citation
scan, confirmed-journal fold-then-compact, complete activation-reader
enumeration, environmental-frequency tier decision, importance bounded
invariant, G7 citation-scan yield gate). The earlier milestone sketch that
lived here has been superseded by that doc to keep a single source of truth
(it also carried a stale de-saturation milestone the recovered design
rejects — resolved in the goal).

