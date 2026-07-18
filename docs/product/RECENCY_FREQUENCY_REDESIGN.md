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

## M0 — FIX-NOW gear triage (from the recovered `rf_gear_triage.md`)

Defects that are wrong under ANY design direction. No design decisions
embedded; each row is independently landable.

- [ ] **M0.1 WM episode-ID type filter (the session-pollution fix).** Skip
      `item_type=="episode"` WM entries in `_working_memory_pool`
      (candidate_pool.py:203-205), the single-pool injection
      (pipeline.py:543-548), and the WM seed step (pipeline.py:1190-1197).
      Keep episode entries in the buffer (recent-queries/context uses
      unaffected). Files: retrieval/candidate_pool.py, retrieval/pipeline.py
      (AFTER sibling merge). DoD: `run_bisect.py` — S_BASE reach@10 15→≥32,
      FRESH unchanged at 21/32; unit test that a WM episode entry never
      appears in the entity pool. Verification: session rig + full lite suite.
- [ ] **M0.2 Gate session-state writes.** WM writes honor `record_access`
      (or a new explicit `session_state=False` recall kwarg)
      (primary_results.py:187-194, 301-308). Files:
      retrieval/primary_results.py. DoD: drift probe — consecutive identical
      recalls with `record_access=False` are byte-stable (ids AND scores;
      today: pool 36→42, scores drift); benchmarks/evals then measure the
      system they ship. Verification: session-rig drift probe + M4.3 fresh-AB
      determinism 12/12.
- [ ] **M0.3 MMR embedding-coverage hardening.** Fetch embeddings for all
      MMR candidates, or penalize missing-embedding candidates instead of the
      current free pass, or exclude non-entity ids before the window cut
      (pipeline.py:1917-1922; mmr.py:73-80). Files: retrieval/mmr.py,
      retrieval/pipeline.py (after sibling merge). DoD: unit test — injecting
      N phantom ids into the pool does not change MMR selection among scored
      entities; diag_mmr.py q1 trace shows sourdough (0.4674) no longer
      demoted below lower-scored unpenalized entities. Verification: session
      rig; oracle rig unchanged (22/36 @5).
- [ ] **M0.4 Fresh-cfg clone in `_activation_pool`.** Pass the live cfg
      instead of constructing `ActivationConfig()` (candidate_pool.py:140).
      DoD: test — an env/config `B_mid` override reaches pool ranking.
- [ ] **M0.5 Router-overwrite trap made explicit.** Either `apply_route`
      scales cfg weights instead of overwriting, or (minimum) a loud comment
      + a test asserting that `_WEIGHT_PROFILES` is the single tuning point
      (router.py:100-108). DoD: a cfg-level `weight_activation=0` either
      takes effect or the illusion is documented + tested; no third voided
      experiment arm.
- [ ] **M0.6 `compute_activation` signature drift.** Thread
      `consolidated_strength` through the ~10 omitting call sites
      (spreading.py:48,64; goals.py:102; dream.py:222,384; microglia.py:386;
      candidate_pool.py:143; display readers may stay). DoD: grep gate — zero
      remaining `compute_activation(state.access_history, now, cfg)` 3-arg
      ranking-path calls; unit test that a cs-seeded zero-access entity is
      seedable/visible to goals. Precondition for any M3.3 flip.
- [ ] **M0.7 Exploration/TS aliasing recorded.** Document + test that
      `ts_enabled=False` re-enables the deterministic `exploration_weight`
      term (scorer.py:264-268; config.py:184,366); TS A/B rigs updated to
      pin it. DoD: rig configs name both knobs explicitly.
- [ ] **M0.8 Predicate allowlist closed under canonicalization (gear 2).**
      The client-proposal allowlist checks the PRE-canonical predicate but
      apply canonicalizes at commit (`extraction/apply.py:346-356`), so
      `LIVES_IN` passes then stores as `LOCATED_IN` while proposing
      `LOCATED_IN` (the vocabulary actually in the graph) is *rejected* — the
      zero-edge-graph trap from M3.1. 3-part fix: (a) canonicalize before the
      allowlist check (`client_proposals.py:242`); (b) make
      `ALLOWED_CLIENT_PREDICATES` canonical-closed (add LOCATED_IN, REQUIRES,
      HAS_ROLE, CREATED, LIKES, DISLIKES, AIMS_FOR, EXPERT_IN,
      COLLABORATES_WITH, LEADS); (c) reject LOUDLY — serialize rejected rows
      (`status="rejected"`) and emit per-edge reasons + the allowlist in the
      error payload, on partial commits too (today only the all-zero case is
      visible). Do NOT invent synonym mappings (INTERESTED_IN stays rejected,
      but visibly, so the agent retries with PREFERS). **Coordinate with the
      `fix-commit-policy` sibling — overlapping scope on
      `commit_policy.py:126-133`.** DoD: LOCATED_IN commits; proposed and
      stored predicate agree; partial-commit response carries per-edge reasons.

## M1 — Access semantics: confirmed use (D1) [gated on Decision F1]

- [ ] **M1.1 Single ranking-relevant writer.** Confirmed-use feedback
      (retrieval/feedback.py:458,553-561) appends access events; behind
      `access_on_confirmed_use` flag, default off. DoD: feedback event with
      entity ids appends to `access_history` (test); no new storage.
- [ ] **M1.2 Demote surfacing writers.** P1 (primary_results.py:285-299) and
      P9 (context_builder.py:1203-1216 — after sibling merge) stop writing
      the ranking prior when the flag is on (telemetry stream or nothing —
      Decision F2). DoD: recall + get_context under the flag record zero
      ranking-prior events (store-spy tests); prune/mature inputs unchanged
      (their tests green).
- [ ] **M1.3 Signal source wiring.** Per F1: operator-surface feedback tool
      and/or harness citation protocol emits `used` interactions. DoD: one
      end-to-end trace — surfaced memory → usage signal → access event →
      activation change; adoption harness captures it.

## M2 — De-saturation + router rebalance (D2/D3) [EVAL-GATED]

- [ ] **M2.1 Neutralize now (flag-ready, default unflipped).** Activation
      component 0 in all `_WEIGHT_PROFILES` rows + `weight_activation=0`,
      shipped behind a config switch; default behavior byte-identical (arm
      B0 proves inertness). DoD: M4.1 arms A/A2/C byte-identical pre/post.
- [ ] **M2.2 De-saturated normalization.** Implement rank-relative or
      z-scored activation (or retuned B_mid/B_scale/d) meeting the D2
      numeric targets. Files: activation/engine.py, config.py. DoD:
      judge-math scenario table re-run hits targets (1@1s ≤0.6; in-session
      1-vs-16 Δ≥0.30; ordering constraints); M4.1 arm B (usage history, term
      ON) reach@10 ≥ 21 — the collapse must not reproduce.
- [ ] **M2.3 Real-corpus rerun (the flip gate).** M4.1 arms A/B/E on a
      real-corpus brain (live-copy read-only or organic capture corpus) +
      `engram continuity --against-live --organic`. DoD: report committed
      under experiments/; only then may M2.1's switch flip default.

## M3 — Importance prior calibration + flip (D4) [EVAL-GATED; needs M0.6]

- [ ] **M3.1 Magnitude calibration.** Arithmetic tests per COGNITIVE_CORE
      M3.3 (one-shot durable ≈ 5-access mundane at 30d) re-derived against
      the M2.2 normalization; cap revisited per F6. DoD: worked-number test
      file mirrors rf_judge_math scenarios.
- [ ] **M3.2 Flip `importance_prior_enabled` default.** Only after M0.6 +
      M3.1 + no regression on M4.1/oracle rigs. DoD: eval report committed.

## M4 — Durability + store unification (D6)

- [ ] **M4.1 Periodic snapshot save** (shell + MCP stdio; interval or
      dirty-count trigger; keep last-clean-exit ownership rules,
      mcp/server.py:425-451). DoD: kill -9 mid-session loses ≤ interval of
      accesses (test with fake clock).
- [ ] **M4.2 Graph rows: wire or delete** (per F3). Wire = call
      `snapshot_to_graph` (storage/memory/activation.py:159-175) on
      save cadence; delete = drop columns + rewrite prune's pre-filter
      honestly (sqlite/graph.py:1529-1558; helix/graph.py:2755-2813). DoD:
      prune pre-filter semantics documented + tested either way.

## M5 — Episode channel + vocabulary honesty (D7/D8)

- [ ] **M5.1 Document Step 5.05/5.06 as THE episode recency model** (docs +
      test coverage for undated-episode no-boost edge, pipeline.py:1681-1801).
- [ ] **M5.2 TS default per F4** — if flipped: `ts_enabled=False` +
      exploration_weight explicitly pinned; if kept: per-entity seeded draws
      first (M4.3 precondition). DoD: repeat-stability ≥ 33/36 session-shaped
      on the TS rig either way.

---

## Calibration / eval gates (acceptance thresholds)

| Gate | Instrument | Threshold |
|------|-----------|-----------|
| Session-pollution fix (M0.1-0.3) | run_bisect.py, n=36 | session-shaped reach@10 ≥ 32 (parity); state-clean unchanged 21/32; drift probe byte-stable with record_access=False |
| Determinism | M4.3 fresh-AB | 12/12 byte-identical repeats; session repeat-stability ≥ 33/36 |
| De-saturation (M2.2) | rf_judge_math scenarios | 1@1s ≤ 0.60; in-session 1-vs-16 Δ ≥ 0.30; 50-over-60d > 1@1d; stale-200-cap < week-regular |
| Activation re-admission (M2.3) | M4.1 arms A/B/E, real corpus | arm B reach@10 ≥ arm E − 2 (no usage-history collapse); benchmark regime byte-identical |
| Surfacing regression | oracle rig | reach@5 ≥ 22/36 maintained on candidates-source arm |
| Importance flip (M3.2) | M3.1 arithmetic + M4.1/oracle rigs | worked-number invariants hold; no rig regression |
| Any default flip | continuity gate | `engram continuity --against-live --organic` green |

## Decisions needed from founder

1. **F1 — Approve the agent-used signal source (unblocks M1):** the design
   RESOLVES the old "where does a usage signal come from?" blocker — the
   ingestion-time citation scan derives agent-used events from the next
   `observe()` turn on the *frozen* 9-tool surface, no new tool, no protocol
   change (D2/§2). The decision is now narrower: approve the citation-scan
   mechanism (surfaced-entity ring buffer + next-turn `_matches_entity_name`
   scan), plus whether confirmed/corrected also gets a public feedback path or
   stays operator-surface-only. Reject the scan ⇒ M1 falls back to
   operator-surface-only confirmed use (weaker but real).
2. **F2 — Fate of surfacing/delivery recording (P1/P9):** telemetry-only
   stream or removed outright once confirmed-use lands; get_context is the
   highest-traffic surface — its closed loop (P9 + layer-3 ordering) needs an
   explicit verdict.
3. **F3 — Graph-row access columns:** wire `snapshot_to_graph` on cadence, or
   delete the columns and rewrite prune's pre-filter honestly.
4. **F4 — Thompson sampling: KILL vs. flip-off** (the triage escalates this).
   The gear triage recommends **deleting** TS outright (reward channel is
   structurally absent — un-eval-gateable), which **supersedes** the recorded
   DECISIONS-2026-07-17 "SEEDED-ON, flip parked" and GOAL decision 3. Options:
   (a) KILL — delete `score_candidates_thompson`, the pipeline branch,
   `activation/feedback.py`, `ts_*` knobs and the `ts_alpha/ts_beta` fields
   (~1 day incl. snapshot compat); (b) FLOOR — one-line `ts_enabled=False`
   default now (mechanism-justified, ~1hr) and leave the code dormant; (c) hold
   entirely. Needs explicit ack because it reverses a prior recorded decision.
5. **F5 — Router TEMPORAL/FREQUENCY posture:** zero activation in profiles
   (this doc's D3) vs retune to small nonzero values after M2.2.
6. **F6 — Importance-prior magnitude:** accept cs cap 0.05 (standalone act ≈
   0.64 under current sigmoid — large and deliberate) or recalibrate cap
   against the M2.2 normalization.
7. **F7 — Episode frequency prior:** build (greenfield — cue hit_count is the
   only per-episode usage counter today) or park indefinitely.
8. **F8 — Judge panel:** the design and triage are now real (recovered from
   the journal); only the adversarial judge panel was degraded (payload-
   starved, scores void). Run a corrected three-lens panel against this doc
   before build sign-off, or waive it and proceed on the design's own
   pre-empted-objections section.
9. **F9 — Temporal vocabulary unification (D8):** promote to a milestone or
   leave parked.

## Parked (dated reasons)

- **Global activation-weight zeroing default flip** — parked 2026-07-17:
  collapse magnitude from a 36-question synthetic rig; shipped config already
  inert; gate = M2.3 real-corpus rerun (DECISIONS).
- **`ts_enabled=False` default flip** — parked 2026-07-17:
  INCONCLUSIVE-on-lift / CONFIRMED-on-mechanism; unpark path in F4 (M4.3).
- **`entity_episode_traversal_source="candidates"` default flip** — parked
  2026-07-17: oracle ceiling on planted structure; needs real-corpus eval +
  RRF-vs-cosine scale normalization in the same stack (M3.1/DECISIONS).
- **M4.4 confirmed-use writer default-on** — parked 2026-07-17: no shipped
  client emits usage signals; implementing now records ~0 events = another
  silent-inert. Unpark = F1 (DECISIONS).
- **Entity-budget re-enable (`passage_first_entity_budget≥1`)** — parked
  2026-07-17: the flag couples five behaviors (cartography §11.1); re-enable
  only after M0 + M1 + M2 decouple recording from surfacing.
- **Episode frequency prior** — parked 2026-07-17: greenfield with no design
  pressure yet; cue hit_count covers the cue-surfaced slice (F7).
- **Cue-feedback read-path writes + conv-fingerprint capture pollution
  probes** — parked 2026-07-17: not exercised by the session rig (no cues in
  that brain, recall-only sessions); cannot explain the measured gap; worth
  separate small probes (rf_session_pollution closing note).
- **Reconsolidation cross-process fix** — parked 2026-07-17: 5-min in-RAM
  window vs 2h brain cadence is structurally dead (cartography P6); superseded
  by D1's usage-signal framing.

## Completion

This goal is DONE when: M0 fully landed with the session rig at parity; F1–F9
each carry a founder decision; M1–M5 rows landed behind their gates with
experiment reports committed or explicitly parked with dated reasons; all
regression instruments green on the final stack; and the R/F knob census
(cartography §10) contains zero rows whose status is "illusory" or
"accidentally vacuous."
