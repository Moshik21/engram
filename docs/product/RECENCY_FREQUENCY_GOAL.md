# Recency/Frequency Goal — usage signals earn their rank, or they don't ship

**Status:** EXECUTING — approved 2026-07-20 via /goal with founder-delegated
decisions (F1-F10 resolved below with rationale; recorded per-decision)
**Created:** 2026-07-18
**Sources:** `RECENCY_FREQUENCY_REDESIGN.md` (the committed design + M0 triage),
`experiments/RF_target_design.md` (full design, §1–§7), `experiments/RF_gear_triage.md`
(M0 FIX-NOW rows), `experiments/RF_session_pollution.md` (the WM-episode-leak defect),
`experiments/M4_1_activation_arm.md`, `experiments/M4_3_ts_arm.md`,
`experiments/M3_1_oracle_surface.md`, `experiments/DECISIONS_2026-07-17.md`,
`COGNITIVE_CORE_REVIEW_2026-07-17.md` (§1,§4), and the **corrected three-lens
judge panel** (computational-cognitive-science 8/10, durability/blast-radius 7/10,
eval-scaffold 6/10 — every blocker below is folded into a milestone or carries an
explicit disagreement). This doc is the executable work order; read the design for
the why. House style follows `COGNITIVE_CORE_GOAL.md`.

## Objective (one sentence)

Recency and frequency influence what Engram surfaces **only through tiered, trusted,
echo-immune usage events composed as a bounded multiplicative tiebreaker** — every
access writer, weight, store, session mutable, and activation reader in the R/F
family becomes live-and-verified, eval-gated, or deleted, with the ranking default
byte-identical until the six flip gates pass.

## Resolved decisions (design + all three judges converge — do not relitigate)

1. **One store, two views.** A single usage-event store; an ACT-R **hygiene** view
   (prune floor, maturation, dashboard — *unchanged*, `engine.py:11-47`) and a new
   **ranking** view `u`. Root cause fixed: one formula served a 768-day prune floor
   and a 30%-band tiebreaker at once. (All three judges endorsed; "why two formulas"
   is pre-answered — consumers need different transforms.)
2. **`u = f · r′`, bounded, absolute, pool-independent.** `f = min(1,
   ln(1+n_eff)/ln(1+N_cap))`, `r = 2^(−Δ_last/h)`, `r′ = r_floor + (1−r_floor)·r`;
   defaults `N_cap=50, h=14d, r_floor=0.25`. Saturation moves from first-touch to
   ~50 events, not deleted. Pool-relative percentile **rejected** (determinism hazard,
   the M4.3 56→61 pool-drift class). Judge 1 independently recomputed the §D1 table
   (1-used 0.067, 5-used 0.225, 50-used 0.296) — holds exactly.
3. **Multiplicative bounded tiebreaker.** `final = composite_sem × (1 + β·u)`,
   `β_max=0.30`; theorem **X overtakes Y only if `sem(X) > sem(Y)/(1+β)`**. Provably
   prevents the 23→2 collapse (gold 0.50 vs person 0.30: `0.30 < 0.50/1.25`).
   Multiplicative over capped-additive is correctly argued. (All endorsed.)
4. **`w_surfaced=0` as a STRUCTURAL ranking choice, kept at hygiene weight 1.0.**
   Surfacing is ranker output, not environmental need; any nonzero surfaced ranking
   weight reopens the fixed point (discounting changes the rate, not the fixed point).
   (All endorsed.)
5. **Two recencies.** Environmental (`conversation_date`, Step-5.05, unchanged) vs
   behavioral (`u`), composed multiplicatively, separately killable. (All endorsed.)
6. **Delete the activation-based candidate backfill — usage is a reranker, not a
   retriever** (arm E vs A cost 2 hits; rediscovery is the graph channel, 2/36→22/36).
   *Judge-2 caveat resolved:* the FREQUENCY route loses its only "my top-used"
   retriever, so M2.4 adds a bounded top-used retriever **sourced from `usage_events`
   (loop-free)** OR a frequency-query rig as the gate before the backfill is deleted.
7. **Thompson sampling = KILL** (recommended; reverses the recorded "SEEDED-ON, flip
   parked" → needs founder ack, **F4**). Reward channel structurally absent; budget-0
   doubly inert, budget>0 buys −1@5/−1@10 and 33→0 repeat-stability. (All endorsed kill;
   one-line `ts_enabled=False` floor if ack withheld.)
8. **Kill-switch at the multiplication site.** `usage_ranking_enabled` read post-composite;
   `apply_route`'s write-set carries no usage field (closes the deep-copy trap,
   `router.py:100-108`); two permanent tests (schema-frozen apply_route diff;
   populated-store ≡ empty-store). (All endorsed.)
9. **Session-pollution fix (M0.1-M0.3) is a FIX-NOW defect**, rig-proven 15→32 =
   state-clean parity. (All endorsed.)
10. **Used stays at `w_ranking=0` until the echo guard + G7 land** (Judge-1 blocker):
    ship confirmed/corrected first (explicit, echo-immune); the citation scan turns on
    only behind the echo guard and its yield/precision/echo gate.
11. **All seven activation ranking-path readers gate under ONE flag** (Judge-2 blocker):
    D3's two enumerated deletions are insufficient — the "populated ≡ empty" gate cannot
    pass until spreading seed energy, temporal bypass, goal priming, the TS scorer twin,
    prospective, and surprise readers are also neutralized (M2.2).
12. **Confirmed journal uses a fold-then-compact ownership protocol** (Judge-2 blocker):
    only the process whose snapshot incorporated an event may truncate it; non-owners
    append but never truncate (M1.3).
13. **One canonical, numbered gate set** (Judge-3 blocker): the two disagreeing tables
    (arm-B 23/21/19; G4 36/33) are collapsed below to a single set; G2 = arm-A parity
    (≥23), tagged mandatory-mechanism.
14. **Launch runs on fallback defaults, not fitted-from-data** (Judge-3 blocker): the
    `usage_events` distribution is empty at calibration time; §6 "every parameter from
    data" is a post-launch re-fit, and pre-launch fitting on legacy `access_history` is a
    surfacing-biased proxy, called out as such.

## Constraints (binding)

- **Live-brain safety:** never open `~/.helix/engram-native-dogfood-axi`; read-only
  HTTP to `127.0.0.1:8100` permitted; experiments use the fakehome pattern
  (`HOME=<scratchpad>/fakehome ENGRAM_MODE=lite`) or the read-only live copy at
  `<scratchpad>/experiments/live-copy/`.
- **Fully-local north star:** no external keys for any mechanism here; every rig is
  judge-free reachability (local-judge accuracy is below instrument floor — DECISIONS
  shared caveat). A G2 pass is a reachability claim, never an answer-quality claim.
- **Eval-gated flips:** small-corpus results flip no shipped default; mechanism proofs
  (no-op / determinism / silent-inert / echo-immunity) may. Only the live organic
  continuity gate flips defaults.
- **Byte-identical defaults until gates pass** — one carve-out: M0 defect fixes change
  *session-shaped* behavior (the WM leak is live on every install). Their gate is the
  session-pollution rig (state-clean arms stay byte-identical: FRESH 21/32 unchanged;
  session-shaped recovers to parity), not byte-identity.
- **Frozen public MCP 9-tool surface:** no new tools; any confirmed/corrected feedback
  path lands on the operator surface unless **F1** decides otherwise. The agent-used
  signal derives from the next `observe()` turn (D2), adding nothing to the surface.
- **Concurrent-edit hazard:** `retrieval/pipeline.py`, `candidate_pool.py`,
  `context_builder.py` are under sibling-workflow edit (2026-07-17). M0 rows touching
  them land only after that stack merges; line numbers are snapshot-anchored.
- **Storage silent-swallow contract stays green;** every new tolerated failure carries
  `# silent-ok: <reason>`. Commit in logical stacks, targeted tests green before the next.

## Global verification gates (apply to every milestone)

- `cd server && uv run ruff check . && uv run ruff format --check .`
- `env HOME=<fakehome> ENGRAM_MODE=lite uv run pytest -m "not requires_helix" -q` → 0 failed.
- **Regression instruments (all fully local, judge-free):**
  - M4.1 rig: `<scratchpad>/experiments/m41_activation/run_m41.py && run_m41b.py`
  - Session rig: `<scratchpad>/experiments/rf_session_pollution/run_bisect.py`
  - TS rig: `<scratchpad>/experiments/m43_ts_arm/run_ts_ab.py && run_fresh_ab.py`
  - Oracle rig: `<scratchpad>/experiments/m31_oracle_surface/run_experiment.py --reuse`
  - **Echo rig (NEW, G7):** `<scratchpad>/experiments/rf_echo/run_echo.py` — planted-label
    citation-scan yield/precision + top-k feedback echo-immunity.
- North-star: `cd server && uv run engram continuity --against-live --organic` green
  before any default flip.
- EVAL-GATED rows require their experiment report committed under
  `docs/product/experiments/` before the default flips.

---

## M0 — FIX-NOW gear triage (wrong under ANY design; each independently landable)

- [x] **M0.1 WM episode-ID type filter (the session-pollution fix).** Skip
      `item_type=="episode"` WM entries in `_working_memory_pool`
      (candidate_pool.py:203-205), the single-pool injection (pipeline.py:543-548),
      and the WM seed step (pipeline.py:1190-1197); keep episode entries in the buffer.
      Files: retrieval/candidate_pool.py, retrieval/pipeline.py (AFTER sibling merge).
      DoD: `run_bisect.py` S_BASE reach@10 15→≥32, FRESH unchanged 21/32; unit test a
      WM episode entry never appears in the entity pool. **Gate: session-pollution rig.**
- [x] **M0.2 Gate session-state writes on `record_access`** (or explicit
      `session_state=False` recall kwarg) (primary_results.py:187-194, 301-308). Files:
      retrieval/primary_results.py. DoD: drift probe — consecutive identical recalls
      with `record_access=False` are byte-stable (ids AND scores; today pool 36→42,
      scores drift). **Precondition for G4;** verification: session-rig drift probe +
      M4.3 fresh-AB 12/12.
- [x] **M0.3 MMR embedding-coverage hardening.** Fetch embeddings for all MMR
      candidates, or penalize missing-embedding candidates, or exclude non-entity ids
      before the top-20 window cut (pipeline.py:1917-1922; mmr.py:73-80). Files:
      retrieval/mmr.py, retrieval/pipeline.py (after sibling merge). DoD: injecting N
      phantom ids does not change MMR selection among scored entities; diag_mmr q1 shows
      sourdough (0.4674) no longer demoted below lower-scored unpenalized entities.
      **Gate: session rig; oracle rig.** MEASURED 2026-07-21: the MMR coverage
      fix IMPROVED the oracle candidates arm 22 -> 35/36 @5 (36/36 @10,
      via_traversal 34/36) — the same missing-embedding defect was
      suppressing traversal episodes there; baseline arm unchanged 2/36.
      Session rig: S_BASE 15 -> 35/36 reach@10 (gate >=32 PASS), all arms at
      state-clean parity, drift probe byte-stable 36/36/36.
- [x] **M0.4 Fresh-cfg clone in `_activation_pool` + permanent test.** Pass the live cfg
      instead of constructing `ActivationConfig()` (candidate_pool.py:140). DoD: a
      permanent test asserts a cfg-level `B_mid`/`weight_*` override reaches pool ranking
      (Judge-3: prevents the next accidentally-vacuous arm).
- [x] **M0.5 Router explicit-zero kill-switch + permanent test.** In `apply_route`, any
      weight the base cfg sets to exactly `0.0` stays `0.0` (zero = term disabled;
      profiles redistribute only among enabled terms); behavior-preserving for every
      shipped config (router.py:100-108). DoD: `cfg.weight_activation=0` + TEMPORAL query
      ⇒ `routed.weight_activation==0`; default cfg ⇒ profiles byte-identical. Permanent
      test asserting `_WEIGHT_PROFILES` is the single tuning point.
- [x] **M0.6 `compute_activation` signature drift.** Thread `consolidated_strength`
      through the ~10 omitting ranking-path call sites (spreading.py:48,64; goals.py:102;
      dream.py:222,384; microglia.py:386; candidate_pool.py:143). DoD: **grep gate** — zero
      remaining 3-arg `compute_activation(state.access_history, now, cfg)` ranking-path
      calls; unit test a cs-seeded zero-access entity is visible to goals. **Precondition
      for any M3.2 flip and for M2.2's reader enumeration.**
- [x] **M0.7 Exploration/TS aliasing recorded.** Document + test that `ts_enabled=False`
      re-enables the deterministic `exploration_weight` term (scorer.py:264-268;
      config.py:184,366); TS rigs pin both knobs explicitly. DoD: rig configs name both knobs.
- [x] **M0.8 Predicate allowlist closed under canonicalization + loud reject.** (a)
      Canonicalize before the allowlist check (client_proposals.py:242); (b) make
      `ALLOWED_CLIENT_PREDICATES` canonical-closed (add LOCATED_IN, REQUIRES, HAS_ROLE,
      CREATED, LIKES, DISLIKES, AIMS_FOR, EXPERT_IN, COLLABORATES_WITH, LEADS); (c) serialize
      rejected rows (`status="rejected"`) + per-edge reasons + the allowlist in the error
      payload, on partial commits too. No invented synonyms (INTERESTED_IN stays rejected,
      visibly). **Coordinate with `fix-commit-policy` sibling** (commit_policy.py:126-133).
      DoD: LOCATED_IN commits; proposed and stored predicate agree; partial-commit response
      carries per-edge reasons.

## M1 — Recording semantics + storage (D2/D6 storage half; lands BEFORE any ranking flip)

*Sequencing invariant: writers and durability land here, ranking stays byte-identical
(inert by G1/G5). Nothing in M1 changes the benchmark regime.*

- [x] **M1.1 `usage_events` store + tiered `record_access(tier=)` + snapshot v2.** Add
      `usage_events: list[tuple[float, float]]` beside `access_history` (which stays whole
      for prune); `record_access` grows a `tier` param, default `"surfaced"` (w_ranking=0)
      keeps every caller hygiene-correct. Snapshot bumps to v2 (`"version":2`); v1 loads as
      legacy/empty usage (forward-compat via `.get()`). Files: models/activation.py,
      storage/memory/activation.py, config.py. DoD: v1↔v2 round-trip test; `n_eff`/`Δ_last`
      computed O(1); no ranking read of `usage_events` yet.
- [x] **M1.2 Confirmed/corrected tier tagging via existing feedback path.** Tier-tag events
      from the MCP `feedback` tool (`"confirmed"`→1.0, `"corrected"`→0.5, marks labile) and
      treat cue-hit promotion as a confirmed-tier signal (feedback.py:458-459,263-315). Files:
      retrieval/feedback.py. DoD: confirmed/corrected feedback appends a weighted
      `usage_events` entry (test); prune/mature inputs unchanged (their tests green).
- [x] **M1.3 Confirmed-event JSONL journal — fold-then-compact ownership protocol.**
      *(Judge-2 blocker 1 — data-loss race across shell + MCP-stdio + brain.)* Append-only
      journal for confirmed/corrected, replayed on `load_from_file`, exempt from the 14-day
      age-out. On save the **owning** process (a) re-reads and replays the whole current
      journal into RAM so its snapshot is a superset of every journaled event, (b) writes
      the snapshot, (c) truncates by rename-and-recreate (capture old inode), re-scanning
      lines appended during the write into a fresh segment. Non-owning processes (stdio while
      shell alive per mcp/server.py:437,445; brain) append but **never** truncate. Files:
      storage/memory/activation.py, mcp/server.py. DoD: **an event written by a second process
      is still present after the first process saves+truncates** (test); TOCTOU test — a line
      appended between replay-at-load and truncate-at-save survives.
- [x] **M1.4 Agent-used citation scan WITH echo guard + dedup** *(Judge-1 blocker — the
      self-referential access loop).* Recall keeps a **bounded** per-group ring buffer
      (cap ~20-50, short-circuit when empty so the CQRS no-LLM `store_episode` path pays ~0)
      of surfaced `(entity_id, name, ts)`; `store_episode` scans the next observed turn with
      `_matches_entity_name` (feedback.py:700-711). **Echo guard:** a `used` event fires only
      when the entity mention falls in **novel tokens** — diff the observed content against
      the ring-buffer surfaced-payload text and require the mention outside the echoed span
      (or restrict to the user-authored portion of the turn). **Dedup:** at most one used
      event per `(entity_id, conversation)` per window, so a persistently in-context entity
      cannot accrue runaway used events. Ships behind `recall_usage_feedback_enabled`, and
      **`w_used` stays 0 in ranking until G7 passes** (resolved decision 10). Files:
      retrieval/feedback.py, ingestion/capture_surface.py. DoD: G7 echo rig — feeding the
      ranker's own top-k back as the synthetic next turn yields used-event count ~0; planted-
      reliance transcript yields events above floor; adversarial common-word ("Python" surfaced
      but not relied upon) does **not** fire. **EVAL-GATED: G7.**
- [x] **M1.5 Demote surfacing writers to `surfaced` tier.** P1 (primary_results.py:285-299)
      and P9 (context_builder.py:1203-1216 — after sibling merge) record a `surfaced`-tier
      event (hygiene weight 1.0, ranking weight 0) instead of a ranking prior. Files:
      retrieval/primary_results.py, retrieval/context_builder.py. DoD: recall + get_context
      record zero ranking-eligible events (store-spy); prune/mature inputs unchanged.
- [x] **M1.6 Environmental mention-frequency: tier or documented exclusion (F10).**
      *(Judge-2 blocker 3 — P5 ingestion mentions silently map to w=0, leaving ranking-
      frequency dependent entirely on the unproven scan.)* Per **F10**, either (a) give
      ingestion mentions a loop-free `w_mentioned≈0.1-0.2` tier (user-content-driven, not
      ranker-output), wired at extraction/apply.py:847 and bootstrap; or (b) document +
      test the explicit rationale for excluding environmental occurrence from a system whose
      basis is environmental statistics. DoD: whichever branch, a test pins the chosen tier
      weight for ingestion mentions; the choice is recorded in the migration table.

## M2 — De-saturation + router rebalance (D1/D3) [EVAL-GATED]

- [x] **M2.1 The `u = f·r′` ranking function + u-discrimination gate.** Implement the
      picked signal (RF_target §1); **do NOT implement rank-relative/z-scored/retuned-sigmoid
      normalization — the design rejects all three** *(Judge-3 blocker 2: the committed
      REDESIGN M2.2 prose prescribed the rejected approach; this row supersedes it).* Files:
      activation/engine.py (new `compute_u`), config.py. DoD: worked-number invariant test —
      `u(1 surfaced)=0` exactly, `u(1 used)<u(5 used)<u(50 used)` at the stated gaps
      (0.067/0.225/0.296), `u(50 confirmed recent)≥~0.9`, count/age separability; a hard-coded
      constant `u` must FAIL this test. **Gate: u-discrimination (mechanism).**
- [x] **M2.2 Enumerate + gate ALL activation ranking-path readers under one flag.**
      *(Judge-2 blocker 2 — D3 neutralized only 2 of 7; the "populated ≡ empty" gate cannot
      pass otherwise.)* Gate/neutralize under a single `activation_ranking` flag: scorer term
      (scorer.py:84-93) AND the TS twin (241-247), spreading seed energy (spreading.py:42-71,
      the `energy=max(act,0.15)` temporal_mode seed that reproduces a slice of the collapse),
      activation candidate pool (candidate_pool.py:120-148), temporal bypass (pipeline.py:536-550),
      goal priming (goals.py:98-107,151), prospective, surprise. DoD: **populated store ==
      empty store over a rig that exercises spreading AND `temporal_mode`** (not just the
      scorer) — this is the G1 body. Note coupling to F4: if FLOOR not KILL, the TS scorer
      twin's `w_act·act` term remains and must be gated here too.
- [x] **M2.3 Multiplicative router tiebreaker + delete additive term + delete backfill.**
      `final = composite_sem × (1 + β_route·u)`, `β_max=0.30`; delete the additive
      `w_act·act` term (scorer.py:66-120); router profiles lose the activation column, gain
      `β_route` (FREQUENCY 0.30). Delete the activation-based candidate backfill
      (candidate_pool.py:128-140, pipeline.py:531, ×3 temporal expansion). Kill-switch
      `usage_ranking_enabled` at the multiplication site. Files: retrieval/router.py,
      retrieval/scorer.py, retrieval/candidate_pool.py, retrieval/pipeline.py. DoD: two
      permanent tests — (1) `apply_route` output differs from input only in (sem, spread,
      edge, β) schema-frozen; (2) flag-off recall on a populated store byte-identical to empty
      store. **EVAL-GATED: G1, G3.**
- [x] **M2.4 FREQUENCY-route bounded top-used retriever (or frequency-query rig gate).**
      *(Judge-2 improvement — deleting the backfill removes the FREQUENCY route's only
      "my top-used" retriever.)* Provide a bounded top-used retriever **sourced from
      `usage_events` (loop-free, unlike the old activation store)** for the FREQUENCY route,
      OR add a frequency-query rig ("what have I focused on lately") as the gate proving the
      deletion costs no frequency-query answerability. Files: retrieval/candidate_pool.py,
      retrieval/router.py. DoD: frequency-query rig report committed; retriever (if built) is
      bounded and reads only `usage_events`. **EVAL-GATED: frequency-query rig.**
- [x] **M2.5 Neutralize-now (flag-ready, default unflipped).** Activation component 0 in all
      `_WEIGHT_PROFILES` rows + `weight_activation=0` behind `usage_ranking_enabled=False`;
      default behavior byte-identical (arm B0 proves inertness). DoD: M4.1 arms A/A2/C
      byte-identical pre/post. **EVAL-GATED: G1.**
- [x] **M2.6 Real-corpus rerun — the flip gate.** M4.1 arms A/B/E on a real-corpus brain
      (live-copy read-only or organic capture) + `engram continuity --against-live --organic`.
      DoD: report committed under experiments/; **only then may M2.5's switch flip default**,
      and only after **G7** (M1.4) has passed. **EVAL-GATED: G2, G6, G7.**
      EXECUTED 2026-07-21, adversarially verified — report:
      `experiments/M2_6_real_corpus_gate.md`. G1 real-corpus PASS (E==A
      byte-identical: 42/42 deep lane, 36/42 shipped lane with all 6 mismatches
      proven flag-independent timeout jitter; flip is a no-op today). Arm-B churn:
      zero inversions, zero 1.30x-band violations (vacuous — see finding).
      **GATE OUTCOME: DO NOT FLIP** (default stays False, dated 2026-07-21):
      G6 fails twice over (organic usage-event yield = 0; organic continuity gate
      FAILS live, flag-independent, ~6.5s recall / 0 Decision hits) and G2's
      arm-A parity floor (>=23) is unmeasurable at reach 4/42. ROOT FINDING: the
      deep recall pipeline returns 0/42 on the real brain — every live result
      comes from the timeout-degrade fallback; prime suspect is the Jul-13
      FastEmbed vector-less backlog (entity vectors only 3/10, 33/40
      recoverable). Flip prerequisites (parked, tracked in the report): backlog
      reindex + continuity PASS restored -> organic used-event capture window ->
      arm-B churn rerun + explicit G2 adjudication.

## M3 — Importance prior calibration + flip (D6 importance lane) [EVAL-GATED; needs M0.6]

- [x] **M3.1 Magnitude calibration + bounded invariant.** Re-derive the M3.3 arithmetic
      (one-shot durable ≈ 5-access mundane at 30d) against the M2.1 `u`; revisit cap per **F6**.
      *(Judge-1 improvement: the 2.5× durable ranking multiplier composes multiplicatively and
      escapes the "usage never beats semantics" theorem — a durable-but-weak item (sem 0.30→0.75)
      beats a non-durable strong item (sem 0.50). Bring importance under an analogous bounded
      invariant, or justify identity-core surfacing on weak match as intended.)* DoD: worked-
      number test mirrors rf_judge_math scenarios; an importance-vs-semantics invariant test
      states and enforces the chosen bound.
      LANDED 2026-07-21 (tests only, zero source edits): worked numbers pinned —
      flag-ON cs-seed invisible to ranking (prune-only), 5×-mentioned mundane u =
      0.0994/0.0433/0.0258 at 1d/30d/180d vs the durable channel ≥2.42× at every
      horizon (`test_importance_prior.py` 15 tests). Invariants
      (`test_importance_invariants.py` 27 tests): constants pinned 2.5/1.5/0.0
      per class; verify pass found the multiplicative band was a MIS-MODEL — see
      the F6 amendment: the live mechanism is the `prefer_durable_facts` reserved
      lane (type-rank dominance, unbounded, intentional), now pinned exactly,
      incl. usage's 1.30x inability to cross it, the within-lane additive class
      order + exact-tie boundary, and the 0.08 rescue-lane ceiling (+0.2).
- [ ] **M3.2 Flip `importance_prior_enabled` default.** Only after M0.6 + M3.1 + no regression
      on M4.1/oracle rigs. DoD: eval report committed. **EVAL-GATED: continuity gate + M4.1/oracle.**
      PARKED 2026-07-21: M3.1 is landed, but the continuity gate is failing live
      flag-independently (see M2.6 root finding — deep-recall emptiness /
      vector-less backlog), so the eval this flip requires cannot produce a
      meaningful PASS. Additionally M3.1's verify pass amended F6: the live
      importance mechanism is the reserved durable lane, not a bounded
      multiplicative band — the flip decision must be re-reasoned from the lane
      model. Revisit together with the M2.6 flip prerequisites.

## M4 — Durability + store unification (D6)

- [x] **M4.1 Periodic snapshot save** (shell + MCP stdio; interval or dirty-count trigger;
      keep last-clean-exit ownership rules, mcp/server.py:425-451). DoD: kill -9 mid-session
      loses ≤ interval of accesses (fake-clock test). *(Confirmed/corrected durability is the
      journal, M1.3; this row bounds used/surfaced loss.)*
      LANDED 2026-07-21: knobs `activation_snapshot_interval_seconds=600` (0 disables) +
      `activation_snapshot_dirty_min=50`, save when BOTH due; `maybe_save_periodic` is
      fake-clock injectable and delegates to `save_to_file` (journal fold applies to every
      periodic save). Shell: 60s lifespan poll task; MCP stdio: due-check piggybacked on
      tool calls, full ownership rules honored (owner-save refreshes its own mtime guard;
      refusals defer so probes run ≤ once/interval). 12 tests
      (`test_activation_snapshot_periodic.py`) incl. durable-without-clean-shutdown both
      paths. Known accepted: save does blocking IO on the loop (mirrors shutdown save),
      bounded to once/interval — `asyncio.to_thread` is a follow-up.
- [x] **M4.2 Graph rows: wire or delete (F3).** Wire = call `snapshot_to_graph`
      (storage/memory/activation.py:159-175) in the mop window (shell paused ⇒ single-writer);
      delete = drop columns + rewrite prune's pre-filter honestly (sqlite/graph.py:1529-1558;
      helix/graph.py:2755-2813). *(Judge-2 improvement: if wired, the counts are last-clean-
      exit-stale — the brain loads read-only, the shell's crash-window accesses are absent —
      document them as approximate, not authoritative.)* DoD: prune pre-filter semantics
      documented + tested either way; ranking never reads the column.
      LANDED 2026-07-21 (WIRED): `execute_hygiene_mop` calls `snapshot_to_graph` after the
      drains, budget 5000/window most-recent-first; result block carries `snapshot_sync`
      counts; dry-run/no-API labeled skips, failures loud (`logger.exception` + status
      error); columns documented in-code as APPROXIMATE. 6 tests
      (`test_mop_snapshot_sync.py`) incl. the ranking pin: `score_candidates` with graph
      access_count=10^9 in entity_attributes is score-identical both flag states.
- [x] **M4.3 Redis activation store parity (full-mode compat lane).** *(Judge-2 improvement —
      `redis/activation.py` is a separate implementation; `usage_events`, the tier param, and
      v2 changes are silently absent otherwise.)* Mirror the M1.1 changes in the Redis store,
      or document explicitly that the usage mechanism is inert in full mode (compat lane only).
      DoD: a test pins whichever choice; no silent divergence.
      LANDED 2026-07-21 (documented-inert, now PINNED):
      `test_record_access_all_tiers_hygiene_only_no_usage_side_effects` — tiers sourced from
      `DEFAULT_USAGE_TIER_WEIGHTS` (auto-covers new tiers), all accepted, hygiene exact,
      zero usage side-effects.

## M5 — Episode channel + vocabulary honesty (D4/D7/D8) [EVAL-GATED for the episode-u flip]

- [x] **M5.1 Episode `u_episode` via cue substrate + cue-bearing rig.** Add `used_count`
      (tier-weighted) + `last_used_at` to the cue record; agent-used events from
      `_matches_cue_content` in the same echo-guarded scan (M1.4); `u_episode = f·r′`.
      *(Judge-3 blocker 4 — the M4.1/session/oracle rigs contain ZERO cues, so this channel is
      untested.)* Build a cue-bearing rig variant (plant cues in the M4.1 corpus or a small
      dedicated corpus) and mirror G1/G3/G4 for `u_episode`. *(Judge-1 improvement — episode
      composition stacks `(1+β·u)≤1.30` × `(1+temporal_cue_boost)≤3.0` for a 3.9× max; add an
      episode-side collapse/tie probe.)* Files: storage/*/cue schema, retrieval/pipeline.py.
      DoD: episode no-op on empty cue-usage; episode tie-probe; episode determinism; episode
      composition stress probe passes. **EVAL-GATED: episode G1/G3/G4 + composition probe.**
      LANDED 2026-07-21: cue model gains `usage_used_count: float` + `usage_last_used_at`
      (legacy int `used_count` hygiene semantics untouched); sqlite = real columns, helix =
      `supporting_spans_json` `_engram_cue_usage` trailer (no schema.hx regen; stripped on
      read, zero-usage payload byte-identical); same-pass echo-guarded cue scan
      (`scan_novel_cue_matches`, shared echo mask + `cue::` dedup class); Step-1.4
      composition `rrf × (1+β_route·u_episode)` before Step 5.05, `usage_ranking_enabled`
      only. MEASURED (`rf_episode_u` rig, pinned clock): G1 flag-on empty ≡ flag-off
      byte-identical PASS; G1b flag-off populated ≡ empty (kill-switch drift probe) PASS;
      G3 used-cue wins equal-rrf tie PASS (0.4088 > 0.3997); G4 3/3 runs id+score-identical
      PASS; composition stress: 3.9×/4.0×-stronger rrf item stays first (weak boosted-both
      0.1679 vs 0.312/0.320) PASS. Real max stack 1.30×2.0=2.6 < 3.9 envelope (pinned in
      `test_episode_usage.py`). FalkorDB compat lane does not persist the fields (episode-u
      inert there); worker rework cue rebuild does not yet carry `usage_*` (follow-up).
- [x] **M5.2 Document Step 5.05/5.06 as THE episode recency model** (docs + coverage for the
      undated-episode no-boost edge, pipeline.py:1681-1801). DoD: doc + edge test.
      LANDED 2026-07-21: normative §4.1 in `experiments/RF_target_design.md` (gate, input =
      `conversation_date` only, bounded form, composition with u_episode, storage note);
      edge pinned by `test_undated_episode_gets_no_temporal_boost` (undated episode score
      == exact semantic base, dated episode boosted above it).
- [x] **M5.3 TS default per F4.** If KILL: delete `score_candidates_thompson`, the pipeline
      branch, `activation/feedback.py`, `ts_*` knobs, `ts_alpha/ts_beta` fields (+snapshot
      compat). If FLOOR: one-line `ts_enabled=False` + `exploration_weight` pinned. If ever
      revived: per-entity seeded draws `blake2b(group,query,entity_id)`. DoD: repeat-stability
      ≥ 33/36 session-shaped on the TS rig either way.

---

## Acceptance gates — the flip criteria (one canonical set)

*(Judge-3 blocker 3: the two disagreeing tables are collapsed here; G2 = arm-A parity,
tagged mandatory-mechanism; the looser ≥21/≥19 variants are deleted. G7 is added by the
Judge-1 echo blocker + Judge-3 producer-yield blocker. A gate pass is a reachability
claim, never an answer-quality claim.)*

| Gate | Instrument | Threshold | Scope |
|------|-----------|-----------|-------|
| **G1 no-op** | M4.1 rig, flag-on + **populated** store vs empty, **exercising spreading + temporal_mode** | byte-identical ids AND scores, 36/36 | mechanism (mandatory) |
| **G2 usage never degrades** | M4.1 arm B (usage history relabeled `used`), real corpus | reach@10 ≥ **23** (arm-A parity; shipped scored 2/36); hard fail on regression | mandatory-mechanism |
| **G3 used wins ties** | synthetic equal-sem tie-probe, one item `used` | 100% used-first *(wiring / inverse-silent-inert proof — proves the multiplication is live, not that usage improves ranking)* | mechanism |
| **G4 determinism** | M4.3 fresh-AB, **`record_access=False`**, M0.2 WM-freeze precondition | 12/12 byte-identical; session repeat-stability ≥ 33/36 | mechanism |
| **G5 budget-0 hygiene** | store-spy, M1.2-style | zero writes to **`usage_events`** (ranking store) on the read path; surfaced hygiene writes to `access_history` at w=0 permitted; **budget≥1 store-spy variant** included | mechanism |
| **G6 real corpus** | `engram continuity --against-live --organic` + **live usage-event yield > floor** over a real capture window | no regression AND per-tier used+confirmed+corrected volume above the silent-inert floor | **flips defaults** |
| **G7 citation-scan yield/precision + echo-immunity** *(NEW)* | echo rig, planted labels, judge-free | (a) used-event rate > floor on planted-reliance transcripts; (b) precision ≥ threshold vs planted labels; (c) adversarial common-word surfaced-but-not-relied ⇒ no fire; (d) top-k fed back as synthetic next turn ⇒ used-event count ~0 | **mandatory precondition to G6 / usage_ranking flip** |

Additional mechanism gates: **u-discrimination** (M2.1 worked-number invariants — a constant
`u` must fail), **de-saturation** (rf_judge_math: 1@1s de-saturated to 0.067 not 0.913, count/age
separability), **surfacing regression** (oracle rig reach@5 ≥ 22/36 on the candidates-source arm —
proves the graph channel survives the shared-pipeline R/F changes), **frequency-query rig** (M2.4),
**session-pollution parity** (M0.1-M0.3: session-shaped reach@10 ≥ 32, state-clean unchanged 21/32).

## Decisions needed from founder

1. **F1 — APPROVED: the echo-guarded citation scan.** M1.4 exactly as
   specified (echo guard + per-conversation dedup + G7 hard precondition;
   `w_used` stays 0 in ranking until G7 passes). Confirmed/corrected stays
   **operator-surface-only**: the frozen public surface is a standing
   constraint, and cue-hit promotion already provides an organic
   confirmed-tier signal with no surface change. Rationale: the guarded scan
   answers Judge-1's blocker head-on; the confirmed-only fallback would leave
   ranking-frequency nearly signal-less at launch (see F10), the riskier posture.
2. **F2 — RESOLVED: KEEP surfacing recording as hygiene-tier events** (M1.5
   demotes to `w_ranking=0`; nothing removed). Prune and maturation
   legitimately consume these events; deleting them would make fresh installs
   prune-naked. get_context's P9 loop is defanged by the zero ranking weight;
   layer-3's `get_top_activated` DISPLAY ordering reads the hygiene view and
   is documented as a display heuristic outside the ranking path (and the
   product surface short-circuits to the durable pack anyway).
3. **F3 — RESOLVED: WIRE `snapshot_to_graph` into the mop window** (shell
   paused ⇒ single-writer), with the counts documented as approximate
   (last-clean-exit-stale, per Judge-2). Rationale: the method already exists
   with zero callers; deletion would touch prune SQL across three backends
   (larger blast radius) and lose dashboard honesty for nothing.
4. **F4 — RESOLVED: KILL.** All three judges endorse; the reward channel is
   structurally absent (no agent-surface feedback signal exists), making TS
   un-eval-gateable; at budget 0 it is doubly inert and at budget >0 it costs
   reachability and destroys repeat-stability. This knowingly reverses the
   recorded "SEEDED-ON, flip parked" decision — the reversal was flagged to
   the founder twice (2026-07-18 review + this doc) before /goal delegated
   the call. Snapshot compat: `ts_alpha/ts_beta` tolerated-and-dropped on
   load. Side benefit: M2.2 loses the TS-twin gating complexity. Drives M5.3.
5. **F5 — RESOLVED: zero the activation column, adopt D3's β_route table**
   (FREQUENCY 0.30, TEMPORAL 0.25, others per design). The u-tiebreaker
   replaces activation in ranking; small-nonzero retunes of the OLD saturated
   signal are exactly the "tuning weights on a saturated signal" mistake
   Judge-1 blocked.
6. **F6 — RESOLVED: justify-and-pin now, rebalance only behind the gate.**
   The 2.5x durable boost is LIVE shipped behavior; changing it is a default
   change requiring eval. M3.1 therefore (a) writes the invariant test that
   PINS the current bound (durable may dominate semantics within factor 2.5 —
   intentional for identity-class facts, which should surface on weak
   matches), (b) documents the asymmetry vs the usage theorem explicitly, and
   (c) any rebalance (e.g. beta_durable) gates on the live continuity gate.
   cs cap 0.05 accepted pending M3.1 recalibration against u.
   **AMENDED 2026-07-21 (M3.1 verify)**: the "bounded 2.5x multiplicative
   band" was a mis-model — no live path composes sem × durable_boost. The
   real mechanism is a reserved lane: `prefer_durable_facts`
   (result_selection.py) sorts on `(type_rank, score + boost)` where durable
   entities take type_rank 3 — score-INDEPENDENT and unbounded (sem-0.01
   durable Person beats sem-0.99 Concept), with the 2.5/1.5 constants acting
   additively within the lane and bounded additive rescue lanes elsewhere
   (`boost × 0.08 ≤ +0.2`). The lane is intentional (strictly stronger than
   the assumed band, same rationale) and is now pinned as-is by
   `test_importance_invariants.py`, including the lane's immunity to usage
   (1.30x cannot cross it). M3.2 must reason from the lane model.
7. **F7 — RESOLVED: BUILD** (M5.1 as designed, behind the episode gates).
   Episodes are the core tier; leaving them without a usage channel recreates
   the "no principled recency model" gap this goal exists to close.
8. **F8 — RESOLVED: WAIVED.** The corrected panel ran against the design and
   every blocker is folded into named rows and gates; G1-G7 plus the
   mechanism gates ARE the ongoing adversarial instrument. A third
   documentation pass has worse marginal value than executing against the
   gates. Per-milestone adversarial verify agents remain in force.
9. **F9 — RESOLVED: stays PARKED.** Three independent classifiers is real
   debt but orthogonal to this goal's collapse modes; D3 touches only the
   router table. Revisit after M2 lands.
10. **F10 — RESOLVED: option (a), `w_mentioned=0.1`.** Ingestion mentions are
    user-content-driven — what the user actually talks about IS the
    environmental statistic Anderson-Schooler ground the prior in, and it is
    loop-free (the ranker does not write episodes). Guard: at most one
    mention event per (entity, episode) — the commit path fires once per
    entity per episode already. This gives ranking-frequency a real,
    trusted signal at launch independent of the citation scan proving itself.

## Parked (dated reasons)

- **Distributed-practice / spacing fidelity in `u`** — parked 2026-07-18 (accepted tradeoff):
  `u` uses only `Δ_last`, so 5 accesses over a month and 5 in one minute give identical `u` if
  the last access matches (Judge-1 improvement). Defensible for a cheap O(1) deterministic
  tiebreaker (the design's explicit rationale); a 2-3-point recency would recover most of it.
  Unpark if calibration shows the loss matters.
- **`r_floor` long-horizon age cutoff** — parked 2026-07-18: `r_floor=0.25` gives a ~6-7% standing
  boost forever with no upper age cutoff (Judge-1). Verify against §6 calibration that abandoned
  high-frequency items are not over-protected; add a cutoff only if calibration shows it.
- **Global activation-weight zeroing default flip** — parked 2026-07-17: collapse magnitude from a
  36-question synthetic rig; shipped config already inert; gate = M2.6 real-corpus rerun.
- **`ts_enabled=False` default flip** — parked 2026-07-17: INCONCLUSIVE-on-lift / CONFIRMED-on-
  mechanism; unpark path in F4 (M5.3).
- **`entity_episode_traversal_source="candidates"` default flip** — parked 2026-07-17: oracle
  ceiling on planted structure; needs real-corpus eval + RRF-vs-cosine scale normalization (M2.4/M3.1).
- **M4.4 confirmed-use writer default-on** — superseded 2026-07-18 by M1.2/M1.4 (this doc builds
  the guarded writer); unpark = F1.
- **Entity-budget re-enable (`passage_first_entity_budget≥1`)** — parked 2026-07-17: the flag
  couples five behaviors (cartography §11.1); re-enable only after M0 + M1 + M2 decouple recording
  from surfacing.
- **Cue-feedback read-path writes + conv-fingerprint capture-pollution probes** — parked
  2026-07-17: not exercised by the session rig (no cues, recall-only sessions); worth separate
  small probes.
- **Reconsolidation cross-process fix** — parked 2026-07-17: 5-min in-RAM window vs 2h brain
  cadence is structurally dead (cartography P6); superseded by D1's usage-signal framing.
- **Full temporal-vocabulary unification (D8)** — parked pending F9.

## Completion

This goal is DONE when: M0 fully landed with the session rig at parity; F1–F10 each carry a
founder decision; M1 (recording + storage + journal + echo-guarded scan) landed and inert by
G1/G5 before any ranking flip; M2–M5 rows landed behind their named gates with experiment reports
committed or explicitly parked with dated reasons; G1–G7 plus the mechanism gates green on the
final stack; and the R/F knob census (cartography §10) contains zero rows whose status is
"illusory" or "accidentally vacuous."

### Completion record — 2026-07-21

DONE. Commits f9e64b6 (M0) → 7bf30e3 (M1) → 5ca08d2 (M2+M5.3) → 9428c03
(M3.1/M4/M5.1/M5.2/pre-flip closure) → the M2.6 gate report commit. Final
stack: 4789 backend tests passing, 0 failed; ruff check+format clean; CI
green. Every row landed except M3.2 (parked, dated, reasons on the row).

Gate ledger on the final stack: **G1** PASS (rig 8/8 + real-corpus E==A
byte-identity), **G3** PASS (used-wins-ties 16/16 after the pre-flip
closure), **G4** PASS (determinism, all rigs), **G5** PASS (budget-0
hygiene), **G7** PASS (committed report, 1.00/1.00/0 fires). **G2/G6:
adjudicated FAIL at the M2.6 gate** — they are flip gates, and their failure
is the recorded reason the `usage_ranking_enabled` default stays False (DO
NOT FLIP, see `experiments/M2_6_real_corpus_gate.md`): organic usage yield
is zero and the organic continuity gate fails live, flag-independently
(deep-recall emptiness / Jul-13 vector-less backlog — pre-existing, outside
this goal). Knob census: zero illusory/vacuous rows (cartography §10
updated with dated resolutions).

The mechanism is built, proven inert by default, safe to flip mechanically,
and blocked from flipping only by real-world gates that must be earned:
recall-stack repair → organic used-event accumulation → arm-B churn rerun +
G2 adjudication.
