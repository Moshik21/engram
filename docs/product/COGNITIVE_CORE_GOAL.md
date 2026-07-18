# Cognitive Core Goal — make the designed system and the shipped system the same system

**Status:** EXECUTING — approved 2026-07-17 (/goal); M1+M2 landed and pushed
(8a4797b, f984b91), M3 code rows landed behind flags (supersession_enabled,
importance_prior_enabled, hygiene_mop_merge_enabled — all default off; M3.2
vocabulary is live), I1-I3 filed under docs/product/investigations/ with the
I2 race fixed. Remaining: M3.1 oracle experiment, M4 eval arms, release tag.
Notable premise correction from M3.4: exclusive-predicate supersession for
location/employment classes already shipped unconditionally — the flag gates
the CLASS EXPANSION (USES_VERSION/NAMED/PREFERS) and provenance recording.
**Created:** 2026-07-17
**Sources:** `COGNITIVE_CORE_REVIEW_2026-07-17.md` (findings, file:line
evidence), `COGNITIVE_CORE_FIXES.md` (deliverable detail). This doc is the
work order; read those for the why.

## Objective (one sentence)

Every cognitive mechanism Engram ships is either **live and verified on the
default install**, **eval-gated behind an explicit experiment**, or
**deleted** — no more computed-but-unconsumed cognition, no more
theory-in-name-only.

## Decisions (resolved 2026-07-17, founder-delegated best judgment)

1. **Schema formation: DELETE.** Force-disabled even in standard, in-code
   admission that no retrieval path reads its output, 445 lines + 29 tests
   of pure maintenance. Git history preserves it; resurrect if a consumer
   is ever designed. (Basis: review §2 table, config.py standard-branch
   comment.)
2. **Tier system: COLLAPSE, keep the working core.** Keep `mat_tier` +
   differential decay in scoring (the load-bearing part) and the
   identity_core→semantic auto-promotion (the only graduation that works).
   Delete the maturation and semanticize scanning phases and their windows
   — four independent defects each prevent graduation; the machinery never
   earned its complexity. W3.3 (importance-as-prior) becomes the second
   road to durable ranking. Revisit only if a W4 eval arm shows tier
   graduation would have earned its keep.
3. **Thompson sampling: SEEDED-ON.** Fix the negative-feedback flood, seed
   the RNG for deterministic recall, keep `ts_enabled=True` until the W4.3
   arm decides the default. Rationale: at entity budget 0 TS cannot affect
   surfaced results, so seeded-on is behavior-preserving today, and it
   avoids changing ranking configs ahead of the eval.
4. **Waves 1+2: proceed without further per-row review** once this doc is
   approved. Wave 3 rows land behind config flags or experiments as marked;
   Wave 4 arms report before any default flips.

## Constraints (unchanged, binding)

- Fully-local north star: no new external-key dependencies; extraction
  ladder stays agent → Ollama → narrow.
- Public MCP 9-tool surface FROZEN.
- Live-brain safety: never multi-open `~/.helix/engram-native-dogfood-axi`;
  no engram commands that open the live dir while the shell runs; tests use
  the fakehome pattern; read-only HTTP to 127.0.0.1:8100 is allowed.
- Storage silent-swallow contract stays green; every new tolerated failure
  carries `# silent-ok: <reason>`.
- Commit in logical stacks, each stack's targeted tests green before the
  next; no `git reset --hard`.

## Global verification gates (apply to every milestone)

- `cd server && uv run ruff check . && uv run ruff format --check .`
- `env HOME=<fakehome> ENGRAM_MODE=lite uv run pytest -m "not requires_helix" -q`
  → 0 failed.
- Native subset (helix id-resolution, transport hardening, swallow contract,
  vector serialization, consolidation store) → no new failures vs baseline.
- One live bounded mop window green after each wave that touches
  brain/consolidation code (`engram brain run --tier mop --budget 100
  --deadline-seconds 600`, `--force` only if on battery and actively
  supervised).
- Rows marked EVAL-GATED additionally require their experiment report
  committed under `docs/product/experiments/` before the default flips.

---

## M1 — Stop the live bleeding (bug-class; no design change)

- [x] **M1.1 Prune sees usage.** Load the activation snapshot in the brain
      path and in `engram mcp` stdio; save on clean MCP exit. The shell
      writes the snapshot at shutdown (main.py:473) and the brain pauses
      the shell first, so a fresh file exists at mop start — it is simply
      never loaded (storage/bootstrap.py, brain_cli.py:400-415,
      mcp/server.py:305). DoD: brain-path store warm after open (unit test);
      live mop log reports nonzero loaded entries; prune.py:76-99
      protections exercised in a test with a populated snapshot.
- [x] **M1.2 Thompson sampling: seed + stop the flood.** No feedback
      recording when entity results were discarded by budget
      (pipeline.py:2060-2071); RNG seeded (scorer.py:231) from a stable
      input. DoD: identical scores on repeated identical recall (test);
      zero activation writes on budget-0 recall (store-spy test).
- [x] **M1.3 Fail-closed dream/microglia.** Identity-core fetch failure or
      connectivity-check failure skips the destructive pass for the cycle
      (dream.py:381-387, 538-548; microglia.py:116-119), logged + tagged.
      DoD: raising-store unit tests show LTD/associations/demotion skipped.
- [x] **M1.4 Replay: terminal state + routing respect.** Empty extraction
      flips a terminal marker; selector skips episodes parked by worker
      routing and bootstrap docs (replay.py:261-267, 427-452;
      worker_routing.py:99,127,162; project_bootstrap.py:455-461). DoD:
      parked/replayed-empty episodes not re-selected (tests); next live mop
      replay count reflects real work only.
- [x] **M1.5 Hot-path guards.** Spreading re-score awaits degrade with
      marker instead of killing recall (pipeline.py:1305-1314). DoD:
      raising-store test returns results with degraded-stage marker.
- [x] **M1.6 Scorer contract honesty.** Merge/infer embedding failures
      tagged + counted + visible in cycle results (merge_scorer.py:368,381,
      412; infer_scorer.py:190-204,247). DoD: scorer tests; embedding-absent
      counter appears in cycle summary.

## M2 — Cheap correctness + honesty

- [x] **M2.1 Cue auto-recall gate.** Resolve 0.26-ceiling < 0.3-gate
      (exempt or rescale; document intent). DoD: strong-cue test surfaces
      (or documented never-auto-surface decision + dead search removed).
- [x] **M2.2 entity_type normalization at commit.** DoD: lowercase
      extractor output gets typed semantics (test).
- [x] **M2.3 mat_tier single source of truth.** Model field added; native
      column round-trips (models/entity.py, helix/graph.py:588,859). DoD:
      tier persists across native write/read (test). (Survives decision 2 —
      identity_core promotion still writes tiers.)
- [x] **M2.4 Artifact merge exemption + class filter.** Artifact/Schema
      types exempt from fuzzy merge (merge.py:822); `artifact_class`
      filters `search_artifacts` (artifacts.py:158). DoD: same-name
      two-project artifact test stays unmerged; class-filtered search test.
- [x] **M2.5 Deletion batch.** feedback_events store, spreading_bonus,
      NeuroplasticityEngine, update_arousal_ema + knob, `_TIER_PHASES
      ["mop"]`, ClarificationIntent graph-entity minting, ~17 dead knobs
      (review list), **schema formation phase + tests (decision 1)**,
      **maturation + semanticize phases (decision 2)** with their config and
      the quiet/standard flags that enable them; CLAUDE.md/README claims
      updated to match. DoD: full suite green; grep proves no dangling
      references; docs updated in the same stack.
- [x] **M2.6 DecisionMaterializer noise stop.** Human-shaped names, dogfood
      predicates behind config (decision_materializer.py:242-259). DoD: no
      `:decision_statement:` names minted (test); suppression regexes in
      promotion.py become dead-lettered (left in place, one release).

## M3 — Design corrections (each behind flag or experiment)

- [ ] **M3.1 Graph→episode surfacing.** (a) Oracle experiment: feed
      `append_entity_linked_episodes` from top-K entity candidates on the
      depth-eval; report reachability/accuracy (docs/product/experiments/).
      (b) If positive: productionize behind `entity_episode_traversal_source
      ="candidates"`, fix RRF-vs-cosine scale sharing in the same stack
      (episode_traversal.py:127-136; pipeline.py:1308-1314). EVAL-GATED.
- [x] **M3.2 Durable types via local extraction.** Decision/Preference/
      Commitment/Correction in the LLM prompt vocabulary AND narrow
      patterns (extraction/prompts.py:21-23; narrow/entity_extractor.py:
      385-407). DoD: per-type extraction tests; organic-capture probe
      produces a Decision from natural text with zero client proposals.
- [x] **M3.3 Importance as a prior.** `consolidated_strength` set at write
      time for remember()/identity/durable commits; bounded; compact's
      absorbed mass decays (engine.py:29-34; compact.py:103-134). DoD:
      arithmetic test (one-shot durable ≈ 5-access mundane at 30d); no
      regression on the eval arm. EVAL-GATED for default-on.
- [x] **M3.4 Supersession (strategic bet).** Commit-time contradiction
      detection for narrow high-value classes (location, preference,
      version, name) sets `valid_to` on the superseded fact; current-value
      preference at retrieval. DoD: "moved to Denver" unit family; depth
      -eval knowledge-update slice report. EVAL-GATED for scope expansion.
- [x] **M3.5 Bounded merge in the mop.** Exact-identifier +
      high-confidence-name dedup slice inside `execute_hygiene_mop`,
      budgeted (hygiene_ops.py). DoD: live window merges seeded dup pairs
      within budget and deadline.

## M4 — Instrument the theory

- [ ] **M4.1 Activation arm** (`record_access=True` — adapter.py:431-436).
      Report per-question-type deltas → ACT-R stays a ranking term or
      becomes telemetry.
- [ ] **M4.2 Surfacing arm** = M3.1(a); doubles as the Gate-G revisit with
      a mechanism that can actually move episodes.
- [ ] **M4.3 TS arm** (seeded-on vs off) → decides `ts_enabled` default
      (decision 3 sunset clause).
- [ ] **M4.4 Confirmed-use access recording** for the eval arm (feedback
      path exists — retrieval/feedback.py:183) so the recency/frequency
      prior reflects usage, not ranker output. Design note first; EVAL-GATED.

## Investigations (subagent-sized, run alongside any milestone)

- [x] **I1 get_context/context_builder adversarial audit** (highest-traffic
      surface; records access at context_builder.py:851,1145).
- [x] **I2 Evidence drains vs corroboration gate** (hygiene_ops.py:148-215
      vs the count≥2 Layer-3 promotion requirement).
- [x] **I3 MCP concurrent-open** — targeted concurrency test for two stdio
      sessions on one native dir; then design (advisory lock or shared
      open). No fix landed without the test first.

## Completion

This goal is DONE when: M1+M2 fully checked; M3 rows each either landed
behind their gate with experiment reports committed, or explicitly parked
with a dated reason; M4 arms reported with a decision recorded per arm;
I1-I3 reports filed; all global gates green on the final stack; and
CLAUDE.md + README describe only mechanisms that actually run.
