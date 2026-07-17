# Cognitive Core Fixes — execution objective

**Status:** DRAFT — pending founder review
**Created:** 2026-07-17, from `COGNITIVE_CORE_REVIEW_2026-07-17.md` (all
findings there are CONFIRMED with file:line unless marked SUSPECTED)
**Owner:** founder + coding agents

Principles: fix the live install first; behavior changes that could move eval
numbers are flagged EVAL-GATED and land behind config or an experiment;
deletions of dead machinery are batched and reversible (git). Every row lists
its files and its verification. Live-brain safety rules apply throughout
(never multi-open the native dir; shell/brain discipline unchanged).

---

## Wave 1 — Stop the live bleeding (bugs; no design changes)

- [ ] **W1.1 Prune sees usage: load the activation snapshot in the brain.**
      The brain pauses the shell before running (shell shutdown saves the
      snapshot at main.py:473), so a fresh snapshot exists on disk when the
      mop starts — the brain just never loads it. Add optional snapshot load
      to `open_local_stores` (storage/bootstrap.py) or the brain path
      (brain_cli.py:400-415) so `MemoryActivationStore` starts warm;
      prune.py:76-99 protections then apply. Also load in `engram mcp` stdio
      (mcp/server.py:305) — the flagship path currently starts blank — and
      save on clean MCP exit.
      Files: storage/bootstrap.py, brain_cli.py, mcp/server.py,
      storage/memory/activation.py. Verify: new test — brain-path store has
      entries after open when snapshot exists; live mop log shows nonzero
      loaded count.
- [ ] **W1.2 Thompson sampling: seed it and stop the negative flood.**
      (a) Skip TS feedback entirely when the entity budget discarded the
      results (`returned_ids` empty because `passage_first_entity_budget=0`)
      — pipeline.py:2060-2071; today every recall records negatives for
      every candidate and does pool-sized activation writes on the read
      path. (b) Seed the sampler (scorer.py:231 `random.Random(None)`) from
      a stable input (config seed or query hash) so recall is deterministic.
      Whether `ts_enabled` should default False entirely is EVAL-GATED
      (W4.3).
      Files: retrieval/pipeline.py, retrieval/scorer.py, config.py.
      Verify: same query twice → identical scores; recall with budget 0
      performs zero activation writes (assert via store spy).
- [ ] **W1.3 Fail-closed protection in dream/microglia.** Identity-core
      fetch failures currently blank the protected set and proceed
      (dream.py:381-387, microglia.py:116-119); a failed connectivity check
      means "assume disconnected" → maximal surprise → MORE speculative
      edges (dream.py:538-548). On failure: skip the destructive pass for
      that cycle and log; tag per the silent-swallow contract.
      Files: consolidation/phases/dream.py, consolidation/phases/microglia.py.
      Verify: unit tests with a raising store — LTD/associations/microglia
      demotion do not run; contract test stays green.
- [ ] **W1.4 Replay: stop the treadmill, respect routing.** (a) On empty
      extraction, flip a terminal marker (e.g. `projection_state=
      "replayed_empty"` or set last_projection_reason) instead of leaving
      the episode eligible forever (replay.py:261-267 vs 427-452). (b) The
      selector must respect `last_projection_reason` and source: skip
      episodes parked CUE_ONLY by worker routing (auto-capture floor, triage
      skip, meta-discourse — worker_routing.py:99,127,162) and bootstrap
      docs (project_bootstrap.py:455-461). The 2026-03-20 Layer-3 fix
      depends on this.
      Files: consolidation/phases/replay.py. Verify: unit tests — parked
      episode not selected; empty-extraction episode not re-selected next
      run; live mop replay counts drop to real work only.
- [ ] **W1.5 Guard the hot-path awaits.** Step 4.5 spreading re-score
      (pipeline.py:1305-1314) has no try/timeout — post-hardening, one
      storage raise kills the whole recall. Catch NativeQueryError →
      degrade with an explicit marker (lookup.py house style).
      Files: retrieval/pipeline.py. Verify: unit test — raising
      similarity/store degrades spreading stage, recall still returns.
- [ ] **W1.6 Merge/infer scorers: honor the storage contract.** The bare
      excepts that zero embeddings (merge_scorer.py:368,381,412) and the
      neutral-0.5 on missing embeddings in infer (infer_scorer.py:190-204,
      247) re-swallow the hardened raises one layer up. Tag with
      `# silent-ok:` where tolerance is justified; count and log embedding
      -absent scoring so a broken embedder (cf. Jul 13–17 outage) is visible
      in cycle results. Making infer *block* on missing embeddings is a
      behavior change → W3.4.
      Files: consolidation/scorers/merge_scorer.py, infer_scorer.py.
      Verify: targeted scorer tests; grep-level contract if we extend the
      static test beyond storage/.

## Wave 2 — Cheap correctness and honesty (small, mostly config/model)

- [ ] **W2.1 Cue auto-recall gate is unsatisfiable.** Ceiling 0.40×0.65 =
      0.26 < `auto_recall_min_score=0.3` (pipeline.py:789; config.py:149,
      694, 1783) — cues are searched then always filtered off the auto
      surface. Either exempt cue results from the min-score gate or rescale.
      Decide intent first: was the cue layer ever meant to auto-surface?
      Files: retrieval/auto_recall.py or config.py. Verify: unit test with a
      strong cue → surfaces (or a documented decision that it must not).
- [ ] **W2.2 Normalize `entity_type` at commit.** Open case-sensitive
      string; lowercase writes are invisible to every typed behavior
      (durable boost, packets, merge gate). Normalize to the canonical
      TitleCase set at commit (extraction/commit_policy.py or
      graph_manager write path); keep unknown types as-is but cased.
      Files: extraction/commit_policy.py (or equivalent), one migration-free
      normalization on read for comparisons. Verify: lowercase "person" from
      an extractor gets durable-type semantics in a unit test.
- [ ] **W2.3 `mat_tier`: one source of truth.** Add the field to the Entity
      model so the native column stops silently defaulting to "episodic"
      both directions (models/entity.py:13-50; helix/graph.py:588,859), OR
      delete the column and keep attributes-JSON as the only store.
      Recommend: add the model field — cheap, and the column already exists.
      Files: models/entity.py, storage/helix/graph.py. Verify: roundtrip
      test — tier set by maturation persists and reads back from native.
- [ ] **W2.4 Artifact disambiguation + merge exemption.** (a) Exempt
      `entity_type in {"Artifact", "Schema"}` from fuzzy merge (merge.py:822
      currently exempts only identity_core) — two projects' README.md must
      not merge (SUSPECTED-strong finding #7 in the review). (b) Make
      `artifact_class` a real filter in `search_artifacts` (artifacts.py:158)
      so conversation-provenance records stop polluting file search.
      Files: consolidation/phases/merge.py, retrieval/artifacts.py.
      Verify: merge test with two same-named artifacts in one group → kept
      separate; artifact search filtered by class.
- [ ] **W2.5 Dead-machinery deletion batch.** Delete: `feedback_events`
      store (storage/sqlite/feedback.py — zero call sites),
      `ActivationState.spreading_bonus` (never written),
      NeuroplasticityEngine (no consumer), `update_arousal_ema` + its knob,
      `_TIER_PHASES["mop"]` dead entry (brain_cli.py:43-48), ClarificationIntent
      entity creation (write-only — keep the adjudication request row,
      drop the graph entity, adjudication_service.py:131-147), and the ~17
      dead config knobs listed in the review. DECISION NEEDED: schema
      formation (445 lines + 29 tests, force-disabled, no consumer) — delete
      now or keep dark pending a retrieval consumer.
      Files: as listed. Verify: full lite suite green; ruff; grep proves no
      dangling references.
- [ ] **W2.6 DecisionMaterializer: generalize or gate.** It hardcodes
      Engram-dogfood predicates (decision_materializer.py:242-259) and mints
      the `X:decision_statement:Y` names that promotion.py:115-122 then
      suppresses. Short term: stop minting the noise-shaped names (compose a
      human name), keep the dogfood predicate list behind config. Long term
      belongs to W3.2 (durable extraction vocabulary).
      Files: retrieval/decision_materializer.py. Verify: existing decision
      tests + no `:decision_statement:` names created.

## Wave 3 — Design corrections (EVAL-GATED; each behind config or experiment)

- [ ] **W3.1 Graph→episode surfacing (the binding-lever bet).** The
      traversal exists and is default-enabled but feeds from final results,
      which entity-budget 0 empties (episode_traversal.py:27-76, 127-136).
      Step 1 (experiment, ~a day): oracle-surface run — feed
      `append_entity_linked_episodes` from the top-K entity *candidate pool*
      on the depth-eval set; measure reachability + accuracy deltas. Step 2:
      if positive, productionize behind `entity_episode_traversal_source=
      "candidates"` and fix the RRF-vs-cosine scale mismatch while in there
      (pipeline.py:1308-1314 vs helix/search.py:100-106 — normalize or
      bucket the two score families before they share `weight_semantic`).
      Files: retrieval/episode_traversal.py, retrieval/pipeline.py,
      evaluation harness. Verify: experiment report first; then A/B flag.
- [ ] **W3.2 Durable types reachable by local extraction.** Add Decision/
      Preference/Commitment/Correction to the LLM extraction prompt
      vocabulary (extraction/prompts.py:21-23) and teach the narrow
      extractor the cheap patterns ("I prefer", "we decided", "actually,
      it's X not Y" — narrow/entity_extractor.py:385-407). Without this the
      durable machinery acts on an empty set for organic capture.
      Files: extraction/prompts.py, extraction/narrow/entity_extractor.py.
      Verify: extraction unit tests per type; organic gate probe finds a
      Decision extracted from a natural sentence, not a client proposal.
- [ ] **W3.3 Importance as a prior.** Set `consolidated_strength` at write
      time for remember()/identity/durable commits (it already sits inside
      the ACT-R ln-sum, engine.py:29-34) — the principled replacement for
      the five ad-hoc guard rails. Bound it, and decay compact's absorbed
      mass (compact.py:103-134) so it stops being a monotone accumulator.
      Files: ingestion write path, activation/engine.py,
      consolidation/phases/compact.py. Verify: one-shot remember() fact
      retains ranking parity with a 5-access mundane entity at 30 days
      (unit-level arithmetic test); eval arm unchanged-or-better.
- [ ] **W3.4 Tier system: make it real or shrink it.** Fix window
      starvation with a scan cursor (maturation.py:77-80, semantic_
      transition.py:42-45); stop bumping `updated_at` on unchanged scans
      (maturation.py:132-144); recalibrate maturity components to personal
      scale (8 distinct predicates → ~3; CV-based regularity → burst-tolerant)
      OR collapse the tier system to {episodic, identity_core-semantic} and
      delete maturation/semanticize. DECISION NEEDED: which direction —
      recommend collapse unless W4 evals show tier decay earning its keep.
      Files: consolidation/phases/maturation.py, semantic_transition.py,
      maturity_features.py, config.py. Verify: on a dogfood-sized fixture,
      a 6-month-old 5-episode entity graduates (if kept); prune resistance
      reflects tier.
- [ ] **W3.5 Supersession / bi-temporal current-value (the strategic bet).**
      Write-path invalidation: when a committed fact contradicts an existing
      one (same subject+predicate class), set `valid_to` on the superseded
      edge/entity attribute instead of accumulating both. The plumbing
      exists (valid_to semantics already in queries); what's missing is the
      detection + invalidation step at commit. Scope to the narrow
      high-value classes first (locations, preferences, versions, names).
      This is roadmap rank [4]-build-first and the theory reviewer's top
      bet; it directly targets LongMemEval knowledge-update bleed.
      Files: extraction/apply.py or commit_policy.py, graph write path,
      retrieval current-value preference. Verify: depth-eval
      knowledge-update slice; unit tests ("I moved to Denver" → Seattle
      edge gets valid_to, current-value query returns Denver).
- [ ] **W3.6 Bounded merge slot in the mop.** Dedup is the one dead phase
      the fragmented-graph evidence actually demands on consumer installs.
      Give `execute_hygiene_mop` a small budgeted merge pass (exact
      -identifier + high-confidence name tiers only, no LLM), consistent
      with the mop's drain philosophy (hygiene_ops.py).
      Files: hygiene_ops.py, consolidation/phases/merge.py (callable slice).
      Verify: live mop window merges known dup pairs within budget; window
      duration stays bounded.

## Wave 4 — Instrument the theory (evals decide, not vibes)

- [ ] **W4.1 Activation arm:** one depth-eval/LongMemEval arm with
      `record_access=True` (adapter currently pins False —
      benchmark/longmemeval/adapter.py:431-436). Reports per-question-type
      deltas. Decides whether ACT-R stays a ranking term or becomes
      telemetry.
- [ ] **W4.2 Surfacing arm:** the W3.1 oracle experiment is the gate for
      graph→episode productionization (and effectively re-runs Gate G with
      a mechanism that can actually move episodes).
- [ ] **W4.3 TS arm:** seeded-TS vs TS-off. Decides `ts_enabled` default.
- [ ] **W4.4 Break the closed loop (design follow-up):** record access on
      *confirmed use* (feedback path exists — retrieval/feedback.py:183)
      rather than on surfacing, at least for the eval arm, so the
      Anderson–Schooler prior reflects usage, not ranker output.

## Investigations (from the cross-checker's uncovered areas)

- [ ] **I1 get_context/context_builder audit** — highest-traffic surface,
      zero adversarial coverage to date; it also records entity access
      (context_builder.py:851,1145) → same rich-get-richer loop.
- [ ] **I2 Evidence drains vs corroboration** — verify aggressive stale/
      already-exists rejection (hygiene_ops.py:148-215, evidence_drain.py)
      doesn't destroy the count≥2 corroboration evidence the Layer-3
      proper-name gate needs.
- [ ] **I3 MCP concurrent-open lock** — two `engram mcp` stdio sessions on
      one native dir have no lock (mcp/server.py:305); design a shared-open
      or advisory-lock story; targeted concurrency test first.

## Sequencing

Wave 1 is a single focused batch (est. one session), all bug-class, each row
independently testable — recommend executing immediately. Wave 2 next (one
session, includes the two DECISION rows). Wave 3 rows are independent of each
other; W3.1's experiment and W3.5 can start in parallel once Wave 1 lands.
Wave 4 arms ride whenever the eval rig is warm. Investigations are
subagent-sized and can run alongside any wave.

## Decisions needed from founder

1. **W2.5**: delete schema formation now, or keep dark?
2. **W3.4**: fix the tier system or collapse it?
3. **W1.2/W4.3**: keep TS enabled (seeded) pending the eval arm, or default
   it off now?
4. Scope check: Wave 1 + Wave 2 proceed without further review?
