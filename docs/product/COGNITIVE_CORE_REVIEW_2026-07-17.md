# Cognitive Core Review — ACT-R, Graph, Consolidation, Memory Model

**Date:** 2026-07-17
**Method:** 6 parallel deep-read agents (activation math, retrieval pipeline,
consolidation system, memory/data model, theory soundness, dead-feature sweep)
+ 1 adversarial cross-checker that re-verified the top claims against code.
All findings cite file:line and are CONFIRMED unless marked SUSPECTED.
**Status:** findings report — no fixes applied in this pass (except where noted
as already fixed in the same-day refactor batch).

---

## 0. Executive verdict

The ACT-R mathematical core is faithful and well-calibrated. The storage and
ingestion machinery is real. But the shipped product runs a small fraction of
the designed cognition: **14 of 18 consolidation phases never execute on a
default install** (shell runs no phases; the 2-hourly brain runs only
drains/adjudication/replay/prune), the graph cannot surface a recall result
by configuration (`passage_first_entity_budget=0`), and several of the
"brain-like" mechanisms are either profile-gated into oblivion or wired to a
consumer that discards their output. The effective live system is:
**episode-vector RRF search + temporal boosts + cue layer, with hygiene
drains** — which is ALSO the configuration that wins the evals. The designed
system and the validated system are different systems; most defects below are
in the gap between them.

## 1. Confirmed high-severity findings (live-install impact)

1. **Prune runs blind.** Prune executes only in the cold-brain process, which
   builds a fresh empty `MemoryActivationStore`; the activation snapshot
   load/save exists only in the REST shell lifecycle (main.py:135/473). Every
   prune protection — access double-check, 0.05 activation floor, emotional
   resistance — sits inside `if state:` (prune.py:76-99) and is skipped for
   every candidate. Mitigation: candidates must also have zero relationships
   and be >14d old, so the exposure is relationship-less but genuinely-used
   entities. `snapshot_to_graph` (the only graph-side access sync) has zero
   production callers, so graph-row `access_count` is permanently stale.
2. **Thompson sampling is a default-on noise source with a poisoned loop.**
   `ts_enabled=True` + no seed → unseeded Beta jitter (≤0.08·sem) in every
   recall score (scorer.py:231,261-266) — a mechanical source of the known
   recall nondeterminism in any config with entity slots > 0. Worse: with the
   default entity budget of 0, `returned_ids` is always empty, so **every
   recall records negative feedback against every candidate**
   (pipeline.py:2060-2071) — ts_beta grows monotonically, positives never
   fire, and each recall performs pool-sized activation writes on the read
   path.
3. **The graph→surface path is configured off, and the near-miss fix is
   cheap.** All entity-side stages (spreading, inhibition, salience, scorer,
   reranker, MMR) execute and are then discarded (`scored[:0]`,
   pipeline.py:2007-2012). The cross-checker refuted "no graph→episode path
   exists": `RecallEpisodeTraversal.append_entity_linked_episodes`
   (retrieval/episode_traversal.py:27-76) is wired and default-enabled — but
   it expands only entity entries in the FINAL results, which budget=0
   empties. Feeding it from the entity candidate pool instead is the ~20%
   remaining work of a graph→episode surfacing channel — the direct fix for
   the "Melanie→pottery connected-but-not-surfaced" failure.
4. **Merge degenerates to exact-identifier matching without embeddings.**
   Embedding failure → `except: pass` → emb=0; on a sparse graph the ensemble
   ceiling (~0.495) is below the reject threshold (0.55), so only the exact
   -identifier short-circuit merges (merge_scorer.py:321-422). Meanwhile
   infer treats missing embeddings as neutral 0.5 — a no-embedding brain
   can't merge duplicates but keeps minting inferred edges. Note: the
   FastEmbed model was silently broken Jul 13–17, i.e. this degenerate mode
   was the *actual* mode. The bare excepts here re-swallow the newly-hardened
   storage raises one layer up.
5. **Maturation/semanticize are structurally unreachable.** Maturation scans
   only the 50 most-recently-updated entities, and its own feature-cache
   writes bump `updated_at` even when the tier is unchanged, re-locking the
   window (maturation.py:132-144; helix sorts updated_at DESC). Semanticize
   has the same newest-20 window and needs 5 scans of the same episode.
   Maturity math: realistic personal-graph ceiling ≈0.60-0.65 < 0.70
   threshold (8 distinct predicate types required for full richness). Net:
   nothing graduates except identity_core auto-promote — "episodic→semantic"
   is aspirational. Additionally `mat_tier` never persists to the native
   column (model has no field; pydantic drops the kwarg; hasattr always
   False — helix/graph.py:588,859): the real tier lives only in attributes
   JSON.
6. **Replay treadmill + routing bypass.** The replay selector checks only
   status/window/projection_state (replay.py:427-452) — not
   `last_projection_reason` or source — so it re-extracts bootstrap docs and
   system discourse that worker routing deliberately parked as CUE_ONLY
   (including the doc-noise the 2026-03-20 Layer-3 fix exists to block).
   Empty extraction → `no_new_info` **without** flipping state → the same
   episodes re-extract every window, forever. And episodes older than the
   720h window can never be replayed at all: old CUE_ONLY is permanent
   landfill.
7. **Fail-open identity protection.** `get_identity_core_entities` failures
   are swallowed (`except: pass`) in dream LTD and microglia
   (dream.py:381-387, microglia.py:116-119) — a raising store silently
   removes identity protection. Dream associations invert the same way: a
   failed connectivity check means "assume disconnected" → maximal surprise →
   MORE speculative edges exactly when storage errors (dream.py:538-548).
   Latent on default installs (those phases don't run); live under
   standard/monolith.

## 2. Dead or display-only features (the feature-level silent-inert list)

| Feature | Status |
|---|---|
| Multi-signal triage scorer + online calibration | Never runs in default (worker off, triage phase never runs); CalibrationState has no persistence — resets every 2h brain window even under standard |
| Reranker | `provider="noop"` default; quiet disables outright; even `local` reranks only the discarded entity list (`reranker_rerank_episodes=False`) — production "rerank" is fictional |
| Scorer calibration snapshots | Written each engine cycle, read only by eval CLI — no scorer consumes them |
| `feedback_events` store | Table schema with zero producers/consumers |
| `ActivationState.spreading_bonus` | Never written; always 0.0 |
| NeuroplasticityEngine | Logs recommendations, adjusts nothing, no consumer, off by default |
| Schema formation | Force-disabled even in standard ("no retrieval path reads Schema/INSTANCE_OF") — 445 lines + 29 tests for no consumer |
| ClarificationIntent entities | Created with `activation_current=0.95` that (a) isn't used by scoring, (b) isn't persisted by helix create, (c) is never search-indexed or read — write-only |
| PressureAccumulator | Display-only in shell (actuating consumer is monolith scheduler); two inputs never incremented |
| Graph embeddings fusion (`weight_graph_structural=0.1`) | Multiplies zero in every real deployment (node2vec off in quiet; sqlite lacks the getter; standard+helix feeds discarded slots) |
| Inhibitory spreading / state-dependent / goal priming | Profile-gated to conservative/standard — consumer installs never see them; `update_arousal_ema` helper + its knob are dead even then |
| DecisionMaterializer | Dogfood-only (hardcodes Engram predicates); mints the very `X:decision_statement:Y` noise names that promotion.py then regex-suppresses at 4 call sites |
| ~17 config knobs | Unread outside config.py (verified by scripted grep; list in review transcript) |

**Live and working as designed:** emotional salience (write→score, ≤0.08, on
entity results), cue layer + cue-hit promotion feedback, near-miss surfacing,
open_work/debt counters (drive the mop trigger), differential tier decay *in
scoring* (0.5/0.3 — though only identity_core ever reaches the tier that uses
it), the no-record-on-spread rule (holds everywhere), atlas (dashboard-only by
design). Also corrected: quiet force-sets recall wave2, so AutoRecall/cue/conv
-context are live on default installs — "recall_profile=off" is not the
shipped reality.

## 3. Score-scale and gate defects (retrieval correctness)

- **RRF-normalized ranks and raw cosines share one formula.** Search
  candidates carry rank-normalized scores (top hit = 1.0 always); spreading
  -discovered and backfilled candidates carry raw cosine (~0.4 for a good
  neighbor) into the same `weight_semantic` term — graph-discovered items are
  systematically buried even when slots exist (pipeline.py:1308-1314 vs
  helix/search.py:100-106).
- **`seed_threshold=0.3` against normalized RRF is a rank cutoff**, not a
  relevance test → nearly every candidate seeds → `edge_proximity≈1.0`
  uniformly → the 0.15 edge weight is a constant, not a discriminator.
- **Cues can never pass the auto-recall gate:** ceiling 0.40×0.65=0.26 <
  `auto_recall_min_score=0.3` — cue layer is searched then always filtered
  off the auto surface (explicit recall keeps them).
- **Unprotected awaits on the hot path:** step 4.5's `batch_get` +
  `compute_similarity` have no try/timeout; post-hardening a storage raise
  kills the whole recall (pipeline.py:1305-1314). Temporal loops do O(N)
  sequential `get_episode_by_id` round-trips.

## 4. Theory verdict

- **ACT-R's prior is wrong for an assistant, and the loop is closed.**
  Anderson–Schooler justify recency/frequency by *environmental* statistics;
  Engram's access stream is generated by its own ranker (surfacing records
  access), so ACT-R amplifies ranker bias into permanent forgetting.
  Exploration/rediscovery counterweights (0.05/0.02) are ~10× weaker than the
  loop. One-shot high-value facts fall to a ~0.2 composite handicap within a
  week — the compensations (identity_core manual flag, salience regex, prune
  floors, LTD skip, summary-overwrite block) are five ad-hoc guard rails
  around a missing concept: **importance as a prior**. The principled fix
  already exists in the equation: set `consolidated_strength` at write time
  for remember()/identity facts (it sits inside the ln-sum, engine.py:29-34).
- **Assistants need supersession, not disuse decay.** Nothing handles
  contradiction/invalidation ("I moved to Denver"). The `valid_to` plumbing
  exists but no write path invalidates superseded facts. This is where
  episode-vector systems bleed on knowledge-update questions, it is ranked
  [4]→build-first in the strategic roadmap, and both the theory reviewer and
  the eval evidence point at it as the single highest-leverage bet.
- **ACT-R is epistemically dark:** benchmarks run `record_access=False`, so
  the 55-63% number cannot detect whether activation helps or hurts. The
  "faithful ACT-R" spreading strategy is benchmark-only (`spreading_strategy
  ="bfs"` in production). Run one eval arm with activation live.
- **Biology scorecard** (by what each phase computes, not its name):
  microglia detectors = real hygiene (type-pair table, cosine floor, meta
  -regex) but the C1q/C3 tag-confirm-demote cascade is cosplay around a lint
  pass; dream Hebbian/associations = costly tuning of a channel the evals
  showed net-neutral, with noise-level thresholds (sim floor 0.2);
  reconsolidation-as-implemented is *inverted* (fires on redundancy, not
  prediction error; blind `[:200]` concatenation that microglia later cleans)
  and structurally dead cross-process (5-min in-RAM window vs 2h brain
  cadence); CLS episodic→semantic has load-bearing tier decay but no actual
  gist extraction — the phase that would do it (reflect) ships dark. Compact's
  absorbed mass never decays (monotone `consolidated_strength` accumulator →
  systematic overestimate for old compacted entities) and absorbs at 0.5 even
  for semantic-tier entities scored at 0.3.
- **What ACT-R does earn:** cheap, deterministic, local recency/frequency
  ranking signal with lazy computation — worth keeping as one term among
  several, not as the memory model.

## 5. Memory model / categorization answer

- **More taxonomy is the wrong answer.** Decision/Preference/Correction/
  Person already carry real ranking semantics (2.5× durable boost,
  short-circuit set, packet typing). The gap is (a) **retention**: prune
  ignores entity_type entirely — a Preference and a bare Concept age
  identically unless a client proposal set identity_core; and (b)
  **reachability**: the LLM extraction prompt and the narrow extractor
  cannot emit the durable types at all — fully-local-without-agent-proposals,
  the durable machinery acts on an empty set (the north-star gate passes via
  its own Decision proposals — it validates the client-proposal path, not
  organic capture). Fix = per-fact retention policy attached at commit time +
  add durable types to both extractor vocabularies, not new types.
- **`Artifact` is overloaded** (bootstrapped files, conversation decision
  -provenance records, context builder) — `artifact_class` exists in
  attributes but is never a search filter; conversation records pollute file
  search. SUSPECTED-strong: two bootstrapped projects in one group merge
  their same-named files (README.md ↔ README.md pass every merge gate;
  only identity_core is exempt).
- **entity_type is an open case-sensitive string** — benchmark corpus writes
  lowercase and becomes invisible to all typed behavior; one defensive
  `.lower()` in the codebase proves the hazard is known. Normalize at commit.
- **CUE_ONLY is a healthy routing tier but currently a landfill** (see 1.6):
  the exits either violate routing intent (replay re-extracting parked
  content) or can't reach old episodes (720h window). It needs: state flip on
  no_new_info, routing-reason respect, and either an explicit landfill
  contract or an age-based promotion path.
- Two half-linked tier systems (episode `memory_tier` column vs entity
  `attributes["mat_tier"]`), plus reflect hijacking `memory_tier=
  "observation"` as a discriminator that semanticize can overwrite.

## 6. Recommended plan (ranked, consistent with the tiering strategy)

1. **Stop the live bleeding (hours):** load the activation snapshot in the
   brain/`open_local_stores` path so prune sees usage; skip TS feedback when
   no entity results returned; seed the TS RNG (or default `ts_enabled=False`);
   guard the two unprotected hot-path awaits; fix fail-open identity
   protection (3 sites); replay: flip state on `no_new_info` + respect
   `last_projection_reason`/source.
2. **Bi-temporal supersession (the strategic bet):** write-path invalidation
   of superseded facts using the existing valid_to plumbing; gate on the
   depth-eval. Highest expected eval + product lift (knowledge-update
   questions).
3. **Retrieval surfacing (cheap, testable):** oracle-surface experiment
   (force-include seed 1-hop neighbors' episodes, let similarity re-score);
   if it moves reachability, productionize graph→episode expansion by feeding
   `append_entity_linked_episodes` from the candidate pool. Fix the
   RRF-vs-cosine scale mismatch while in there. Put the local cross-encoder
   on the episode channel (`reranker_rerank_episodes=True`).
4. **Make the tier system real or remove it:** cursor/watermark for
   maturation+semanticize windows; stop bumping updated_at on unchanged
   scans; recalibrate maturity math to personal-scale; persist mat_tier
   properly (model field), or accept identity_core as the only durable tier
   and delete the rest.
5. **Importance as a prior:** set `consolidated_strength` at write time for
   remember()/durable/identity facts; retire ad-hoc guard rails as it takes
   over. Bound and decay compact's absorbed mass.
6. **Honesty pass on the phase system:** either schedule warm/cold windows
   that actually run merge/mature/reindex on consumer installs, or delete the
   dead phases and their claims (schema formation, neuroplasticity, feedback
   _events, spreading_bonus, ~17 dead knobs, `_TIER_PHASES["mop"]`).
   Dedup (merge) is the one phase the fragmented-graph evidence actually
   demands on consumer installs — give it a bounded slot in the mop.
7. **Instrument the theory:** one LongMemEval arm with `record_access=True`;
   one arm with TS off; report per-question-type deltas. Data decides whether
   activation stays a ranking term or becomes telemetry.

## Corrections to the agent reports (my own verification)

- "Unbounded access history without compact" — REFUTED: `record_access`
  trims at `max_history_size=200` (activation/engine.py:76-77).
- "No graph→episode path exists" — REFUTED as stated; the traversal exists
  and is default-enabled but starved (see 1.3).
- "13 of 18 / 11 of 16 phases dead" — reconciled to **14 of 18** by the
  cross-checker (registry has 18 incl. reflect/immunity; mop runs 4).
- The three `find_*_all` native routes were present in the compiled binary
  all along (zero-param queries emit no Input struct — audit-script artifact,
  not a code bug).

## Cross-checker's uncovered areas (flagged for a future pass)

1. `get_context`/context_builder — the highest-traffic agent surface had no
   adversarial coverage in this review (it also records entity access →
   the same rich-get-richer loop, unexamined).
2. The evidence drains (`reject_junk_evidence`/`drain_already_exists`/
   `drain_stale`) — the most destructive pass that actually runs live —
   were not audited for interaction with the count≥2 corroboration gate.
3. Concurrent multi-open of the native graph via two `engram mcp` stdio
   sessions on the same data dir — no lock exists on that path (the brain
   flock protects only brain-vs-shell). Needs a targeted concurrency test.
