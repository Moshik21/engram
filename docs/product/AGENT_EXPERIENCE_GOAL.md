# Goal: The Agent-Experience Repair — write-side attention, question-space recall, index integrity

**Status: EXECUTING — D1–D5 resolved 2026-07-22 under founder-delegated judgment
(pattern of the two predecessor goals; product not live, brain/schema
modification pre-authorized)**
**Drafted: 2026-07-22, from the first agent-experience evaluation**
**Predecessor: RECENCY_FREQUENCY_GOAL.md (complete; usage flip armed behind organic yield)**

## Why this goal exists — the evidence

The first agent-as-user evaluation (2026-07-22, grader held full ground truth;
battery at `server/tests/rigs/agent_experience_battery.json`) scored the live
recall surface **3 hits / 1 weak / 1 wrong-hit / 5 misses** on 10 ground-truth
questions — on the same brain where mechanism gates show reach@10 = 42/42.
Both numbers are true: the mechanisms work, the *experience* doesn't. Five
defect signatures, one shared root:

1. **Bootstrap 500s every session start** — `create_entity` collides with a
   stale native BM25 document (silent cascade-delete failure months earlier).
2. **Squatter entities** — a milestone observation became a sentence-long
   entity name scoring 0.99 on unrelated ranking queries.
3. **Hook-noise pollution** — task-notification/command-output episodes are
   vector-indexed with full standing and rank top-3 on technical queries.
4. **Answer locality** — multi-fact observations never surface for their
   natural single-fact questions. Aggravated by the 2026-07-21 emergency
   backfill: it wrote ONE vector per episode from 1,200-char truncated text
   (the designed `index_episode` path writes full-content + per-chunk
   vectors), so the 8,918-episode historical corpus is coarsely indexed.
5. **Identity never surfaces** — identity_core entities are unreachable by
   candidate generation for identity-phrased queries; the durable result
   lane has no candidate-stage feeder.

**The shared root**: Engram spends all its intelligence at read time and none
at write time. Capture is deliberately dumb (CQRS — keep it), but dumb
capture silently became *uniform indexing*: every byte gets equal standing in
semantic space. The attention gate that biology applies at encoding exists in
the codebase (the 8-signal triage scorer, ~2ms) — it is wired to extraction,
not to indexing.

## Design principles

- **P1 Write-side attention**: what enters *semantic space* is a decision,
  not a default. Storage stays universal; embedding is earned.
- **P2 Question-space indexing**: agents query in question-space; content
  lives in answer-space. Stop bridging badly with one embedding — index the
  anticipated questions themselves, proposed by the observing agent (the
  extraction-ladder philosophy applied to indexing; zero LLM cost).
- **P3 Index integrity is an immune function**: graph↔index drift (orphan
  BM25 docs, vector-less rows, duplicate vectors) is a recurring species —
  it gets a dedicated bounded drain, not per-incident firefighting.
- **P4 Names are identifiers, not content**: retrievability comes from
  evidence and summaries; a name longer than a name is a summary in disguise.
- **P5 Non-use is forgetting evidence, never ranking evidence**:
  surfaced-but-never-used feeds the pruner/demoter offline (M4.1's lesson:
  behavioral echo must not touch live rank).
- **P6 The battery is the gate**: mechanism gates (G1–G7 class) stay, but
  every milestone here is additionally judged by the agent-experience battery
  score. Baseline: 3/10.

## Founder decisions required

- [x] **D1 — APPROVED (additive-only).** M0's by-id vector/doc probes require
      new native queries → schema regen + native rebuild. This is the
      operational cost that has been deferred three times (drain census, write
      idempotency, backfill dedup all need it). Approve one supervised regen
      window, or reject and accept census-blind workarounds permanently.
      RESOLUTION NOTE: additive QUERY definitions only, zero node/edge schema
      changes; validated on the 17GB clone before any live window; the live
      regen runs supervised with the shell paused (same protocol as the
      vector-write window).
- [x] **D2 — decided by M0.4 capability check; DEFAULT DEMOTE.** If helix native supports
      vector deletion, machinery-class episodes can be removed from HNSW; if
      not, they get class-based ranking demotion (recap_penalty pattern) and
      exclusion at query time. Decision follows a capability check in M0.
- [x] **D3 — CLI flag (`--question`, repeatable) mapping into the observation payload.** `axi observe --questions`
      flag (explicit, per-call) vs structured observation body (parsed
      contract) vs both. Client packs and the clawhub skill must teach it.
- [x] **D4 — DEMOTION first; prune only after a measurement window.** Feed surfaced/used ratio to prune
      (episodes eligible for soft-delete) or to a mop-time demotion score
      (kept but down-weighted). Recommend demotion first, prune only after a
      measurement window.
- [x] **D5 — SALIENCE-GATED SUBSET.** Full corpus through real
      `index_episode` (~12h of mop windows at 50/window; proper chunk
      vectors) vs salience-gated subset (skip machinery class — likely ~40%
      smaller). Recommend gated subset.

## Milestones

### M0 — Keystone primitive + index-integrity immune drain [UNBLOCKS EVERYTHING]

- [x] **M0.1 By-id probes in schema.hx**: `find_episode_vectors_by_ids`,
      `find_cue_vectors_by_ids`, `find_bm25_doc_by_id` (entity/episode/cue) —
      mirrors the existing `find_entity_vectors_by_ids`. Requires D1 regen
      window. DoD: probes return exact presence for known-written ids
      (the ANN-census under-reporting instrument bug dies here).
- [x] **M0.2 Consistency drain** in the hygiene mop: bounded cursor-driven
      diff of graph rows vs index docs, both directions — orphan BM25 docs
      (the bootstrap-500 root), vector-less rows, duplicate vectors (the
      drains/backfill created known duplicates on clone paths). Repairs
      logged with counts; debt scoreboard row. DoD: mop report carries
      `index_consistency`; the live brain's stale `create_entity` BM25 doc is
      repaired; bootstrap returns 200 on a fresh session.
- [x] **M0.3 Write-conflict self-healing**: deterministic state-conflict
      errors on native writes (doc-already-exists class) attempt one
      reconcile-and-retry, then surface as hygiene debt — never a silent 500
      loop. DoD: test with a planted stale doc; bootstrap-500 regression test.
- [x] **M0.4 Vector-delete capability check** (feeds D2): can helix native
      remove/replace HNSW vectors and BM25 docs? Document the answer with a
      probe script; it decides M1's noise treatment.

### M1 — Write-side attention + honest re-index [P1]

- [x] **M1.1 Machinery-class detection at capture**: cheap deterministic
      classifier (protocol frames, tool-use ids, task-notification shapes,
      exit-code dumps) → episode `salience_class` field. Zero LLM. DoD:
      classifier corpus test on real hook captures from the dogfood brain;
      false-positive rate measured on genuine content.
- [x] **M1.2 Salience-gated embedding**: machinery-class episodes are stored
      and BM25/grep-reachable but get NO capture-time vectors; the mop drain
      skips them. Existing noise handled per D2. DoD: fresh-install smoke
      extended — noise episode stored but absent from vector space; battery
      queries no longer surface task-notification episodes.
- [ ] **M1.3 Historical re-index through real `index_episode`** (per D5
      scope): cursor-swept mop windows replace the coarse single-vector
      backfill with proper full+chunk vectors; presence-aware via M0.1 (no
      duplicates). DoD: sampled episodes show chunk vectors; battery
      answer-locality questions (recall-outage, fastembed-outage, ts-kill)
      re-scored.
- [x] **M1.4 Squatter guard**: extraction commit policy caps entity names at
      6 tokens (excess folds into summary); observation-sourced entities
      require ≥2-episode corroboration before full ranking weight (extends
      the existing proper-name gate). DoD: the live squatter entity is
      repaired (renamed/demoted) by a mop pass; regression test.

### M2 — Question-space observe [P2; accelerates the RF flip]

- [x] **M2.1 Protocol**: `axi observe` accepts anticipated questions (shape
      per D3). Each becomes a cue record embedded in question-space pointing
      at the episode; agent-proposed entities accepted on the same call
      (agent-as-extractor for indexing). DoD: contract test; skill +
      client-packs updated (clawhub republish reminder).
- [x] **M2.2 Recall integration**: question-cues compete in the cue lane
      (existing machinery — cue_recall path, M5.1 usage substrate). No new
      ranking paths. DoD: battery question stored WITH proposed questions is
      recalled at rank 1 by its natural question; cue_hit_count grows in
      normal dogfood use.
- [ ] **M2.3 Flip interplay measured**: question-cues raise cue surfacing →
      organic used-tier yield accelerates. Report yield rate before/after a
      one-week window. **This is the RF flip's accelerator — when yield > 0,
      execute the armed flip per M2.5/RECENCY_FREQUENCY_GOAL (E==A + latency
      spot-check only).**

### M3 — Durable candidate feeder [P—identity]

- [x] **M3.1** identity_core + durable-class entities (bounded: ≤64) are
      always present in the candidate pool (zero search cost; scoring
      decides). Flag `durable_candidate_feeder_enabled`, default off until
      battery + continuity show no regression, then flip. DoD: battery
      founder-identity question hits; flag-off byte-identity; continuity
      PASS unchanged.

### M4 — Surfaced-never-used decay [P5; needs accumulated data]

- [x] **M4.1** Track surfaced/used ratio per episode/entity from the existing
      usage substrate (surfaced counts exist; used counts exist). Mop-time
      consumer per D4 (demotion recommended first). Bounded, tier-aware,
      identity/durable exempt. DoD: planted chronic-surfaced-never-used items
      demote after N windows; used items never demote; live ranking untouched
      (byte-identity outside the mop).

### M5 — The suites become the gate [P6]

- [x] **M5.1 Battery rig**: `agent_experience_battery.json` gets a scored
      runner (containment@3 over the axi recall surface + get_context),
      CI-runnable against a seeded corpus, dogfood-runnable against live.
      DoD: score reported per milestone lands in this doc.
- [x] **M5.2 Fresh-agent suite** (the advanced one): a scripted
      Sonnet-level agent with EMPTY context + only Engram tools answers the
      battery + novel questions; measured against a no-memory control:
      answerability lift, token cost, tool-call count. This replaces the
      "grader holds ground truth" crutch. DoD: suite runs end-to-end locally
      (fully-local constraint: local judge or deterministic containment
      scoring); baseline recorded.

## Gates

| Gate | Instrument | Threshold |
|---|---|---|
| **B1 battery** | agent_experience battery containment@3 | ≥ 7/10 by goal end (baseline 3/10); no question regresses |
| **B2 zero bootstrap errors** | 20 consecutive session starts | 0 × 500s |
| **B3 no mechanism regression** | RF gate suite (G1/G3/G4/G5/G7) + continuity | all green on final stack |
| **B4 noise exclusion** | battery + spot queries | 0 machinery-class episodes in any top-3 |
| **B5 byte-identity** | flag-off drift probes | all new ranking-adjacent paths byte-identical off |
| **B6 fresh-agent lift** | M5.2 suite | memory-agent > no-memory control on answerability; report committed |

## Completion

DONE when: D1–D5 carry founder decisions; M0–M5 rows landed or
parked-with-dated-reason; B1–B6 green on the final stack with reports
committed under `docs/product/experiments/`; the RF flip status is resolved
(executed if yield arrived, else re-parked with the measured yield rate); and
the battery score table below is filled.

| Checkpoint | Battery score |
|---|---|
| Baseline 2026-07-22 | 3/10 (1 weak, 1 wrong-hit) |
| Post phase-1+2, bootstrap healed (2026-07-24) | **6/10**, B4 machinery-clean |
| Post recency chunk re-index (2026-07-24) | **3/10** — regressed by recall latency (see assessment), NOT mechanisms |
| Blocked at | latency ceiling (native HNSW + durable-rescue fallback cascade); B1 ≥7 unmet |


## Session completion assessment (2026-07-24)

Committed b906316 (phase 1) → 200fcaf (phase 2) → 2fbd4c9 (live window).
Suite 5032+ green, ruff clean, CI green through 200fcaf.

**Landed + verified (mechanisms all work):** M0 keystone (by-id probes,
consistency drain, write-conflict self-heal, capability check — all live on
the regenerated 198-route binary); M1.1/M1.2/M1.4 (machinery classifier
0 FP / 1.00 recall, salience-gated embedding, squatter guard); M2.1/M2.2
(question-space observe, e2e verbatim + paraphrase recall); M3.1 (durable
feeder, flag-off byte-identical); M4.1 (surfaced-never-used decay, P5
boundary pinned — zero retrieval refs); M5.1/M5.2 (battery + fresh-agent
runners).

**LIVE PROOF:**
- **B2 PASS** — bootstrap 500→200. The BM25 upsert heals orphan docs on
  first re-create (`replaced stale document ... idempotent upsert`). The
  months-long every-session 500 is dead.
- **B4 PASS** — battery machinery-clean, 0 machinery-class episodes in any
  top-3 (hook-noise pollution eliminated).
- **B5 PASS** — flag-off byte-identity pinned for every ranking-adjacent
  path (durable feeder, usage decay).
- **B3** — continuity still PASSES live (819ms < 2000ms); RF gate suite
  green.

**BLOCKED — B1 (battery ≥7/10) NOT met; the honest finding:**
Battery went 3→6 (bootstrap heal + machinery-clean + squatter/feeder), then
6→3 after the recency-weighted 312-episode chunk re-index. Root cause is
NOT the mechanisms: the enlarged vector index raised recall 438→819ms,
tipping episode queries past the 1500ms primary budget into an expensive
**durable-entity-rescue fallback (1.5s)** that discards the ~44 found
candidates and degrades. The battery is **latency-gated**. The vector
search itself is fine (~985ms, finds candidates); the fixed pipeline
overhead (stats 250ms + preflight 250ms) + the fallback cascade is the
ceiling.

**M1.3 PARKED** (built + tested, gated `reindex_sweep_enabled=False`): the
chunk answer-locality fix is correct in principle but exposes the recall
latency ceiling on large native brains; it needs a latency-aware redesign
(two-stage retrieval — main vector lane first, chunk lane only on miss — or
coarser chunking) before running unattended. Left ON it would grow the
index corpus-wide and risk the continuity gate. This becomes the head of
the answer-locality task (#14) and folds into the recall-latency /
fallback-cascade work.

**Net:** this program made Engram markedly more RELIABLE (bootstrap heal,
machinery-clean, index-integrity immune drain, self-healing BM25) and built
every write-side-attention mechanism, but the retrieval-QUALITY battery is
now provably blocked by a recall-latency ceiling (native HNSW + fallback
cascade at 17GB scale) — the same infrastructure frontier as the original
deep-recall saga, one layer up. That ceiling is the true next lever.
