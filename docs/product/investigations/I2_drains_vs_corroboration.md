# I2 — Evidence drains vs corroboration gate (2026-07-17)

VERDICT: RACE EXISTS — structural starvation, two concrete interleavings, both live under mop-only scheduling.

**Corroboration side (what the gate needs)**
- Bare proper_name evidence defers at extraction: conf 0.55 (narrow/entity_extractor.py:505,513), entity threshold 0.70 (config.py:2542), defer band 0.15 → `defer`, persisted immediately with `status="deferred"`, `deferred_cycles=0` (projection_execution.py:276-324).
- Adjudication counts corroboration ONLY over co-loaded open rows: `get_pending_evidence` filters `status IN ('pending','deferred','approved')` (sqlite/graph.py:2477; helix/graph.py:3617-3625 OPEN_EVIDENCE_STATUSES), groups by key `entity:{name}:{entity_type}` (evidence_adjudication.py:603-609), and holds proper_name rows when `group_count < 2` (evidence_adjudication.py:292-319). A **rejected** first instance is invisible to this count forever.
- Each hold calls `_defer_evidence` → `deferred_cycles += 1` (evidence_adjudication.py:307-308, 539-551). Mop runs every 2h (LaunchAgent), so cycles advance ~12/day.

**Drain side (what kills the instances)**
- `select_stale_low_value_evidence` selects a row when `cycles >= min_deferred_cycles` **OR** `age >= max_age_days` (skip predicate evidence_drain.py:527 is AND of the negations). High-signal-type exemption (line 504) does NOT protect bare proper names: Layer-1 typed them Concept, and HIGH_SIGNAL_ENTITY_TYPES (promotion.py:15-28) has Person but not Concept.
- Interleaving A (default config): the corroboration hold itself advances the stale clock. Instance 1 held 5 passes → cycles=5 at ~10h old → next window's stale drain rejects it as `drain_stale:stale_uncorroborated` (hygiene_ops.py:205-218 runs BEFORE the adjudication call at :240-252; the phase's own internal drain at evidence_adjudication.py:141-165 also runs before grouping at :228). Even if the drain misses it, the forced path at cycles>=5 force-rejects it (`forced` bypasses the hold at :299 but `should_force_commit_evidence` → False for Concept, evidence_drain.py:535-549 → rejected at evidence_adjudication.py:334-361). Effective corroboration window ≈ **10-12h**, not the nominal 21 days. Instance 2 arriving later always sees group_count=1 → held → dies on the same clock. Permanent starvation unless two mentions land within ~5 mop windows.
- Interleaving B (recovery mode, worse): at `len(deferred) >= 200`, hygiene_ops.py:201-203 sets `min_cycles=0` → skip predicate `cycles < 0 and not age_ok` is unconditionally False → **every** non-client-proposal, non-high-signal deferred row is selected regardless of age; the advertised 3-day floor (stale_days=3.0) is dead code because `age_ok` is only consulted when `cycles < min_cycles`. A proper-name row deferred minutes earlier is drained in the same/next 2h window, before adjudication ever counts it. Window ≈ 0-2h.
- Not implicated: `reject_junk_evidence` (pattern-based; legit names pass, evidence_drain.py:248-313) and `drain_already_exists` (casefold-exact name must already exist in the graph, evidence_drain.py:456-479 — only fires post-promotion, which is the intended dedup).

**Aggravator (even without the drains)**: passing the hold ≠ promotion. group_count=2 gives +0.05/cycle persisted boost (evidence_adjudication.py:242-257); 0.55 needs 3 co-present cycles to reach 0.70 (4 for the >500-entity 0.75 threshold), but the cycles>=5 forced-reject fires first unless the two mentions arrive within ~1-2 cycles of each other. Also the count is computed only within a confidence-DESC window of 200 rows (config `consolidation_evidence_adjudication_limit`), so on a large backlog two 0.55-conf instances may never co-load.

**Minimal fix** (two clauses, one file each):
1. `evidence_drain.py select_stale_low_value_evidence`: exempt corroboration-gated rows from the cycles shortcut and the recovery floor — if `"proper_name" in row["corroborating_signals"] and "identity_pattern" not in ...`, select only when `age_days >= 21.0` (the un-lowered config default), ignoring `min_deferred_cycles`/recovery overrides. (Signals are present on rows: sqlite/graph.py:51.)
2. `evidence_adjudication.py`: don't advance `deferred_cycles` when deferring for `proper_name_needs_corroboration` (the hold at :307 is "waiting for data", not a failed adjudication) — pass a flag into `_defer_evidence` or skip the increment for that reason. This keeps the cycles>=5 forced-reject from consuming the corroboration window.

Both are additive guards; the drains keep clearing genuine sludge (junk-classified, redundant, aged-out) at full rate.

Files read (no changes made — investigation was read-only as specified):
- /Users/konnermoshier/Engram/server/engram/hygiene_ops.py
- /Users/konnermoshier/Engram/server/engram/consolidation/evidence_drain.py
- /Users/konnermoshier/Engram/server/engram/consolidation/phases/evidence_adjudication.py
- /Users/konnermoshier/Engram/server/engram/extraction/commit_policy.py, extraction/promotion.py, extraction/narrow/entity_extractor.py, ingestion/projection_execution.py, storage/sqlite/graph.py, storage/helix/graph.py, config.py

Tests/lint: none run — no files modified.