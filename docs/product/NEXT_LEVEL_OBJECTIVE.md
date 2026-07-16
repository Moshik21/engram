# Engram Next-Level Objective

**Status:** DRAFT — pending founder review
**Created:** 2026-07-16 (from the 44-agent full-system audit; conclusions in project memory `deep-audit-2026-07-16`)
**Owner:** founder + coding agents
**Supersedes:** the "Recommended next work" ordering in `docs/AGENT_HANDOFF_2026-07-16.md` §7

---

## Objective (verifiable)

> Engram's thin live core (episode-vector recall + harness-proposed durables + identity-core context) becomes
> **boring-reliable**, the memory gets a **real metabolism** (every queue has a consumer), the north-star metric
> becomes **impossible to self-satisfy**, and a **second person can install it** — all fully local, all verified.

**Definition of done** (every box checked, each independently verifiable):

- [ ] `git status` clean on `main`; CI fully green including lint.
- [ ] 7 consecutive dogfood days: shell availability ≥ 99.5%; no brain window pauses the shell > 60s unless it
      performed real mutations; zero stranded-shell incidents under deliberate kill testing.
- [ ] `open_work` trends down week-over-week; the deferred-54 plateau is resolved through adjudication
      (commit or reject — not bigger drain budgets).
- [ ] `engram continuity --against-live` PASSES with `promote_if_missing=false` (aged organic Decision).
- [ ] Fresh-machine install = one command; `engram doctor` goes red on all four known silent failure modes
      (broken ONNX, unreachable Ollama→narrow, vectors off, stale embed cache).
- [ ] One verified backup/restore round-trip of the dogfood brain.
- [ ] Docs tell the truth (no monolith marketing, no corrupted model ids, no "CURRENT" traps).

**Not the objective:** LongMemEval score, new public MCP tools, new consolidation phases, Rust rewrite,
multi-device sync, federated anything.

---

## Milestone map

Dependency order. M0 is prerequisite to everything (total-loss protection). M1–M3 can interleave.
M4 gates the strategic decision G. M5–M7 are parallelizable after M1.

| # | Milestone | One-line outcome |
|---|-----------|------------------|
| M0 | Protect the truth | Working tree committed in green stacks; brain data backed up |
| M1 | Availability & exclusivity | Shell never stranded; graph never double-opened; no-op windows cost 0 downtime |
| M2 | Metabolism | Every queue has a consumer; debt trends down; triggers fire only on actionable debt |
| M3 | Silent-inert purge | No computed-but-inert paths on the live route (audit's confirmed set) |
| M4 | Honest metric v2 | Gate measures aged organic continuity + availability + precision |
| M5 | Recall quality gates | Rescue is intent-gated; all 9 public tools actually work on the consumer profile |
| M6 | Second-user install | engramctl owns both LaunchAgents; doctor probes real providers; one wizard |
| M7 | Docs & surface truth | Repo tells the current story to agents and contributors |
| G | Graph-tier portfolio decision | Evidence-gated keep/park decision on the depth tier |

---

## M0 — Protect the truth

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 0.1 | Commit the tree in the audited 8-stack order: embeddings-cache → index-completeness → hygiene-debt/drains → loop-steward(+dashboard) → operator-mop CLI → hot-cold split → recall polish → docs. Hunk-split the 5 mixed files (`config.py`, `__main__.py`, `scheduler.py`, `main.py`, `installer/engramctl`). | Each stack: `ruff check` + `ruff format --check` clean, targeted tests green, then full lite suite green at the end; `git status` clean | [ ] |
| 0.2 | Do NOT commit `server/showcase-export.md` (generated demo artifact) — delete or gitignore. | `git check-ignore server/showcase-export.md` passes or file gone | [ ] |
| 0.3 | Fix the F821 (`scheduler.py:171` missing `Any` import) + 20 ruff errors + 45 format diffs as part of their stacks. | CI lint job green | [ ] |
| 0.4 | One-off backup of `~/.helix/engram-native-dogfood-axi` (17G) to external/secondary storage before any further live-graph work. | Backup exists; restore-readability spot-checked | [ ] |

**Rule:** no `git reset --hard`, no history rewrites, stacks land in dependency order (import edges verified in audit).

---

## M1 — Availability & exclusivity (the P0 cluster)

The pause-shell design's single-writer invariant becomes a mechanism, not choreography.

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 1.1 | Pause-marker file + unconditional resume: brain writes `shell-paused-by-brain` marker before stopping the shell; any brain start (and `engramctl start`) resumes the shell if the marker exists and the shell is down. Resume no longer conditioned on `if paused` within the same run. | Test: `kill -9` brain mid-window → next brain run (or engramctl) restores shell. Unit tests on marker lifecycle | [ ] |
| 1.2 | Abort on pause failure: if `_pause_shell` cannot confirm the shell is down (probe false-negative, stop timeout, engramctl failure), the brain **exits non-zero without opening the graph**. Delete the "proceeding carefully" path. | Unit test: simulated stop failure → no engine open, exit 1 | [ ] |
| 1.3 | Power/wake gate + deadline: skip the run (exit 0, no pause) when on battery in DarkWake (`pmset -g ps`/`-g systemstate` check); wrap `_run_cycle` in `asyncio.wait_for(~1800s)`; report `time.monotonic()` duration alongside wall clock and log "system slept during run" when they diverge. | Unit tests on gate + deadline; brain-status gains monotonic field; no future 10h "runs" | [ ] |
| 1.4 | No-op pre-flight: query actionable debt via the still-running shell's read-only HTTP **before** pausing; skip the window entirely when there is no eligible work. | 24h dogfood log check: windows with no work show zero shell downtime | [ ] |
| 1.5 | `engram serve` (and `engramctl start`) take/observe `~/.engram/brain.lock` (shared/non-blocking probe): refuse or wait when a brain window holds it. | Test: start during held lock refuses with clear message | [ ] |
| 1.6 | Role-gate shutdown consolidation: `run_shutdown_consolidation` is a no-op unless `runtime_role == monolith`. | Shell stop produces no `trigger="shutdown"` cycle row; test | [ ] |
| 1.7 | Gate the CLI double-openers: `engram loop steward-once`, `engram hygiene report\|mop`, `engram index` check shell health first (wire the dead `_server_reachable`), and either route through shell HTTP or require `--force-local` with the shell confirmed down. Fix `hooks/session-steward-nudge.sh` to stop instructing a start-then-double-open ritual. | With shell up, each command refuses local open; tests | [ ] |
| 1.8 | Lock-contention loser must not write `brain-status.json` (currently clobbers the winner's record). | Unit test on the RuntimeError path | [ ] |
| 1.9 | Auto-drain `~/.engram/capture-queue.jsonl` on shell startup (replay endpoint exists; nothing calls it — 43 real captures currently rotting). | Startup log shows drain; queue empties; test | [ ] |
| 1.10 | REST trigger gate: `POST /api/consolidation/trigger` (and operator MCP `trigger_consolidation`) returns 409 with "use engram brain run" when `runtime_role == shell`. | Test + SKILL.md correction (M7) | [ ] |

---

## M2 — Metabolism (every queue gets a consumer)

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 2.1 | Bounded `evidence_adjudication` pass inside the brain window (budgeted commit-or-reject; increments `deferred_cycles`). This is the only legitimate exit for the deferred-54 and the 419 open adjudications. | deferred_evidence < 10 within a week; `deferred_cycles` advancing; audit records written | [ ] |
| 2.2 | Zero-LLM `replay` tier scheduled in the brain cadence (exact-substring linking against the clean agent-curated entity set) — gives the ~8.6k dormant observe episodes their consumer. | cue_only + pending trending down week-over-week | [ ] |
| 2.3 | `should_mop` computed on **actionable** debt only (work a scheduled tier can actually reduce) and **honored** — no more unconditional drains. Fix the cue_only metric conflation (trigger counts episode projection state; cue hygiene only clears cue rows). | Brain log shows skipped windows; trigger metric moves when work completes | [ ] |
| 2.4 | Watermark/index the cue-hygiene scan (currently full 8.4k-row scan per window for 0 eligible). | Mop duration for no-op case < 30s | [ ] |
| 2.5 | Persist activation state across restarts (snapshot on shutdown / restore on start — `snapshot_to_graph` exists, is never called; ACT-R history currently wiped every 2h). | Activation survives a shell restart; test | [ ] |

---

## M3 — Silent-inert purge (audit-confirmed instances)

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 3.1 | Broken-model embeddings: return `[]` (whole-batch failure) or raise — never `[[]]`; clear the `_model_broken` latch on successful repair; one-shot audit script counts and repairs present-but-empty vectors already in stores. | Unit test on failure contract; audit script reports 0 empty vectors | [ ] |
| 3.2 | Noop-embed guard in `_run_mop`: set provider before `EngramConfig()` construction (or set the field directly). | Unit test: manual mop resolves provider noop | [ ] |
| 3.3 | Wire the dead background count-refresh in `storage/diagnostics.py` (`_track_count_refresh` has zero call sites); counts recover after a timed-out snapshot instead of pinning stale. | Dashboard counts refresh after induced timeout; test | [ ] |
| 3.4 | Loop-steward honor gaps: apply the adjustment overlay in `brain_cli._run_cycle` (non-mop tiers); honor the intake floor wherever the brain drains QUEUED episodes; resolve the file/sidecar dual-write divergence (recommend: drop the sidecar, file is authoritative) — or route CLI through the async dual-write. | `test_loop_shell_honor` extended to brain paths; status/apply/clear agree across CLI/API/MCP | [ ] |
| 3.5 | Fix `worker_auto_capture_extract_score_floor` `or`-coercion (legal 0.0 becomes 0.85). | Unit test with floor=0.0 | [ ] |
| 3.6 | Fix latent `AttributeError`: `projection_execution.py` references `_evidence_adjudication_service` never assigned on `EvidenceProjectionExecutor`. | Test with `active_adjudication_enabled=True` | [ ] |
| 3.7 | Honest hygiene reports: stop reporting always-zero `event_bus` pressure from fresh accumulators (report debt-only, or plumb the real accumulator). | Report fields match reality; test | [ ] |
| 3.8 | Steward-once `--mop` honors `--adjustment-path`; API/MCP apply honors (or removes) `skip_continuity_check`; clamp caps derive from live env config, not default-constructed `ActivationConfig()`. | Unit tests on each | [ ] |

---

## M4 — Honest metric v2

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 4.1 | Continuity gate v2: aged organic Decision hit — `promote_if_missing=false`, target Decision ≥ 7 days old that survived ≥ N consolidation cycles. The current gate can write a synthetic Decision and immediately recall it. | Gate FAILS on a brain whose organic Decisions are unreachable; PASSES live | [ ] |
| 4.2 | Decision precision@5 (WEEKLY_NORTH_STAR's anti-metric: 0 decision_statement scrap in top-5) added to the scorecard. | Scorecard shows the number weekly | [ ] |
| 4.3 | Shell availability % + max brain-window duration become first-class metrics (computed from logs/brain-status; shown in `engramctl status` and the dashboard scorecard). | Numbers visible; the overnight-outage class is now detectable | [ ] |
| 4.4 | Doctor/engramctl surface brain anomalies: last run error, staleness, repeated no-op windows, wall-vs-monotonic sleep flag. | Simulated anomaly turns status/doctor output red | [ ] |

---

## M5 — Recall quality gates (public surface must be real)

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 5.1 | Intent-gate the durable-first name rescue (uncommitted) — fire only for decision/preference-shaped queries, or merge rescue hits into the deep pool instead of replacing it. Today any query naming "Engram"/"Konner" short-circuits the entire pipeline into name-stub results. | Query-class eval: entity-name queries AND paraphrase/episodic queries both return correct shape | [ ] |
| 5.2 | Aggregate wall-clock bound on the rescue (per-probe-only today; up to ~7.6s theoretical, runs up to 3× per recall). | Timeout test; recall p95 unchanged or better | [ ] |
| 5.3 | **DECISION NEEDED:** `intend` on the quiet profile — enable `prospective_memory_enabled` for quiet, or pull `intend` from the public 9. A frozen public tool must not be a silent no-op. | Chosen path implemented + tested; GOLDEN_LOOP.md updated | [ ] |
| 5.4 | Surface-aware system prompt: strip operator-tool instructions (`route_question`, `search_artifacts`) when surface=public; fix adoption nudges that suggest nonexistent tools. Resolve the intend/list_intentions asymmetry per 5.3. | Public-surface prompt contains no unavailable tool names; test | [ ] |
| 5.5 | **DECISION NEEDED:** REST exposure — at minimum change `ServerConfig.host` default to `127.0.0.1` and remove REST-bypass teaching from SKILL.md; optionally add a REST surface policy mirroring the MCP freeze. | Bare `engram serve` binds localhost; SKILL.md consistent with freeze | [ ] |
| 5.6 | Paraphrase-robust Decision surfacing experiment (evidence for G): reserve 1 durable-entity slot in passage-first assembly (vs today's 0) and/or index Decision summaries in the episode-tier semantic index. Run as an A/B against the depth-eval scaffold. | A/B result recorded; kept only if it wins | [ ] |

---

## M6 — Second-user install

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 6.1 | engramctl templates + installs **both** LaunchAgents (`dev.engram.local`, `dev.engram.brain` with `--tier mop`) during quickstart; `engramctl brain status\|install` subcommand. Fix the example plist (currently `--tier auto` = full 16-phase cycle every 2h, dev-repo venv path, no embed guard). | Fresh-machine (or clean-user) install: one command → both agents loaded, correct tiers | [ ] |
| 6.2 | Doctor live-provider probes: materialize the configured fastembed model from the configured cache and embed one string; probe Ollama when extraction is auto/ollama; report the resolved extraction+embedding rungs; run vector-completeness (`engram index` logic) against the live brain **safely** (via shell HTTP or with the lock held); check brain-status health. | Doctor red on each of the four simulated failure modes | [ ] |
| 6.3 | One setup wizard: delete or delegate `python -m engram setup` to the engramctl env-writer (it currently defaults standard/all/rework, omits role/surface/cache-path, and prints a false "vector search disabled" claim). | Single code path writes .env; quiet/shell/wave2 defaults; no false claims | [ ] |
| 6.4 | `engram backup` (LMDB-safe copy of the native data dir + `~/.engram` state) with documented restore. | Round-trip restore verified on a copy | [ ] |
| 6.5 | Privacy hygiene minimum: document what `~/.engram/capture-queue.jsonl` and the graph store in plaintext (verbatim cross-project prompts today); decide whether to scrub queue entries after drain. | Documented; queue entries removed after successful replay | [ ] |

---

## M7 — Docs & surface truth

| ID | Deliverable | Verify | Status |
|----|-------------|--------|--------|
| 7.1 | Rewrite or delete `AGENTS.md` (corrupted "Codex Haiku (`Codex-haiku-4-5-20251001`)" model id; 15-tool/FalkorDB-era architecture). | File matches on-disk reality | [ ] |
| 7.2 | README: retire the 27-tool badge, always-on-brain framing, Docker-first quickstart; present the 9-tool golden loop + quiet shell + cold brain. | README matches product contract | [ ] |
| 7.3 | Rename `docs/CURRENT_HANDOFF.md` → `docs/ARCHIVE_HANDOFF_2026-07-10.md`, leave a pointer stub. | Grep for "current handoff" finds the live doc | [ ] |
| 7.4 | Update project `CLAUDE.md`: quiet profile, `runtime_role`, brain/hygiene/loop CLIs, actual phase registry count, current recall architecture. | New-agent smoke: instructions produce working commands | [ ] |
| 7.5 | Sync `skills/engram-memory/SKILL.md` + client packs (remove REST-bypass teaching, fix phase counts and unsupported request bodies, add brain-window "connection refused is transient — retry" guidance) and **republish to clawhub** (standing reminder). | Skill matches surface; republish done | [ ] |
| 7.6 | Status headers on all `docs/design/*.md` (implemented / superseded / parked); stamp `SAAS_NEXT_STEPS.md` + the FalkorDB-default section of the OpenClaw strategy DEPRECATED. | Every design doc has a status line | [ ] |

---

## G — Graph-tier portfolio decision (evidence-gated; after M2 + M4 run ≥ 2 weeks)

The audit's central strategic fact: live product value flows through episode vectors + harness proposals +
name rescue; the graph/consolidation tier is deliberately dark in results (`passage_first_entity_budget=0`
per the tiering strategy) and until M2 had no consumer. With M2 feeding it (adjudication + replay on the
clean agent-curated entity set) and M4 measuring honestly (aged-organic hit rate, precision@5, availability
cost), decide:

- **Keep-and-open**: 5.6's A/B shows durable-entity slots or graph neighbors improving aged-organic
  hit/precision → open the tier incrementally (entity slot > 0, graph pool budget raised from 75ms).
- **Keep-dark**: no measurable lift → tier stays validation-gated; cap its ops cost (brain cadence
  can lengthen; steward/hygiene stay minimal); revisit the May roadmap items (vector-gap, bi-temporal
  current-value, write-side synthesis) as the next depth investments instead.

**Decision inputs required:** ≥ 2 weeks of M4 metrics, the 5.6 A/B, and per-week brain-window cost.
No further graph-tier plumbing investment before this decision.

---

## Guardrails (unchanged from audit + handoff)

- Dogfood stays `ENGRAM_RUNTIME_ROLE=shell` + `quiet` + wave2 + quantized nomic (`…-Q`, `FASTEMBED_CACHE_PATH=…/fastembed/hf`). No monolith + standard flips (Jetsam history).
- Public MCP surface stays frozen at the 9 tools; changes only via an explicit decision recorded here (5.3 is the one open case).
- No Rust rewrite; no capture outbox unless M1.4 fails to make no-op windows zero-outage; no new consolidation phases; no multi-device sync build.
- Surgical diffs; never delete working code before its replacement is verified; never open the live native data dir from a second process (M1 makes this mechanical).

## Open decisions for founder review

1. **5.3** — `intend`: enable prospective memory on quiet, or drop it from the public 9?
2. **5.5** — REST: localhost-bind + doc fix only, or full REST surface policy mirroring the MCP freeze?
3. **3.4** — Loop-steward sidecar: drop the Helix dual-write (file authoritative), or fix CLI to dual-write?
4. **Steward ritual** — fold `steward-once` into `brain run` pre-mop (deterministic, no agent involvement) and keep the harness ritual as operator-optional?
5. **M0.4 backup target** — where does the 17G backup live (external disk, NAS, cloud-encrypted)?
6. **G criteria** — agree the 2-week window + the three decision inputs before starting M5.6.

## How to mark progress

1. Implement + verify the row (verify column is the gate, not the diff).
2. Flip `[ ]` → `[x]` here.
3. Operator-facing changes get a one-line note in the live handoff doc.
