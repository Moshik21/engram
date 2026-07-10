# Current Handoff

**Last updated:** 2026-07-09

**Read the START HERE section first.** Everything below the archive divider is
historical context from prior milestones — useful for archaeology, not for
re-deriving current dogfood truth.

---

## START HERE — Strategy + Golden Loop (2026-07-09)

### TL;DR

**North star is not LongMemEval.** Product success = *fresh session / other agent
surfaces 1 high-signal prior Decision without opening a doc.*

Architecture direction (keep):

1. **Passive capture** — `observe` / harness auto-observe (cheap, complete)
2. **Sparse agent promotion** — `remember` with `proposed_*` (0–5 per **compaction window**)
3. **Deliberate consolidation** — offline hygiene, not always-on LLM ETL

Helix **native PyO3 is custom Engram work**, not official Helix. Treat schema /
install drift as Engram-owned.

### Shipped this session (code)

| Area | Change |
|------|--------|
| Promotion policy | `extraction/promotion.py` — high-signal types, recap reject, window budget |
| Compaction window | 0–5 remembers per window; reset on `compaction_id`, compaction source, or **4h idle** (not multi-day session lifetime) |
| Client proposals | Span auto-fill; high-signal verified floor; commit path for span-verified proposals |
| Decision names | `validate_entity_name` allows ≤24 words for Decision/Preference/client proposals (was hard 5-word reject) |
| Auto-capture worker | Auto sources stay cue-only unless triage score ≥ 0.85 |
| Recall ranking | Prefer durable entity facts over session recap / cue packets |
| Recall rescue | Preflight timeout no longer aborts deep recall; durable entity name rescue surfaces Decisions when hybrid search times out on loaded brain |

### Golden path (verified live 2026-07-09)

```
remember(Decision + proposed_entities/relationships, span-verified)
  → 2 entities + 1 relationship persisted
  → entity search finds Decision
  → recall returns Decision via durable_entity_rescue (loaded 17G brain still slow)
```

Strategy Decisions stored: “LongMemEval is not Engram north star”, “Prefer sparse
agent promotion”, “Prefer markdown handoffs until proven”.

### PreCompact hook (Claude Code)

- Source: `hooks/pre-compact.sh` → installed at `~/.engram/hooks/pre-compact.sh`
- Claude settings: `PreCompact` → that script (async)
- On compact: writes `~/.engram/promotion-window.json` with `compaction_id`
- MCP `remember` reads that file and **resets the 0–5 promotion window**
- Also auto-observes `source=claude:precompact` (best-effort)

### get_context durable pack (2026-07-10)

`get_context` now prefers **graph Decision/Preference/… packets** (same rescue as
explicit recall + type listing), before project-file fallback. Session-start
continuity no longer requires the agent to call `recall` first for strategy facts.

### Product plan + release gates

- Plan / checkboxes: `docs/product/CONTINUITY_PRODUCT_PLAN.md`
- Golden loop contract: `docs/GOLDEN_LOOP.md`
- Continuity smoke: `uv run engram continuity --smoke` (CI job `continuity`)
- Client-promoted high-signal entities → **identity_core**
- `decision_statement` scrap filtered from rescue, context, and entity search
- Operator note: **open_work_count is hygiene, not product success**

### Recall budget defaults (tuned 2026-07-09)

| Knob | Old | New |
|------|-----|-----|
| `recall_budget_explicit_ms` | 2000 | **4000** |
| `recall_budget_explicit_search_ms` | 650 | **1500** |
| `recall_fast_preflight_timeout_ms` | 250 | **400** |
| `recall_fast_fallback_timeout_ms` | 100 | **250** |

### Still true / still broken

- Loaded-brain hybrid recall can still be slow; durable entity rescue is a safety
  net, not a full ranking redesign.
- Open deferred evidence backlog remains huge; do not chase LongMemEval fitting.
- Cross-encoder merge can still fail/timeout on full cycles.
- Prefer markdown/git handoffs until multi-agent dogfood is routine.

### Operator runtime (Konner's machine)

| Item | Value |
|------|-------|
| Server | LaunchAgent `dev.engram.local` — `engramctl status` |
| Health | `http://127.0.0.1:8100/health` |
| MCP | `http://127.0.0.1:8100/mcp` |
| Config | `~/.engram/.env` |
| Data | `~/.helix/engram-native-dogfood-axi` |
| Install note | Local editable `engram` tool may be installed from `~/Engram/server`; helix-native pin is separate |

Restart: `engramctl stop && engramctl start`

---

## ARCHIVE — Dogfood Status (2026-07-01)

### TL;DR (historical)

Engram dogfood is **healthy and live** on native Helix PyO3. Recall hot-path
fixes and junk evidence drain are **shipped and applied**. Open-work backlog
dropped from ~22k to ~13k. The remaining ~13k deferred rows are **borderline
facts**, not junk — they drain slowly via warm-tier `evidence_adjudication`
(200/cycle) while `engram serve` stays up. Full consolidation cycles still
**timeout during merge** (cross-encoder). Do not re-audit architecture or
re-discover drain vs adjudication — the answers are below.

### Operator runtime (Konner's machine)

| Item | Value |
|------|-------|
| Server | LaunchAgent `dev.engram.local` — `engramctl status` |
| Health | `http://127.0.0.1:8100/health` |
| MCP (Cursor/Grok) | `http://127.0.0.1:8100/mcp` |
| Config | `~/.engram/.env` |
| Data dir | `~/.helix/engram-native-dogfood-axi` (~17 GB) |
| Logs | `~/.engram/logs/engram.log` |
| Mode | `ENGRAM_MODE=helix`, `ENGRAM_HELIX__TRANSPORT=native` |
| Profiles | consolidation `standard`, recall `all`, integration `rework` |
| Worker | enabled (`ENGRAM_ACTIVATION__WORKER_ENABLED=true`) |

Restart: `engramctl stop && engramctl start`

### Live brain metrics (2026-07-01)

| Metric | Value | Notes |
|--------|-------|-------|
| Episodes | 5,434 | Harness auto-capture active |
| Cues | 5,295 | 97.4% cue coverage |
| Projected | 3,991 | 74% cue→projection conversion |
| Entities | 1,155 | Down from ~1,246 (merge/prune) |
| Relationships | 346 | |
| **Open work** | **13,364** | Was ~21,847 before drain |
| Deferred evidence | 13,171 | Borderline — adjudication queue |
| Pending evidence | 91 | |
| Pending edge requests | 89 | Server LLM adjudication **off** |
| Consolidation cycles | 10 | Scheduler active |
| Junk drain candidates | 24 | Was ~8,450 rejected live |

Entity mix: 555 Artifact, 248 Technology, 163 Decision, 162 Concept, 12
Project, 8 Schema, 1 Person.

### Brain loop status

```text
Capture -> Cue -> Project -> Recall -> Consolidate
 active    97%    active     ready      active (scheduler on)
```

- **Capture:** harness `api_auto_observe` writing episodes
- **Cue:** 97.4% coverage; triage skips low-signal episodes
- **Project:** worker projecting high-score episodes; 0 failures / 0 dead letters
- **Recall:** `recall_profile=all`; probe-timeout guards fixed via `GatedGraphStore`
- **Consolidate:** three-tier scheduler + backlog trigger (open work > 500 → warm
  tier every ~5 min cooldown)

Verify live state without re-deriving:

```bash
curl -s http://127.0.0.1:8100/api/lifecycle/summary | python3 -m json.tool
# MCP tools: get_graph_state, get_lifecycle_summary, get_consolidation_status
```

### Shipped this milestone (commits on `main`)

**PR1–PR3 — recall + backlog** (`2dfbae4` → `8728762`)

- Helix fast stats (`exact=False` count-only) — stats probe no longer blocks graph
  expansion on timeout
- Cached adjudication metrics on recall hot path
- Backlog-driven warm consolidation when `open_work_count > 500`
- Tier timestamp persistence (SQLite + Helix sidecar)
- Harness observe normalization (`normalize_harness_observe_content`)
- `GatedGraphStore` — single gate after graph expansion; blocks secondary graph
  reads on probe timeout (structural fix, not whack-a-mole guards)

**Evidence drain CLI** (`5271e53`, `3feb0d8`)

- `python -m engram.consolidation.drain_evidence --mode audit|reject-junk`
- Junk rules in `evidence_drain.py`: path-like names, bootstrap spans, markdown
  fragments, low-confidence identity, harness UUID paths, etc.
- Helix `update_evidence_status` fixed to persist `deferred_cycles` / `confidence`
- Id cache priming on `find_evidence_by_status` for fast bulk reject
- **Live applied:** ~8,450 junk rows rejected; re-audit shows ~24 junk left

### Backlog mechanics (do not re-investigate)

Two separate tools — easy to confuse:

| Mechanism | What it does | Automatic? |
|-----------|--------------|------------|
| **`drain_evidence` CLI** | Rejects obvious junk | **No** — manual only |
| **`evidence_adjudication` phase** | Reviews 200 open rows/cycle; promote, re-defer, or force-commit after 5 cycles | **Yes** — warm-tier scheduler |

`get_pending_evidence(limit=200)` pulls `pending` + `deferred` + `approved` open
rows, sorted by confidence. MCP being connected does not drive drain; keeping
`engram serve` up does drive adjudication.

Earlier live adjudication: **194 processed, 0 materialized** — queue count drops
slowly because most borderline rows get re-deferred until corroboration,
threshold, or 5-cycle force-commit.

Key config (`server/engram/config.py`):

- `consolidation_open_work_backlog_threshold=500`
- `consolidation_open_work_backlog_cooldown_seconds=300`
- `consolidation_tier_warm_seconds=7200` (2h normal; backlog bypasses)
- `evidence_forced_commit_cycles=5`
- `edge_adjudication_server_enabled=false` (89 pending edge requests idle)

### Known issues (accepted — do not spend tokens re-diagnosing)

1. **Merge phase timeouts** — full consolidation cycles killed during merge
   cross-encoder on loaded 17G brain. Hot triage runs fine; warm/cold full passes
   are heavy. Prioritize phase isolation or merge budget tuning if tackling this.
2. **Low adjudication materialization yield** — borderline deferred facts cycle
   without committing. Expected at current thresholds; force-commit at 5 cycles is
   the safety valve.
3. **MCP lifecycle underreports scheduler** — `get_lifecycle_summary` via MCP
   shows `schedulerActive: false`; REST `/api/lifecycle/summary` is authoritative.
4. **Stale shutdown cycle artifacts** — interrupted `trigger=shutdown` cycles can
   appear `running` in audit store; `isRunning: false` on live API is truth.
5. **`CURRENT_HANDOFF.md` archive below** — pre-2026-07-01 entries document
   earlier milestones (startup latency, adoption gates, GraphManager extractions).
   Do not treat them as current dogfood blockers unless this section says so.

### Next levers (prioritized)

1. **Let scheduler run** — warm adjudication at 200/cycle with backlog trigger;
   monitor `open_work_count` over days via `get_graph_state` or lifecycle API.
2. **Reject remaining junk** (optional, quick):
   `uv run python -m engram.consolidation.drain_evidence --mode reject-junk --yes --helix-data-dir ~/.helix/engram-native-dogfood-axi`
3. **Enable server LLM edge adjudication** (optional) — 89 pending edge requests;
   set `edge_adjudication_server_enabled=true` in `~/.engram/.env`, restart.
4. **LLM curator phase** (not built) — discussed for bulk borderline deferred;
   would be new consolidation phase, not an extension of drain.
5. **Merge timeout fix** — if full cycles needed: cross-encoder budget, skip tier-1
   under pressure, or run merge in isolated subprocess with timeout.
6. **Adoption evidence** — recall evaluation signals still thin; harness capture
   is active but confirmation/correction metrics need live-harness transcripts.

### Key files (this milestone)

| Area | Path |
|------|------|
| Fast Helix stats | `server/engram/storage/helix/graph.py` |
| Recall graph gate | `server/engram/retrieval/recall_graph_gate.py` |
| Recall pipeline | `server/engram/retrieval/pipeline.py` |
| Backlog scheduler | `server/engram/consolidation/scheduler.py` |
| Evidence drain | `server/engram/consolidation/evidence_drain.py` |
| Drain CLI | `server/engram/consolidation/drain_evidence_cli.py` |
| Evidence adjudication | `server/engram/consolidation/phases/evidence_adjudication.py` |
| Harness normalization | `server/engram/ingestion/capture_surface.py` |
| Tests | `server/tests/test_evidence_drain.py`, `test_helix_stats.py`, recall gate tests |

### Commands cheat sheet

```bash
# Status / health
engramctl status
curl -s http://127.0.0.1:8100/health | python3 -m json.tool
curl -s http://127.0.0.1:8100/api/lifecycle/summary | python3 -m json.tool

# Drain audit / reject junk
cd server && uv run python -m engram.consolidation.drain_evidence \
  --mode audit --helix-data-dir ~/.helix/engram-native-dogfood-axi
cd server && uv run python -m engram.consolidation.drain_evidence \
  --mode reject-junk --yes --helix-data-dir ~/.helix/engram-native-dogfood-axi

# Manual consolidation (may timeout on merge)
cd server && uv run python -m engram.consolidation \
  --profile standard --group default --no-dry-run

# Tests (affected suites)
cd server && uv run pytest tests/test_evidence_drain.py \
  tests/test_helix_stats.py -v -m "not requires_helix"

# Lint
cd server && uv run ruff check .
```

### Agent protocol reminder

Engram is the portable memory authority across harnesses. On session start:
`get_context(project_path=...)` before substantive answers; `recall` when prior
context could change the response. Harness auto-capture is active — do not
duplicate with per-turn `observe`. Use `remember` for high-signal cross-context
facts with `proposed_entities` + `proposed_relationships`. Empty graph = onboarding,
not proof Engram is useless.

---

## Stable context (unchanged since 2026-05-19 closeout)

Build Engram into a production-grade "one brain per person" memory runtime:

```text
Capture -> Cue -> Project -> Recall -> Consolidate
```

Preferred local path: Helix native PyO3 (`ENGRAM_MODE=helix`,
`ENGRAM_HELIX__TRANSPORT=native`). SQLite/lite is smoke/demo fallback only.

Core brain-runtime goal **passed closeout audit 2026-05-19**. Remaining work is
adoption evidence, dogfood scale hardening, and backlog hygiene — not
re-architecting the loop.

AI-harness adoption: MCP connection alone is insufficient. Agents must use Engram
as portable memory authority even when project-local files exist; treat empty
runtime as onboarding; bootstrap when `artifactCount` is low. Project-local files
own repo conventions; Engram owns user facts, preferences, durable decisions,
corrections, goals, and cross-harness continuity.

---

## Historical Handoff Log (archive — pre-2026-07-01)

Latest native PyO3 startup-latency follow-up: local FastEmbed no longer
constructs the ONNX model during store creation when the configured model has a
known dimension, and predicate context-gating embeddings now warm in a tracked
background task after readiness instead of blocking `_startup`. The installed
tool had to be reinstalled with `uv tool install --force --no-cache ...`
because uv reused the same-version local wheel cache on the first reinstall.
After the real reinstall, `engramctl stop && engramctl start` returned healthy
in `24.59s` on the loaded 4.0G native store, with the server-side app startup
path from first Python log to health reduced to about `1.5s`: FastEmbed was
`configured lazy`, Atlas/consolidation/conversation stores initialized in the
startup window without materializing the model, and predicate embedding warmup
completed later with `predicates=32`. A confirmed startup matrix then produced
`/private/tmp/engram-dogfood-startup-20260528-104401` with `12 pass, 0 warn,
0 fail, 0 skip`; its start-runtime step completed in `16.333s`, and the final
runtime is healthy on LaunchAgent PID `13527`. Remaining startup work is now
the pre-app/LaunchAgent/import delay before the first Python log, not Helix
store construction or FastEmbed initialization inside readiness.

Latest resumed native PyO3 dogfood pass: the remaining `medium` degraded samples
were not coming from explicit recall anymore; they were MCP read-tool
auto-recall gate samples. The middleware checked `identity_core`,
`project_home`, and `explicit_recall` packets before falling through to medium,
but it ignored the fresh `session_recent` packets that now carry Codex
continuity evidence. Auto-recall now includes `session_recent` in the
startup-safe recent-packet lookup, so read-tool enrichment can return the fresh
observation and record `modeExecuted=cached` instead of timing out in medium.
AXI value reporting also now surfaces operation/source/status/skip-reason
counts and recent problem samples, so future medium timeouts identify whether
they are `auto_recall_gate`, explicit recall, packet cache, or another source.
After reinstall/restart to LaunchAgent PID `95327`, live marker
`sapphiremarker`/`zirconmarker` proved the path: MCP `recall` returned the
fresh `session_recent` packet with `duration_ms=0.7616`,
`skip_reason=cache_satisfied`, and attached
`recalled_context.gate.modeExecuted=cached`; MCP
`get_context` returned the same packet with `duration_ms=0.1077`; AXI recall was
`cache_satisfied` in `1.1359ms`; AXI context returned from packet cache in
`0.0625ms`; and `engram axi value --json` reported read-path p95 `22.4422ms`,
read cache hit rate `1.0`, and zero read degradation/timeouts. The `medium`
mode now shows `operation_counts.auto_recall_gate=2`,
`skip_reason_counts.cache_satisfied=2`, `timeout_count=0`, and
`degraded_count=0`. Focused backend/AXI tests passed with `88 passed`; ruff
passed on touched files; full startup validation passed against the current
runtime. Caveat: the reinstall restart itself still took `1:22` wall time, with
server startup logs reaching healthy after roughly 30 seconds, so startup
latency remains a live goal item even though the read-tool medium timeout is
fixed.

Latest native PyO3 dogfood performance hardening checkpoint: MCP `observe` no
longer awaits recall middleware on the write path, agent capture waits are
bounded to `100ms`, and `session_recent` packets remain in the in-memory packet
cache with `sync_persistent=false`. Context now lets strong marker/date/id
`session_recent` matches satisfy exact topic queries without loaded-store or
project-file enrichment, while broad recent-only topics still enrich. Normal
graph invalidation preserves the `session_recent` scope so projection or
materialization does not erase the fresh raw observation before the next agent
turn. The native Helix migration path also writes a
`vector_integrity_verification_version` marker after a successful verification
and skips that expensive migration-only scan on subsequent dogfood starts; the
first patched restart still verified the existing 4.0G store, and the next
restart skipped the vector scan. After reinstall/restart to PID `68316`, two
MCP observe samples with `rubymarker`/`emeraldmarker` returned bounded
`capture_store_timeout` diagnostics at `101ms`; MCP `get_context` then returned
the exact two `session_recent` packets in `0.3724ms`, AXI context returned them
in `0.4443ms`, and AXI recall was `cache_satisfied` in `6.9803ms`, with no
degradation, timeout, or budget miss. `engram axi value --json` reported
read-path cache hit rate `1.0`, read-path p95 `208.0112ms` at the value layer
with zero read budget misses/degradation/timeouts, and write-path p95
`114.9663ms`. Full startup validation passed; the lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-101318` with `13 pass, 0 warn,
0 fail, 0 skip`. Focused Python gates passed with `182 passed, 3 skipped`;
focused native Helix migration tests passed with `14 passed`; ruff and
`git diff --check` passed.

Latest native PyO3 dogfood recall closeout: the loaded-store no-evidence tail is
now bounded by cached project packets after fast preflight. Explicit recall can
use same-project home packets or identity packets as weak fallback context after
a fast preflight miss/timeout, caps that preflight to the shorter fallback
budget when such packets already exist, and syncs the persistent packet cache
when `project_path` is supplied so post-restart recalls do not start from an
empty in-memory cache. GraphManager also avoids paying duplicate legacy BM25
fallback work when direct record-backed cue/episode search methods are present,
and recall pool stats now preserve timeout behavior while supporting stores
that still reject `exact=False`. After reinstall/restart, a true miss
`zzpersist noartifact yonderplasm quibbleflux 20260528 final true miss tail`
returned three project packets in `100.1129ms` with
`skipReason=preflight_timeout_context_packet_fallback`, no deep recall timings,
no degradation, and no budget miss. After the confirmed lifecycle matrix
restarted the runtime again, `zzaftermatrix noartifact yonderplasm quibbleflux
20260528 final true miss tail` still returned useful project context in
`101.9488ms` with the same non-degraded lifecycle. Final runtime status is
healthy on LaunchAgent PID `28314`; `engram axi value --json` reports read-path
p95 `65.7748ms`, cache hit rate `1.0`, and zero read budget misses,
degraded reads, or timeouts. Full startup validation passed after the final
reinstall, including a live MCP catalog with 27 tools, `remember` present,
`recall.project_path`, and a `cache_satisfied` recall probe at `query_time_ms=1.2`.
The refreshed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-071304` with
`13 pass, 0 warn, 0 fail, 0 skip`. Focused backend gates passed with
`94 passed, 2 skipped`, ruff passed on the touched retrieval files/tests, and
`git diff --check` is clean. The goal remains active for longer real Codex
dogfood continuity evidence, but the no-evidence loaded-store recall tail is no
longer the current blocker.

Latest packet-cache diagnostics follow-up: a resumed 2026-05-28 goal turn
rechecked the live native PyO3 dogfood store after `6dc928f`. Runtime was
healthy on port `8100`, storage was still the 4.0G native data dir, and the
on-disk LaunchAgent `ThrottleInterval` was restored from the experiment back to
`10`. Hot AXI/MCP context and recall paths were cache-satisfied: AXI context
returned from packet cache in `0.0994ms`, AXI recall in `2.0336ms`, MCP
`get_context` in `0.0556ms`, and MCP `recall` in `25.1052ms`, with no
degradation or budget miss. After explicit packet-cache clears, isolated AXI
context initially rebuilt useful project-file packets in `70.5027ms`; MCP cold
context returned useful project-file packets in `194.023ms` with
`loaded_store_context_preflight=76.0465ms`; and isolated AXI cold recall
returned project-file packets in `532.0314ms`, mostly from bounded empty
preflight/live-search timings rather than packet formatting. A post-restart
empty-cache AXI context then exposed the real first-turn project-file scan
edge: `project_file_fallback=7022.0675ms` and a wall-budget miss. Context now
returns a small `project_file_pending` packet after the soft wait while the full
project-file scan caches the exact topic in the background. After reinstall and
restart to LaunchAgent PID `33029`, `engramctl start` completed in `21.439s`;
after `engram axi packet-cache clear`, the cold AXI context probe returned in
`80.5724ms` with `projectFileFallbackPending=1.0`, no degradation, and no
budget miss; a follow-up same-topic context hit warmed project packets in
`0.0566ms`. `engram axi packet-cache --json` reported two fresh persistent
`project_home` entries at
`/Users/konnermoshier/.helix/engram-native-dogfood-axi/packet-cache.sqlite3`,
and `engram axi value --json` reported read-path p95 `80.793ms`, cache hit rate
`0.5`, and zero read degradation/timeouts/budget misses. The code also adds a
read-only `GET /api/knowledge/packet-cache` plus `engram axi packet-cache`
summary command so agents can inspect cache entry count, scopes, hits,
persistence, and sidecar path without clearing the cache. Focused AXI/API/context
tests passed with `164 passed`; ruff and `git diff --check` passed on touched
files. A follow-up explicit recall patch now skips deep loaded-store search
when fast preflight times out and bounded project-file fallback packets are
ready. After reinstall/restart to LaunchAgent PID `37398` and
`engram axi packet-cache clear`, the cold no-evidence AXI recall probe returned
three `project_home` packets in `277.7469ms` with
`skipReason=preflight_timeout_project_file_fallback`, no `recallSearch` timing,
no degradation, and no budget miss. `engram axi value --json` then reported
read-path p95 `276.7128ms`, zero read budget misses, and the remaining degraded
sample was an MCP `auto_recall_gate` timeout from the live status check, not
explicit recall. `engram axi packet-cache --json` reported two fresh persistent
`project_home` entries after the probe. A follow-up LaunchAgent startup pass
then traced the remaining `72.043s` matrix start tail to the supervised restart
path, not the server or native data path: a direct foreground `engram serve`
against the same 4.0G dogfood native Helix directory started on a temporary port
in about `1s`. `engramctl start` now avoids killing a freshly bootstrapped
RunAtLoad job and repairs legacy LaunchAgent plists from `/bin/zsh -lc` to
`/bin/zsh -c`, including cleanup for the accidentally duplicated `-lc` form.
After installing the repaired `engramctl`, the local plist is clean
(`ProgramArguments: /bin/zsh, -c, source ~/.engram/.env ...`), a direct
supervised restart held at `5.427s`, and the refreshed lifecycle matrix at
`/private/tmp/engram-dogfood-startup-20260528-113523` passed with
`13 pass, 0 warn, 0 fail, 0 skip`; its `start runtime` step was `6.124s`.
Post-matrix AXI context returned useful project context in `71.7778ms`, AXI
recall was `cache_satisfied` in `2.9092ms`, MCP `get_context` returned useful
project packets in `32.6169ms`, and MCP `recall` was `cache_satisfied` in
`35.0792ms`, all without degradation or budget misses. `engram axi value
--json` reported read-path p95 `71.8991ms`, read cache hit rate `1.0`, and zero
read degradation/timeouts/budget misses. The next performance target is more
real Codex-session continuity evidence rather than a known recall/startup tail.
Second real Codex-session sample after commit `1b98a19`: the resumed Codex turn
called MCP `get_context` before work and got useful project packets in
`249.4908ms` with no degradation or budget miss; MCP `recall` then hit cache in
`1.4862ms`. Matching AXI probes showed the warmed same-topic context path using
project-file cache rescue in `2.5496ms`, same-topic recall `cache_satisfied` in
`32.0827ms`, and a fresh no-evidence recall returning a project packet in
`101.3667ms` with `preflight_timeout_context_packet_fallback`. A brand-new
context-miss topic then hit cache rescue in `4.3832ms`; MCP `get_context` on
that topic hit cache in `0.2735ms`, and MCP `recall` was `cache_satisfied` in
`45.9587ms`. The current `engram axi value --json` read path still reports zero
budget misses, degraded reads, or timeouts. This is the first post-commit
evidence that the `1b98a19` cache/fallback behavior holds across a resumed real
Codex turn without another runtime change.
Third Codex-session/restart sample: the next resumed turn intentionally probed
the apparent next bottleneck, a fresh topic-specific `get_context` for write-path
capture/observe context. The first MCP context call found useful project packets
but paid a cold project-file scan (`duration_ms=581.6269`,
`project_file_fallback=550.3093ms`). The same call created the stable
same-project cache entry. After an actual `engramctl stop && engramctl start`
restart to PID `35144`, the first new context topic
`post restart stable rescue verdant zephyr 20260528 first context` returned via
project-file cache rescue in `2.3597ms`, and the matching AXI recall returned
three project packets in `106.2332ms` with
`preflight_timeout_context_packet_fallback` and no degradation. MCP
`get_context` on that same post-restart topic hit packet cache in `0.0617ms`;
MCP `recall` was `cache_satisfied` in `89.6023ms`. Post-restart
`engram axi value --json` reported read-path p95 `106.2332ms`, cache hit rate
`1.0`, and zero read budget misses, degraded reads, or timeouts. This proves the
stable project packet survives restart once warmed. The follow-up installed
runtime check found one more soft-wait edge: the MCP context helper could still
wait for a slow project-file scan to finish before trying same-project cache
rescue. That helper now waits only the loaded-store soft budget, then lets the
project-file payload builder rescue from stable cache while the fresh scan keeps
warming the exact topic. After reinstall/restart to LaunchAgent PID `40680`, the
first fresh MCP topic with no usable stable sidecar entry rebuilt in
`937.6488ms`; once that stable entry existed, fresh AXI context used
`project_file_cache_rescue` in `10.3801ms` and exact repeat context hit cache in
`0.0413ms`. Fresh MCP topics were bounded without degradation (`104.0038ms` and
`138.8205ms`) because the local scan completed quickly before rescue was needed.
After the final no-project guardrail reinstall/restart, the runtime is healthy on
LaunchAgent PID `41982`, and a fresh AXI context probe used
`project_file_cache_rescue` in `2.238ms`.
The remaining cold edge is first-ever stable project packet creation after the
packet cache has no same-project home entry.
Current continuation checkpoint: HEAD `5d3554d` was clean and pushed before
this pass, and the installed dogfood runtime was still running native PyO3 Helix
from that worktree. A resumed real Codex turn then exposed a cache invalidation
edge: after normal MCP capture/projection, stable same-project `project_home`
file packets had been invalidated, so the next fresh MCP `get_context` returned
useful packets but spent `715.1811ms` total with
`project_file_fallback=709.6484ms`. Broad graph/episode packet-cache
invalidation now preserves cache entries whose packets are all
`trust.source=project_file`; explicit clears and mutable cue/entity/relationship
packet invalidation still clear stale packet views. After reinstall/restart to
LaunchAgent PID `45085`, AXI context seeded project-file cache rescue in
`2.2862ms`. A live MCP `observe` stored `ep_4c0605de51da`; after background
projection ingested it, the `project_home` project-file rows remained
uninvalidated. Post-observe MCP `get_context` returned useful packets in
`83.4848ms` with no degradation; AXI context then hit
`project_file_cache_rescue` in `3.608ms`; AXI recall was `cache_satisfied` in
`2.3008ms`; and `engram axi value --json` reported read-path p95
`83.8738ms`, read cache hit rate `0.7778`, and zero read budget misses,
degraded reads, or timeouts. Focused context/recall/packet-cache tests passed
with `112 passed`, ruff passed on touched files, and `git diff --check` is
clean. The goal remains active for longer real Codex dogfood continuity
evidence, but normal observe/projection no longer erases stable project-file
rescue packets.
Post-`1029cf7` reinstall/restart dogfood evidence: the local package was
reinstalled from the clean checkout and the LaunchAgent was restarted to PID
`48229`. AXI home stayed startup-safe and healthy; fresh AXI context for
`post reinstall 1029cf7 live context probe 20260528` used project-file cache
rescue in `2.8982ms`, and the matching AXI recall was `cache_satisfied` on
repeat in `1.9462ms`. A forced no-evidence AXI recall
`zz1029cf7 noartifact quibbleplasm yonderflux 20260528 forced miss` returned
three project packets in `102.0141ms` with
`preflight_timeout_context_packet_fallback`, no degradation, and no empty
timeout payload. MCP `get_context` returned useful packets in `161.2856ms`, and
MCP `recall` was `cache_satisfied` in `5.6553ms`. The live MCP catalog probe
still exposes 27 tools including `remember` and `recall.project_path`; startup
validation passed; and the confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-075531` with
`13 pass, 0 warn, 0 fail, 0 skip`, leaving the post-matrix runtime healthy on
PID `52284`. Post-matrix `engram axi value --json` reports read-path p95
`73.4648ms`, read cache hit rate `1.0`, and zero read budget misses, degraded
reads, or timeouts. Recall-quality sample `ers_145ab372177d` records this as
useful real Codex dogfood evidence. No new code patch was justified by this
sample; the next useful work is more real-session continuity evidence and
watching write-path samples, not another speculative recall/context change.
Latest real-session recall-priority fix: the next resumed Codex turn found a
packet-cache relevance regression rather than a loaded-store latency issue.
Fresh session packet `ep_971656d12ca9` surfaced once, then repeat recall for
the same resumed-goal query was starved by newer `project_home` rows because
`cached_context_recall_packet_payloads()` fetched `session_recent`,
`identity_core`, and `project_home` together under one global recency limit. The
fallback now gathers `session_recent`, `identity_core`, and `project_home`
scope groups separately, dedupes them, and lets the existing relevance filter
choose the useful packets. After reinstall/restart to LaunchAgent PID `62723`,
live MCP observe stored fresh marker `ep_0352d83b5ece`; two newer AXI context
calls warmed project-home packets; and repeated AXI recalls for
`recent priority liveproof ultramarine 20260528 fresh observation` returned the
recent observation first in `3.2818ms` and `2.2598ms` with
`skipReason=cache_satisfied`, no degradation, and no budget miss. The focused
backend gate passed with `113 passed`; ruff passed on the touched retrieval
files/tests; startup validation passed all 14 checks against PID `62723` with
the live MCP catalog still exposing 27 tools including `remember` and
`recall.project_path`. Post-validation `engram axi value --json` reports
read-path p95 `81.5933ms`, read cache hit rate `0.95`, and zero read budget
misses, degraded reads, or timeouts. The goal remains active for longer real
Codex continuity evidence, but fresh observations are no longer starved by
project-home cache recency.
Latest project-file context usefulness pass: live dogfood after `1a52a5f`
showed a new first-use quality edge. The runtime was fast and non-degraded, but
while a fresh local project-file scan was still running, context could return
older stable `project_file_cache_rescue` packets for a very specific topic. The
context fallback now waits one bounded `context_fast_preflight_soft_wait_ms`
window for the in-flight project-file scan before falling back to stable rescue,
so quick local scans can return the current exact project evidence on the first
call. After reinstall/restart to LaunchAgent PID `70588`, AXI context for
`fresh observations are no longer starved by project-home cache recency live
softwait firstuse 20260528` returned the current `docs/CURRENT_HANDOFF.md`
packet first with no degradation or budget miss (`durationMs=1083.3078`,
`projectFileFallback=926.1808ms`, `projectFileFallbackSoftWait=75.8946ms`);
the exact repeat hit packet cache in `0.0475ms`. MCP `get_context` for the same
topic hit cache in `0.069ms`, MCP `recall` was `cache_satisfied` in `23.2ms`,
and a forced no-evidence AXI recall still returned project context in
`100.7934ms` with `preflight_timeout_context_packet_fallback`, not an empty
timeout payload. A stricter follow-up now filters stable
`project_file_cache_rescue` packets through topic relevance before returning
them; live AXI context for `stable project-file rescue packets relevant topic
soft-wait filter 20260528` returned only relevant current handoff/AXI-plan
packets in `130.8898ms` with no budget miss or degradation. The full startup
validator passed all 14 checks against PID `74441`, and the final confirmed
lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-082753` with
`13 pass, 0 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on PID
`75670`. Post-matrix `engram axi value --json` reports read-path p95
`73.0892ms`, cache hit rate `1.0`, and zero read budget misses, degraded reads,
or timeouts. The remaining performance target is lowering the cold exact
project-file scan cost, not rescue relevance or degraded/empty recall behavior.
Latest recent project-file reuse pass: the next resumed Codex turn showed that
the exact local parser was not the real bottleneck (`_project_file_fallback_packets`
measured about `19ms` in-process), but the live context surface could still miss
nearby cached project-file packets and pay another scan/soft-wait path. Context
cache lookup now asks the packet cache for recent same-project `project_home`
packets when direct relevant cache lookup has not already found loaded-store or
exact project-file context. Those packets must be current-version
`trust.source=project_file`, same project, and topic relevant; if the query has
specific tokens such as dates or hyphenated terms, recent reuse now requires a
specific-token match instead of accepting weak broad-word overlap. Duplicate
stable packets are upgraded to `_cache_scope=project_file_recent_reuse`, so they
can satisfy context without waiting for loaded-store preflight or a fresh file
scan. After reinstall/restart to PID `81443`, AXI context for `stable
project-file rescue packets relevant topic soft-wait filter 20260528 final3
nearby reuse` hit `project_file_recent_reuse=2` in `0.0451ms`, and MCP
`get_context` for the same family hit `project_file_recent_reuse=2` in
`0.1312ms`, both with no degradation or budget miss. Startup validation passed
against PID `81443`; the final confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-085123` with
`13 pass, 0 warn, 0 fail, 0 skip`, leaving PID `82520` healthy. Immediately
after that matrix, a fresh topic rebuilt bounded project-file context in
`99.5414ms`, and the nearby topic then hit recent reuse in `0.0447ms`. Focused
retrieval/cache tests passed with `117 passed`; ruff and `git diff --check`
passed. The remaining latency target is cold startup/runtime warmup and
first-ever project packet creation, not nearby-topic context reuse.
Latest cache-specificity hardening: a fresh real probe showed that the recent
reuse win was slightly too broad. Queries with a new marker such as `orchid`
could be satisfied by older project-file packets that only matched generic terms
and the shared date. Recent project-file context reuse now requires all
distinctive non-generic query tokens to be present before it can skip the
bounded project scan, and explicit recall applies the same guard before treating
non-exact project-file packets as `cache_satisfied`. Exact-topic project-file
fallback packets still satisfy repeat calls. After reinstall/restart to PID
`84563`, AXI context for `live dogfood loaded-store recall context trace
orchid2 20260528 specificity probe` no longer hit stale recent reuse; it rebuilt
project-file packets in `1020.0549ms` with no degradation or budget miss. The
matching AXI recall no longer reported `cache_satisfied`; it ran bounded recall
and returned project-file fallback packets with `skipReason=null`,
`durationMs=1139.2205`, and no degradation. Repeating the exact context hit
cache in `0.0413ms`, and MCP `recall` for the exact MCP topic was
`cache_satisfied` in `2.4848ms`. Focused retrieval/cache tests now pass with
`119 passed`; ruff and `git diff --check` pass. Startup validation passed
against PID `84563`; the final lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-090414` with
`13 pass, 0 warn, 0 fail, 0 skip`, leaving PID `85767` healthy. Post-matrix
value reports read-path p95 `85.9715ms`, read cache hit rate `1.0`, and zero
read budget misses, degradation, or timeouts. The next performance target is
reducing first-miss project-file fallback cost without reopening noisy cache
satisfaction.
Latest immediate-rescue pass: the next cold-restart probe showed a smaller
latency leak after cache-specificity hardening. A first AXI context call with a
usable same-project rescue packet still paid the project-file soft wait:
`coldstart first request latency lapismarker 20260528 project file fallback`
returned in `129.2005ms` with `projectFileFallbackSoftWait=75.0566`. Context
now checks topic-relevant same-project `project_file_cache_rescue` packets
before waiting on either loaded-store preflight or the project-file fallback
soft wait, while the fresh project-file scan continues in the background and
updates the exact-topic cache. The same specificity guards still apply, so
generic overlap does not satisfy a new marker query. After reinstall/restart to
PID `97485`, AXI context for `early rescue mcp axial garnetmarker 20260528
project file fallback` returned via `project_file_cache_rescue=1` in
`46.3217ms` with `projectFileFallbackSoftWait=0.0`; MCP `get_context` for
`early rescue mcp sapphiremarker 20260528 project file fallback` returned via
the same rescue path in `5.8787ms` with no loaded-store preflight wait; and MCP
`recall` for that topic was `cache_satisfied` in `43.1145ms`. Focused
retrieval/cache tests passed with `119 passed`; ruff and `git diff --check`
passed; the installer doctor passed; startup validation passed against PID
`97485`; and the confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260528-092158` with
`13 pass, 0 warn, 0 fail, 0 skip`, leaving PID `99293` healthy. Post-matrix
value reports read-path p95 `4.3265ms`, read cache hit rate `1.0`, and zero
read budget misses, degradation, or timeouts. Overall p95 is now dominated by
write-side `api_auto_observe`, not read-path context/recall.

Latest dogfood performance note: the native PyO3 path now uses generated bulk
Helix stats routes for evaluation graph-state refresh. The previous
server-backed `engram evaluate --server-url http://127.0.0.1:8100` run after a
fresh dogfood restart still returned `graph_state_timeout` and zero graph totals.
The stats path now queries cue rows and projected episode/entity links in bulk,
falls back to the older per-episode route set for old native builds, and the
bundled PyO3 route source rebuilds with 180 routes. Live restart on 2026-05-27
logged `routes=180`; the server-backed evaluation report returned `1414`
episodes, `901` entities, `8109` relationships, Capture `ready`, Cue
`attention`, and Project `active`. AXI context/recall and MCP get_context/recall
all returned useful packets without degradation after that restart. The follow-up
startup matrix at `/private/tmp/engram-dogfood-startup-20260527-112804` passed
with `13 pass, 0 warn, 0 fail, 0 skip`. A final report-service guard now marks
runtime-only graph stats as `graph_state_unavailable` instead of presenting a
cold fallback as a real empty graph; after restart on PID `14146`, server-backed
evaluation immediately returned `1416` episodes, `901` entities, and `8109`
relationships with no report degradation.

Latest AXI latency follow-up: `engram axi recall --json` now preserves REST
diagnostics, including recall-search, preflight, packet-cache, and fallback
stage timings. The loaded native dogfood store still showed a hard query where
fast preflight hit its `250ms` cap and deep search hit its `650ms` cap, but the
bounded project-file fallback returned useful packets. GraphManager now races
record-backed cue and episode fast searches so a stalled cue-record lookup cannot
starve a ready episode-record hit. Cached packets explicitly marked
`trust.source=project_file` can also satisfy the next matching explicit recall,
while generic file-context packets remain conservative. After reinstall/restart
on PID `31237`, the previously degraded AXI query returned
`status=ok`, `skipReason=cache_satisfied`, three packets, and
`durationMs=43.9192`; a second cached project-file query returned in
`durationMs=1.2747`. Fresh live-cost value evidence reported read-path
`p95_added_latency_ms=43.9192`, `cache_hit_rate=1.0`, and zero timeouts or
budget misses. The report endpoint still surfaced `evaluation_context_timeout`
and `live_cost_runtime_only` degradations for full evaluation context, so the
next performance target is report/context aggregation, not AXI repeat recall.
The next report pass moved live-cost reporting off consolidation-context scans,
starts native graph-stats warmup during REST startup, replaces stale graph-stats
and consolidation-context background tasks after 30 seconds, and adds native
count routes for entities, episodes, relationships, and cues. After rebuilding
and reinstalling `helix-native`, startup logs showed `routes=184`; direct native
count routes reported `901` entities, `1427` episodes, `3963` relationships, and
`1332` cues. The running server now returns both normal and `liveCost=true`
brain-loop reports with no degradation and graph totals populated
(`1427` episodes, `901` entities, `3963` relationships, `1332` cues). AXI recall
for the current dogfood status returned from cache in `54.7065ms` server-side
with no budget miss, AXI context rebuilt topic-specific project packets in
`534.6486ms`, and `python3 scripts/dogfood_startup_validation.py --skip-slow`
passed with `12 pass, 0 warn, 0 fail, 2 skip`. A repeat-cache follow-up now tags
project-file fallback packets with their exact topic/project cache key. On the
live runtime, the first fresh AXI context call for
`repeat project-file fallback cache live 20260527 exact marker residual latency`
built five project-file packets in `1142.4237ms`; the second identical call hit
packet cache in `0.1268ms` server-side with no preflight or file scan. MCP
`get_context` for the same topic also hit cache in `0.1611ms`, and MCP `recall`
returned three cached packets in `2.1714ms`. The quick startup validator still
passed with `12 pass, 0 warn, 0 fail, 2 skip` after reinstall.
The following live Codex turn then exposed write-path risk: MCP `observe`
successfully captured the continuation but spent `36503ms` in raw
`capture_store`, driving write-path p95 to `36593.9551ms`. Raw episode
persistence is now bounded by `capture_store_timeout_ms` and, on timeout, the
raw write plus queued/cue side effects finish in the background like cue/vector
work already did. After reinstall/restart on PID `94293`, REST observe samples
returned with `captureStore=122ms` and `8ms`, and a real MCP `observe` returned
in about `1.28s` wall with `capture_store=415ms`, `cue_store=123ms`, and only
bounded side-effect timeouts around `103ms`.

Latest startup-matrix closeout: the full matrix initially failed because native
`HelixGraphStore.get_stats()` returned the new count-only fast stats packet to
the disposable doctor smoke, leaving `projection_metrics.state_counts` empty even
after the smoke had projected episodes. Exact `get_stats()` is now the default
again, recall's pool-sizing stats path explicitly requests `exact=False`, and
deterministic smoke disables live capture deferral with synchronous raw/cue
storage. Installed `engram evaluate --smoke --mode helix --format json` returned
no coverage gaps with `3` projected episodes, `3` linked episode entities, and
one consolidation cycle; `engramctl doctor --format json` passed with disposable
Helix smoke readiness measured; and the confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-130309` with
`13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime stayed healthy on
LaunchAgent PID `19164`; repeat AXI context for the validation topic hit packet
cache in `0.1597ms`, MCP `get_context` hit in `0.0261ms`, and MCP `recall` hit
cache in `3.184ms`.

Latest AXI CLI routing note: live dogfood then exposed that AXI subcommands with
their own `--project` / `--topic` options were overwriting global values when the
global flags appeared before the subcommand, so `engram axi --project ... --topic
... context` could silently fall back to CWD inference and miss the intended
topic-specific fast path. The duplicate subcommand options now use suppressed
defaults so absent subcommand flags no longer erase global values, while
subcommand-local flags still override. After reinstalling the local AXI CLI,
global-before-subcommand probes from `/tmp` returned useful packets without
degradation: fresh AXI context rebuilt project-file packets in `562.4966ms`,
fresh AXI recall returned three project packets in `636.2156ms`, and
`engram axi --json doctor --project /Users/konnermoshier/Engram` passed. The
startup validator now warns when a hook trace records `project=/`, and the matrix
propagates validation warnings instead of flattening them to pass. With those
stricter checks, the confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-134521` with
`11 pass, 2 warn, 0 fail, 0 skip`; both warnings are the same Codex SessionStart
root-project trace, while stop/start/stale-PID lifecycle checks still pass.
Post-matrix status reported LaunchAgent PID `59643`; AXI context from `/tmp`
returned five project-file packets in `699.2828ms`, AXI recall returned three
project packets in `1132.7942ms`, MCP `get_context` returned five useful
project-file packets in `129.0879ms`, and MCP `recall` hit cache in `3.2646ms`,
all without degradation or budget misses. Managed Codex and Claude Code AXI hooks
now use `engram axi hook-run`, which reads the hook stdin JSON `cwd` instead of
shell `$PWD`; both local hook configs were rewritten with that command after
reinstall. Manual `hook-run` smoke with stdin `{"cwd":"/Users/konnermoshier/Engram"}`
returned a healthy packet with `brain.project=/Users/konnermoshier/Engram`. The
existing validator warning will remain until a fresh real Codex SessionStart
trace replaces the older `project=/` row.
The validator now also compares each installed hook config mtime against the
latest SessionStart trace timestamp, so it warns when startup evidence predates
the current hook command. Current skip-slow validation reports
`11 pass, 1 warn, 0 fail, 2 skip`: Codex and Claude Code both need fresh
SessionStart traces after the hook-run reinstall, and Codex still has the older
root-project row. Fresh live read probes remain healthy: AXI context returned
three loaded-store packets in `78.3556ms`, AXI recall was `cache_satisfied` in
`1.5127ms`, MCP `get_context` returned five project-file packets in `123.034ms`,
and MCP `recall` was `cache_satisfied` in `2.4123ms`.
`engram axi doctor --hooks codex claude-code --require-hook-run
--require-followup --json` now shares the same freshness rule and fails with
`stale_session_start_run` for both clients instead of accepting pre-reinstall
startup evidence. Codex's hook payload also reports `last_run_project_root=true`.
Focused regression gates passed for AXI CLI/hooks and startup-warning coverage
(`46 passed`), ruff passed on the touched AXI/validation files, and
`git diff --check` is clean.
The refreshed full lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-140202` with
`11 pass, 2 warn, 0 fail, 0 skip`; `engramctl doctor` and the live MCP catalog
passed, and both warnings are still the expected stale/root hook-run evidence.
Post-matrix runtime stayed healthy on LaunchAgent PID `74501`. Fresh post-matrix
probes stayed bounded without degradation: AXI context returned five project-file
fallback packets in `1205.4917ms`, AXI recall returned three project-file packets
in `1559.5697ms`, MCP `get_context` returned five project-file packets in
`142.7577ms`, and MCP `recall` was `cache_satisfied` in `54.5339ms`.
The next fallback-quality pass fixed two real dogfood issues in that bounded
path. Project-file fallback now scans a larger bounded topic prefix, scores
adjacent line-wrapped evidence, caps historical term-count inflation, and gives
`docs/CURRENT_HANDOFF.md` priority when it is relevant, so current evidence beats
older append-only matrix mentions. Explicit recall's context-packet fallback now
filters project-file packets by `project_path`, preventing cached packets from
another repo from satisfying an Engram-scoped recall. After reinstall/restart,
AXI context for `startup matrix 20260527 tiecheck gold` returned the current
`20260527-140202` `CURRENT_HANDOFF.md` packet in `537.2311ms`; repeat AXI
context hit cache in `0.1081ms`; AXI recall was `cache_satisfied` in `18.1105ms`;
MCP `get_context` hit the same packet in `0.0629ms`; and MCP `recall` was
`cache_satisfied` in `24.3204ms`. Focused context/recall surface tests passed
with `72 passed`, and ruff passed on the touched retrieval files/tests. The
refreshed full lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-142608` with
`11 pass, 2 warn, 0 fail, 0 skip`; after that restart, LaunchAgent PID `9661`
was healthy, cold-ish AXI context returned the current handoff packet in
`952.857ms`, and repeat AXI recall hit cache in `1.8494ms`.
The next cache-quality follow-up tightened that same project-file path again.
Adjacent line scoring now only joins true wrapped-continuation lines, so a
matching line no longer drags an unrelated previous line into the packet
summary. Project-file fallback packets are versioned (`version=2`), exact
context cache hits require the current version, and explicit recall ignores old
unversioned project-file fallback rows before deciding whether a cached packet
can satisfy the query. After reinstall/restart on PID `21404`, AXI recall for
`startup matrix 20260527 tiecheck diamond` rebuilt current Engram project-file
evidence in `1249.1747ms` with `project_file_recall_fallback`; AXI context for
`native PyO3 dogfood performance continuation cleanline 20260527` rebuilt clean
handoff evidence in `931.882ms`. Repeats then hit cache: AXI context
`0.2474ms`, AXI recall `cache_satisfied` in `74.5458ms`, MCP `get_context`
`0.1637ms`, and MCP `recall` `cache_satisfied` in `2.4684ms`. Focused
context/recall tests now pass with `75 passed`, ruff passed, skip-slow
validation reports `11 pass, 1 warn, 0 fail, 2 skip`, and the refreshed
lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260527-144207`
with `11 pass, 2 warn, 0 fail, 0 skip`. Post-matrix runtime is healthy on
LaunchAgent PID `23794`; the remaining warnings are still only stale/root
SessionStart hook evidence awaiting fresh real Codex and Claude Code sessions.
The following live packet-quality pass fixed the remaining mid-sentence and
mid-word summary rough edges in that fallback. Wrapped lowercase continuation
lines now join to their previous line when the previous line does not end a
sentence, and reusable packet snippets truncate on word boundaries with `...`
instead of cutting tokens. After reinstall/restart, fresh AXI context for
`startup matrix 20260527 tiecheck diamond project_file_recall_fallback
continuationproof2` rebuilt the current handoff packet in `785.6527ms` with the
summary starting at the `startup matrix ...` line and ending at
`native PyO3...` instead of `native PyO3 dogfood p`. Repeats hit the fast
path: AXI context `0.1051ms`, AXI recall `cache_satisfied` in `52.3059ms`, MCP
`get_context` `0.0807ms`, and MCP `recall` `cache_satisfied` in `1.1909ms`.
Focused context/recall surface tests now pass with `77 passed`, ruff passed,
and `git diff --check` is clean. The refreshed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-145536` with
`11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on LaunchAgent
PID `37314`, and the remaining warnings are still only stale/root SessionStart
hook evidence.
The next continuation-window follow-up made the evidence lines match the same
quality bar as summaries. Project-file matching now ignores lines with no
direct query-term hit, walks a bounded chain of previous continuation lines, and
trims unrelated prior sentences before using a wrapped previous line. After
reinstall/restart, fresh AXI context for
`evidence project_file_recall_fallback wrappedwindow liveproof 20260527
chainfixed2` rebuilt the current handoff packet in `747.8033ms`; MCP
`get_context` returned cached evidence lines starting with the full
`startup matrix ...` line and `After reinstall...`, with no `evidence in...` or
`can satisfy...` starts. Repeats hit cache: AXI context `0.179ms`, AXI recall
`cache_satisfied` in `3.8688ms`, MCP `get_context` `1.5832ms`, and MCP
`recall` `cache_satisfied` in `1.0518ms`. Focused context/recall surface tests
now pass with `80 passed`, ruff passed, and `git diff --check` is clean. The
refreshed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-151148` with
`11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on LaunchAgent
PID `48671`, and the remaining warnings are still only stale/root SessionStart
hook evidence.
The next latency follow-up tightened repeat-cache behavior for topic-specific
fallback packets. Context, explicit recall, and auto-recall now read resident
packet caches with `sync_persistent=False` on hot paths; topic-specific context
prebuilds project-file fallback packets while the bounded loaded-store preflight
runs; and exact project-file fallback packets can satisfy repeated context and
recall calls even when their generic file summaries do not contain the query's
unusual terms. After reinstall/restart, the synthetic miss query
`xafnorb quexilate zumbrel frobnicate mintcase exactcache5` rebuilt project-file
packets in AXI context `618.2917ms` (`cacheRelevanceMiss=2.5779ms`,
`projectFileFallback=564.7988ms`), then repeated through packet cache in
`0.047ms`. AXI recall for the same topic was `cache_satisfied` in `0.7253ms`
and `0.585ms` on two runs. Focused context/recall tests now pass with
`81 passed`, ruff passed, and `git diff --check` is clean. Skip-slow startup
validation reports `11 pass, 1 warn, 0 fail, 2 skip`; the refreshed lifecycle
matrix produced `/private/tmp/engram-dogfood-startup-20260527-153357` with
`11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on LaunchAgent
PID `57186`, and the only remaining warnings are still stale/root SessionStart
hook evidence. On that post-matrix runtime, the same synthetic topic rebuilt
project-file context in `41.1342ms`, then repeated from cache in `0.6055ms`, and
AXI recall was `cache_satisfied` in `0.5183ms`.
The next usefulness follow-up tightened packet relevance so context caches reject
lone date/id matches and explicit recall ignores generated `why_now` text before
deciding a packet satisfies a query. After reinstall/restart, weak synthetic
query `qvanta noexisting loadedstore miss tail 20260527 probeB` no longer
returned the stale loaded-store dogfood packets: AXI context reported
`cache_relevance_miss` and synthesized project-file fallback packets in
`44.7505ms` (`projectFileFallback=42.4623ms`). AXI recall for that same topic was
`cache_satisfied` in `0.6213ms` from the exact project-file fallback cache, not a
loaded-store false positive. A fresh recall-first probe
`qvanta noexisting loadedstore miss tail 20260527 probeC` ran bounded recall in
`228.2368ms`, found no memory results, and returned three project-file packets
with `fallbackStatus=context_packet_fallback`. Focused context/recall tests now
pass with `83 passed`, ruff passed, and `git diff --check` is clean. Live value
after the probe set reports `0%` budget misses, `0%` degradation, `75%`
read-path cache hit rate, and p95 `223.127ms` over five read-path samples.
Skip-slow startup validation reports `11 pass, 1 warn, 0 fail, 2 skip`; the
refreshed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-154316` with
`11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on LaunchAgent
PID `59654`, and the remaining warnings are still stale/root SessionStart hook
evidence.
The next write-path latency follow-up split MCP write side-effect budgets and
made write-tool auto-recall cache-only. `observe`/`remember` still cache fresh
session packets and can surface already-warm context, but a cache miss no longer
runs a medium recall probe on the write response path. Before this pass, live
value showed `mcp_observe` p95 `178.0555ms`, `api_auto_observe` p95
`314.8745ms`, and stale `medium` recall timeouts. After reinstall/restart on
PID `63522`, live MCP observe probe `obsF` returned in `85.9ms` wall time with
`capture_store=11ms`, `cue_store=39ms`, `live_turn_timeout=11.6503ms`, and
`recall_middleware=0.4302ms`; live value showed write-path p95 `65.566ms` and
cache-miss `medium` auto-recall skipped in `0.0775ms`. AXI context for
`write auto recall cache-only short live-turn timeout obsF` returned loaded-store
cue packets in `30.4921ms`, and AXI recall was `cache_satisfied` from the fresh
`mcp_observe` recent packet in `0.3212ms`. Focused backend tests now pass with
`129 passed, 2 skipped`, ruff passed, and `git diff --check` is clean. Skip-slow
startup validation reports `11 pass, 1 warn, 0 fail, 2 skip`; the refreshed
lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260527-155526`
with `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
LaunchAgent PID `64398`. A post-matrix MCP observe probe `obsG` kept
`recall_middleware=0.3129ms` and `live_turn_timeout=12.5198ms`; live value after
that probe showed read-path p95 `2.444ms`, `0%` read/write degradation, and
write-path p95 still dominated by matrix `api_auto_observe` at `155.1062ms`.
The next operator-friction pass corrected the startup validator's AXI evidence
guidance. When SessionStart hook evidence is stale or records `project=/`, the
printed next action now requires a new interactive Codex or Claude Code session
from the target project and says manual `agent-followup` traces do not refresh
SessionStart evidence. A nested `codex exec` run from this repo proved real MCP
adoption by calling Engram `get_context` and returning `ENGRAM_SESSIONSTART_PROBE`,
but it did not emit a SessionStart hook row, so the validator correctly keeps
the stale/root startup warning. Focused validation tests for stale/root guidance
passed, ruff passed, and `git diff --check` is clean for the validator/test
slice. Current full startup validation still reports `13 pass, 1 warn, 0 skip`;
the warning is real and remains limited to stale/root SessionStart evidence that
requires fresh interactive Codex and Claude Code sessions. Recall evaluation
sample `ers_0147444122dc` records this nested Codex run as MCP adoption evidence
with unnecessary memory for the trivial probe.
The follow-up interactive Codex TUI probe from `/Users/konnermoshier/Engram`
then produced a real SessionStart hook row:
`timestamp=2026-05-28T00:24:34.184814Z`, `operation=hook-run`,
`origin=session-start-hook`, `project=/Users/konnermoshier/Engram`,
`durationMs=11`, `status=healthy`. The startup validator now treats both the
legacy `operation=home` and current `operation=hook-run` as startup evidence, so
the Codex stale/root warning cleared. A Claude Code print-mode probe also ran
the managed SessionStart hook before its prompt-argument error and wrote
`timestamp=2026-05-28T00:28:48.891213Z`, `operation=hook-run`,
`project=/Users/konnermoshier/Engram`, `durationMs=12`, `status=healthy`. Full
startup validation now reports all startup checks passing, including
read-only AXI hooks plus startup/follow-up evidence for both Codex and
Claude Code. The installed AXI home packet now uses the active trace client for
capture suggestions (`--source claude-code`, `--source codex`, or generic
`--source axi`) instead of hard-coding Codex. Focused AXI/startup-validation
coverage passed with `39 passed`, ruff passed, and live value reports read-path
p95 `104.0759ms`, read cache hit rate `0.8545`, and zero read budget
misses/degraded reads/timeouts.
The next resumed dogfood pass verified the restarted runtime from the installed
local tool. `engramctl stop && engramctl start` brought the LaunchAgent back on
PID `86463`; full startup validation passed with all checks green, including
doctor, live MCP catalog, Codex/Claude/OpenClaw config, and AXI hook traces. AXI
home stayed startup-safe; AXI context for
`live dogfood loaded-store context performance restart 20260528` returned five
project-file packets in `49.1589ms` (`projectFileFallback=35.2116ms`); AXI
recall for the same query returned five loaded-store episode results and three
packets in `235.963ms` with `fallbackStatus=fast_preflight_hit`; MCP
`get_context` then hit packet cache in `0.0496ms`; MCP `recall` was
`cache_satisfied` in `0.2021ms`. The confirmed lifecycle matrix produced
`/private/tmp/engram-dogfood-startup-20260527-173419` with
`13 pass, 0 warn, 0 fail, 0 skip`, leaving the runtime healthy on PID `87404`.
Post-matrix probes stayed bounded without empty timeout payloads: forced-miss
AXI recall returned a relevant historical diagnostic episode in `19.1024ms`,
fresh AXI context fallback returned useful packets in `16.7976ms`, and a broad
AXI recall hit cache in `0.4516ms`. Final live value reports read-path p95
`83.526ms`, read cache hit rate `0.6667`, and zero budget misses/degraded
reads/timeouts. The dogfood performance hardening pass is committed and pushed
at `e59be43`, with a clean worktree after push.
A real Codex continuation then recorded session-continuity sample
`esc_35ecade2bbf7` and recall-quality sample `ers_7aa46915657c` against that
clean checkpoint. The sample recovered the active dogfood performance goal and
current runtime state, with shell probes confirming HEAD `e59be43`, native PyO3
LaunchAgent PID `87404`, and no degraded recall/context path. The follow-up
`engram axi value --json` reports continuity lift `0.075`, useful packet rate
`0.6429`, memory-need precision `0.9286`, read-path p95 `87.0364ms`, read cache
hit rate `0.5714`, and zero read budget misses/degraded reads/timeouts. This is
one more real Codex dogfood sample, not enough by itself to close the long-running
goal.
The same continuation lowered agent-facing raw capture wait without changing the
global explicit-write default: MCP observe and REST auto-observe now pass a
per-write `capture_store_timeout_ms=250`. After reinstall/restart on LaunchAgent
PID `67368`, live MCP observe returned in the client window with
`capture_store=169ms`, `cue_store_timeout=251ms`, `live_turn_timeout=13.097ms`,
and `recall_middleware=1.3739ms`; live REST auto-observe deferred raw capture at
`captureStoreTimeout=252ms`. `engram axi value --json` then reported write-path
p95 `440.135ms`, read-path p95 `0.1838ms`, `0%` degradation, and `0%` budget
misses. Focused capture/validation tests passed with `166 passed, 2 skipped`,
ruff passed, and `git diff --check` is clean.
The follow-up startup-safe read warmup now has `/api/knowledge/runtime/fast`
schedule a non-blocking, in-memory project-file prefix warmup. It does not touch
the graph or write packet-cache entries, but it moves local file prefix reads out
of the first real topic-specific context call after AXI session-start. After
reinstall/restart on LaunchAgent PID `69492`, the AXI home probe for
`/Users/konnermoshier/Engram` stayed startup-safe and triggered the warmup.
Fresh AXI context for `project prefix warmup liveproof citrine 20260527`
returned five useful project-file packets with no degradation in
`durationMs=183.2692` (`projectFileFallback=137.6598`), then repeated in
`0.0474ms`; AXI recall for the same topic was `cache_satisfied` in `0.7942ms`.
Fresh MCP `get_context` for `project prefix warmup mcp liveproof beryl 20260527`
returned useful project-file packets with `duration_ms=127.3182` and
`project_file_fallback=29.3808`; MCP recall was `cache_satisfied` in
`1.1467ms`. Installed-user startup validation reports `11 pass, 1 warn, 0 fail,
2 skip`; the warning remains the honest stale/root SessionStart proof that
needs fresh Codex and Claude Code sessions. Focused runtime/context tests passed
with `139 passed`, ruff passed, and `git diff --check` is clean.
The next live pass isolated project-file fallback from the default executor
because the resumed Codex MCP context probe showed `project_file_fallback`
queueing at `806.2496ms` even though the same local packet builder profiles at
about `25ms` after prefix warmup. Project-file context fallback, recall
fallback, and runtime-fast prefix warmup now use a small dedicated executor so
native Helix/storage/background work cannot starve local-file rescue packets.
After reinstall/restart on LaunchAgent PID `71087`, AXI home triggered warmup;
fresh MCP `get_context` for `dedicated project executor mcp fallback liveproof
jasper 20260527` returned five project-file packets with no degradation in
`149.254ms` (`project_file_fallback=141.8561`), and MCP recall was
`cache_satisfied` in `0.5977ms`. Fresh AXI context for `dedicated project
executor axi fallback liveproof onyx 20260527` returned loaded-store cue packets
in `77.4707ms`, and AXI recall was `cache_satisfied` in `0.5423ms`.
`engram axi value --json` then reported read-path p95 `149.6347ms`, cache hit
rate `0.7857`, and zero budget misses, degraded operations, or timeouts.
Installed-user startup validation stayed at `11 pass, 1 warn, 0 fail, 2 skip`;
the warning is still only stale/root SessionStart proof. Focused runtime/context
tests passed with `182 passed`, ruff passed, and `git diff --check` is clean.
The next AXI follow-up made project-scoped AXI context skip loaded-store
preflight with or without an explicit topic. AXI remains the startup-safe
project packet lane; long-tail memory lookup stays available through
`engram axi recall` and MCP `get_context`. After reinstall/restart on LaunchAgent
PID `74526`, packet-cache clear plus a cold topic-specific AXI context showed no
loaded-store preflight (`cacheRelevanceMiss=0.3322ms`) but still paid one cold
prefix scan at `projectFileFallback=285.6067ms`; the next fresh topic returned
in `57.0956ms` with `projectFileFallback=32.1867ms`, and exact repeat hit cache
in `0.0461ms`. Startup validation reports `13 pass, 1 warn`; the only warning
remains stale/root real SessionStart proof. The final focused suite passed with
`184 passed`, ruff passed, and `git diff --check` is clean.
The next fresh-context pass reduced duplicate project-file scans inside the
packet builder. Summary matching and evidence-claim extraction now share the
same topic-match pass instead of rescanning each candidate file. Local profiling
dropped the Engram fallback builder from roughly `18-22ms` to `11-16ms`. After
reinstall/restart on LaunchAgent PID `75796`, AXI home warmed the project; a
fresh AXI context for `duplicate matching lines liveproof garnet fresh2
20260527` built project-file packets in `22.6326ms`, and the exact repeat hit
cache in `0.0393ms`. MCP `get_context` for the matching liveproof topic showed
`project_file_fallback=24.8746ms`; total `duration_ms=175.392ms` is now mostly
the intentional `context_fast_preflight_timeout_ms=100` loaded-store miss budget
plus transport/presentation overhead. MCP recall hit cache in `1.0761ms`.
`engram axi value --json` reported read-path p95 `175.694ms`, cache hit rate
`0.625`, and zero budget misses, degraded operations, or timeouts. Startup
validation stayed at `13 pass, 1 warn`, with the same stale/root SessionStart
warning. The focused runtime/context suite passed with `185 passed`; ruff and
`git diff --check` passed.
The follow-up instrumentation pass split MCP loaded-store context preflight
timings into search and packet assembly on hits, and now reports
`loaded_store_context_preflight` on project-file fallback responses when the
loaded-store preflight misses. After reinstall/restart on LaunchAgent PID
`76806`, a repeated useful goal-continuation context hit packet cache in
`0.0772ms`. A fresh loaded-store miss
`cold project file fallback variance mcp probe jasper 20260527` returned useful
project-file packets without degradation in `103.96ms`, with
`loaded_store_context_preflight=99.7659ms` and
`project_file_fallback=23.6974ms`; AXI context for a comparable fresh project
topic returned in `69.46ms` with `projectFileFallback=21.6849ms`. One earlier
post-restart fresh MCP context sample showed a transient cold project-file build
at `1228.0362ms`, which now remains visible in `engram axi value` p95 rather
than being hidden. Startup validation stayed at `13 pass, 1 warn`, with the
same stale/root SessionStart warning. The focused runtime/context suite passed
with `185 passed`; ruff and `git diff --check` passed. A real Codex recall
evaluation sample `ers_5fba1e65db42` recorded that Engram surfaced relevant
goal-continuation context for this session.
The next pass added a soft wait for MCP loaded-store context preflight. Quick
loaded-store hits still win, but once project-file context is ready the first
response no longer waits the full `context_fast_preflight_timeout_ms=100` miss
budget. The new `context_fast_preflight_soft_wait_ms` default is `75ms`; late
loaded-store work continues in the background and can still populate packet
cache. After reinstall/restart on LaunchAgent PID `77735`, AXI home warmed the
project. A useful goal-continuation MCP context still returned loaded-store cue
packets in `85.4594ms`
(`loaded_store_context_search=52.4431ms`,
`loaded_store_context_packet_assembly=0.0627ms`). A fresh miss
`softwait miss qxjv norel zaffron plinket 20260527` returned useful project
packets in `25.7279ms`, with `project_file_fallback=23.1409ms` and only
`loaded_store_context_preflight=17.2334ms`; MCP recall for the same query hit
cache in `0.8398ms`. `engram axi value --json` reported read-path p95
`85.4594ms`, cache hit rate `0.7143`, and zero budget misses, degraded reads,
or timeouts. Startup validation stayed at `13 pass, 1 warn`, with the same
stale/root SessionStart warning. The focused runtime/context suite passed with
`186 passed`; ruff and `git diff --check` passed.
The next cache-rescue pass targeted the remaining first-post-restart variance.
Before the final tweak, a fresh MCP miss after reinstall/restart still spent
`750.092ms` in cold project-file fallback. MCP now treats cached same-project,
current-version project-file packets as a rescue path while a fresh project scan
is still pending, and only that rescue path is allowed to sync persistent packet
cache; the initial strict cache lookup remains in-memory. The background scan
still refreshes the exact topic cache when it completes. After reinstall/restart
on LaunchAgent PID `80961`, the first fresh live MCP miss
`persistent rescue first post restart miss zibble norvax klym 20260527`
did not need rescue because loaded-store context won in `30.344ms`
(`loaded_store_context_search=21.6139ms`). AXI recall for the same topic hit
cache in `0.6234ms`, and AXI context hit project cache in `0.049ms`. After full
startup validation, `engram axi value --json` reports read-path p95 `80.397ms`,
read cache hit rate `0.8`, and zero budget misses/degraded reads/timeouts. Full
startup validation reported `13 pass, 1 warn, 0 skip`: doctor and live MCP
catalog passed, and the only warning remains stale/root SessionStart trace
proof. Focused runtime/context tests now pass with `558 passed, 13 skipped`;
ruff and `git diff --check` passed. Recall evaluation sample
`ers_5491a90a404a` recorded the post-restart live MCP result.

Latest live-client note: the REST-mounted HTTP MCP endpoint now serves the
advertised `/mcp` URL directly, starts FastMCP's session manager from the parent
REST lifespan, answers Claude Code's plain `GET /mcp` discovery probe, and
refcounts overlapping MCP lifespans so stateless HTTP probes cannot close shared
stores mid-initialization. `curl` initialize and escalated `claude mcp list`
both verify `engram: http://127.0.0.1:8100/mcp (HTTP) - Connected`; full
Claude prompt-run adoption has now been validated with a raw Claude Code
`--output-format stream-json` transcript. The adoption verifier infers live
client metadata from raw Claude stream records, and the raw
`/private/tmp/engram-claude-live-raw.jsonl` run passed
`engram adoption --require-live-evidence` with observed `claim_authority`,
`get_context`, `recall`, and `remember`.
Follow-up live-client note: a fresh isolated Claude Code 2.1.144 print-mode
attempt was run against a local lite REST/MCP server at `127.0.0.1:8100`, but
the current CLI session reported `Not logged in - Please run /login` before any
Engram tool calls could execute; the raw failed stream was captured at
`/private/tmp/engram-claude-live-20260519-stream.jsonl` and should be treated
as blocked evidence, not adoption success. This run also confirmed current
Claude Code requires `--verbose` with `--output-format stream-json`; the
adoption template now prints that exact capture command before the validation
commands.
The adoption verifier now preserves that kind of failed Claude stream as
parseable blocked evidence: Claude stream-json logs with no Engram tool calls
but with `authentication_failed`, "Not logged in", or failed `engram`
`mcp_servers` init records produce a failed adoption report with explicit
`live_harness_authentication_failed` / `live_harness_mcp_server_failed`
failures instead of an `invalid_calls_transcript` parse error.
`engram evaluate` now keeps those blocker fields in adoption evidence,
multi-client adoption summaries, Markdown reports, release evidence summaries,
and the dashboard Evaluate release-evidence view, so auth/MCP setup blockers
remain visible instead of collapsing into a generic failed gate.
Adoption evidence release packaging now also rejects proxy summaries that do
not prove real protocol execution: a report must carry a nonzero call count,
nonempty expected and observed before-answer tools, observed `get_context` and
`recall`, and an observed Capture tool before it can be measured.
Human-label evidence now applies the same anti-proxy rule at the sample level:
recall samples must include reviewable query text, notes, and recall-quality
label fields; session samples must include reviewable scenario text, notes, and
session-continuity label fields, so release evidence cannot pass with
source-tagged but empty sample shells. Adoption and human-label artifacts also
require parseable ISO-style captured timestamps instead of arbitrary text.
Live adoption evidence must also name a transcript `source`, and
adoption validation reports now carry `kind:
engram_adoption_validation_report` so release packaging can reject generic
hand-shaped adoption summaries. Adoption release evidence also rejects template
placeholder client/source metadata and synthetic transcript source labels.
Current blocked live-client evidence refresh: `claude mcp list` verified the
local REST/MCP server as connected, but a constrained Claude Code 2.1.144
print-mode run still exited before tool execution with `Not logged in - Please
run /login`. The raw stream is
`/private/tmp/engram-claude-adoption-20260519-stream.jsonl`, and
`uv run engram adoption --authority /private/tmp/engram-live-claim-authority.json
--calls /private/tmp/engram-claude-adoption-20260519-stream.jsonl
--require-live-evidence --require-client "Claude Code"
--report-out /private/tmp/engram-claude-adoption-20260519-report.json
--format markdown` correctly failed with 0 calls, blockers
`mcp_server_failed` and `authentication_failed`, session
`dd1cdf01-d4d1-47e2-8d48-4b1e3bccd578`, and missing `captured_at`. Adoption
validation Markdown now also prints the specific failed MCP server list
(`['engram']`) so blocked reports identify which server failed to initialize.
New authority-prep path: `engram authority` now generates the same
`claim_authority` payload locally from the configured brain before a client run.
`uv run engram authority --mode lite --sqlite-path
/private/tmp/engram-authority-cli-20260519.db --project-path
/Users/konnermoshier/Engram --user-message "I prefer Engram as the portable
memory authority across AI harnesses." --file-memory-present --out
/private/tmp/engram-authority-cli-20260519.json --format markdown` passed and
produced required before-answer tools `bootstrap_project`, `get_context`, and
`recall`, with `remember` capture. Feeding that JSON into `engram adoption
--template --client Cursor` produced the expected live transcript template and
manual Cursor/Windsurf transcript block.
Shared local CLI runtime cleanup: lifecycle snapshots and local authority
payload generation now create graph/activation/search stores through
`engram.storage.bootstrap.create_local_runtime_stores()` and close them through
`close_if_supported()`. `authority_cli.py` no longer imports private helpers
from `lifecycle_cli.py`, keeping the authority-prep path on the same public
bootstrap boundary used by other runtime entrypoints. Focused Ruff, storage,
lifecycle, authority CLI tests, `git diff --check`, and a live lite
`engram authority` Markdown smoke passed for this slice.
Follow-up shared close cleanup: `engram evaluate` live-report loading now also
uses `engram.storage.bootstrap.close_if_supported()` for graph, consolidation,
and evaluation stores instead of carrying a local close helper. This keeps the
CLI/operator cleanup path aligned with lifecycle and authority commands while
leaving the existing evaluation graph-store construction seam intact.
Focused Ruff, live-report tests, storage bootstrap tests, `git diff --check`,
and a disposable lite `engram evaluate --no-saved-samples --format json` smoke
passed for this follow-up.
Runtime-store close coverage now includes all three stores returned by
`create_local_runtime_stores()`: lifecycle, authority, and live evaluation
reports close search, activation, and graph resources through
`close_if_supported()`. The evaluation CLI now uses the shared runtime-store
factory directly, closing the full-mode Redis activation resource instead of
discarding it after creating only the graph handle.
Focused Ruff, live evaluation report tests, lifecycle CLI tests, authority CLI
tests, `git diff --check`, and disposable lite `engram evaluate`, `engram
authority`, and `engram lifecycle` smoke commands passed after the runtime-triple
close follow-up.
Projected/consolidated smoke cleanup now follows the same rule: the smoke
runner uses `close_if_supported()` from storage bootstrap and closes search,
activation, and graph resources in its `finally` block. A focused regression
keeps the runtime-store triple visible so full-mode smoke runs do not leak the
activation resource while closing search and graph.
Focused Ruff, focused smoke tests, `git diff --check`, and a disposable lite
`engram evaluate --smoke --mode lite --replace --format json` operator run
passed, with all six evaluation signals measured and release evidence still
correctly reporting `needs_evidence`.
One-shot consolidation CLI cleanup now also uses the shared storage bootstrap
boundary: it creates the consolidation audit store through
`create_consolidation_store_for_graph()` and closes consolidation, search,
activation, and graph resources through `close_if_supported()` even when phase
validation exits early. This keeps the preferred native Helix path on a Helix
consolidation store instead of silently writing a separate local SQLite audit
file for Helix runs.
Benchmark adapter cleanup follows the same ownership rule: LongMemEval closes
search, activation, and graph resources via `close_if_supported()`, while the
showcase adapter closes its search index before the graph store and then clears
temporary state. Focused benchmark adapter regressions guard both paths.

Runtime-state probes now reinforce the same adoption contract: shared REST/MCP
`get_runtime_state()` payloads include `agentAdoption.beforeAnswer`,
`requiredNextTools`, `doNotTreatEmptyAsFailure`, and concrete
`claim_authority`/`bootstrap_project` guidance when artifacts are missing,
stale, or the graph is fresh. The MCP tool docstring now says a required
`beforeAnswer` sequence must be followed before answering. This closes the
failure mode where an agent pings Engram, sees `artifactCount: 0` and
`lastObservedAt: null`, then treats Engram as optional instead of onboarding it.
The dashboard API client now types this runtime payload, and the sidebar
connection status surfaces the same state as an Onboarding/Bootstrap badge when
the runtime says empty Engram needs adoption actions.

REST/MCP recall alignment note: explicit recall responses now use a shared
top-level lifecycle contract. REST `/api/knowledge/recall` and MCP `recall()`
both include `operation: recall`, the original query, and Recall-stage metadata
(`stage`, explicit mode, result count, packet count). MCP result items now also
surface stable entity IDs plus cue episode source/timestamps/counters that REST
already exposed, so agents and the dashboard can reason about Recall results
without translating between disconnected shapes.

Latest broad backend gate: `uv run pytest -m "not requires_docker and not
requires_helix" -q` now passes with
`3409 passed, 43 skipped, 236 deselected` after the release-evidence anti-proxy
hardening for adoption reports, human-label sample text/fields, parseable
captured timestamps, adoption report kind, live adoption source metadata,
placeholder metadata rejection, synthetic source rejection, runtime-adoption
dashboard bridge, REST empty-runtime guidance coverage, release-evidence
dashboard surface, REST/MCP recall lifecycle response alignment, GraphManager
private static helper extraction from replay/infer/apply paths, the MCP
auto-recall Capture-helper boundary, blocked Claude stream-json adoption
classification, the `agentAdoption.beforeAnswer` runtime-state contract, and
the evidence-bundle provenance/status semantics.
The previous broad run first
exposed two static contract drifts: FastMCP's root mount needed manifest
classification by its nested advertised `/mcp` route, and REST `auto_observe`
had JSON parsing/skip branching back in the route. Both remain covered.

Latest dashboard gate: `pnpm test -- --run` passes with `218 passed, 1 skipped`
and `pnpm build` passes with the existing Vite large-chunk warning after the
runtime-adoption, release-evidence dashboard bridge, and typed recall response
contract. The focused release-evidence slice also passed
`pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx`
with `55 passed` plus `pnpm exec tsc --noEmit`; the focused native dashboard
smoke for the recall response type passed with `1 passed, 1 skipped`.
Final closeout also ran `pnpm test -- --run src/test/LifecyclePanel.test.tsx
src/test/nativeDashboardSmoke.test.tsx`, which passed with 4 passed and 1
opt-in live-native smoke skip, followed by another successful `pnpm build` with
the same large-chunk warning.

Latest native PyO3 operator gate: `uv run engram evaluate --smoke --mode helix
--helix-data-dir /private/tmp/engram-native-goal-20260519-data --sqlite-path
/private/tmp/engram-native-goal-20260519-labels.db --replace --format json`
passed on the no-Docker native path, initializing `helix_native`, projecting 3
episodes, completing 1 triage consolidation cycle, reporting zero coverage
gaps, and measuring all 6 evaluation signals. Reopening the same native graph
with `uv run engram evaluate --mode helix --helix-data-dir
/private/tmp/engram-native-goal-20260519-data --sqlite-path
/private/tmp/engram-native-goal-20260519-labels.db
--require-evaluation-signals --format json` also passed, proving the operator
gate reloads the persisted native graph and saved evaluation labels. `uv run
engram lifecycle --mode helix --helix-data-dir
/private/tmp/engram-native-goal-20260519-data --sqlite-path
/private/tmp/engram-native-goal-20260519-labels.db --format json` passed on
the same store with 3 episodes, 3 cues, 3 projected memories, 1 completed
cycle, and ready stage statuses. `uv run engram doctor --mode helix
--helix-data-dir
/private/tmp/engram-native-goal-20260519-data --skip-server --format json`
passed with a ready Capture -> Cue -> Project -> Recall -> Consolidate
lifecycle snapshot and a disposable helix smoke with 6/6 evaluation signals
measured. The native evaluation report still correctly reports
`release_evidence: needs_evidence` until real human-label and live-adoption
artifacts are attached; that is a release-hardening signal, not a blocker for
the core PyO3-native brain-loop goal.
The populated native parity test now also asserts REST and MCP
`get_runtime_state` include ready `agentAdoption` guidance for the bootstrapped
PyO3 brain, proving the adoption contract is present on the preferred no-Docker
full backend path.
Final closeout reran the native import, native surface manifest, GraphManager
facade boundary, group-scope, storage bootstrap, brain-loop report, native
REST/MCP parity, native dashboard WebSocket, and explicit consolidation phase
tests. Results: `helix_native import ok`, 142 focused backend tests passed, 4
REST/MCP observe/remember/recall tests passed with 1 existing skip, 2 MCP
lifecycle tests passed, 2 populated native parity tests passed, and the
consolidation phase suite passed with 224 passed and 8 skipped.

SQLite/lite shutdown stability note: the live adoption run exposed a nonfatal
shutdown-consolidation `cannot commit transaction - SQL statements in progress`
warning during the dream phase. The root cause was SQLite
`update_relationship_weight()` committing after `UPDATE ... RETURNING` while
reciprocal/duplicate relationship matches could leave unconsumed returned rows.
That helper now consumes all returned rows before commit, with regression
coverage in `tests/test_consolidation_graph_methods.py`.

GraphManager facade cleanup note: production runtime modules no longer call
private `GraphManager._*` static helpers. Extraction summary merging now calls
`is_meta_summary()` directly, consolidation replay uses
`merge_entity_attributes()` and `apply_relationship_fact()`, and edge inference
uses `apply_relationship_fact()` directly. The facade-boundary test now guards
against future `GraphManager._*` private static calls from `server/engram/**`
outside `graph_manager.py`, alongside the existing `manager._*` access scan.

MCP auto-recall capture boundary note: the auto-recall middleware's
`auto_observe=True` side effect now records piggybacked read-tool turns through
the shared Capture-stage `store_observation()` helper instead of calling
`manager.store_episode()` directly from Recall policy code. Focused
auto-recall tests now guard that the helper remains routed through the Capture
boundary while preserving the existing `source="tool_piggyback"` behavior.

Auto-capture compatibility note: REST `/api/knowledge/auto-observe` now parses
request JSON behind the route-facing Capture surface helper instead of letting
FastAPI reject hook-shaped payloads before Engram can classify them or putting
that branching back in the public route. Installed hook payloads with explicit
`content` still use the same Capture -> Cue path, while raw Claude
`UserPromptSubmit` payloads, stream-json `message` records, and
`last_assistant_message` stop records normalize into role/project-tagged
captures. Malformed or unsupported async hook traffic returns a skipped capture
contract instead of a pre-route `422`. The adoption verifier also recognizes
REST `/api/knowledge/auto-observe` hook traces as `observe`-equivalent Capture
evidence for cheap-capture protocols, while still requiring explicit
`remember` evidence when `claim_authority()` classifies the message as
high-signal. Fresh `engram hooks` installs now generate first-party
AutoCapture hook scripts when missing, preserve existing hook scripts, and write
successful REST auto-observe capture records to
`~/.engram/adoption-trace.jsonl` (or `ENGRAM_ADOPTION_TRACE_FILE`) so operators
have machine-readable capture evidence for the adoption verifier. The verifier
now accepts multiple `--calls` inputs, so a Claude stream-json log can be merged
with the hook-generated adoption trace in one validation run. When live evidence
is required, merged transcripts with conflicting session/thread IDs now fail, so
a stale cumulative adoption trace cannot satisfy a current client run. Operators
can pass `--session-id <client-session-id>` during validation to filter a
cumulative hook trace to the current client session instead of manually trimming
JSONL. `tests/test_mcp_adoption_cli.py` now also executes the generated
`capture-prompt.sh` hook against a local auto-observe endpoint and validates the
actual trace file with a matching Claude stream-json transcript, so hook
generation, REST capture evidence, multi-file transcript merging, and
session-filtered adoption validation are covered together. The interactive
`engram hooks` output now prints the matching live-evidence verifier command,
including the default hook trace path and `--session-id` filter. The
`engram adoption --template` JSON/Markdown output now carries the same
validation-command guidance: one command for a single live wrapper transcript
and one for Claude stream-json plus the AutoCapture trace; when `--client` is
provided, both generated commands include `--require-client` and
`--report-out adoption-report.json`. The adoption verifier also supports
`--require-client <label>` directly so cross-harness gates can assert that a
Cursor/Windsurf run is not accidentally satisfied by Claude Code evidence. With
`--require-live-evidence`, `--session-id` now also requires session evidence; a
sessionless wrapper transcript cannot satisfy a session-filtered live gate.
`engram adoption --report-out adoption-report.json` writes the JSON validation
artifact directly for release packaging while still printing the requested
JSON/Markdown output. The release handoff's human-label command now includes
`--human-label-template-out human-label-template.json`, and `engram evaluate`
writes that fillable JSON template while still printing JSON/Markdown.
The live-adoption template now also emits a copyable
`manual_transcript_markdown` block with client/captured/session/source metadata
plus `Before answer` and `Capture` tool-call sections. That gives Cursor,
Windsurf, and copied MCP UI logs the same first-class collection path as Claude
stream-json without requiring raw JSON export from the client.
Adoption validation reports now include a
`release_evidence` handoff section with the prefilled human-label template
command and final `engram evaluate --require-release-evidence` command so the
validated live-client artifact points directly at release packaging.

Evaluation release-gate note: `engram evaluate` now has a separate
human-labeled harness evidence gate via `--human-label-artifact` and
`--require-human-label-evidence`, plus `--human-label-template` to print the
schema, starter examples, and validation command operators should fill from a
real harness run. With `--adoption-report adoption-report.json`, the template
pre-fills the client, capture timestamp, and session metadata that the release
gate later cross-checks, but template prefill now rejects failed or non-live-
gated adoption reports before labels are collected. Template prefill also
validates and preserves `--additional-adoption-report` plus
`--require-adoption-clients` when release labels are being collected for a
multi-client package; supplemental reports require a primary `--adoption-report`
so label metadata stays tied to one live harness run. The artifact is attached
to JSON/Markdown reports and evidence bundles, but it explicitly rejects
untouched placeholder templates and
smoke, benchmark, showcase, fixture, deterministic, simulated, or synthetic
sources. Loaded artifacts now carry their SHA-256 digest in the evidence summary
and Markdown report, so archived bundles point back to the exact reviewed
`human-labels.json`. Human-label evidence now also requires the
`engram_human_label_evidence` artifact kind and requires recall/session samples
to carry real per-sample `source` values that match the artifact-level source,
so release labels cannot pass with generic, untraceable, or mixed-source
samples.
Evidence bundles now also include a top-level
`source_sha256` map for report, benchmark, human-label, adoption, and sample
files, plus package/git provenance (`engram_version`, commit, branch, dirty
state, and short status), making release artifacts directly traceable to their
input files and to the code revision that produced the bundle. Bundles now also
distinguish `status: recorded` / `gate_profile: record_only` for ungated
archives from `status: passed` gate runs, and set `release_ready: true` only
when `--require-release-evidence` actually passed.
`engram evaluate` can also attach and require a passed
`engram adoption --format json` report via `--adoption-report` and
`--require-adoption-evidence`; when both adoption and human-label evidence are
present, client/session metadata must point at the same live harness run.
Release packaging can now add `--require-adoption-client <client>` so a
Cursor/Windsurf package cannot accidentally pass with a Claude report; the
attached adoption report must have been validated with the matching
`engram adoption --require-client <client>` gate. It can also attach repeated
`--additional-adoption-report <path>` artifacts plus
`--require-adoption-clients <client...>` to prove broader live-client diversity
across Cursor/Windsurf/second-client release packages. When release/adoption
evidence is required, attached additional reports must also be measured and
unblocked. The adoption report must have been validated with
`--require-live-evidence`; a passing transcript without that live gate now fails
release/adoption evidence, and the adoption report's own `release_evidence`
handoff now stays blocked until that validation flag was used.
`--require-release-evidence` now enforces measured evaluation signals, real
human-label evidence, and passed adoption evidence as one operator gate. It
also defaults human-label thresholds to 10 recall samples and 3 session samples
unless explicitly overridden, while standalone human-label gates keep the 1/1
local-check default.
Deterministic benchmark bundles remain useful local proof; they no longer stand
in for production/staging human-reviewed harness sessions.

Dashboard release-evidence note: the Evaluate panel now preserves and renders
`human_label_evidence`, `adoption_evidence`,
`additional_adoption_evidence`, and `adoption_client_evidence` from
`GET /api/evaluation/brain-loop/report`. Operators can see human-label sample
counts, adoption calls/tool coverage, required vs observed live MCP clients,
and evidence failures in the same runtime-quality view instead of only in CLI
Markdown/JSON output. The report now also carries a computed
`release_evidence` summary so REST/MCP/CLI/dashboard all share the same
`measured`, `failed`, `needs_signals`, or `needs_evidence` readiness state
instead of each surface inferring release status from optional artifacts.

## Current Milestone

The audit milestone, P0 public-contract slices, and several P1 runtime-service
slices are complete enough to build from. See
`docs/design/brain-runtime-audit.md` for the lifecycle map, drift list, and next
build queue.

What changed in this pass:

- Added the brain runtime audit and P0/P1/P2/P3 build queue.
- Updated stale consolidation documentation from 12 phases to the current
  17-phase engine contract.
- Updated dashboard consolidation test fixtures/assertions to cover all 17
  phases.
- Marked remaining Helix-dependent tests so the SQLite/lite backend gate can
  run without a local HelixDB instance.
- Re-ran the explicit broad non-Docker/non-external-Helix backend gate after
  the latest native parity, lifecycle, consolidation phase-contract, and shared
  consolidation presenter/projection-plan/default-group/replay/projection-yield
  group-scope/static-guard/Recall-gate coverage/persistence work; it now passes
  with 3320 tests, 43 skips, and 236 external-service tests deselected after
  the latest entity-probe recall, consolidation phase-catalog, episode
  ingestion, offline replay, capture dedup, and native surface manifest
  extractions plus the GraphManager, REST/MCP memory, and consolidation
  presenter-boundary guards, native evidence-update normalization, the native
  default-group config inheritance contract, the episode-worker runtime-store,
  batching, scoring, routing, and event parsing boundaries, and the MCP
  auto-recall middleware execution boundary, the MCP Capture write
  orchestration boundary, the MCP explicit recall tool surface boundary, and
  the MCP entity/fact lookup, artifact-search, context, and question-route tool
  middleware boundaries plus the knowledge-chat tool-use loop and response-turn
  orchestration extractions plus the REST evaluation-report runtime-boundary,
  evaluation-signal CLI hard gate, doctor evaluation-signal readiness reporting,
  Python 3.13 event-loop test harness cleanup, the date-stable Helix dashboard
  analytics unit fixture, direct REST engine-dispatch/store-service guard,
  stricter native manifest evidence verifier, the chat rate-limit execution
  helper, the chat persistence scheduler helper, shared companion-store
  bootstrap, explicit notification/scheduler dependencies, and the public
  smoke cue-feedback facade, plus REST/MCP shutdown stop/close facade cleanup,
  shutdown consolidation helper, REST chat SSE runtime extraction, REST chat
  response-surface extraction, MCP memory authority/onboarding prompt contract,
  dashboard WebSocket auth route-boundary extraction, REST health route-boundary
  extraction, REST consolidation trigger scheduling extraction, static guards,
  REST consolidation status pressure/config extraction, adoption transcript
  stdin validation, self-reported file-memory bypass classification, the copied
  Claude file-memory bypass transcript regression, the public REST route
  control-flow guard, the atlas warning/response helper extraction, the
  whole-runtime private GraphManager access guard, and the
  `--min-evaluation-signal-evidence` evaluation gate.
  The GraphManager guard now
  covers the remaining service-backed compatibility adapters too, and the MCP
  identity-core mutation, MCP consolidation trigger, and MCP entity graph
  resources now route through service-backed manager facades. REST/MCP
  recall-need graph-probe helpers now use a manager facade instead of
  constructing probes from private stores in the transport layer, and
  REST/MCP intention-list presentation now lives in the prospective-memory
  service. MCP `intend` now reads its effective default threshold through that
  same service boundary. REST/MCP live conversation context helpers now route
  through `ConversationRuntimeService`-backed manager facades instead of
  reading `manager._conv_context` or `manager._search` in the transport layer.
  REST chat/recall/remember public-policy decisions now route through
  `PublicSurfacePolicyService`, and MCP lifecycle summary reads now use manager
  facades for activation config and lifecycle graph-store access. MCP
  recall-response enrichment now uses manager facades for triggered intentions,
  near misses, recall item access counts, and surprise connections. REST
  dashboard stats now uses a graph-state service facade instead of route-local
  graph-store reads for top-connected and growth timeline payloads. REST
  activation monitor snapshot and curve reads now use the same graph-state
  service boundary instead of route-local app-state store reads. REST episode
  dashboard listing now also uses `GraphStateService` through
  `GraphManager.list_episode_summaries()` instead of formatting paginated
  episode/cue payloads directly from the route. REST and MCP lifecycle summary
  reads now call `GraphManager.get_lifecycle_summary()` backed by
  `LifecycleSummaryService`, so transport layers no longer assemble the
  brain-loop summary directly. Dashboard WebSocket activation-monitor snapshots
  now call a route-facing graph-state helper instead of calling the manager
  directly or recomputing activation from app-state graph/activation stores in
  the socket loop. REST
  notification reads/dismissal and MCP `memory_notifications` piggybacking now
  share `NotificationSurfaceService` instead of formatting or surfacing
  notifications directly from route/tool code, and the dashboard WebSocket
  `dismiss_notification` command now uses that same service boundary instead of
  reading the app notification store directly. The dashboard WebSocket auth
  setup now reads config through `get_config()` as well, so
  `server/engram/api/websocket.py` no longer imports `_app_state` directly. REST
  knowledge-chat rate limiting now reads the optional rate limiter through
  `get_rate_limiter()` and passes execution through
  `check_api_chat_rate_limit()`, so `server/engram/api/knowledge.py` is
  app-state-free and does not call `rate_limiter.*` directly. REST `/health`
  now uses `get_graph_store()`, `get_config()`, and
  `get_mode()`, and `build_api_health_surface()`, so the health route is also
  free of direct app-state reads and direct graph-store probe/status assembly. The
  public-surface guard now generates coverage for every API route module except
  `api/deps.py`, making direct route-local `_app_state` reads a guarded
  regression. It also guards direct manager dispatch across REST API routes and
  `server/engram/mcp/server.py`; the only allowed direct manager method call
  there is MCP shutdown closing runtime resources. REST API routes are also
  statically guarded against direct `engine.*` dispatch and direct route-local
  store/service method dispatch now, REST route functions are guarded to only
  directly await route-facing helpers, and decorated MCP public surfaces are
  guarded against direct store/session method dispatch and arbitrary direct
  awaited runtime calls.
- Added `server/engram/quality/native_surface_manifest.py` plus static
  coverage tests so every current public REST route and MCP tool/resource/
  prompt is classified against PyO3-native parity evidence. The verifier now
  requires runtime evidence to name an actual native parity helper/test and
  requires dashboard/operator/static evidence to point at an existing repo
  artifact. This makes the native path the auditable default for new public
  surfaces instead of a best-effort follow-up.
- Added `docs/design/brain-runtime-completion-audit.md` as the current
  goal-readiness checkpoint. It maps each explicit objective requirement to
  concrete repo evidence, keeps the verdict at close pending final closeout
  audit, and separates core-goal blockers from later release-hardening evidence.
- Added doctor evaluation-signal readiness output so `engram doctor` now carries
  the smoke report's six-signal readiness summary in JSON metadata and Markdown,
  alongside the existing Capture -> Cue -> Project -> Recall -> Consolidate
  smoke totals and coverage gaps.
- Updated the native surface manifest and completion audit to record that the
  Helix native doctor path reports evaluation-signal readiness, so future
  operator-readiness reviews do not treat this as a missing PyO3 path.
- Centralized REST companion-store creation plus shared CLI/MCP evaluation
  stores in `server/engram/storage/bootstrap.py`. REST startup now creates
  atlas, consolidation, evaluation, and conversation stores through that module,
  while MCP startup, lifecycle CLI, evaluation CLI, and projected/consolidated
  smoke share the same consolidation/evaluation lite borrowed-DB and Helix
  shared-client paths.
- Moved the remaining route-facing borrowed SQLite consolidation fallback into
  `server/engram/storage/bootstrap.py`. MCP consolidation trigger fallback,
  lifecycle summary fallback, and graph-health metrics now call bootstrap
  helpers instead of probing `_db` or constructing `SQLiteConsolidationStore`
  locally.
- Removed another hidden app-state path from notification presentation and
  consolidation scheduling. MCP `memory_notifications` now calls the pure
  `build_mcp_notifications_surface()` helper with an explicit
  `NotificationSurfaceService` from `api/deps.py`; `notifications.surface`
  no longer imports application state. `ConsolidationScheduler` receives the
  active graph store from REST startup and passes it to the temporal scanner
  instead of importing app state from `engram.main`.
- Updated the projected/consolidated smoke cue-feedback check to use
  `GraphManager.apply_memory_interaction()` and a public cue result lookup
  instead of recording cue hits through manager private methods.
- Added `server/tests/test_graph_manager_facade_boundaries.py` so the core
  lifecycle entrypoints on `GraphManager` are statically guarded as service
  delegates. The guard currently covers store, project, one-shot ingestion,
  bootstrap/runtime, epistemic route/evidence, recall, entity-probe recall,
  context, and graph-state facades, plus evidence adjudication, artifact search,
  decision materialization, lookup, forgetting, prospective memory, context,
  graph-state, and related access/interaction compatibility adapters. It now
  also scans runtime modules for direct `manager._*`, `graph_manager._*`, or
  `_manager._*` access outside `server/engram/graph_manager.py`.
- Extracted MCP identity-core mutation out of route-local graph writes.
  `server/engram/retrieval/identity_core.py` owns mark/unmark behavior,
  `GraphManager.mark_identity_core()` is the compatibility facade, and
  `mark_identity_core` in MCP now calls the facade instead of reaching into
  `manager._graph`.
- Extracted MCP consolidation trigger orchestration out of route-local manager
  private state. `server/engram/consolidation_trigger.py` owns ad hoc
  `ConsolidationEngine` construction/execution and graph-stats capture,
  `GraphManager.trigger_consolidation_cycle()` is the compatibility facade,
  and MCP `trigger_consolidation` now formats the returned cycle instead of
  building an engine from `manager._graph`, `_activation`, `_search`, `_cfg`,
  and `_extractor`.
- Moved MCP identity-core and consolidation-control response assembly behind
  route-facing helpers. `server/engram/retrieval/identity_core.py` now owns MCP
  identity-core dispatch, and `server/engram/consolidation_trigger.py` now owns
  MCP trigger dispatch, active-store/shared-DB fallback resolution,
  consolidation status reads, and cycle-summary shaping while the MCP transport
  keeps JSON wrapping and session-state store references.
- Extracted MCP entity graph resources out of route-local graph/activation
  reads. `GraphStateService` now owns entity profile and one-hop neighbor
  resource views, `GraphManager.get_entity_profile()` and
  `get_entity_neighbors()` are compatibility facades, and MCP resources now
  call those facades instead of formatting from `manager._graph` and
  `_activation`.
- Moved REST/MCP recall-need graph-probe helpers behind
  `GraphManager.get_recall_need_graph_probe()`. The transport helpers now ask
  the runtime for the graph probe instead of constructing `GraphProbe` from
  `manager._graph` and `_activation`.
- Moved REST/MCP intention-list presentation behind
  `ProspectiveMemoryService.list_intention_views()`. API and MCP list handlers
  now preserve their existing response shapes by calling
  `GraphManager.list_intention_views()` instead of reading `manager._cfg` and
  `_activation` to compute warmth in the transport layer.
- Moved MCP `intend` default-threshold reporting behind
  `ProspectiveMemoryService.effective_activation_threshold()`.
  `GraphManager.effective_intention_threshold()` is the compatibility facade,
  so MCP no longer reads `manager._cfg.prospective_activation_threshold`.
- Moved REST/MCP live conversation context access behind
  `ConversationRuntimeService`. `GraphManager` now exposes compatibility
  facades for active conversation context, live-turn embedding, turn counts,
  session entity names, recent turns, and live-turn ingestion; REST chat/route
  helpers and MCP auto-recall/route helpers use those facades instead of
  reading `manager._conv_context` or `manager._search`.
- Moved public REST route policy behind
  `PublicSurfacePolicyService`. `GraphManager` now exposes compatibility
  facades for activation config, memory-need graph-probe enablement,
  client-adjudication response visibility, explicit recall packet limits, chat
  tool recall semantics, post-response recall feedback, chat retry safety-net
  enablement, and chat runtime policy. REST remember/recall/chat helpers use
  those facades instead of reading `manager._cfg`.
- Moved MCP lifecycle summary graph/config access behind manager facades.
  `GraphStateService.get_graph_store()` backs
  `GraphManager.get_lifecycle_graph_store()`, and MCP lifecycle summary now
  passes `get_lifecycle_graph_store()` plus `get_activation_config()` into the
  shared lifecycle builder instead of reading private manager fields.
- Moved REST entity detail and mutations behind manager service facades.
  `GraphStateService.get_entity_detail()` owns the `/api/entities/{id}` detail
  view with activation and facts, while `EntityMutationService` owns PATCH
  profile updates and DELETE soft-delete plus activation clearing. The
  `/api/entities` route now preserves its existing JSON shapes without reading
  `manager._graph`, `_activation`, or `_cfg`.
- Moved REST entity detail/mutation response assembly behind
  `server/engram/retrieval/entity_surface.py`. The route no longer owns
  get/update/delete manager dispatch, sparse update payload construction,
  404 status mapping, or the shared entity-not-found response payload.
- Moved REST admin benchmark loading behind `BenchmarkLoadService`.
  `GraphManager.load_benchmark_corpus()` now owns the generated corpus group
  scoping and active-store load path, so `/api/admin/load-benchmark` no longer
  reaches into `manager._graph`, `_activation`, or `_search`. The route now
  uses `build_api_benchmark_load_surface()` instead of calling the manager
  method directly.
- Moved REST graph neighborhood and temporal graph reads behind
  `GraphStateService`. `GraphManager.get_graph_neighborhood()` and
  `get_temporal_graph()` now own the dashboard graph payload construction, so
  `/api/graph/neighborhood` and `/api/graph/at` no longer read
  `manager._graph`, `_activation`, or `_cfg` while preserving the existing JSON
  shapes. Route-facing helpers in `server/engram/retrieval/graph_state.py` now
  also own REST missing-entity and invalid-timestamp payloads for those routes.
- Moved REST atlas snapshot/history/region response assembly behind
  `server/engram/retrieval/atlas_surface.py`. `/api/graph/atlas`,
  `/api/graph/atlas/history`, and `/api/graph/regions/{region_id}` now keep
  tenant lookup, atlas-service dependency lookup, logging, and JSON wrapping in
  the route while the helper owns representation metadata, snapshot/history row
  shaping, region/snapshot 404 payloads, and service dispatch.
- Moved MCP recall-response enrichment behind `RecallResponseStateService`.
  `GraphManager` now exposes facades for draining triggered-intention views,
  latest near misses, recall item access counts, and surprise-connection views,
  so MCP `recall` and piggyback middleware no longer read `manager._activation`,
  `_last_near_misses`, `_surprise_cache`, or `_triggered_intentions`.
- Moved REST dashboard stats behind `GraphStateService`.
  `GraphManager.get_dashboard_stats()` now owns `/api/stats` top-activated
  formatting plus top-connected and growth timeline reads, so the stats route
  no longer reaches into the graph store directly. The stats route now also
  uses a route-facing graph-state helper instead of calling the manager method
  directly.
- Moved REST activation monitor reads behind `GraphStateService`.
  `GraphManager.get_activation_snapshot()` and `get_activation_curve()` now own
  `/api/activation/snapshot` and `/api/activation/{entity_id}/curve`, so the
  activation route no longer reads active graph, activation, or config stores
  directly from app state. The curve route now also uses the graph-state
  route-facing response helper for missing-entity 404 payloads instead of
  raising a route-local `HTTPException`, and the snapshot route uses a
  route-facing graph-state helper instead of calling the manager method
  directly.
- Moved REST episode dashboard reads behind `GraphStateService`.
  `GraphManager.list_episode_summaries()` now owns `/api/episodes` paginated
  episode/cue payload construction, so the route no longer reads the graph store
  or formats episode/cue state locally while preserving status/source filters,
  cursor pagination, projection fields, cue counters, and timestamp formatting.
  The episode route now also uses a route-facing graph-state helper instead of
  calling the manager method directly.
- Moved REST/MCP lifecycle summary reads behind `LifecycleSummaryService`.
  `GraphManager.get_lifecycle_summary()` now owns the shared
  `Capture -> Cue -> Project -> Recall -> Consolidate` summary call for
  `/api/lifecycle/summary` and MCP `get_lifecycle_summary`, so both transports
  preserve the shared payload without directly calling `build_lifecycle_summary`
  or passing graph/config facades themselves.
- Moved MCP lifecycle-summary audit-store wiring behind
  `server/engram/lifecycle_summary.py`. MCP `get_lifecycle_summary` now delegates
  active audit-store reader construction, inactive-engine placeholder wiring, and
  limit clamping to a route-facing helper while keeping JSON wrapping in the
  tool handler.
- Moved the dashboard WebSocket activation-monitor snapshot behind the existing
  graph-state activation surface helper. `subscribe.activation_monitor` now
  sends `build_api_activation_snapshot_surface()` payloads instead of calling
  the manager directly, reading `_app_state` graph/activation/config stores, or
  running activation math inside the socket loop.
- Moved REST/MCP notification surfacing behind `NotificationSurfaceService`.
  REST `/api/knowledge/notifications`, REST dismiss, and MCP
  `memory_notifications` piggybacking now share one group-scoped presentation
  and dismissal boundary instead of reading the app notification store and
  formatting payloads in route/tool code. The notification surface now also
  owns REST list/dismiss response envelopes and missing-service fallbacks, so
  `server/engram/api/knowledge.py` keeps only tenant lookup and JSON wrapping
  for those routes.
- Moved the dashboard WebSocket notification dismiss command behind the same
  notification surface boundary. `dismiss_notification` now calls
  `NotificationSurfaceService.dismiss_notifications()` with the connected brain
  group instead of reading `_app_state["notification_store"]` directly in the
  socket loop.
- Moved dashboard WebSocket event/command payload shaping into
  `server/engram/api/websocket_surface.py`. `forward_events`,
  `activation_snapshot_loop`, and `receive_commands` now delegate event
  flattening, `pong`, `resync`, activation snapshot envelopes, and
  `dismiss_notification` dispatch to route-facing helpers while the socket
  route keeps connection auth, subscription task lifecycle, and JSON transport.
- Moved dashboard WebSocket auth config lookup behind the existing API config
  dependency. The route now uses `get_config().auth` with the previous
  `AuthConfig()` fallback if app config is unavailable, and the public-surface
  guard asserts the socket route does not import `_app_state` directly.
- Moved REST knowledge-chat rate limiting behind the API dependency layer.
  `chat()` now calls `get_rate_limiter()` instead of importing `_app_state`,
  while `server/engram/retrieval/chat_runtime.py` owns
  `check_api_chat_rate_limit()` and the shared 429 payload. The public-surface
  guard now also asserts `server/engram/api/knowledge.py` stays free of direct
  app-state reads and direct `rate_limiter.*` dispatch.
- Moved REST health app-state access behind API dependencies. `/health` now uses
  `get_graph_store()` for the probe, `get_config().default_group_id` for the
  one-brain stats scope, and `get_mode()` for the response mode; the
  public-surface guard now asserts `server/engram/api/health.py` does not import
  `_app_state` directly.
- Moved REST health response assembly into `server/engram/api/health_surface.py`.
  The helper owns graph-store probing, default-brain stats checks, service
  status aggregation, and the public `HealthResponse`, while `/health` keeps
  dependency lookup and response return.
- Generalized the public-surface app-state guard to every API route module. The
  guard now discovers `server/engram/api/*.py` dynamically, excludes only
  `__init__.py` and `deps.py`, and fails if any route module imports
  `_app_state` directly.
- Moved the REST evaluation report's consolidation-cycle reads behind
  `build_api_brain_loop_evaluation_surface()` in
  `server/engram/evaluation/report_service.py`. The helper now loads recent
  cycles and calibration snapshots through
  `ConsolidationEngine.get_recent_evaluation_context()`, then calls the shared
  report assembly boundary with REST snapshot labeling. The route no longer
  reaches into `engine._store` or calls engine methods directly, and the
  public-surface guard now rejects route-local `engine._*` access plus direct
  REST route `engine.*` dispatch.
- Moved REST/MCP consolidation audit reads behind
  `ConsolidationAuditReader`. REST status/history/detail now call public
  `ConsolidationEngine` reader facades and the shared
  `serialize_cycle_detail()` presenter instead of reading `engine._store` or
  assembling audit-record payloads in the route. MCP evaluation report inputs,
  MCP consolidation status, and MCP lifecycle summaries use the same reader, and
  lifecycle summaries now accept an explicit `consolidation_reader` instead of a
  synthetic engine object with private store state.
- Moved REST consolidation control/read response assembly behind route-facing
  helpers in `server/engram/consolidation_trigger.py`. REST trigger conflict and
  acknowledgement payloads, background-cycle execution, status pressure/latest
  cycle shaping, history cycle lists, and cycle-detail 404/detail payloads now
  live beside the MCP consolidation-control helpers while
  `server/engram/api/consolidation.py` keeps tenant lookup, dependency lookup,
  background-task registration, and HTTP wrapping.
- Moved knowledge-chat rich tool-event shaping behind
  `server/engram/retrieval/chat_events.py`. Recall/fact-to-UI-event selection,
  chat-recall round-tripping, and AI SDK synthetic tool payload-pair
  construction now live in a retrieval-side presenter. The REST route keeps only
  SSE wrapping for those payloads.
- Moved knowledge-chat tool execution payload shaping behind
  `server/engram/retrieval/chat_tools.py`. Retrieval code now owns
  recall/search_entities/search_facts dispatch payloads, chat recall packet
  shaping, entity/fact LLM payloads, fact deduplication, unknown-tool
  responses, and the non-streaming Anthropic tool-use loop/result accumulation.
  The retrieval helper also owns the chat tool schema, Anthropic text-block
  extraction, and the JSON-string compatibility wrapper used by legacy tests.
  The REST route keeps client construction and SSE framing.
- Moved knowledge-chat recall feedback and retry policy behind
  `server/engram/retrieval/chat_feedback.py` and
  `server/engram/retrieval/chat_tools.py`. Retrieval code now owns
  used/dismissed memory interaction application, generic memory-free response
  detection, retry gating, retry system-prompt construction, and retry provider
  execution. The REST route keeps client construction and SSE framing.
- Moved knowledge-chat response-turn orchestration behind
  `server/engram/retrieval/chat_runtime.py`. The REST route now delegates chat
  memory-need analysis, memory-guidance text, live conversation hydration,
  assistant-turn recording, recent-turn extraction, chat runtime policy lookup,
  epistemic-evidence dispatch, baseline context dispatch, system-prompt
  assembly, sliding-window message assembly, tool-use loop invocation, retry
  policy application, recall feedback, and route-neutral chat stream payload
  construction to retrieval code. The REST route keeps rate-limiter dependency
  lookup, conversation helper invocation, Anthropic client construction, SSE
  wrapping, and persistence scheduler invocation.
- Moved REST/MCP explicit recall result and packet assembly behind
  `server/engram/retrieval/recall_surface.py`. REST still keeps its HTTP
  response shape, but the Recall-stage manager call, packet analysis, packet
  assembly, API/MCP recall item presentation, MCP entity-name/access-count
  resolution, MCP near-miss/surprise side-channel enrichment, MCP query timing,
  MCP recall-session flags, and MCP recall middleware invocation now share one
  retrieval-side boundary. `server/engram/mcp/server.py` keeps only manager/
  session lookup, config fallback, tool signature, and JSON wrapping for
  explicit recall.
- Moved sync/async recall-need threshold resolution and memory-need analysis
  recording behind helpers in `server/engram/retrieval/control.py`. REST, MCP,
  chat runtime, chat tool execution, and explicit recall surface code now share
  those manager-facade compatibility adapters instead of each owning private
  copies.
- Moved REST/MCP artifact-search result assembly behind helpers in
  `server/engram/retrieval/artifacts.py`. REST keeps `projectPath`, MCP keeps
  `project_path` plus recall middleware, but both surfaces now share artifact
  hit loading and item serialization.
- Moved REST/MCP deterministic question-route assembly behind
  `server/engram/retrieval/epistemic_route.py`. REST and MCP now share route
  history normalization and the manager `route_question` call while preserving
  their transport-specific response wrapping and MCP recall middleware.
- Moved REST/MCP prospective-memory intention surfaces behind helpers in
  `server/engram/retrieval/prospective.py`. REST and MCP now share intention
  create/list/dismiss manager calls and acknowledgement shaping while preserving
  HTTP/MCP JSON wrapping. REST intention validation/not-found payload bodies,
  REST create/dismiss status mapping, and MCP create/dismiss error payloads now
  live in the same helper module instead of `server/engram/api/knowledge.py` or
  `server/engram/mcp/server.py`.
- Moved REST/MCP forget entity/fact surfaces behind helpers in
  `server/engram/retrieval/forgetting.py`. REST keeps its entity-first behavior
  when both targets are supplied; MCP keeps its exactly-one-target validation,
  but both surfaces share target dispatch and fact-field normalization. REST
  missing-target and error-status response mapping now also lives in the same
  route-facing helper instead of `server/engram/api/knowledge.py`.
- Moved REST/MCP explicit preference-feedback validation and manager dispatch
  behind `server/engram/retrieval/preference_feedback.py`. REST keeps 400/404
  HTTP mapping, but rating validation, `record_explicit_feedback` dispatch,
  REST error payloads, and MCP invalid-rating error payloads now share
  route-facing helpers.
- Moved REST/MCP project bootstrap and runtime-state route calls behind shared
  surface helpers. `server/engram/ingestion/project_bootstrap.py` now owns the
  transport-facing bootstrap manager call plus REST skipped-status mapping, and
  `server/engram/retrieval/runtime_state.py` owns the runtime-state manager call
  used by REST and MCP. The existing `ProjectBootstrapService` and
  `RuntimeStateService` remain the deeper lifecycle owners.
- Moved REST/MCP public entity/fact lookup surfaces behind helpers in
  `server/engram/retrieval/lookup.py`. REST entity and fact search keep their
  camelCase `items` payloads, MCP entity/fact search keeps raw lookup payloads,
  and MCP's missing-query validation plus recall-middleware invocation now share
  the lookup surface helper instead of living in `server/engram/mcp/server.py`.
- Moved REST/MCP public agent-context response assembly behind helpers in
  `server/engram/retrieval/context_builder.py`. REST keeps camelCase count/token
  keys, MCP keeps the raw `get_context` shape, retrieval-side MCP helpers own
  recall/notification middleware invocation, and `MemoryContextBuilder` remains
  the deeper tiered context assembly service.
- Moved knowledge-chat conversation persistence behind
  `server/engram/retrieval/chat_persistence.py`. The REST chat route still owns
  SSE wrapping, Anthropic client construction, and HTTP status mapping, but
  conversation validation/creation, group-scoped not-found handling and payload
  shaping, completed-turn persistence scheduling, completed-turn persistence,
  and recalled entity tagging now live in retrieval-side helpers.
- Moved REST conversation CRUD behind
  `server/engram/retrieval/conversation_persistence.py`. The conversations API
  now delegates group-scoped listing, creation, message reads/appends,
  title updates, deletes, REST response-envelope shaping, and not-found
  translation/status mapping to shared helpers instead of encoding store calls,
  payload bodies, or 404 branches in the route.
- Moved REST/MCP episode adjudication request loading behind
  `server/engram/ingestion/adjudication_surface.py`. REST and MCP remember
  surfaces now share the compatibility lookup for post-write adjudication work
  items before passing them to the shared memory-write presenters.
- Moved REST/MCP adjudication resolution response assembly behind the same
  `server/engram/ingestion/adjudication_surface.py` module. REST keeps camelCase
  response keys, MCP keeps snake_case response keys, and both now share the
  `submit_adjudication_resolution(..., source="client_adjudication")` manager
  dispatch.
- Moved REST/MCP live conversation manager-facade helpers into
  `server/engram/retrieval/context.py`. REST chat and MCP recall piggybacking
  now share the defensive sync/async/type filtering around conversation
  context, embed functions, turn counts, recent turns, session entity names,
  and live-turn ingestion.
- Moved REST/MCP brain-loop evaluation report assembly behind
  `server/engram/evaluation/report_service.py`. REST and MCP still supply their
  transport-specific stores and snapshot source labels, while the REST helper
  also owns engine-derived cycle-context loading. Graph-state reads, runtime
  Recall metric snapshot persistence/reload, saved label reads, and report
  construction now share one service boundary.
- Moved MCP evaluation report consolidation-input loading behind the same report
  service boundary. `server/engram/evaluation/report_service.py` now owns active
  MCP audit-store reads, cycle-limit clamping, and calibration snapshot loading
  before building the MCP report, so `mcp/server.py` no longer carries a local
  evaluation-input loader.
- Moved REST/MCP evaluation label writes behind
  `server/engram/evaluation/label_service.py`. REST and MCP now share the
  recall-quality and session-continuity sample construction, count clamping,
  active `group_id` persistence, and write acknowledgement presentation through
  route-facing helpers.
- Moved REST/MCP public Capture write dispatch behind
  `server/engram/ingestion/capture_surface.py`. REST and MCP observe/remember
  paths now share conversation-date parsing, attachment construction, raw
  observation storage, Capture -> Project ingest dispatch, client-adjudication
  loading, and memory-write presentation. REST auto-observe also delegates
  enablement, short-content skip, dedup skip, raw observation storage, and
  memory-write presentation to the same capture surface while REST routes keep
  config/dependency lookup and JSON wrapping. MCP keeps its transport-specific
  session accounting, live-turn ingestion, and recall middleware there too.
- Moved MCP Capture write orchestration behind the same capture surface module.
  `build_mcp_remember_write_surface()`, `build_mcp_observe_write_surface()`,
  and `build_mcp_attachment_observe_write_surface()` now own MCP session
  activity updates, live-turn recording, adjudication-request loading, memory
  write presentation, and recall middleware invocation for the public write
  tools. `server/engram/mcp/server.py` keeps manager/session lookup, JSON
  wrapping, and tool signatures.
- Moved reusable MCP auto-recall policy helpers behind
  `server/engram/retrieval/auto_recall.py`. Cooldown/topic deduplication,
  compact recall-query extraction, per-tool recall gating, first-call session
  prime planning, middleware side-effect planning, lite/medium entity-probe
  response shaping, and additive MCP response enrichment now live outside
  `server/engram/mcp/server.py`.
- Moved MCP recall middleware execution behind the same retrieval helper module.
  `run_mcp_recall_middleware()` now owns plan execution for middleware
  auto-observe, read-tool live-turn ingestion, first-call session prime, lite
  auto-recall, triggered-intention draining, notification lookup, and additive
  response enrichment. `server/engram/mcp/server.py` keeps only the compatibility
  wrapper that supplies MCP session/global dependencies and JSON tool wiring.
- Moved MCP auto-recall result compaction into the same retrieval helper module.
  `compact_auto_recall_surface()` now owns the score filter, compact entity
  summary, top-fact truncation, cue-episode payload, packet attachment, and
  no-surfaceable-results decision; `_auto_recall_full()` now delegates that
  result shape instead of building the MCP response inline.
- Moved MCP recall enrichment attachment into the same retrieval helper module.
  `apply_mcp_recall_enrichment()` now owns the additive response keys for
  session context, recalled context, triggered intentions, and memory
  notifications.
- Added `server/tests/test_public_surface_presenter_boundaries.py` so REST and
  MCP observe/remember/recall/chat recall surfaces stay tied to the shared
  ingestion and retrieval presenters instead of drifting back to local response
  formatting.
- Added `server/tests/test_consolidation_presenter_boundaries.py` so REST, MCP,
  and CLI consolidation cycle/status/history/detail outputs stay tied to the
  shared consolidation presenter.
- Refreshed the dashboard completion-readiness gate. Full Vitest passes with
  the existing React `act(...)`/canvas warnings, and the production build
  passes with the existing large-chunk warning; the live native dashboard smoke
  now also passes when REST is started against the seeded PyO3 data directory
  with both app and auth default group set to `native_brain`.
- Normalized Helix native `update_evidence` optional string payloads so
  `commit_reason=None` or `committed_id=None` are sent as empty strings instead
  of JSON null to the PyO3 query layer. This addresses the decode-error shape
  seen during live native dashboard smoke shutdown consolidation.
- Tightened the native/default-group config contract. `ENGRAM_DEFAULT_GROUP_ID`
  now drives `auth.default_group_id` when the auth default group is omitted, so
  unauthenticated REST can follow the intended one-brain group without also
  setting `ENGRAM_AUTH__DEFAULT_GROUP_ID`; explicit auth overrides still win,
  including an explicit `default`. README and Helix install docs now show the
  auth env var as a commented override so copied native-mode env blocks do not
  accidentally defeat inheritance.
- Tightened the cue-usefulness gate so reports no longer treat cue existence as
  enough coverage. Projected/consolidated smoke now records a surfaced cue
  feedback check, and the PyO3 native Helix schema/runtime now persists the
  same `EpisodeCue` feedback counters used by lite mode.
- Added a static Helix schema contract guard so the server Helix schema,
  native `helixdb-cfg/db/schema.hx`, and generated PyO3 query source cannot
  drift on `EpisodeCue` fields or the `update_cue_by_episode` feedback route.
- Extended that native schema guard to `Entity` provenance fields and the
  graph-embedding cleanup route. PyO3 native now stores and returns
  `source_episode_ids`, evidence counts, and evidence span bounds on entity
  create/read/update, and the rebuilt native engine exposes
  `delete_graph_embed_vector`.
- Closed the next graph-embedding cleanup contract on the native path.
  `HelixSearchIndex.get_graph_embeddings()` now reads Helix vector-search
  `data` payloads, and a PyO3 native regression proves graph embeddings can be
  synced, retrieved, cleared through `delete_graph_embed_vector`, and verified
  absent afterward.
- Closed the graph-embedding phase replacement contract on the native path.
  `GraphEmbedPhase` now has a PyO3 native regression proving a full retrain
  clears stale native graph vectors before syncing the new trained vectors, so
  the Consolidate-stage behavior is covered beyond mocks.
- Closed native open-work queue parity for consolidation adjudication. Helix
  now exposes status-filtered evidence/adjudication queries, `HelixGraphStore`
  reads the same open statuses as lite mode (`pending`, `deferred`, `approved`
  evidence and `pending`, `deferred`, `error` adjudication requests), and a
  PyO3 native regression proves deferred/open work round-trips through the
  preferred no-Docker backend.
- Surfaced that native/lite open-work queue pressure in the shared stats and
  brain-loop report contract. SQLite and Helix now emit `adjudication_metrics`
  with open evidence/request counts by status, the Consolidate report treats
  live open adjudication work as `attention`, Markdown prints open work beside
  adjudication phase effect, and the dashboard API client/types preserve the
  same fields.
- Tightened MCP consolidation trigger audit-store ownership. MCP
  `trigger_consolidation()` now reuses the active MCP consolidation store,
  including PyO3 native Helix, instead of only creating a SQLite shared-DB
  fallback; the native parity test now proves a triggered dry-run appears as
  the latest MCP consolidation status cycle.
- Extracted evidence/adjudication materialization into
  `server/engram/ingestion/adjudication_service.py`. `GraphManager` remains the
  compatibility facade for REST, MCP, projection execution, and consolidation
  phases, while the service owns open adjudication presentation, stored evidence
  materialization, committed-id mapping, evidence storage-row serialization, and
  client/server adjudication resolution.
- Extracted explicit preference feedback reinforcement into
  `server/engram/retrieval/preference_feedback.py`.
  `GraphManager.record_explicit_feedback()` remains the REST/MCP compatibility
  API, while the recorder owns UserPreference profile creation, PREFERS/AVOIDS
  edge create/strengthen behavior, domain lookup, and `feedback.recorded`
  publication.
- Extracted memory forgetting/correction into
  `server/engram/retrieval/forgetting.py`. `GraphManager.forget_entity()` and
  `GraphManager.forget_fact()` remain compatibility APIs for REST/MCP, while the
  service owns soft-deleting entities, clearing activation, resolving fact
  endpoints, invalidating relationships, and preserving the forget response
  contract.
- Extracted direct entity/fact lookup into
  `server/engram/retrieval/lookup.py`. `GraphManager.resolve_entity_name()`,
  `search_entities()`, and `search_facts()` remain REST/MCP compatibility APIs,
  while the service owns read-only entity search, activation-score decoration,
  fact search/name resolution, and epistemic fact filtering.
- Extracted agent context assembly into
  `server/engram/retrieval/context_builder.py`. `GraphManager.get_context()`,
  `_template_briefing()`, `_render_tier()`, and `_entity_to_context_data()` remain
  compatibility APIs for REST, MCP, and older tests, while `MemoryContextBuilder`
  owns tiered identity/project/recent/intention/pinned-context assembly,
  deterministic briefing cache behavior, token budgeting, project-neighbor
  injection, and context access events.
- Extracted prospective-memory intention management into
  `server/engram/retrieval/prospective.py` via `ProspectiveMemoryService`.
  `GraphManager.create_intention()`, `list_intentions()`, `dismiss_intention()`,
  `delete_intention()`, `migrate_flat_intentions()`, `_update_intention_fire()`,
  and `update_intention_meta()` remain compatibility APIs, while the service owns
  graph-embedded intention creation, v1 flat-table fallback, TRIGGERED_BY edges,
  list filtering, soft/hard dismiss behavior, fire-count updates, metadata
  updates, and intention lifecycle events.
- Extracted graph-state reads into `server/engram/retrieval/graph_state.py`.
  `GraphManager.get_graph_state()` remains the REST/MCP/lifecycle/evaluation
  compatibility API, while `GraphStateService` owns graph stats enrichment,
  top-activated entity materialization, active/dormant counts, recall/epistemic
  metrics attachment, entity-type filtering, and optional relationship-edge
  expansion.
- Moved MCP graph-state tool and graph/entity resource response assembly behind
  `server/engram/retrieval/graph_state.py` helpers. MCP now delegates graph-state
  tool payloads, graph stats resource payloads, entity profile resources, and
  entity neighbor resources to retrieval-side surface helpers while
  `GraphStateService` remains the deeper read-model owner.
- Moved the REST entity-neighbor convenience route onto the same graph-state
  helper. `/api/entities/{entity_id}/neighbors` now calls
  `build_api_graph_neighborhood_surface()` directly instead of importing and
  invoking the `/api/graph/neighborhood` route function, and the public-surface
  guard forbids that route-to-route coupling from returning.
  - Verification: `uv run pytest
    tests/test_api_endpoints.py::TestEntityDetail::test_get_entity_neighbors
    tests/test_mcp_graph_state_surfaces.py
    tests/test_public_surface_presenter_boundaries.py -q` passed with 149
    tests; `uv run ruff check engram/api/entities.py
    tests/test_public_surface_presenter_boundaries.py` passed.
- Extracted epistemic question routing into
  `server/engram/retrieval/epistemic_route.py`. `GraphManager.route_question()`
  and `_build_epistemic_route()` remain compatibility APIs for REST, MCP, and
  evidence gathering, while `EpistemicRouteService` owns memory-need analysis
  integration, graph-probe use, surface capability derivation, question-frame
  construction, evidence-plan construction, answer-contract application, payload
  formatting, and route metrics recording.
- Extracted epistemic evidence gathering into
  `server/engram/retrieval/epistemic_evidence.py`.
  `GraphManager.gather_epistemic_evidence()` remains the chat/API compatibility
  API, while `EpistemicEvidenceService` owns route-guided memory/artifact/runtime
  source queries, project bootstrap before artifact reads, memory/artifact/runtime
  claim construction, claim-state summary generation, two-pass answer-contract
  reconciliation, stale-artifact miss detection, execution metrics, and
  `EpistemicBundle` assembly.
- Extracted runtime-state reads into `server/engram/retrieval/runtime_state.py`.
  `GraphManager.get_runtime_state()` remains the REST/MCP/epistemic-evidence
  compatibility API, while `RuntimeStateService` owns effective mode/config
  reporting, feature flags, project artifact freshness counts, latest observed
  artifact timestamp selection, recall/epistemic metrics attachment, and
  generated-at timestamps.
- Extracted decision graph materialization into
  `server/engram/ingestion/decision_materializer.py`. Conversation capture and
  artifact bootstrap still call through `GraphManager` compatibility wrappers,
  while `DecisionMaterializer` owns committed conversation-decision extraction,
  conversation-record artifact upsert, artifact-claim decision linking,
  Decision entity upsert/reinforcement, supersession edges, and idempotent
  relationship creation.
- Extracted consolidation cycle completion into
  `server/engram/consolidation/completion.py`. `ConsolidationEngine` still owns
  phase-loop orchestration, cancellation, capability validation, and non-fatal
  phase failures, while `ConsolidationCycleCompletionService` owns final
  duration stamping, final store update, post-cycle learning event publication,
  successful-cycle finalization, and the final `consolidation.completed` event.
- Extracted structure-aware entity indexing into
  `server/engram/ingestion/entity_indexer.py`. Projection post-processing,
  adjudication materialization, decision materialization, and project bootstrap
  still call the `GraphManager._index_entity_with_structure()` compatibility
  wrapper, while `StructureAwareEntityIndexer` owns predicate-weighted
  relationship context expansion and enriched search-index payload creation.
- Extracted artifact search/read behavior into
  `server/engram/retrieval/artifacts.py`. REST, MCP, runtime state, and
  epistemic evidence still call through `GraphManager` compatibility wrappers,
  while `ArtifactSearchService` owns project artifact listing, optional
  bootstrap-before-read, search-index/fallback lookup, lexical claim fallback,
  stale detection, and `ArtifactHit` construction.
- Extracted project bootstrap writes into
  `server/engram/ingestion/project_bootstrap.py`. REST, MCP, artifact search,
  and epistemic evidence still call through `GraphManager.bootstrap_project()`,
  while `ProjectBootstrapService` owns project entity create/refresh, bootstrap
  file expansion, artifact entity upsert, cue-only bootstrap episode capture,
  PART_OF links, artifact-decision materialization calls, and bootstrap events.
- Extracted lightweight entity-probe recall into
  `server/engram/retrieval/entity_probe.py`. MCP auto-recall and other callers
  still use `GraphManager.recall_lite()` and `GraphManager.recall_medium()`,
  while `EntityProbeRecallService` owns mention extraction, session-cache use,
  FTS candidate probing, optional embedding rerank, top-fact rendering,
  confidence/freshness labels, and token-budget packing.
- Reduced the broad local gate warning noise by replacing benchmark-corpus
  `datetime.utcnow()` / `datetime.utcfromtimestamp()` calls with Engram's naive
  UTC helper and timezone-aware timestamp conversion. The broad warning count
  dropped from 11,474 to 500 while preserving the 2501/43/236 result.
- Continued that warning cleanup in production runtime code by moving
  prospective-memory, temporal parsing, graph-embedding storage, and dream
  association timestamps onto Engram's UTC helpers. The broad gate still passes
  at 2501/43/236 and now reports 462 warnings.
- Removed the largest remaining test-fixture warning block from schema
  formation tests by using the same UTC helper in the local entity factory. The
  broad gate still passes at 2501/43/236 and now reports 106 warnings.
- Cleaned the next consolidation fixture warning cluster in replay, prune,
  maturation, microglia, and predicate-enriched embedding tests. The broad gate
  still passes at 2501/43/236 and now reports 20 warnings.
- Removed the remaining broad-gate warnings by cleaning the last datetime
  fixtures and async-mark mismatch in consolidation graph methods, MCP facts,
  prospective memory, structural merge, structure-aware embeddings, and
  proactive recall tests. The non-Docker/non-Helix backend gate now passes with
  zero warnings reported.
- Fixed numeric identifier validation so SKU/part-like codes can be projected
  as `Identifier` entities instead of being rejected before normalization.
- Added a shared recall presenter in `server/engram/retrieval/presenter.py`.
  REST recall, MCP recall, and knowledge-chat recall now format `entity`,
  `episode`, and `cue_episode` results from one semantic contract.
- Added recall presenter contract tests so REST-style, MCP-style, and chat
  summaries stay aligned while preserving each surface's naming convention.
- Added a shared observe/remember presenter in
  `server/engram/ingestion/presenter.py`. REST and MCP write paths now share
  lifecycle semantics for capture, cue, projection mode, projection status, and
  adjudication request formatting while preserving existing top-level response
  keys.
- Made the GraphSAGE normalization test deterministic after the full lite gate
  exposed random all-zero post-ReLU output as a test flake, not a write-path
  regression.
- Extracted `GraphManager.store_episode()` into
  `server/engram/ingestion/capture_service.py`. `GraphManager` remains the
  compatibility facade, while the service owns raw episode creation, queued
  events, deterministic cue generation/indexing, initial projection-state
  updates, and decision-graph capture side effects.
- Extracted one-shot `GraphManager.ingest_episode()` sequencing into
  `server/engram/ingestion/episode_ingestion.py`. REST, MCP, benchmarks, and
  older integration tests still call the `GraphManager` compatibility API,
  while `EpisodeIngestionService` owns the store-then-project workflow,
  proposed entity/relationship forwarding, attachment/conversation metadata
  forwarding, and the existing return-episode-id-on-projection-failure
  behavior.
- Extracted REST offline capture replay into
  `server/engram/ingestion/offline_replay.py`. The `/api/knowledge/replay-queue`
  route still owns tenant resolution, manager dependency lookup, and JSON
  wrapping, while `OfflineReplayService` owns queue draining, short-content
  skips, dedup skips, active `group_id` store calls, failed-entry accounting,
  and replay counts. `build_api_offline_replay_surface()` now owns service
  construction and the `status: replayed` REST acknowledgement payload.
- Extracted auto-observe/replay deduplication into
  `server/engram/ingestion/dedup.py`. The API keeps `_DEDUP_CACHE` and
  `_dedup_check()` compatibility handles for existing tests and monkeypatches,
  while `CaptureDedupCache` owns hash generation, TTL duplicate detection,
  stale eviction, and max-entry cleanup.
- Added `server/engram/ingestion/projection_state.py` as the shared
  episode/cue projection-state synchronization helper. `GraphManager`,
  `EpisodeCaptureService`, `EpisodeWorker`, triage, and replay now use it for
  the touched state transitions.
- Tightened the remaining GraphManager cue-feedback promotion path so hot cue
  recall schedules episode projection through `sync_projection_state()` instead
  of separately writing episode status, episode projection state, and cue
  projection metadata.
- Finished the next projection-state audit pass for production runtime paths.
  Worker system-discourse skips and project-bootstrap artifact suppression now
  use `sync_projection_state()`, and the remaining direct `update_episode()`
  calls inspected were non-projection metadata writes or storage serialization.
- Extracted projection execution into
  `server/engram/ingestion/projection_service.py`. `GraphManager.project_episode()`
  remains the compatibility facade, while `EpisodeProjectionService` now owns
  duplicate/system-discourse skips, projection planning, evidence/legacy
  projection execution, apply/index/post-processing, completion events, and
  retry/dead-letter state handling.
- Added `server/engram/ingestion/projection_execution.py` and moved the legacy
  extractor -> apply -> relationship-write path behind `LegacyProjectionExecutor`.
  `EpisodeProjectionService` still orchestrates lifecycle, but it no longer owns
  the legacy graph-apply mechanics directly.
- Expanded `server/engram/ingestion/projection_execution.py` with
  `EvidenceProjectionExecutor`. The evidence hot path now owns evidence bundle
  building, optional edge-adjudication work item storage, commit/defer
  decisions, committed evidence materialization, and deferred/committed evidence
  persistence outside the main projection lifecycle service.
- Added `ProjectionLifecycleResult` as the typed Project-stage outcome
  contract. `EpisodeProjectionService.project_episode()` and
  `GraphManager.project_episode()` now return it for projected and skipped
  outcomes, while preserving existing lifecycle events and allowing current
  callers to ignore the return value.
- Added `group_id` to `ProjectionPlan` and made the planner carry the episode's
  active brain group into legacy projection. `EpisodeProjector` now forwards
  `episode_id`/`group_id` only to extractors that declare support for those
  keywords, and `NarrowExtractorAdapter` uses them when building narrow
  evidence bundles instead of always using `default`.
- Tightened consolidation replay's remaining one-brain boundary. Replay now
  passes the cycle `group_id` into linked-episode entity reads and group-aware
  extractors, so deferred/native narrow replay cannot rebuild evidence metadata
  under the raw `default` brain.
- Tightened the remaining projection-yield and semantic-transition entity-link
  reads. Worker feedback, triage feedback, and semantic coverage now read linked
  entities with the active brain `group_id` instead of relying on storage
  defaults.
- Added `server/engram/consolidation/lifecycle.py` as the first Consolidate
  lifecycle contract. It builds a typed selected phase plan and normalizes
  phase/cycle lifecycle event payloads while keeping `ConsolidationEngine` as
  the compatibility surface and leaving phase execution in place.
- Tightened that Consolidate lifecycle contract so requested phase names are
  validated against the actual engine phase registry before a cycle starts.
  Unknown phase names now fail fast instead of silently producing a zero-work
  scheduled or CLI cycle; `python -m engram.consolidation --phases <bad-name>`
  now exits with a clean operator error instead of a traceback.
- Tightened the consolidation CLI operator contract for failed cycles. It still
  prints the structured cycle JSON, but failed or cancelled cycles now say
  `Consolidation failed`/`cancelled` instead of `complete` and exit nonzero.
  The JSON output now includes cycle-level and phase-level `error` fields so
  operator tooling can inspect failures without scraping stderr.
- Aligned MCP `trigger_consolidation` with that structured error contract.
  The tool response now includes cycle-level and phase-level `error` fields,
  and failed cycles describe themselves as failed in the summary text.
- Aligned MCP `get_consolidation_status` with the latest-cycle read contract.
  It still reports `is_running=false` because MCP cycles are synchronous, but
  now reads the active consolidation store and returns `latest_cycle` through
  the shared consolidation presenter, including cycle and phase errors.
- Aligned REST consolidation status/history/detail reads with the same cycle
  serializer. Status and history now include cycle `error`, phase `error`, and
  phase `duration_ms`, matching detail responses and dashboard types.
- Updated the dashboard consolidation slice to preserve `latest_cycle` from
  `/api/consolidation/status` by merging it into the cycle list. Cycle and
  phase errors now survive the status-refresh path even before history refresh
  completes.
- Surfaced cycle-level errors in the dashboard Consolidation panel's cycle
  list. Failed cycles now show their error below the timestamp instead of
  requiring the user to open detail first.
- Corrected the dashboard API client type for REST consolidation trigger
  responses. The backend returns `status`, `group_id`, and `dry_run`; the
  dashboard no longer claims a nonexistent `cycle_id`.
- Added `server/engram/consolidation/presenter.py` as the shared cycle/phase
  result presenter. REST, MCP, and CLI consolidation outputs now reuse the same
  serializer, total calculation, and failed-cycle description semantics while
  preserving each surface's existing `id`/`cycle_id` naming convention.
- Moved the consolidation operator summary payload into that shared presenter.
  REST status/history/detail, MCP trigger/status, CLI JSON, lifecycle summary,
  and dashboard types now share the same `summary.total_processed`,
  `summary.total_affected`, and `summary.description` contract instead of MCP
  and CLI rebuilding the totals/description separately.
- Added `server/engram/consolidation/phase_registry.py` as the shared backend
  phase order/tier contract. Scheduler tiering and engine tests now read the
  same phase registry instead of carrying independent 17-phase lists, and
  `ConsolidationEngine` validates constructed runtime phase order against that
  registry on startup.
- Added `server/engram/consolidation/phase_catalog.py` as the Consolidate-stage
  phase construction boundary. `ConsolidationEngine` still owns the run loop,
  cancellation, capability validation, non-fatal phase failures, completion, and
  events, while the catalog owns concrete runtime phase assembly and registry
  order validation.
- Added `dashboard/src/constants/consolidation.ts` as the dashboard-side
  17-phase mirror. Consolidation panel fixtures now build from that list, and
  quest-mode phase descriptions are typed against it so adjudication or future
  phase drift is caught by the dashboard build. Added
  `server/tests/test_dashboard_phase_contract.py` to compare that dashboard
  mirror against the backend phase registry.
- Added `server/engram/consolidation/phase_runner.py` as the next Consolidate
  runtime boundary. `ConsolidationEngine` still owns the cycle loop,
  cancellation, and non-fatal phase error handling, while the runner owns one
  phase execution, direct audit-record persistence, new decision-trace/label
  persistence, and merge/prune removed-node discovery.
- Added `server/engram/consolidation/events.py` as the Consolidate event
  publishing boundary. Cycle start/end, phase start/complete/fail, graph deltas,
  and learning update notifications now flow through
  `ConsolidationEventPublisher`, which uses the typed lifecycle contracts for
  event payloads while preserving existing event names and legacy keys.
- Added `server/engram/consolidation/learning.py` as the post-cycle learning
  boundary. `ConsolidationLearningService` now owns distillation example
  generation/persistence, calibration history collection, calibration snapshot
  generation/persistence, and artifact counts returned to the event publisher.
- Added `server/engram/consolidation/finalization.py` as the post-cycle
  finalization boundary. `ConsolidationFinalizationService` now owns pinned
  context refresh after successful consolidation cycles, leaving
  `ConsolidationEngine` focused on cycle state, phase selection, cancellation,
  and event ordering.
- Added `server/engram/consolidation/capabilities.py` as the Consolidate-stage
  preflight boundary. `ConsolidationCapabilityValidator` now owns selected-phase
  runtime method checks for graph, activation, and search stores while
  preserving the existing failed-cycle error contract.
- Re-ran the broad non-Docker/non-Helix backend gate after the capability
  extraction. It passed with 2504 tests, 43 skips, and 236 external-service
  tests deselected; one transient `aiosqlite` event-loop-close thread warning
  surfaced at `tests/test_summary_merge.py::TestIsMetaSummary::test_knowledge_graph_node`,
  then did not reproduce in the isolated summary-merge test or the adjacent
  structure-embeddings plus summary-merge rerun.
- Added the first P2 dashboard lifecycle surface in
  `dashboard/src/components/LifecyclePanel.tsx`. The default observatory view is
  now Brain Loop, mapping existing stats, episodes, cue summaries, recall
  results, activation, scheduler state, and consolidation cycles into
  Capture -> Cue -> Project -> Recall -> Consolidate.
- Added `GET /api/lifecycle/summary` in `server/engram/api/lifecycle.py` as the
  first backend lifecycle summary contract. The dashboard now loads this shared
  Capture/Cue/Project/Recall/Consolidate summary through `lifecycleSlice`, while
  WebSocket episode, graph, activation snapshot, and consolidation events
  refresh it.
- Extracted the lifecycle summary builder into
  `server/engram/lifecycle_summary.py` and added MCP `get_lifecycle_summary`.
  REST and MCP now return the same Capture/Cue/Project/Recall/Consolidate
  summary contract instead of leaving headless clients behind the dashboard.
- Reused the shared consolidation cycle presenter inside that lifecycle summary
  contract. `consolidate.latestCycle` now carries the same cycle-level and
  phase-level error fields as REST consolidation status/history/detail, MCP,
  and CLI outputs instead of using a duplicate local serializer.
- Extended the P3 brain-loop evaluation report's consolidation summary with
  `latest_cycle.error`. The report already counted cycle-level errors; it now
  exposes the actual cycle error text next to phase errors for REST, MCP, CLI,
  dashboard, and smoke consumers.
- Updated the brain-loop Markdown renderer to print that latest cycle error on
  the Consolidate summary line, so CLI/operator reports no longer hide the
  top-level consolidation failure while JSON carries it.
- Surfaced that cycle-level consolidation failure reason in the dashboard
  Evaluation panel. The Consolidate card now shows a compact `latest error`
  row when `latest_cycle.error` is present, while API client tests keep the raw
  field preserved through mapping.
- Surfaced the same failure reason in the primary Brain Loop dashboard. The
  Lifecycle panel's Consolidate stage now appends the latest cycle error to the
  stage summary when present, with wrapping so longer operator errors do not
  overflow the card.
- Tightened shared lifecycle Consolidate health for non-fatal phase failures.
  Completed cycles with a phase-level `status: error` or phase `error` now mark
  Consolidate as `attention`, and the Brain Loop fallback path displays the
  first phase issue instead of reporting a ready completed cycle.
- Aligned the headless lifecycle Markdown renderer with that same issue
  selection. `engram lifecycle --format markdown` now prints a cycle-level
  error first, or the first phase-level Consolidate issue when the cycle itself
  completed without a top-level error.
- Aligned `engram doctor --format markdown` with the same lifecycle issue
  selection so the diagnostic snapshot does not reduce phase-level Consolidate
  failures to a bare `attention` status.
- Tightened `engram doctor` lifecycle snapshot status. Loading the snapshot is
  still a pass when all stages are ready/active, but any lifecycle stage with
  `attention` now makes the `lifecycle_snapshot` check warn, carrying stage
  statuses and the Consolidate phase issue in metadata.
- Tightened the direct consolidation CLI summary for non-fatal phase errors.
  Completed cycles still exit successfully, but if any phase has an error the
  human summary now says `completed with warnings` and stderr names the first
  phase issue instead of saying clean `complete`.
- Moved completed-with-warning wording into the shared consolidation presenter.
  MCP `trigger_consolidation` summaries now say `cycle with warnings` for
  completed cycles with phase errors instead of returning a clean completed
  cycle description.
- Added `phase_issue` to the shared consolidation cycle summary. REST, MCP,
  lifecycle, doctor, CLI, and dashboard consumers can now read the first
  phase-level issue directly instead of re-scanning phase arrays, while phase
  `error` fields remain available.
- Extended the P3 brain-loop evaluation report and dashboard Evaluate panel to
  use `latest_cycle.phase_issue` when a completed cycle has no top-level cycle
  error. Markdown now prints `Phase issue: ...`, and the dashboard shows it in
  the Consolidate latest-issue row.
- Tightened the P3 Consolidate-stage evaluation status. Recent cycle or phase
  issues now make the evaluation report's Consolidate stage `attention` instead
  of leaving the stage `ready` while `error_count` and `latest_cycle.phase_issue`
  report a problem.
- Updated the dedicated dashboard Consolidation panel to surface `phase_issue`
  in the cycle list when `cycle.error` is empty, so completed-with-warning
  cycles do not look clean unless the user opens detail.
- The dashboard Consolidation detail timeline now renders each phase's `error`
  text below the phase name, so opening a completed-with-warning cycle shows the
  concrete failing phase message instead of only a red status dot.
- Tightened the dashboard Consolidation panel's completed-with-warning visual
  state. Completed cycles with `phase_issue` now render with warning color and a
  `warning` detail badge instead of using the same green status treatment as a
  clean completed cycle.
- Extended the no-bind native dashboard fixture smoke with the same
  completed-with-warning consolidation shape. The fixture now carries
  `phase_issue` through lifecycle, evaluation, consolidation status/history,
  cycle detail, and verifies the Consolidation panel's warning detail state
  without requiring a live REST bind.
- Added `phase_issue` to the typed `consolidation.completed` lifecycle event
  payload. WebSocket/event consumers now receive the same completed-with-warning
  issue field as REST, MCP, CLI, lifecycle, doctor, and dashboard reads.
- Updated dashboard quest-mode WebSocket handling for `consolidation.completed`.
  If the event carries `phase_issue`, the quest log now records a lower-XP
  warning message instead of a clean `Quest completed!` celebration.
- Re-ran the broad backend and dashboard gates after the shared `phase_issue`
  warning contract and later projection-plan group metadata work. The backend
  non-Docker/non-Helix suite now passes at 2580/43/236, the dashboard build
  passes with only the existing large-chunk
  warning, and the full dashboard test suite passes when run with one Vitest
  worker. An unconstrained dashboard run hit worker startup timeouts after 97
  tests had already passed, so keep the low-worker command for local full-suite
  verification on this machine.
- Updated `engram lifecycle --format markdown` to include the latest
  consolidation cycle error on the Consolidate line, keeping the headless
  lifecycle snapshot aligned with lifecycle JSON and dashboard Brain Loop
  output.
- Extended the Recall stage in that lifecycle summary with prospective-memory
  intention counts. REST, MCP, CLI, and the dashboard now see active intentions,
  refresh-context intentions, after-consolidation pinned contexts, pinned-result
  count, needs-refresh count, and latest refresh timestamp as part of Recall
  state. The dashboard `LifecycleSummary` type now treats that intention summary
  as a required backend contract, native dashboard smoke fixtures include it,
  and the CLI Markdown renderer prints active intention/pinned counts in the
  Recall stage line.
- Refreshed the audit's Drift And Gaps section so it no longer describes the
  original P0/P2/P3 gaps as still open. The current remaining risks are
  GraphManager re-accumulation, future surface contract drift, phase-list drift,
  quest-mode dominance, longer native endurance gates, live AI-harness adoption
  evidence, and deeper benchmark integration. Docker Helix/full-mode is now
  treated as a separate compatibility/integration lane because PyO3 native is the
  full-backend completion path for no-Docker operator readiness.
- Added `engram lifecycle` as a headless CLI snapshot for the same shared
  lifecycle summary. It resolves the configured engine mode, including Helix
  native PyO3, and prints Markdown or JSON without requiring REST, MCP, or the
  dashboard to be running.
- Embedded the shared lifecycle snapshot in `engram doctor`. Doctor JSON and
  Markdown now include a `lifecycle_snapshot` check plus `lifecycle_summary`
  output by default, with `--no-lifecycle` available for config-only runs.
- Clarified the external agent/client contract: README now lists
  `/api/lifecycle/summary`, `get_lifecycle_summary`, and evaluation endpoints,
  while the OpenClaw/Engram skill documents when agents should inspect the
  lifecycle snapshot. The skill and install docs now say 16 consolidation
  phases and 26 MCP tools.
- Wired Brain Loop stage cards into existing dashboard drilldowns. Capture opens
  Feed, Cue and Project open Stats, Recall opens Knowledge, and Consolidate
  opens Consolidation while selecting the latest known cycle when available.
- Added lifecycle drilldown context to the dashboard store. Brain Loop card
  clicks now carry their originating stage into the target panel, while ordinary
  sidebar/search/entity navigation clears that context. Capture opens Feed with
  the active-capture status filter selected; Cue and Project open Stats with the
  relevant section highlighted and scrolled into view when needed.
- Added Recall-stage context inside the Knowledge drilldown. The Recall card now
  opens Knowledge with the active recall context strip visible and focused, even
  when no pulse entities are currently loaded.
- Added the first P3 local evaluation report in
  `server/engram/evaluation/brain_loop_report.py`, plus
  `server/scripts/brain_loop_report.py`. It builds a JSON-serializable and
  Markdown report for Capture, Cue, Project, Recall, and Consolidate from
  SQLite/lite stats, optional JSON exports, labeled recall samples, session
  continuity samples, recent consolidation cycles, and calibration snapshots.
- Added `server/engram/evaluation/store.py` as the first persisted local
  evaluation sample store. SQLite/lite mode can now store labeled recall
  decisions and session-continuity labels, and the brain-loop report CLI reads
  those saved samples by default unless `--no-saved-samples` is passed.
- Added `server/engram/api/evaluation.py` and app wiring for
  `POST /api/evaluation/recall-samples`,
  `POST /api/evaluation/session-samples`, and
  `GET /api/evaluation/brain-loop/report`. REST can now record local evaluation
  labels and return the same brain-loop report contract used by the CLI.
- Added `engram evaluate` as the first-class CLI surface for the same local
  brain-loop report. `server/scripts/brain_loop_report.py` is now a thin
  compatibility wrapper over `engram.evaluation.cli`.
- Added the P3 dashboard evaluation drilldown in
  `dashboard/src/components/EvaluationPanel.tsx`. The sidebar now exposes
  Evaluate, the dashboard loads `GET /api/evaluation/brain-loop/report` through
  `evaluationSlice`, and lifecycle-affecting WebSocket events refresh the
  report.
- Added MCP participation in the P3 evaluation loop. `engram mcp` now
  initializes the local `SQLiteEvaluationStore`, exposes
  `record_recall_evaluation`, `record_session_continuity_evaluation`, and
  `get_evaluation_report`, and REST/MCP write acknowledgements share
  `server/engram/evaluation/presenter.py`.
- Ran a live P3 evaluation smoke with the lite backend and dashboard connected.
  A seeded episode plus recall/session-continuity labels flowed through
  `GET /api/evaluation/brain-loop/report` and rendered in the Evaluate
  drilldown with measured recall and continuity signals.
- Added the operator-facing P3 label path in the Evaluate drilldown. The
  dashboard API client and `evaluationSlice` now write recall and
  session-continuity labels to the existing REST endpoints, refresh the report
  after successful writes, and expose saving state to the UI.
- Added a missed-recall evaluation signal to the P3 loop. Recall labels now
  carry optional `recall_needed` / `recallNeeded` state through SQLite, REST,
  MCP, the shared presenter, CLI/report builder, smoke labels, and the
  dashboard Evaluate form. Reports now expose memory-need recall, missed-recall
  rate, need-label counts, needed count, and missed count, so Engram can measure
  "memory should have been recalled but was not" instead of only precision for
  triggered recall.
- Closed the remaining transient broad-gate warning source by making
  structural-merge SQLite integration tests close their in-memory graph stores.
  The broad non-Docker/non-Helix backend gate now passes cleanly again with 2505
  tests, 43 skips, 236 external-service tests deselected, and no warning
  summary.
- Made cue usefulness more visible in the P3 evaluation surface. The Markdown
  brain-loop report now includes cue-to-projection conversion, and the Evaluate
  drilldown shows surfaced cues, selected rate, used rate, projection conversion,
  and near-miss rate instead of hiding conversion behind the raw contract.
- Made consolidation calibration quality visible in the same Evaluate stage
  card. Consolidate now shows top-phase calibration accuracy and expected
  calibration error alongside cycle, affected-item, error, and snapshot counts.
- Brought the headless Markdown report up to the same calibration standard. The
  Consolidate section now appends the top labeled calibration phase's accuracy
  and expected calibration error instead of reporting only snapshot count.
- Added `server/scripts/projected_consolidated_smoke.py` as the repeatable P3
  projected/consolidated smoke. It boots local stores against disposable
  storage, seeds queued episodes, runs the real triage consolidation phase,
  stores recall and continuity labels, builds the shared brain-loop report, and
  fails if projection/consolidation coverage gaps remain.
- Promoted that smoke into the first-class `engram evaluate --smoke` command.
  The reusable implementation now lives in `server/engram/evaluation/smoke.py`;
  `server/scripts/projected_consolidated_smoke.py` remains a thin compatibility
  wrapper.
- Added `engram doctor` as the first local diagnostic gate. It loads config,
  checks the configured SQLite path, resolves runtime mode, optionally checks
  the REST `/health` endpoint, and runs the same disposable brain-loop smoke.
- Pivoted the preferred local operator path to Helix native PyO3. Lite remains
  the disposable smoke/demo fallback, but `engram lifecycle`, `engram doctor`,
  setup, and CLI mode overrides should treat `ENGRAM_MODE=helix` with
  `ENGRAM_HELIX__TRANSPORT=native` as the main no-Docker full-backend path.
- Made the local evaluation report follow the same runtime direction.
  `engram evaluate --mode helix` now reads graph stats and consolidation cycles
  from the resolved Helix backend while keeping SQLite as the local label store.
  `engram evaluate --smoke --mode helix` now runs the same projected/consolidated
  smoke against native PyO3 Helix with a disposable label DB/data directory,
  while bare `engram evaluate --smoke` remains the lite fallback.
- Made `engram doctor` follow the same runtime direction for its brain-loop
  smoke. When diagnostics resolve to Helix, doctor now runs the disposable
  projected/consolidated smoke in native PyO3 mode and records `mode: helix` in
  the smoke check metadata; lite remains the fallback for bare smoke or
  unsupported smoke modes.
- Added explicit native data-directory targeting to `engram lifecycle` and
  `engram doctor`. `--helix-data-dir` now lets operators inspect a specific
  PyO3 Helix brain without relying on env-only configuration; doctor uses that
  directory for the lifecycle snapshot while keeping its projected/consolidated
  smoke on disposable native storage.
- Added the same explicit native data-directory targeting to runtime startup.
  `engram serve --mode helix --helix-data-dir ...` and
  `engram mcp --mode helix --helix-data-dir ...` now set native Helix transport
  plus `ENGRAM_HELIX__DATA_DIR` before booting REST or MCP, so the preferred
  no-Docker path does not require env-only setup.
- Aligned the native Makefile shortcuts with that same startup contract.
  `make up-native NATIVE_DATA_DIR=...` now invokes
  `engram serve --mode helix --helix-data-dir ...`, and
  `make mcp-native NATIVE_DATA_DIR=...` invokes
  `engram mcp --mode helix --transport streamable-http --helix-data-dir ...`.
  The shortcuts also now force `ENGRAM_HELIX__TRANSPORT=native` even when no
  explicit data directory is passed, so `make up-native` and `make mcp-native`
  cannot silently use the HTTP Helix transport.
- Aligned the public local installer with the same native direction. The
  one-click installer now validates `helix`, `lite`, `auto`, `full`, and
  `openclaw`, forwards explicit local modes into `engramctl setup --mode ...`,
  and `engramctl setup --mode helix` writes local lifecycle config with
  `ENGRAM_MODE=helix` plus `ENGRAM_HELIX__TRANSPORT=native` instead of falling
  through to interactive/default-lite behavior or the Docker full setup path.
  README and lite install docs now present Helix native as the main no-Docker
  growth path, with lite as fallback and Docker full as legacy.
- Made the public one-click Helix path fail honestly when native support is not
  present. `scripts/install.sh helix` now requests `engram[local,native]`, and
  `engramctl setup --mode helix` runs a no-smoke doctor check to verify that the
  `helix_native` PyO3 runtime is importable. If the installed package cannot
  supply native support yet, setup exits with source-build remediation instead
  of leaving a broken native config.
- Made local updates preserve that native contract. `engramctl update` now reads
  the local `ENGRAM_MODE`; Helix-native local installs upgrade
  `engram[local,native]` and rerun the same native verification instead of
  blindly upgrading `engram[local]` like a lite install.
- Tightened native mode resolution. Explicit Helix native mode now checks that
  the `helix_native` PyO3 extension is importable before returning
  `EngineMode.HELIX`; missing native support now fails early with a build/install
  remediation instead of surfacing later during store initialization.
- Fixed Helix graph stats so native lifecycle/evaluation reports count episodes,
  cue coverage, projection attempts, and projected episode entity yield instead
  of reporting zero episodes for populated Helix brains.
- Brought MCP evaluation reports into the same backend contract. MCP now opens a
  consolidation audit store for the resolved runtime, so
  `get_evaluation_report` can include Helix native consolidation cycles instead
  of only reading cycle context when the graph store exposes a SQLite `_db`.
- Brought MCP lifecycle summaries into that same consolidation-store contract.
  `get_lifecycle_summary` now passes the active MCP consolidation store into
  the shared lifecycle builder, so Helix native cycle counts and latest-cycle
  details appear in headless lifecycle snapshots.
- Extended populated native parity to MCP consolidation controls. The native MCP
  test now verifies `get_consolidation_status` and dry-run
  `trigger_consolidation` against the active PyO3 graph, including completed
  latest-cycle status, dry-run status, populated graph stats, phase results,
  and summary totals.
- Extended populated native parity to the MCP graph-stats resource. The native
  MCP test now calls `graph_stats_resource()` separately from the graph-state
  tool and verifies active PyO3 episode/entity counts, TestMemory type counts,
  cue coverage, projected cue counts, projection state counts, and projection
  yield.
- Extended populated native parity to MCP evaluation-label writes. The native
  MCP test now records recall-quality and session-continuity samples through
  `record_recall_evaluation` and `record_session_continuity_evaluation`, then
  requires the MCP evaluation report to see the extra native-brain samples.
- Extended populated native parity to REST dashboard read surfaces. The native
  REST test now verifies `/api/stats`, `/api/activation/snapshot`,
  `/api/activation/{entity_id}/curve`, `/api/graph/neighborhood`,
  `/api/graph/at`, and `/api/episodes` against the active PyO3 brain, including
  graph counts, top-connected entities, growth timeline, activation curve data,
  graph neighborhood/temporal edges, and cue-bearing episode listing.
- Tightened the native `/api/episodes` coverage to include status filtering and
  cursor pagination.
  The populated native REST test now filters the cue-bearing observe episode by
  `status=queued` and verifies the serialized episode projection state stays
  aligned with the serialized cue projection state, while allowing the cue
  policy to choose `cued`, `scheduled`, or `queued`. It also fetches two
  one-item pages and verifies the second page advances to a different episode.
- Fixed and covered native atlas persistence. `HelixAtlasStore` now writes the
  schema's `region_label` field and deletes existing materialized snapshot
  children through available find + hard-delete queries instead of non-existent
  bulk-delete query names. The populated native REST test now verifies atlas
  refresh, history, snapshot lookup, region drilldown, and same-snapshot-ID
  upsert cleanup against PyO3 Helix.
- Extended native parity to the dashboard WebSocket. A PyO3 native WebSocket
  test now opens `/ws/dashboard`, verifies ping/pong, forwards native-brain
  event-bus messages, ignores wrong-group events, and checks resync stays scoped
  to `native_brain`.
- Quieted EventBus hook scheduling in sync contexts. `publish()` now checks for
  a running event loop before creating fire-and-forget hook tasks, so sync
  TestClient and CLI publishes do not emit a no-current-event-loop deprecation
  warning while async runtime hooks still execute.
- Removed the remaining native parity UTC deprecation warnings. MCP session
  timestamps and GraphManager freshness labels now use Engram's shared
  `utc_now()` helper instead of `datetime.utcnow()` while preserving the
  existing naive-UTC timestamp convention.
- Hardened native runtime shutdown ownership. Helix stores now distinguish
  owned versus borrowed shared `HelixClient` instances so in-process native
  engines and HTTP/gRPC clients are closed by the owning graph store. MCP
  lifespan shutdown now stops its worker, removes/closes the Redis publisher,
  and closes evaluation, consolidation, search, activation, and graph stores.
- Added a populated PyO3 native evaluation smoke and live-report reopen check.
  Native smoke now verifies Capture -> Cue -> Project -> Recall -> Consolidate
  on Helix native without Docker, and the reopened `engram evaluate --mode helix`
  report preserves projection yield for non-default `group_id` brains.
- Made the projected/consolidated smoke genuinely deterministic after broader
  verification showed activation profile presets could rewrite the smoke's
  `triage_extract_ratio` back to the standard profile default. The smoke now
  reapplies its narrow extractor, full triage ratio, disabled worker, disabled
  background consolidation, and graph-embedding-off settings after profile
  post-init.
- Added missing SQLite episode schema/migration columns for `skipped_meta` and
  `skipped_triage`, matching the projection-state helper's update contract for
  fresh lite-mode databases.
- Added populated native REST/MCP surface parity coverage. The new
  `tests/test_native_surface_parity.py` seeds a PyO3 Helix brain through the
  canonical smoke, reopens it through FastAPI, checks lifecycle/evaluation/recall
  REST endpoints, then points MCP lifecycle/evaluation/recall tools at the same
  runtime objects.
- Extended populated native parity to REST health. The native REST test now
  calls `/health` first and verifies the reopened PyO3 runtime reports
  `status=healthy`, `mode=helix`, and a healthy graph store before exercising
  lifecycle, evaluation, recall, and mutation surfaces.
- Extended populated native parity to the REST admin benchmark loader without
  adding the full benchmark corpus to the regression path. The native REST test
  patches `CorpusGenerator` to a tiny fixture, calls
  `/api/admin/load-benchmark`, verifies active `native_brain` group override,
  writes fixture entities/relationships through the live PyO3 graph store, and
  reads them back through public entity search and fact lookup.
- Extended the populated native REST/MCP parity coverage to evaluation writes.
  The test now posts recall-quality and session-continuity labels through REST
  on the native brain with stale/wrong group fields in the payload, verifies
  they are stored under `native_brain`, and checks the MCP evaluation report
  sees the same persisted sample counts.
- Extended populated native parity to REST consolidation reads. The native REST
  test now verifies `/api/consolidation/status`, `/api/consolidation/history`,
  and `/api/consolidation/cycle/{id}` against the completed PyO3 native smoke
  cycle and checks phase/detail collections from the active Helix-backed
  consolidation store.
- Extended populated native parity to REST consolidation trigger. The native
  REST test now calls `/api/consolidation/trigger?dry_run=true` after the core
  lifecycle assertions, verifies the trigger payload uses `native_brain`, and
  waits for the dry-run manual cycle to appear completed in native history.
- Extended populated native parity to REST notifications. The native REST test
  now seeds `native_brain` and wrong-group notifications in the app notification
  store, verifies `/api/knowledge/notifications` pending/since reads stay scoped
  to the active PyO3 brain, dismisses through `/api/knowledge/notifications/dismiss`,
  and verifies dismissed notifications leave the pending feed while retaining
  dismissal metadata in recent reads.
- Tightened REST notification dismissal for one-brain-per-person semantics.
  `NotificationStore.dismiss()` and `dismiss_batch()` now accept an optional
  `group_id`, `/api/knowledge/notifications/dismiss` passes the active tenant
  group, and native parity now proves an active `native_brain` dismiss request
  cannot dismiss a wrong-group notification by ID.
- Tightened the dashboard WebSocket notification-dismiss command the same way.
  `dismiss_notification` now passes the connected tenant group, so a dashboard
  socket cannot dismiss another brain's notification even if it sends a valid
  wrong-group notification ID.
- Corrected the lite dashboard WebSocket contract wording and added a
  non-default tenant regression. The endpoint docstring now describes the
  authenticated tenant group's event bus, and `tests/test_websocket.py` proves
  a configured `tenant_brain` socket receives only that group's events instead
  of falling back to raw `default`.
- Tightened OIDC tenant fallback for the same non-default brain path.
  `OIDCValidator.validate_token()` now receives the configured
  `default_group_id`, and the middleware falls back to that configured group
  when an OIDC token lacks the configured group claim instead of silently
  routing to raw `default`.
- Extended PyO3/Helix cue usefulness parity. `EpisodeCue` nodes in the Helix
  schema now persist cue metadata, feedback counters, policy score, projection
  attempts, and feedback timestamps; `HelixGraphStore` round-trips those fields
  and includes them in `get_stats().cue_metrics` instead of hardcoding native
  cue usefulness to zero.
- Tightened the Redis event bridge's one-brain channel contract. The MCP-side
  publisher now ignores events for groups other than the configured bridge
  group, and the REST-side subscriber publishes received events into its
  subscribed channel group instead of falling back to serialized `default`.
  Mismatched serialized groups are dropped, preventing cross-brain lifecycle or
  dashboard event routing through Redis.
- Hardened the PyO3 Helix graph store's internal-ID caches for the same
  one-brain boundary. Entity and episode Helix IDs are now cached by
  `(group_id, external_id)` for resolver paths, while bare external-ID caches
  are kept only when the ID is known to map to a single Helix node. This keeps
  native Helix lookups group-scoped even if import or fixture data reuses an
  external entity or episode ID across brains.
- Fixed the memory activation snapshot write path so activation metadata can be
  persisted back to graph entity rows through the real graph-store contract.
  `MemoryActivationStore.snapshot_to_graph()` now passes the recorded
  `group_id` for each activated entity, falls back to `default` for ungrouped
  demo/test state, and converts epoch `last_accessed` values to Engram's
  storage-compatible naive UTC datetimes before calling `update_entity()`.
- Hardened the Redis/full activation store's group index contract without
  requiring a live Redis service in local tests. `set_activation()` and
  `batch_set()` now preserve the existing hash `group_id` and refresh the
  matching `act_group:{group_id}` sorted-set index, `record_access()` writes
  that index immediately, `clear_activation()` removes the indexed member, and
  the fast-path `get_top_activated(group_id=...)` rechecks each candidate hash's
  current group so a stale sorted-set entry cannot surface another brain's
  activation state.
- Made episode-entity linking group-aware across the graph-store protocol and
  production write paths. `link_episode_entity()` now accepts an optional
  `group_id`; Helix resolves episode/entity internal IDs through group-scoped
  caches when provided, Falkor matches both nodes under the same group, and the
  projection apply, replay, and benchmark corpus loaders pass the active
  episode group instead of relying on raw-ID or `default` fallback resolution.
- Tightened the episode-entity read side to match that write contract.
  `get_episode_entities(group_id=...)` now filters linked entities by the same
  group as the episode in SQLite and Falkor, and Helix filters returned entity
  nodes by `group_id` after resolving the active episode. This protects
  projection-yield, replay, and consolidation graph methods from surfacing
  legacy or malformed cross-group HAS_ENTITY links.
- Aligned the auth-exempt health probe with configured one-brain defaults.
  `/health` now calls `graph_store.get_stats(group_id=config.default_group_id)`
  instead of an unscoped `get_stats()` call that falls back to raw `default`.
  This keeps local/native installs whose configured brain is `native_brain`
  from using the wrong group during graph-store health checks.
- Aligned `engram evaluate` operator defaults with the configured one-brain
  default. `engram evaluate --smoke` and `engram evaluate --from-json` now use
  `EngramConfig.default_group_id` when `--group-id` is omitted instead of
  falling back to raw `default`, keeping PyO3/native operator checks pointed at
  the configured brain unless the operator explicitly overrides it.
- Extended populated native parity to project topology surfaces. The native
  REST test now bootstraps a temporary project with stale/wrong group fields,
  verifies the artifact search endpoint returns the README hit from the active
  `native_brain`, and confirms the bootstrap adds a cue-only capture without
  inflating projected-memory counts. The same bootstrapped project is now
  checked through MCP `bootstrap_project` as an idempotent
  `already_bootstrapped` project, then searched through MCP `search_artifacts`
  so headless agent artifact lookup is tied to the same PyO3 native brain. REST
  `/api/knowledge/runtime` and MCP `get_runtime_state` now verify the same
  native runtime mode and artifact freshness state for that project.
- Extended populated native parity to epistemic routing. REST
  `/api/knowledge/route` and MCP `route_question` now classify the same
  project-grounded reconcile question against the active PyO3 project context,
  verifying the shared evidence plan routes to artifacts and discourages raw
  facts.
- Extended populated native parity to MCP route auto-observe. The native MCP test
  now temporarily enables tool-call recall, calls `route_question` with a long
  project-grounded question, and verifies the middleware stores a `tool_piggyback`
  episode plus cue in the active PyO3 `native_brain`.
- Extended populated native parity to prospective memory intentions. REST
  `/api/knowledge/intentions` and MCP `intend`/`list_intentions`/
  `dismiss_intention` now create, list, soft-disable, and list disabled native
  intentions in `native_brain`.
- Extended populated native parity to edge adjudication resolution. The native
  test now creates pending adjudication work items in PyO3 Helix, resolves them
  through REST `/api/knowledge/adjudicate` and MCP `adjudicate_evidence`, and
  verifies the rejected resolution payload plus persisted request status under
  `native_brain`.
- Extended populated native parity to conversation persistence. The native REST
  test now creates a conversation, appends user/assistant messages, lists the
  conversation, and reads messages back from the active `native_brain` while
  ignoring stale/wrong group fields in the request payload.
- Extended that native conversation coverage to update/delete semantics. The
  REST parity test now renames the conversation, deletes it, verifies deleted
  conversations return 404, and checks PyO3 Helix message nodes are cleaned up
  through existing find + hard-delete queries instead of a non-existent bulk
  delete query.
- Extended populated native parity to the chat stream path. The native REST test
  now mocks the Anthropic client, posts to `/api/knowledge/chat`, verifies the
  streamed finish event returns a conversation id, reconstructs the streamed
  assistant text, and reads the persisted user/assistant turn back from native
  Helix.
- Extended populated native parity to the forget mutation surface. The native
  REST test now creates an explicit entity in `native_brain`, calls
  `/api/knowledge/forget` with stale/wrong group fields, verifies entity search
  no longer returns it, and verifies activation state is cleared. The same
  populated runtime also calls the MCP `forget` tool against another explicit
  native entity and verifies search plus activation are cleared.
- Extended populated native parity to explicit feedback. The native REST test
  now calls `/api/knowledge/feedback` on an explicit `native_brain` entity and
  verifies a `PREFERS` edge from `UserPreference`; the MCP `feedback` tool does
  the same for an `AVOIDS` edge. This exposed and fixed a publisher mismatch:
  `record_explicit_feedback()` now supports the real group-scoped sync
  `EventBus.publish()` and the older async test-double shape.
- Extended populated native parity to identity-core mutation. MCP
  `mark_identity_core` now marks and unmarks an explicit `native_brain` entity
  and reads the flag back from the active PyO3 Helix graph.
- Extended populated native parity to context assembly. The native REST test now
  calls `/api/knowledge/context` with a topic hint and verifies camelCase
  context counts/token estimates; the same runtime calls the MCP `get_context`
  tool and verifies the raw manager shape over the active `native_brain`.
- Extended populated native parity to MCP notification piggybacking. The native
  MCP test now temporarily enables notification surfacing, seeds `native_brain`
  and wrong-group notifications, calls `get_context`, and verifies only the active
  PyO3 brain notification appears in `memory_notifications`.
- Extended populated native parity to MCP auto-recall piggybacking. The native
  MCP test now creates an explicit `native_brain` anchor entity, temporarily
  enables tool-call auto-recall, calls `search_entities`, and verifies the
  response attaches `recalled_context` from `recall_lite` over the active PyO3
  graph.
- Updated MCP prompt guidance to name the shared brain-loop contract. The
  system prompt now explicitly describes `Capture -> Cue -> Project -> Recall
  -> Consolidate`, tying `observe`, `remember`, `recall`, and consolidation to
  the same lifecycle language used by REST, tests, docs, and dashboard surfaces.
- Extended populated native parity to direct entity/fact lookup. The native REST
  test now creates explicit `native_brain` entities plus a `USES` relationship,
  verifies `/api/entities/search` and `/api/knowledge/facts`, then the same
  runtime verifies MCP `search_entities` and `search_facts` against a second
  explicit native relationship. REST entity detail and neighbor endpoints, plus
  MCP `get_graph_state` and entity profile/neighbor resources, now verify the
  same explicit native relationship from the graph-detail surfaces.
- Extended populated native parity to REST entity mutation. The native REST test
  now patches a standalone TestMemory entity through `/api/entities/{id}`, reads
  the updated entity back from PyO3 Helix, soft-deletes it through the same REST
  entity route, and verifies detail/search plus activation no longer surface it.
- Extended populated native parity to MCP write tools. After the stable
  lifecycle/evaluation/recall assertions, the native test now calls MCP
  `remember` and `observe`, verifies their shared lifecycle write contract, and
  reads the created episodes plus observe cue back from the active PyO3 Helix
  graph under `native_brain`.
- Extended populated native parity to REST text observe. REST
  `/api/knowledge/observe` now verifies the Capture -> Cue lifecycle write
  contract, conversation date preservation, and episode/cue persistence in the
  active PyO3 Helix graph under `native_brain`.
- Extended populated native parity to REST auto-observe. REST
  `/api/knowledge/auto-observe` now verifies hook-style Capture -> Cue writes,
  session/conversation metadata, cue persistence, and duplicate-content
  `dedup_skipped` lifecycle semantics in the active PyO3 Helix graph under
  `native_brain`.
- Extended populated native parity to attachment capture. REST
  `/api/knowledge/observe-image` and `/api/knowledge/observe-file`, plus MCP
  `observe_image` and `observe_file`, now verify image/file lifecycle metadata
  and read the persisted attachment episodes plus cues back from the active
  PyO3 Helix graph under `native_brain`.
- Hardened native artifact search for no-embedding/no-BM25 misses. Artifact
  search now adds a bounded lexical fallback over bootstrapped Artifact
  entities, and Helix entity reads now tolerate the JSON-string attribute
  updates produced by existing GraphManager paths instead of returning
  `attributes=None` after artifact episode metadata updates.
- Fixed recall episode context for non-default native brains by passing the
  active `group_id` into `GraphManager.recall()` calls to
  `get_episode_entities()`.
- Extracted raw recall result assembly into
  `server/engram/retrieval/result_builder.py`. `GraphManager.recall()` still
  owns retrieval orchestration, cue feedback, traversal, and side effects, but
  the raw `entity`, `episode`, and `cue_episode` dictionaries now come from a
  retrieval-side builder with focused tests.
- Extracted recall episode expansion into
  `server/engram/retrieval/episode_traversal.py`. `GraphManager.recall()` still
  sequences the Recall stage, but entity-linked episode traversal and temporal
  contiguity expansion now live behind a retrieval-side service that preserves
  active `group_id`, duplicate filtering, merged-episode filtering, and
  synthetic episode score metadata.
- Extracted near-miss lookup and payload formatting into
  `server/engram/retrieval/near_miss.py`. `GraphManager.recall()` still records
  cue near-miss feedback through `_record_cue_hit()`, but entity near-miss
  formatting, cue/episode eligibility checks, merged-episode filtering, and
  near-miss payload construction now live in a retrieval-side helper.
- Extracted retrieval priming updates into
  `server/engram/retrieval/priming.py`. `GraphManager.recall()` still sequences
  the Recall stage and owns the priming buffer, but one-hop neighbor lookup,
  top-N entity filtering, TTL calculation, and boost writes now live in a
  retrieval-side updater.
- Wrapped relevance-confidence application in
  `server/engram/retrieval/confidence.py` via `RecallConfidenceApplier`.
  `GraphManager.recall()` still decides when the Recall stage reaches
  confidence scoring, but the feature flag, empty-result guard, error isolation,
  and call into `apply_relevance_confidence()` now live with the confidence
  code.
- Extracted recall conversation fingerprint recording into
  `server/engram/retrieval/context.py` via
  `RecallConversationFingerprintRecorder`. `GraphManager.recall()` still
  sequences when recall-query context is recorded, but the feature flag,
  no-context guard, provider embedding adapter, non-fingerprinting ingest mode,
  and `recall_query:<source>` tagging now live with the conversation retrieval
  utilities.
- Extracted Recall-stage working-memory writes into
  `server/engram/retrieval/working_memory.py` via
  `RecallWorkingMemoryUpdater`. `GraphManager.recall()` still sequences recalled
  entity/episode/query updates, but the disabled-buffer no-op and write-through
  policy now live beside the working-memory buffer instead of being inline in
  the facade.
- Extracted Recall-stage entity interaction telemetry into
  `server/engram/retrieval/feedback.py` via `RecallInteractionRecorder`.
  `GraphManager.recall()` still decides when entity results have an
  interaction, but publishing `recall.interaction` events and recording
  recall-need interaction samples now live in the retrieval feedback module.
- Extracted true Recall-stage entity access recording into
  `server/engram/retrieval/feedback.py` via `RecallEntityAccessRecorder`.
  `GraphManager` still decides when access should count, but activation access
  writes, `activation.access` event payloads, and labile reconsolidation-window
  marking now live in the retrieval feedback module.
- Extracted Recall-stage cue feedback into
  `server/engram/retrieval/feedback.py` via `RecallCueFeedbackRecorder`.
  `GraphManager` still decides when a cue hit or cue near-miss should be
  recorded, but cue counters, cue hit/near-miss/policy/promotion events,
  recall-need cue samples, and hot-cue projection scheduling now live in the
  retrieval feedback module and use the shared projection-state sync helper.
- Extracted explicit post-response memory feedback into
  `server/engram/retrieval/feedback.py` via `RecallMemoryInteractionApplier`.
  `GraphManager.apply_memory_interaction()` remains the compatibility facade,
  but dedupe, cue feedback dispatch, entity access, Thompson-sampling
  positive/negative feedback, recall interaction events, and recall-need
  interaction samples now live in the retrieval feedback module.
- Extracted Recall current-state result selection into
  `server/engram/retrieval/result_selection.py`. `GraphManager.recall()` still
  sequences final result shaping, but the rule that current/currently/now
  queries prefer entity state over raw episode/cue history now has direct tests
  and a retrieval-side helper.
- Extracted Recall request policy into `server/engram/retrieval/request_policy.py`.
  `GraphManager.recall()` still calls the retrieval pipeline, but near-miss
  fetch-window sizing, primary/near-miss result splitting, and ranking-feedback
  learning decisions for passive vs true-usage interactions now live in pure
  retrieval helpers with direct tests.
- Extracted primary Recall result materialization into
  `server/engram/retrieval/primary_results.py` via
  `RecallPrimaryResultMaterializer`. `GraphManager.recall()` still sequences
  retrieval and post-processing, but primary entity/episode/cue result assembly,
  merged-episode filtering, working-memory writes, entity access recording, cue
  feedback dispatch, relationship fetches, and entity interaction telemetry now
  live behind a retrieval-side materialization boundary.
- Extracted Recall near-miss materialization into
  `server/engram/retrieval/near_miss.py` via `RecallNearMissMaterializer`.
  `GraphManager.recall()` still stores `_last_near_misses` for downstream
  surfaces, but entity near-miss formatting, cue near-miss context lookup,
  cue near-miss feedback, refreshed cue payload lookup, and near-miss payload
  assembly now live behind the retrieval near-miss boundary.
- Extracted Recall post-processing into
  `server/engram/retrieval/post_process.py` via `RecallPostProcessor`.
  `GraphManager.recall()` now delegates the post-primary tail for
  entity-linked episode expansion, temporal expansion, current-state filtering,
  query working-memory writes, retrieval priming, near-miss materialization,
  relevance-confidence application, and recall-query fingerprint recording.
- Extracted Recall-stage orchestration into
  `server/engram/retrieval/service.py` via `RecallService`.
  `GraphManager.recall()` is now a compatibility facade over the service, which
  owns request policy, retrieval pipeline invocation, primary/near-miss
  splitting, primary result materialization, post-processing, and the final
  near-miss/result return contract.
- Added an opt-in dashboard native smoke harness at
  `dashboard/src/test/nativeDashboardSmoke.test.tsx`. It is skipped by default,
  but when `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1` and `VITE_API_URL` point at a
  populated native REST server, it fetches lifecycle, evaluation, and recall
  data through the dashboard API client and renders the Lifecycle panel from
  that live native contract.
- Added a no-bind native dashboard fixture smoke to the same test file. The
  default lane now exercises dashboard API-client normalization for native-shaped
  lifecycle, evaluation, recall, episode listing, and consolidation
  status/history payloads, then renders the Brain Loop Lifecycle panel,
  Consolidation panel, and Memory Feed without requiring a local REST bind. The
  fixture asserts native episode cue/projection-state preservation so Capture
  and Cue do not regress while the live native REST smoke remains opt-in.
  The live native REST smoke also passes when pointed at an approved local
  native REST bind.
- Re-ran the live dashboard native smoke after the adjudication-pressure
  evaluation contract. The seeded PyO3 REST server still feeds lifecycle,
  evaluation, and recall into the dashboard API client and Lifecycle panel.
- Quieted optional native graph-embedding vector misses. `NativeTransport` now
  treats Helix native's missing-HNSW-entrypoint response like an empty result,
  so recall can fall back cleanly when graph embeddings have not been trained
  without logging error noise during the preferred PyO3 path.
- Re-ran the broad backend and dashboard gates after the adjudication-pressure
  and native-transport fallback changes. The non-Docker/non-Helix backend suite
  remains green, and the full dashboard tests/build still pass with only the
  existing warning classes.
- Added repeated native REST reopen coverage. The native surface parity test now
  seeds a populated PyO3 brain once, then starts and stops the FastAPI app three
  times against the same native data directory, verifying lifecycle,
  evaluation, and recall on every reopen.
- Added bounded native write/read load coverage. The native surface parity test
  now writes five additional memories through the public REST `remember` path,
  runs repeated recall queries against the populated PyO3 brain, and verifies
  lifecycle/evaluation totals after the load without binding a local socket.
- Extended that bounded native load coverage to the REST offline replay path.
  The same populated PyO3 test now replays one queued offline entry carrying a
  wrong `group_id`, verifies the endpoint stores it under the active native
  brain, and treats replayed entries as captured/cued rather than immediately
  projected.
- Added a larger multi-batch native ASGI soak. The native surface parity suite
  now writes 12 additional memories across three `remember` batches, checks
  lifecycle totals and recall after each batch, shuts the runtime down, reopens
  the same PyO3 data directory, and verifies lifecycle/evaluation/recall remain
  coherent for the expanded `native_brain`.
- Added an operator-style native load smoke to `engram evaluate --smoke`.
  `--smoke-load-count N` adds deterministic load episodes before projection,
  and `--smoke-recall-rounds N` runs post-projection recall checks. This gives
  the preferred PyO3 path a repeatable command outside pytest for checking
  larger Capture -> Cue -> Project -> Recall -> Consolidate runs.
- Extended that operator smoke with a duration-based recall soak.
  `--smoke-min-duration-seconds N` keeps recall active against the populated
  PyO3 brain after projection, `--smoke-pause-seconds` optionally spaces the
  loops, and the JSON report records duration recall checks plus elapsed time.
- Verified an actual hour-scale native recall soak with that operator command.
  The PyO3 run used disposable native data and label storage, held Recall active
  for 3600.599 seconds, completed 10,362 sustained recall checks, and returned
  no coverage gaps.
- Hardened that operator smoke across the triage batch boundary. A 120-load
  native smoke exposed that one triage-only cycle projected 100 of 123 episodes;
  the smoke runner now keeps running deterministic triage cycles until the
  expected projected count is reached and records `smoke.cycle_count` plus
  `smoke.cycle_ids`.
- Recovered the full non-Docker/non-Helix backend gate after the native
  `group_id` recall fix exposed a SQLite/Falkor graph-store contract mismatch.
  `get_episode_entities()` now accepts optional `group_id` across the protocol,
  SQLite, FalkorDB, and Helix implementations; the shared storage contract checks
  both unscoped and group-scoped calls.
- Closed a REST offline replay queue group-boundary leak. `/api/knowledge/replay-queue`
  now always replays drained entries into the current request tenant group
  instead of trusting any stale or injected `group_id` carried by the queued
  entry payload.
- Added non-default group regression coverage for project bootstrap artifact
  writes. Project paths remain topology/context inside a brain: Project and
  Artifact entities, bootstrap episodes, and cue-only projection-state sync all
  stay in the active `group_id`.
- Added a projection freshness signal to the P3 evaluation path. Helix native
  graph stats now compute average capture-to-projection lag from episode
  timestamps instead of returning `0.0`, and the shared brain-loop Markdown plus
  dashboard Evaluate card surface projection latency and processing duration
  beside yield/failure metrics.
- Added a projection backlog signal to the same Project-stage evaluation
  contract. The shared report now derives tracked count, projected rate, and
  backlog rate from projection state counts, and the dashboard Evaluate card
  surfaces backlog pressure beside projected count, failure rate, latency, and
  processing duration.
- Added a Recall-stage gate-latency signal to the P3 evaluation contract. The
  shared brain-loop report now carries recall-need analyzer and graph-probe
  average/p95 latency from the existing runtime controller metrics, the
  Markdown report prints analyzer/probe p95, and the dashboard Evaluate card
  surfaces both p95 timings beside labeled recall quality.
- Added a Recall-stage gate-control signal to the same evaluation contract. The
  shared report now carries runtime surfaced/used/dismissed interaction counts,
  selected/confirmed/corrected counts, graph override count, adaptive-threshold
  enabled state, and active recall thresholds. Markdown prints the control
  summary, and the dashboard Evaluate view now has a Recall Gate card for
  trigger volume, runtime use, graph lift/probe trigger, graph overrides, and
  resonance threshold posture.
- Tightened `engram evaluate --smoke` so the deterministic projected/
  consolidated smoke now runs one real recall-gate check, records memory-need
  analysis plus surfaced recall feedback, and builds its report from
  `GraphManager.get_graph_state()` instead of raw graph-store stats. The smoke
  report now proves Recall gate latency/control fields can be populated by the
  runtime, not only by synthetic report fixtures.
- Verified that same recall-gate smoke path against Helix native PyO3. The
  disposable native run produced `recall.total_analyses=1`,
  `recall.trigger_count=1`, analyzer p95 latency, `family_contributions.keyword=1`,
  `control.surfaced_count=1`, `smoke.gate_recall_checks=1`, and no coverage
  gaps for `native_brain`.
- Hardened the projected/consolidated smoke verifier around those Recall gate
  metrics. `assert_smoke_report()` now fails if the report lacks gate analysis,
  a trigger, analyzer latency, surfaced recall feedback, or the smoke
  `gate_recall_checks` counter, and the smoke footer prints `gate_checks`.
- Made the P3 evaluation coverage gaps distinguish labeled recall samples from
  actual Recall Gate runtime coverage. Reports now flag `recall gate needs
  runtime analyses` when labels exist but no gate analyses were captured, and
  flag `recall gate latency needs analyzer samples` when gate analysis exists
  without analyzer latency.
- Added structured P3 evaluation signal readiness to the shared report contract.
  `evaluation_signals` now records status, evidence count, current metric, and
  a gap for cue usefulness, projection yield, recall quality, false recall,
  triage calibration, and consolidation effect. REST exposes the map through the
  brain-loop report, MCP returns it through `get_evaluation_report`, the
  Markdown report prints it, and the dashboard Evaluate panel renders a Signal
  Readiness section so missing evidence is visible before any production-grade
  evaluation claim. Fast MCP coverage and populated native PyO3 MCP parity now
  assert the six-signal map is measured when matching evidence is present.
- Tightened the projected/consolidated smoke verifier around that readiness map.
  `assert_smoke_report()` now fails if any required evaluation signal is missing,
  not measured, evidence-free, or metric-free, so smoke success is tied to the
  same cue/projection/recall/calibration/consolidation evidence that the
  operator report displays.
- Added an operator CLI hard gate for the same readiness map. `engram evaluate
  --require-evaluation-signals` now exits non-zero for normal JSON/live reports
  when any required signal is missing, unmeasured, evidence-free, or metric-free,
  so benchmark-labeled exports and reopened native reports can be promoted to
  blocking checks without running the deterministic smoke harness.
  `--min-evaluation-signal-evidence N` now lets release/benchmark gates require
  more than one smoke-sized evidence record per measured signal.
  `--benchmark-artifact results.json --require-benchmark-evidence` now attaches
  deterministic showcase benchmark evidence to the report and fails if the
  `engram_full` baseline lacks enough scenarios, pass rate, fairness contract,
  or transcript hashes. The Markdown brain-loop report now includes a Benchmark
  Evidence section when this artifact is attached.
  `--evidence-bundle brain-loop-evidence.json` writes a single JSON artifact
  with the report, attached benchmark evidence, source paths, and gate thresholds
  after the requested gates pass.
  Current disposable proof: generated
  `/private/tmp/engram-evidence-showcase/results.json` with the deterministic
  showcase benchmark for `temporal_override`, then ran `uv run engram evaluate
  --smoke --require-evaluation-signals --benchmark-artifact
  /private/tmp/engram-evidence-showcase/results.json
  --require-benchmark-evidence --min-benchmark-scenarios 1
  --min-benchmark-pass-rate 1.0 --evidence-bundle
  /private/tmp/engram-brain-loop-evidence.json --format json`. It passed and
  wrote an `engram_brain_loop_evidence_bundle` with all six evaluation signals
  measured and benchmark evidence attached. Treat this as proof of the packaging
  path, not as final production evidence.
  Follow-up quick benchmark proof: generated
  `/private/tmp/engram-evidence-showcase-quick-20260518/results.json` with
  deterministic showcase `quick` mode, seed `7`, and only the `engram_full`
  baseline, then ran `uv run engram evaluate --smoke
  --require-evaluation-signals --benchmark-artifact
  /private/tmp/engram-evidence-showcase-quick-20260518/results.json
  --require-benchmark-evidence --min-benchmark-scenarios 4
  --min-benchmark-pass-rate 1.0 --evidence-bundle
  /private/tmp/engram-brain-loop-evidence-quick-20260518.json --format json`.
  It passed with benchmark status `measured`, four available quick scenarios,
  pass rate `1.0`, false recall `0.0`, four transcript hashes, an `engram_full`
  fairness contract, and all six evaluation signals measured. Treat this as a
  stronger local deterministic gate, not as final production evidence.
  Full deterministic benchmark proof: generated
  `/private/tmp/engram-evidence-showcase-full-20260518/results.json` with
  deterministic showcase `full` mode, seeds `7, 19, 31`, and only the
  `engram_full` baseline across all 13 scenario transcripts, then ran
  `uv run engram evaluate --smoke --require-evaluation-signals
  --benchmark-artifact /private/tmp/engram-evidence-showcase-full-20260518/results.json
  --require-benchmark-evidence --min-benchmark-scenarios 39
  --min-benchmark-pass-rate 1.0 --evidence-bundle
  /private/tmp/engram-brain-loop-evidence-full-20260518.json --format json`.
  It passed with benchmark status `measured`, 39 available scenario runs, 39
  passed, pass rate `1.0`, false recall `0.0`, 13 transcript hashes, an
  `engram_full` fairness contract, and all six evaluation signals measured.
  Treat this as the strongest local benchmark-labeled gate so far; broader live
  AI-harness adoption remains release-hardening evidence, not the core
  brain-loop completion blocker.
  Latest broad backend gate after the adoption-template and full-benchmark
  evidence updates: `uv run pytest -m "not requires_docker and not requires_helix" -q`
  passed with 3320 tests, 43 skips, and 236 deselections in 192.79s.
  `--from-json` also recognizes already-rendered brain-loop report artifacts, so
  saved JSON reports can be verified directly instead of being misread as raw
  graph stats. Partial report-shaped JSON now fails fast with the missing
  required report sections instead of being silently rebuilt as an empty raw
  stats report. The report-artifact shape helpers now live in
  `engram.evaluation.brain_loop_report`, not in the CLI, so future REST/MCP or
  benchmark tooling can share the same artifact contract. The partial-report
  helper returns false for complete report artifacts, so future callers can use
  it directly without depending on CLI check ordering. The shared verifier,
  report-artifact helpers, and failure formatter are also exported from
  `engram.evaluation` for package-level reuse.
- Registered `engram evaluate --mode helix --require-evaluation-signals` in the
  native surface manifest as an operator hard gate, with manifest coverage
  proving the gate remains tracked beside the native smoke and doctor paths.
- Updated the Helix native install guide with the same hard gate so the
  preferred PyO3 operator path documents smoke, lifecycle/doctor, and measured
  evaluation-readiness verification together.
- Updated the lite install guide with the same `engram evaluate` report path and
  a warning that `--require-evaluation-signals` is only appropriate after a lite
  brain has cue feedback, projection yield, recall labels, triage calibration,
  and consolidation history. This keeps SQLite/lite viable as the dev/demo path
  without implying a fresh disposable brain can satisfy production-readiness
  gates.
- Tightened the shared signal verifier itself. `unmeasured_evaluation_signals()`
  now reports missing signals in the same lifecycle order as
  `EVALUATION_SIGNAL_ORDER`, and direct report-helper tests cover measured,
  missing, not-measured, no-evidence, and no-metric outcomes instead of relying
  only on smoke/CLI coverage. Smoke and CLI now share
  `evaluation_signal_failure_message()` so their hard-gate failure formatting
  cannot drift independently.
- Updated the no-bind native dashboard smoke fixture to carry measured
  `evaluation_signals` and cue/calibration/consolidation effect evidence. The
  fixture now asserts the dashboard Evaluate panel renders Signal Readiness as
  `6/6 measured` instead of silently defaulting missing native-shaped fields to
  `needs_data`.
- Persisted the latest Recall Gate runtime metrics snapshot through the local
  evaluation store and taught CLI/REST/MCP reports to merge that snapshot only
  when current in-memory stats are weaker. The PyO3 native smoke still proves
  Recall Gate fields in-process; reopened native live reports now recover
  analyzer latency, trigger counts, and surfaced feedback from the persisted
  snapshot instead of losing runtime coverage after process reopen. Runtime
  metric snapshots are pruned to the latest 25 per group so repeated report
  reads do not grow the local evaluation DB without bound. `SQLiteEvaluationStore`
  now also tracks whether it owns its SQLite connection, so lite-mode stores
  initialized with the graph store's borrowed connection do not close the shared
  graph DB during report cleanup.
- Matched that borrowed-connection ownership guard in `SQLiteConsolidationStore`
  so lite-mode consolidation history shares the graph store's SQLite connection
  without closing it during consolidation/report cleanup.
- Added the same ownership-aware close behavior to `SQLiteConversationStore`.
  Standalone SQLite conversation stores now close their own connection, while
  lite-mode stores initialized with the graph store's borrowed connection leave
  the shared graph DB open for normal runtime shutdown.
- Added the same borrowed-connection ownership guard to `SQLiteFeedbackStore`
  so implicit feedback storage can share a lite-mode graph DB connection without
  owning shutdown of that connection.
- Added explicit borrowed-connection regression coverage for `SQLiteAtlasStore`,
  which already followed the ownership rule, so materialized atlas snapshots are
  now covered by the same lite-mode shared-DB contract.
- Made lite search storage ownership-aware. `FTS5SearchIndex` and
  `SQLiteVectorStore` now close owned standalone connections while preserving
  borrowed graph DB connections, and `HybridSearchIndex.close()` delegates to
  its FTS/vector/provider components instead of being a no-op.
- Added a shared borrowed-connection contract test across the lite/evaluation
  SQLite stores that accept caller-owned DB handles. The guard covers atlas,
  consolidation, conversation, evaluation, feedback, FTS, and vector storage so
  future close-path changes fail before they can break the shared lite runtime
  connection.
- Added `server/engram/storage/bootstrap.py` as the shared runtime
  initialization contract for companion stores that can borrow the lite graph
  DB connection. It now also owns atlas, consolidation, evaluation, and
  conversation store creation for REST startup plus consolidation/evaluation
  store creation for MCP startup, lifecycle CLI, evaluation CLI, and
  projected/consolidated smoke. Those entrypoints share the lite borrowed-DB and
  Helix shared-client paths instead of repeating `graph_store._db` and
  `graph_store._helix_client` checks.
  The same module now owns borrowed in-memory consolidation-store creation for
  fallback readers, so MCP trigger resolution, lifecycle summary fallback reads,
  and graph-health SQLite metrics use one helper instead of local private DB
  probes.
- Removed app-state reads from shared runtime helpers outside the dependency
  layer. `server/engram/notifications/surface.py` is now a pure presenter for
  MCP notification piggybacking, `server/engram/api/deps.py` owns optional
  notification-service construction from app state, and
  `ConsolidationScheduler` receives its graph-store dependency explicitly for
  temporal scans. Public-surface guards now reject the old
  `build_mcp_notifications_surface_from_state()` path and assert
  `notifications.surface` plus `consolidation/scheduler.py` do not read
  `_app_state` directly.
- Updated the projected/consolidated smoke's cue-feedback path to record
  surfaced cue feedback through `GraphManager.apply_memory_interaction()`.
  The smoke no longer uses private manager graph or cue-hit methods while
  proving Recall feedback can be written back into the Cue stage.
- Added the shared close helper and `GraphManager.close_runtime_resources()` so
  MCP shutdown closes owned search, activation, and graph resources through the
  manager facade instead of reaching into private manager fields.
- Extended that runtime shutdown boundary to REST startup-owned resources.
  `server/engram/main.py` now stops subscriber/worker/pressure/scheduler
  resources through `stop_if_supported()`, closes companion stores and
  aclose-only clients through `close_if_supported()`, then closes search,
  activation, and graph stores through `GraphManager.close_runtime_resources()`
  when the manager is available. Direct store closes remain only as a
  startup-failure fallback.
  MCP shutdown now uses the same shared close helper for its Redis publisher,
  evaluation store, and consolidation store, and the same stop helper for its
  episode worker.
  `tests/test_public_surface_presenter_boundaries.py` now statically guards
  both shutdown paths against drifting away from the shared stop/close boundary.
- Moved shutdown consolidation orchestration out of `server/engram/main.py`.
  `run_shutdown_consolidation()` in `server/engram/consolidation_trigger.py`
  now owns the cancel/running-engine, disabled-config, and final
  `trigger="shutdown"` cycle decision. The static public-surface guard rejects
  `is_running`, `cancel`, and `run_cycle` from `main._shutdown()`. Focused
  helper coverage now also proves failed shutdown cycles are logged and do not
  break shutdown, and `main._shutdown()` delegates the active engine/config/logger
  into the helper.
- Moved REST knowledge-chat SSE stream orchestration out of
  `server/engram/api/knowledge.py`. `stream_api_chat_sse_events()` in
  `server/engram/retrieval/chat_runtime.py` now owns AI SDK SSE start/step/text/
  finish/error framing, Anthropic client construction, response-turn execution,
  and best-effort conversation persistence scheduling. The public-surface guard
  rejects returning `_sse`, inline event-stream generators, direct Anthropic
  construction, `run_chat_response_turn()`, and `schedule_chat_turn_persistence()`
  to the REST route.
- Moved the remaining REST knowledge-chat response setup out of the route body.
  `build_api_chat_stream_response_surface()` in
  `server/engram/retrieval/chat_runtime.py` now owns chat rate-limit handling,
  optional conversation-store resolution, conversation not-found payloads,
  session entity lookup, and SSE stream construction. `chat()` now keeps tenant
  lookup, dependency lookup, and HTTP response wrapping, and the public-surface
  guard forbids reintroducing direct chat rate-limit, conversation resolution,
  not-found payload, or SSE stream assembly in the FastAPI route.
- Moved `/health` dependency fallback out of the public handler.
  `build_api_health_response()` in `server/engram/api/health_runtime.py` now owns
  optional graph-store resolution plus the default-group fallback before calling
  the shared health surface. `health_check()` is now a one-call route wrapper,
  and the public-surface guard rejects direct `get_graph_store()`,
  `get_config()`, `get_mode()`, `get_stats()`, or `build_api_health_surface()`
  usage in the route.
- Moved REST consolidation trigger background scheduling out of the route body.
  `build_api_consolidation_trigger_response_surface()` now owns the running
  check, public trigger payload, and `BackgroundTasks.add_task()` scheduling for
  `run_api_consolidation_cycle()`. `trigger_consolidation()` keeps tenant and
  engine lookup plus response wrapping, and the public-surface guard rejects
  direct task scheduling or direct trigger/run helper usage in the route.
- Moved REST consolidation status pressure/config selection out of the route
  body. `build_api_consolidation_status_response_surface()` now owns the
  activation-config selection for pressure reporting before calling the shared
  status surface. `consolidation_status()` keeps tenant/dependency lookup plus
  JSON wrapping, and the public-surface guard rejects direct activation-config
  extraction or direct status-surface calls in the route.
- Added a route-control-flow guard for the remaining public REST/WebSocket
  handlers. Atlas warning logging and response wrapping now live in
  `build_api_atlas_json_response()`, so the only allowed route branches now are
  chat JSON-vs-stream response wrapping and WebSocket auth/session
  try/excepts. Any new `if`/`try`/loop/match-style control flow inside a
  decorated FastAPI route must be explicitly justified in the guard or moved
  into a route-facing helper.
- Tightened the MCP adoption contract after a live agent ignored Engram despite
  the MCP server being connected. `ENGRAM_SYSTEM_PROMPT` now explicitly says
  Engram is the portable, cross-context source of truth for user facts,
  preferences, corrections, durable decisions, relationships, goals,
  commitments, and long-tail recall; project-local files own only repo-specific
  conventions and current-task scratch notes. It also tells agents that an empty
  runtime (`artifactCount: 0`, `lastObservedAt: null`, zero recall/evaluation
  stats) is an onboarding state and should trigger `bootstrap_project(project_path)`
  when a project path is available, not a reason to route around Engram. README
  automatic-memory behavior and prompt tests now cover that authority/onboarding
  contract.
- Added the explicit MCP `claim_authority(project_path)` surface for agents that
  need a callable source-of-truth decision. It returns a machine-readable
  authority contract, Engram-owned vs project-local memory responsibilities,
  onboarding state, recommended `bootstrap_project`/`get_context`/`recall`
  actions, the current runtime-state payload, and the brain-loop lifecycle. The
  tool is registered in the native surface manifest, covered by focused
  project-runtime/MCP-wrapper/prompt/static/native-manifest tests, and listed in
  README as the 27th MCP tool.
- Extended `claim_authority()` into a deterministic adoption harness. The tool
  now accepts `user_message` and `file_memory_present`, then returns an
  `agent_protocol` with `required_tools_before_answer` plus a capture routing
  decision. The covered failure case is an agent with file memory visible and an
  empty Engram runtime: the payload requires `bootstrap_project`, `get_context`,
  and `recall` before answering, and routes high-signal cross-context facts to
  Engram `remember` instead of treating file-local memory as a substitute.
- Added `validate_agent_protocol_calls()` so real MCP client logs or thin
  harnesses can prove they followed the returned `agent_protocol`. It validates
  required pre-answer tool order, required Engram capture calls, unexpected
  Engram writes for project-local scratch, and the specific failure where
  visible file memory substitutes for Engram.
- Added a real stdio MCP-client adoption check in
  `tests/test_mcp_authority_client_adoption.py`. It starts `engram mcp` in
  isolated lite/noop mode, calls `claim_authority(project_path, user_message,
  file_memory_present=True)`, executes the returned bootstrap/context/recall
  and capture tools, then scores the observed transcript with
  `validate_agent_protocol_calls()`. That live path exposed a contract gap:
  missing project artifacts were reported as `ready` when other runtime metrics
  existed. `_build_onboarding()` now treats missing or stale project artifacts
  as `needs_project_bootstrap` and requires `bootstrap_project` before judging
  recall usefulness.
- Strengthened the real-harness installation guidance. The MCP system prompt
  now tells agents to follow `agent_protocol.required_tools_before_answer` in
  order and use the returned `capture` decision after `claim_authority()`.
  `engram setup` prints an "Agent adoption checklist" for Claude Code, Cursor,
  Windsurf, and similar MCP clients, and README's Claude Code guidance now
  includes the same authority/protocol/bootstrap/capture contract instead of
  only saying to call `get_context()`.
- Added a reproducible adoption transcript verifier. `engram adoption
  --authority claim-authority.json --calls mcp-calls.jsonl` loads a saved
  `claim_authority()` response plus JSON/JSONL tool-call records, runs
  `validate_agent_protocol_calls()`, and exits nonzero when a real client skips
  required pre-answer tools, substitutes file-local memory for Engram, misses
  required capture, or writes to Engram for project-local scratch. README and
  `engram setup` now point real harnesses to this command.
- Made `claim_authority()` self-describing for real harness validation. Its
  `agent_protocol` now includes a `verification` block with the `engram adoption`
  command, required JSONL transcript fields (`phase`, `tool`), allowed phase
  values, and an example transcript derived from the returned capture decision.
  This lets Claude/Cursor/Windsurf-style harnesses record the right evidence
  without reading README first.
- Added a stricter live-harness evidence mode for the adoption verifier.
  `claim_authority().agent_protocol.verification` now includes
  `live_evidence_command` and a JSON wrapper schema with required `client` and
  `capturedAt` metadata. `engram adoption --require-live-evidence` fails with
  `missing_live_harness_evidence` if a passing tool-name transcript lacks that
  metadata, or if those fields are still template placeholders, so handcrafted
  JSONL or an untouched generated template cannot be mistaken for current
  Claude/Cursor/Windsurf completion evidence.
- Added a live-adoption transcript template mode. `engram adoption --authority
  claim-authority.json --template` reads the saved authority payload, emits the
  expected JSON wrapper and call sequence, and supports Markdown output for
  operator copy/paste. This keeps real harness collection aligned with the
  returned `agent_protocol` without making the template itself count as evidence.
- Hardened `engram adoption` for common real MCP log shapes. The verifier now
  normalizes prefixed tool names such as `mcp__engram__recall`, nested `tool`,
  `function`, `tool_call`, and `toolCall` records, and `stage` as an alias for
  `phase`, while keeping the actual required phase/tool contract unchanged. It
  also accepts Claude Code `--output-format stream-json` logs directly by
  extracting Engram `tool_use` blocks, mapping `observe`/`remember` to
  `capture`, and mapping the other Engram tools to `before_answer`. It also
  accepts explicit plaintext/Markdown harness notes with
  `before_answer`/`capture` headings plus common `Before answer`/`pre-answer`
  aliases and Engram tool lines, so copied Claude, Cursor, or Windsurf session
  notes can be checked without hand-converting them to JSON first. Malformed
  copied notes now return a structured `invalid_calls_transcript` report instead
  of surfacing a parser exception. The command also accepts `--calls -` to read
  copied transcript notes from stdin. If copied chat notes do not expose raw
  tool logs but the agent explicitly says it ignored Engram or used file-local
  memory as the primary memory path, the verifier now classifies the notes as a
  failed adoption transcript instead of rejecting them as unusable.
- Fixed `claim_authority()` verifier metadata for project-local scratch. The
  returned `agent_protocol.verification` now includes `capture_required`, and
  the example transcript only includes a `capture` record when the protocol
  actually requires an Engram `observe` or `remember` call. Project-local
  scratch examples no longer emit a fake `"<none>"` capture tool.
- Tightened the public route-local orchestration guard. The static
  public-surface test now discovers every decorated REST API route and fails if
  any route is missing from `PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES`, so new
  transport handlers cannot bypass the named boundary map silently.
- Moved dashboard WebSocket session orchestration out of route-local nested
  functions. `run_dashboard_websocket_session()` in
  `server/engram/api/websocket_runtime.py` now owns event forwarding, command
  handling, activation-monitor task lifecycle, bus subscription cleanup, and
  WebSocket disconnect cancellation cleanup. `dashboard_ws()` keeps auth,
  accept, dependency lookup, and one route-facing runtime call. The
  public-surface guard now also fails when decorated API route handlers define
  nested functions, so new transport handlers cannot hide lifecycle loops inside
  a route body.
- Moved dashboard WebSocket tenant authentication out of the route body.
  `resolve_dashboard_websocket_tenant()` in `server/engram/api/websocket_auth.py`
  now owns the configured `AuthConfig()` fallback, header resolution, and browser
  `?token=` bearer fallback before the socket is accepted. `dashboard_ws()` now
  delegates auth and rejection to route-facing helpers, and the public-surface
  guard forbids the route from reintroducing `resolve_tenant_from_scope`,
  `Headers`, or query-param auth parsing directly.
- Flattened the MCP recall middleware adapter. `_recall_middleware()` now passes
  the named `_ingest_live_tool_turn()` helper into `run_mcp_recall_middleware()`
  instead of defining a nested runtime callback in `mcp/server.py`. The static
  boundary tests now assert that `_recall_middleware()` delegates through the
  named helper and contains no nested runtime callbacks.
- Added the MCP public-surface counterpart to the route nested-orchestration
  guard. Decorated MCP tools/resources/prompts now fail the public-surface suite
  if they define nested functions inside the public handler body, keeping MCP
  handlers as lookup/JSON transport wrappers over named runtime surfaces.
- Added `server/engram/ingestion/worker_runtime.py` so `EpisodeWorker` receives
  explicit graph, activation, and search stores from REST/MCP startup instead of
  reaching through `GraphManager` private fields. `GraphManager` keeps
  `get_episode_worker_runtime_stores()` as a compatibility accessor for direct
  worker construction in tests and legacy internal callers.
- Added `server/engram/ingestion/worker_batching.py` so adjacent auto-capture
  turn merging, primary cue rebuild, and merged-away cue retirement live in a
  Cue-stage ingestion helper instead of `EpisodeWorker`. The worker now keeps
  queue consumption, deterministic scoring, and projection routing.
- Added `server/engram/ingestion/worker_scoring.py` so deterministic worker
  triage scoring, multi-signal scorer access, goal boost lookup, and
  projection-yield feedback live behind a scoring service. `EpisodeWorker`
  delegates scoring and outcome feedback while keeping event routing and
  Project-stage dispatch.
- Added `server/engram/ingestion/worker_routing.py` so duplicate projection
  guards, system-discourse cue-only skips, worker skip/defer projection-state
  sync, and the "project now" routing flag live outside the worker loop.
  `EpisodeWorker` now keeps event consumption, batch timing, and Project-stage
  dispatch.
- Updated the no-bind native dashboard smoke fixture with the same Recall gate
  payload shape emitted by the PyO3 smoke. The dashboard API client test path
  now verifies native-shaped analyzer latency, trigger count, surfaced recall
  feedback, and threshold mapping without requiring a bound local REST server.
  The same no-bind smoke now renders the Evaluation panel from that native-shaped
  report and checks the Recall Gate card.
- Added a consolidation effect-rate signal to the P3 evaluation contract. The
  shared report now derives overall and per-phase affected/processed rates, the
  Markdown report prints cycle effect percentage, and the dashboard Evaluate
  card surfaces consolidation effect beside cycles, affected count, errors,
  snapshots, accuracy, and ECE.
- Added an adjudication-pressure signal to the Consolidate evaluation contract.
  The shared report now summarizes evidence/edge adjudication phase runs,
  affected/unaffected counts, effect rate, and errors; Markdown prints that
  pressure beside consolidation totals, and the dashboard Evaluate card surfaces
  adjudication effect plus unaffected count.
- Tightened the Consolidate calibration-quality contract. Evaluation coverage
  gaps now distinguish saved calibration snapshots from quality-scored
  calibration evidence, so a report with unscored snapshots cannot claim the
  calibration-quality gate is covered. The nested calibration status now reports
  `needs_quality` for unscored snapshots instead of `measured`, and the
  Consolidate stage reports `attention` while calibration quality is incomplete
  or while completed cycles have no saved calibration snapshots. The Markdown
  report now prints `needs labeled decisions` beside `needs_quality` calibration
  snapshots so CLI/headless output matches the dashboard operator wording.
- Made `engram doctor` Markdown diagnostics list individual brain-loop smoke
  coverage gaps instead of only printing a count, so operator diagnostics expose
  the same concrete Recall/Consolidate gap names as the evaluation report.
- Expanded the static one-brain group-scope guard so every production Python
  module is checked for silent `group_id or "default"` narrowing, instead of
  relying on a curated REST/MCP/storage/retrieval/consolidation directory list.
- Carried that calibration-quality state into the dashboard Evaluate panel.
  When the backend reports `needs_quality`, the Calibration card now shows the
  missing labeled-decision state instead of rendering a misleading `0 labels /
  accuracy n/a / ECE n/a` summary, and the API client now has explicit coverage
  proving the backend `needs_quality` status survives normalization.
- Surfaced post-cycle finalization as part of the Consolidate lifecycle event
  contract. Successful cycles now pass the pinned-context refresh count through
  `consolidation.completed` under `payload.finalization.refreshedPinnedContexts`
  instead of discarding the finalizer result inside `ConsolidationEngine`.
- Extended native prospective-memory intention parity to hard deletes. The
  populated PyO3 native surface test now covers REST `DELETE
  /api/knowledge/intentions/{id}?hard=true` and MCP `dismiss_intention(...,
  hard=True)`, then verifies the deleted intentions no longer appear when
  listing disabled intentions for the active `native_brain`.
- Aligned REST prospective-memory creation/listing with the MCP
  refresh-context intention contract. REST `POST /api/knowledge/intentions` now
  accepts `refresh_trigger`, REST listing exposes `refreshTrigger`,
  `lastRefreshed`, and `hasPinnedResult` for pinned context intentions, and the
  populated PyO3 native surface test verifies REST and MCP
  `refresh_context`/`after_consolidation` intentions in `native_brain`.
- Tightened prospective-memory create acknowledgements across REST and MCP.
  REST now echoes `triggerType` and `refreshTrigger`; MCP now echoes
  `refresh_trigger`; and the populated native test verifies both activation and
  refresh-context create responses before listing the created intentions.
- Synced public prospective-memory guidance with that contract. README,
  `skills/engram-memory/SKILL.md`, and the MCP prompt now document
  `trigger_text`/`action_text` creation fields, refresh-context pinned queries,
  `refresh_trigger="after_consolidation"`, and the REST listing metadata
  (`refreshTrigger`, `lastRefreshed`, `hasPinnedResult`).

## Verified

- Backend lifecycle focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_cycle_context.py tests/test_triage_phase.py
  tests/test_rework_integration.py -q`
  - Result: 41 passed, 2 skipped.
- Backend lite gate:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2498 passed, 43 skipped, 236 deselected, with existing warnings.
- Backend lint on touched Python files:
  `uv run ruff check ...`
  - Result: passed.
- Dashboard consolidation test:
  `pnpm test -- --run src/test/ConsolidationPanel.test.tsx`
  - Result: 8 passed, with existing React `act(...)` warnings.
- Dashboard build:
  `pnpm run build`
  - Result: passed, with existing large chunk warning.
- Recall presenter/API/MCP focus:
  `uv run pytest tests/test_recall_presenter.py
  tests/test_knowledge_api.py::TestRecall
  tests/test_knowledge_api.py::TestChatRecallHelpers::test_execute_tool_recall_formats_cue_episode
  tests/test_autorecall.py::TestRecallSetsLastTime -q`
  - Result: 15 passed, with existing `datetime.utcnow()` deprecation warnings.
- Recall presenter lint:
  `uv run ruff check engram/retrieval/presenter.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_recall_presenter.py
  tests/test_knowledge_api.py tests/test_autorecall.py`
  - Result: passed.
- Observe/remember presenter/API/MCP focus:
  `uv run pytest tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py::TestObserve tests/test_knowledge_api.py::TestRemember
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_autorecall.py::TestObserveWithAutoRecall
  tests/test_autorecall.py::TestRememberWithAutoRecall -q`
  - Result: 17 passed, 2 skipped, with existing `datetime.utcnow()` warnings.
- Auto-observe skip/success focus:
  `uv run pytest tests/test_auto_observe.py::test_auto_observe_endpoint
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_auto_observe.py::test_auto_observe_short_content_skipped -q`
  - Result: 3 passed.
- Observe/remember presenter lint:
  `uv run ruff check engram/ingestion/presenter.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py tests/test_autorecall.py`
  - Result: passed.
- GraphSAGE flake focus:
  `uv run pytest tests/test_gnn.py::TestGraphSAGEInference::test_output_normalized -q`
  and `uv run ruff check tests/test_gnn.py`
  - Result: passed.
- Capture service/facade focus:
  `uv run pytest tests/test_capture_service.py tests/test_episode_cues.py
  tests/test_cqrs_split.py::TestStoreEpisode
  tests/test_auto_observe.py::test_auto_observe_endpoint
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_auto_observe.py::test_auto_observe_short_content_skipped -q`
  - Result: 13 passed, 2 skipped.
- Capture service/write-path focus:
  `uv run pytest tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py::TestObserve tests/test_knowledge_api.py::TestRemember
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_autorecall.py::TestObserveWithAutoRecall
  tests/test_autorecall.py::TestRememberWithAutoRecall
  tests/test_capture_service.py -q`
  - Result: 20 passed, 2 skipped, with existing `datetime.utcnow()` warnings.
- Decision-graph capture focus:
  `uv run pytest
  tests/test_mcp_tools.py::TestSearchFacts::test_search_hides_epistemic_facts_by_default
  tests/test_mcp_tools.py::TestSearchFacts::test_question_form_does_not_materialize_decision_entity
  tests/test_mcp_tools.py::TestSearchFacts::test_explicit_commitment_materializes_decision_entity -q`
  - Result: 3 passed.
- Capture service lint:
  `uv run ruff check engram/ingestion/capture_service.py engram/graph_manager.py
  tests/test_capture_service.py tests/test_episode_cues.py tests/test_cqrs_split.py
  tests/test_auto_observe.py`
  - Result: passed.
- Projection/cue state sync focus:
  `uv run pytest tests/test_projection_state_sync.py tests/test_capture_service.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_cqrs_split.py::TestProjectEpisode
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_formats_cue_results_and_tracks_hits
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_promotes_hot_cue_to_scheduled_projection
  tests/test_recall_feedback.py -q`
  - Result: 60 passed, 6 skipped, with existing warnings.
- Projection/cue state sync lint:
  `uv run ruff check engram/ingestion/projection_state.py
  engram/ingestion/capture_service.py engram/graph_manager.py engram/worker.py
  engram/consolidation/phases/triage.py engram/consolidation/phases/replay.py
  tests/test_projection_state_sync.py tests/test_capture_service.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py tests/test_cqrs_split.py
  tests/test_episode_retrieval.py tests/test_recall_feedback.py`
  - Result: passed.
- Projection service/facade focus:
  `uv run pytest tests/test_cqrs_split.py tests/test_projection_state_sync.py
  tests/test_capture_service.py -q`
  - Result: 20 passed.
- Projection service broader runtime focus:
  `uv run pytest tests/test_cqrs_split.py tests/test_episode_worker.py
  tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_recall_feedback.py tests/test_rework_integration.py -q`
  - Result: 64 passed, 6 skipped, with existing `datetime.utcnow()` warnings.
- Projection service lint:
  `uv run ruff check engram/ingestion/projection_service.py
  engram/graph_manager.py tests/test_cqrs_split.py`
  - Result: passed.
- Legacy projection executor focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_projection_state_sync.py tests/test_capture_service.py -q`
  - Result: 22 passed.
- Legacy projection executor broader runtime focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_recall_feedback.py tests/test_rework_integration.py -q`
  - Result: 66 passed, 6 skipped, with existing `datetime.utcnow()` warnings.
- Legacy projection executor lint:
  `uv run ruff check engram/ingestion/projection_execution.py
  engram/ingestion/projection_service.py engram/graph_manager.py
  tests/test_projection_execution.py tests/test_cqrs_split.py`
  - Result: passed.
- Evidence projection executor focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_projection_state_sync.py tests/test_capture_service.py -q`
  - Result: 24 passed.
- Evidence projection executor broader runtime focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_recall_feedback.py tests/test_rework_integration.py
  tests/test_evidence_storage.py tests/test_evidence_adjudication.py -q`
  - Result: 87 passed, 24 skipped, with existing `datetime.utcnow()` warnings.
- Projection lifecycle result focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_projection_state_sync.py tests/test_capture_service.py -q`
  - Result: 26 passed.
- Projection lifecycle result broader runtime focus:
  `uv run pytest tests/test_projection_execution.py tests/test_cqrs_split.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_recall_feedback.py tests/test_rework_integration.py
  tests/test_evidence_storage.py tests/test_evidence_adjudication.py -q`
  - Result: 89 passed, 24 skipped, with existing `datetime.utcnow()` warnings.
- Consolidation lifecycle contract focus:
  `uv run pytest tests/test_consolidation_engine.py -q`
  - Result: 20 passed.
- Consolidation lifecycle broader focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_cycle_context.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`
  - Result: 116 passed, 22 skipped, with existing `datetime.utcnow()` warnings.
- Consolidation phase runner focus:
  `uv run pytest tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py -q`
  - Result: 22 passed.
- Consolidation phase runner broader focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_cycle_context.py
  tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`
  - Result: 118 passed, 22 skipped, with existing `datetime.utcnow()` warnings.
- Consolidation phase runner lint:
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/phase_runner.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`
  - Result: passed.
- Consolidation event publisher focus:
  `uv run pytest tests/test_consolidation_events.py
  tests/test_consolidation_engine.py tests/test_consolidation_phase_runner.py -q`
  - Result: 25 passed.
- Consolidation event publisher broader focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_cycle_context.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`
  - Result: 121 passed, 22 skipped, with existing `datetime.utcnow()` warnings.
- Consolidation event publisher lint:
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/events.py engram/consolidation/phase_runner.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`
  - Result: passed.
- Consolidation learning service focus:
  `uv run pytest tests/test_consolidation_learning.py
  tests/test_consolidation_engine.py tests/test_consolidation_events.py
  tests/test_consolidation_phase_runner.py -q`
  - Result: 28 passed.
- Consolidation learning service broader focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_learning.py tests/test_consolidation_events.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_cycle_context.py
  tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`
  - Result: 124 passed, 22 skipped, with existing `datetime.utcnow()` warnings.
- Consolidation learning service lint:
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/learning.py engram/consolidation/events.py
  engram/consolidation/phase_runner.py tests/test_consolidation_learning.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`
  - Result: passed.
- Dashboard lifecycle focus:
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx`
  - Result: 20 passed, with existing React `act(...)` and SVG casing warnings.
- Dashboard full test run:
  `pnpm test -- --run`
  - Result: 191 passed, with existing React `act(...)`, canvas, and SVG casing
    warnings.
- Dashboard build:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard visual smoke:
  `pnpm exec vite --host 127.0.0.1`
  - Result: Brain Loop rendered in Safari at `http://127.0.0.1:5173/` with all
    five lifecycle stages visible and no first-viewport overlap. The backend was
    not running, so the visible API 500 was expected for this UI-only smoke.
- Lifecycle summary API focus:
  `uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary -q`
  - Result: 2 passed.
- Lifecycle summary/API adjacent focus:
  `uv run pytest tests/test_api_endpoints.py::TestStats
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_api_endpoints.py::TestEpisodes -q`
  - Result: 14 passed.
- Full API endpoint file:
  `uv run pytest tests/test_api_endpoints.py -q`
  - Result: 35 passed.
- Lifecycle summary API lint:
  `uv run ruff check engram/api/lifecycle.py engram/main.py
  tests/test_api_endpoints.py`
  - Result: passed.
- Lifecycle dashboard/store/WebSocket focus:
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx src/test/store.test.ts
  src/test/useWebSocket.test.ts src/test/apiClient.test.ts`
  - Result: 61 passed, with existing React `act(...)` and SVG casing warnings.
- Dashboard full test run after lifecycle summary wiring:
  `pnpm test -- --run`
  - Result: 191 passed, with existing React `act(...)`, canvas, and SVG casing
    warnings.
- Dashboard build after lifecycle summary wiring:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard lifecycle drilldown focus:
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx`
  - Result: 21 passed, with existing React `act(...)` and SVG casing warnings.
- Dashboard full test run after stage-card drilldowns:
  `pnpm test -- --run`
  - Result: 192 passed, with existing React `act(...)`, canvas, and SVG
    casing warnings.
- Dashboard build after stage-card drilldowns:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard visual smoke after stage-card drilldowns:
  `pnpm exec vite --host 127.0.0.1`
  - Result: Brain Loop rendered in Chrome at `http://127.0.0.1:5173/` with
    all five clickable stage cards visible. Clicking Capture opened the Feed
    drilldown. The backend was not running, so the visible API 500/proxy
    failures were expected for this UI-only smoke.
- Dashboard lifecycle context focus:
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/components.test.tsx src/test/mediumComponents.test.tsx
  src/test/store.test.ts`
  - Result: 93 passed, with existing React `act(...)`, canvas, and SVG casing
    warnings.
- Dashboard full test run after lifecycle context:
  `pnpm test -- --run`
  - Result: 195 passed, with existing React `act(...)`, canvas, and SVG casing
    warnings.
- Dashboard build after lifecycle context:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard visual smoke after lifecycle context:
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`
  - Result: Chrome reached the Stats drilldown from a lifecycle card. The
    backend was not running, so the Stats panel had no live data and the API
    500/proxy failures were expected for this UI-only smoke.
- Dashboard recall drilldown focus:
  `pnpm test -- --run src/test/components.test.tsx
  src/test/LifecyclePanel.test.tsx`
  - Result: 42 passed, with existing React `act(...)` and canvas warnings.
- Dashboard full test run after recall context:
  `pnpm test -- --run`
  - Result: 196 passed, with existing React `act(...)`, canvas, and SVG casing
    warnings.
- Dashboard build after recall context:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard visual smoke after recall context:
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`
  - Result: Chrome opened Knowledge from the Recall card and showed the focused
    `Recall Context` strip with an empty active-entity state. The backend was
    not running, so API 500/proxy failures were expected for this UI-only smoke.
- Brain loop evaluation report focus:
  `uv run pytest tests/test_brain_loop_report.py tests/test_evaluation_store.py
  -q`
  - Result: 5 passed.
- Brain loop evaluation API focus:
  `uv run pytest tests/test_api_endpoints.py::TestEvaluation
  tests/test_brain_loop_report.py tests/test_evaluation_store.py -q`
  - Result: 7 passed.
- Brain loop evaluation report lint:
  `uv run ruff check engram/api/evaluation.py engram/api/deps.py
  engram/main.py engram/evaluation/brain_loop_report.py
  engram/evaluation/store.py engram/evaluation/__init__.py
  scripts/brain_loop_report.py tests/test_api_endpoints.py
  tests/test_brain_loop_report.py
  tests/test_evaluation_store.py`
  - Result: passed.
- Brain loop report CLI smoke:
  `uv run python scripts/brain_loop_report.py --sqlite-path
  /private/tmp/engram-brain-report-smoke.db --format json`
  - Result: passed; emitted an empty-brain JSON report with expected coverage
    gaps and no external services.
- First-class evaluation CLI smoke:
  `uv run python -m engram evaluate --sqlite-path
  /private/tmp/engram-cli-evaluate-smoke.db --format json`
  - Result: passed; emitted the same empty-brain JSON report contract.
- Dashboard evaluation fallback focus:
  `pnpm test -- --run src/test/components.test.tsx`
  - Result: 42 passed, with existing React `act(...)` and jsdom canvas warnings.
- Dashboard full test run after evaluation drilldown:
  `pnpm test -- --run`
  - Result: 200 passed, with existing React `act(...)`, SVG casing, and jsdom
    canvas warnings.
- Dashboard build after evaluation drilldown:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Dashboard evaluation visual smoke:
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`
  - Result: Chrome opened the Evaluate route at `http://127.0.0.1:5173/` and
    showed the manual `Load Evaluation` fallback without first-viewport
    overlap. The backend was not running, so API 500/proxy failures were
    expected for this UI-only smoke.
- MCP evaluation-label/report focus:
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses
  tests/test_api_endpoints.py::TestEvaluation tests/test_brain_loop_report.py
  tests/test_evaluation_store.py -q`
  - Result: 13 passed, 2 skipped, with existing `datetime.utcnow()` warnings.
- MCP/REST evaluation lint:
  `uv run ruff check engram/mcp/server.py engram/api/evaluation.py
  engram/evaluation/presenter.py engram/evaluation/__init__.py
  tests/test_mcp_tools.py tests/test_api_endpoints.py
  tests/test_brain_loop_report.py tests/test_evaluation_store.py`
  - Result: passed.
- Live evaluation API smoke:
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-live-eval-smoke.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0 uv run engram serve
  --host 127.0.0.1 --port 8100`
  plus `curl` calls to seed one observed episode, one recall evaluation label,
  one session-continuity label, and
  `GET /api/evaluation/brain-loop/report`.
  - Result: passed. The report returned 1 episode, cue coverage 100%, Recall
    status `measured`, memory-need precision 100%, false recall 33.3%, useful
    packet rate 66.7%, Continuity status `measured`, session continuity lift
    0.5, open-loop recovery 100%, and the expected projected/consolidation
    coverage gaps.
- Live dashboard evaluation smoke:
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`
  against the running API at `http://127.0.0.1:8100`.
  - Result: Chrome opened `http://127.0.0.1:5173/`, the Evaluate drilldown
    rendered the live report values above, and the first viewport had no
    obvious overlap.
- Dashboard evaluation-label focus:
  `pnpm test -- --run src/test/apiClient.test.ts src/test/store.test.ts
  src/test/components.test.tsx`
  - Result: 84 passed, with existing React `act(...)` and jsdom canvas
    warnings.
- Dashboard full test run after operator label capture:
  `pnpm test -- --run`
  - Result: 205 passed, with existing React `act(...)`, SVG casing, and jsdom
    canvas warnings.
- Dashboard build after operator label capture:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Live dashboard label smoke:
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-label-smoke.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0 uv run engram serve
  --host 127.0.0.1 --port 8100` plus
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`.
  - Result: Chrome submitted a Recall label through the Evaluate UI; the form
    reached `stored` and the report refreshed to Recall `measured` with 1
    label. Direct REST calls on the same backend stored recall and
    session-continuity labels, and `GET /api/evaluation/brain-loop/report`
    returned Recall sample count 2, memory-need precision 100%, useful packet
    rate 66.7%, false recall 33.3%, Continuity sample count 1, session
    continuity lift 0.4, and open-loop recovery 100%.
  - Caveat: the Chrome smoke environment also emitted a large stream of
    `/api/knowledge/auto-observe` requests, which made the Continuity UI submit
    fail after the page reported offline. Use a clean browser context or disable
    that auto-observe source for future live UI smokes.
- Clean dashboard smoke and auto-observe guard:
  `uv run python -m engram evaluate --smoke --sqlite-path
  /private/tmp/engram-dashboard-clean-smoke-disabled-20260512.db --replace
  --group-id default --format json`, then
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-clean-smoke-disabled-20260512.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0
  ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false uv run engram serve --host
  127.0.0.1 --port 8100`, plus
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`.
  - Result: a separate Chrome profile with extensions disabled rendered Brain
    Loop with 3 episodes, 3 cues, 3 projected memories, and 1 consolidation
    cycle. Evaluate rendered the same report, submitted a Recall label and a
    Continuity label through the UI, and refreshed to Recall sample count 2 and
    Continuity sample count 2. REST report stayed at 3 seeded episodes with no
    coverage gaps.
  - Guard check: repeated external `/api/knowledge/auto-observe` requests still
    hit the backend, but `ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false` returned
    skipped `reason=disabled` and prevented DB pollution.
- Auto-observe guard regression:
  `uv run pytest tests/test_auto_observe.py::test_auto_observe_endpoint
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_auto_observe.py::test_auto_observe_short_content_skipped
  tests/test_auto_observe.py::test_auto_observe_can_be_disabled
  tests/test_config.py::TestEngramConfig::test_default_config -q`
  - Result: 5 passed.
- Auto-observe guard lint:
  `uv run ruff check engram/config.py engram/api/knowledge.py
  tests/test_auto_observe.py`
  - Result: passed.
- Shared lifecycle REST/MCP focus:
  `uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract -q`
  - Result: 3 passed.
- Shared lifecycle REST/MCP lint:
  `uv run ruff check engram/lifecycle_summary.py engram/api/lifecycle.py
  engram/mcp/server.py tests/test_mcp_tools.py`
  - Result: passed.
- Diff hygiene:
  `git diff --check`
  - Result: passed.
- Lifecycle CLI focus:
  `uv run pytest tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract -q`
  - Result: 5 passed.
- Lifecycle CLI lint:
  `uv run ruff check engram/lifecycle_cli.py engram/lifecycle_summary.py
  engram/api/lifecycle.py engram/mcp/server.py engram/__main__.py
  tests/test_lifecycle_cli.py tests/test_mcp_tools.py`
  - Result: passed.
- Lifecycle CLI smoke:
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-lifecycle-cli-smoke-20260512.db
  uv run python -m engram lifecycle --format json --episodes 1 --cycles 1`
  - Result: passed outside the sandbox after the first sandboxed attempt hit
    the existing `uv` cache permission boundary. The command returned the
    zero-state Capture/Cue/Project/Recall/Consolidate summary for group
    `default`.
- Doctor lifecycle focus:
  `uv run pytest tests/test_doctor.py tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract -q`
  - Result: 9 passed.
- Doctor lifecycle lint:
  `uv run ruff check engram/doctor.py engram/lifecycle_cli.py
  engram/lifecycle_summary.py engram/api/lifecycle.py engram/mcp/server.py
  engram/__main__.py tests/test_doctor.py tests/test_lifecycle_cli.py
  tests/test_mcp_tools.py`
  - Result: passed.
- Doctor lifecycle smoke:
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-lifecycle-config-20260512.db
  uv run python -m engram doctor --mode lite --skip-server --no-smoke --format json`
  - Result: passed outside the sandbox after the first sandboxed attempt hit
    the existing `uv` cache permission boundary. The report included a passing
    `lifecycle_snapshot` check and a zero-state lifecycle summary for group
    `default`.
- External lifecycle docs audit:
  searched README/docs/skill text for old phase-count and tool-count public
  wording.
  - Result: no stale public phase/tool count matches.
- External lifecycle docs contract check:
  `rg "/api/lifecycle/summary|get_lifecycle_summary|/api/evaluation/brain-loop/report"
  README.md skills/engram-memory/SKILL.md`
  - Result: README and the Engram skill expose the lifecycle/evaluation
    contracts.
- Projected/consolidated P3 smoke:
  `uv run python scripts/projected_consolidated_smoke.py`
  - Result: passed. The report returned 3 captured/cued episodes, 3 projected
    episodes, 2 linked entities, Recall and Continuity `measured`, 1 completed
    triage consolidation cycle, 1 measured calibration snapshot, and no coverage
    gaps.
- First-class projected/consolidated smoke:
  `uv run python -m engram evaluate --smoke --group-id operator_brain --format json`
  - Result: passed with the same no-gap projected/consolidated report contract
    and `group_id` set to `operator_brain`.
- Doctor CLI smoke:
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-config.db uv run python -m
  engram doctor --mode lite --skip-server --format json`
  - Result: passed. Config, SQLite parent, mode resolution, and brain-loop
    smoke checks were `pass`; server was skipped.
- Doctor config-only smoke:
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-config.db uv run python -m
  engram doctor --mode lite --skip-server --no-smoke --format markdown`
  - Result: passed and rendered the Markdown check list.
- Doctor tests:
  `uv run pytest tests/test_doctor.py -q`
  - Result: 3 passed.
- Projected/consolidated smoke regression:
  `uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_brain_loop_report.py tests/test_evaluation_store.py -q`
  - Result: 7 passed.
- Projected/consolidated smoke lint:
  `uv run ruff check engram/doctor.py engram/evaluation/cli.py
  engram/evaluation/smoke.py engram/evaluation/__init__.py engram/__main__.py
  scripts/projected_consolidated_smoke.py tests/test_doctor.py
  tests/test_projected_consolidated_smoke.py`
  - Result: passed.
- Cue-feedback projection-state sync focus:
  `uv run pytest tests/test_episode_retrieval.py tests/test_recall_feedback.py
  tests/test_projection_state_sync.py
  tests/test_api_endpoints.py::TestLifecycleSummary -q`
  - Result: 32 passed, 8 skipped, with existing AsyncMock coroutine warnings in
    `test_episode_retrieval.py`.
- Cue-feedback projection-state sync lint:
  `uv run ruff check engram/graph_manager.py engram/ingestion/projection_state.py
  tests/test_episode_retrieval.py tests/test_recall_feedback.py
  tests/test_projection_state_sync.py`
  - Result: passed.
- Projection-state runtime audit focus:
  `uv run pytest tests/test_episode_worker.py tests/test_project_bootstrap.py
  tests/test_projection_state_sync.py -q`
  - Result: 36 passed, with existing warnings.
- Projection-state runtime audit lint:
  `uv run ruff check engram/worker.py engram/graph_manager.py
  tests/test_episode_worker.py tests/test_project_bootstrap.py
  tests/test_projection_state_sync.py`
  - Result: passed.
- Consolidation finalization focus:
  `uv run pytest tests/test_consolidation_finalization.py
  tests/test_consolidation_engine.py tests/test_consolidation_cycle_context.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_learning.py
  tests/test_consolidation_events.py -q`
  - Result: 37 passed, 2 skipped.
- Consolidation finalization lint:
  `uv run ruff check engram/consolidation/finalization.py
  engram/consolidation/engine.py tests/test_consolidation_finalization.py
  tests/test_consolidation_engine.py`
  - Result: passed.
- Consolidation capability validator focus:
  `uv run pytest tests/test_consolidation_capabilities.py
  tests/test_consolidation_engine.py -q`
  - Result: 24 passed.
- Consolidation capability validator adjacent focus:
  `uv run pytest tests/test_consolidation_capabilities.py
  tests/test_consolidation_engine.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_events.py tests/test_consolidation_learning.py
  tests/test_consolidation_finalization.py -q`
  - Result: 35 passed.
- Consolidation capability validator lint:
  `uv run ruff check engram/consolidation/capabilities.py
  engram/consolidation/engine.py tests/test_consolidation_capabilities.py
  tests/test_consolidation_engine.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after capability extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2504 passed, 43 skipped, 236 deselected, with 1 transient
    `aiosqlite` event-loop-close thread warning reported at
    `tests/test_summary_merge.py::TestIsMetaSummary::test_knowledge_graph_node`.
- Summary-merge warning reproduction checks:
  `uv run pytest tests/test_summary_merge.py -q`
  and
  `uv run pytest tests/test_structure_embeddings.py tests/test_summary_merge.py -q`
  - Result: 35 passed and 40 passed, with no reproduced warning.
- Native Helix CLI path focus:
  `uv run pytest tests/test_cli_main.py tests/test_lifecycle_cli.py
  tests/test_doctor.py tests/test_setup.py -q`
  - Result: 28 passed.
- Native evaluation/Helix stats focus:
  `uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_lifecycle_cli.py tests/test_doctor.py -q`
  - Result: 16 passed.
- MCP Helix lifecycle/evaluation focus:
  `uv run pytest
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_evaluation_report_reads_active_consolidation_store -q`
  - Result: 3 passed.
- Native shutdown/ownership focus:
  `uv run pytest tests/test_helix_stats.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples -q`
  - Result: 7 passed.
- Native shutdown/ownership lint:
  `uv run ruff check engram/storage/helix/graph.py
  engram/storage/helix/search.py engram/storage/helix/atlas.py
  engram/storage/helix/consolidation.py
  engram/storage/helix/conversations.py engram/storage/factory.py
  engram/main.py engram/mcp/server.py tests/test_helix_stats.py
  tests/test_mcp_tools.py`
  - Result: passed.
- Native evaluation smoke:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-native-path-eval-data uv run
  python -m engram evaluate --mode helix --no-saved-samples --format json`
  - Result: passed. It initialized `helix_native` and returned the expected
    empty local report contract without Docker.
- Populated native smoke and reopen check:
  `uv run python -m engram evaluate --smoke --mode helix --sqlite-path
  /private/tmp/engram-native-smoke-labels-20260512.db --helix-data-dir
  /private/tmp/engram-native-smoke-data-20260512 --replace --group-id
  native_brain --format json`
  - Result: passed. It initialized `helix_native`, captured/cued/projected 3
    episodes, persisted 1 completed triage consolidation cycle, saved a
    measured calibration snapshot plus recall/continuity labels, and returned
    no coverage gaps.
- Reopened native report:
  `uv run python -m engram evaluate --mode helix --sqlite-path
  /private/tmp/engram-native-smoke-labels-20260512.db --helix-data-dir
  /private/tmp/engram-native-smoke-data-20260512 --group-id native_brain
  --format json`
  - Result: passed. The reopened report preserved 3 episodes, 3 cues, 3
    projected memories, linked-entity projection yield, 1 consolidation cycle,
    measured recall/continuity labels, and no coverage gaps.
- Evaluation/doctor native smoke regression focus:
  `uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_lifecycle_cli.py tests/test_doctor.py -q`
  - Result: 20 passed.
- Native REST/MCP surface parity:
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 3 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST/MCP evaluation-write parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST consolidation-read parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST notification parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native project artifact parity focus:
  `uv run pytest tests/test_project_bootstrap.py::test_artifact_search_uses_lexical_fallback_when_index_misses tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 2 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native MCP project artifact parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native MCP project bootstrap parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST/MCP runtime-state parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native MCP consolidation-control parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST/MCP route parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native MCP route auto-observe parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST/MCP intention parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST/MCP adjudication parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native conversation/chat persistence parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 3 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_conversations_api.py tests/test_knowledge_api.py::TestChat -q`
  - Result: 9 passed.
  `uv run ruff check tests/test_native_surface_parity.py tests/test_conversations_api.py tests/test_knowledge_api.py`
  - Result: passed.
- Native REST/MCP forget parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST/MCP feedback parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_feedback_tool.py -q`
  - Result: 9 passed.
  `uv run ruff check engram/graph_manager.py tests/test_native_surface_parity.py tests/test_feedback_tool.py`
  - Result: passed.
- Native MCP identity-core parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST/MCP context parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native MCP notification piggyback parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native MCP auto-recall piggyback parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST consolidation trigger parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST health parity focus:
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- Native REST admin benchmark-loader parity focus:
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- Native REST episode status-filter and cursor parity focus:
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- MCP prompt brain-loop language:
  `uv run pytest tests/test_mcp_prompts.py -q`
  - Result: 10 passed.
  `uv run ruff check engram/mcp/prompts.py tests/test_mcp_prompts.py`
  - Result: passed.
- Native REST/MCP entity/fact lookup parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native graph detail/resource parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST entity mutation parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native MCP write parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST observe parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native REST auto-observe parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
- Native REST/MCP attachment capture parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native replay-queue parity focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_rest_surfaces_handle_bounded_remember_recall_load -q`
  - Result: 1 passed.
- Native multi-batch load/reopen focus:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_rest_surfaces_survive_multi_batch_load_and_reopen -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 4 passed, with existing `datetime.utcnow()` deprecation warnings.
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- Native operator load smoke:
  `uv run python -m engram evaluate --smoke --mode helix --sqlite-path
  /private/tmp/engram-native-stress-120-labels-20260513.db --helix-data-dir
  /private/tmp/engram-native-stress-120-data-20260513 --replace --group-id
  native_brain --smoke-load-count 120 --smoke-recall-rounds 5 --format json`
  - Result: passed after adding multi-cycle smoke support. The report returned
    123 captured/cued/projected episodes, 2 completed consolidation cycles, no
    coverage gaps, and `smoke.recall_checks: 30`.
- Native operator duration-smoke focus:
  `uv run python -m engram evaluate --smoke --mode helix --sqlite-path
  /private/tmp/engram-native-duration-labels-20260513.db --helix-data-dir
  /private/tmp/engram-native-duration-data-20260513 --replace --group-id
  native_brain --smoke-load-count 6 --smoke-recall-rounds 1
  --smoke-min-duration-seconds 0.01 --format json`
  - Result: passed. The report returned 9 captured/cued/projected episodes, no
    coverage gaps, `smoke.recall_checks: 6`,
    `smoke.duration_recall_checks: 6`, and
    `smoke.duration_elapsed_seconds: 0.04`.
  `uv run pytest tests/test_projected_consolidated_smoke.py -q`
  - Result: 6 passed.
  `uv run ruff check engram/evaluation/cli.py engram/evaluation/smoke.py
  tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_helix_stats.py tests/test_native_transport.py -q`
  - Result: 10 passed.
  `git diff --check -- README.md docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md server/engram/evaluation/cli.py
  server/engram/evaluation/smoke.py
  server/tests/test_projected_consolidated_smoke.py`
  - Result: passed.
- Native hour-scale recall soak:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-native-hour-soak-data-20260513
  ENGRAM_SQLITE__PATH=/private/tmp/engram-native-hour-soak-labels-20260513.db
  ENGRAM_EMBEDDING__PROVIDER=noop uv run python -m engram evaluate --smoke
  --mode helix --sqlite-path
  /private/tmp/engram-native-hour-soak-labels-20260513.db --helix-data-dir
  /private/tmp/engram-native-hour-soak-data-20260513 --replace --group-id
  native_brain --smoke-load-count 6 --smoke-recall-rounds 1
  --smoke-min-duration-seconds 3600 --smoke-pause-seconds 2 --format json`
  - Result: passed outside the sandbox after the first sandboxed attempt hit the
    existing `uv` cache permission boundary. The run initialized
    `helix_native`, returned 9 captured/cued/projected episodes, 1 completed
    triage consolidation cycle, no coverage gaps, `smoke.recall_checks: 6`,
    `smoke.duration_recall_checks: 10362`, and
    `smoke.duration_elapsed_seconds: 3600.599`.
  `git diff --check -- docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
- Consolidation requested-phase validation:
  `uv run pytest tests/test_consolidation_engine.py -q`
  - Result: 24 passed.
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/lifecycle.py tests/test_consolidation_engine.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_capabilities.py -q`
  - Result: 32 passed.
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-consolidation-unknown-phase-20260513.db
  uv run python -m engram.consolidation --phases missing_phase`
  - Result: exited 2 outside the sandbox after the first sandboxed attempt hit
    the existing `uv` cache permission boundary. The CLI printed
    `Consolidation failed: Unknown consolidation phase(s): missing_phase`
    without a traceback.
  `git diff --check -- server/engram/consolidation/cli.py
  server/engram/consolidation/engine.py server/engram/consolidation/lifecycle.py
  server/tests/test_consolidation_engine.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
- Consolidation phase registry focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_scheduler.py -q`
  - Result: 37 passed.
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/phase_registry.py engram/consolidation/scheduler.py
  tests/test_consolidation_engine.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_scheduler.py tests/test_consolidation_events.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_capabilities.py -q`
  - Result: 45 passed.
  `git diff --check -- server/engram/consolidation/engine.py
  server/engram/consolidation/phase_registry.py server/engram/consolidation/scheduler.py
  server/tests/test_consolidation_engine.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after phase-registry runtime guard:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2512 passed, 43 skipped, 236 deselected in 126.62s.
- Dashboard consolidation phase-contract focus:
  `pnpm test -- --run src/test/ConsolidationPanel.test.tsx`
  - Result: 8 passed, with the existing React `act(...)` warning in the
    loading-state test.
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, SVG casing,
    and jsdom canvas warnings.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
  `git diff --check -- dashboard/src/constants/consolidation.ts
  dashboard/src/constants/rpg.ts dashboard/src/test/ConsolidationPanel.test.tsx`
  - Result: passed.
- Backend/dashboard phase-order contract:
  `uv run pytest tests/test_dashboard_phase_contract.py
  tests/test_consolidation_engine.py tests/test_consolidation_scheduler.py -q`
  - Result: 38 passed.
  `uv run ruff check tests/test_dashboard_phase_contract.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after dashboard phase-order contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2513 passed, 43 skipped, 236 deselected in 324.81s.
- Consolidation CLI failed-cycle operator contract:
  `uv run pytest tests/test_consolidation_cli.py
  tests/test_consolidation_engine.py tests/test_consolidation_capabilities.py -q`
  - Result: 31 passed after adding structured cycle/phase error fields.
  `uv run ruff check engram/consolidation/cli.py tests/test_consolidation_cli.py`
  - Result: passed.
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-consolidation-cli-unknown-phase-20260513b.db
  uv run python -m engram.consolidation --phases missing_phase`
  - Result: exited 2 outside the sandbox after the first sandboxed attempt hit
    the existing `uv` cache permission boundary. The CLI printed
    `Consolidation failed: Unknown consolidation phase(s): missing_phase`
    without a traceback.
- Backend broad non-Docker/non-Helix gate after consolidation CLI contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2515 passed, 43 skipped, 236 deselected in 363.81s.
- MCP consolidation trigger error contract:
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors -q`
  - Result: 1 passed.
  `uv run pytest tests/test_mcp_tools.py -q`
  - Result: 56 passed, 2 skipped.
  `uv run ruff check engram/mcp/server.py tests/test_mcp_tools.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after MCP consolidation error contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2516 passed, 43 skipped, 236 deselected in 365.74s.
- REST consolidation read error contract:
  `uv run pytest tests/test_api_endpoints.py::TestConsolidationAPI::test_status_and_history_include_cycle_and_phase_errors -q`
  - Result: 1 passed.
  `uv run pytest tests/test_api_endpoints.py -q`
  - Result: 38 passed.
  `uv run ruff check engram/api/consolidation.py tests/test_api_endpoints.py`
  - Result: passed.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
- Backend broad non-Docker/non-Helix gate after REST consolidation read contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2517 passed, 43 skipped, 236 deselected in 426.86s.
- Dashboard consolidation trigger response contract:
  `pnpm test -- --run src/test/ConsolidationPanel.test.tsx
  src/test/apiClient.test.ts src/test/store.test.ts`
  - Result: 49 passed, with the existing React `act(...)` warning in the
    consolidation panel loading-state test.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, SVG casing,
    and jsdom canvas warnings.
- Shared consolidation presenter contract:
  `uv run pytest tests/test_consolidation_presenter.py
  tests/test_consolidation_cli.py
  tests/test_api_endpoints.py::TestConsolidationAPI::test_status_and_history_include_cycle_and_phase_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors -q`
  - Result: 6 passed.
  `uv run pytest tests/test_consolidation_presenter.py
  tests/test_consolidation_cli.py tests/test_api_endpoints.py
  tests/test_mcp_tools.py -q`
  - Result: 98 passed, 2 skipped.
  `uv run ruff check engram/consolidation/presenter.py
  engram/api/consolidation.py engram/consolidation/cli.py engram/mcp/server.py
  tests/test_consolidation_presenter.py tests/test_consolidation_cli.py
  tests/test_api_endpoints.py tests/test_mcp_tools.py`
  - Result: passed.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
- Shared consolidation operator-summary contract:
  `uv run pytest tests/test_consolidation_presenter.py
  tests/test_api_endpoints.py::TestConsolidationAPI
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_reports_completed_phase_warnings
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_consolidation_status_includes_latest_cycle
  -q`
  - Result: 7 passed.
  `uv run ruff check engram/consolidation/presenter.py
  engram/consolidation/cli.py engram/mcp/server.py
  tests/test_consolidation_presenter.py tests/test_api_endpoints.py
  tests/test_mcp_tools.py`
  - Result: passed.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
- Backend broad non-Docker/non-Helix gate after shared operator summary:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2573 passed, 43 skipped, 236 deselected in 111.88s.
- Backend broad non-Docker/non-Helix gate after shared consolidation presenter:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2519 passed, 43 skipped, 236 deselected in 1300.65s.
- Backend broad non-Docker/non-Helix gate after requested-phase validation:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2510 passed, 43 skipped, 236 deselected in 114.74s.
- Consolidation finalization event contract focus:
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_finalization.py -q`
  - Result: 28 passed.
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_finalization.py
  tests/test_consolidation_learning.py tests/test_consolidation_phase_runner.py -q`
  - Result: 33 passed.
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/events.py engram/consolidation/finalization.py
  engram/consolidation/lifecycle.py tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_finalization.py`
  - Result: passed.
- Lifecycle Recall intention summary focus:
  `uv run pytest tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py::TestLifecycleSummary::test_lifecycle_summary_maps_brain_loop_contract
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract -q`
  - Result: 9 passed.
  `uv run ruff check engram/lifecycle_summary.py tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py tests/test_mcp_tools.py`
  - Result: passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/store.test.ts
  src/test/LifecyclePanel.test.tsx`
  - Result: 43 passed.
  `pnpm exec tsc --noEmit`
  - Result: passed after making Recall intention summary required in the
    dashboard lifecycle type.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/store.test.ts
  src/test/LifecyclePanel.test.tsx src/test/nativeDashboardSmoke.test.tsx`
  - Result: 44 passed, 1 skipped.
  `uv run pytest tests/test_lifecycle_cli.py -q`
  - Result: 7 passed.
  `uv run ruff check engram/lifecycle_cli.py tests/test_lifecycle_cli.py`
  - Result: passed.
  `uv run python -m engram lifecycle --sqlite-path
  /private/tmp/engram-lifecycle-markdown-intentions-20260513.db --group-id
  cli_brain --episodes 0 --cycles 1 --top-n 0 --format markdown`
  - Result: passed. The Recall stage line includes
    `intentions 0 | pinned 0`.
- Broad local gate after lifecycle/finalization contract work:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2508 passed, 43 skipped, 236 deselected in 111.17s.
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped. Existing React `act(...)`, SVG casing, and
    canvas-environment warnings remain.
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
  `git diff --check`
  - Result: passed.
- Native evaluation/surface regression focus:
  `uv run pytest tests/test_native_surface_parity.py
  tests/test_projected_consolidated_smoke.py tests/test_helix_stats.py
  tests/test_brain_loop_report.py tests/test_lifecycle_cli.py tests/test_doctor.py
  -q`
  - Result: 23 passed, with existing `datetime.utcnow()` deprecation warnings.
- Cue-usefulness report focus:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed.
- Calibration Markdown report focus:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed.
- Recall result-builder focus:
  `uv run pytest tests/test_recall_result_builder.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 13 passed.
- Recall API/rework focus:
  `uv run pytest tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 9 passed.
- Native/lite smoke lint:
  `uv run ruff check engram/graph_manager.py tests/test_native_surface_parity.py
  engram/evaluation/cli.py engram/evaluation/smoke.py
  engram/storage/protocols.py engram/storage/sqlite/graph.py
  engram/storage/falkordb/graph.py engram/storage/helix/graph.py
  engram/storage/helix/native_transport.py tests/storage/contract.py
  tests/test_projected_consolidated_smoke.py tests/test_helix_stats.py
  tests/test_doctor.py`
  - Result: passed.
- Recall result-builder lint:
  `uv run ruff check engram/graph_manager.py
  engram/retrieval/result_builder.py tests/test_recall_result_builder.py`
  - Result: passed.
- Recall episode-traversal focus:
  `uv run pytest tests/test_recall_episode_traversal.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 13 passed.
- Recall episode-traversal lint:
  `uv run ruff check engram/graph_manager.py
  engram/retrieval/episode_traversal.py tests/test_recall_episode_traversal.py`
  - Result: passed.
- Recall API/rework focus after traversal extraction:
  `uv run pytest tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 9 passed.
- Recall near-miss focus:
  `uv run pytest tests/test_recall_near_miss.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_tracks_cue_near_misses
  -q`
  - Result: 5 passed.
- Recall near-miss lint:
  `uv run ruff check engram/graph_manager.py engram/retrieval/near_miss.py
  tests/test_recall_near_miss.py`
  - Result: passed.
- Recall/API/rework focus after near-miss extraction:
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
- Recall priming focus:
  `uv run pytest tests/test_recall_priming.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_entity_results_are_typed_and_prime_neighbors
  -q`
  - Result: 5 passed.
- Recall priming lint:
  `uv run ruff check engram/graph_manager.py engram/retrieval/priming.py
  tests/test_recall_priming.py`
  - Result: passed.
- Recall/API/rework focus after priming extraction:
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
- Relevance-confidence focus:
  `uv run pytest tests/test_relevance_confidence.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 33 passed.
- Relevance-confidence lint:
  `uv run ruff check engram/graph_manager.py engram/retrieval/confidence.py
  tests/test_relevance_confidence.py`
  - Result: passed.
- Recall/API/rework focus after confidence extraction:
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
- Conversation fingerprint focus:
  `uv run pytest tests/test_conversation_retrieval.py::TestConversationFingerprinter
  tests/test_conversation_retrieval.py::TestRecallConversationFingerprintRecorder
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 16 passed.
- Conversation fingerprint lint:
  `uv run ruff check engram/graph_manager.py engram/retrieval/context.py
  tests/test_conversation_retrieval.py`
  - Result: passed.
- Recall/API/rework focus after conversation fingerprint extraction:
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
- Recall working-memory extraction focus:
  `uv run pytest tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 31 passed.
  `uv run pytest tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py
  tests/test_recall_feedback.py -q`
  - Result: 47 passed, 6 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/working_memory.py engram/graph_manager.py
  tests/test_working_memory.py`
  - Result: passed.
- Recall interaction-feedback extraction focus:
  `uv run pytest tests/test_recall_feedback.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 14 passed, 6 skipped.
  `uv run pytest tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 49 passed, 6 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/feedback.py engram/graph_manager.py
  tests/test_recall_feedback.py`
  - Result: passed.
- Recall entity-access extraction focus:
  `uv run pytest tests/test_recall_feedback.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 15 passed, 6 skipped.
  `uv run pytest tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 50 passed, 6 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/feedback.py engram/graph_manager.py
  tests/test_recall_feedback.py`
  - Result: passed.
- Recall cue-feedback extraction focus:
  `uv run pytest tests/test_recall_feedback.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 15 passed, 7 skipped.
  `uv run pytest tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 50 passed, 7 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/feedback.py engram/graph_manager.py
  tests/test_recall_feedback.py`
  - Result: passed.
- Recall explicit feedback-applier extraction focus:
  `uv run pytest tests/test_recall_feedback.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 15 passed, 8 skipped.
  `uv run pytest tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 50 passed, 8 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/feedback.py engram/graph_manager.py
  tests/test_recall_feedback.py`
  - Result: passed.
- Recall current-state result-selection focus:
  `uv run pytest tests/test_recall_result_selection.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_current_state_queries_prefer_entities_over_episodes
  -q`
  - Result: 4 passed.
  `uv run pytest tests/test_recall_result_selection.py tests/test_recall_feedback.py
  tests/test_working_memory.py tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 53 passed, 8 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/result_selection.py engram/graph_manager.py
  tests/test_recall_result_selection.py`
  - Result: passed.
- Recall request-policy extraction focus:
  `uv run pytest tests/test_recall_request_policy.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_tracks_cue_near_misses
  tests/test_recall_feedback.py::TestRecallFeedback::test_surfaced_recall_skips_access_but_emits_interaction
  tests/test_recall_feedback.py::TestRecallFeedback::test_explicit_recall_can_still_record_access_and_emit_interaction
  -q`
  - Result: 4 passed, 2 skipped.
  `uv run pytest tests/test_recall_request_policy.py tests/test_recall_result_selection.py
  tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 56 passed, 8 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/request_policy.py engram/graph_manager.py
  tests/test_recall_request_policy.py`
  - Result: passed.
- Recall primary-result materializer extraction focus:
  `uv run pytest tests/test_recall_primary_results.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 13 passed.
  `uv run pytest tests/test_recall_primary_results.py tests/test_recall_request_policy.py
  tests/test_recall_result_selection.py tests/test_recall_feedback.py
  tests/test_working_memory.py tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_near_miss.py
  tests/test_recall_episode_traversal.py tests/test_recall_result_builder.py -q`
  - Result: 59 passed, 8 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed.
  `uv run ruff check engram/retrieval/primary_results.py engram/graph_manager.py
  tests/test_recall_primary_results.py`
  - Result: passed.
- Recall near-miss materializer extraction focus:
  `uv run pytest tests/test_recall_near_miss.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_tracks_cue_near_misses
  -q`
  - Result: 6 passed.
  `uv run pytest tests/test_recall_near_miss.py tests/test_recall_primary_results.py
  tests/test_recall_request_policy.py tests/test_recall_result_selection.py
  tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_episode_traversal.py
  tests/test_recall_result_builder.py -q`
  - Result: 60 passed, 8 skipped.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed in 513.53s.
  `uv run ruff check engram/retrieval/near_miss.py engram/graph_manager.py
  tests/test_recall_near_miss.py`
  - Result: passed.
- Recall post-processor extraction focus:
  `uv run pytest tests/test_recall_post_process.py -q`
  - Result: 1 passed.
  `uv run pytest tests/test_recall_post_process.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 11 passed.
  `uv run pytest tests/test_recall_post_process.py tests/test_recall_near_miss.py
  tests/test_recall_primary_results.py tests/test_recall_request_policy.py
  tests/test_recall_result_selection.py tests/test_recall_feedback.py
  tests/test_working_memory.py tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_episode_traversal.py
  tests/test_recall_result_builder.py -q`
  - Result: 61 passed, 8 skipped.
  `uv run ruff check engram/retrieval/post_process.py engram/graph_manager.py
  tests/test_recall_post_process.py`
  - Result: passed.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed in 33.26s.
  `git diff --check`
  - Result: passed.
- Recall service extraction focus:
  `uv run pytest tests/test_recall_service.py tests/test_recall_post_process.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes -q`
  - Result: 12 passed.
  `uv run pytest tests/test_working_memory.py::TestGraphManagerWorkingMemory::test_recall_populates_working_memory
  tests/test_recall_service.py -q`
  - Result: 2 passed.
  `uv run pytest tests/test_recall_service.py tests/test_recall_post_process.py
  tests/test_recall_near_miss.py tests/test_recall_primary_results.py
  tests/test_recall_request_policy.py tests/test_recall_result_selection.py
  tests/test_recall_feedback.py tests/test_working_memory.py
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_recall_priming.py tests/test_recall_episode_traversal.py
  tests/test_recall_result_builder.py -q`
  - Result: 62 passed, 8 skipped.
  `uv run ruff check engram/retrieval/service.py engram/retrieval/post_process.py
  engram/graph_manager.py tests/test_recall_service.py
  tests/test_recall_post_process.py`
  - Result: passed.
  `uv run ruff check tests/test_working_memory.py tests/test_recall_service.py
  engram/retrieval/service.py engram/graph_manager.py`
  - Result: passed.
  `uv run pytest tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes
  tests/test_knowledge_api.py::TestRecall
  tests/test_rework_integration.py::test_observe_creates_cue_and_surfaced_recall_stays_latent
  tests/test_rework_integration.py::test_used_cue_feedback_schedules_worker_projection_and_enables_entity_recall
  -q`
  - Result: 19 passed in 28.37s.
  `git diff --check`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after Recall post-processor and
  Recall service extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2493 passed, 43 skipped, 234 deselected, with existing warning
    noise.
- Backend broad non-Docker/non-Helix gate after native atlas/WebSocket,
  conversation, auto-recall, consolidation-trigger, MCP prompt,
  admin benchmark-loader, and episode-list filter/cursor parity:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with 11,474 warnings before
    the benchmark-corpus UTC cleanup.
- Benchmark corpus UTC warning cleanup:
  `uv run ruff check engram/benchmark/corpus.py tests/benchmark/test_corpus.py`
  - Result: passed.
  `uv run pytest tests/benchmark/test_corpus.py -q`
  - Result: 38 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with 500 warnings. The
    remaining warnings were existing datetime uses outside the benchmark corpus,
    two known async-mark warnings, and one aiosqlite thread warning in the broad
    lane.
- Production UTC helper cleanup:
  `uv run ruff check engram/retrieval/prospective.py
  engram/extraction/temporal.py engram/embeddings/graph/storage.py
  engram/consolidation/phases/dream.py tests/test_prospective_v2.py
  tests/test_temporal.py tests/test_graph_embed_storage.py
  tests/test_dream_associations.py`
  - Result: passed.
  `uv run pytest tests/test_prospective_v2.py tests/test_temporal.py
  tests/test_graph_embed_storage.py tests/test_dream_associations.py -q`
  - Result: 85 passed, with 5 remaining test-fixture datetime warnings.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with 462 warnings.
- Schema formation fixture UTC cleanup:
  `uv run ruff check tests/test_schema_formation.py`
  - Result: passed.
  `uv run pytest tests/test_schema_formation.py -q`
  - Result: 31 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with 106 warnings.
- Consolidation fixture UTC cleanup:
  `uv run ruff check tests/test_consolidation_replay.py
  tests/test_consolidation_prune.py tests/test_memory_maturation.py
  tests/test_microglia.py tests/test_predicate_enriched_embeddings.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_replay.py
  tests/test_consolidation_prune.py tests/test_memory_maturation.py
  tests/test_microglia.py tests/test_predicate_enriched_embeddings.py -q`
  - Result: 95 passed, 1 skipped.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with 20 warnings.
- Final broad warning cleanup:
  `uv run ruff check tests/test_consolidation_graph_methods.py
  tests/test_mcp_tools.py tests/test_prospective_v2.py
  tests/test_structural_merge.py tests/test_structure_embeddings.py
  tests/test_proactive_recall.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_graph_methods.py
  tests/test_mcp_tools.py tests/test_prospective_v2.py
  tests/test_structural_merge.py tests/test_structure_embeddings.py
  tests/test_proactive_recall.py -q`
  - Result: 154 passed, 2 skipped, with transient aiosqlite thread warnings in
    the combined focused run; `tests/test_structure_embeddings.py` passes cleanly
    alone.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2501 passed, 43 skipped, 236 deselected, with zero warnings
    reported.
- Native conversation update/delete parity:
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
  `uv run pytest tests/test_conversations_api.py tests/test_knowledge_api.py::TestChat -q`
  - Result: 9 passed.
  `uv run ruff check engram/api/conversations.py engram/api/knowledge.py
  engram/storage/helix/conversations.py tests/test_native_surface_parity.py`
  - Result: passed.
- REST replay queue group isolation:
  `uv run pytest tests/test_knowledge_api.py::TestReplayQueue
  tests/test_knowledge_api.py::TestRecall -q`
  - Result: 8 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 45 passed in 104.09s.
  `uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_api_endpoints.py::TestEvaluation -q`
  - Result: 4 passed in 53.86s.
  `uv run ruff check engram/api/knowledge.py tests/test_knowledge_api.py`
  - Result: passed.
- Project bootstrap group semantics:
  `uv run pytest tests/test_project_bootstrap.py::test_bootstrap_project_artifacts_preserve_active_group
  tests/test_project_bootstrap.py::test_bootstrap_artifact_episode_syncs_cue_only_state -q`
  - Result: 2 passed.
  `uv run pytest tests/test_project_bootstrap.py -q`
  - Result: 14 passed, no warnings after replacing test `datetime.utcnow()`
    usage with `utc_now()` and stubbing unrelated graph-change publication in
    the PART_OF edge test.
  `uv run ruff check tests/test_project_bootstrap.py`
  - Result: passed.
- Native PyO3 surface suite after conversation fingerprint extraction:
  `uv run pytest tests/test_native_surface_parity.py
  tests/test_projected_consolidated_smoke.py tests/test_helix_stats.py
  tests/test_brain_loop_report.py tests/test_lifecycle_cli.py
  tests/test_doctor.py -q`
  - Result: 23 passed, 2 warnings.
- Native doctor smoke behavior focus:
  `uv run pytest tests/test_doctor.py
  tests/test_projected_consolidated_smoke.py::test_evaluate_cli_smoke_flag_passes_helix_mode
  tests/test_setup.py::test_collect_config_helix_uses_native_transport
  tests/test_cli_main.py -q`
  - Result: 9 passed.
- Native doctor live smoke:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run python -m engram
  doctor --mode helix --skip-server --format json`
  - Result: passed with `brain_loop_smoke` metadata `mode: helix`,
    `episodes: 3`, `projected: 3`, `cycles: 1`, and no coverage gaps. The
    first sandboxed run was blocked by `uv` cache permissions, then passed after
    approved cache access.
- Native doctor/smoke/parity regression:
  `uv run pytest tests/test_native_surface_parity.py
  tests/test_projected_consolidated_smoke.py tests/test_doctor.py -q`
  - Result: 14 passed, 2 warnings.
- Native lifecycle/doctor explicit data-dir focus:
  `uv run pytest tests/test_lifecycle_cli.py tests/test_doctor.py -q`
  - Result: 12 passed.
- Native lifecycle explicit data-dir smoke:
  `uv run python -m engram lifecycle --mode helix --helix-data-dir
  /private/tmp/engram-native-cli-dir-data-20260512 --group-id native_cli_dir
  --format json`
  - Result: passed. The lifecycle snapshot reopened the seeded native PyO3 data
    directory and returned 3 episodes, 3 cues, 3 projected episodes, and 1
    consolidation cycle for `native_cli_dir`.
- Native doctor explicit data-dir smoke:
  `uv run python -m engram doctor --mode helix --helix-data-dir
  /private/tmp/engram-native-cli-dir-data-20260512 --group-id native_cli_dir
  --skip-server --format json`
  - Result: passed. The lifecycle snapshot read the seeded native directory,
    while `brain_loop_smoke` used a disposable native data directory and still
    returned `mode: helix`, 3 episodes, 3 projected, 1 cycle, and no coverage
    gaps.
- Native lifecycle/doctor operator regression:
  `uv run pytest tests/test_lifecycle_cli.py tests/test_doctor.py
  tests/test_projected_consolidated_smoke.py tests/test_native_surface_parity.py -q`
  - Result: 20 passed, 2 warnings.
- Native runtime CLI data-dir focus:
  `uv run pytest tests/test_cli_main.py -q`
  - Result: 3 passed.
- Native runtime CLI parser smoke:
  `uv run python -m engram serve --help` and
  `uv run python -m engram mcp --help`
  - Result: passed. Both help outputs include `--helix-data-dir`.
- Native runtime/lifecycle/doctor operator focus:
  `uv run pytest tests/test_cli_main.py tests/test_lifecycle_cli.py
  tests/test_doctor.py -q`
  - Result: 15 passed.
- Native runtime operator regression:
  `uv run pytest tests/test_cli_main.py tests/test_lifecycle_cli.py
  tests/test_doctor.py tests/test_projected_consolidated_smoke.py
  tests/test_native_surface_parity.py -q`
  - Result: 23 passed, 2 warnings.
- Native Makefile command smoke:
  `make -n up-native NATIVE_DATA_DIR=/private/tmp/engram-make-native-data`
  and
  `make -n mcp-native NATIVE_DATA_DIR=/private/tmp/engram-make-native-data`
  - Result: passed. The dry-run commands route through `engram serve --mode
    helix --helix-data-dir ...` and `engram mcp --mode helix --transport
    streamable-http --helix-data-dir ...`.
- Backend broad non-Docker/non-Helix gate after recall builder extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2446 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after traversal extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2449 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after near-miss extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2453 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after priming extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2457 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after confidence extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2459 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after conversation fingerprint extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2461 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after native doctor smoke alignment:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2462 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after explicit native data-dir CLI alignment:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2464 passed, 41 skipped, 234 deselected, with existing warnings.
- Backend broad non-Docker/non-Helix gate after runtime startup native data-dir alignment:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2466 passed, 41 skipped, 234 deselected, with existing warnings.
- `git diff --check`
  - Result: passed.
- Public installer native-mode alignment:
  `uv run pytest tests/test_public_installer_local_modes.py tests/test_setup.py -q`
  - Result: 21 passed.
  `uv run ruff check tests/test_public_installer_local_modes.py`
  - Result: passed.
  `bash -n scripts/install.sh` and `bash -n installer/engramctl`
  - Result: passed.
- Public installer native-package and runtime verification guard:
  `uv run pytest tests/test_public_installer_local_modes.py
  tests/test_storage_resolver.py tests/test_setup.py -q`
  - Result: 29 passed.
  `uv run ruff check tests/test_public_installer_local_modes.py
  tests/test_storage_resolver.py engram/storage/resolver.py`
  - Result: passed.
  `bash -n installer/engramctl`
  - Result: passed.
- Native transport shortcut and resolver guard:
  `uv run pytest tests/test_storage_resolver.py tests/test_cli_main.py
  tests/test_public_installer_local_modes.py -q`
  - Result: 12 passed.
  `uv run ruff check engram/storage/resolver.py tests/test_storage_resolver.py
  tests/test_public_installer_local_modes.py`
  - Result: passed.
  `make -n up-native` and `make -n mcp-native`
  - Result: passed. Both dry-runs now include
    `ENGRAM_HELIX__TRANSPORT=native`.
  `make -n up-native NATIVE_DATA_DIR=/private/tmp/engram-native-data` and
  `make -n mcp-native NATIVE_DATA_DIR=/private/tmp/engram-native-data`
  - Result: passed. Both dry-runs preserve `--helix-data-dir ...` while forcing
    native transport.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2473 passed, 41 skipped, 234 deselected, with existing warnings.
- Dashboard native smoke default lane:
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx
  src/test/LifecyclePanel.test.tsx`
  - Result: 2 passed, 1 skipped. The native smoke is intentionally skipped
    unless `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1` is set.
- Dashboard native no-bind fixture lane:
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx
  src/test/LifecyclePanel.test.tsx src/test/apiClient.test.ts`
  - Result: 8 passed, 1 skipped. The default native fixture smoke exercises
    native-shaped lifecycle/evaluation/recall payloads without a REST bind; the
    live REST smoke remains intentionally skipped unless
    `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1` is set.
- Dashboard typecheck:
  `pnpm exec tsc --noEmit`
  - Result: passed.
- Dashboard evaluation label contract focus:
  `pnpm test -- --run src/test/components.test.tsx src/test/store.test.ts
  src/test/apiClient.test.ts`
  - Result: 84 passed, with existing React `act(...)` and canvas warnings.
- Missed-recall evaluation backend focus:
  `uv run pytest tests/benchmark/test_metrics.py tests/test_brain_loop_report.py
  tests/test_evaluation_store.py tests/test_api_endpoints.py::TestEvaluation
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_records_recall_evaluation_sample
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 30 passed.
- Missed-recall evaluation backend lint:
  `uv run ruff check engram/benchmark/metrics.py
  engram/evaluation/brain_loop_report.py engram/evaluation/store.py
  engram/evaluation/presenter.py engram/api/evaluation.py engram/mcp/server.py
  engram/evaluation/smoke.py tests/benchmark/test_metrics.py
  tests/test_brain_loop_report.py tests/test_evaluation_store.py
  tests/test_api_endpoints.py tests/test_mcp_tools.py
  tests/test_native_surface_parity.py`
  - Result: passed.
- Missed-recall native PyO3 parity:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- Dashboard missed-recall evaluation contract focus:
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx
  src/test/store.test.ts`
  - Result: 84 passed, with existing React `act(...)` and canvas warnings.
- Dashboard missed-recall typecheck:
  `pnpm exec tsc --noEmit`
  - Result: passed.
- Dashboard missed-recall build:
  `pnpm run build`
  - Result: passed, with existing large chunk warning.
- Full dashboard tests after missed-recall evaluation:
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
- Structural-merge warning cleanup focus:
  `uv run pytest tests/test_structural_merge.py tests/test_structure_embeddings.py
  tests/test_summary_merge.py -q`
  - Result: 60 passed.
- Structural-merge warning cleanup lint:
  `uv run ruff check tests/test_structural_merge.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after missed-recall evaluation and
  structural-merge cleanup:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2505 passed, 43 skipped, 236 deselected, with no warnings reported.
- Dashboard cue-usefulness evaluation focus:
  `pnpm test -- --run src/test/components.test.tsx`
  - Result: 43 passed, with existing React `act(...)` and canvas warnings.
- Dashboard consolidation-calibration evaluation focus:
  `pnpm test -- --run src/test/components.test.tsx`
  - Result: 43 passed, with existing React `act(...)` and canvas warnings.
- Dashboard build after cue-usefulness evaluation UI:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Full dashboard tests after native fixture smoke:
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
- Dashboard build:
  `pnpm run build`
  - Result: passed, with existing large chunk warning.
- Live dashboard native smoke:
  - Seed command passed:
    `uv run python -m engram evaluate --smoke --mode helix --sqlite-path
    /private/tmp/engram-dashboard-native-smoke-labels-20260513.db
    --helix-data-dir /private/tmp/engram-dashboard-native-smoke-data-20260513
    --replace --group-id native_brain --format json`
  - REST startup passed through `.venv/bin/engram serve --mode helix
    --helix-data-dir /private/tmp/engram-dashboard-native-smoke-data-20260513
    --host 127.0.0.1 --port 8102` after approved local bind access.
  - `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1
    VITE_API_URL=http://127.0.0.1:8102 pnpm test -- --run
    src/test/nativeDashboardSmoke.test.tsx`
  - Result: 2 passed. The live smoke reached the native PyO3 REST server,
    fetched lifecycle/evaluation/recall, and rendered the Lifecycle panel.
  - Follow-up default lane:
    `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: 1 passed, 1 skipped.
  - Dashboard typecheck:
    `pnpm exec tsc --noEmit`
  - Result: passed.
- Projection freshness evaluation focus:
  `uv run pytest tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 12 passed.
- Projection freshness lint/typecheck:
  `uv run ruff check engram/storage/helix/graph.py
  engram/evaluation/brain_loop_report.py tests/test_helix_stats.py
  tests/test_brain_loop_report.py`
  - Result: passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
- Projection freshness dashboard focus:
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx`
  - Result: 48 passed, with existing React `act(...)` and canvas warnings.
- Projection freshness native PyO3 parity:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
- Projection backlog evaluation focus:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed.
  `uv run pytest tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 12 passed.
- Projection backlog dashboard focus:
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx
  src/test/store.test.ts`
  - Result: 84 passed, with existing React `act(...)` and canvas warnings.
- Projection backlog native PyO3 parity:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
- Consolidation effect-rate evaluation focus:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed.
  `uv run pytest tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 12 passed.
- Consolidation effect-rate lint/typecheck:
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py`
  - Result: passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
- Consolidation effect-rate dashboard focus:
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx
  src/test/store.test.ts`
  - Result: 84 passed, with existing React `act(...)` and canvas warnings.
- Adjudication-pressure evaluation focus:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed.
  `uv run pytest tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 12 passed.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py`
  - Result: passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx
  src/test/store.test.ts`
  - Result: 84 passed, with existing React `act(...)` and canvas warnings.
- Native graph-embedding fallback noise:
  `uv run pytest tests/test_native_transport.py -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_transport.py tests/test_helix_stats.py
  tests/test_brain_loop_report.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 13 passed.
  `uv run ruff check engram/storage/helix/native_transport.py
  engram/evaluation/brain_loop_report.py tests/test_native_transport.py
  tests/test_brain_loop_report.py`
  - Result: passed.
- Live dashboard native smoke after adjudication-pressure contract:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-dashboard-native-smoke-data-20260513-adjudication
  ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-native-smoke-labels-20260513-adjudication.db
  ENGRAM_EMBEDDING__PROVIDER=noop uv run engram evaluate --smoke --mode helix
  --helix-data-dir /private/tmp/engram-dashboard-native-smoke-data-20260513-adjudication
  --sqlite-path /private/tmp/engram-dashboard-native-smoke-labels-20260513-adjudication.db
  --group-id native_brain --format json --replace`
  - Result: passed; the seeded report included `consolidate.adjudication` and
    no coverage gaps.
  `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1 VITE_API_URL=http://127.0.0.1:8102
  pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: 2 passed against the seeded PyO3 REST server. The repeat server log
    contained normal lifecycle/evaluation/recall/dashboard reads without the
    previous `search_graph_embed_vectors` HNSW errors.
- Broad backend gate after adjudication/native-transport changes:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2506 passed, 43 skipped, 236 deselected.
- Full dashboard tests after adjudication/native-transport changes:
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
- Dashboard build after adjudication/native-transport changes:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Consolidation effect-rate native PyO3 parity:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
- Backend broad non-Docker/non-Helix gate after the Project/Consolidate P3
  evaluation signal additions:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2505 passed, 43 skipped, 236 deselected.
- Full dashboard tests after the Project/Consolidate P3 evaluation signal
  additions:
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
- Dashboard build after the Project/Consolidate P3 evaluation signal additions:
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Recall gate-latency evaluation signal:
  `uv run pytest tests/test_brain_loop_report.py
  tests/test_api_endpoints.py::TestEvaluation
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples -q`
  - Result: 6 passed after adding gate latency and then gate-control coverage.
  `uv run pytest tests/test_brain_loop_report.py tests/test_api_endpoints.py
  tests/test_mcp_tools.py -q`
  - Result: 97 passed, 2 skipped.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py tests/test_api_endpoints.py tests/test_mcp_tools.py`
  - Result: passed.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/store.test.ts
  src/test/components.test.tsx`
  - Result: 84 passed, with existing React `act(...)` and jsdom canvas warnings.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2519 passed, 43 skipped, 236 deselected in 173.79s after the
    smoke verifier hardening.
- Native dashboard fixture Recall gate payload:
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: 1 passed, 1 skipped after rendering the Lifecycle and Evaluation
    panels from native-shaped lifecycle/evaluation/recall payloads.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run`
  - Result: 206 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
- Native dashboard fixture consolidation/episode payloads:
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: 1 passed, 1 skipped after adding native-shaped consolidation
    status/history and episode listing payloads, rendering Lifecycle,
    Evaluation, Consolidation, and Memory Feed panels.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run`
  - Result: 207 passed, 1 skipped, with the existing React `act(...)`, canvas,
    and SVG casing warnings.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning around the 3D/
    knowledge bundles.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2522 passed, 43 skipped, 236 deselected in 245.37s.
  `pnpm run build`
  - Result: passed, with the existing large chunk warning.
- Lifecycle Consolidate phase-error health:
  `uv run pytest tests/test_lifecycle_cli.py -q`
  - Result: 11 passed after completed cycles with phase-level errors were
    marked as `attention` in the shared lifecycle summary and lifecycle
    Markdown printed the first phase issue.
  `uv run ruff check engram/lifecycle_cli.py engram/lifecycle_summary.py
  tests/test_lifecycle_cli.py`
  - Result: passed.
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx`
  - Result: 3 passed after the Brain Loop fallback path displayed the first
    phase-level Consolidate issue and marked the stage as attention.
- Doctor Markdown lifecycle issue focus:
  `uv run pytest tests/test_doctor.py tests/test_lifecycle_cli.py -q`
  - Result: 19 passed after doctor Markdown reused lifecycle cycle/phase issue
    selection for the embedded lifecycle snapshot and doctor warned when the
    lifecycle snapshot itself had attention stages.
  `uv run ruff check engram/doctor.py engram/lifecycle_cli.py
  tests/test_doctor.py tests/test_lifecycle_cli.py`
  - Result: passed.
- Consolidation CLI completed-with-warning focus:
  `uv run pytest tests/test_consolidation_cli.py -q`
  - Result: 3 passed after completed cycles with phase errors printed
    `completed with warnings` and emitted the first phase issue on stderr.
  `uv run ruff check engram/consolidation/cli.py tests/test_consolidation_cli.py`
  - Result: passed.
- Shared consolidation presenter/MCP warning wording:
  `uv run pytest tests/test_consolidation_presenter.py tests/test_consolidation_cli.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_reports_completed_phase_warnings -q`
  - Result: 8 passed.
  `uv run ruff check engram/consolidation/presenter.py
  engram/consolidation/cli.py tests/test_consolidation_presenter.py
  tests/test_consolidation_cli.py tests/test_mcp_tools.py`
  - Result: passed.
- Shared consolidation cycle `phase_issue` contract:
  `uv run pytest tests/test_consolidation_presenter.py
  tests/test_consolidation_cli.py tests/test_lifecycle_cli.py tests/test_doctor.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_reports_completed_phase_warnings -q`
  - Result: 27 passed.
  `uv run ruff check engram/consolidation/presenter.py
  engram/consolidation/cli.py engram/lifecycle_cli.py engram/lifecycle_summary.py
  engram/doctor.py tests/test_consolidation_presenter.py
  tests/test_consolidation_cli.py tests/test_lifecycle_cli.py tests/test_doctor.py
  tests/test_mcp_tools.py`
  - Result: passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run src/test/LifecyclePanel.test.tsx`
  - Result: 3 passed.
- P3 evaluation `phase_issue` display:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 3 passed. Re-run after Consolidate-stage warning status alignment
    stayed at 3 passed.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py`
  - Result: passed. Re-run after Consolidate-stage warning status alignment
    also passed.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/components.test.tsx`
  - Result: 49 passed, with existing jsdom canvas and React `act(...)` warnings.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx
  src/test/components.test.tsx`
  - Result: 2 test files passed; 45 passed, 1 skipped, with existing jsdom
    canvas and React `act(...)` warnings.
- Dashboard Consolidation panel `phase_issue` display:
  `pnpm test -- --run src/test/ConsolidationPanel.test.tsx`
  - Result: 10 passed, with existing React `act(...)` warnings. Re-run after
    warning-state styling stayed at 10 passed.
  `pnpm exec tsc --noEmit`
  - Result: passed. Re-run after warning-state styling also passed.
- Native dashboard fixture completed-with-warning coverage:
  `pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx
  src/test/ConsolidationPanel.test.tsx src/test/apiClient.test.ts
  src/test/components.test.tsx`
  - Result: 4 test files passed; 60 passed, 1 skipped, with existing jsdom
    canvas and React `act(...)` warnings.
  `pnpm exec tsc --noEmit`
  - Result: passed.
  `pnpm test -- --run --maxWorkers=1`
  - Result: 15 test files passed; 212 passed, 1 skipped, with existing jsdom
    canvas, React `act(...)`, and SVG casing warnings.
  `pnpm run build`
  - Result: passed with the existing Vite large-chunk warning.
- Consolidation lifecycle event `phase_issue` payload:
  `uv run pytest
  tests/test_consolidation_engine.py::test_consolidation_lifecycle_result_payloads_preserve_legacy_keys
  tests/test_consolidation_events.py -q`
  - Result: 4 passed.
  `uv run ruff check engram/consolidation/lifecycle.py
  tests/test_consolidation_engine.py tests/test_consolidation_events.py`
  - Result: passed.
- Dashboard quest-mode consolidation warning event:
  `pnpm test -- --run src/test/useWebSocket.test.ts`
  - Result: 8 passed.
  `pnpm exec tsc --noEmit`
  - Result: passed.
- Broad gates after shared `phase_issue` warning contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2533 passed, 43 skipped, 236 deselected after the notification
    group-scope fixes.
  `pnpm run build`
  - Result: passed with the existing Vite large-chunk warning.
  `pnpm test -- --run`
  - Result: runner failure, not assertion failure. Vitest reported worker
    startup timeouts after 97 tests had passed.
  `pnpm test -- --run --maxWorkers=1`
  - Result: 15 test files passed; 212 passed, 1 skipped.
- Recall gate metrics in projected/consolidated smoke:
  `uv run pytest tests/test_projected_consolidated_smoke.py::test_projected_consolidated_smoke_produces_full_report
  tests/test_projected_consolidated_smoke.py::test_evaluate_cli_smoke_load_options_extend_report -q`
  - Result: 2 passed.
  `uv run pytest tests/test_projected_consolidated_smoke.py -q`
  - Result: 6 passed.
  `uv run ruff check engram/evaluation/smoke.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  - Follow-up verifier hardening kept `tests/test_projected_consolidated_smoke.py`
    at 6 passed and ruff passed.
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-native-recall-gate-smoke-data-20260513
  ENGRAM_SQLITE__PATH=/private/tmp/engram-native-recall-gate-smoke-labels-20260513.db
  ENGRAM_EMBEDDING__PROVIDER=noop uv run engram evaluate --smoke --mode helix
  --helix-data-dir /private/tmp/engram-native-recall-gate-smoke-data-20260513
  --sqlite-path /private/tmp/engram-native-recall-gate-smoke-labels-20260513.db
  --group-id native_brain --format json --replace`
  - Result: passed outside the sandbox after the first sandboxed attempt hit the
    existing `uv` cache permission boundary. Native output included
    `recall.trigger_count=1`, analyzer p95 latency, `control.surfaced_count=1`,
    and no coverage gaps.
- Native prospective-memory hard-delete parity:
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- Native prospective-memory hard-delete lint:
  `uv run ruff check tests/test_native_surface_parity.py`
  - Result: passed.
- REST/MCP refresh-context intention contract:
  `uv run ruff check engram/api/knowledge.py tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- REST/MCP intention create acknowledgement contract:
  `uv run ruff check engram/api/knowledge.py engram/mcp/server.py
  tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
- Group-scoped REST notification dismissal:
  `uv run pytest tests/test_websocket.py tests/test_notifications.py::TestNotificationStore
  tests/test_knowledge_api.py::TestNotifications -q`
  - Result: 24 passed.
  `uv run ruff check engram/api/websocket.py engram/notifications/store.py
  engram/api/knowledge.py tests/test_websocket.py tests/test_notifications.py
  tests/test_knowledge_api.py tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_dashboard_websocket_uses_native_group -q`
  - Result: 1 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2533 passed, 43 skipped, 236 deselected.
- Native Helix graph cache group scoping:
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 5 passed.
  `uv run pytest
  tests/test_helix_stats.py
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 6 passed.
  `uv run ruff check engram/storage/helix/graph.py tests/test_helix_stats.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/helix/graph.py
  server/tests/test_helix_stats.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2535 passed, 43 skipped, 236 deselected.
- Memory activation snapshot graph contract:
  `uv run pytest tests/storage/test_activation_store.py -q`
  - Result: 14 passed.
  `uv run pytest tests/storage/test_activation_store.py
  tests/test_activation_api.py tests/test_integration_lite.py -q`
  - Result: 19 passed, 11 skipped.
  `uv run ruff check engram/storage/memory/activation.py
  tests/storage/test_activation_store.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/memory/activation.py
  server/tests/storage/test_activation_store.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2537 passed, 43 skipped, 236 deselected.
- Redis/full activation group-index contract:
  `uv run pytest tests/storage/test_redis_activation_store.py -q`
  - Result: 4 passed.
  `uv run pytest tests/storage/test_activation_store.py
  tests/storage/test_redis_activation_store.py tests/test_activation_api.py -q`
  - Result: 23 passed.
  `uv run ruff check engram/storage/redis/activation.py
  tests/storage/test_redis_activation_store.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/redis/activation.py
  server/tests/storage/test_redis_activation_store.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2541 passed, 43 skipped, 236 deselected.
- Group-aware episode-entity linking:
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 6 passed.
  `uv run pytest tests/test_projection_execution.py
  tests/test_consolidation_replay.py tests/test_benchmark.py
  tests/test_helix_stats.py -q`
  - Result: 37 passed, 6 skipped.
  `uv run pytest tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces -q`
  - Result: 1 passed.
  `uv run pytest tests/test_projection_apply.py
  tests/test_projection_execution.py tests/test_consolidation_replay.py
  tests/test_benchmark.py tests/test_helix_stats.py -q`
  - Result: 41 passed, 6 skipped.
  `uv run ruff check engram/storage/protocols.py
  engram/storage/sqlite/graph.py engram/storage/falkordb/graph.py
  engram/storage/helix/graph.py engram/extraction/apply.py
  engram/consolidation/phases/replay.py engram/benchmark/corpus.py
  tests/test_helix_stats.py tests/test_consolidation_replay.py
  tests/test_projection_apply.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/protocols.py
  server/engram/storage/sqlite/graph.py server/engram/storage/falkordb/graph.py
  server/engram/storage/helix/graph.py server/engram/extraction/apply.py
  server/engram/consolidation/phases/replay.py server/engram/benchmark/corpus.py
  server/tests/test_helix_stats.py server/tests/test_consolidation_replay.py
  server/tests/test_projection_apply.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2542 passed, 43 skipped, 236 deselected.
- Group-aware episode-entity reads:
  `uv run pytest tests/test_consolidation_graph_methods.py
  tests/test_helix_stats.py -q`
  - Result: 17 passed.
  `uv run pytest tests/test_consolidation_graph_methods.py
  tests/test_projection_apply.py tests/test_projection_execution.py
  tests/test_consolidation_replay.py tests/test_benchmark.py
  tests/test_helix_stats.py -q`
  - Result: 52 passed, 6 skipped.
  `uv run ruff check engram/storage/sqlite/graph.py
  engram/storage/falkordb/graph.py engram/storage/helix/graph.py
  tests/test_consolidation_graph_methods.py tests/test_helix_stats.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/sqlite/graph.py
  server/engram/storage/falkordb/graph.py server/engram/storage/helix/graph.py
  server/tests/test_consolidation_graph_methods.py server/tests/test_helix_stats.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2544 passed, 43 skipped, 236 deselected.
- Health graph-store group probe:
  `uv run pytest tests/test_api_endpoints.py::test_health_uses_configured_default_group
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 2 passed.
  `uv run pytest tests/test_api_endpoints.py -q`
  - Result: 39 passed.
  `uv run ruff check engram/api/health.py tests/test_api_endpoints.py`
  - Result: passed.
  `git diff --check -- server/engram/api/health.py
  server/tests/test_api_endpoints.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2545 passed, 43 skipped, 236 deselected.
- Helix all-group entity/episode/analytics/neighborhood read parity:
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 19 passed.
  `uv run ruff check engram/storage/helix/graph.py tests/test_helix_stats.py`
  - Result: passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `git diff --check -- server/engram/storage/helix/graph.py
  server/engram/storage/helix/schema.hx helixdb-cfg/db/schema.hx
  server/tests/test_helix_stats.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2557 passed, 43 skipped, 236 deselected.
- Helix consolidation cycle cache group scoping:
  `uv run pytest tests/test_helix_consolidation_store.py -q`
  - Result: 2 passed.
  `uv run pytest tests/test_consolidation_store.py
  tests/test_helix_consolidation_store.py -q`
  - Result: 15 passed.
  `uv run ruff check engram/storage/helix/consolidation.py
  tests/test_helix_consolidation_store.py`
  - Result: passed.
  `git diff --check -- server/engram/storage/helix/consolidation.py
  server/tests/test_helix_consolidation_store.py docs/CURRENT_HANDOFF.md
  docs/design/brain-runtime-audit.md`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2559 passed, 43 skipped, 236 deselected.
- Helix search native/vector fallback group scoping:
  `uv run pytest tests/storage/helix/test_helix_search_index.py -q`
  - Result: 11 passed, 21 skipped.
  `uv run ruff check engram/storage/helix/search.py
  tests/storage/helix/test_helix_search_index.py`
  - Result: passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2563 passed, 43 skipped, 236 deselected.
- SQLite hybrid-search omitted-group parity:
  `uv run pytest tests/test_sqlite_hybrid_search_group_scope.py -q`
  - Result: 2 passed.
  `uv run ruff check engram/storage/sqlite/hybrid_search.py
  engram/storage/sqlite/vectors.py engram/embeddings/graph/storage.py
  tests/test_sqlite_hybrid_search_group_scope.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2565 passed, 43 skipped, 236 deselected.
- Redis vector embedding omitted-group parity:
  `uv run pytest tests/storage/test_redis_search_group_scope.py -q`
  - Result: 1 passed.
  `uv run ruff check engram/storage/vector/redis_search.py
  tests/storage/test_redis_search_group_scope.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2566 passed, 43 skipped, 236 deselected.
- Static group-scope regression guard:
  `uv run pytest tests/test_group_scope_static_contract.py -q`
  - Result: 1 passed.
  `uv run ruff check tests/test_group_scope_static_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2567 passed, 43 skipped, 236 deselected.
- Showcase evaluation adapter group scoping:
  `uv run pytest tests/benchmark/test_showcase_adapter_group_scope.py -q`
  - Result: 1 passed.
  `uv run pytest
  tests/benchmark/test_showcase_runner.py::test_engram_full_golden_scenarios
  -q`
  - Result: 6 passed.
  `uv run ruff check engram/benchmark/showcase/adapters.py
  tests/benchmark/test_showcase_adapter_group_scope.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2568 passed, 43 skipped, 236 deselected.
- Benchmark default brain-group alignment:
  `uv run pytest tests/benchmark/test_locomo.py
  tests/benchmark/test_memory_need_eval.py
  tests/benchmark/test_echo_chamber_group_scope.py -q`
  - Result: 26 passed.
  `uv run ruff check engram/benchmark/locomo/adapter.py
  engram/benchmark/locomo/runner.py engram/benchmark/memory_need.py
  engram/benchmark/echo_chamber.py tests/benchmark/test_locomo.py
  tests/benchmark/test_memory_need_eval.py
  tests/benchmark/test_echo_chamber_group_scope.py`
  - Result: passed.
  `rg -n "group_id: str = \"default\"|group_id: str \| None = \"default\"|group_id=\"default\""
  server/engram/benchmark -g '*.py'`
  - Result: no matches.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2570 passed, 43 skipped, 236 deselected.
- Explicit recall packet-analysis group scoping:
  `uv run pytest
  tests/test_knowledge_api.py::TestRecall::test_recall_packet_analysis_uses_tenant_group
  tests/test_knowledge_api.py::TestChatRecallHelpers::test_execute_tool_recall_packet_analysis_uses_tool_group
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_recall_packet_analysis_uses_active_group
  -q`
  - Result: 3 passed.
  `uv run pytest tests/test_knowledge_api.py tests/test_mcp_tools.py -q`
  - Result: 107 passed, 2 skipped.
  `uv run ruff check engram/api/knowledge.py engram/mcp/server.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2573 passed, 43 skipped, 236 deselected.
- Explicit chat memory-need group scoping:
  `_analyze_chat_memory_need()` now requires a caller-provided `group_id`.
  The production chat route already passes the active tenant group; focused
  helper tests now do the same so future chat analyzer callers cannot silently
  fall back to raw `default`.
  `uv run pytest tests/test_knowledge_api.py::TestChatMemoryNeedHelpers
  tests/test_knowledge_api.py::TestChatRecallHelpers::test_execute_tool_recall_packet_analysis_uses_tool_group
  -q`
  - Result: 4 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 48 passed.
  `uv run ruff check engram/api/knowledge.py tests/test_knowledge_api.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2580 passed, 43 skipped, 236 deselected in 164.00s.
- Consolidation replay group-scoped extraction and link reads:
  `uv run pytest tests/test_consolidation_replay.py -q`
  - Result: 27 passed.
  `uv run ruff check engram/consolidation/phases/replay.py
  tests/test_consolidation_replay.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2581 passed, 43 skipped, 236 deselected in 124.70s.
- Projection-yield and semantic-transition group-scoped linked-entity reads:
  `uv run pytest
  tests/test_memory_maturation.py::test_semantic_transition_promotes_on_coverage
  tests/test_triage_phase.py::test_triage_projection_outcome_reads_entities_in_cycle_group
  tests/test_episode_worker.py::test_worker_projection_outcome_reads_entities_in_group
  -q`
  - Result: 3 passed.
  `uv run pytest tests/test_memory_maturation.py tests/test_triage_phase.py
  tests/test_episode_worker.py -q`
  - Result: 68 passed.
  `uv run ruff check engram/consolidation/phases/semantic_transition.py
  engram/consolidation/phases/triage.py engram/worker.py
  tests/test_memory_maturation.py tests/test_triage_phase.py
  tests/test_episode_worker.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2583 passed, 43 skipped, 236 deselected in 138.37s.
- Static group-scoped graph-call contract:
  `uv run pytest tests/test_group_scope_static_contract.py -q`
  - Result: 2 passed.
  `uv run ruff check tests/test_group_scope_static_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2584 passed, 43 skipped, 236 deselected in 121.85s.
- Prospective-memory public guidance sync:
  `git diff --check -- README.md skills/engram-memory/SKILL.md
  server/engram/mcp/prompts.py`
  - Result: passed.
  `uv run ruff check engram/mcp/prompts.py`
  - Result: passed.
- Projection plan group metadata:
  `uv run pytest tests/test_projection_planner.py
  tests/test_projection_projector.py tests/test_narrow_adapter.py
  tests/test_projection_execution.py -q`
  - Result: 19 passed.
  `uv run ruff check engram/extraction/models.py
  engram/extraction/planner.py engram/extraction/projector.py
  engram/extraction/narrow_adapter.py engram/graph_manager.py
  tests/test_projection_planner.py tests/test_projection_projector.py
  tests/test_narrow_adapter.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after projection plan group metadata:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2575 passed, 43 skipped, 236 deselected in 135.62s.
- Redis event bridge group routing:
  `uv run pytest tests/test_redis_bridge.py -q`
  - Result: 3 passed.
  `uv run ruff check engram/events/redis_bridge.py tests/test_redis_bridge.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after Redis event bridge routing:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2578 passed, 43 skipped, 236 deselected in 137.34s.
- Evaluation CLI configured default group:
  `uv run pytest
  tests/test_projected_consolidated_smoke.py::test_evaluate_cli_smoke_uses_configured_default_group
  tests/test_projected_consolidated_smoke.py::test_evaluate_cli_from_json_uses_configured_default_group
  tests/test_projected_consolidated_smoke.py::test_evaluate_cli_smoke_flag_passes_helix_mode
  -q`
  - Result: 3 passed.
  `uv run pytest tests/test_projected_consolidated_smoke.py -q`
  - Result: 8 passed.
  `uv run ruff check engram/evaluation/cli.py
  tests/test_projected_consolidated_smoke.py`
  - Result: passed.
- Backend broad non-Docker/non-Helix gate after evaluation CLI default group:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2580 passed, 43 skipped, 236 deselected in 139.39s.
- Recall Gate coverage-gap contract:
  `uv run pytest tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 12 passed.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2585 passed, 43 skipped, 236 deselected in 115.15s.
- Recall Gate runtime-metric persistence:
  `uv run pytest tests/test_brain_loop_report.py tests/test_evaluation_store.py
  tests/test_projected_consolidated_smoke.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_recall_runtime_snapshot
  tests/test_api_endpoints.py::TestEvaluation -q`
  - Result: 25 passed.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  engram/evaluation/store.py engram/evaluation/__init__.py
  engram/evaluation/cli.py engram/evaluation/smoke.py
  engram/api/evaluation.py engram/mcp/server.py
  tests/test_brain_loop_report.py tests/test_evaluation_store.py
  tests/test_projected_consolidated_smoke.py tests/test_mcp_tools.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2593 passed, 43 skipped, 236 deselected in 120.94s.
- SQLite consolidation store borrowed-connection ownership:
  `uv run pytest tests/test_consolidation_store.py -q`
  - Result: 14 passed.
  `uv run pytest tests/test_consolidation_store.py tests/test_evaluation_store.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 28 passed.
  `uv run ruff check engram/consolidation/store.py
  tests/test_consolidation_store.py engram/evaluation/store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2594 passed, 43 skipped, 236 deselected in 160.66s.
- SQLite conversation store borrowed-connection ownership:
  `uv run pytest tests/test_conversations_api.py -q`
  - Result: 4 passed.
  `uv run pytest tests/test_conversations_api.py tests/test_consolidation_store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 32 passed.
  `uv run ruff check engram/storage/sqlite/conversations.py
  tests/test_conversations_api.py engram/consolidation/store.py
  tests/test_consolidation_store.py engram/evaluation/store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2595 passed, 43 skipped, 236 deselected in 122.94s.
- SQLite feedback store borrowed-connection ownership:
  `uv run pytest tests/storage/test_sqlite_feedback_store.py -q`
  - Result: 1 passed.
  `uv run pytest tests/storage/test_sqlite_feedback_store.py
  tests/test_conversations_api.py tests/test_consolidation_store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 33 passed.
  `uv run ruff check engram/storage/sqlite/feedback.py
  tests/storage/test_sqlite_feedback_store.py
  engram/storage/sqlite/conversations.py tests/test_conversations_api.py
  engram/consolidation/store.py tests/test_consolidation_store.py
  engram/evaluation/store.py tests/test_evaluation_store.py
  tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2596 passed, 43 skipped, 236 deselected in 158.77s.
- SQLite atlas store borrowed-connection ownership:
  `uv run pytest tests/storage/test_sqlite_atlas_store.py -q`
  - Result: 1 passed.
  `uv run pytest tests/storage/test_sqlite_atlas_store.py
  tests/storage/test_sqlite_feedback_store.py tests/test_conversations_api.py
  tests/test_consolidation_store.py tests/test_evaluation_store.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 34 passed.
  `uv run ruff check tests/storage/test_sqlite_atlas_store.py
  tests/storage/test_sqlite_feedback_store.py engram/storage/sqlite/atlas.py
  engram/storage/sqlite/feedback.py engram/storage/sqlite/conversations.py
  tests/test_conversations_api.py engram/consolidation/store.py
  tests/test_consolidation_store.py engram/evaluation/store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2597 passed, 43 skipped, 236 deselected in 123.48s.
- SQLite hybrid search owned/borrowed connection ownership:
  `uv run pytest tests/test_sqlite_hybrid_search_group_scope.py -q`
  - Result: 4 passed.
  `uv run pytest tests/test_sqlite_hybrid_search_group_scope.py
  tests/storage/test_sqlite_atlas_store.py tests/storage/test_sqlite_feedback_store.py
  tests/test_conversations_api.py tests/test_consolidation_store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 38 passed.
  `uv run ruff check engram/storage/sqlite/search.py
  engram/storage/sqlite/vectors.py engram/storage/sqlite/hybrid_search.py
  tests/test_sqlite_hybrid_search_group_scope.py
  tests/storage/test_sqlite_atlas_store.py
  tests/storage/test_sqlite_feedback_store.py engram/storage/sqlite/atlas.py
  engram/storage/sqlite/feedback.py engram/storage/sqlite/conversations.py
  tests/test_conversations_api.py engram/consolidation/store.py
  tests/test_consolidation_store.py engram/evaluation/store.py
  tests/test_evaluation_store.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2599 passed, 43 skipped, 236 deselected in 117.07s.
- SQLite borrowed-connection shared contract:
  `uv run pytest tests/storage/test_sqlite_borrowed_connection_contract.py -q`
  - Result: 7 passed.
  `uv run pytest tests/storage/test_sqlite_borrowed_connection_contract.py
  tests/storage/test_sqlite_atlas_store.py tests/storage/test_sqlite_feedback_store.py
  tests/test_conversations_api.py tests/test_consolidation_store.py
  tests/test_evaluation_store.py tests/test_sqlite_hybrid_search_group_scope.py -q`
  - Result: 36 passed.
  `uv run ruff check tests/storage/test_sqlite_borrowed_connection_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2606 passed, 43 skipped, 236 deselected in 112.57s.
- Calibration-quality coverage-gap contract:
  `uv run pytest tests/test_brain_loop_report.py -q`
  - Result: 7 passed.
  `uv run ruff check engram/evaluation/brain_loop_report.py
  tests/test_brain_loop_report.py`
  - Result: passed.
  `uv run pytest tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py tests/test_cli_main.py
  tests/test_lifecycle_cli.py tests/test_doctor.py -q`
  - Result: 38 passed.
  `uv run pytest tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py tests/test_api_endpoints.py
  tests/test_mcp_tools.py -q`
  - Result: 116 passed, 2 skipped.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2608 passed, 43 skipped, 236 deselected in 174.50s.
- Doctor smoke coverage-gap Markdown:
  `uv run pytest tests/test_doctor.py -q`
  - Result: 9 passed.
  `uv run ruff check engram/doctor.py tests/test_doctor.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2609 passed, 43 skipped, 236 deselected in 119.84s.
- Production-wide group-scope static guard:
  `uv run pytest tests/test_group_scope_static_contract.py -q`
  - Result: 2 passed.
  `uv run ruff check tests/test_group_scope_static_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2609 passed, 43 skipped, 236 deselected in 129.01s.
- Dashboard WebSocket tenant-group subscription:
  `uv run pytest tests/test_websocket.py -q`
  - Result: 6 passed.
  `uv run ruff check engram/api/websocket.py tests/test_websocket.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2610 passed, 43 skipped, 236 deselected in 171.41s.
- OIDC configured default-group fallback:
  `uv run pytest tests/security/test_websocket_auth.py tests/test_websocket.py -q`
  - Result: 12 passed.
  `uv run ruff check engram/security/oidc.py engram/security/middleware.py
  engram/api/websocket.py tests/security/test_websocket_auth.py
  tests/test_websocket.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2611 passed, 43 skipped, 236 deselected in 133.79s.
- Helix cue usefulness metric parity:
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 20 passed.
  `uv run ruff check engram/storage/helix/graph.py tests/test_helix_stats.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2612 passed, 43 skipped, 236 deselected in 110.59s.
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-cue-metrics-native-20260514
  uv run python -m engram doctor --mode helix --skip-server --no-smoke
  --format json`
  - Result: passed; native engine initialized with 167 routes.
- Cue usefulness smoke gate and PyO3 native schema rebuild:
  `make build-native`
  - Result: passed; rebuilt and installed `helix_native` after syncing the
    native generated query source with the expanded `EpisodeCue` contract.
  `uv run pytest
  tests/test_projected_consolidated_smoke.py::test_projected_consolidated_smoke_supports_native_helix
  -q`
  - Result: 1 passed after the final metadata-sync rebuild.
  `uv run pytest tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py -q`
  - Result: 39 passed.
  `uv run ruff check engram/storage/helix/graph.py
  engram/evaluation/brain_loop_report.py engram/evaluation/smoke.py
  tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2615 passed, 43 skipped, 236 deselected in 170.86s.
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-cue-metrics-smoke-20260514b
  ENGRAM_SQLITE__PATH=/private/tmp/engram-cue-metrics-smoke-20260514b.db
  uv run python -m engram evaluate --smoke --mode helix
  --helix-data-dir /private/tmp/engram-cue-metrics-smoke-20260514b
  --sqlite-path /private/tmp/engram-cue-metrics-smoke-20260514b.db
  --group-id native_brain --format json --replace`
  - Result: passed; native engine initialized with 168 routes, coverage gaps
    were empty, `cue.surfaced_count=1`, `smoke.cue_feedback_checks=1`,
    `project.projected_count=3`, and `recall.trigger_count=1`.
- Helix schema drift guard:
  `uv run pytest tests/test_helix_schema_contract.py -q`
  - Result: 5 passed after extending the guard to Entity provenance and
    graph-embedding vector deletion.
  `uv run ruff check tests/test_helix_schema_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Earlier result: 2617 passed, 43 skipped, 236 deselected in 114.20s.
- Native entity provenance parity:
  `make build-native`
  - Result: passed; rebuilt and installed `helix_native` from the generated
    source carrying Entity provenance fields and `delete_graph_embed_vector`.
  `uv run python -c "import helix_native, tempfile;
  e=helix_native.HelixEngine(data_dir=tempfile.mkdtemp(prefix='engram-route-count-'),
  num_workers=1); print(len(e.list_routes()));
  print('delete_graph_embed_vector' in e.list_routes())"`
  - Result: 169 routes and `True`.
  `uv run pytest tests/test_helix_schema_contract.py
  tests/test_native_entity_provenance.py -q`
  - Result: 6 passed.
  `uv run pytest
  tests/test_projected_consolidated_smoke.py::test_projected_consolidated_smoke_supports_native_helix
  -q`
  - Result: 1 passed.
  `uv run ruff check tests/test_helix_schema_contract.py
  tests/test_native_entity_provenance.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2621 passed, 43 skipped, 236 deselected in 129.49s.
- Native graph-embedding cleanup parity:
  `uv run pytest
  tests/storage/helix/test_helix_search_index.py::test_native_graph_embed_vectors_round_trip_and_clear
  -q`
  - Result: 1 passed.
  `uv run pytest tests/storage/helix/test_helix_search_index.py -q`
  - Result: 12 passed, 21 skipped.
  `uv run ruff check engram/storage/helix/search.py
  tests/storage/helix/test_helix_search_index.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2622 passed, 43 skipped, 236 deselected in 122.76s.
- Native graph-embedding phase replacement parity:
  `uv run pytest
  tests/test_graph_embed_phase.py::TestHelixDBSync::test_native_full_retrain_replaces_stale_graph_vectors
  -q`
  - Result: 1 passed.
  `uv run pytest tests/test_graph_embed_phase.py -q`
  - Result: 21 passed.
  `uv run ruff check tests/test_graph_embed_phase.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2623 passed, 43 skipped, 236 deselected in 117.99s.
- Native open adjudication/evidence status parity:
  `make build-native`
  - Result: passed; rebuilt and installed `helix_native` with
    `find_evidence_by_status` and `find_adjudications_by_status`.
  `uv run python -c "import helix_native, tempfile;
  e=helix_native.HelixEngine(data_dir=tempfile.mkdtemp(prefix='engram-route-count-'),
  num_workers=1); routes=e.list_routes(); print(len(routes));
  print('find_evidence_by_status' in routes);
  print('find_adjudications_by_status' in routes)"`
  - Result: 171 routes, `True`, `True`.
  `uv run pytest tests/test_native_open_adjudication_status.py
  tests/test_helix_schema_contract.py
  tests/test_helix_stats.py::test_helix_pending_evidence_includes_open_non_pending_statuses
  tests/test_helix_stats.py::test_helix_update_evidence_finds_deferred_open_status
  tests/test_helix_stats.py::test_helix_pending_adjudications_include_deferred_and_error
  -q`
  - Result: 10 passed.
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 25 passed.
  `uv run ruff check engram/storage/helix/graph.py tests/test_helix_stats.py
  tests/test_helix_schema_contract.py tests/test_native_open_adjudication_status.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2628 passed, 43 skipped, 236 deselected in 115.70s.
- Native/lite open adjudication pressure visibility:
  `uv run pytest tests/test_adjudication_metrics.py tests/test_brain_loop_report.py
  tests/test_helix_stats.py::test_helix_stats_counts_episodes_cues_and_projection_yield
  tests/test_native_open_adjudication_status.py -q`
  - Result: 12 passed.
  `uv run pytest tests/test_helix_stats.py -q`
  - Result: 25 passed.
  `uv run ruff check engram/storage/open_work.py engram/storage/sqlite/graph.py
  engram/storage/helix/graph.py engram/evaluation/brain_loop_report.py
  tests/test_adjudication_metrics.py tests/test_brain_loop_report.py
  tests/test_helix_stats.py tests/test_native_open_adjudication_status.py`
  - Result: passed.
  `pnpm test -- --run src/test/apiClient.test.ts`
  - Result: 6 passed.
  `pnpm test -- --run`
  - Result: 214 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG casing warnings.
  `pnpm run build`
  - Result: passed, with the existing Vite large chunk warning.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2630 passed, 43 skipped, 236 deselected in 140.77s.
- MCP consolidation trigger active-store contract:
  `uv run pytest
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_uses_active_audit_store_for_native_graph
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  -q`
  - Result: 2 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run pytest tests/test_mcp_tools.py -q`
  - Result: 61 passed, 2 skipped.
  `uv run pytest tests/test_native_surface_parity.py -q`
  - Result: 5 passed.
  `uv run ruff check engram/mcp/server.py tests/test_mcp_tools.py
  tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 137.37s.
- Evidence/adjudication service boundary:
  `uv run pytest tests/test_edge_adjudication.py tests/test_evidence_adjudication.py
  tests/test_projection_execution.py -q`
  - Result: 16 passed, 7 skipped.
  `uv run pytest
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_surfaces_adjudication_requests
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_adjudicate_evidence_forwards_resolution
  tests/test_knowledge_api.py::TestRemember::test_remember_returns_adjudication_requests
  tests/test_knowledge_api.py::TestRemember::test_adjudicate_endpoint_materializes_resolution
  -q`
  - Result: 4 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py
  engram/ingestion/adjudication_service.py engram/ingestion/projection_execution.py
  tests/test_edge_adjudication.py tests/test_evidence_adjudication.py
  tests/test_projection_execution.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 112.33s.
- Preference feedback service boundary:
  `uv run pytest tests/test_calibrate.py tests/test_feedback_tool.py
  tests/test_preference_boost.py -q`
  - Result: 22 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py
  engram/retrieval/preference_feedback.py tests/test_feedback_tool.py
  tests/test_calibrate.py tests/test_preference_boost.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 112.67s.
- Memory forgetting service boundary:
  `uv run pytest tests/test_mcp_tools.py::TestForgetEntity
  tests/test_mcp_tools.py::TestForgetFact tests/test_knowledge_api.py::TestForget -q`
  - Result: 13 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/forgetting.py
  tests/test_mcp_tools.py tests/test_knowledge_api.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 109.58s.
- Direct entity/fact lookup service boundary:
  `uv run pytest tests/test_mcp_tools.py::TestSearchEntities
  tests/test_mcp_tools.py::TestSearchFacts tests/test_mcp_tools.py::TestResolveEntityName
  tests/test_knowledge_api.py::TestFacts -q`
  - Result: 18 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/lookup.py
  tests/test_mcp_tools.py tests/test_knowledge_api.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 110.70s.
- Agent context builder service boundary:
  `uv run pytest
  tests/test_project_bootstrap.py::test_get_context_project_neighbors_in_layer2
  tests/test_tiered_context.py tests/test_template_briefing.py
  tests/test_variable_resolution.py tests/test_mcp_tools.py::TestGetContext
  tests/test_knowledge_api.py::TestContext -q`
  - Result: 36 passed, 11 skipped.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/context_builder.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 108.87s.
- Prospective-memory intention service boundary:
  `uv run pytest tests/test_prospective_v2.py -q`
  - Result: 39 passed.
  `uv run pytest tests/test_consolidation_finalization.py
  tests/test_lifecycle_cli.py::test_lifecycle_summary_includes_recall_intention_summary -q`
  - Result: 4 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/prospective.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 111.13s.
- Graph-state read service boundary:
  `uv run pytest tests/test_mcp_tools.py::TestGetGraphState
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 6 passed.
  `uv run pytest
  tests/test_lifecycle_cli.py::test_lifecycle_summary_includes_recall_intention_summary
  tests/test_lifecycle_cli.py::test_lifecycle_summary_uses_shared_consolidation_cycle_contract
  tests/test_brain_loop_report.py::test_brain_loop_report_accepts_graph_state_and_lifecycle_cycle_shape
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  -q`
  - Result: 4 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/graph_state.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 112.33s.
- Epistemic route service boundary:
  `uv run pytest
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_route_endpoint_classifies_reconcile
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_route_endpoint_classifies_inspect
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_route_endpoint_exposes_compare_scopes
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 4 passed.
  `uv run pytest tests/test_piggyback_context.py::TestRecallMiddleware::test_adds_recalled_context
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_route_question_auto_observes
  tests/test_epistemic_routing.py -q`
  - Result: 18 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/epistemic_route.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 115.08s.
- Epistemic evidence service boundary:
  `uv run pytest
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_gather_epistemic_evidence_prefers_artifacts_for_project_truth
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 2 passed.
  `uv run pytest
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_route_endpoint_classifies_reconcile
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_route_endpoint_exposes_compare_scopes
  tests/test_epistemic_routing.py tests/test_knowledge_api.py::TestChatContextHelpers -q`
  - Result: 21 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/epistemic_evidence.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 115.05s.
- Runtime-state service boundary:
  `uv run pytest
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_get_runtime_state_reports_artifact_freshness
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_runtime_endpoint_returns_epistemic_metrics
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_gather_epistemic_evidence_prefers_artifacts_for_project_truth
  -q`
  - Result: 3 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/runtime_state.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2631 passed, 43 skipped, 236 deselected in 113.47s.
- Decision materializer service boundary:
  `uv run pytest
  tests/test_mcp_tools.py::TestSearchFacts::test_search_hides_epistemic_facts_by_default
  tests/test_mcp_tools.py::TestSearchFacts::test_question_form_does_not_materialize_decision_entity
  tests/test_mcp_tools.py::TestSearchFacts::test_explicit_commitment_materializes_decision_entity
  tests/test_mcp_tools.py::TestSearchFacts::test_artifact_decision_materializer_links_decision_to_artifact
  tests/test_capture_service.py::test_capture_service_runs_decision_materializer_with_episode_context
  -q`
  - Result: 5 passed.
  `uv run pytest
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_facts_endpoint_hides_epistemic_edges_by_default
  -q`
  - Result: 1 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py
  engram/ingestion/decision_materializer.py tests/test_mcp_tools.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2632 passed, 43 skipped, 236 deselected in 111.64s.
- Consolidation cycle completion service boundary:
  `uv run pytest tests/test_consolidation_completion.py
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_full_cycle_completes
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_successful_cycle_runs_finalization
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_completed_event_includes_finalization_metrics
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_stage3_learning_artifacts_persisted
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_phase_capability_validation_fails_cycle
  -q`
  - Result: 7 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/completion.py tests/test_consolidation_completion.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2634 passed, 43 skipped, 236 deselected in 112.19s.
- Structure-aware entity indexer service boundary:
  `uv run pytest tests/test_entity_indexer.py
  tests/test_structure_embeddings.py tests/test_predicate_enriched_embeddings.py -q`
  - Result: 14 passed.
  `uv run pytest
  tests/test_mcp_tools.py::TestSearchFacts::test_artifact_decision_materializer_links_decision_to_artifact
  tests/test_consolidation_completion.py -q`
  - Result: 3 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py
  engram/ingestion/entity_indexer.py tests/test_entity_indexer.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2635 passed, 43 skipped, 236 deselected in 114.49s.
- Artifact search/read service boundary:
  `uv run pytest tests/test_artifact_search_service.py
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_bootstrap_creates_searchable_artifacts
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_get_runtime_state_reports_artifact_freshness
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_gather_epistemic_evidence_prefers_artifacts_for_project_truth
  tests/test_project_bootstrap.py::test_artifact_search_uses_lexical_fallback_when_index_misses
  -q`
  - Result: 5 passed.
  `uv run pytest
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_artifact_search_endpoint_returns_bootstrapped_hits
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 2 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/artifacts.py
  tests/test_artifact_search_service.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2636 passed, 43 skipped, 236 deselected in 113.89s.
- Project bootstrap service boundary:
  `uv run pytest tests/test_project_bootstrap.py -q`
  - Result: 15 passed.
  `uv run pytest
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_bootstrap_creates_searchable_artifacts
  tests/test_mcp_tools.py::TestEpistemicArtifacts::test_gather_epistemic_evidence_prefers_artifacts_for_project_truth
  tests/test_artifact_search_service.py -q`
  - Result: 3 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/ingestion/project_bootstrap.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2636 passed, 43 skipped, 236 deselected in 114.68s.
- Lightweight entity-probe recall service boundary:
  `uv run pytest tests/test_recall_lite.py -q`
  - Result: 38 passed.
  `uv run pytest tests/test_piggyback_context.py tests/test_autorecall.py -q`
  - Result: 64 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py engram/retrieval/entity_probe.py
  tests/test_recall_lite.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2638 passed, 43 skipped, 236 deselected in 112.33s.
- Consolidation phase catalog boundary:
  `uv run pytest tests/test_consolidation_engine.py -q`
  - Result: 27 passed.
  `uv run pytest tests/test_consolidation_cycle_context.py
  tests/test_consolidation_scheduler.py tests/test_schema_formation.py::test_engine_has_17_phases
  -q`
  - Result: 17 passed, 2 skipped.
  `uv run pytest tests/test_dashboard_phase_contract.py -q`
  - Result: 1 passed.
  `uv run pytest tests/test_memory_maturation.py::test_engine_phase_order -q`
  - Result: 1 passed.
  `uv run ruff check engram/consolidation/engine.py
  engram/consolidation/phase_catalog.py tests/test_consolidation_engine.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2639 passed, 43 skipped, 236 deselected in 115.19s.
- Episode ingestion service boundary:
  `uv run pytest tests/test_episode_ingestion_service.py
  tests/test_cqrs_split.py::TestIngestEpisodeWrapper
  tests/test_knowledge_api.py::TestRemember -q`
  - Result: 9 passed.
  `uv run pytest tests/test_knowledge_api.py::TestObserve
  tests/test_knowledge_api.py::TestRemember tests/test_mcp_tools.py::TestJSONResponses -q`
  - Result: 21 passed, 2 skipped.
  `uv run pytest tests/test_integration_lite.py tests/test_remember_v2.py -q`
  - Result: 12 passed, 12 skipped.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/graph_manager.py
  engram/ingestion/episode_ingestion.py tests/test_episode_ingestion_service.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2641 passed, 43 skipped, 236 deselected in 112.58s.
- Offline replay service boundary:
  `uv run pytest tests/test_offline_replay_service.py
  tests/test_knowledge_api.py::TestReplayQueue::test_replay_queue_uses_current_tenant_group
  -q`
  - Result: 3 passed.
  `uv run pytest tests/test_offline_replay_service.py
  tests/test_knowledge_api.py::TestReplayQueue
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 124 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_rest_surfaces_handle_bounded_remember_recall_load
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/api/knowledge.py engram/ingestion/offline_replay.py
  tests/test_offline_replay_service.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2643 passed, 43 skipped, 236 deselected in 112.22s.
- Capture dedup service boundary:
  `uv run pytest tests/test_capture_dedup.py tests/test_auto_observe.py::TestDedupCache
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_knowledge_api.py::TestReplayQueue::test_replay_queue_uses_current_tenant_group -q`
  - Result: 9 passed.
  `uv run pytest
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 1 passed.
  `uv run ruff check engram/api/knowledge.py engram/ingestion/dedup.py
  tests/test_capture_dedup.py tests/test_auto_observe.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2646 passed, 43 skipped, 236 deselected in 112.42s.
- Native surface manifest coverage:
  `uv run pytest tests/test_native_surface_manifest.py -q`
  - Result: 5 passed.
  `uv run ruff check engram/quality/native_surface_manifest.py
  tests/test_native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_manifest.py
  tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces
  -q`
  - Result: 5 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2650 passed, 43 skipped, 236 deselected in 114.02s.
- GraphManager facade-boundary guard:
  `uv run pytest tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 12 passed.
  `uv run ruff check tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2662 passed, 43 skipped, 236 deselected in 112.40s.
- REST/MCP presenter-boundary guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 12 passed.
  `uv run ruff check tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2674 passed, 43 skipped, 236 deselected in 113.31s.
- Consolidation presenter-boundary guard:
  `uv run pytest tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 6 passed.
  `uv run ruff check tests/test_consolidation_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2680 passed, 43 skipped, 236 deselected in 111.41s.
- Dashboard completion-readiness refresh:
  `pnpm test -- --run`
  - Result: 214 passed, 1 skipped, with existing React `act(...)`, canvas
    `getContext`, and SVG casing warnings.
  `npm run build`
  - Result: passed, with existing large chunk warning.
- Live dashboard native smoke:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-dashboard-live-native-data-20260515
  ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-live-native-labels-20260515.db
  uv run python -m engram evaluate --smoke --mode helix
  --helix-data-dir /private/tmp/engram-dashboard-live-native-data-20260515
  --sqlite-path /private/tmp/engram-dashboard-live-native-labels-20260515.db
  --group-id native_brain --format json --replace`
  - Result: passed; seeded a PyO3 native `native_brain` with no coverage gaps.
  First REST attempt:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native ... uv run engram serve
  --host 127.0.0.1 --port 8102 --mode helix --helix-data-dir ...`
  then
  `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1 VITE_API_URL=http://127.0.0.1:8102
  pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: failed as useful config drift evidence; REST served
    `groupId=default` while the seeded brain was `native_brain`.
  Corrected REST attempt:
  `ENGRAM_MODE=helix ENGRAM_DEFAULT_GROUP_ID=native_brain
  ENGRAM_AUTH__DEFAULT_GROUP_ID=native_brain
  ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-dashboard-live-native-data-20260515
  ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-live-native-labels-20260515.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0
  ENGRAM_ACTIVATION__RERANKER_ENABLED=false
  ENGRAM_ACTIVATION__CONSOLIDATION_ENABLED=false uv run engram serve
  --host 127.0.0.1 --port 8102 --mode helix --helix-data-dir
  /private/tmp/engram-dashboard-live-native-data-20260515`
  then
  `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1 VITE_API_URL=http://127.0.0.1:8102
  pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result: 2 passed.
  - Residual note: shutdown exited cleanly, but logged native `update_evidence`
    decode errors during its shutdown consolidation attempt.
- Native evidence-update null normalization:
  `uv run pytest
  tests/test_helix_stats.py::test_helix_update_evidence_normalizes_null_optional_strings
  tests/test_helix_stats.py::test_helix_update_evidence_finds_deferred_open_status -q`
  - Result: 2 passed.
  `uv run ruff check engram/storage/helix/graph.py tests/test_helix_stats.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2681 passed, 43 skipped, 236 deselected in 141.87s.
  `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1 VITE_API_URL=http://127.0.0.1:8102
  pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`
  - Result after restarting patched native REST with `native_brain` defaults:
    2 passed. Shutdown no longer logged native `update_evidence` decode errors;
    only existing graph-embedding not-persisted/too-few-entities warnings
    remained.
- Native default-group config inheritance:
  `uv run pytest tests/test_config.py -q`
  - Result: 13 passed.
  `uv run ruff check engram/config.py tests/test_config.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2687 passed, 43 skipped, 236 deselected in 160.73s.
- Expanded GraphManager facade-boundary guard:
  `uv run pytest tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 50 passed.
  `uv run ruff check tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2725 passed, 43 skipped, 236 deselected in 125.96s.
- Whole-runtime GraphManager private-field access guard:
  `uv run ruff check tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 91 passed in 0.78s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3303 passed, 43 skipped, 236 deselected in 110.33s.
- MCP identity-core service boundary:
  `uv run pytest tests/test_identity_core_service.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 67 passed.
  `uv run ruff check engram/retrieval/identity_core.py engram/graph_manager.py
  engram/mcp/server.py tests/test_identity_core_service.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2730 passed, 43 skipped, 236 deselected in 122.97s.
- MCP consolidation trigger service boundary:
  `uv run pytest tests/test_consolidation_trigger_service.py tests/test_mcp_tools.py
  -k "mcp_trigger_consolidation" tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result for the matching MCP trigger cases: 3 passed, 136 deselected.
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 75 passed.
  `uv run ruff check engram/consolidation_trigger.py engram/graph_manager.py
  engram/mcp/server.py tests/test_consolidation_trigger_service.py
  tests/test_mcp_tools.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2735 passed, 43 skipped, 236 deselected in 199.69s.
  Latest MCP trigger store-resolution helper check:
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_uses_active_audit_store_for_native_graph
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_reports_completed_phase_warnings
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
  `uv run ruff check engram/consolidation_trigger.py engram/mcp/server.py
  tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP identity-core/consolidation control public-surface boundary:
  `uv run pytest tests/test_identity_core_service.py
  tests/test_consolidation_trigger_service.py tests/test_mcp_tools.py -k
  "identity_core or mcp_trigger_consolidation"
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 7 passed, 190 deselected.
  `uv run ruff check engram/retrieval/identity_core.py
  engram/consolidation_trigger.py engram/mcp/server.py
  tests/test_identity_core_service.py tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py tests/test_mcp_tools.py`
  - Result: passed.
- MCP consolidation status public-surface boundary:
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_consolidation_status_includes_latest_cycle
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 130 passed.
  `uv run ruff check engram/consolidation_trigger.py engram/mcp/server.py
  tests/test_consolidation_trigger_service.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py`
  - Result: passed.
- REST consolidation control/read public-surface boundary:
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_api_endpoints.py::TestConsolidationAPI
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 137 passed.
  `uv run ruff check engram/consolidation_trigger.py
  engram/api/consolidation.py tests/test_consolidation_trigger_service.py
  tests/test_api_endpoints.py tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py`
  - Result: passed.
- MCP entity graph-resource service boundary:
  `uv run pytest tests/test_graph_state_resource_views.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 74 passed.
  `uv run ruff check engram/retrieval/graph_state.py engram/graph_manager.py
  engram/mcp/server.py tests/test_graph_state_resource_views.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2742 passed, 43 skipped, 236 deselected in 134.57s.
- REST/MCP recall-need graph-probe facade:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 73 passed.
  `uv run ruff check engram/api/knowledge.py engram/mcp/server.py
  engram/graph_manager.py tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2744 passed, 43 skipped, 236 deselected in 123.87s.
- REST/MCP intention-list presentation boundary:
  `uv run pytest tests/test_prospective_intention_views.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 79 passed.
  `uv run ruff check engram/retrieval/prospective.py engram/api/knowledge.py
  engram/mcp/server.py engram/graph_manager.py
  tests/test_prospective_intention_views.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2750 passed, 43 skipped, 236 deselected in 116.97s.
- MCP `intend` threshold-default facade:
  `uv run pytest tests/test_prospective_intention_views.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 82 passed.
  `uv run ruff check engram/retrieval/prospective.py engram/mcp/server.py
  engram/graph_manager.py tests/test_prospective_intention_views.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_graph_manager_facade_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2753 passed, 43 skipped, 236 deselected in 117.61s.
- REST/MCP conversation-runtime facade:
  `uv run pytest tests/test_conversation_runtime_service.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 100 passed.
  `uv run pytest tests/test_knowledge_api.py -k "ChatContextHelpers or
  ChatMemoryNeedHelpers or ChatEndpoint or ExecuteTool" -q`
  - Result: 6 passed, 42 deselected.
  `uv run pytest tests/test_autorecall.py tests/test_piggyback_context.py -q`
  - Result: 64 passed.
  `uv run pytest tests/test_mcp_tools.py -k "route_question or observe or
  remember" -q`
  - Result: 2 passed, 1 skipped, 61 deselected.
  `uv run ruff check engram/retrieval/context.py engram/graph_manager.py
  engram/api/knowledge.py engram/mcp/server.py
  tests/test_conversation_runtime_service.py tests/test_knowledge_api.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2775 passed, 43 skipped, 236 deselected in 115.08s.
- Public-surface policy and MCP lifecycle facade:
  `uv run pytest tests/test_public_surface_policy.py tests/test_knowledge_api.py
  tests/test_lifecycle_cli.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 178 passed.
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract -q`
  - Result: 1 passed.
  `uv run ruff check engram/public_surface_policy.py engram/graph_manager.py
  engram/api/knowledge.py engram/mcp/server.py engram/lifecycle_summary.py
  engram/retrieval/graph_state.py tests/test_public_surface_policy.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py tests/test_lifecycle_cli.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2797 passed, 43 skipped, 236 deselected in 145.17s.
- REST entity route service boundary:
  `uv run pytest tests/test_entity_mutation_service.py
  tests/test_graph_state_resource_views.py tests/test_api_endpoints.py -k
  "Entity" tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 33 passed, 138 deselected.
  `uv run ruff check engram/api/entities.py engram/retrieval/graph_state.py
  engram/retrieval/entity_mutation.py engram/graph_manager.py
  tests/test_entity_mutation_service.py tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2809 passed, 43 skipped, 236 deselected in 532.21s.
- REST entity public-surface boundary:
  `uv run pytest tests/test_entity_surface.py
  tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 126 passed.
  `uv run ruff check engram/retrieval/entity_surface.py engram/api/entities.py
  tests/test_entity_surface.py tests/test_api_endpoints.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST admin benchmark-loader service boundary:
  `uv run pytest tests/test_benchmark_loader.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 124 passed.
  `uv run ruff check engram/benchmark_loader.py engram/api/admin.py
  engram/graph_manager.py tests/test_benchmark_loader.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2812 passed, 43 skipped, 236 deselected in 1000.84s.
  Latest route-facing helper check:
  `uv run pytest tests/test_benchmark_loader.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 238 passed.
  `uv run ruff check engram/api/admin.py engram/benchmark_loader.py
  tests/test_benchmark_loader.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST graph route service boundary:
  `uv run pytest tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py -k "GraphNeighborhood or GraphAt"
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 13 passed, 165 deselected.
  `uv run pytest tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 127 passed.
  `uv run pytest tests/test_graph_state_resource_views.py -q`
  - Result: 10 passed.
  `uv run pytest tests/test_api_endpoints.py::TestGraphNeighborhood
  tests/test_api_endpoints.py::TestGraphAt -q`
  - Result: 8 passed.
  `uv run ruff check engram/api/graph.py engram/graph_manager.py
  engram/retrieval/graph_state.py tests/test_graph_state_resource_views.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2821 passed, 43 skipped, 236 deselected in 135.24s.
- MCP graph-state/resource public-surface boundary:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_mcp_tools.py::TestGetGraphState tests/test_graph_state_resource_views.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 142 passed.
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 126 passed.
  `uv run ruff check engram/retrieval/graph_state.py engram/api/graph.py
  engram/mcp/server.py tests/test_mcp_graph_state_surfaces.py
  tests/test_graph_state_resource_views.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP recall-response state boundary:
  `uv run pytest tests/test_recall_response_state.py tests/test_piggyback_context.py
  tests/test_autorecall.py -q`
  - Result: 69 passed.
  `uv run pytest tests/test_mcp_tools.py -k "recall or
  remember_forwards_client_proposals or remember_surfaces_adjudication"
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 20 passed, 1 skipped, 181 deselected.
  `uv run ruff check engram/retrieval/response_state.py
  engram/graph_manager.py engram/mcp/server.py tests/test_recall_response_state.py
  tests/test_piggyback_context.py tests/test_mcp_tools.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2832 passed, 43 skipped, 236 deselected in 441.33s.
- REST dashboard stats service boundary:
  `uv run pytest tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py -k "Stats" tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 8 passed, 179 deselected.
  `uv run pytest tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 135 passed.
  `uv run pytest tests/test_api_endpoints.py::TestStats -q`
  - Result: 6 passed.
  `uv run ruff check engram/api/stats.py engram/graph_manager.py
  engram/retrieval/graph_state.py tests/test_graph_state_resource_views.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2835 passed, 43 skipped, 236 deselected in 322.04s.
- REST activation monitor service boundary:
  `uv run pytest tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py -k "Activation" tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 9 passed, 188 deselected.
  `uv run pytest tests/test_api_endpoints.py::TestActivation
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 142 passed.
  `uv run ruff check engram/api/activation.py engram/graph_manager.py
  engram/retrieval/graph_state.py tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2845 passed, 43 skipped, 236 deselected in 249.85s.
- REST activation curve response surface:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_activation_api.py::TestActivationCurve
  tests/test_api_endpoints.py::TestActivation
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 150 passed.
- REST dashboard graph-state read route-facing helpers:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_api_endpoints.py::TestStats
  tests/test_api_endpoints.py::TestActivation
  tests/test_api_endpoints.py::TestEpisodes tests/test_activation_api.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 173 passed.
  `uv run ruff check engram/api/activation.py engram/api/episodes.py
  engram/api/stats.py engram/retrieval/graph_state.py
  tests/test_mcp_graph_state_surfaces.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST episode dashboard read service boundary:
  `uv run pytest tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py -k "Episodes or episode_summary"
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 7 passed, 193 deselected.
  `uv run pytest tests/test_api_endpoints.py::TestEpisodes
  tests/test_graph_state_resource_views.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 162 passed.
  `uv run ruff check engram/api/episodes.py engram/graph_manager.py
  engram/retrieval/graph_state.py tests/test_graph_state_resource_views.py
  tests/test_api_endpoints.py tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2848 passed, 43 skipped, 236 deselected in 262.89s.
- REST/MCP lifecycle summary service boundary:
  `uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py -k "lifecycle_summary" -q`
  - Result: 4 passed, 63 deselected.
  `uv run pytest tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 143 passed.
  `uv run ruff check engram/api/lifecycle.py engram/mcp/server.py
  engram/graph_manager.py engram/lifecycle_summary.py tests/test_mcp_tools.py
  tests/test_graph_manager_facade_boundaries.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2851 passed, 43 skipped, 236 deselected in 338.44s.
  Latest route-facing REST lifecycle helper check:
  `uv run pytest tests/test_lifecycle_cli.py::test_api_lifecycle_summary_surface_forwards_runtime_context
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 146 passed.
  `uv run ruff check engram/api/lifecycle.py engram/lifecycle_summary.py
  tests/test_lifecycle_cli.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Combined dirty route-helper gate:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_lifecycle_cli.py::test_api_lifecycle_summary_surface_forwards_runtime_context
  tests/test_api_endpoints.py::TestStats
  tests/test_api_endpoints.py::TestActivation
  tests/test_api_endpoints.py::TestEpisodes
  tests/test_api_endpoints.py::TestEntityDetail::test_get_entity_neighbors
  tests/test_api_endpoints.py::TestLifecycleSummary tests/test_activation_api.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 177 passed.
  `uv run ruff check engram/api/activation.py engram/api/entities.py
  engram/api/episodes.py engram/api/lifecycle.py engram/api/stats.py
  engram/lifecycle_summary.py engram/retrieval/graph_state.py
  tests/test_lifecycle_cli.py tests/test_mcp_graph_state_surfaces.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `/Applications/Xcode.app/Contents/Developer/usr/bin/git diff --check`
  - Result: passed.
  Combined dirty route-helper gate after the admin/WebSocket/chat/MCP helper updates:
  `uv run pytest tests/test_benchmark_loader.py tests/test_consolidation_trigger_service.py
  tests/test_recall_surface.py
  tests/test_mcp_graph_state_surfaces.py
  tests/test_lifecycle_cli.py::test_api_lifecycle_summary_surface_forwards_runtime_context
  tests/test_api_endpoints.py::TestStats
  tests/test_api_endpoints.py::TestActivation
  tests/test_api_endpoints.py::TestEpisodes
  tests/test_api_endpoints.py::TestEntityDetail::test_get_entity_neighbors
  tests/test_api_endpoints.py::TestLifecycleSummary tests/test_activation_api.py
  tests/test_websocket.py tests/security/test_websocket_auth.py
  tests/test_knowledge_api.py::TestChatMemoryNeedHelpers
  tests/test_knowledge_api.py::TestChat
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_recall_packet_analysis_uses_active_group
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_includes_failure_errors
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_uses_active_audit_store_for_native_graph
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_trigger_consolidation_reports_completed_phase_warnings
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 226 passed.
  `uv run ruff check engram/api/admin.py engram/api/activation.py
  engram/api/entities.py engram/api/episodes.py engram/api/knowledge.py
  engram/api/lifecycle.py engram/api/stats.py engram/api/websocket.py
  engram/benchmark_loader.py engram/consolidation_trigger.py
  engram/lifecycle_summary.py engram/mcp/server.py
  engram/retrieval/chat_runtime.py engram/retrieval/graph_state.py
  engram/retrieval/recall_surface.py tests/test_benchmark_loader.py
  tests/test_consolidation_trigger_service.py tests/test_knowledge_api.py
  tests/test_lifecycle_cli.py tests/test_recall_surface.py
  tests/test_mcp_graph_state_surfaces.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `rg -n "await manager\\.|manager\\.get_|manager\\.list_|return await manager|= await manager"
  server/engram/api/*.py server/engram/mcp/server.py`
  - Result: no REST API matches remain; remaining matches are MCP recall-need
    graph probe, full auto-recall, session prime, and live-turn piggyback
    compatibility paths.
- MCP lite/medium auto-recall route-helper boundary:
  `uv run pytest tests/test_auto_recall_policy.py
  tests/test_recall_lite.py::TestAutoRecallLiteWiring
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 173 passed.
  `uv run ruff check engram/mcp/server.py engram/retrieval/auto_recall.py
  tests/test_auto_recall_policy.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest direct manager-dispatch scan:
  `rg -n "await manager\\.|manager\\.get_|manager\\.list_|return await manager|= await manager"
  server/engram/api/*.py server/engram/mcp/server.py`
  - Result: no REST API matches remain; remaining MCP matches are the
    recall-need graph probe, full auto-recall, session prime, and live-turn
    piggyback compatibility paths.
- MCP auto-recall/session-prime/middleware dispatch helper boundary:
  `uv run pytest tests/test_auto_recall_policy.py tests/test_autorecall.py
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 245 passed.
  `uv run ruff check engram/mcp/server.py engram/retrieval/auto_recall.py
  tests/test_auto_recall_policy.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_tools.py -k "observe or remember or
  route_question or recall" -q`
  - Result: 6 passed, 2 skipped, 58 deselected.
  `uv run pytest tests/test_recall_lite.py::TestAutoRecallLiteWiring -q`
  - Result: 4 passed.
  `/Applications/Xcode.app/Contents/Developer/usr/bin/git diff --check`
  - Result: passed.
  Latest direct manager-dispatch scan:
  `rg -n "await manager\\.|manager\\.get_|manager\\.list_|return await manager|= await manager"
  server/engram/api/*.py server/engram/mcp/server.py`
  - Result: no matches. REST API routes and `server/engram/mcp/server.py` no
    longer directly dispatch the scanned manager read/write calls.
- REST/MCP post-write adjudication gate and REST offline replay route-helper boundary:
  `uv run pytest tests/test_adjudication_surface.py tests/test_offline_replay_service.py
  tests/test_knowledge_api.py::TestRemember::test_remember_returns_adjudication_requests
  tests/test_knowledge_api.py::TestReplayQueue::test_replay_queue_uses_current_tenant_group
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_surfaces_adjudication_requests
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 171 passed.
  `uv run ruff check engram/api/knowledge.py engram/mcp/server.py
  engram/ingestion/adjudication_surface.py engram/ingestion/offline_replay.py
  tests/test_adjudication_surface.py tests/test_offline_replay_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `/Applications/Xcode.app/Contents/Developer/usr/bin/git diff --check`
  - Result: passed.
  Latest broad route manager-attribute scan:
  `rg -n "\\bmanager\\." server/engram/api/*.py server/engram/mcp/server.py`
  - Result: no route-body manager attribute access remains; remaining hits are
    docstrings only.
- MCP triggered-intention and notification enrichment helper boundary:
  `uv run pytest tests/test_auto_recall_policy.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 206 passed.
  `uv run pytest tests/test_notifications.py::TestNotificationSurfaceService::test_build_mcp_notifications_surface_from_state_uses_surface_service
  tests/test_piggyback_context.py::TestRecallMiddleware::test_adds_memory_notifications
  tests/test_piggyback_context.py::TestRecallMiddleware::test_get_context_notification_fallback
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 158 passed.
  `uv run ruff check engram/mcp/server.py engram/retrieval/auto_recall.py
  engram/notifications/surface.py tests/test_auto_recall_policy.py
  tests/test_notifications.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Broad backend non-Docker/non-external-Helix gate after the route-boundary dirty slice:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3115 passed, 43 skipped, 236 deselected in 130.30s.
- MCP lifecycle-summary public-surface boundary:
  `uv run pytest
  tests/test_lifecycle_cli.py::test_mcp_lifecycle_summary_surface_forwards_store_reader_and_clamped_limits
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_manager_facade
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_clamps_limits
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 122 passed.
  `uv run ruff check engram/lifecycle_summary.py engram/mcp/server.py
  tests/test_lifecycle_cli.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Dashboard WebSocket activation monitor service boundary:
  `uv run pytest
  tests/test_websocket.py::TestWebSocket::test_activation_monitor_snapshot_uses_dashboard_payload
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 57 passed.
  `uv run pytest tests/test_websocket.py tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 69 passed.
  `uv run ruff check engram/api/websocket.py tests/test_websocket.py
  tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest route-facing helper check:
  `uv run pytest tests/test_websocket.py tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 160 passed.
  `uv run ruff check engram/api/websocket.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2853 passed, 43 skipped, 236 deselected in 292.08s.
- Dashboard WebSocket command/event surface boundary:
  `uv run ruff check engram/api/websocket.py engram/api/websocket_surface.py
  tests/test_websocket.py tests/test_websocket_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_websocket_surface.py tests/test_websocket.py
  tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 176 passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3123 passed, 43 skipped, 236 deselected in 127.87s.
- REST/MCP notification surface service boundary:
  `uv run pytest tests/test_notifications.py::TestNotificationSurfaceService
  tests/test_knowledge_api.py::TestNotifications tests/test_piggyback_context.py
  -k "notification" tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 5 passed, 83 deselected.
  `uv run pytest tests/test_notifications.py
  tests/test_knowledge_api.py::TestNotifications tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 133 passed.
  `uv run pytest tests/test_mcp_tools.py -k "notification or recall" -q`
  - Result: 4 passed, 1 skipped, 60 deselected.
  `uv run ruff check engram/notifications/surface.py engram/api/deps.py
  engram/main.py engram/api/knowledge.py engram/mcp/server.py
  tests/test_notifications.py tests/test_knowledge_api.py
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST notification response-envelope helpers:
  `uv run pytest tests/test_notifications.py
  tests/test_knowledge_api.py::TestNotifications
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 169 passed.
  `uv run ruff check engram/notifications/surface.py engram/api/knowledge.py
  tests/test_notifications.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2858 passed, 43 skipped, 236 deselected in 697.63s.
- Dashboard WebSocket notification-dismiss service boundary:
  `uv run pytest tests/test_websocket.py tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 73 passed.
  `uv run pytest tests/test_notifications.py tests/test_websocket.py
  tests/security/test_websocket_auth.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 120 passed.
  `uv run ruff check engram/api/websocket.py tests/test_websocket.py
  tests/security/test_websocket_auth.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check -- server/engram/api/websocket.py
  server/tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2859 passed, 43 skipped, 236 deselected in 706.19s.
- Dashboard WebSocket auth config dependency boundary:
  `uv run pytest tests/test_websocket.py tests/security/test_websocket_auth.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 75 passed.
  `uv run ruff check engram/api/websocket.py tests/test_websocket.py
  tests/security/test_websocket_auth.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2861 passed, 43 skipped, 236 deselected in 422.31s.
- REST knowledge-chat rate-limiter dependency boundary:
  `uv run pytest tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 69 passed.
  `uv run pytest tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 111 passed.
  `uv run ruff check engram/api/deps.py engram/api/knowledge.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2862 passed, 43 skipped, 236 deselected in 858.13s.
- REST health dependency boundary:
  `uv run pytest
  tests/test_api_endpoints.py::test_health_uses_configured_default_group
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 66 passed.
  `uv run pytest tests/test_api_endpoints.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 109 passed.
  `uv run ruff check engram/api/deps.py engram/api/health.py
  tests/test_api_endpoints.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2864 passed, 43 skipped, 236 deselected in 117.15s.
- REST health response surface boundary:
  `uv run pytest tests/test_health_surface.py
  tests/test_api_endpoints.py::test_health_uses_configured_default_group
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
  `uv run ruff check engram/api/health.py engram/api/health_surface.py
  tests/test_health_surface.py tests/test_api_endpoints.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3129 passed, 43 skipped, 236 deselected in 127.64s.
- Generated API route app-state guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 75 passed.
  `uv run ruff check tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `rg -n "_app_state" server/engram/api`
  - Result: only `server/engram/api/deps.py` contains `_app_state`.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 2874 passed, 43 skipped, 236 deselected in 117.13s.
- REST evaluation report consolidation-store facade:
  `uv run pytest
  tests/test_consolidation_engine.py::TestConsolidationEngine::test_recent_evaluation_context_reads_store_snapshots
  tests/test_api_endpoints.py::TestEvaluation
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 81 passed.
  `uv run pytest tests/test_api_endpoints.py tests/test_consolidation_engine.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 148 passed.
  `uv run ruff check engram/consolidation/engine.py engram/api/evaluation.py
  tests/test_consolidation_engine.py tests/test_api_endpoints.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `rg -n "engine\\._store|get_recent_evaluation_context"
  server/engram/api/evaluation.py server/engram/consolidation/engine.py
  server/tests/test_consolidation_engine.py
  server/tests/test_public_surface_presenter_boundaries.py`
  - Result: route now calls `get_recent_evaluation_context`; no route-local
    `engine._store` read remains.
  - Note: the broad non-Docker/non-Helix rerun was interrupted while still
    running when the commit checkpoint was requested; the latest
    route-orchestration broad gate now passes with 3155 passed, 43 skipped, and
    236 deselected.
- Consolidation audit reader and REST/MCP consolidation route cleanup:
  `uv run pytest tests/test_lifecycle_cli.py
  tests/test_consolidation_presenter.py
  tests/test_consolidation_presenter_boundaries.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 101 passed.
  `uv run pytest tests/test_api_endpoints.py::TestConsolidationAPI
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses -q`
  - Result: 19 passed, 2 skipped.
  `uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_learning.py
  tests/test_graph_manager_facade_boundaries.py -q`
  - Result: 119 passed.
  `uv run pytest tests/test_api_endpoints.py
  tests/test_mcp_tools.py::TestJSONResponses -q`
  - Result: 60 passed, 2 skipped.
  `uv run ruff check engram/consolidation/audit_reader.py
  engram/consolidation/engine.py engram/consolidation/presenter.py
  engram/api/consolidation.py engram/mcp/server.py engram/lifecycle_summary.py
  engram/lifecycle_cli.py engram/graph_manager.py
  tests/test_consolidation_presenter.py
  tests/test_consolidation_presenter_boundaries.py
  tests/test_public_surface_presenter_boundaries.py tests/test_mcp_tools.py
  tests/test_lifecycle_cli.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- REST knowledge-chat rate-limit response surface:
  `uv run pytest
  tests/test_knowledge_api.py::TestChat::test_build_api_chat_rate_limit_surface
  tests/test_knowledge_api.py::TestChat::test_chat_rate_limit_returns_shared_surface
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 130 passed.
- REST knowledge-chat rate-limit execution helper:
  `uv run pytest tests/test_knowledge_api.py::TestChat::test_build_api_chat_rate_limit_surface tests/test_knowledge_api.py::TestChat::test_check_api_chat_rate_limit_returns_none_when_allowed tests/test_knowledge_api.py::TestChat::test_check_api_chat_rate_limit_returns_shared_surface_when_blocked tests/test_knowledge_api.py::TestChat::test_chat_rate_limit_returns_shared_surface tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 169 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 61 passed.
- Knowledge-chat persistence scheduler helper:
  `uv run pytest tests/test_chat_persistence.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 175 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 61 passed.
- Knowledge-chat rich tool-event presenter:
  `uv run pytest tests/test_chat_events.py tests/test_recall_presenter.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 86 passed.
  `uv run pytest tests/test_knowledge_api.py::TestChat -q`
  - Result: 6 passed.
  `uv run pytest tests/test_knowledge_api.py tests/test_chat_events.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 131 passed.
  `uv run ruff check engram/retrieval/chat_events.py engram/api/knowledge.py
  tests/test_chat_events.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest tool-result accumulation check:
  `uv run pytest tests/test_chat_events.py tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 173 passed.
  `uv run pytest tests/test_knowledge_api.py tests/test_chat_events.py -q`
  - Result: 62 passed.
  Latest stream-payload helper check:
  `uv run pytest tests/test_chat_events.py tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 172 passed.
  `uv run ruff check engram/api/knowledge.py engram/retrieval/chat_events.py
  tests/test_chat_events.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3138 passed, 43 skipped, 236 deselected in 114.10s.
- Shared storage bootstrap initialization boundary:
  `uv run pytest tests/storage/test_storage_bootstrap.py
  tests/storage/test_sqlite_borrowed_connection_contract.py -q`
  - Result: 12 passed.
  `uv run pytest tests/test_lifecycle_cli.py tests/test_consolidation_cli.py
  tests/test_projected_consolidated_smoke.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_manager_facade
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_clamps_limits -q`
  - Result: 37 passed.
  `uv run pytest tests/test_api_endpoints.py tests/test_conversations_api.py
  tests/test_activation_api.py -q`
  - Result: 53 passed.
  `uv run pytest tests/test_auto_observe.py tests/test_native_surface_manifest.py -q`
  - Result: 16 passed, 3 skipped.
  `uv run ruff check engram/storage/bootstrap.py engram/main.py
  engram/mcp/server.py engram/lifecycle_cli.py engram/evaluation/smoke.py
  engram/consolidation/cli.py tests/storage/test_storage_bootstrap.py`
  - Result: passed.
  Latest companion-store creation follow-up:
  `uv run ruff check engram/storage/bootstrap.py engram/main.py
  engram/mcp/server.py engram/lifecycle_cli.py engram/evaluation/cli.py
  engram/evaluation/smoke.py tests/storage/test_storage_bootstrap.py`
  - Result: passed.
  `uv run pytest tests/storage/test_storage_bootstrap.py -q`
  - Result: 16 passed after the REST atlas/conversation creation helpers and
    borrowed consolidation fallback helpers were added.
  `uv run pytest tests/storage/test_sqlite_borrowed_connection_contract.py -q`
  - Result: 7 passed.
  `uv run pytest tests/test_lifecycle_cli.py
  tests/test_projected_consolidated_smoke.py tests/test_doctor.py
  tests/test_cli_main.py -q`
  - Result: 50 passed.
  `uv run pytest tests/test_mcp_tools.py -q`
  - Result: 64 passed, 2 skipped.
  `uv run pytest tests/test_api_endpoints.py tests/test_conversations_api.py
  tests/test_activation_api.py tests/test_auto_observe.py
  tests/test_native_surface_manifest.py -q`
  - Result: 75 passed, 3 skipped after REST startup switched atlas and
    conversation store creation to the shared bootstrap helper.
  `uv run pytest tests/storage/test_storage_bootstrap.py
  tests/test_consolidation_trigger_service.py tests/test_lifecycle_cli.py
  tests/test_graph_health.py -q`
  - Result: 41 passed, 5 skipped after route-facing borrowed consolidation
    fallbacks moved to bootstrap.
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-bootstrap-shared-stores-20260518.db
  uv run python -m engram lifecycle --mode lite --format json --episodes 1
  --cycles 1`
  - Result: passed and returned the shared lifecycle JSON loop.
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-bootstrap-shared-stores-20260518.db
  uv run python -m engram evaluate --mode lite --no-saved-samples --format json
  --cycles 1`
  - Result: passed and returned the shared brain-loop evaluation JSON loop with
    expected empty-DB coverage gaps.
- Runtime resource shutdown facade:
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_graph_manager_facade_boundaries.py tests/storage/test_storage_bootstrap.py -q`
  - Result: 95 passed.
  `uv run ruff check engram/storage/bootstrap.py engram/graph_manager.py
  engram/mcp/server.py tests/test_mcp_tools.py
  tests/test_graph_manager_facade_boundaries.py tests/storage/test_storage_bootstrap.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3141 passed, 43 skipped, 236 deselected in 125.18s.
- Episode worker runtime-store boundary:
  `uv run pytest tests/test_episode_worker.py tests/test_graph_manager_facade_boundaries.py
  tests/test_group_scope_static_contract.py tests/test_auto_observe.py
  tests/test_rework_integration.py -q`
  - Result: 132 passed, 3 skipped.
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_graph_manager_facade_boundaries.py
  tests/test_api_endpoints.py::test_health_uses_configured_default_group
  tests/test_auto_observe.py tests/test_episode_worker.py -q`
  - Result: 127 passed, 3 skipped.
  `uv run ruff check engram/worker.py engram/ingestion/worker_runtime.py
  engram/graph_manager.py engram/main.py engram/mcp/server.py
  tests/test_episode_worker.py tests/test_graph_manager_facade_boundaries.py
  tests/test_group_scope_static_contract.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3145 passed, 43 skipped, 236 deselected in 128.48s.
- Episode worker auto-capture batching boundary:
  `uv run pytest tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py tests/test_rework_integration.py -q`
  - Result: 134 passed, 3 skipped.
  `uv run ruff check engram/worker.py engram/ingestion/worker_batching.py
  engram/ingestion/worker_runtime.py engram/graph_manager.py engram/main.py
  engram/mcp/server.py tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3147 passed, 43 skipped, 236 deselected in 126.86s.
- Episode worker deterministic scoring boundary:
  `uv run pytest tests/test_worker_scoring.py tests/test_worker_batching.py
  tests/test_episode_worker.py tests/test_graph_manager_facade_boundaries.py
  tests/test_group_scope_static_contract.py tests/test_auto_observe.py
  tests/test_rework_integration.py -q`
  - Result: 137 passed, 3 skipped.
  `uv run ruff check engram/worker.py engram/ingestion/worker_batching.py
  engram/ingestion/worker_runtime.py engram/ingestion/worker_scoring.py
  engram/graph_manager.py engram/main.py engram/mcp/server.py
  tests/test_worker_scoring.py tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3150 passed, 43 skipped, 236 deselected in 138.92s.
- Episode worker projection routing boundary:
  `uv run pytest tests/test_worker_routing.py tests/test_worker_scoring.py
  tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py tests/test_rework_integration.py -q`
  - Result: 142 passed, 3 skipped.
  `uv run ruff check engram/worker.py engram/ingestion/worker_batching.py
  engram/ingestion/worker_routing.py engram/ingestion/worker_runtime.py
  engram/ingestion/worker_scoring.py engram/graph_manager.py engram/main.py
  engram/mcp/server.py tests/test_worker_routing.py tests/test_worker_scoring.py
  tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3155 passed, 43 skipped, 236 deselected in 136.44s.
- Episode worker event parsing and content loading boundary:
  `uv run pytest tests/test_worker_events.py tests/test_worker_routing.py
  tests/test_worker_scoring.py tests/test_worker_batching.py
  tests/test_episode_worker.py tests/test_auto_observe.py tests/test_rework_integration.py -q`
  - Result: 55 passed, 3 skipped.
  `uv run pytest tests/test_worker_events.py tests/test_worker_routing.py
  tests/test_worker_scoring.py tests/test_worker_batching.py
  tests/test_episode_worker.py tests/test_graph_manager_facade_boundaries.py
  tests/test_group_scope_static_contract.py tests/test_auto_observe.py
  tests/test_rework_integration.py -q`
  - Result: 147 passed, 3 skipped.
  `uv run ruff check engram/worker.py engram/ingestion/worker_events.py
  engram/ingestion/worker_batching.py engram/ingestion/worker_routing.py
  engram/ingestion/worker_runtime.py engram/ingestion/worker_scoring.py
  engram/graph_manager.py engram/main.py engram/mcp/server.py
  tests/test_worker_events.py tests/test_worker_routing.py tests/test_worker_scoring.py
  tests/test_worker_batching.py tests/test_episode_worker.py
  tests/test_graph_manager_facade_boundaries.py tests/test_group_scope_static_contract.py
  tests/test_auto_observe.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3160 passed, 43 skipped, 236 deselected in 128.29s.
- Knowledge-chat tool execution payload boundary:
  `uv run pytest tests/test_chat_tools.py
  tests/test_knowledge_api.py::TestChatRecallHelpers tests/test_chat_events.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 105 passed.
  Latest JSON-wrapper cleanup check:
  `uv run pytest tests/test_chat_tools.py
  tests/test_knowledge_api.py::TestChatRecallHelpers
  tests/test_knowledge_api.py::TestChatRecallFeedbackHelpers::test_execute_tool_uses_selected_semantics_when_usage_feedback_enabled
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 166 passed.
  Latest chat-tool schema check:
  `uv run pytest tests/test_chat_tools.py tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 172 passed.
  `uv run ruff check engram/retrieval/chat_tools.py engram/api/knowledge.py
  tests/test_chat_tools.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Knowledge-chat tool-use loop boundary:
  `uv run pytest tests/test_chat_tools.py
  tests/test_knowledge_api.py::TestChat::test_chat_tool_use_loop
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
  `uv run pytest tests/test_chat_tools.py tests/test_chat_events.py
  tests/test_chat_feedback.py tests/test_chat_persistence.py
  tests/test_knowledge_api.py::TestChat -q`
  - Result: 27 passed.
  `uv run pytest tests/test_knowledge_api.py tests/test_chat_tools.py
  tests/test_chat_events.py tests/test_chat_feedback.py tests/test_chat_persistence.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 237 passed.
  `uv run ruff check engram/api/knowledge.py engram/retrieval/chat_tools.py
  tests/test_chat_tools.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Knowledge-chat recall feedback and retry policy boundary:
  `uv run pytest tests/test_chat_feedback.py tests/test_chat_tools.py
  tests/test_knowledge_api.py tests/test_chat_events.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 153 passed.
  `uv run ruff check engram/retrieval/chat_feedback.py
  engram/retrieval/chat_tools.py engram/api/knowledge.py
  tests/test_chat_feedback.py tests/test_chat_tools.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- REST entity response surface:
  `uv run pytest tests/test_entity_surface.py
  tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 143 passed.
- Knowledge-chat memory-need and live-context runtime boundary:
  `uv run pytest tests/test_chat_feedback.py tests/test_chat_tools.py
  tests/test_knowledge_api.py tests/test_chat_events.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 145 passed.
  `uv run ruff check engram/retrieval/chat_runtime.py
  engram/retrieval/chat_feedback.py engram/retrieval/chat_tools.py
  engram/api/knowledge.py tests/test_chat_feedback.py tests/test_chat_tools.py
  tests/test_knowledge_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
  Latest route-facing chat runtime helper check:
  `uv run pytest tests/test_knowledge_api.py::TestChatMemoryNeedHelpers
  tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 161 passed.
  `uv run ruff check engram/api/knowledge.py engram/retrieval/chat_runtime.py
  tests/test_knowledge_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest full chat-focused suite:
  `uv run pytest tests/test_knowledge_api.py tests/test_chat_tools.py
  tests/test_chat_events.py tests/test_chat_feedback.py tests/test_chat_persistence.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 239 passed.
  Latest response-turn orchestration check:
  `uv run pytest tests/test_knowledge_api.py::TestChatMemoryNeedHelpers
  tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 174 passed.
  `uv run ruff check engram/api/knowledge.py engram/retrieval/chat_runtime.py
  tests/test_knowledge_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest chat prompt/message surface check:
  `uv run pytest tests/test_knowledge_api.py::TestChatMemoryNeedHelpers
  tests/test_knowledge_api.py::TestChatContextHelpers
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 170 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 57 passed.
  `uv run ruff check engram/api/knowledge.py engram/retrieval/chat_runtime.py
  tests/test_knowledge_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: superseded by latest route-orchestration broad gate:
    3155 passed, 43 skipped, 236 deselected in 136.44s.
- REST/MCP explicit recall surface boundary:
  `uv run pytest tests/test_chat_feedback.py tests/test_chat_tools.py
  tests/test_knowledge_api.py tests/test_chat_events.py
  tests/test_mcp_tools.py::TestJSONResponses tests/test_autorecall.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 199 passed, 2 skipped.
  `uv run ruff check engram/retrieval/recall_surface.py
  engram/retrieval/chat_runtime.py engram/retrieval/chat_feedback.py
  engram/retrieval/chat_tools.py engram/api/knowledge.py engram/mcp/server.py
  tests/test_chat_feedback.py tests/test_chat_tools.py tests/test_knowledge_api.py
  tests/test_mcp_tools.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- MCP explicit recall enrichment surface:
  `uv run pytest tests/test_recall_surface.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_recall_packet_analysis_uses_active_group
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 121 passed.
  `uv run ruff check engram/retrieval/recall_surface.py engram/mcp/server.py
  tests/test_recall_surface.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest MCP route-local resolver check:
  `uv run pytest tests/test_recall_surface.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_recall_packet_analysis_uses_active_group
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 153 passed.
  `uv run ruff check engram/mcp/server.py engram/retrieval/recall_surface.py
  tests/test_recall_surface.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP explicit recall tool surface:
  `uv run pytest tests/test_recall_surface.py tests/test_autorecall.py::TestRecallSetsLastTime
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_recall_packet_analysis_uses_active_group
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_recall_includes_recalled_context
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 168 passed.
  `uv run ruff check engram/retrieval/recall_surface.py engram/mcp/server.py
  tests/test_recall_surface.py tests/test_autorecall.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3165 passed, 43 skipped, 236 deselected in 132.45s.
- Shared recall-control manager compatibility helpers:
  `uv run pytest tests/test_recall_control_helpers.py tests/test_chat_feedback.py
  tests/test_chat_tools.py tests/test_knowledge_api.py tests/test_chat_events.py
  tests/test_mcp_tools.py::TestJSONResponses tests/test_autorecall.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 201 passed, 2 skipped.
  `uv run ruff check engram/retrieval/control.py
  engram/retrieval/recall_surface.py engram/retrieval/chat_runtime.py
  engram/retrieval/chat_feedback.py engram/retrieval/chat_tools.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_recall_control_helpers.py
  tests/test_chat_feedback.py tests/test_chat_tools.py tests/test_knowledge_api.py
  tests/test_mcp_tools.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- REST/MCP artifact search surface boundary:
  `uv run pytest tests/test_artifact_search_surface.py
  tests/test_artifact_search_service.py
  tests/test_knowledge_api.py::TestEpistemicEndpoints::test_artifact_search_endpoint_returns_bootstrapped_hits
  tests/test_mcp_tools.py::TestEpistemicArtifacts
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 97 passed.
  `uv run ruff check engram/retrieval/artifacts.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_artifact_search_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- MCP artifact-search tool middleware boundary:
  `uv run pytest tests/test_artifact_search_surface.py
  tests/test_mcp_tools.py::TestEpistemicArtifacts tests/test_lookup_surfaces.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 169 passed.
  `uv run ruff check engram/retrieval/artifacts.py engram/retrieval/lookup.py
  engram/mcp/server.py tests/test_artifact_search_surface.py
  tests/test_lookup_surfaces.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Combined lookup/artifact read-tool middleware gate:
  `uv run pytest tests/test_artifact_search_surface.py tests/test_lookup_surfaces.py
  tests/test_mcp_tools.py::TestEpistemicArtifacts tests/test_mcp_tools.py::TestSearchEntities
  tests/test_mcp_tools.py::TestSearchFacts tests/test_mcp_tools.py::TestJSONResponses
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 228 passed, 2 skipped.
  `uv run ruff check engram/retrieval/artifacts.py engram/retrieval/lookup.py
  engram/mcp/server.py tests/test_artifact_search_surface.py tests/test_lookup_surfaces.py
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST/MCP deterministic question-route surface boundary:
  `uv run pytest tests/test_epistemic_route_surface.py
  tests/test_knowledge_api.py::TestEpistemicEndpoints
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 115 passed, 2 skipped.
  `uv run ruff check engram/retrieval/epistemic_route.py
  engram/api/knowledge.py engram/mcp/server.py
  tests/test_epistemic_route_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- MCP question-route tool middleware boundary:
  `uv run pytest tests/test_epistemic_route_surface.py
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_route_question_auto_observes
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 163 passed.
  `uv run ruff check engram/retrieval/epistemic_route.py engram/mcp/server.py
  tests/test_epistemic_route_surface.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST/MCP prospective-memory intention surface boundary:
  `uv run pytest tests/test_prospective_surface.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_native_surface_parity.py -k "intention or intentions" -q`
  - Result: 2 passed, 99 deselected.
  `uv run pytest tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_prospective_surface.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 207 passed, 2 skipped.
  `uv run ruff check engram/retrieval/prospective.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_prospective_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_prospective_surface.py
  tests/test_chat_persistence.py
  tests/test_conversations_api.py::TestConversationOwnership::test_chat_rejects_other_group_conversation_id
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 132 passed.
  `uv run ruff check engram/retrieval/prospective.py
  engram/retrieval/chat_persistence.py engram/api/knowledge.py
  tests/test_prospective_surface.py tests/test_chat_persistence.py
  tests/test_conversations_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- REST prospective-memory/entity/conversation response surfaces:
  `uv run pytest tests/test_prospective_surface.py tests/test_entity_surface.py
  tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations
  tests/test_conversation_persistence.py tests/test_conversations_api.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
- MCP prospective-memory intention response surfaces:
  `uv run pytest tests/test_prospective_surface.py
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed, 2 skipped.
- REST/MCP forget surface boundary:
  `uv run pytest tests/test_forgetting_surface.py tests/test_knowledge_api.py::TestForget
  tests/test_mcp_tools.py::TestForgetEntity tests/test_mcp_tools.py::TestForgetFact
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 112 passed.
  `uv run ruff check engram/retrieval/forgetting.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_forgetting_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- REST forget response surface:
  `uv run pytest tests/test_forgetting_surface.py
  tests/test_knowledge_api.py::TestForget
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 139 passed.
- REST/MCP explicit preference-feedback surface boundary:
  `uv run pytest tests/test_preference_feedback_surface.py tests/test_feedback_tool.py
  tests/test_public_surface_presenter_boundaries.py tests/test_native_surface_parity.py
  -k "feedback" -q`
  - Result: 11 passed, 103 deselected.
  `uv run pytest tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_preference_feedback_surface.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 211 passed, 2 skipped.
  `uv run ruff check engram/retrieval/preference_feedback.py
  engram/api/knowledge.py engram/mcp/server.py
  tests/test_preference_feedback_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `git diff --check`
  - Result: passed.
- MCP preference-feedback error surface:
  `uv run pytest tests/test_preference_feedback_surface.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_feedback_returns_error_payload_for_invalid_rating
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 123 passed.
  `uv run ruff check engram/retrieval/preference_feedback.py
  engram/mcp/server.py tests/test_preference_feedback_surface.py
  tests/test_mcp_tools.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST preference-feedback response surface:
  `uv run pytest tests/test_preference_feedback_surface.py
  tests/test_knowledge_api.py::TestExplicitFeedback
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 135 passed.
- Knowledge-chat conversation persistence boundary:
  `uv run pytest tests/test_chat_persistence.py
  tests/test_conversations_api.py::TestConversationOwnership::test_chat_rejects_other_group_conversation_id
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 93 passed.
  `uv run pytest tests/test_knowledge_api.py::TestChat tests/test_chat_events.py
  tests/test_chat_persistence.py -q`
  - Result: 14 passed.
  `uv run pytest tests/test_knowledge_api.py -q`
  - Result: 48 passed.
  `uv run ruff check engram/retrieval/chat_persistence.py
  engram/api/knowledge.py tests/test_chat_persistence.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest focused prospective/chat error-payload helper gate:
  `uv run pytest tests/test_prospective_surface.py tests/test_chat_persistence.py
  tests/test_conversations_api.py::TestConversationOwnership::test_chat_rejects_other_group_conversation_id
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 132 passed.
- REST conversation persistence boundary:
  `uv run pytest tests/test_conversation_persistence.py
  tests/test_conversations_api.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 127 passed.
  `uv run ruff check engram/retrieval/conversation_persistence.py
  engram/api/conversations.py tests/test_conversation_persistence.py
  tests/test_conversations_api.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST conversation/entity response surface:
  `uv run pytest tests/test_entity_surface.py
  tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations
  tests/test_conversation_persistence.py tests/test_conversations_api.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 157 passed.
- REST/MCP adjudication request surface:
  `uv run pytest tests/test_adjudication_surface.py
  tests/test_knowledge_api.py::TestRemember::test_remember_returns_adjudication_requests
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_surfaces_adjudication_requests
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 104 passed.
  `uv run pytest tests/test_knowledge_api.py::TestRemember
  tests/test_mcp_tools.py::TestJSONResponses -q`
  - Result: 20 passed, 2 skipped.
  `uv run ruff check engram/ingestion/adjudication_surface.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_adjudication_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST/MCP adjudication resolution surface:
  `uv run pytest tests/test_adjudication_surface.py
  tests/test_knowledge_api.py::TestRemember::test_adjudicate_endpoint_materializes_resolution
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_adjudicate_evidence_forwards_resolution
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 117 passed.
  `uv run ruff check engram/ingestion/adjudication_surface.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_adjudication_surface.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST/MCP public Capture write surface:
  `uv run pytest tests/test_capture_surface.py tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py::TestRemember::test_remember_forwards_client_proposals
  tests/test_knowledge_api.py::TestRemember::test_remember_returns_adjudication_requests
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_forwards_client_proposals
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_surfaces_adjudication_requests
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 128 passed.
  `uv run ruff check engram/ingestion/capture_surface.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_capture_surface.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest REST auto-observe capture policy check:
  `uv run pytest tests/test_auto_observe.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: superseded by latest focused capture/static gate:
    186 passed, 3 skipped.
  `uv run ruff check engram/api/knowledge.py engram/ingestion/capture_surface.py
  tests/test_auto_observe.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  Latest REST observe/remember envelope extraction check:
  `uv run pytest tests/test_capture_surface.py
  tests/test_knowledge_api.py::TestObserve tests/test_knowledge_api.py::TestRemember
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 177 passed.
  `uv run ruff check engram/api/knowledge.py engram/ingestion/capture_surface.py
  tests/test_capture_surface.py tests/test_public_surface_presenter_boundaries.py
  tests/test_knowledge_api.py`
  - Result: passed.
  Latest combined REST capture/static gate:
  `uv run pytest tests/test_capture_surface.py tests/test_auto_observe.py
  tests/test_memory_write_presenter.py tests/test_knowledge_api.py::TestObserve
  tests/test_knowledge_api.py::TestRemember
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 196 passed, 3 skipped.
  `uv run ruff check engram/api/knowledge.py engram/ingestion/capture_surface.py
  engram/retrieval/chat_runtime.py tests/test_capture_surface.py
  tests/test_auto_observe.py tests/test_knowledge_api.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP Capture write orchestration surface:
  `uv run pytest tests/test_capture_surface.py
  tests/test_mcp_tools.py::TestJSONResponses tests/test_autorecall.py
  tests/test_piggyback_context.py tests/test_auto_recall_policy.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 275 passed, 2 skipped.
  `uv run pytest tests/test_auto_observe.py::test_observe_response_message
  tests/test_capture_surface.py tests/test_mcp_tools.py::TestJSONResponses
  tests/test_autorecall.py tests/test_piggyback_context.py
  tests/test_auto_recall_policy.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 276 passed, 2 skipped.
  `uv run pytest tests/test_mcp_tools.py -q`
  - Result: 64 passed, 2 skipped.
  `uv run ruff check engram/ingestion/capture_surface.py engram/mcp/server.py
  engram/retrieval/auto_recall.py tests/test_capture_surface.py
  tests/test_auto_observe.py tests/test_mcp_tools.py tests/test_autorecall.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3164 passed, 43 skipped, 236 deselected in 121.46s.
- Combined REST/MCP route-surface extraction gate:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_entity_surface.py tests/test_capture_surface.py
  tests/test_memory_write_presenter.py
  tests/test_adjudication_surface.py tests/test_context_surface.py tests/test_lookup_surfaces.py
  tests/test_project_runtime_surfaces.py tests/test_tiered_context.py
  tests/test_project_bootstrap.py tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations tests/test_knowledge_api.py::TestObserve
  tests/test_knowledge_api.py::TestRemember::test_remember_forwards_client_proposals
  tests/test_knowledge_api.py::TestRemember::test_remember_returns_adjudication_requests
  tests/test_knowledge_api.py::TestRemember::test_adjudicate_endpoint_materializes_resolution
  tests/test_knowledge_api.py::TestEpistemicEndpoints tests/test_knowledge_api.py::TestFacts
  tests/test_mcp_tools.py::TestGetGraphState
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_forwards_client_proposals
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_remember_surfaces_adjudication_requests
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_adjudicate_evidence_forwards_resolution
  tests/test_mcp_tools.py::TestEpistemicArtifacts tests/test_mcp_tools.py::TestSearchEntities
  tests/test_mcp_tools.py::TestSearchFacts
  tests/test_graph_state_resource_views.py
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_search_entities_calls_middleware
  tests/test_piggyback_context.py::TestRecallMiddleware::test_get_context_notification_fallback
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 222 passed, 11 skipped.
  `uv run ruff check` on touched ingestion/retrieval/API/MCP modules and focused tests.
  - Result: passed.
  `/Library/Developer/CommandLineTools/usr/bin/git diff --check`
  - Result: passed.
- REST/MCP live conversation facade helpers:
  `uv run pytest tests/test_conversation_runtime_service.py
  tests/test_knowledge_api.py::TestChatContextHelpers tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 135 passed.
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses
  tests/test_knowledge_api.py::TestChat -q`
  - Result: 22 passed, 2 skipped.
  `uv run ruff check engram/retrieval/context.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_conversation_runtime_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Shared REST/MCP evaluation report assembly:
  `uv run pytest tests/test_evaluation_report_service.py
  tests/test_api_endpoints.py::TestEvaluation
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 93 passed.
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses -q`
  - Result: 16 passed, 2 skipped.
  `uv run ruff check engram/evaluation/report_service.py
  engram/api/evaluation.py engram/mcp/server.py
  tests/test_evaluation_report_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP evaluation report public-surface boundary:
  `uv run pytest tests/test_evaluation_report_service.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_recall_runtime_snapshot
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_evaluation_report_reads_active_consolidation_store
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 126 passed.
  `uv run ruff check engram/evaluation/report_service.py engram/mcp/server.py
  tests/test_evaluation_report_service.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Shared REST/MCP evaluation label writes:
  `uv run pytest tests/test_evaluation_label_service.py
  tests/test_api_endpoints.py::TestEvaluation
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_records_recall_evaluation_sample
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_records_session_continuity_evaluation_sample
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 129 passed.
  `uv run ruff check engram/evaluation/label_service.py
  engram/api/evaluation.py engram/mcp/server.py
  tests/test_evaluation_label_service.py tests/test_api_endpoints.py
  tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Combined service-boundary focused gate:
  `uv run pytest tests/test_conversation_runtime_service.py
  tests/test_adjudication_surface.py tests/test_conversation_persistence.py
  tests/test_chat_persistence.py tests/test_evaluation_label_service.py
  tests/test_evaluation_report_service.py tests/test_auto_recall_policy.py
  tests/test_chat_events.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 139 passed.
  `uv run ruff check` on the touched consolidation/API/evaluation/retrieval/MCP
  modules and focused tests.
  - Result: passed.
- Route-facing public-surface cleanup combined gate:
  `uv run pytest tests/test_preference_feedback_surface.py
  tests/test_recall_surface.py tests/test_identity_core_service.py
  tests/test_consolidation_trigger_service.py tests/test_lifecycle_cli.py
  tests/test_evaluation_report_service.py tests/test_evaluation_label_service.py
  tests/test_api_endpoints.py::TestEvaluation
  tests/test_api_endpoints.py::TestConsolidationAPI tests/test_mcp_tools.py
  tests/test_conversation_persistence.py tests/test_conversations_api.py
  tests/test_notifications.py tests/test_knowledge_api.py::TestNotifications
  tests/test_offline_replay_service.py tests/test_knowledge_api.py::TestReplayQueue
  tests/test_prospective_surface.py tests/test_chat_persistence.py
  tests/test_mcp_graph_state_surfaces.py tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 310 passed, 2 skipped.
  `uv run ruff check` on the touched route-surface modules and focused tests.
  - Result: passed.
  `/Applications/Xcode.app/Contents/Developer/usr/bin/git diff --check`
  - Result: passed.
- REST atlas public-surface helper:
  `uv run pytest tests/test_atlas_surface.py
  tests/test_api_endpoints.py::TestGraphAtlas
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 140 passed.
- Expanded route-facing public-surface gate with activation/atlas/entity/conversation/intention/feedback/forget:
  `uv run pytest tests/test_mcp_graph_state_surfaces.py
  tests/test_activation_api.py::TestActivationCurve
  tests/test_api_endpoints.py::TestActivation tests/test_entity_surface.py
  tests/test_api_endpoints.py::TestEntityDetail
  tests/test_api_endpoints.py::TestEntityMutations
  tests/test_conversation_persistence.py tests/test_conversations_api.py
  tests/test_prospective_surface.py tests/test_forgetting_surface.py
  tests/test_knowledge_api.py::TestForget tests/test_preference_feedback_surface.py
  tests/test_knowledge_api.py::TestExplicitFeedback
  tests/test_knowledge_api.py::TestChat::test_build_api_chat_rate_limit_surface
  tests/test_knowledge_api.py::TestChat::test_chat_rate_limit_returns_shared_surface
  tests/test_recall_surface.py
  tests/test_identity_core_service.py tests/test_consolidation_trigger_service.py
  tests/test_lifecycle_cli.py tests/test_evaluation_report_service.py
  tests/test_evaluation_label_service.py tests/test_api_endpoints.py::TestEvaluation
  tests/test_api_endpoints.py::TestConsolidationAPI tests/test_mcp_tools.py
  tests/test_notifications.py tests/test_knowledge_api.py::TestNotifications
  tests/test_offline_replay_service.py tests/test_knowledge_api.py::TestReplayQueue
  tests/test_chat_persistence.py tests/test_atlas_surface.py
  tests/test_api_endpoints.py::TestGraphAtlas
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_presenter_boundaries.py -q`
  - Result: 386 passed, 2 skipped.
- Broad backend non-Docker/non-external-Helix gate:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3076 passed, 43 skipped, 236 deselected in 151.58s.
- MCP auto-recall policy boundary:
  `uv run pytest tests/test_auto_recall_policy.py tests/test_autorecall.py
  tests/test_piggyback_context.py -q`
  - Result: 79 passed.
  `uv run pytest tests/test_recall_lite.py::TestAutoRecallLiteWiring
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 90 passed.
  `uv run pytest tests/test_mcp_tools.py -k "observe or remember or
  route_question or recall" -q`
  - Result: 6 passed, 2 skipped, 57 deselected.
  `uv run ruff check engram/retrieval/auto_recall.py engram/mcp/server.py
  tests/test_auto_recall_policy.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP auto-recall response enrichment boundary:
  `uv run pytest tests/test_auto_recall_policy.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 121 passed.
  `uv run pytest tests/test_mcp_tools.py -k "observe or remember or
  route_question or recall" -q`
  - Result: 6 passed, 2 skipped, 57 deselected.
  `uv run ruff check engram/retrieval/auto_recall.py engram/mcp/server.py
  tests/test_auto_recall_policy.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP auto-recall middleware execution boundary:
  `uv run pytest tests/test_auto_recall_policy.py tests/test_piggyback_context.py
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 230 passed, 2 skipped.
  `uv run pytest tests/test_auto_recall_policy.py tests/test_piggyback_context.py
  tests/test_mcp_tools.py::TestJSONResponses tests/test_autorecall.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 268 passed, 2 skipped.
  `uv run ruff check engram/retrieval/auto_recall.py engram/mcp/server.py
  tests/test_auto_recall_policy.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3161 passed, 43 skipped, 236 deselected in 125.14s.
- REST/MCP project bootstrap/runtime-state surface boundary:
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_project_bootstrap.py tests/test_knowledge_api.py::TestEpistemicEndpoints
  tests/test_mcp_tools.py::TestEpistemicArtifacts
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 129 passed.
  `uv run ruff check engram/ingestion/project_bootstrap.py
  engram/retrieval/runtime_state.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_project_runtime_surfaces.py tests/test_project_bootstrap.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `/Library/Developer/CommandLineTools/usr/bin/git diff --check`
  - Result: passed.
- REST/MCP public entity/fact lookup surface boundary:
  `uv run pytest tests/test_lookup_surfaces.py tests/test_knowledge_api.py::TestFacts
  tests/test_mcp_tools.py::TestSearchEntities tests/test_mcp_tools.py::TestSearchFacts
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_search_entities_calls_middleware
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 127 passed.
  `uv run ruff check engram/retrieval/lookup.py engram/api/entities.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_lookup_surfaces.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- MCP entity/fact lookup tool middleware boundary:
  `uv run pytest tests/test_lookup_surfaces.py
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_search_entities_calls_middleware
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
  `uv run pytest tests/test_lookup_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 207 passed, 2 skipped.
  `uv run pytest tests/test_mcp_tools.py::TestSearchEntities
  tests/test_mcp_tools.py::TestSearchFacts tests/test_lookup_surfaces.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 180 passed.
  `uv run ruff check engram/retrieval/lookup.py engram/mcp/server.py
  tests/test_lookup_surfaces.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- REST/MCP public agent-context surface boundary:
  `uv run pytest tests/test_context_surface.py tests/test_lookup_surfaces.py
  tests/test_project_runtime_surfaces.py tests/test_tiered_context.py
  tests/test_project_bootstrap.py tests/test_knowledge_api.py::TestEpistemicEndpoints
  tests/test_knowledge_api.py::TestFacts tests/test_mcp_tools.py::TestEpistemicArtifacts
  tests/test_mcp_tools.py::TestSearchEntities tests/test_mcp_tools.py::TestSearchFacts
  tests/test_piggyback_context.py::TestToolMiddlewareIntegration::test_search_entities_calls_middleware
  tests/test_piggyback_context.py::TestRecallMiddleware::test_get_context_notification_fallback
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 159 passed, 11 skipped.
  `uv run ruff check engram/retrieval/context_builder.py
  engram/retrieval/lookup.py engram/ingestion/project_bootstrap.py
  engram/retrieval/runtime_state.py engram/api/entities.py engram/api/knowledge.py
  engram/mcp/server.py tests/test_context_surface.py tests/test_lookup_surfaces.py
  tests/test_project_runtime_surfaces.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `/Library/Developer/CommandLineTools/usr/bin/git diff --check`
  - Result: passed.
- MCP context tool middleware boundary:
  `uv run pytest tests/test_context_surface.py
  tests/test_piggyback_context.py::TestRecallMiddleware::test_get_context_notification_fallback
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 163 passed.
  `uv run ruff check engram/retrieval/context_builder.py engram/mcp/server.py
  tests/test_context_surface.py tests/test_piggyback_context.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
- Combined MCP read-tool middleware gate:
  `uv run pytest tests/test_artifact_search_surface.py tests/test_context_surface.py
  tests/test_epistemic_route_surface.py tests/test_lookup_surfaces.py
  tests/test_mcp_tools.py::TestEpistemicArtifacts tests/test_mcp_tools.py::TestSearchEntities
  tests/test_mcp_tools.py::TestSearchFacts tests/test_mcp_tools.py::TestJSONResponses
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 234 passed, 2 skipped.
  `uv run ruff check engram/retrieval/artifacts.py
  engram/retrieval/context_builder.py engram/retrieval/epistemic_route.py
  engram/retrieval/lookup.py engram/mcp/server.py
  tests/test_artifact_search_surface.py tests/test_context_surface.py
  tests/test_epistemic_route_surface.py tests/test_lookup_surfaces.py
  tests/test_piggyback_context.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_tools.py -q`
  - Result: 64 passed, 2 skipped.
- MCP read-tool middleware regression guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 162 passed.
  This now includes a static check that `server/engram/mcp/server.py` has no
  direct `await _recall_middleware(...)` calls and that REST API routes plus
  `server/engram/mcp/server.py` do not dispatch manager methods directly except
  MCP shutdown resource closing.
- REST evaluation report runtime-boundary guard:
  `uv run pytest tests/test_evaluation_report_service.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 169 passed.
  This covers the route-facing REST evaluation report helper, engine-context
  clamping through `ConsolidationEngine.get_recent_evaluation_context()`, and
  the new static check that REST API routes do not dispatch `engine.*` methods
  directly.
- REST route store/service dispatch guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 165 passed.
  This includes the guard that REST API route modules may obtain stores/services
  through dependencies but must pass them to surface helpers instead of calling
  store/service methods directly in route bodies.
- REST route awaited-helper guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 167 passed.
  This keeps API route functions limited to directly awaiting route-facing
  helpers such as `build_*`, `check_api_chat_rate_limit()`,
  `resolve_chat_conversation()`, and `run_chat_response_turn()`.
- MCP public surface store/session dispatch guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 166 passed.
  This covers decorated MCP tools, resources, and prompts, while leaving MCP
  startup/shutdown store initialization outside the guard.
- MCP public surface awaited-helper guard:
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 168 passed.
  This keeps decorated MCP tools/resources/prompts limited to directly awaiting
  `build_*` helpers plus the consolidation trigger store resolver.
- Combined dirty route-helper lint/check:
  `uv run ruff check` on touched API/retrieval/MCP modules and focused tests.
  - Result: passed.
  `git diff --check`
  - Result: passed.
- Evaluation signal CLI hard gate:
  `uv run ruff check engram/__main__.py engram/evaluation/cli.py tests/test_projected_consolidated_smoke.py tests/test_cli_main.py`
  - Result: passed.
  `uv run python -m engram --help`
  - Result: passed; top-level examples include
    `engram evaluate --require-evaluation-signals`.
  `uv run python -m engram evaluate --help`
  - Result: passed; evaluate options include `--require-evaluation-signals`,
    and `--from-json` documents saved brain-loop report artifacts.
  `uv run pytest tests/test_cli_main.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 20 passed.
  This includes subprocess coverage for
  `python -m engram evaluate --from-json <saved-report>
  --require-evaluation-signals --format json` accepting measured report
  artifacts, exiting non-zero for unmeasured report artifacts, and rejecting
  partial report-shaped JSON with a clear missing-section error.
  `uv run ruff check engram/quality/native_surface_manifest.py tests/test_native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_native_surface_manifest.py -q`
  - Result: 6 passed.
  Combined focused gate:
  `uv run ruff check engram/evaluation/brain_loop_report.py engram/evaluation/cli.py engram/evaluation/smoke.py engram/__main__.py engram/quality/native_surface_manifest.py tests/test_brain_loop_report.py tests/test_projected_consolidated_smoke.py tests/test_cli_main.py tests/test_native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_brain_loop_report.py tests/test_cli_main.py tests/test_projected_consolidated_smoke.py tests/test_native_surface_manifest.py tests/test_decomposer.py::TestDecomposeQuery tests/test_emotional_salience.py::TestTriageFormula::test_async_score_uses_new_weights tests/test_emotional_salience.py::TestTriageFormula::test_personal_floor_kicks_in tests/test_emotional_salience.py::TestTriageFormula::test_emotional_disabled_no_floor tests/test_emotional_salience.py::TestPruneResistance::test_emotional_entity_survives_pruning -q`
  - Result: 53 passed.
  `git diff --check`
  - Result: passed.
- Doctor evaluation-signal readiness output:
  `uv run ruff check engram/doctor.py engram/quality/native_surface_manifest.py
  engram/__main__.py engram/evaluation/cli.py engram/evaluation/smoke.py
  tests/test_doctor.py tests/test_native_surface_manifest.py
  tests/test_cli_main.py tests/test_projected_consolidated_smoke.py`
  - Result: passed.
  `uv run pytest tests/test_doctor.py tests/test_native_surface_manifest.py
  tests/test_cli_main.py tests/test_projected_consolidated_smoke.py -q`
  - Result: 38 passed. This includes direct JSON-output coverage for the doctor
    smoke evaluation-signal metadata plus the existing evaluate hard-gate and
    projected/consolidated smoke coverage.
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-eval-signals-20260518.db
  uv run python -m engram doctor --mode lite --skip-server --format markdown`
  - Result: passed; Markdown output includes `Evaluation signals: 6/6 measured`
    in the Brain Loop Smoke section.
  `ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-eval-signals-json-20260518.db
  uv run python -m engram doctor --mode lite --skip-server --format json`
  - Result: passed; JSON output includes
    `brain_loop_smoke.metadata.evaluation_signals.ready: true`, `measured: 6`,
    `required: 6`, and measured statuses for cue usefulness, projection yield,
    recall quality, false recall, triage calibration, and consolidation effect.
  `uv run python -m engram doctor --help`
  - Result: passed; help output describes the doctor as a lifecycle snapshot and
    brain-loop smoke with evaluation-signal readiness, and `--no-smoke`
    explicitly says it skips the readiness summary.
  `uv run ruff check engram/doctor.py engram/__main__.py tests/test_doctor.py
  tests/test_cli_main.py`
  - Result: passed.
  `uv run pytest tests/test_doctor.py tests/test_cli_main.py -q`
  - Result: 20 passed. This includes fully measured and partially measured
    doctor smoke evaluation-signal metadata coverage, and verifies doctor marks
    the smoke check failed and exits non-zero if the readiness summary is not
    ready. It also covers top-level help and `doctor --help` readiness wording.
- Python 3.13 event-loop test harness cleanup:
  `uv run ruff check tests/test_decomposer.py tests/test_emotional_salience.py`
  - Result: passed.
  `uv run pytest tests/test_decomposer.py::TestDecomposeQuery tests/test_emotional_salience.py::TestTriageFormula::test_async_score_uses_new_weights tests/test_emotional_salience.py::TestTriageFormula::test_personal_floor_kicks_in tests/test_emotional_salience.py::TestTriageFormula::test_emotional_disabled_no_floor tests/test_emotional_salience.py::TestPruneResistance::test_emotional_entity_survives_pruning -q`
  - Result: 10 passed. These tests now use `asyncio.run()` instead of
    `asyncio.get_event_loop().run_until_complete(...)`, which Python 3.13 no
    longer tolerates when no loop is set in the main thread.
- Helix dashboard analytics date-stability fix:
  `uv run ruff check tests/test_helix_stats.py`
  - Result: passed.
  `uv run pytest tests/test_helix_stats.py::test_helix_dashboard_analytics_without_group_use_all_group_queries -q`
  - Result: 1 passed. The unscoped dashboard analytics fixture now freezes
    `utc_now()` so its fixed `2026-05-14` records remain inside the rolling
    growth window after the real calendar advances.
- Broad backend non-Docker/non-external-Helix gate after doctor evaluation-signal
  readiness reporting, defensive doctor failure-path coverage, Helix dashboard
  analytics date-stability, and the shared companion-store bootstrap follow-up:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3235 passed, 43 skipped, 236 deselected in 144.83s.
- Focused explicit-dependency/public-smoke follow-up:
  `uv run ruff check engram/evaluation/smoke.py engram/consolidation/scheduler.py
  engram/main.py engram/notifications/surface.py engram/api/deps.py
  engram/mcp/server.py engram/storage/bootstrap.py
  tests/test_consolidation_scheduler.py tests/test_notifications.py
  tests/test_public_surface_presenter_boundaries.py
  tests/storage/test_storage_bootstrap.py`
  - Result: passed.
  `uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_consolidation_scheduler.py tests/test_notifications.py
  tests/test_public_surface_presenter_boundaries.py tests/test_piggyback_context.py
  tests/storage/test_storage_bootstrap.py -q`
  - Result: 291 passed in 5.19s.
  `git diff --check`
  - Result: passed.
- Broad backend non-Docker/non-external-Helix gate after explicit notification/
  scheduler dependencies and public smoke cue-feedback facade:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3238 passed, 43 skipped, 236 deselected in 169.82s.
- REST/MCP runtime shutdown facade follow-up:
  `uv run ruff check engram/main.py engram/mcp/server.py engram/storage/bootstrap.py
  tests/test_main_shutdown.py tests/storage/test_storage_bootstrap.py
  tests/test_mcp_tools.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_main_shutdown.py tests/storage/test_storage_bootstrap.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 189 passed in 1.66s.
  `uv run pytest tests/test_main_shutdown.py tests/storage/test_storage_bootstrap.py -q`
  - Result after preferring `aclose()` in the shared close helper: 18 passed in
    0.91s.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result after adding runtime shutdown static guards: 174 passed in 1.57s.
  `uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result after moving MCP Redis publisher shutdown onto `close_if_supported()`:
    175 passed in 1.55s.
  `uv run pytest tests/test_main_shutdown.py
  tests/test_consolidation_engine.py::TestShutdownTrigger
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result after adding shutdown-helper delegation coverage:
    183 passed in 1.67s.
  `uv run pytest tests/test_main_shutdown.py tests/storage/test_storage_bootstrap.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_public_surface_presenter_boundaries.py
  tests/test_consolidation_trigger_service.py -q`
  - Result after extracting shutdown consolidation orchestration:
    209 passed in 3.54s.
  `uv run pytest tests/test_api_endpoints.py tests/test_conversations_api.py
  tests/test_activation_api.py tests/test_auto_observe.py tests/test_mcp_tools.py -q`
  - Result: 133 passed, 5 skipped in 68.52s.
  `uv run pytest tests/test_main_shutdown.py tests/storage/test_storage_bootstrap.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result after adding runtime shutdown stop/static guards: 195 passed in
    2.46s.
  `git diff --check`
  - Result: passed.
- Broad backend non-Docker/non-external-Helix gate after REST/MCP runtime
  shutdown facade follow-up:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after shutdown-helper delegation coverage:
    3251 passed, 43 skipped, 236 deselected in 190.15s.
- Knowledge-chat SSE runtime and MCP authority prompt follow-up:
  `uv run ruff check engram/mcp/prompts.py tests/test_mcp_prompts.py
  engram/api/knowledge.py engram/retrieval/chat_runtime.py
  tests/test_chat_runtime_stream.py tests/test_knowledge_api.py
  tests/test_native_surface_parity.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_prompts.py tests/test_chat_runtime_stream.py
  tests/test_knowledge_api.py::TestChat tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 199 passed in 23.97s.
- Broad backend non-Docker/non-external-Helix gate after knowledge-chat SSE
  runtime extraction and MCP authority/onboarding prompt contract:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3256 passed, 43 skipped, 236 deselected in 184.62s.
- MCP claim-authority callable contract follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py engram/mcp/server.py
  engram/mcp/prompts.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py
  tests/test_public_surface_presenter_boundaries.py tests/test_native_surface_parity.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_public_surface_presenter_boundaries.py
  tests/test_native_surface_manifest.py -q`
  - Result: 199 passed in 5.18s.
- Broad backend non-Docker/non-external-Helix gate after MCP
  `claim_authority()` surfaced as a public tool:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3260 passed, 43 skipped, 236 deselected in 226.06s.
- MCP claim-authority adoption harness follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py engram/mcp/server.py
  engram/mcp/prompts.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_native_surface_parity.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 200 passed in 4.02s.
- Broad backend non-Docker/non-external-Helix gate after adding
  `claim_authority()` `agent_protocol` adoption routing:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3261 passed, 43 skipped, 236 deselected in 198.84s.
- MCP authority protocol transcript validation follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py engram/mcp/server.py
  engram/mcp/prompts.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 204 passed in 4.48s.
- Broad backend non-Docker/non-external-Helix gate after MCP authority protocol
  transcript validation:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3265 passed, 43 skipped, 236 deselected in 250.91s.
- MCP stdio-client adoption validation follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py engram/mcp/server.py
  engram/mcp/prompts.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_tools.py
  tests/test_mcp_prompts.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 206 passed in 3.46s.
- Broad backend non-Docker/non-external-Helix gate after MCP stdio-client
  adoption validation:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3267 passed, 43 skipped, 236 deselected in 191.30s.
- MCP adoption installer/prompt guidance follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py engram/mcp/server.py
  engram/mcp/prompts.py engram/setup.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_tools.py
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 224 passed in 4.15s.
- Broad backend non-Docker/non-external-Helix gate after MCP adoption guidance:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3267 passed, 43 skipped, 236 deselected in 154.77s.
- MCP adoption transcript verifier CLI follow-up:
  `uv run ruff check engram/mcp/adoption_cli.py engram/__main__.py
  engram/setup.py tests/test_mcp_adoption_cli.py tests/test_cli_main.py
  tests/test_setup.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_adoption_cli.py tests/test_cli_main.py
  tests/test_setup.py -q`
  - Result: 29 passed in 0.47s.
- Full MCP adoption/verifier focused gate:
  `uv run ruff check engram/retrieval/memory_authority.py
  engram/mcp/adoption_cli.py engram/mcp/server.py engram/mcp/prompts.py
  engram/setup.py engram/__main__.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_setup.py
  tests/test_cli_main.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_cli_main.py
  tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 235 passed in 3.85s.
- Broad backend non-Docker/non-external-Helix gate after adoption verifier CLI:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3272 passed, 43 skipped, 236 deselected in 217.85s.
- Self-describing `claim_authority()` verifier metadata follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py
  tests/test_project_runtime_surfaces.py tests/test_mcp_tools.py
  tests/test_mcp_authority_client_adoption.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_authority_client_adoption.py -q`
  - Result: 13 passed in 1.96s.
- Full MCP adoption/verifier focused gate after self-describing protocol metadata:
  `uv run ruff check engram/retrieval/memory_authority.py
  engram/mcp/adoption_cli.py engram/mcp/server.py engram/mcp/prompts.py
  engram/setup.py engram/__main__.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_setup.py
  tests/test_cli_main.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_cli_main.py
  tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 235 passed in 2.92s.
- Broad backend non-Docker/non-external-Helix gate after self-describing
  `claim_authority()` verifier metadata:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3272 passed, 43 skipped, 236 deselected in 201.99s.
- MCP adoption verifier real-log normalization follow-up:
  `uv run ruff check engram/mcp/adoption_cli.py tests/test_mcp_adoption_cli.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_adoption_cli.py -q`
  - Result: 5 passed in 0.04s.
- Full MCP adoption/verifier focused gate after real-log normalization:
  `uv run ruff check engram/retrieval/memory_authority.py
  engram/mcp/adoption_cli.py engram/mcp/server.py engram/mcp/prompts.py
  engram/setup.py engram/__main__.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_setup.py
  tests/test_cli_main.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_cli_main.py
  tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 236 passed in 3.69s.
- Broad backend non-Docker/non-external-Helix gate after adoption verifier
  real-log normalization:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3273 passed, 43 skipped, 236 deselected in 191.39s.
- `claim_authority()` verifier metadata capture-required follow-up:
  `uv run ruff check engram/retrieval/memory_authority.py
  tests/test_project_runtime_surfaces.py tests/test_mcp_tools.py
  tests/test_mcp_authority_client_adoption.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_authority_client_adoption.py -q`
  - Result: 13 passed in 1.30s.
- Full MCP adoption/verifier focused gate after capture-required metadata:
  `uv run ruff check engram/retrieval/memory_authority.py
  engram/mcp/adoption_cli.py engram/mcp/server.py engram/mcp/prompts.py
  engram/setup.py engram/__main__.py tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py tests/test_mcp_prompts.py tests/test_setup.py
  tests/test_cli_main.py tests/test_native_surface_parity.py
  tests/test_public_surface_presenter_boundaries.py
  engram/quality/native_surface_manifest.py`
  - Result: passed.
  `uv run pytest tests/test_project_runtime_surfaces.py
  tests/test_mcp_authority_client_adoption.py tests/test_mcp_adoption_cli.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_claim_authority_returns_onboarding_contract
  tests/test_mcp_prompts.py tests/test_setup.py tests/test_cli_main.py
  tests/test_native_surface_manifest.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 236 passed in 2.53s.
- Broad backend non-Docker/non-external-Helix gate after capture-required
  verifier metadata:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3273 passed, 43 skipped, 236 deselected in 153.72s.
- Public REST route boundary-map coverage guard:
  `uv run ruff check tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 176 passed in 0.91s.
- Broad backend non-Docker/non-external-Helix gate after the public REST route
  boundary-map coverage guard:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3274 passed, 43 skipped, 236 deselected in 161.21s.
- Dashboard WebSocket route-runtime extraction:
  `uv run ruff check engram/api/websocket.py engram/api/websocket_runtime.py
  tests/test_websocket.py tests/test_websocket_surface.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_websocket.py tests/test_websocket_surface.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 185 passed in 7.65s.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 178 passed in 2.68s.
- Broad backend non-Docker/non-external-Helix gate after dashboard WebSocket
  route-runtime extraction:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3276 passed, 43 skipped, 236 deselected in 291.07s.
- MCP recall middleware adapter flattening:
  `uv run ruff check engram/mcp/server.py
  tests/test_public_surface_presenter_boundaries.py
  tests/test_piggyback_context.py tests/test_auto_recall_policy.py`
  - Result: passed.
  `uv run pytest tests/test_piggyback_context.py tests/test_auto_recall_policy.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 234 passed in 2.96s.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 180 passed in 1.82s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3278 passed, 43 skipped, 236 deselected in 186.20s.
- MCP public nested-orchestration guard:
  `uv run ruff check tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 181 passed in 1.07s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3279 passed, 43 skipped, 236 deselected in 182.60s.
- Dashboard WebSocket auth route-boundary follow-up:
  `uv run ruff check engram/api/websocket.py engram/api/websocket_auth.py
  tests/test_websocket.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_websocket.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 191 passed in 7.78s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3287 passed, 43 skipped, 236 deselected in 155.06s.
- REST knowledge-chat response-surface follow-up:
  `uv run ruff check engram/api/deps.py engram/api/knowledge.py
  engram/retrieval/chat_runtime.py tests/test_chat_runtime_stream.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_chat_runtime_stream.py
  tests/test_knowledge_api.py::TestChat
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 198 passed in 12.18s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3289 passed, 43 skipped, 236 deselected in 138.95s.
- REST health route-boundary follow-up:
  `uv run ruff check engram/api/health.py engram/api/health_runtime.py
  tests/test_health_runtime.py tests/test_api_endpoints.py
  tests/test_health_surface.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_health_runtime.py
  tests/test_api_endpoints.py::test_health_uses_configured_default_group
  tests/test_health_surface.py tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 191 passed in 1.74s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3292 passed, 43 skipped, 236 deselected in 113.72s.
- REST consolidation trigger route-boundary follow-up:
  `uv run ruff check engram/api/consolidation.py
  engram/consolidation_trigger.py tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 198 passed in 1.03s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3295 passed, 43 skipped, 236 deselected in 113.47s.
- REST consolidation status route-boundary follow-up:
  `uv run ruff check engram/api/consolidation.py
  engram/consolidation_trigger.py tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_consolidation_trigger_service.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 201 passed in 1.06s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3298 passed, 43 skipped, 236 deselected in 110.88s.
- Public REST route control-flow guard:
  `uv run ruff check tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 187 passed in 0.98s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3299 passed, 43 skipped, 236 deselected in 111.04s.
- REST atlas warning/response route-boundary follow-up:
  `uv run ruff check engram/api/graph.py engram/retrieval/atlas_surface.py
  tests/test_atlas_surface.py tests/test_public_surface_presenter_boundaries.py`
  - Result: passed.
  `uv run pytest tests/test_atlas_surface.py
  tests/test_public_surface_presenter_boundaries.py -q`
  - Result: 197 passed in 1.09s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result: 3302 passed, 43 skipped, 236 deselected in 113.60s.
- Adoption verifier plaintext transcript support:
  `uv run ruff check engram/mcp/adoption_cli.py tests/test_mcp_adoption_cli.py
  tests/test_cli_main.py`
  - Result: passed.
  `uv run pytest tests/test_mcp_adoption_cli.py tests/test_cli_main.py -q`
  - Result after adding stdin transcript support: 16 passed in 0.35s.
  `uv run pytest tests/test_mcp_adoption_cli.py tests/test_cli_main.py -q`
  - Result after adding self-reported file-memory bypass transcript support:
    17 passed in 0.32s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after adding plaintext phase aliases:
    3280 passed, 43 skipped, 236 deselected in 175.72s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after adding structured transcript parse failures:
    3282 passed, 43 skipped, 236 deselected in 221.46s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after adding stdin transcript support:
    3283 passed, 43 skipped, 236 deselected in 175.94s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after adding self-reported file-memory bypass transcript support:
    3284 passed, 43 skipped, 236 deselected in 118.97s.
  `uv run ruff check engram/mcp/adoption_cli.py tests/test_mcp_adoption_cli.py
  tests/test_cli_main.py`
  - Result after adding the copied Claude file-memory bypass transcript
    regression: passed.
  `uv run pytest tests/test_mcp_adoption_cli.py tests/test_cli_main.py -q`
  - Result after adding the copied Claude file-memory bypass transcript
    regression: 18 passed in 0.30s.
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  - Result after adding the copied Claude file-memory bypass transcript
    regression: 3300 passed, 43 skipped, 236 deselected in 111.42s.
- Dashboard calibration-quality UI contract:
  `pnpm test -- --run src/test/components.test.tsx`
  - Result: 45 passed, with existing canvas/act warnings.
  `pnpm test -- --run src/test/apiClient.test.ts src/test/store.test.ts
  src/test/nativeDashboardSmoke.test.tsx`
  - Result: 44 passed, 1 skipped.
  `npm run build`
  - Result: passed, with existing large chunk warning.
  `pnpm test -- --run`
  - Result: 214 passed, 1 skipped, with existing canvas/act warnings.
- Dashboard verification refresh after the current REST/MCP adoption and
  route-boundary pass:
  `pnpm test -- --run`
  - Result: 214 passed, 1 skipped, with existing React `act(...)`, canvas, and
    SVG/jsdom warning noise.
  `pnpm build`
  - Result: passed, with the existing Vite large chunk warning.
- Dashboard verification refresh after adoption-template and full-benchmark
  evidence updates:
  `pnpm test -- --run`
  - Result: 214 passed, 1 skipped, with the same existing React `act(...)`,
    canvas, and SVG/jsdom warning noise.
  `pnpm build`
  - Result: passed, with the existing Vite large chunk warning.

## Next Best Task

Continue P3 from the audit, with the recent native parity and evaluation
visibility work treated as done:

1. Pause broad recall extraction unless a concrete small boundary is obvious.
   Entity-linked and temporal episode traversal, near-miss formatting, priming
   updates, relevance-confidence, conversation fingerprint recording, and
   working-memory writes are now out of `GraphManager`. Entity interaction
   telemetry/recall-need samples and true entity access/reconsolidation marking
   are also out of the facade. Cue feedback and hot-cue scheduling are also now
   in `RecallCueFeedbackRecorder`, and explicit post-response feedback is now
   in `RecallMemoryInteractionApplier`. Current-state result selection is now
   in `retrieval/result_selection.py`, and request policy is now in
   `retrieval/request_policy.py`. Primary result materialization is now in
   `RecallPrimaryResultMaterializer`, and near-miss materialization is now in
   `RecallNearMissMaterializer`. The post-primary tail is now in
   `RecallPostProcessor`, and request policy, retrieval invocation,
   primary/near-miss splitting, primary materialization, post-processing, and
   final result/near-miss return are now in `RecallService`. Lightweight
   `recall_lite`/`recall_medium` entity-probe behavior is now in
   `retrieval/entity_probe.py` as well. Pause further Recall extraction unless
   another smaller concrete contract emerges; the current facade is thin enough
   to shift attention back to native parity or small
   consolidation contracts.
2. Keep expanding use of `sync_projection_state()` if future projection paths
   surface direct episode/cue drift.
3. Keep `ConsolidationEngine` as the cycle-loop facade unless a concrete
   lifecycle concern can be extracted with a small, tested boundary. The
   post-cycle pinned-context finalizer is now observable through
   `consolidation.completed`, and requested phase names now fail fast when they
   are outside the engine's shared phase registry. Scheduler tiers and engine
   phase-order tests also read that registry now. Concrete runtime phase
   construction is now in `consolidation/phase_catalog.py`, so do not repeat
   those slices.
4. Keep the opt-in dashboard native smoke in the regression rotation when local
   REST bind access is available:
   `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1 VITE_API_URL=http://127.0.0.1:8102
   pnpm test -- --run src/test/nativeDashboardSmoke.test.tsx`.
5. Continue PyO3 native parity with any remaining REST/MCP/dashboard surfaces
   that are not yet covered by the populated native parity tests, or with a
   small consolidation contract. REST health, REST admin benchmark-loader plumbing,
   project bootstrap/artifact search across REST and MCP, conversation
   persistence/update/delete, chat stream persistence, live dashboard native
   smoke, REST consolidation trigger/status/history/detail, REST notifications,
   REST atlas materialization,
   no-bind dashboard native fixture coverage for lifecycle/evaluation/recall/
   episode/consolidation payloads and the Lifecycle/Evaluation/Consolidation/
   Memory Feed panels,
   dashboard WebSocket group scoping, runtime-state freshness, MCP consolidation
   controls, route planning, context assembly, MCP notification/auto-recall piggybacking,
   direct entity/fact lookup, graph detail/resources, MCP graph state, MCP graph
   stats resource, REST entity mutation, REST auto-observe, REST/MCP
   remember/observe writes, MCP evaluation-label writes, MCP route auto-observe,
   REST/MCP image/file attachment capture,
   prospective-memory intentions including refresh-context and hard delete, edge adjudication resolution,
   forget/feedback/identity-core mutations, native graph-embedding sync/read/clear,
   native graph-embedding full-retrain replacement,
   native open evidence/adjudication status queues,
   native/lite open adjudication stats/report visibility,
   MCP consolidation trigger active-store persistence,
   evidence/adjudication service boundary,
   preference feedback service boundary,
   memory forgetting service boundary,
   direct entity/fact lookup service boundary,
   agent context builder service boundary,
   prospective-memory intention service boundary,
   graph-state read service boundary,
   epistemic route service boundary,
   epistemic evidence service boundary,
   runtime-state service boundary,
   decision materializer service boundary,
   consolidation cycle completion service boundary,
   structure-aware entity indexer service boundary,
   artifact search/read service boundary,
   project bootstrap service boundary,
   multi-batch ASGI load/reopen, operator load smoke, the duration-based
   operator recall-soak control, and an actual one-hour native recall soak are
   now covered. The next good candidates are another currently uncovered
   REST/MCP surface or a small consolidation contract, not
   another repeat of rest-health/admin-benchmark/lifecycle/evaluation/recall/bootstrap/artifact
   search/consolidation-read/notifications/rest-dashboard-reads/episode-list-filters/rest-atlas/
   dashboard-websocket/native-dashboard-fixture/runtime-state/mcp-consolidation/
   mcp-consolidation-trigger-active-store/
   route/conversation/chat/context/mcp-notifications/entity/fact/graph-detail/
   graph-neighborhood/graph-temporal/graph-state/graph-stats-resource/
   entity-mutation/auto-observe/write/observe/mcp-evaluation-writes/
   route-auto-observe/attachment/intentions/adjudication/forget/feedback/
   identity-core/graph-embedding-cleanup/graph-embedding-phase/open-adjudication-statuses/
   open-adjudication-stats/evidence-adjudication-service/preference-feedback-service/
   memory-forgetting-service/entity-fact-lookup-service/context-builder-service/
   prospective-memory-service/graph-state-service/epistemic-route-service/
   epistemic-evidence-service/runtime-state-service/decision-materializer-service/
   consolidation-cycle-completion-service/entity-indexer-service/artifact-search-service/
   project-bootstrap-service/entity-probe-service/phase-catalog-service/
   episode-ingestion-service/offline-replay-service/capture-dedup-service/
   native-surface-manifest/graph-manager-facade-boundary-guard/
   public-surface-presenter-boundary-guard/
   consolidation-presenter-boundary-guard/native-evidence-update-null-normalization/
   default-group-config-inheritance/expanded-graph-manager-facade-boundary-guard/
   mcp-identity-core-service-boundary/mcp-consolidation-trigger-service-boundary/
   mcp-entity-resource-service-boundary/rest-mcp-graph-probe-facade/
   rest-mcp-intention-list-presentation-boundary/mcp-intend-threshold-default-facade/
   rest-mcp-conversation-runtime-facade/public-surface-policy-facade/
   rest-mcp-lifecycle-summary-facade/rest-entity-route-service-boundary/
   rest-admin-benchmark-loader-service-boundary/rest-graph-route-service-boundary/
   mcp-recall-response-state-boundary/rest-dashboard-stats-service-boundary/
   rest-activation-monitor-service-boundary/rest-episode-dashboard-read-service-boundary/
   consolidation-audit-reader-boundary/
   knowledge-chat-event-presenter-boundary/
   mcp-auto-recall-policy-boundary/
   websocket-activation-monitor-service-boundary/rest-mcp-notification-surface-boundary/
   websocket-notification-dismiss-surface-boundary/
   websocket-auth-config-dependency-boundary/
   knowledge-chat-rate-limiter-dependency-boundary/rest-health-dependency-boundary/
   generated-api-route-app-state-guard/
   rest-mcp-runtime-shutdown-stop-close-boundary/
   shutdown-consolidation-helper-boundary/
   knowledge-chat-sse-stream-runtime-boundary/
   mcp-memory-authority-onboarding-contract/
   mcp-claim-authority-tool-contract.
6. Validate real AI-harness adoption behavior against the stdio
   client contract. The known failure mode is an agent seeing Engram connected
   but choosing file-local memory because Engram looked empty or overlapped in
   responsibility. `claim_authority()` now returns the required
   `agent_protocol`, `validate_agent_protocol_calls()` scores compact client
   transcripts, and `tests/test_mcp_authority_client_adoption.py` proves a real
   stdio MCP client can follow the protocol end to end. The setup wizard and
   README now include the same adoption checklist, and `engram adoption` can
   generate a live-harness transcript template or validate recorded real-client
   transcripts. `claim_authority()` now returns the verifier command and
   transcript schema inside `agent_protocol`. The copied Claude transcript that
   triggered this slice is now a failing adoption regression: Engram was
   reachable, the runtime looked empty, and the agent admitted file memory
   stayed primary. The next adoption slice should run the same verifier against
   a current live Claude, Cursor, Windsurf, or similar harness transcript and
   use any failures to tighten harness-specific instructions. The verifier
   already handles common prefixed MCP tool names, nested log-record shapes,
   Claude Code stream-json tool-use logs, explicit plaintext/Markdown notes,
   stdin transcript input, placeholder metadata rejection, and copied chat
   admissions where the agent says it ignored Engram in favor of file-local
   memory. A local Claude Code 2.1.143 print-mode attempt was made with a strict
   disposable Engram stdio MCP config, but Claude returned `Not logged in -
   Please run /login`; rerun after `claude /login` or with another authenticated
   harness.
7. Keep the P3 evaluation loop focused on real evidence and persistence paths,
   not duplicate display work. Cue usefulness, projection yield, projection backlog,
   projection freshness/latency, recall gate latency, recall gate-control
   posture, false recall, memory-need recall / missed-recall rate, continuity,
   consolidation cycles, consolidation effect-rate, adjudication phase pressure,
   live open adjudication/evidence work pressure,
   calibration accuracy, and calibration ECE are now visible in
   REST/CLI/dashboard contracts. The Recall gate fields are also covered by the
   projected/consolidated smoke, native PyO3 smoke, smoke verifier, and no-bind
   native dashboard fixture. The shared report now also exposes
   `evaluation_signals` readiness for cue usefulness, projection yield, recall
   quality, false recall, triage calibration, and consolidation effect, and the
   projected/consolidated smoke verifier now requires each of those signals to be
   measured. Report coverage gaps now distinguish label-only recall evidence from
   actual runtime gate analysis. Evaluation reports now persist and reload the
   latest runtime gate metrics snapshot, so reopened native live reports do not
   lose the in-process smoke proof. The CLI hard gate now also supports
   `--min-evaluation-signal-evidence N`, which is the next operator knob to use
   when promoting real or benchmark-labeled evidence beyond smoke coverage.
   `--require-benchmark-evidence` can pair that report with a showcase
   `results.json` artifact, enforce scenario/pass-rate/fairness/hash gates, and
   render the attached evidence in Markdown for operator review. Use
   `--evidence-bundle` when the next slice needs a durable completion artifact.
   Benchmark and
   evaluation defaults now use explicit non-default groups for showcase,
   LoCoMo, memory-need fixtures, and the
   echo-chamber simulation, and explicit recall packet analysis uses the active
   REST/MCP/chat-tool brain group, so do not repeat those group-scoping slices
   either; a future evaluation slice should add a genuinely new measured
   signal, persistence path, or benchmark integration.
   Recall-need threshold resolution and memory-need analysis recording now share
   `retrieval.control` helpers, so do not recreate route/tool-local sync/async
   manager-facade adapters for those behaviors.
   REST/MCP artifact search now shares retrieval artifact-surface helpers, so
   do not rebuild artifact-hit serialization in REST/MCP routes.
   REST/MCP project bootstrap and runtime-state route calls now share
   `ingestion.project_bootstrap` and `retrieval.runtime_state` surface helpers,
   so keep those manager calls and REST skipped-status mapping out of public
   transport code.
   REST/MCP public entity/fact lookup now shares `retrieval.lookup` surface
   helpers, so keep REST camelCase search shaping, MCP raw lookup shaping, and
   MCP search validation out of public transport code.
   REST/MCP public agent-context response assembly now shares
   `retrieval.context_builder` surface helpers, so keep REST camelCase
   count/token shaping and MCP raw context shaping out of public transport code.
   REST/MCP adjudication resolution now shares `ingestion.adjudication_surface`
   helpers, so keep client-adjudication dispatch and REST/MCP outcome shaping out
   of public transport code.
   REST/MCP public Capture writes now share `ingestion.capture_surface` helpers,
   so keep conversation-date parsing, attachment construction, raw observe
   storage dispatch, Capture -> Project ingest dispatch, MCP write session
   activity, MCP live-turn recording, MCP adjudication loading, MCP write
   presentation, and MCP write recall middleware out of public transport code.
   REST entity detail/update/delete now shares `retrieval.entity_surface`
   helpers, so keep manager dispatch, sparse update payload construction, and
   not-found payload/status shaping out of the REST route.
   MCP graph-state tool and graph/entity resources now share
   `retrieval.graph_state` surface helpers, so keep graph tool dispatch, graph
   stats shaping, entity profile resource dispatch, and entity-neighbor resource
   dispatch out of `mcp/server.py`.
   REST dashboard stats, activation snapshot, and episode list reads now share
   `retrieval.graph_state` surface helpers too, so keep those manager dispatch
   calls out of public API routes.
   REST graph neighborhood, entity-neighbor, and temporal graph routes also share
   `retrieval.graph_state` surface helpers, so keep manager dispatch,
   missing-entity payloads, temporal timestamp parsing, and invalid-timestamp
   payloads out of public route handlers.
   REST atlas snapshot/history/region routes now share
   `retrieval.atlas_surface` helpers, so keep representation metadata,
   snapshot/history serialization, atlas service dispatch, and region/snapshot
   not-found payloads out of `server/engram/api/graph.py`.
   REST and MCP consolidation controls/read payloads now share route-facing
   helpers in `consolidation_trigger.py`, so keep REST trigger/status/history/
   detail payload shaping and MCP consolidation-status/trigger cycle-summary
   shaping plus MCP trigger-store fallback resolution out of public transport
   code. MCP still owns JSON wrapping and session-state store references.
   REST/MCP lifecycle summary now shares route-facing lifecycle helpers, so keep
   API runtime-context manager call wiring, audit-store reader construction,
   inactive-engine placeholder wiring, and limit clamping out of public
   transport handlers.
   Dashboard WebSocket activation monitor snapshots now share the graph-state
   activation snapshot helper, so keep activation snapshot manager dispatch out
   of the socket loop.
   MCP explicit recall near-miss/surprise enrichment now shares
   `retrieval.recall_surface`, and explicit MCP recall query timing, recall
   session flags, and recall middleware invocation now live there too. Keep
   those side-channel fields and recall-session side effects out of
   `mcp/server.py`; the tool wrapper still owns JSON wrapping.
   REST/MCP deterministic question routing now shares
   `retrieval.epistemic_route` helpers, so keep route history normalization and
   the manager route call out of public transport code.
   REST/MCP prospective-memory intention surfaces now share
   `retrieval.prospective` helpers, so keep intention create/list/dismiss
   manager calls, acknowledgement shapes, REST intention validation/not-found
   payload/status bodies, and MCP intention error payloads out of public
   transport code.
   Knowledge-chat conversation persistence now shares `retrieval.chat_persistence`
   helpers, so keep conversation validation/creation, group-scoped not-found
   handling and payload shaping, completed-turn persistence scheduling,
   completed-turn persistence, and recalled entity tagging out of the REST chat
   route.
   REST/MCP forget entity/fact surfaces now share `retrieval.forgetting`
   helpers, so keep target dispatch, fact-field normalization, missing-target
   payloads, and REST 400/404 response mapping out of public transport code.
   REST/MCP explicit preference feedback now shares `retrieval.preference_feedback`
   validation/dispatch, so keep rating validation, manager feedback calls, REST
   feedback error payloads, and MCP invalid-rating error payload shaping out of
   public transport code.
7. Use `docs/design/brain-runtime-completion-audit.md` before any completion
   claim and follow its current verdict near the top of the file. The
   GraphManager compatibility facade is now guarded more broadly, and the
   consolidation audit store reads are now behind `ConsolidationAuditReader`.
   Knowledge-chat rich tool-event
   shaping, chat tool execution payloads, chat tool-use loop/result
   accumulation, chat recall feedback/retry policy, chat response-turn orchestration,
   chat memory-need/live-context runtime, chat rate-limit execution, and chat persistence scheduling,
   REST/MCP explicit recall result/packet
   assembly, REST/MCP artifact search, REST/MCP project bootstrap/runtime-state
   route calls, REST/MCP public entity/fact lookup, REST/MCP public agent-context
   response assembly, REST/MCP deterministic question routing, REST/MCP
   prospective-memory intentions, and chat conversation persistence are also
   behind shared helpers now. REST/MCP forget target dispatch shares a retrieval
   helper, REST/MCP explicit preference feedback shares a retrieval helper, REST
   conversation CRUD has group-scoped persistence and response-envelope/status helpers,
   and SSE framing plus Anthropic client construction plus completed-turn
   persistence scheduling remain in the route as intentional transport/client
   concerns. REST/MCP episode adjudication
   request loading plus client-enabled surfacing gates now share an ingestion
   helper, and REST/MCP adjudication
   resolution now shares the same adjudication surface module. REST/MCP public
   Capture write dispatch, MCP write orchestration, and REST offline replay
   manager dispatch now share ingestion helpers. REST
   entity detail/mutation response/status
   assembly now shares a retrieval entity-surface helper. MCP graph-state and
   graph/entity resource response assembly now shares retrieval graph-state
   surface helpers, and REST dashboard stats/activation snapshot/episode-list
   reads plus graph neighborhood/entity-neighbor/temporal route response
   assembly now use the same graph-state surface module. REST and MCP consolidation
   control/read response assembly now share route-facing helpers, and MCP
   identity-core response assembly has a route-facing helper. REST/MCP live
   conversation manager-facade helpers are centralized in `retrieval.context`.
   REST/MCP lifecycle summary runtime-context/audit-store/limit wiring now
   shares lifecycle helpers. REST/MCP
   brain-loop evaluation report assembly now shares a service boundary too, and
   REST evaluation-report engine-context loading plus MCP evaluation report
   audit-store/cycle-snapshot loading now live in that report service. REST/MCP
   evaluation label writes now share label write surface helpers. MCP explicit
   recall near-miss/surprise enrichment now lives
   in the recall surface helper. MCP auto-recall cooldown, query extraction,
   per-tool gating,
   first-call session-prime planning, middleware side-effect planning, and
   middleware plan execution are also now in retrieval policy code, and
   lite/medium dispatch, full auto-recall dispatch, first-call context-prime
   dispatch, triggered-intention draining, middleware auto-observe storage,
   read-tool live-turn ingestion, lite/full auto-recall result compaction, plus
   additive response enrichment now live there too. MCP notification state
   lookup for piggyback `memory_notifications` is now in
   `notifications.surface`. The direct manager-dispatch scan
   across REST API routes and `server/engram/mcp/server.py` now returns no
   matches except MCP shutdown resource closing, and REST API routes are guarded
   against direct `engine.*` dispatch. `EpisodeWorker` runtime store access now uses explicit
   `EpisodeWorkerRuntimeStores` from REST/MCP startup, with a `GraphManager`
   compatibility accessor for direct construction, so do not repeat worker
   private-store extraction. Worker adjacent-turn batching, primary cue rebuild,
   and merged-away cue retirement now live in `ingestion.worker_batching`, so do
   not repeat that worker Cue-stage extraction. Worker deterministic scoring,
   multi-signal scorer access, goal boost lookup, and projection-yield feedback
   now live in `ingestion.worker_scoring`, so do not repeat that worker
   scoring extraction. Worker duplicate projection guards, system-discourse cue-only skips, and
   skip/defer projection-state sync now live in `ingestion.worker_routing`, so
   do not repeat worker routing extraction. Worker raw EventBus parsing and
   compact auto-capture content loading now live in `ingestion.worker_events`,
   so do not repeat worker event-shape extraction. MCP entity/fact search,
   artifact-search, context, and question-route tool recall-middleware
   invocation now live in retrieval-side tool-surface helpers too, so do not
   move those side effects back into `server/engram/mcp/server.py`. The next
   high-leverage slice is real live-harness evidence: either a second MCP
   client adoption transcript, or a staging/production `human-labels.json`
   artifact plus matching adoption report that passes `engram evaluate
   --require-release-evidence`. The CLI gate and template for human-labeled
   harness evidence now exist and are separate from deterministic benchmark
   proof. The local benchmark-labeled evaluation gate is strong for this
   milestone: the full deterministic showcase bundle passed 39/39
   `engram_full` scenario runs with pass rate `1.0`, false recall `0.0`, 13
   transcript hashes, a fairness contract, and all six evaluation signals
   measured.
   Packaging plan: treat the current dirty scope as one cohesive milestone
   commit unless a reviewer asks for a split. If split, use route/runtime
   boundaries, MCP adoption authority/verifier/template, and evaluation
   gates/docs. Stage only `README.md`, `docs/CURRENT_HANDOFF.md`,
   `docs/design/brain-runtime-audit.md`,
   `docs/design/brain-runtime-completion-audit.md`, the touched/new
   `server/engram/**` runtime/API/MCP/evaluation/setup/manifest files, and the
   touched/new `server/tests/**` files. Do not stage `/private/tmp` evidence
   artifacts. Current gates: `git diff --check`, focused adoption/evaluation
   tests, and `uv run pytest -m "not requires_docker and not requires_helix" -q`
   passing with 3320 tests, 43 skips, and 236 deselections.
8. Keep quest mode as a drilldown or alternate presentation, not the primary
   explanation of the brain loop.

Do not start by rewriting `GraphManager` wholesale. Keep extracting one shared
contract or service boundary at a time, with focused tests around each move.
