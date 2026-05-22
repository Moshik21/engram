# Memory Value and Latency Plan

Date: 2026-05-21

Status: implementation in progress

Owner surface: recall, AXI, evaluation, and dashboard runtime contracts

Current checkpoint:

- Phase 1 is implemented for runtime cost metrics: aggregate memory operation
  metrics persist in `SQLiteEvaluationStore`, the brain-loop report exposes
  `memory_value`, dashboard Evaluation renders cost/benefit, and AXI has
  `engram axi value`. Explicit REST/MCP `observe` and `remember` write
  surfaces now record agent-facing memory operation samples, including skipped
  auto-observe outcomes, and metrics recording is best-effort so telemetry
  cannot fail a capture write.
- Phase 2 is implemented as shared budget telemetry and gate outcome reporting:
  `RecallBudget` profiles exist, GraphManager attaches budget/degraded fields,
  auto-recall skips are recorded with skip reasons, explicit REST/MCP/AXI
  recall stages are timeboxed before packet assembly, explicit packet assembly
  is timeboxed, MCP/REST `get_context()` surfaces degrade instead of waiting
  indefinitely, MCP session-prime is capped by the startup budget, lite/medium
  MCP auto-recall probes are capped by the auto-lite budget, and knowledge-chat
  packet assembly degrades inside the chat budget. AXI startup remains a short
  metadata probe, while context/recall/value follow-up commands are bounded
  after real native dogfood runs showed the 5 second follow-up bound was too
  tight for loaded Helix stores. Context and recall stay at 10 seconds; the
  value/report follow-up uses a 20 second default because the full brain-loop
  report measured at roughly 12 seconds on the live native dogfood store.
  Live capture also keeps the raw episode and graph cue synchronous but treats
  cue vector indexing as best-effort under
  `capture_cue_vector_index_timeout_ms`, so observe/remember capture does not
  inherit slow embedding/vector latency from a loaded native store.
- Phase 3 has the packet-cache foundation: serialized packet payloads can be
  cached, hit/miss metrics are recorded, runtime/home surfaces show cache warmth,
  `get_context()` and `engram axi context` consume cached project/identity
  packets, graph mutation paths invalidate the cache, and runtime entrypoints
  attach a SQLite sidecar so native/local packet caches survive process restarts.
  Consolidation finalization invalidates cached packets for affected entities,
  relationships, and episodes after graph-mutating cycles; adjudication
  materialization/rejection invalidates stale packet views. Legacy lite SQLite
  databases are migrated before schema indexes are created, so default local
  AXI startup no longer fails on older entity facet columns.
- Phase 4 has dogfood replay through `engram dogfood replay` with redacted local
  transcript parsing, mode-comparison reports, an opt-in human-label template
  export, explicit `engram dogfood import-labels` ingestion into the local
  evaluation store, and `engram dogfood export-evidence` conversion into the
  standard `engram_human_label_evidence` artifact used by `engram evaluate`
  gates. A real redacted Codex transcript replay has passed; remaining Phase 4
  work is filling/importing reviewed labels from real dogfood transcripts.
  A real local AXI hook trace replay also passed and now reports measured
  startup/follow-up counts, operation/status counts, and duration stats.
  Default dogfood prepare/review outputs are hash-based and suppress
  prompt-derived query hints unless the operator explicitly opts into including
  transcript content for local review. Review/import/export/finalize gates now
  reject copied placeholder label text, so suggested command templates cannot
  accidentally satisfy the real human-label evidence gate. `dogfood review`
  withholds import/export closeout commands until the reviewed labels satisfy
  the configured sample thresholds.
- Phase 5 has packet-level trust summaries. Dashboard Knowledge now shows packet
  trust/provenance drilldowns, and Evaluation shows cache, budget, degraded,
  dismissed, corrected, stale-packet, and corrected-packet metrics. Auto-recall
  gate metadata is carried with full auto-recall results, including cached and
  budget-skipped outcomes. Explicit recall lifecycle/budget telemetry is
  retained in the dashboard store and shown in Knowledge even when no packets
  surfaced because the bounded recall path degraded. Packet trust now also
  carries per-memory confirmed/corrected/dismissed counts and last feedback
  timestamps when those interactions have happened in the current runtime.
  Remaining trust work is real dogfood label evidence.

Latest verification checkpoint:

- Backend focused plan suite: `405 passed, 13 skipped`.
- Dogfood replay/review focused suite: `47 passed`.
- Dashboard suite: `224 passed, 1 skipped`.
- Dashboard degraded-report focused tests: `59 passed`.
- Dashboard production build: passed with existing Vite chunk-size warnings.
- `git diff --check`: passed.
- Worktree native REST smoke on a temporary port proved the bounded recall path:
  `engram axi recall ... --server-url http://127.0.0.1:8121 --timeout 10`
  returned `status=degraded`, `skipReason=recall_timeout`, and
  `durationMs≈1200` instead of hanging.
- The live dogfood runtime has been reinstalled from this worktree with the
  local `helix-native` PyO3 package and restarted as LaunchAgent
  `dev.engram.local` on `127.0.0.1:8100`. `engramctl status` reports healthy
  `helix native (PyO3)` mode using
  `/Users/konnermoshier/.helix/engram-native-dogfood-axi`. Live AXI, REST, and
  MCP recall now all return bounded degraded responses with
  `skipReason=recall_timeout` and `durationMs≈1200` instead of hanging at the
  10 second client timeout.
- Native PyO3 health checks now use an in-process transport readiness check
  instead of a fake-group graph query, avoiding false `/health` degradation on
  loaded native stores where the graph query can exceed the 2 second public
  health timeout. After reinstall/restart, three consecutive live `/health`
  calls returned `status=healthy` and `services.graph_store=healthy`.
- After the same live runtime surfaced slow capture, cue vector indexing was
  timeboxed. Before the change, live REST observe measured roughly 8-9 seconds
  and one MCP observe took roughly 93 seconds. After reinstall/restart from this
  worktree, a single live REST observe measured roughly 0.43 seconds and MCP
  observe measured under 2 seconds while still returning
  `captureStatus=stored` and `projectionStatus=queued`.
- After reinstalling this worktree again with capture write metrics, the live
  native runtime recorded a REST observe memory operation sample in
  `/api/knowledge/runtime`: `source_counts.api_observe=1`, `result_count=1`,
  `status_counts.ok=1`, and `duration_ms.avg≈200`.
- REST auto-observe hook capture now defers synchronous decision
  materialization for `auto:*` sources so hook writes stay in the Capture/Cue
  lane and project later. Before the change, live auto-observe probes on the
  loaded native store measured roughly 2-3 seconds and cost samples briefly
  surfaced around 10 seconds. After reinstall/restart, a live
  `/api/knowledge/auto-observe` probe returned in roughly 0.77 seconds and
  `/api/knowledge/runtime` reported `source_counts.api_auto_observe=1` with
  `duration_ms.p95≈722`.
- After importing the reviewed dogfood labels, `engram axi value --server-url
  http://127.0.0.1:8100 --timeout 20 --json` reports `status=measured`,
  `operation_count=2`, and `p95_added_latency_ms≈693`.
- The live `/api/evaluation/brain-loop/report` path is now bounded around
  expensive graph-state reads. Before the change, the loaded native store could
  exceed a 30 second client timeout; after reinstall/restart, the endpoint
  returned in roughly 2.27 seconds with `degraded=true`,
  `skip_reason=graph_state_timeout`, and measured `memory_value.cost` preserved
  from runtime metrics. This keeps `engram axi value --timeout 20` usable while
  still surfacing degraded evidence.
- Dashboard Evaluation now maps and renders top-level degraded brain-loop
  reports. When REST falls back because a report read timed out, the Evaluation
  panel shows the degraded fallback count and skip reason instead of only
  showing operation-level degraded rates.
- `ENGRAM_EMBEDDING__PROVIDER=noop uv run engram evaluate --format json --memory-value`
  reports measured cost and measured benefit with 2 reviewed recall samples and
  1 reviewed session sample.
- Native dogfood value evidence passes:
  `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --mode helix --sqlite-path "$HOME/.engram/engram.db" --memory-value --require-memory-value --human-label-artifact "$EVIDENCE" --require-human-label-evidence --min-human-recall-samples 1 --min-human-session-samples 1 --format json`
  reports `memory_value.status=measured`.
- Native PyO3 Helix smoke passes:
  `ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --smoke --mode helix --replace --require-evaluation-signals --require-memory-value --format json`
  returned `coverage_gaps=[]` and `memory_value.status=measured` on the smoke
  fixture. Release evidence still reports missing human-label/adoption evidence,
  which is expected for the synthetic smoke.

## Purpose

This plan covers seven connected topics:

1. Performance instrumentation
2. Memory value report
3. Cached memory packets
4. Recall budgets
5. Recall gating
6. Dogfood against real transcripts
7. Trust visibility

The goal is to make Engram's benefit visible and bounded. Engram should improve
agent continuity without making every turn slower, noisier, or less trustworthy.
The target shape is not "always run deep recall." The target shape is:

```text
cheap startup packet -> deterministic need gate -> bounded recall path -> value/latency report -> visible trust signals
```

## Repo Findings

The repo already has most of the primitives needed for this work. The missing
piece is a unified value and latency loop across AXI, MCP, REST, evaluation, and
the dashboard.

### Recall Control

Existing files:

- `server/engram/retrieval/need.py`
- `server/engram/retrieval/signals.py`
- `server/engram/retrieval/control.py`
- `server/engram/retrieval/auto_recall.py`
- `server/engram/retrieval/feedback.py`
- `server/engram/retrieval/chat_feedback.py`

Current state:

- `analyze_memory_need()` already performs deterministic need analysis and
  records analyzer/probe latency on `MemoryNeed`.
- Pragmatic recall signals exist for unstated memory need such as bare names,
  possessive relational references, definite references, and continuation
  markers.
- `RecallNeedController` already keeps rolling runtime metrics:
  `total_analyses`, `trigger_count`, `surfaced_count`, `selected_count`,
  `used_count`, `dismissed_count`, `confirmed_count`, `corrected_count`,
  `false_recall_rate`, `surfaced_to_used_ratio`, `graph_lift_rate`,
  `probe_trigger_rate`, graph overrides, family contributions, and
  analyzer/probe latency summaries.
- MCP auto-recall is already split between lite/medium entity probe and the
  older full recall path.
- Recall feedback already distinguishes `surfaced`, `selected`, `used`,
  `dismissed`, `confirmed`, and `corrected`.
- Chat feedback can infer used vs dismissed recall targets from the final
  response text and apply post-response interaction updates.

Gap:

- The controller measures the gate and interactions, but not end-to-end recall
  wall time, packet-cache hit/miss, skipped-by-budget decisions, or explicit
  "value over baseline" outcomes.

### Packet and Planner Path

Existing files:

- `server/engram/models/recall.py`
- `server/engram/retrieval/plan.py`
- `server/engram/retrieval/pipeline.py`
- `server/engram/retrieval/packets.py`
- `server/engram/retrieval/recall_surface.py`
- `server/tests/test_recall_planner.py`
- `server/tests/test_recall_packets.py`

Current state:

- `MemoryPacket` exists and carries packet type, title, summary, why-now,
  confidence, belief map, entity ids, relationship ids, episode ids, evidence
  lines, provenance, and supporting intents.
- `assemble_memory_packets()` already builds fact, state, timeline, open-loop,
  intention, episode, and cue packets from raw recall results.
- `RecallPlan`, `RecallIntent`, and `RecallTrace` exist. `retrieve()` can run
  planner-driven multi-intent recall and attach planner support to scored
  results.
- REST and MCP explicit recall share `recall_surface.py`, but explicit recall is
  still a live retrieval operation every time.

Gap:

- Packets are assembled per recall call. There is no durable or semi-durable
  packet cache for fast startup, hot project state, identity core, recent
  decisions, or high-confidence long-tail facts.

### AXI Startup and Hooks

Existing files:

- `server/engram/axi/cli.py`
- `server/engram/axi/surfaces.py`
- `server/engram/axi/client.py`
- `server/engram/axi/hooks.py`
- `docs/axi-interface-plan.md`

Current state:

- `engram axi` provides a read-only home packet.
- Home probes health, runtime, and storage concurrently with a 2.5 second probe
  cap.
- `engram axi context`, `recall`, `storage`, `doctor`, `observe`, `remember`,
  `bootstrap`, and managed hooks exist.
- Hook traces write JSONL records with operation, status, duration, client,
  origin, project, budget, and timeout.
- Codex and Claude Code managed read-only session-start hooks exist.
- `engram axi doctor --hooks ... --require-hook-run --require-followup` can
  verify startup and follow-up evidence from traces.

Gap:

- AXI traces are currently local hook-run evidence, not first-class recall value
  evidence. They do not yet connect a startup/context/recall command to whether
  memory was useful, skipped, cached, or too slow.

### Evaluation and Dashboard

Existing files:

- `server/engram/evaluation/store.py`
- `server/engram/evaluation/report_service.py`
- `server/engram/evaluation/brain_loop_report.py`
- `server/engram/evaluation/cli.py`
- `server/engram/evaluation/adoption_evidence.py`
- `server/engram/api/evaluation.py`
- `dashboard/src/store/evaluationSlice.ts`
- `dashboard/src/api/client.ts`
- `dashboard/src/store/types.ts`
- `dashboard/src/components/EvaluationPanel.tsx`

Current state:

- The evaluation store persists recall labels, session continuity labels, and
  recall runtime metric snapshots.
- REST and MCP can write recall/session labels and load the shared brain-loop
  evaluation report.
- The report already includes recall precision, need recall, missed recall,
  useful packet rate, false recall, session continuity lift, open-loop recovery,
  temporal correctness, analyzer/probe p95 latency, and recall gate control
  counters.
- The dashboard already renders Evaluate cards for recall quality, recall gate,
  continuity, release evidence, adoption evidence, and human-label evidence.

Gap:

- The report does not yet answer: "How much time did Engram add?", "What was
  served from cache?", "How often did Engram skip itself correctly?", "Which
  recall mode was used?", and "What value did this produce compared with no
  memory or startup-only memory?"

### Benchmarks and Dogfood

Existing files:

- `server/engram/benchmark/showcase/*`
- `server/engram/benchmark/longmemeval/*`
- `server/engram/evaluation/benchmark_evidence.py`
- `server/engram/mcp/adoption_cli.py`
- `docs/dogfood-startup-validation-goal.md`
- `docs/CURRENT_HANDOFF.md`

Current state:

- Showcase benchmark scenarios compare Engram against constrained baselines
  such as recent context, markdown memory, context summary, external proxy
  baselines, and ablations.
- `engram evaluate` can attach benchmark artifacts and gate benchmark evidence.
- Adoption validation can parse live MCP client transcripts and distinguish
  successful use from blocked client/auth/MCP failures.
- The dogfood startup plan already covers native runtime readiness, hooks,
  AXI, and client startup validation.

Gap:

- There is no replay harness that takes real local transcripts and runs the same
  task under "Engram off", "AXI startup only", "gated lite recall", and "deep
  recall" modes with a shared latency/value schema.

## Product Contract

Engram should have four runtime tiers for agents:

1. `home`: read-only AXI startup packet. It must stay fast and degrade cleanly.
2. `cached_context`: prebuilt packets for identity, current project, recent
   decisions, and hot open loops. This should be the default low-latency memory
   path.
3. `gated_recall`: deterministic memory-need analysis decides whether to run
   recall and at what budget.
4. `deep_recall`: explicit user/agent search or high-confidence memory need.
   This can be slower, but must be bounded and visible.

The system should track both sides of the tradeoff:

- Benefit: useful packet rate, memory-need precision/recall, continuity lift,
  open-loop recovery, temporal correctness, avoided misses, confirmed recall.
- Cost: wall-clock latency, timeout/degraded counts, cache hit rate, recall
  trigger rate, false recall rate, dismissed rate, token budget used.

## Topic 1: Performance Instrumentation

### Current Grounding

Existing instrumentation:

- `MemoryNeed.analyzer_latency_ms`
- `MemoryNeed.probe_latency_ms`
- `RecallNeedController` latency summaries
- MCP explicit recall `query_time_ms`
- AXI hook trace `durationMs`
- Project/consolidation processing durations in evaluation reports

### Gaps

- No common event schema for memory operations.
- No wall-clock breakdown for recall stages: gate, cache, search, graph
  expansion, scoring, packet assembly, presenter serialization.
- No persistent record of budget decisions, timeouts, or degraded fallback.
- No dashboard/API view for end-to-end memory cost.

### Plan

Add a lightweight runtime telemetry boundary, preferably under:

```text
server/engram/retrieval/performance.py
server/engram/models/memory_value.py
```

The first implementation should be in-process and event-bus friendly. It does
not need a heavy observability stack.

Add a `MemoryOperationTrace` model:

```python
operation_id: str
group_id: str
surface: "mcp" | "rest" | "axi" | "chat" | "benchmark"
operation: "home" | "context" | "auto_recall" | "explicit_recall" | "chat_recall"
mode: "none" | "cached" | "lite" | "medium" | "deep"
started_at: float
duration_ms: float
budget_ms: float | None
budget_tokens: int | None
status: "completed" | "skipped" | "timeout" | "degraded" | "error"
skip_reason: str | None
cache_hit: bool | None
cache_key: str | None
result_count: int
packet_count: int
error: str | None
```

Instrument these entrypoints first:

- `build_home_payload()` in `server/engram/axi/surfaces.py`
- `build_context_payload()` and `build_recall_payload()` in AXI surfaces
- `build_mcp_explicit_recall_tool_surface()` in
  `server/engram/retrieval/recall_surface.py`
- `build_full_auto_recall_surface()` and `build_lite_auto_recall_surface()` in
  `server/engram/retrieval/auto_recall.py`
- `MemoryContextBuilder.get_context()` in
  `server/engram/retrieval/context_builder.py`
- knowledge chat recall path via `server/engram/retrieval/chat_runtime.py` and
  `chat_feedback.py`

Persist summaries through `SQLiteEvaluationStore`, not raw traces at first.
Add a retention-limited table:

```text
evaluation_memory_operation_metrics
```

Store aggregated windows by group, surface, operation, and mode:

- count
- completed/skipped/timeout/degraded/error counts
- avg/p50/p95 duration
- avg packet count
- avg result count
- cache hit rate
- budget miss rate

### Verification

- Unit tests for trace model serialization and latency aggregation.
- Focused tests around recall surface and AXI trace behavior.
- Evaluation report test proving memory-operation metrics survive process
  restart through the evaluation store.
- Dashboard API mapping test for new fields.

## Topic 2: Memory Value Report

### Current Grounding

The current evaluation report already has the right conceptual home:

- `evaluation_recall_samples`
- `evaluation_session_samples`
- `evaluation_recall_runtime_metrics`
- `build_brain_loop_report()`
- `EvaluationPanel`

Existing report metrics cover recall quality and continuity, but they do not
combine value with cost.

### Gaps

- No single "benefit vs latency" section.
- No mode comparison between no memory, startup-only, cached packets, lite
  recall, medium recall, and deep recall.
- No report field that answers whether Engram was net useful for an operator's
  dogfood period.

### Plan

Extend the brain-loop report with a new top-level section:

```json
"memory_value": {
  "status": "measured|needs_samples|needs_latency|needs_baseline",
  "window": {"sample_count": 0, "operation_count": 0},
  "cost": {
    "avg_added_latency_ms": 0,
    "p95_added_latency_ms": 0,
    "timeout_rate": 0,
    "cache_hit_rate": 0
  },
  "benefit": {
    "useful_packet_rate": 0,
    "memory_need_precision": 0,
    "memory_need_recall": 0,
    "session_continuity_lift": 0,
    "open_loop_recovery_rate": 0,
    "temporal_correctness": 0
  },
  "modes": {
    "cached": {...},
    "lite": {...},
    "medium": {...},
    "deep": {...}
  },
  "recommendation": {
    "default_mode": "cached+gated_lite",
    "reason": "..."
  }
}
```

Add helpers in `brain_loop_report.py`:

- `_memory_value_summary(...)`
- `_memory_cost_summary(...)`
- `_memory_mode_summary(...)`
- `memory_value_failure_message(...)` for future gates

Add CLI visibility:

```bash
engram evaluate --memory-value
engram evaluate --require-memory-value
```

The first command can be report-only. The second should be a future release gate
that requires both cost and label evidence.

Add AXI visibility:

```bash
engram axi value --budget 800 --timeout 20
```

This should return a compact summary for agents:

```text
operation: value
status: measured
default_mode: cached+gated_lite
cost:
  p95_added_latency_ms: 120
  cache_hit_rate: 0.82
benefit:
  useful_packet_rate: 0.71
  continuity_lift: 0.34
next[2]{cmd,reason}:
  engram axi context --project "$PWD",Use cached project context
  engram axi recall "query" --limit 5,Use explicit recall for long-tail searches
```

Implementation progress:

- `engram axi value` now calls the REST brain-loop evaluation report and returns
  a compact agent-facing memory value payload with latency, cache hit, budget
  miss, skipped count, benefit metrics, and next context/recall commands. It
  has a 20 second default timeout because the native-store report is a measured
  follow-up, not the metadata-only startup packet.
- `engram evaluate --memory-value` prints the operator memory-value section
  directly, and `--require-memory-value` exits non-zero unless both cost and
  benefit evidence are measured. Evidence bundles record that gate when used.
- `engram axi` home includes packet-cache warmth from runtime state so startup
  can show whether cached packet context is available without constructing it.

### Verification

- Extend `server/tests/test_evaluation_report_service.py` and
  `server/tests/test_evaluation_store.py`.
- Extend dashboard `apiClient.test.ts` and `store.test.ts`.
- Add `EvaluationPanel` rendering for cost/benefit summary.

## Topic 3: Cached Memory Packets

### Current Grounding

Existing packet assembly is good but live-only:

- `MemoryPacket` model exists.
- `assemble_memory_packets()` can build useful packet shapes.
- `MemoryContextBuilder` has a short-lived briefing cache.
- `EntityProbeRecallService` has a session-scoped entity cache.
- AXI home deliberately points to context rather than loading full context.

### Gaps

- No persistent packet cache.
- No invalidation model tied to observe/remember/project/consolidate changes.
- Startup hooks cannot cheaply show "what matters now" beyond the home packet
  and context pointer.

### Plan

Add a packet cache read model:

```text
server/engram/retrieval/packet_cache.py
```

Initial cache scopes:

- `identity_core`: durable user identity and preferences.
- `project_home:{project_path_hash}`: current project state, open loops, recent
  decisions, stale artifact warnings.
- `recent_decisions`: recent durable decisions across contexts.
- `hot_open_loops`: unresolved commitments or pending work.
- `session_prime:{topic_hash}`: short TTL session packet for startup/resume.

Cache entry fields:

```python
cache_key: str
group_id: str
scope: str
topic_hint: str | None
project_path: str | None
packets_json: str
source_entity_ids_json: str
source_episode_ids_json: str
source_relationship_ids_json: str
created_at: float
updated_at: float
expires_at: float | None
invalidated_at: float | None
build_duration_ms: float
hit_count: int
last_hit_at: float | None
```

Use existing packet assembly rather than creating a second packet schema. The
cache should store serialized `MemoryPacket.to_dict()` output.

Implementation progress:

- `server/engram/retrieval/packet_cache.py` now provides an in-process packet
  cache with TTL, max-entry eviction, hit counts, source id indexing, and
  invalidation by entity, episode, relationship, scope, or group.
- Explicit REST/MCP recall and full auto-recall packet assembly now check the
  packet cache before rebuilding serialized packet payloads and record
  `packet_cache` memory-operation samples for cache hits and misses.
- `GraphManager` owns packet-cache helpers and invalidates packet cache entries
  on observe/session-prime changes, projection, forget-entity, and forget-fact.
- `get_context()` now loads cached `identity_core` and `project_home` packets
  into the context tier, refreshes those scopes from live context entities, and
  reports packet-cache hit metadata. REST context forwards `cachedPackets` and
  `packetCache`, and `engram axi context` compacts cached packets with trust
  fields under the output budget.
- `MemoryPacketCache` now supports an optional SQLite sidecar. Runtime
  entrypoints derive a mode-appropriate path, using the native Helix data
  directory for PyO3 Helix and a SQLite sibling sidecar for lite/local modes.
  Storage diagnostics include the packet-cache sidecar so operators can see its
  location and size while Engram is running.
- Consolidation finalization now invalidates packet-cache entries touched by the
  cycle context, including merged/inferred/pruned/matured/schema entities,
  transitioned episodes, and demoted relationships. Evidence/adjudication
  materialization paths invalidate stale packet views after accepted or terminal
  adjudication outcomes.
- The blunt operator fallback is now wired through REST and AXI as
  `engram axi packet-cache clear`, which clears tenant-local cached packet
  entries without deleting graph memory, labels, or native Helix storage.

Invalidation:

- On `observe`, invalidate only short TTL session packets if no projection has
  happened.
- On `remember` or successful projection, invalidate scopes containing touched
  entity/episode ids.
- On consolidation merge/forget/adjudication, invalidate affected entity and
  relationship packet entries.
- Provide and document the blunt fallback:

```bash
engram axi packet-cache clear
```

First consumers:

- `get_context()` includes cached packet summaries before live context tiers
  when project/identity packets are present and fresh.
- `engram axi context` consumes the REST forwarded cached project/identity
  packets and shows compact trust metadata under the output budget.
- `build_full_auto_recall_surface()` attaches cached packets for known
  `MemoryNeed` types before running deep retrieval.

Do not make `engram axi` home run cache construction synchronously. Home can
report cache status and suggest `engram axi context`.

### Verification

- Packet cache unit tests for create, hit, expiry, invalidation by entity,
  invalidation by episode, and stale fallback.
- AXI context test proving cached packets are returned under budget.
- Recall surface test proving cached packets do not count as live recall unless
  surfaced.
- Evaluation report test proving cache hit rate appears in memory value metrics.

## Topic 4: Recall Budgets

### Current Grounding

Existing budget pieces:

- AXI has output token budgets and HTTP timeouts.
- `MemoryContextBuilder` scales entity limits and timeouts based on max tokens.
- `auto_recall_token_budget` and cache TTL exist for lite recall.
- Recall planner has intent count and subquery limits.
- Showcase benchmark scenarios have explicit retrieval/evidence/answer budgets.

### Gaps

- No single recall budget object passed across gate, planner, retrieval, packet
  assembly, and presenter.
- Time budgets are mostly local constants.
- Timeout behavior is not consistently reported as a first-class degraded state.

### Plan

Add a `RecallBudget` model:

```python
surface: str
mode: str
max_wall_ms: int
max_search_ms: int
max_graph_ms: int
max_packet_ms: int
max_results: int
max_packets: int
max_output_tokens: int
allow_deep_recall: bool
allow_embeddings: bool
allow_graph_probe: bool
allow_cache_only: bool
```

Default profiles:

- `startup`: cache-only, no deep recall, strict wall budget.
- `auto_lite`: analyzer plus lite/medium entity probe only.
- `auto_deep`: analyzer plus bounded full recall, only for high-confidence
  memory need.
- `explicit`: user/agent asked for recall, larger budget.
- `chat`: bounded by streaming response deadline.
- `benchmark`: deterministic profile from scenario budget.

Config additions:

```text
recall_budget_startup_ms
recall_budget_auto_lite_ms
recall_budget_auto_deep_ms
recall_budget_explicit_ms
recall_budget_chat_ms
recall_budget_cache_ttl_seconds
recall_budget_timeout_degrades
```

Implementation progress:

- `server/engram/retrieval/budgets.py` now defines `RecallBudget` plus shared
  startup, auto-lite, auto-deep, explicit, and chat profiles.
- Runtime memory operation samples now carry budget milliseconds, budget tokens,
  budget-miss flags, degraded flags, and status/skip-reason counts.
- `GraphManager.recall()`, `recall_lite()`, `recall_medium()`, and
  `get_context()` attach budget telemetry. Explicit REST/MCP recall search,
  explicit REST/MCP packet assembly, MCP session-prime context, MCP/REST
  `get_context()` surfaces, and lite/medium MCP auto-recall probes are now
  timeboxed by the shared budgets. Timeout paths return degraded/partial
  payloads with `context_timeout`, `recall_timeout`, or `packet_timeout`
  telemetry instead of blocking the whole agent response.
- Knowledge-chat recall packet assembly now uses the chat budget profile and
  degrades to raw chat recall results with `packet_timeout` telemetry when
  packet assembly exceeds the streaming-safe packet budget.

Apply budgets at these boundaries:

- `analyze_memory_need()` graph probe: skip graph probe if budget disallows it.
- `execute_recall_plan()`: cap sub-intents and per-intent budgets.
- `retrieve()`: wrap optional slow branches with budget checks.
- `assemble_memory_packets()`: cap packet count and packet assembly time.
- AXI/REST/MCP presenters: report budget used and degraded flags.

The first implementation can avoid invasive cancellation by checking a deadline
between stages and using `asyncio.wait_for()` around known expensive calls:

- project context recall in `MemoryContextBuilder`
- graph lookup/neighbor expansion in context builder
- MCP session-prime `get_context()` startup load
- lite/medium MCP auto-recall entity probes
- explicit recall in MCP/REST surfaces
- packet cache rebuilds

### Verification

- Tests where graph probe is skipped by budget.
- Tests where packet assembly times out and response still returns raw results.
- Tests where MCP session-prime, MCP/REST context, and lite/medium auto-recall
  timeout paths return degraded telemetry instead of hanging.
- AXI tests proving timeout/degraded output is compact and nonzero exit behavior
  remains unchanged where expected.
- Benchmark fairness test proving budget profile is recorded.

## Topic 5: Recall Gating

### Current Grounding

Recall gating is already partially implemented:

- `plan_mcp_recall_middleware()` decides if a tool gets auto-recall context.
- `should_recall_for_tool()` gates by tool and config.
- `analyze_memory_need()` decides `should_recall`.
- `RecallCooldown` rate-limits repeated topics.
- `RecallNeedController` can adapt thresholds when enabled.
- `chat_feedback.py` can retry generic memory-free chat responses when memory
  was needed.

### Gaps

- Lite auto-recall still runs through tool middleware after the middleware gate,
  but not always through a shared need/budget decision.
- The system does not explicitly record "skipped and correct" vs "skipped and
  missed."
- Gate decisions are not surfaced to AXI or the dashboard as trust-visible
  decisions.

### Plan

Make `MemoryNeed` the canonical gate result for auto recall, chat recall, and
AXI follow-up guidance.

Add gate outcome categories:

```text
triggered
skipped_ack
skipped_low_signal
skipped_recent_duplicate
skipped_budget
skipped_cache_satisfied
skipped_disabled
forced_explicit
forced_protocol
```

Implementation progress:

- Auto-recall gate skips now emit `auto_recall_gate` memory-operation samples
  for disabled recall, low-signal/ack turns, recent duplicates, recent explicit
  recall, empty results, and errors.
- Brain-loop and dashboard memory-value views now show skipped/error counts in
  addition to latency, cache hit, timeout, and budget-miss metrics.
- `MemoryNeed` now carries `mode_requested`, `mode_executed`, skip reason,
  budget profile, cache-hit/cache-satisfied, and budget-skipped outcomes.
  `RecallNeedController` aggregates those counts directly, and auto-recall
  enrichment includes compact gate metadata when recall fires or cache satisfies
  the request.
- The remaining gate work is to tune policy thresholds against real dogfood
  labels rather than only synthetic/unit evidence.

Extend `RecallNeedController.record_analysis()` to track skipped categories and
mode requested:

```text
mode_requested: none|cached|lite|medium|deep
mode_executed: none|cached|lite|medium|deep
skip_reason
budget_profile
cache_hit
```

Policy:

- Acknowledgements and contained operational commands should skip recall.
- Presupposed references, open loops, temporal updates, project-state questions,
  corrections, and explicit memory questions should trigger at least cached or
  lite recall.
- Graph probe can lift borderline cases only when budget allows it.
- Explicit `recall` remains forced, but still budgeted and measured.
- Empty/fresh Engram is not a reason to skip the pre-response protocol. The
  existing authority/runtime-state contract remains the adoption layer.

### Verification

- Extend `server/tests/test_recall_need.py` for skip reasons and mode choice.
- Extend `server/tests/test_auto_recall_policy.py` for cache-satisfied and
  budget-skipped decisions.
- Add a regression for acknowledgements not running deep recall.
- Add a regression where pragmatic presupposition triggers cached/lite recall.

## Topic 6: Dogfood Against Real Transcripts

### Current Grounding

Existing dogfood/evidence tools:

- AXI hook trace JSONL records command duration and origin.
- `engram adoption` validates live MCP client tool-call transcripts.
- `engram evaluate` attaches benchmark, adoption, and human-label evidence.
- Showcase benchmarks compare multiple constrained baselines.
- Current handoff records that multi-client adoption and human-labeled samples
  are release-hardening, not the core runtime blocker.

### Gaps

- No local replay command for real user transcripts across Engram modes.
- No repeatable A/B comparison that measures both answer quality and latency for
  the same transcript.
- No path to turn dogfood traces into memory-value samples without manual glue.

### Plan

Add a dogfood replay command under evaluation:

```bash
engram evaluate dogfood \
  --transcript path.jsonl \
  --project /path/to/project \
  --modes off,startup,cached,gated_lite,gated_medium,deep \
  --out /tmp/engram-dogfood-report.json
```

If adding a subcommand to `evaluate` is too large for the first pass, add:

```bash
engram dogfood replay ...
```

Input formats:

- AXI hook trace JSONL.
- Claude stream-json parsed by existing adoption CLI helpers.
- Manual wrapper Markdown used by adoption validation.
- Simple JSONL turns:

```json
{"role":"user","content":"...","timestamp":"..."}
{"role":"assistant","content":"...","timestamp":"..."}
```

Modes:

- `off`: no Engram context.
- `startup`: AXI home only.
- `cached`: cached memory packets only.
- `gated_lite`: need gate plus cached/lite recall.
- `gated_medium`: need gate plus medium recall.
- `deep`: explicit/full recall profile.

Outputs:

- latency per turn
- recall decision per turn
- packets surfaced
- packets used if detectable
- user-label placeholders
- suggested labels for manual review
- transcript hash and redaction status
- mode summary

Use the existing showcase benchmark scoring and evaluation sample models where
possible, but keep dogfood separate from synthetic benchmark evidence. Dogfood
must be allowed to contain real transcript metadata and must not be mislabeled
as benchmark evidence.

Implementation progress:

- `engram dogfood replay` now reads local JSONL, AXI trace JSONL, and simple
  Markdown role transcripts, plus real Codex `response_item` session JSONL. It
  hashes/redacts transcript content by default and compares `off`, `startup`,
  `cached`, `gated_lite`, `gated_medium`, and `deep` decisions with a shared
  latency/decision schema. Codex bootstrap payloads such as injected
  `AGENTS.md`, `<environment_context>`, and `<goal_context>` blocks are skipped
  so setup instructions do not become human-label review turns. Exact Codex
  smoke prompts such as `Reply exactly OK.` are also skipped because they prove
  hook/session plumbing, not memory value.
- `engram dogfood prepare` creates the local review bundle in one command:
  redacted replay report, optional merged AXI trace cost evidence from separate
  hook JSONL files, fillable labels, JSON review status, and Markdown review
  summary. The replay report remains redacted; contentful label templates still
  require explicit local opt-in. When a transcript contains only startup/setup
  or smoke content, prepare/review report `trace_only`: measured cost evidence
  is retained, but no import/export/finalize label command is suggested.
- `engram dogfood scan` finds redacted local transcript candidates with
  labelable turns, marks whether the recorded session `cwd` matches the target
  project, supports `--project-only`, and prints prepare commands. This lets
  the operator choose a meaningful dogfood transcript instead of repeatedly
  preparing startup-only files or unrelated project sessions.
- Replay reports now summarize redaction-safe AXI trace evidence when present:
  operation/status/origin/client counts, measured duration avg/p95/max, cache
  hits, session-start/follow-up split, and degraded/timeout counts. This keeps
  real hook latency evidence visible without copying private transcript text.
  Trace-only AXI hook files count as measured replay reports even when they have
  no user turns.
- Dogfood replay reports are explicitly separate from benchmark evidence and
  include label-template metadata only; they do not write recall/session labels
  unless a future explicit opt-in path is added.
- `--label-template` and `--label-template-out` can export redaction-safe review
  templates keyed by transcript and turn hashes without copying transcript
  content. `--label-template-include-content` is the explicit local-review
  opt-in for contentful templates; the default remains redacted.
- `engram dogfood review` summarizes label review readiness without importing:
  reviewed/importable turns, missing label queue, invalid mode references,
  recall/session sample counts, suggested `label-turn`/`label-session`
  commands, local `inspect-turn --include-content` commands for explicit human
  review of source transcript turns, and import/export next commands. Suggested
  turn-label commands omit placeholder `--notes`; reviewers add notes only when
  they have real observed rationale to record. Suggested session-label commands
  also omit placeholder notes. Review output now includes a compact queue
  summary by missing reason and need type plus one recommended next turn, so
  the operator does not have to scan a full command wall before starting
  human review. `dogfood review --need-type ... --command-limit ...` can focus
  suggested commands and the visible queue preview on high-value need
  categories while preserving the full queue counts and readiness gates.
  `dogfood review --include-content --context ...` can also produce a bounded
  local review packet that inlines transcript context for the suggested turns,
  but only after explicit opt-in; default review output remains redacted.
  `dogfood inspect-turn --next --need-type ...`
  opens the next queued turn directly and shows matching `label-turn` commands
  beside the local transcript context. Review and inspect output label the
  memory-needed and memory-not-needed commands as alternatives so the first
  command is not treated as a default. Transcript content is still redacted
  unless the operator passes `--include-content`.
- `engram dogfood label-turn` and `engram dogfood label-session` let the
  reviewer fill the prepared label artifact from the CLI without hand-editing
  nested JSON. These commands mutate only the local label artifact; they do not
  import or export evaluation evidence.
- `engram dogfood import-labels` imports only explicitly reviewed labels into
  `SQLiteEvaluationStore`; imports are idempotent for the same transcript,
  turn, mode, and session sample, keeping reruns from inflating value metrics.
  It keeps transcript bodies out of the imported samples and can dry-run the
  import before writing.
- `engram dogfood export-evidence` converts the same reviewed dogfood label
  artifact into the standard `engram_human_label_evidence` JSON shape, with
  required real source/client/capturedAt/labeler metadata. That artifact can be
  passed directly to `engram evaluate --human-label-artifact` so dogfood labels
  support both the memory-value report and release-style evidence gates without
  manual JSON reshaping. Export validates the would-be evidence before writing
  `human-labels.json`, so placeholder metadata exits non-zero without leaving an
  invalid artifact behind.
- `engram dogfood finalize` is the one-command hard gate for reviewed labels:
  review, idempotent import, optional measured AXI trace-cost import from the
  replay report, evidence export, closeout, and native memory-value evaluation.
  It fails before importing labels or writing evidence if labels are incomplete
  or source/client/capturedAt/labeler metadata is placeholder/invalid, and
  `--skip-evaluate` is explicit when the operator only wants import/export;
  skipped evaluation leaves the finalize report in `needs_evaluation` until the
  native memory-value command is measured.
- `engram dogfood closeout` now turns reviewed label artifacts into a native
  dogfood checklist: it verifies reviewed recall/session label counts, checks
  the exported human-label artifact when present, and stages command guidance
  so incomplete reviews do not advertise doomed import/export/native
  memory-value commands. Once reviewed labels meet the sample minimums it shows
  import/export, and once exported human-label evidence validates it also shows
  the native `engram evaluate --memory-value --require-memory-value` command
  needed to prove the closeout without treating replay output as labels.
  `--require-ready` makes the checklist enforceable, returning a non-zero exit
  until reviewed labels and exported human-label evidence satisfy the configured
  sample minimums. If the expected human-label artifact path is supplied before
  export creates it, closeout reports missing evidence rather than treating the
  absent file as a malformed artifact.
- A redacted real Codex transcript replay has been run successfully, with
  label-template export and dry-run label import confirming that unlabeled turns
  do not enter evaluation evidence accidentally.
- Dogfood closeout is complete for the current local bundle: 2 reviewed recall
  labels and 1 reviewed session label were imported, `human-labels.json` was
  exported from that reviewed artifact, `dogfood finalize` returned
  `status=finalized`, and native PyO3 `--require-memory-value` reports
  `memory_value.status=measured`.

### Current Dogfood Review Runbook

The current local dogfood bundle is:

```bash
LABELS="$HOME/.engram/dogfood-review/engram-019cc52d-with-axi/dogfood-labels.json"
REPLAY="$HOME/.engram/dogfood-review/engram-019cc52d-with-axi/dogfood-replay.json"
EVIDENCE="$HOME/.engram/dogfood-review/engram-019cc52d-with-axi/human-labels.json"
```

The bundle has 36 labelable turns, 2 reviewed recall labels, 1 reviewed session
label, 34 skipped turns, and an exported `human-labels.json`. The reviewed
labels intentionally record no measurable Engram lift for the reviewed local
continuity turns: immediate transcript context was enough, so baseline and
memory scores are both 1.0 for the session sample. To continue reviewing more
turns later, run:

```bash
uv run engram dogfood review --labels "$LABELS" --need-type open_loop --command-limit 2 --format markdown
uv run engram dogfood review --labels "$LABELS" --need-type open_loop --command-limit 2 --include-content --context 1 --format markdown
uv run engram dogfood inspect-turn --labels "$LABELS" --next --need-type open_loop --context 1 --format markdown
```

The review loop is:

```bash
cd server
uv run engram dogfood review --labels "$LABELS" --format markdown
uv run engram dogfood inspect-turn --labels "$LABELS" --turn 0 --context 1 --include-content
uv run engram dogfood label-turn --labels "$LABELS" --turn 0 --memory-needed yes --best-mode cached --helpful-mode cached --notes "real observed reason"
uv run engram dogfood label-session --labels "$LABELS" --scenario "real continuity scenario" --baseline-score 0.2 --memory-score 0.8 --notes "real session review"
uv run engram dogfood review --labels "$LABELS" --require-ready
```

Do not copy placeholder note text such as `<why memory helped>` into labels.
Review output avoids placeholder note arguments, and
review/import/export/finalize reject placeholder text by design if it is entered
manually. Once review is ready, close out with real metadata:

```bash
uv run engram dogfood import-labels --labels "$LABELS" --sqlite-path "$HOME/.engram/engram.db"
uv run engram dogfood export-evidence --labels "$LABELS" --out "$EVIDENCE" --source native_dogfood_harness --client Codex --captured-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --labeler operator
uv run engram dogfood closeout --labels "$LABELS" --human-label-artifact "$EVIDENCE" --sqlite-path "$HOME/.engram/engram.db" --mode helix --require-ready
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram dogfood finalize --labels "$LABELS" --replay-report "$REPLAY" --human-label-artifact "$EVIDENCE" --sqlite-path "$HOME/.engram/engram.db" --source native_dogfood_harness --client Codex --captured-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --labeler operator --mode helix
```

### Verification

- Parser tests for JSONL, adoption-style Markdown, and AXI trace files.
- Parser tests for real Codex `response_item` session JSONL.
- A tiny fixture transcript with expected per-mode decisions.
- Evaluation-store tests proving dogfood output can save recall/session samples
  only when the user explicitly opts in.
- CLI test proving private transcript contents are not printed by default.

## Topic 7: Trust Visibility

### Current Grounding

Existing trust signals:

- `MemoryPacket.provenance`
- packet `why_now`
- `belief_map`
- freshness labels in `EntityProbeRecallService`
- cue projection state and cue hit/selection/use counters
- evaluation dashboard cards
- storage diagnostics and runtime-state onboarding guidance
- release evidence and adoption evidence dashboard sections

### Gaps

- Agents and humans cannot easily see why recall fired, why it skipped, whether
  it was cached, or whether memory was stale.
- Packet provenance exists but is not consistently elevated in AXI or the
  dashboard.
- Corrections/dismissals are tracked, but trust surfaces do not yet tell the
  operator which memories are being penalized or need adjudication.

### Plan

Add a trust view to recall packets and reports.

For each packet returned through REST/MCP/AXI, expose:

```json
"trust": {
  "freshness": "fresh|recent|aging|stale|unknown",
  "source": "cache|live_recall|cue|episode|entity",
  "confidence": 0.0,
  "why_now": "...",
  "provenance_count": 0,
  "evidence_count": 0,
  "belief_status": "supported|uncertain|conflicting|unknown",
  "last_confirmed_at": null,
  "last_corrected_at": null
}
```

Do not remove the lower-level provenance arrays. Add trust as a compact summary
for agent/human scanning.

Dashboard:

- Extend `EvaluationPanel` with "Cost and Trust" metrics:
  - cache hit rate
  - timeout/degraded rate
  - stale packet rate
  - corrected/dismissed counts
  - skipped-by-gate count
- Add a packet drilldown in the Knowledge/Recall area showing:
  - why recall fired
  - mode used
  - budget used
  - packet provenance
  - feedback actions: used, dismissed, corrected, confirmed

AXI:

- Add compact trust fields to `engram axi context` and `engram axi recall`.
- Keep home packet compact. Home can show only high-level trust status:

```text
memory:
  value_status: measured
  trust_status: needs_labels
  cache: warm
  recall_p95_ms: 84
```

MCP:

- Include gate decision and trust summary in auto-recall enrichment when present:

```json
"recalled_context": {
  "source": "recall_lite",
  "gate": {"decision": "triggered", "needType": "project_state"},
  "packets": [...]
}
```

Implementation progress:

- `MemoryPacket` now carries a compact `trust` summary with freshness, source,
  confidence, why-now, provenance count, evidence count, belief status,
  confirmed/corrected/dismissed counts, and last feedback timestamps.
- Packet assembly fills trust summaries for entity, episode, cue, and timeline
  packets. MCP/AXI-style packet dictionaries keep snake_case trust fields, while
  REST explicit recall maps trust to camelCase.
- Dashboard Knowledge now surfaces recalled packet trust/provenance drilldowns,
  including packet-level confirmed/corrected/dismissed feedback. Evaluation
  shows degraded, dismissed, corrected, cache, skip, and budget metrics
  together.
- Brain-loop, REST, MCP, dashboard, and dogfood label-import paths now carry
  stale/corrected packet counts and rates from explicit evaluation labels.
- Auto-recall enrichment now carries gate metadata for executed, cached, skipped,
  and budget-skipped paths.
- Recall interaction feedback invalidates affected packet-cache entries so
  confirmed/corrected/dismissed trust metadata is not hidden behind stale cached
  packets.
- Remaining trust work: run/import real dogfood labels.

### Verification

- Packet serialization tests for trust summary.
- REST/MCP recall surface parity tests.
- AXI context/recall output tests under token budget.
- Dashboard type and component tests.

## Implementation Phases

### Phase 1: Instrument and Report

Outcome: Engram can show memory cost and runtime behavior without changing
recall behavior.

Tasks:

- Add memory operation trace model and aggregator.
- Persist aggregate metrics in `SQLiteEvaluationStore`.
- Extend `build_brain_loop_report()` with `memory_value.cost`.
- Add CLI/report rendering for memory value cost.
- Extend dashboard API mapping and Evaluation panel.

Acceptance:

- `engram evaluate --format json` includes memory operation cost metrics when
  runtime metrics exist.
- Existing tests for recall, evaluation, AXI, and dashboard continue to pass.
- No recall behavior changes yet.

### Phase 2: Shared Budgets and Gate Outcomes

Outcome: recall decisions are explicitly budgeted and skipped decisions are
measured.

Tasks:

- Add `RecallBudget`.
- Thread budgets through auto recall, context, explicit recall, and chat recall.
- Add gate outcome/skip reason fields to recall metrics.
- Add budget timeout/degraded reporting.
- Add focused regressions for skip reasons and budget-limited degradation.

Acceptance:

- Acknowledgement/low-signal turns do not run deep recall.
- Budget-degraded recall returns compact output rather than hanging.
- Evaluation report shows trigger/skipped/budget outcomes.

### Phase 3: Cached Packet Fast Path

Outcome: session startup and common context can use cached, provenance-bearing
packets.

Tasks:

- Add packet cache store and service.
- Build identity/project/recent/open-loop packet scopes.
- Wire cache reads into AXI context and `get_context()`.
- Add invalidation on projection, remember, consolidation, forget, and
  adjudication mutations.
- Track cache hit/miss/build duration in memory operation metrics.

Acceptance:

- `engram axi context --project "$PWD"` can serve cached project packets.
- Cache hit rate appears in memory value report.
- Cache invalidation prevents corrected/forgotten facts from being surfaced as
  fresh cached packets.
- Native/local runtime restarts can reuse fresh packet-cache entries from a
  local SQLite sidecar without running deep recall at startup.

### Phase 4: Real Transcript Dogfood Replay

Outcome: Engram can be compared against itself off/on using real transcript
fixtures and the same value schema.

Tasks:

- Add dogfood transcript parser and replay command.
- Support off/startup/cached/gated_lite/gated_medium/deep modes.
- Produce JSON and Markdown dogfood reports.
- Provide optional label-template export for human review.
- Keep private transcript bodies out of default summaries.

Acceptance:

- A real transcript can produce a mode comparison report with latency and recall
  decisions.
- The report can feed evaluation labels only with explicit operator opt-in.
- Synthetic benchmark evidence and real dogfood evidence remain clearly
  separated.

### Phase 5: Trust Visibility

Outcome: agents and humans can see why memory was used, skipped, cached, stale,
or corrected.

Tasks:

- Add packet trust summaries.
- Extend REST/MCP/AXI packet presentation.
- Extend dashboard Evaluation and Recall/Knowledge drilldowns.
- Add feedback actions where missing.

Acceptance:

- Every surfaced packet has why-now, freshness, source, confidence, and
  provenance summary.
- Brain-loop reports and dashboard Evaluation expose stale/corrected packet
  rates from explicit labels.
- Dashboard shows cost/trust status without requiring log inspection.
- AXI remains compact under default budgets.

## Suggested First Slice

Start with Phase 1 plus the minimal parts of Phase 2:

1. Add memory operation trace aggregation.
2. Record traces around AXI context/recall, MCP explicit recall, auto recall,
   and `get_context()`.
3. Persist aggregate operation metrics in the evaluation store.
4. Extend the brain-loop report with `memory_value.cost`.
5. Render the cost section in CLI Markdown and dashboard Evaluation.

This gives immediate dogfood feedback without changing recall behavior. Once
the cost is visible, tune gating and caching against real numbers instead of
guessing.

## Non-Goals For This Plan

- Replacing Helix native PyO3 as the preferred local full backend.
- Replacing MCP with AXI.
- Capturing transcripts automatically by default.
- Making every session-start hook run deep recall.
- Treating synthetic showcase benchmark evidence as real dogfood evidence.
- Building an external observability dependency before local metrics are useful.

## Decisions

1. Use a small SQLite sidecar for durable packet-cache entries. Native PyO3
   Helix stores the sidecar under the configured Helix data directory; lite and
   local modes use a SQLite sibling sidecar.
2. Keep `engram axi value` as the compact agent-facing value surface and
   `engram evaluate` as the fuller operator report.
3. Keep dogfood replay deterministic and redaction-safe by default; use
   human-label templates and the explicit label import path before automating
   label capture.
4. Keep AXI startup hooks light: metadata/status first, with follow-up commands
   for context, recall, and value instead of doing deep recall at startup.

## Completion Audit

Last audited: 2026-05-21.

The implementation scope is materially complete across the plan phases, but the
goal must stay open until real dogfood labels exist and the native value gate is
run against those labels. Synthetic smoke labels prove the machinery, not
dogfood benefit.

Current requirement status:

- Backend instrumentation and persisted value/cost metrics: implemented through
  `server/engram/retrieval/memory_operations.py`,
  `server/engram/evaluation/store.py`,
  `server/engram/evaluation/brain_loop_report.py`, and focused tests.
- Brain-loop, CLI, AXI, REST/MCP, and dashboard value reports: implemented
  through `server/engram/evaluation/report_service.py`,
  `server/engram/evaluation/cli.py`, `server/engram/axi/surfaces.py`,
  `server/engram/api/evaluation.py`, `server/engram/mcp/server.py`, and
  dashboard Evaluation mappings/components.
- Shared recall budgets and gate outcomes: implemented through
  `server/engram/retrieval/budgets.py`,
  `server/engram/retrieval/auto_recall.py`,
  `server/engram/retrieval/recall_surface.py`, and GraphManager recall/context
  telemetry.
- Cached packet fast paths and invalidation: implemented through
  `server/engram/retrieval/packet_cache.py`,
  `server/engram/retrieval/packet_cache_surface.py`, graph mutation
  invalidation hooks, consolidation finalization, adjudication finalization, and
  the AXI `packet-cache clear` fallback.
- Real transcript dogfood replay/comparison tooling: implemented through
  `server/engram/evaluation/dogfood.py`, including scan, prepare, replay,
  review, inspect-turn, label-turn, label-session, import-labels,
  export-evidence, closeout, and finalize.
- Trust/provenance/freshness/correction/cache/budget visibility: implemented
  through `MemoryPacket.trust`, REST/MCP/AXI packet presentation, dashboard
  Knowledge drilldowns, dashboard Evaluation cost/trust metrics, and evaluation
  label fields for stale/corrected packet evidence.
- Native PyO3 Helix primary path: preserved for live dogfood and smoke; AXI
  startup remains metadata-first and read-only by default.
- Synthetic evidence separation: preserved. Dogfood replay output does not
  populate benefit labels unless explicit reviewed labels are imported, and
  benchmark/smoke evidence does not satisfy the dogfood human-label gate.

Completion gate evidence:

- Current closeout for
  `$HOME/.engram/dogfood-review/engram-019cc52d-with-axi/dogfood-labels.json`
  reports `status=ready_for_native_memory_value`, 2 reviewed recall samples,
  1 reviewed session sample, and measured human-label evidence from
  `$HOME/.engram/dogfood-review/engram-019cc52d-with-axi/human-labels.json`.
- `human-labels.json` is marked `humanLabeled=true`, `source=native_dogfood_harness`,
  `client=Codex`, `labeler=operator`, and contains 2 recall samples plus
  1 session sample.
- `engram dogfood finalize` completed with `status=finalized`, imported the
  reviewed labels, exported evidence, checked closeout, and ran native
  memory-value evaluation.
- Native PyO3
  `engram evaluate --memory-value --require-memory-value --require-human-label-evidence`
  passes against the reviewed dogfood evidence with
  `memory_value.status=measured`.
- AXI `engram axi value --server-url http://127.0.0.1:8100 --timeout 20 --json`
  reports `status=measured`.

## Test Matrix

Backend focused tests:

```bash
cd server
uv run ruff check \
  engram/evaluation/dogfood.py \
  engram/evaluation/report_service.py \
  engram/ingestion/capture_service.py \
  engram/retrieval/auto_recall.py \
  engram/retrieval/context_builder.py \
  engram/retrieval/recall_surface.py \
  tests/test_auto_recall_policy.py \
  tests/test_capture_service.py \
  tests/test_context_surface.py \
  tests/test_dogfood_replay.py \
  tests/test_evaluation_report_service.py
uv run pytest tests/test_recall_need.py \
  tests/test_auto_recall_policy.py \
  tests/test_autorecall.py \
  tests/test_context_surface.py \
  tests/test_recall_feedback.py \
  tests/test_recall_surface.py \
  tests/test_recall_packets.py \
  tests/test_recall_planner.py \
  tests/test_piggyback_context.py \
  tests/test_mcp_tools.py \
  tests/test_evaluation_store.py \
  tests/test_evaluation_report_service.py \
  tests/axi/test_axi_cli.py \
  tests/axi/test_axi_client.py \
  tests/axi/test_axi_surfaces.py \
  tests/test_memory_operation_metrics.py \
  tests/test_packet_cache.py \
  tests/test_recall_budgets.py \
  tests/test_dogfood_replay.py \
  tests/test_evaluation_human_label_evidence.py \
  tests/test_capture_service.py \
  tests/test_capture_surface.py \
  tests/test_auto_observe.py \
  tests/test_helix_client.py \
  tests/test_health_surface.py -q
```

Dashboard focused tests:

```bash
cd dashboard
pnpm exec vitest run
pnpm run build
```

Operator smoke:

```bash
cd server
uv run engram axi --project "$PWD" --budget 800 --timeout 3 --json
uv run engram axi context --project "$PWD" --budget 800 --timeout 10 --json
uv run engram axi value --budget 800 --timeout 20 --json
ENGRAM_EMBEDDING__PROVIDER=noop uv run engram evaluate --format json
```

Native dogfood smoke after behavior changes:

```bash
ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --smoke --mode helix --replace --require-evaluation-signals --require-memory-value --format json
```
