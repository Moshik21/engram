# Brain Runtime Completion Audit

Date: 2026-05-15

This is a readiness audit for the active long-running goal, not a completion
claim. The current verdict is **not complete**. Engram has strong implementation
and verification coverage for the brain-loop contract, especially on the
preferred Helix native PyO3 path, but several requirements still need final
evidence before the goal can be closed.

## Objective Restated

Engram should be a coherent, production-grade, one-brain-per-person memory
runtime for AI agents. The runtime contract is:

```text
Capture -> Cue -> Project -> Recall -> Consolidate
```

The codebase should expose that loop consistently across runtime services,
REST, MCP, CLI/operator tooling, tests, and dashboard UI. PyO3 native Helix is
the preferred full-backend local path; SQLite/lite remains the smoke/demo path.

## Prompt-To-Artifact Checklist

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Audit current architecture and drift | `docs/design/brain-runtime-audit.md`, `docs/CURRENT_HANDOFF.md` | Strong, ongoing |
| Preserve useful dirty-worktree changes | Worktree remains dirty; new work is scoped to added manifest/tests/docs | Needs final packaging discipline |
| Make `Capture -> Cue -> Project -> Recall -> Consolidate` explicit | `server/engram/lifecycle_summary.py`, `dashboard/src/components/LifecyclePanel.tsx`, `server/engram/evaluation/brain_loop_report.py` | Strong |
| Extract capture/observe/store runtime boundaries | `server/engram/ingestion/capture_service.py`, `episode_ingestion.py`, `offline_replay.py`, `dedup.py` | Strong |
| Extract project runtime boundaries | `server/engram/ingestion/projection_service.py`, `projection_execution.py`, `projection_state.py` | Strong |
| Extract recall runtime boundaries | `server/engram/retrieval/service.py`, `presenter.py`, `context_builder.py`, `entity_probe.py`, `graph_state.py` | Strong |
| Extract consolidation orchestration boundaries | `server/engram/consolidation/lifecycle.py`, `phase_runner.py`, `events.py`, `completion.py`, `phase_catalog.py` | Strong |
| Keep `GraphManager` as compatibility facade, not hidden runtime brain | `server/tests/test_graph_manager_facade_boundaries.py` guards 61 core and compatibility delegates across lifecycle, evidence, artifacts, lookup, forgetting, intentions, context, graph state, and recall interactions; consolidation audit reads now use `ConsolidationAuditReader`, and MCP auto-recall lite/full shaping plus enrichment now use retrieval helpers instead of route-local response construction | Strong for `GraphManager`; route-local orchestration audit remains ongoing |
| Align REST and MCP remember/observe/recall semantics | Shared presenters in ingestion/retrieval plus REST/MCP tests | Strong |
| Align backend/dashboard lifecycle contracts | `dashboard/src/components/LifecyclePanel.tsx`, `dashboard/src/constants/consolidation.ts`, backend phase registry tests | Strong |
| Preserve one-brain-per-person `group_id` semantics | `server/tests/test_group_scope_static_contract.py`, native parity tests, active `native_brain` coverage, default-group config inheritance tests | Strong |
| Keep SQLite/lite viable | Broad gate: `2933 passed, 43 skipped, 236 deselected` for `pytest -m "not requires_docker and not requires_helix"` | Strong |
| Make PyO3 native Helix the preferred full path | README/install docs, native smoke, native parity suite, `engram.quality.native_surface_manifest` | Strong |
| Keep Helix/full-mode external tests isolated | `requires_helix`/`requires_docker` deselection and native no-Docker parity | Strong for local gates; Docker/full still separate |
| Build evaluation loop | `server/engram/evaluation/brain_loop_report.py`, REST/MCP label/report surfaces, dashboard Evaluate panel, smoke verifier | Strong, needs more real labeled evidence before production claim |
| Update docs/handoff as decisions become real | `docs/CURRENT_HANDOFF.md`, `docs/design/brain-runtime-audit.md` | Strong, ongoing |
| Do not mark complete until implementation, tests, docs, UI are understandable | This audit says not complete | Blocking |

## Current Verification Evidence

- Backend non-Docker/non-external-Helix gate:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  currently passes with 2933 tests, 43 skips, and 236 deselections.
- GraphManager facade evidence:
  `tests/test_graph_manager_facade_boundaries.py` now has 61 static delegate
  checks for core lifecycle APIs and service-backed compatibility adapters.
- MCP route-orchestration evidence:
  `mark_identity_core` now calls `GraphManager.mark_identity_core()` instead of
  writing through `manager._graph`, with service and public-surface coverage.
  `trigger_consolidation` now calls
  `GraphManager.trigger_consolidation_cycle()` instead of building
  `ConsolidationEngine` from manager private state inside the MCP route.
  MCP entity profile/neighbors resources now call graph-state manager facades
  instead of formatting directly from `manager._graph` and `_activation`.
  REST/MCP graph-probe helpers now call a manager facade instead of building
  `GraphProbe` from private graph and activation stores.
  REST/MCP intention-list handlers now call a prospective-memory presentation
  facade instead of computing warmth from manager private config/activation.
  MCP `intend` now reads the effective default threshold through the same
  prospective-memory facade instead of private config.
  REST/MCP conversation context helpers now call conversation-runtime manager
  facades for active context, embedding, session entity names, recent turns,
  turn counts, and live-turn ingestion instead of reaching into
  `manager._conv_context` or `manager._search`.
  REST remember/recall/chat policy decisions now call
  `PublicSurfacePolicyService`-backed manager facades instead of reading
  `manager._cfg`, and MCP lifecycle summary now uses manager facades for
  lifecycle graph-store and activation-config access.
  REST entity detail/PATCH/DELETE now call graph-state and entity-mutation
  manager facades instead of reading private graph, activation, or config
  fields in `server/engram/api/entities.py`.
  REST admin benchmark loading now calls `GraphManager.load_benchmark_corpus()`
  backed by `BenchmarkLoadService` instead of reading private graph,
  activation, or search stores in `server/engram/api/admin.py`.
  REST graph neighborhood and temporal graph routes now call
  `GraphManager.get_graph_neighborhood()` and `get_temporal_graph()` backed by
  `GraphStateService` instead of reading private graph, activation, or config
  fields in `server/engram/api/graph.py`; route-facing graph-state helpers now
  also own REST missing-entity and invalid-timestamp payloads for those routes.
  MCP recall-response enrichment now calls manager facades backed by
  `RecallResponseStateService` instead of reading private activation,
  near-miss, surprise-cache, or triggered-intention fields in
  `server/engram/mcp/server.py`. MCP auto-recall response payload attachment
  also delegates to `apply_mcp_recall_enrichment()` so the additive
  `session_context`, `recalled_context`, `triggered_intentions`, and
  `memory_notifications` keys share one retrieval-side contract.
  REST dashboard stats now calls `GraphManager.get_dashboard_stats()` backed by
  `GraphStateService` instead of reading the graph store directly in
  `server/engram/api/stats.py`.
  REST activation monitor reads now call
  `GraphManager.get_activation_snapshot()` and `get_activation_curve()` backed
  by `GraphStateService` instead of reading app state, graph store, activation
  store, config, or `compute_activation` directly in
  `server/engram/api/activation.py`.
  REST episode dashboard reads now call
  `GraphManager.list_episode_summaries()` backed by `GraphStateService` instead
  of reading the graph store and formatting paginated episode/cue payloads in
  `server/engram/api/episodes.py`.
  REST/MCP lifecycle summary reads now call
  `GraphManager.get_lifecycle_summary()` backed by `LifecycleSummaryService`
  instead of calling `build_lifecycle_summary` and passing graph/config facades
  from `server/engram/api/lifecycle.py` or `server/engram/mcp/server.py`.
  Dashboard WebSocket activation-monitor snapshots now call
  `GraphManager.get_activation_snapshot()` instead of reading app-state
  graph/activation/config stores and computing activation inside
  `server/engram/api/websocket.py`.
  REST notification reads/dismissal and MCP `memory_notifications`
  piggybacking now call `NotificationSurfaceService` instead of reading and
  formatting the notification store directly in `server/engram/api/knowledge.py`
  or `server/engram/mcp/server.py`; REST list/dismiss response envelopes and
  missing-service fallbacks now use route-facing notification surface helpers.
- Native/default-group config evidence:
  `tests/test_config.py` verifies `ENGRAM_DEFAULT_GROUP_ID` drives
  `auth.default_group_id` when the auth override is omitted, while explicit
  auth overrides still win.
- Native public-surface accounting:
  `server/engram/quality/native_surface_manifest.py` classifies REST routes,
  WebSocket/MCP transports, MCP tools/resources/prompts, dashboard native smoke,
  and operator native smoke evidence.
- Native populated REST/MCP parity:
  `tests/test_native_surface_parity.py::test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces`
  passes with the native surface manifest tests.
- Lifecycle-first UI evidence:
  `dashboard/src/components/LifecyclePanel.tsx` maps Capture, Cue, Project,
  Recall, and Consolidate to stage cards and drilldowns.
- Evaluation-loop evidence:
  `server/engram/evaluation/brain_loop_report.py` and
  `dashboard/src/components/EvaluationPanel.tsx` surface cue usefulness,
  projection yield/backlog/freshness, recall gate quality, continuity,
  consolidation effect, adjudication pressure, and calibration quality.

## Blocking Gaps Before Goal Completion

1. Final facade audit:
   `GraphManager` is much thinner, and core lifecycle plus service-backed
   compatibility facades now have 61 static delegate checks. REST/MCP memory
   presenters and consolidation presenters are also guarded. The first
   route-local cleanup moved MCP identity-core writes behind a service and
   manager facade; the second moved MCP consolidation trigger engine
   construction behind a service and manager facade; the third moved MCP entity
   graph resources behind graph-state service facades; the fourth moved
   REST/MCP graph-probe construction behind a manager facade; the fifth moved
   REST/MCP intention-list presentation behind the prospective-memory service.
   MCP `intend` default-threshold reporting now uses the same service boundary.
   The latest slice moved REST/MCP live conversation context access and
   turn-ingestion helpers behind `ConversationRuntimeService` and manager
   facades. The follow-up moved REST public-policy config reads behind
   `PublicSurfacePolicyService` and moved MCP lifecycle graph/config reads
   behind manager facades. The latest slice moved REST entity detail and
   mutation routes behind graph-state/entity-mutation service facades.
   The follow-up moved REST admin benchmark loading behind a benchmark-loader
   service facade. The latest slice moved REST graph neighborhood and temporal
   graph reads behind `GraphStateService` facades. The follow-up moved MCP
   recall-response transient state behind `RecallResponseStateService`
   facades, REST dashboard stats moved behind `GraphStateService`, and REST
   activation monitor reads plus episode dashboard reads now use the same
   graph-state boundary. REST/MCP lifecycle summary reads now use
   `LifecycleSummaryService` through a manager facade, and WebSocket activation
   monitor snapshots use the existing activation snapshot facade. REST/MCP
   notification surfacing now uses `NotificationSurfaceService`, WebSocket
   notification dismissal uses that same service boundary, and WebSocket auth
   config lookup uses `get_config()`. REST knowledge-chat rate limiting now uses
   `get_rate_limiter()`, and REST health now uses `get_graph_store()`,
   `get_config()`, and `get_mode()`. The REST/MCP
   route-private manager read scan is now clean, and the activation/episode/
   lifecycle/WebSocket activation-monitor/notification scans are clean for
   route-local app-state, store, config, activation-compute, episode/cue
   formatting, lifecycle-summary builder, socket-local activation math, and
   notification formatting/store reads; the WebSocket, knowledge, and health
   routes no longer import `_app_state` directly. The public-surface guard now
   generates coverage for every API route module except `api/deps.py`, and the
   route-local `_app_state` scan is clean outside that dependency layer.
   The REST evaluation report also reads consolidation cycles/calibration
   snapshots through `ConsolidationEngine.get_recent_evaluation_context()`
   instead of `engine._store`; focused evaluation/consolidation/static checks
   passed, with a later route-orchestration broad gate now passing 2933 tests.
   The follow-up moved REST consolidation status/history/detail reads through
   public `ConsolidationEngine` reader facades backed by
   `ConsolidationAuditReader`, moved detail payload assembly into
   `serialize_cycle_detail()`, and moved MCP evaluation report inputs,
   consolidation status, and lifecycle summary cycle reads through the same
   reader. Focused lifecycle/consolidation/API/MCP/static checks passed, and
   route-local `engine._store` plus synthetic lifecycle `consolidation_engine._store`
   usage has been removed.
   REST consolidation trigger/status/history/detail response assembly now also
   uses route-facing helpers in `server/engram/consolidation_trigger.py`,
   covering trigger conflict/acknowledgement payloads, background manual-cycle
   execution, pressure/latest-cycle status shaping, history payloads, and
   cycle-detail 404/detail payloads. Focused consolidation-trigger, REST API,
   public-surface, consolidation-presenter, and Ruff checks passed.
   The knowledge-chat rich event selection is now in
   `server/engram/retrieval/chat_events.py`, with the REST route retaining only
   AI SDK SSE framing. Focused chat-event, recall-presenter, public-surface, and
   full knowledge API checks passed.
   Knowledge-chat tool execution payloads are now in
   `server/engram/retrieval/chat_tools.py`, covering recall/search_entities/
   search_facts LLM payloads, chat recall packet shaping, fact deduplication,
   and unknown-tool responses while the REST route keeps the Anthropic tool-use
   loop and JSON-string compatibility wrapper. Focused chat-tool,
   chat-recall-helper, chat-event, public-surface, and Ruff checks passed.
   Knowledge-chat recall feedback and retry policy are now in
   `server/engram/retrieval/chat_feedback.py`, covering used/dismissed memory
   interaction application, generic memory-free response detection, retry
   gating, and retry system-prompt construction while the REST route keeps the
   actual Anthropic retry call and stream framing. Focused chat-feedback,
   chat-tool, full knowledge API, chat-event, public-surface, Ruff, and
   `git diff --check` gates passed.
   Knowledge-chat memory-need and live-context runtime helpers are now in
   `server/engram/retrieval/chat_runtime.py`, covering chat memory-need
   analysis, memory-guidance text, live conversation hydration, assistant-turn
   recording, and recent-turn extraction while the REST route keeps rate
   limiting, conversation resolution, context fetch, Anthropic tool-loop
   streaming, and final SSE framing. Focused chat-runtime/feedback/tool, full
   knowledge API, chat-event, public-surface, Ruff, and `git diff --check`
   gates passed.
   REST/MCP explicit recall result and packet assembly are now in
   `server/engram/retrieval/recall_surface.py`, covering the explicit
   Recall-stage manager call, recall packet analysis, memory packet assembly,
   API/MCP recall item presentation, and MCP near-miss/surprise side-channel
   enrichment while REST and MCP keep their transport-specific metadata and
   response shapes. Focused knowledge API, MCP JSON-response, autorecall, chat,
   public-surface, Ruff, and `git diff --check` gates passed.
   Recall-need threshold resolution and memory-need analysis recording now share
   `server/engram/retrieval/control.py` helpers, covering sync/async manager
   facade compatibility for REST, MCP, chat runtime, chat tool execution, and
   explicit recall surfaces. Focused recall-control, knowledge API, MCP
   JSON-response, autorecall, chat, public-surface, Ruff, and `git diff --check`
   gates passed.
   REST/MCP artifact search result assembly now shares
   `server/engram/retrieval/artifacts.py` helpers, covering artifact hit loading
   and item serialization while REST keeps `projectPath` and MCP keeps
   `project_path` plus recall middleware enrichment. Focused artifact-search,
   artifact service, REST artifact endpoint, MCP artifact, public-surface, Ruff,
   and `git diff --check` gates passed.
   REST/MCP deterministic question routing now shares
   `server/engram/retrieval/epistemic_route.py` helpers, covering route history
   normalization and the manager `route_question` call while REST keeps HTTP
   response wrapping and MCP keeps recall middleware enrichment. Focused
   route-surface, REST epistemic endpoint, MCP JSON-response, public-surface,
   Ruff, and `git diff --check` gates passed.
   REST/MCP prospective-memory intention surfaces now share
   `server/engram/retrieval/prospective.py` helpers, covering intention create,
   list, and dismiss manager calls plus API/MCP acknowledgement shapes while
   REST keeps HTTP status mapping and MCP keeps JSON error wrappers. REST
   intention validation and not-found payload bodies now live in the same helper
   module. Focused prospective-surface, public-surface, full knowledge API, full
   MCP tool, Ruff, and `git diff --check` gates passed.
   REST/MCP forget entity/fact surfaces now share
   `server/engram/retrieval/forgetting.py` helpers, covering target dispatch and
   fact-field normalization while REST keeps entity-first behavior for dual
   targets and MCP keeps exactly-one-target validation. Focused forget-surface,
   REST forget, MCP forget, public-surface, Ruff, and `git diff --check` gates
   passed.
   REST/MCP explicit preference feedback now shares
   `server/engram/retrieval/preference_feedback.py` helpers, covering public
   rating validation, the `record_explicit_feedback` manager call, and MCP
   invalid-rating error payloads while REST keeps 400/404 HTTP mapping. Focused
   feedback-surface, feedback recorder, full knowledge API, full MCP tool,
   public-surface, Ruff, and `git diff --check` gates passed.
   REST/MCP project bootstrap and runtime-state calls now share route-facing
   helpers. `server/engram/ingestion/project_bootstrap.py` owns the public
   bootstrap manager call and REST skipped-status mapping while
   `ProjectBootstrapService` remains the deeper artifact/cue/graph writer.
   `server/engram/retrieval/runtime_state.py` owns the public runtime-state
   manager call while `RuntimeStateService` remains the runtime/config/artifact
   freshness read model. Focused project-runtime surface, REST bootstrap/runtime,
   MCP runtime, public-surface, and Ruff checks passed.
   REST/MCP public entity/fact lookup now shares route-facing lookup helpers.
   `server/engram/retrieval/lookup.py` still owns the deeper
   `EntityFactLookupService`, and now also owns REST entity/fact search payload
   shaping plus MCP entity/fact search payload shaping and missing-query
   validation. REST keeps camelCase `items`; MCP keeps raw lookup results and
   recall middleware enrichment. Focused lookup-surface, REST facts, MCP
   entity/fact search, MCP middleware, public-surface, and Ruff checks passed.
   REST/MCP public agent-context assembly now shares route-facing context
   helpers. `server/engram/retrieval/context_builder.py` still owns the deeper
   `MemoryContextBuilder`, and now also owns REST context payload shaping and MCP
   raw context manager access. REST keeps camelCase count/token fields; MCP keeps
   the raw `get_context` shape and its recall/notification middleware. Focused
   context-surface, tiered context, REST context/runtime, MCP context middleware,
   public-surface, and Ruff checks passed.
   REST/MCP adjudication resolution now shares ingestion-side surface helpers.
   `server/engram/ingestion/adjudication_surface.py` owns the public
   client-adjudication manager dispatch and API/MCP outcome shaping for resolved
   edge-adjudication work items. REST keeps camelCase IDs; MCP keeps snake_case
   IDs. Focused adjudication-surface, REST adjudicate, MCP adjudicate,
   public-surface, and Ruff checks passed.
   REST/MCP public Capture writes now share route-facing capture helpers.
   `server/engram/ingestion/capture_surface.py` owns public conversation-date
   parsing, attachment construction, raw observation storage dispatch, and
   Capture -> Project ingest dispatch. REST and MCP still keep
   transport-specific session accounting, live-turn ingestion, recall middleware,
   skip handling, and memory-write presenters. Focused capture-surface,
   memory-write presenter, REST remember/adjudication, MCP remember/adjudication,
   public-surface, and Ruff checks passed.
   REST entity detail/update/delete now has a route-facing public-surface helper.
   `server/engram/retrieval/entity_surface.py` owns entity detail manager
   dispatch, sparse update payload construction, delete dispatch, and the shared
   REST not-found payload, while `GraphStateService` and `EntityMutationService`
   remain the deeper service owners. Focused entity-surface, REST entity
   detail/mutation, public-surface, and Ruff checks passed.
   MCP graph-state tool and graph/entity resources now have route-facing public
   surface helpers. `server/engram/retrieval/graph_state.py` now owns MCP graph
   tool dispatch, graph stats resource shaping, entity profile resource
   dispatch, and entity-neighbor resource dispatch on top of the existing
   `GraphStateService` read model. Focused MCP graph-state surface, graph-state
   service/resource, MCP graph-state, public-surface, and Ruff checks passed.
   REST graph neighborhood/temporal route response assembly now also uses
   `server/engram/retrieval/graph_state.py` helpers for manager dispatch,
   not-found payloads, timestamp parsing, and invalid-timestamp payloads.
   MCP identity-core and consolidation controls now have route-facing public
   surface helpers. `server/engram/retrieval/identity_core.py` owns MCP
   identity-core manager dispatch, and `server/engram/consolidation_trigger.py`
   owns MCP trigger dispatch, consolidation status reads, and cycle-summary
   shaping. The MCP transport still owns JSON wrapping and active
   consolidation-store selection. Focused identity-core, consolidation controls,
   MCP trigger/status, public-surface,
   consolidation-presenter, and Ruff checks passed.
   MCP lifecycle summary now has a route-facing public surface helper.
   `server/engram/lifecycle_summary.py` owns active audit-store reader
   construction, inactive-engine placeholder wiring, and limit clamping for MCP
   lifecycle reads. The MCP transport still owns JSON wrapping and session state
   lookup. Focused lifecycle summary, MCP lifecycle, public-surface, and Ruff
   checks passed.
   Knowledge-chat conversation persistence is now in
   `server/engram/retrieval/chat_persistence.py`, covering conversation
   validation/creation, active-`group_id` not-found handling, completed-turn
   persistence, recalled entity tagging, and the chat conversation not-found
   payload body. Focused chat-persistence, conversation ownership, chat,
   public-surface, full knowledge API, and Ruff checks passed.
   REST conversation CRUD now uses
   `server/engram/retrieval/conversation_persistence.py`, covering group-scoped
   listing, creation, message reads/appends, title updates, deletes, and
   not-found translation plus the REST response envelopes and shared not-found
   body. Focused conversation-persistence, conversation API, public-surface, and
   Ruff checks passed.
   REST/MCP post-write adjudication request loading now uses
   `server/engram/ingestion/adjudication_surface.py`, covering the compatibility
   lookup for sync/async manager facades and missing/malformed responses before
   shared memory-write presenters run. Focused adjudication-surface, REST
   remember, MCP JSON-response, public-surface, and Ruff checks passed.
   REST/MCP live conversation manager-facade access now uses helpers in
   `server/engram/retrieval/context.py`, covering defensive sync/async/type
   checks for conversation context, embed functions, turn counts, session entity
   names, recent turns, and live-turn ingestion. Focused conversation-runtime,
   chat-context, MCP piggyback, public-surface, and Ruff checks passed.
   REST/MCP brain-loop evaluation report assembly is now in
   `server/engram/evaluation/report_service.py`, covering graph-state reads,
   runtime Recall metrics snapshot persistence/reload, saved label reads, and
   report construction. MCP active audit-store reads, cycle-limit clamping, and
   calibration snapshot loading now live in that report service too, while REST
   still supplies engine-derived cycle context. Focused evaluation report
   service, REST evaluation, MCP JSON-response, public-surface, and Ruff checks
   passed.
   REST/MCP evaluation label writes are now in
   `server/engram/evaluation/label_service.py`, covering recall-quality and
   session-continuity sample construction, count clamping, active-`group_id`
   persistence, and shared write acknowledgement payloads through route-facing
   helpers. Focused label service, REST
   evaluation, MCP JSON-response, public-surface, and Ruff checks passed.
   The broad non-Docker/non-external-Helix backend gate now passes with 2933
   tests, 43 skips, and 236 deselections after these route-orchestration
   slices.
   MCP auto-recall cooldown/topic deduplication, compact query extraction,
   per-tool recall gating, first-call session-prime planning, and MCP
   middleware side-effect planning now live in
   `server/engram/retrieval/auto_recall.py`. Auto-recall result compaction also
   lives there now, including lite/medium entity-probe response shaping, score
   filtering, compact entity summaries, top-fact truncation, cue-episode
   payloads, packet attachment, and the no-surfaceable-results decision.
   The same helper module now owns additive MCP response enrichment for session
   context, recalled context, triggered intentions, and memory notifications.
   The MCP middleware still owns tool-specific fetching and transport behavior,
   which remains an intentional compatibility surface pending any further
   extraction.
   Completion still needs a continued review of remaining
   route-local orchestration that may hide lifecycle logic, plus explicit notes
   for any intentionally retained compatibility behavior.

2. Final dashboard verification:
   The lifecycle/evaluation UI has a refreshed default gate: full Vitest passes
   with 214 tests and 1 skipped live-native test, and the production build
   passes with the existing large-chunk warning. The live native dashboard smoke
   also passes against a seeded PyO3 REST server when both app and auth default
   groups are set to `native_brain`. The config contract now allows future
   unauthenticated REST runs to omit `ENGRAM_AUTH__DEFAULT_GROUP_ID` when they
   want it to follow `ENGRAM_DEFAULT_GROUP_ID`; this is covered by config tests,
   not by a separate live dashboard smoke yet. The patched rerun after native
   evidence update normalization also shuts down without the previous
   `update_evidence` decode errors. Before closure after any further UI/API
   changes, rerun these gates or explicitly carry this snapshot as the final
   dashboard evidence.

3. Native/full-mode boundary decision:
   PyO3 native is now the main path. Docker Helix/full-mode remains a separate
   explicit gate; the goal needs a final decision on whether Docker/full is
   out-of-scope or must pass before completion.

4. Evaluation confidence:
   The report and labels exist, but a production-grade claim needs a final
   evaluation run with enough real or benchmark-labeled data to justify recall
   quality, calibration quality, projection yield, cue usefulness, false recall,
   and consolidation-effect confidence.

5. Completion packaging:
   The repo has a large dirty worktree. Before closure, the intended files,
   ignored docs, generated artifacts, and unrelated user changes need a final
   packaging/staging plan so the completed work is reproducible.

## Next Concrete Work

Continue the REST/MCP route orchestration audit against the service boundaries
already extracted. The consolidation audit-store and knowledge-chat event
presenter slices are complete, knowledge-chat tool execution payloads have a
retrieval helper, chat recall feedback/retry policy has a retrieval helper,
chat memory-need/live-context runtime has a retrieval helper, chat conversation
persistence and not-found payloads have a helper boundary, REST/MCP explicit recall result/packet
assembly and MCP explicit recall enrichment have a retrieval helper,
recall-control manager compatibility has shared helpers, REST/MCP artifact
search has retrieval helpers, REST conversation CRUD
has group-scoped persistence and response-envelope helpers, REST/MCP project
bootstrap/runtime-state calls have surface helpers, REST/MCP public entity/fact
lookup has surface helpers, REST/MCP public agent-context response assembly has
surface helpers, REST/MCP
adjudication resolution has ingestion-surface helpers, REST/MCP
Capture write dispatch has capture-surface helpers, REST entity detail/mutation
response assembly has a public-surface helper, REST/MCP graph-state resources
and graph route payloads have public-surface helpers, REST and MCP consolidation controls/read payloads have
public-surface helpers, MCP identity-core has a public-surface helper, MCP
lifecycle summary has a public-surface helper, REST/MCP
deterministic question routing has retrieval helpers, REST/MCP
prospective-memory intentions have retrieval helpers, REST/MCP
forget target dispatch has retrieval helpers, REST/MCP explicit preference
feedback has retrieval helpers, REST/MCP
post-write adjudication request loading has an ingestion helper, REST/MCP
live conversation manager-facade access uses retrieval helpers, REST/MCP
evaluation report assembly shares a service, MCP evaluation report audit-store
input loading shares that report service, REST/MCP evaluation label writes share
write-surface helpers, and MCP auto-recall policy helpers have been extracted,
including first-call session-prime planning and middleware side-effect
planning. Auto-recall result compaction and additive MCP response enrichment
have also been extracted for both lite/medium entity-probe and full recall
surfaces. The next likely area is any remaining REST/MCP route-local
orchestration that still hides lifecycle behavior rather than surface transport
details.
