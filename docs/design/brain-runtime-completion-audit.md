# Brain Runtime Completion Audit

Date: 2026-05-18

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
| Preserve useful dirty-worktree changes | Worktree remains dirty; current work is scoped to shared companion-store bootstrap, explicit dependency injection, public smoke feedback, tests, and docs | Needs final packaging discipline |
| Make `Capture -> Cue -> Project -> Recall -> Consolidate` explicit | `server/engram/lifecycle_summary.py`, `dashboard/src/components/LifecyclePanel.tsx`, `server/engram/evaluation/brain_loop_report.py` | Strong |
| Extract capture/observe/store runtime boundaries | `server/engram/ingestion/capture_service.py`, `episode_ingestion.py`, `offline_replay.py`, `dedup.py` | Strong |
| Extract project runtime boundaries | `server/engram/ingestion/projection_service.py`, `projection_execution.py`, `projection_state.py` | Strong |
| Extract recall runtime boundaries | `server/engram/retrieval/service.py`, `presenter.py`, `context_builder.py`, `entity_probe.py`, `graph_state.py` | Strong |
| Extract consolidation orchestration boundaries | `server/engram/consolidation/lifecycle.py`, `phase_runner.py`, `events.py`, `completion.py`, `phase_catalog.py` | Strong |
| Keep `GraphManager` as compatibility facade, not hidden runtime brain | `server/tests/test_graph_manager_facade_boundaries.py` guards 61 core and compatibility delegates across lifecycle, evidence, artifacts, lookup, forgetting, intentions, context, graph state, and recall interactions; consolidation audit reads now use `ConsolidationAuditReader`; MCP auto-recall lite/full dispatch, session prime, auto-observe piggybacking, shaping, enrichment, middleware execution, and entity/fact/artifact/context/question-route tool middleware now use retrieval helpers; the direct manager-dispatch scan across REST API routes and `mcp/server.py` is now guarded, allowing only MCP shutdown resource closing; REST API routes are guarded against direct `engine.*` and store/service method dispatch and are limited to directly awaiting route-facing helpers; decorated MCP public surfaces are guarded against direct store/session method dispatch and arbitrary direct awaited runtime calls | Strong for `GraphManager`; broader route-local orchestration audit remains ongoing |
| Align REST and MCP remember/observe/recall semantics | Shared presenters in ingestion/retrieval plus REST/MCP tests | Strong |
| Align backend/dashboard lifecycle contracts | `dashboard/src/components/LifecyclePanel.tsx`, `dashboard/src/constants/consolidation.ts`, backend phase registry tests | Strong |
| Preserve one-brain-per-person `group_id` semantics | `server/tests/test_group_scope_static_contract.py`, native parity tests, active `native_brain` coverage, default-group config inheritance tests | Strong |
| Keep SQLite/lite viable | Broad gate: `3238 passed, 43 skipped, 236 deselected` for `pytest -m "not requires_docker and not requires_helix"` plus shared lite DB initialization helpers in `server/engram/storage/bootstrap.py` | Strong |
| Make PyO3 native Helix the preferred full path | README/install docs, native smoke, native parity suite, `engram.quality.native_surface_manifest`, native operator gate tracking for `engram evaluate --mode helix --require-evaluation-signals`, and `engram doctor --mode helix` reporting smoke evaluation readiness | Strong |
| Keep Helix/full-mode external tests isolated | `requires_helix`/`requires_docker` deselection and native no-Docker parity | Strong for local gates; Docker/full still separate |
| Build evaluation loop | `server/engram/evaluation/brain_loop_report.py`, REST/MCP label/report surfaces, dashboard Evaluate panel, smoke verifier, structured `evaluation_signals` readiness map, `engram evaluate --require-evaluation-signals`, and doctor smoke readiness output; projected/consolidated smoke and normal CLI reports can now fail if required signals are missing or unmeasured | Strong, needs more real labeled evidence before production claim |
| Update docs/handoff as decisions become real | `docs/CURRENT_HANDOFF.md`, `docs/design/brain-runtime-audit.md` | Strong, ongoing |
| Do not mark complete until implementation, tests, docs, UI are understandable | This audit says not complete | Blocking |

## Current Verification Evidence

- Backend non-Docker/non-external-Helix gate:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  currently passes with 3238 tests, 43 skips, and 236 deselections after the
  doctor readiness failure path was guarded, the Helix dashboard analytics test
  fixture was made date-stable, and REST companion-store plus CLI/MCP
  consolidation/evaluation store creation was centralized in the shared
  bootstrap helper, the notification/scheduler dependencies were made explicit,
  and the smoke cue-feedback path moved onto the public manager facade.
- Shared lite storage bootstrap evidence:
  `server/engram/storage/bootstrap.py` centralizes companion-store
  initialization against the active graph store. REST startup, MCP startup,
  lifecycle CLI, evaluation CLI, and projected/consolidated smoke now share its
  store factory path where applicable. REST startup creates atlas,
  consolidation, evaluation, and conversation stores through that module; MCP
  and CLI/operator paths share its consolidation/evaluation store factories.
  Lite borrowed-DB, borrowed consolidation fallback, and Helix shared-client
  behavior are guarded by `tests/storage/test_storage_bootstrap.py` plus the
  borrowed-connection contract tests.
- Public dependency-boundary evidence:
  MCP notification piggybacking now uses the pure
  `build_mcp_notifications_surface()` presenter with a
  `NotificationSurfaceService` supplied by `api/deps.py`, and the consolidation
  scheduler receives its graph store explicitly for temporal scans. Static
  public-surface tests guard against reintroducing `_app_state` reads in
  `notifications.surface` or `consolidation/scheduler.py`.
- Public feedback-write evidence:
  The projected/consolidated smoke writes surfaced cue feedback through
  `GraphManager.apply_memory_interaction()` instead of manager private cue-hit
  helpers, keeping the smoke verifier on the same public facade used by runtime
  consumers.
- Runtime resource shutdown evidence:
  `GraphManager.close_runtime_resources()` closes owned search, activation, and
  graph stores through `engram.storage.bootstrap.close_if_supported()`, and MCP
  lifespan shutdown now calls that facade instead of reading private manager
  store fields.
- Episode worker runtime-store evidence:
  `server/engram/ingestion/worker_runtime.py` defines the graph, activation, and
  search stores needed by `EpisodeWorker`. REST and MCP startup pass those
  stores explicitly, while `GraphManager.get_episode_worker_runtime_stores()`
  remains a compatibility accessor for direct worker construction.
- Episode worker batching evidence:
  `server/engram/ingestion/worker_batching.py` owns adjacent auto-capture turn
  merging, primary cue rebuild, merged-away cue retirement, and cue re-indexing.
  `EpisodeWorker` now keeps queue consumption, deterministic scoring, and
  projection routing instead of embedding that Cue-stage lifecycle mutation.
- Episode worker scoring evidence:
  `server/engram/ingestion/worker_scoring.py` owns deterministic worker triage
  scoring, multi-signal scorer access, goal boost lookup, and projection-yield
  feedback. `EpisodeWorker` delegates scoring and outcome recording while
  keeping queue event routing and Project-stage dispatch.
- Episode worker routing evidence:
  `server/engram/ingestion/worker_routing.py` owns duplicate projection guards,
  system-discourse cue-only skips, skip/defer projection-state sync, and the
  boolean project-now routing contract. `EpisodeWorker` keeps event
  consumption, batch timing, and Project-stage dispatch.
- Episode worker event evidence:
  `server/engram/ingestion/worker_events.py` owns EventBus payload parsing for
  queued and scheduled-projection episodes plus compact auto-capture content
  loading. `EpisodeWorker` now subscribes, batches, routes, and dispatches
  without embedding raw `episodeId` payload keys or event names.
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
  REST admin benchmark loading now calls a route-facing helper backed by
  `GraphManager.load_benchmark_corpus()` and `BenchmarkLoadService` instead of
  calling the manager directly or reading private graph, activation, or search
  stores in `server/engram/api/admin.py`.
  REST graph neighborhood and temporal graph routes now call
  `GraphManager.get_graph_neighborhood()` and `get_temporal_graph()` backed by
  `GraphStateService` instead of reading private graph, activation, or config
  fields in `server/engram/api/graph.py`; route-facing graph-state helpers now
  also own REST missing-entity and invalid-timestamp payloads for those routes.
  The REST entity-neighbor convenience route now calls the same graph-state
  surface helper directly instead of importing the graph route handler, with
  focused entity-neighbor/graph-state/public-surface coverage.
  REST atlas snapshot/history/region routes now use
  `server/engram/retrieval/atlas_surface.py` for atlas service dispatch,
  representation metadata, snapshot/history serialization, and region/snapshot
  404 payloads instead of assembling those response bodies in
  `server/engram/api/graph.py`.
  MCP recall-response enrichment now calls manager facades backed by
  `RecallResponseStateService` instead of reading private activation,
  near-miss, surprise-cache, or triggered-intention fields in
  `server/engram/mcp/server.py`. MCP auto-recall response payload attachment
  also delegates to `apply_mcp_recall_enrichment()` so the additive
  `session_context`, `recalled_context`, `triggered_intentions`, and
  `memory_notifications` keys share one retrieval-side contract.
  REST dashboard stats now calls a route-facing graph-state helper backed by
  `GraphManager.get_dashboard_stats()` and `GraphStateService` instead of
  reading the graph store or calling the manager directly in
  `server/engram/api/stats.py`.
  REST activation monitor reads now call
  route-facing graph-state helpers backed by `GraphManager.get_activation_snapshot()`,
  `get_activation_curve()`, and `GraphStateService` instead of reading app
  state, graph store, activation store, config, `compute_activation`, or
  route-local curve 404 status/payloads directly in
  `server/engram/api/activation.py`.
  REST episode dashboard reads now call a route-facing graph-state helper backed
  by `GraphManager.list_episode_summaries()` and `GraphStateService` instead of
  calling the manager directly, reading the graph store, or formatting paginated
  episode/cue payloads in `server/engram/api/episodes.py`.
  REST/MCP lifecycle summary reads now call
  `GraphManager.get_lifecycle_summary()` backed by `LifecycleSummaryService`
  instead of calling `build_lifecycle_summary` and passing graph/config facades
  from `server/engram/api/lifecycle.py` or `server/engram/mcp/server.py`, and
  both transports now have route-facing lifecycle helper coverage.
  Dashboard WebSocket activation-monitor snapshots now call
  a route-facing graph-state helper backed by
  `GraphManager.get_activation_snapshot()` instead of calling the manager
  directly, reading app-state graph/activation/config stores, or computing
  activation inside `server/engram/api/websocket.py`.
  Dashboard WebSocket event/command payload shaping now lives in
  `server/engram/api/websocket_surface.py`, covering event flattening, `pong`,
  `resync`, activation snapshot envelopes, and connected-group notification
  dismiss dispatch while the route keeps socket auth, task lifecycle, and JSON
  transport.
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
  and operator native smoke evidence. Its verifier compares REST/MCP surfaces
  to the live route/decorator tables, checks runtime evidence against real
  native parity helper/test function names, and checks dashboard/operator/static
  evidence paths against the repo.
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
   `get_rate_limiter()` plus `check_api_chat_rate_limit()` so limiter execution
   lives outside the route, and REST health now uses `get_graph_store()`,
   `get_config()`, `get_mode()`, and `build_api_health_surface()` so graph-store
   probing plus status aggregation live outside the route. The REST/MCP
   route-private manager read scan is now clean, and the activation/episode/
   lifecycle/WebSocket activation-monitor/notification scans are clean for
   route-local app-state, store, config, activation-compute, episode/cue
   formatting, lifecycle-summary builder, socket-local activation math, and
   notification formatting/store reads; the WebSocket, knowledge, and health
   routes no longer import `_app_state` directly. The public-surface guard now
   generates coverage for every API route module except `api/deps.py`, and the
   route-local `_app_state` scan is clean outside that dependency layer. The
   same guard now rejects direct REST/MCP route manager method dispatches except
   MCP shutdown resource closing, and it rejects direct REST route store/service
   method dispatch so route modules keep dependency lookup separate from
   lifecycle/service execution. REST route functions are also limited to
   directly awaiting route-facing helpers. Decorated MCP public tools,
   resources, and prompts are now guarded against direct store/session method
   dispatch and arbitrary direct awaited runtime calls, while MCP
   startup/shutdown store initialization stays outside that public surface guard.
   The REST evaluation report also reads consolidation cycles/calibration
   snapshots through `build_api_brain_loop_evaluation_surface()` in
   `server/engram/evaluation/report_service.py` instead of calling
   `engine.get_recent_evaluation_context()` from the route or reading
   `engine._store`; focused evaluation/consolidation/static checks passed, and
   the public-surface guard now rejects direct REST route `engine.*` dispatch.
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
   `server/engram/retrieval/chat_events.py`, including rich memory UI event
   selection, chat recall round-tripping, Anthropic tool-result message shaping,
   recall/fact tool JSON accumulation, and AI SDK synthetic tool payload-pair
   construction, with the REST route retaining only SSE wrapping for those
   payloads. Focused chat-event, recall-presenter, public-surface, and full
   knowledge API checks passed.
   Knowledge-chat tool execution payloads are now in
   `server/engram/retrieval/chat_tools.py`, covering recall/search_entities/
   search_facts LLM payloads, chat recall packet shaping, fact deduplication,
   unknown-tool responses, and the non-streaming Anthropic tool-use loop/result
   accumulation plus the chat tool schema, Anthropic text-block extraction, and
   JSON-string compatibility wrapper used by legacy tests, while the REST route
   keeps Anthropic client construction and SSE framing. Focused chat-tool, chat API, chat-event,
   public-surface, and Ruff checks passed.
   Knowledge-chat recall feedback and retry policy are now in
   `server/engram/retrieval/chat_feedback.py`, covering used/dismissed memory
   interaction application, generic memory-free response detection, retry
   gating, and retry system-prompt construction, while
   `server/engram/retrieval/chat_tools.py` owns retry provider execution.
   Focused chat-feedback, chat-tool, full knowledge API, chat-event,
   public-surface, Ruff, and `git diff --check` gates passed.
   Knowledge-chat response-turn orchestration is now in
   `server/engram/retrieval/chat_runtime.py`, covering chat memory-need
   analysis, memory-guidance text, live conversation hydration, assistant-turn
   recording, recent-turn extraction, chat runtime policy lookup, chat
   epistemic-evidence dispatch, baseline context dispatch, system-prompt
   assembly, sliding-window message assembly, tool-use loop invocation, retry
   policy application, recall feedback, route-neutral rich tool stream payloads,
   route-neutral text stream payloads, and chat rate-limit execution/response
   payloads while the REST route keeps rate-limiter dependency lookup, conversation
   helper invocation, Anthropic client construction, SSE wrapping, and
   persistence scheduler invocation. Focused chat-runtime/feedback/tool, prompt/message,
   full knowledge API, chat-event, public-surface, Ruff, and `git diff --check`
   gates passed. The latest response-turn orchestration check passed with 174
   focused tests.
   REST/MCP explicit recall result and packet assembly are now in
   `server/engram/retrieval/recall_surface.py`, covering the explicit
   Recall-stage manager call, recall packet analysis, memory packet assembly,
   API/MCP recall item presentation, MCP entity-name/access-count resolution,
   MCP near-miss/surprise side-channel enrichment, MCP query timing,
   recall-session flag updates, and MCP recall middleware invocation while REST
   and MCP keep their transport-specific response shapes. Focused knowledge API,
   MCP JSON-response, autorecall, chat, public-surface, Ruff, and
   `git diff --check` gates passed.
   Recall-need threshold resolution and memory-need analysis recording now share
   `server/engram/retrieval/control.py` helpers, covering sync/async manager
   facade compatibility for REST, MCP, chat runtime, chat tool execution, and
   explicit recall surfaces. Focused recall-control, knowledge API, MCP
   JSON-response, autorecall, chat, public-surface, Ruff, and `git diff --check`
   gates passed.
   REST/MCP artifact search result assembly now shares
   `server/engram/retrieval/artifacts.py` helpers, covering artifact hit loading
   and item serialization while REST keeps `projectPath` and MCP keeps
   `project_path`. MCP artifact-search tool-surface helpers now also own recall
   middleware invocation for that read tool, so `server/engram/mcp/server.py`
   keeps manager lookup, callback injection, and JSON wrapping. Focused
   artifact-search, artifact service, REST artifact endpoint, MCP artifact,
   public-surface, Ruff, and `git diff --check` gates passed.
   REST/MCP deterministic question routing now shares
   `server/engram/retrieval/epistemic_route.py` helpers, covering route history
   normalization and the manager `route_question` call while REST keeps HTTP
   response wrapping. MCP question-route tool-surface helpers now also own
   recall middleware invocation with `auto_observe=True`, so
   `server/engram/mcp/server.py` keeps session entity lookup, callback
   injection, and JSON wrapping. Focused route-surface, REST epistemic endpoint,
   MCP JSON-response, public-surface, Ruff, and `git diff --check` gates passed.
   REST/MCP prospective-memory intention surfaces now share
   `server/engram/retrieval/prospective.py` helpers, covering intention create,
   list, and dismiss manager calls plus API/MCP acknowledgement shapes while
   REST keeps HTTP wrapping and MCP keeps JSON wrapping. REST intention
   validation/not-found payload bodies, REST create/dismiss status mapping, and
   MCP create/dismiss error payloads now live in the same helper module. Focused prospective-surface, public-surface,
   full knowledge API, full MCP tool, Ruff, and `git diff --check` gates passed.
   REST/MCP forget entity/fact surfaces now share
   `server/engram/retrieval/forgetting.py` helpers, covering target dispatch and
   fact-field normalization while REST keeps entity-first behavior for dual
   targets and MCP keeps exactly-one-target validation. REST missing-target
   payloads and 400/404 response mapping now live in the same route-facing
   helper. Focused forget-surface, REST forget, MCP forget, public-surface,
   Ruff, and `git diff --check` gates passed.
   REST/MCP explicit preference feedback now shares
   `server/engram/retrieval/preference_feedback.py` helpers, covering public
   rating validation, the `record_explicit_feedback` manager call, REST error
   payloads, and MCP invalid-rating error payloads while REST keeps HTTP status
   mapping. Focused feedback-surface, feedback recorder, full knowledge API,
   full MCP tool, public-surface, Ruff, and `git diff --check` gates passed.
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
   validation. REST keeps camelCase `items`; MCP keeps raw lookup results.
   MCP entity/fact lookup tool-surface helpers now also own the recall
   middleware invocation for those read tools, so `server/engram/mcp/server.py`
   keeps manager lookup, callback injection, and JSON wrapping. Focused
   lookup-surface, REST facts, MCP entity/fact search, MCP middleware,
   public-surface, and Ruff checks passed.
   REST/MCP public agent-context assembly now shares route-facing context
   helpers. `server/engram/retrieval/context_builder.py` still owns the deeper
   `MemoryContextBuilder`, and now also owns REST context payload shaping and MCP
   raw context manager access. REST keeps camelCase count/token fields; MCP keeps
   the raw `get_context` shape. MCP context tool-surface helpers now also own
   recall/notification middleware invocation for that read tool, so
   `server/engram/mcp/server.py` keeps manager lookup, callback injection, and
   JSON wrapping. Focused context-surface, tiered context, REST context/runtime,
   MCP context middleware, public-surface, and Ruff checks passed.
   REST/MCP adjudication resolution now shares ingestion-side surface helpers.
   `server/engram/ingestion/adjudication_surface.py` owns the public
   client-adjudication manager dispatch and API/MCP outcome shaping for resolved
   edge-adjudication work items. REST keeps camelCase IDs; MCP keeps snake_case
   IDs. Focused adjudication-surface, REST adjudicate, MCP adjudicate,
   public-surface, and Ruff checks passed.
   REST/MCP public Capture writes now share route-facing capture helpers.
   `server/engram/ingestion/capture_surface.py` owns public conversation-date
   parsing, attachment construction, raw observation storage dispatch, and
   Capture -> Project ingest dispatch. REST observe/image/file/remember now
   route memory-write presentation and adjudication request loading through the
   helper too, while REST routes keep tenant/dependency lookup and JSON
   wrapping. MCP write tools now route through that module for session activity
   updates, live-turn recording, adjudication-request loading, memory-write
   presentation, and recall middleware invocation as well. `server/engram/mcp/server.py`
   keeps manager/session lookup, JSON wrapping, and tool signatures. REST
   auto-observe routes enablement, short-content skip, dedup skip, raw
   observation storage, and memory-write presentation through the same capture
   surface while the route keeps dependency lookup and JSON wrapping. Focused
   capture-surface, memory-write presenter, REST observe/remember/adjudication,
   REST auto-observe, MCP write/adjudication, public-surface, and Ruff checks passed.
   REST entity detail/update/delete now has a route-facing public-surface helper.
   `server/engram/retrieval/entity_surface.py` owns entity detail manager
   dispatch, sparse update payload construction, delete dispatch, 404 status
   mapping, and the shared REST not-found payload, while `GraphStateService` and
   `EntityMutationService` remain the deeper service owners. Focused
   entity-surface, REST entity detail/mutation, public-surface, and Ruff checks
   passed.
   MCP graph-state tool and graph/entity resources now have route-facing public
   surface helpers. `server/engram/retrieval/graph_state.py` now owns MCP graph
   tool dispatch, graph stats resource shaping, entity profile resource
   dispatch, and entity-neighbor resource dispatch on top of the existing
   `GraphStateService` read model. Focused MCP graph-state surface, graph-state
   service/resource, MCP graph-state, public-surface, and Ruff checks passed.
   REST graph neighborhood/temporal route response assembly now also uses
   `server/engram/retrieval/graph_state.py` helpers for manager dispatch,
   not-found payloads, timestamp parsing, invalid-timestamp payloads, and the
   entity-neighbor convenience route.
   REST atlas snapshot/history/region response assembly now uses
   `server/engram/retrieval/atlas_surface.py` helpers for atlas service
   dispatch, representation metadata, snapshot/history serialization,
   passthrough region drill-down payloads, and region/snapshot lookup payloads.
   Focused atlas helper, graph atlas API, and public-surface checks passed.
   MCP identity-core and consolidation controls now have route-facing public
   surface helpers. `server/engram/retrieval/identity_core.py` owns MCP
   identity-core manager dispatch, and `server/engram/consolidation_trigger.py`
   owns MCP trigger dispatch, active-store/shared-DB fallback resolution,
   consolidation status reads, and cycle-summary shaping. The MCP transport
   still owns JSON wrapping and session-state store references. Focused
   identity-core, consolidation controls, MCP trigger/status, public-surface,
   consolidation-presenter, and Ruff checks passed.
   REST and MCP lifecycle summary now have route-facing public surface helpers.
   `server/engram/lifecycle_summary.py` owns the REST runtime-context manager
   call shape plus MCP active audit-store reader construction, inactive-engine
   placeholder wiring, and limit clamping. The routes keep JSON wrapping and
   transport-local dependency/session lookup. Focused lifecycle summary,
   API lifecycle, public-surface, and Ruff checks passed.
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
   body/status mapping. Focused conversation-persistence, conversation API,
   public-surface, and Ruff checks passed.
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
   calibration snapshot loading now live in that report service too, and REST
   engine-derived cycle context loading is behind the route-facing
   `build_api_brain_loop_evaluation_surface()` helper. Focused evaluation report
   service, REST evaluation, MCP JSON-response, public-surface, and Ruff checks
   passed.
   REST/MCP evaluation label writes are now in
   `server/engram/evaluation/label_service.py`, covering recall-quality and
   session-continuity sample construction, count clamping, active-`group_id`
   persistence, and shared write acknowledgement payloads through route-facing
   helpers. Focused label service, REST
   evaluation, MCP JSON-response, public-surface, and Ruff checks passed.
   The broad non-Docker/non-external-Helix backend gate now passes with 3238
   tests, 43 skips, and 236 deselections after these route-orchestration
   slices, the Python 3.13 event-loop test harness cleanup, the doctor
   readiness failure-path guard, the date-stable Helix dashboard analytics
   fixture, the shared companion-store bootstrap follow-up, explicit
   notification/scheduler dependency cleanup, and the public smoke cue-feedback
   facade.
   MCP auto-recall cooldown/topic deduplication, compact query extraction,
   per-tool recall gating, first-call session-prime planning, MCP middleware
   side-effect planning, middleware plan execution, middleware auto-observe,
   read-tool live-turn ingestion, first-call session prime, lite auto-recall,
   triggered-intention draining, notification lookup, additive response
   enrichment, and auto-recall result compaction now live in
   `server/engram/retrieval/auto_recall.py`. That includes lite/medium
   entity-probe response shaping, score filtering, compact entity summaries,
   top-fact truncation, cue-episode payloads, packet attachment, and the
   no-surfaceable-results decision. `server/engram/mcp/server.py` keeps the
   compatibility wrappers plus tool-specific fetching, session/global dependency
   lookup, JSON wrapping, and transport behavior.
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
   The report and labels exist, and the report now exposes structured
   `evaluation_signals` for cue usefulness, projection yield, recall quality,
   false recall, triage calibration, and consolidation effect. The projected/
   consolidated smoke verifier fails if those records are missing or unmeasured.
   The normal `engram evaluate --require-evaluation-signals` CLI path applies
   the same hard gate to JSON exports, live lite reports, and native PyO3
   reports outside the smoke harness. `--from-json` also accepts
   already-rendered brain-loop report JSON so saved report artifacts can be
   verified directly, and the native surface manifest tracks the Helix variant
   as an operator hard gate. `engram doctor` now surfaces the same
   evaluation-signal readiness in its disposable smoke report, so the local
   diagnostic path shows whether the six operator signals are measured instead
   of only reporting smoke totals and coverage-gap counts.
   A production-grade claim still needs a final evaluation run with enough real
   or benchmark-labeled data to move those readiness records from smoke coverage
   to meaningful evidence.

5. Completion packaging:
   The current dirty scope is intentionally bounded to the shared
   companion-store bootstrap follow-up, its focused storage/CLI/MCP tests, and
   audit/handoff docs. Before closure or a commit, the intended files still
   need a final packaging/staging plan so the completed work is reproducible and
   unrelated user changes stay out of scope.

## Next Concrete Work

Continue the REST/MCP route orchestration audit against the service boundaries
already extracted. The consolidation audit-store and knowledge-chat event
presenter slices are complete, knowledge-chat tool execution payloads have a
retrieval helper, the knowledge-chat tool-use loop/result accumulation has a
retrieval helper, chat recall feedback/retry policy has a retrieval helper,
chat response-turn orchestration has a retrieval helper,
chat memory-need/live-context runtime has a retrieval helper, chat conversation
persistence, scheduling, and not-found payloads have a helper boundary, REST/MCP explicit recall result/packet
assembly, MCP explicit recall enrichment, and MCP explicit recall session/timing
side effects have a retrieval helper,
recall-control manager compatibility has shared helpers, REST/MCP artifact
search has retrieval helpers, REST conversation CRUD
has group-scoped persistence and response-envelope/status helpers, REST/MCP project
bootstrap/runtime-state calls have surface helpers, REST/MCP public entity/fact
lookup has surface helpers, REST/MCP public agent-context response assembly has
surface helpers, REST/MCP
adjudication resolution has ingestion-surface helpers, REST/MCP
Capture write dispatch, MCP write orchestration, and offline replay manager
dispatch have capture-surface helpers, REST entity detail/mutation
response/status assembly has a public-surface helper, REST/MCP graph-state resources
plus REST dashboard read and graph route payloads have public-surface helpers, REST atlas
snapshot/history/region payloads have a public-surface helper, REST and MCP consolidation controls/read payloads have
public-surface helpers, MCP identity-core has a public-surface helper, MCP
lifecycle summary has a public-surface helper, REST/MCP
deterministic question routing has retrieval helpers, REST/MCP
prospective-memory intentions have retrieval helpers, REST/MCP
forget target dispatch has retrieval helpers, REST/MCP explicit preference
feedback has retrieval helpers, REST/MCP
post-write adjudication request loading and client-enabled surfacing gates have an ingestion helper, REST/MCP
live conversation manager-facade access uses retrieval helpers, REST/MCP
evaluation report assembly shares a service, REST evaluation report
engine-context loading and MCP audit-store input loading share that report
service, REST/MCP evaluation label writes share
write-surface helpers, dashboard WebSocket command/event payload shaping has a
route-facing helper, shared storage bootstrap initialization plus REST
companion-store, CLI/MCP consolidation/evaluation store creation, and borrowed
consolidation fallback reads have helper coverage, episode-worker runtime
stores have an explicit dependency object,
auto-capture worker batching has a Cue-stage helper, worker deterministic
scoring has an ingestion helper, worker projection routing has an ingestion
helper, worker event parsing and compact auto-content loading have an ingestion
helper, MCP Capture write orchestration has a capture-surface helper, and MCP
explicit recall session/timing side effects plus auto-recall middleware
execution have been extracted,
including first-call session-prime planning, middleware side-effect planning,
auto-recall result compaction, additive MCP response enrichment, lite/medium
entity-probe dispatch, full recall dispatch, session-prime context dispatch,
triggered-intention drain, middleware auto-observe storage, read-tool live-turn
ingestion, and MCP notification state lookup. The next likely area is any
remaining REST/MCP route-local orchestration that still hides lifecycle behavior
rather than surface transport details.
