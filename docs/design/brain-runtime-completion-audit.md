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
Agent adoption is part of the runtime contract: an MCP-connected Engram that
agents ignore because file-local memory is visible or the graph looks empty is
not yet doing its job.

## Prompt-To-Artifact Checklist

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Audit current architecture and drift | `docs/design/brain-runtime-audit.md`, `docs/CURRENT_HANDOFF.md` | Strong, ongoing |
| Preserve useful dirty-worktree changes | Current milestone slices have been committed and pushed through adoption-template/hook-trace coverage; latest status checks should start from a clean `main...origin/main` worktree | Strong, ongoing |
| Make `Capture -> Cue -> Project -> Recall -> Consolidate` explicit | `server/engram/lifecycle_summary.py`, `dashboard/src/components/LifecyclePanel.tsx`, `server/engram/evaluation/brain_loop_report.py` | Strong |
| Extract capture/observe/store runtime boundaries | `server/engram/ingestion/capture_service.py`, `episode_ingestion.py`, `offline_replay.py`, `dedup.py` | Strong |
| Extract project runtime boundaries | `server/engram/ingestion/projection_service.py`, `projection_execution.py`, `projection_state.py` | Strong |
| Extract recall runtime boundaries | `server/engram/retrieval/service.py`, `presenter.py`, `context_builder.py`, `entity_probe.py`, `graph_state.py` | Strong |
| Extract consolidation orchestration boundaries | `server/engram/consolidation/lifecycle.py`, `phase_runner.py`, `events.py`, `completion.py`, `phase_catalog.py` | Strong |
| Keep `GraphManager` as compatibility facade, not hidden runtime brain | `server/tests/test_graph_manager_facade_boundaries.py` guards 61 core and compatibility delegates across lifecycle, evidence, artifacts, lookup, forgetting, intentions, context, graph state, and recall interactions, and now scans runtime modules for direct `manager._*` / `graph_manager._*` / `_manager._*` access outside `graph_manager.py`; consolidation audit reads now use `ConsolidationAuditReader`; MCP auto-recall lite/full dispatch, session prime, auto-observe piggybacking, shaping, enrichment, middleware execution, and entity/fact/artifact/context/question-route tool middleware now use retrieval helpers; the direct manager-dispatch scan across REST API routes and `mcp/server.py` is now guarded, allowing only MCP shutdown resource closing; REST API routes are guarded against direct `engine.*` and store/service method dispatch and are limited to directly awaiting route-facing helpers; decorated REST API routes must have named orchestration-boundary entries; decorated MCP public surfaces are guarded against direct store/session method dispatch and arbitrary direct awaited runtime calls | Strong |
| Align REST and MCP remember/observe/recall semantics | Shared presenters in ingestion/retrieval plus REST/MCP tests | Strong |
| Make agents actually adopt Engram over overlapping file memory | MCP prompt authority contract, README automatic-memory behavior, setup wizard adoption checklist, prompt/setup tests, fresh-runtime bootstrap guidance, `claim_authority(project_path, user_message, file_memory_present)`, `validate_agent_protocol_calls()`, `engram adoption`, `engram adoption --template`, stdio MCP-client adoption coverage, copied Claude transcript regression, REST-mounted HTTP MCP discovery coverage, generated AutoCapture hooks, hook-trace validation, and a live Claude Code stream-json adoption transcript now tell agents Engram owns portable cross-context memory while project files own local scratch/conventions, return an `agent_protocol` with verifier metadata, generate live-harness transcript guidance, validate client transcripts, prove real MCP clients can follow the required pre-answer/capture flow, classify the observed file-memory bypass failure, verify Claude Code can discover `http://127.0.0.1:8100/mcp` when run outside the sandbox, and verify a full prompt-run transcript with observed `claim_authority`, `get_context`, `recall`, and `remember` | Strong for Claude Code and verifier/tooling; broader Cursor/Windsurf/live-client diversity remains future release evidence, not an unresolved verifier gap |
| Align backend/dashboard lifecycle contracts | `dashboard/src/components/LifecyclePanel.tsx`, `dashboard/src/constants/consolidation.ts`, backend phase registry tests | Strong |
| Preserve one-brain-per-person `group_id` semantics | `server/tests/test_group_scope_static_contract.py`, native parity tests, active `native_brain` coverage, default-group config inheritance tests | Strong |
| Keep SQLite/lite viable | Broad gate: `3320 passed, 43 skipped, 236 deselected` for `pytest -m "not requires_docker and not requires_helix"` plus shared lite DB initialization helpers in `server/engram/storage/bootstrap.py` | Strong |
| Make PyO3 native Helix the preferred full path | README/install docs, native smoke, native parity suite, `engram.quality.native_surface_manifest`, native operator gate tracking for `engram evaluate --mode helix --require-evaluation-signals`, and `engram doctor --mode helix` reporting smoke evaluation readiness | Strong |
| Keep Helix/full-mode external tests isolated | `requires_helix`/`requires_docker` deselection and native no-Docker parity | Strong for local gates; Docker/full still separate |
| Build evaluation loop | `server/engram/evaluation/brain_loop_report.py`, REST/MCP label/report surfaces, dashboard Evaluate panel, smoke verifier, structured `evaluation_signals` readiness map, `engram evaluate --require-evaluation-signals`, `--min-evaluation-signal-evidence`, `--require-benchmark-evidence`, `--human-label-template`, `--human-label-artifact`, `--require-human-label-evidence`, `--adoption-report`, `--require-adoption-evidence`, `--evidence-bundle`, and doctor smoke readiness output; projected/consolidated smoke and normal CLI reports can now fail if required signals are missing, unmeasured, below an operator evidence threshold, not paired with a valid showcase benchmark artifact, not paired with a real human-reviewed harness artifact, or not paired with a passed live-client adoption report when release evidence is requested; the full deterministic 39-scenario bundle passed for `engram_full` with pass rate `1.0`, false recall `0.0`, transcript hashes, fairness contract, and all six evaluation signals measured | Strong for local deterministic milestone and gate mechanics; real/labeled production artifact remains future release evidence |
| Update docs/handoff as decisions become real | `docs/CURRENT_HANDOFF.md`, `docs/design/brain-runtime-audit.md` | Strong, ongoing |
| Do not mark complete until implementation, tests, docs, UI are understandable | This audit says not complete | Blocking |

## Current Verification Evidence

- Backend non-Docker/non-external-Helix gate:
  `uv run pytest -m "not requires_docker and not requires_helix" -q`
  currently passes with 3320 tests, 43 skips, and 236 deselections after the
  doctor readiness failure path was guarded, the Helix dashboard analytics test
  fixture was made date-stable, and REST companion-store plus CLI/MCP
  consolidation/evaluation store creation was centralized in the shared
  bootstrap helper, the notification/scheduler dependencies were made explicit,
  the smoke cue-feedback path moved onto the public manager facade, and REST
  shutdown joined MCP in stopping and closing owned runtime resources through
  shared helpers and the manager facade, with shutdown consolidation orchestration
  in a named helper and static guards against reintroducing local stop/close or
  engine-cycle code. REST knowledge-chat SSE orchestration now lives in
  `stream_api_chat_sse_events()` with focused success/error stream tests,
  `build_api_chat_stream_response_surface()` owns route-facing chat rate-limit,
  conversation resolution, not-found, session-entity, and stream setup, and MCP
  instructions plus `claim_authority(project_path,
  user_message, file_memory_present)` now claim Engram's portable-memory
  authority plus empty-runtime bootstrap behavior. `claim_authority()` also
  returns an `agent_protocol` covering required pre-answer tools and capture
  routing when file memory is present, and `validate_agent_protocol_calls()`
  can score whether a client transcript actually followed that contract. A real
  stdio MCP-client test now starts `engram mcp`, follows the protocol, and
  validates the transcript; that live path also tightened onboarding so missing
  or stale project artifacts require bootstrap even when other runtime metrics
  exist. `engram setup` and README now surface the same adoption checklist for
  real MCP clients instead of only advising `get_context()`. `engram adoption`
  now gives operators a reproducible way to validate saved real-client
  transcripts against the same protocol, and `claim_authority()` returns the
  verifier command plus JSONL transcript schema in `agent_protocol.verification`.
  Its verification block also includes a `live_evidence_command` and JSON
  wrapper schema requiring live client metadata; `engram adoption
  --require-live-evidence` fails if a passing transcript lacks `client` and
  `capturedAt`, or if those fields are still template placeholders.
  `engram adoption --authority claim-authority.json --template` now generates
  the expected live-harness JSON wrapper from the saved authority payload so
  operators can collect actual Claude/Cursor/Windsurf logs without hand-shaping
  the transcript.
  The verifier also normalizes common real MCP log shapes such as
  `mcp__engram__recall`, nested `tool` / `function` / `tool_call` records,
  `stage` as an alias for `phase`, and Claude Code `--output-format
  stream-json` tool-use events. Raw Claude Code stream-json logs now also
  satisfy `--require-live-evidence` directly by inferring `client`, `session_id`,
  and `captured_at` from the live stream. A live Claude Code Sonnet prompt run
  against `http://127.0.0.1:8100/mcp` produced
  `/private/tmp/engram-claude-live-raw.jsonl`, and
  `engram adoption --authority /private/tmp/engram-live-claim-authority.json
  --calls /private/tmp/engram-claude-live-raw.jsonl --require-live-evidence
  --format markdown` passed with observed `claim_authority`, `get_context`,
  `recall`, and `remember`. `claim_authority()` also exposes
  `capture_required`, and its example transcript omits capture records when the
  protocol routes content to project-local scratch. The helper also preserves the previous
  failure-swallowing shutdown behavior by logging failed shutdown cycles, and
  `main._shutdown()` has dynamic coverage proving it delegates the active
  engine/config/logger.
- REST-mounted HTTP MCP discovery evidence:
  `server/engram/main.py` now mounts FastMCP's streamable HTTP app at the REST
  root so the SDK-owned `/mcp` route is actually served at the advertised
  `.mcp.json` URL. The parent REST lifespan starts the FastMCP session manager,
  and a narrow `GET /mcp` health shim supports Claude Code's discovery probe
  without intercepting real streamable HTTP POST/SSE traffic. MCP server
  lifespan initialization is refcounted so overlapping stateless HTTP requests
  cannot close shared stores while another request is initializing. Verified
  with `curl` initialize and `claude mcp list` outside the sandbox:
  `engram: http://127.0.0.1:8100/mcp (HTTP) - Connected`.
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
- SQLite relationship-weight update evidence:
  `SQLiteGraphStore.update_relationship_weight()` now consumes all
  `UPDATE ... RETURNING` rows before committing. This prevents dream-phase and
  shutdown-consolidation failures when reciprocal or duplicate relationship
  rows match one weight update. Guarded by
  `tests/test_consolidation_graph_methods.py::test_update_relationship_weight_consumes_all_returning_rows`
  plus the existing dream and shutdown consolidation tests.
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
  graph stores through `engram.storage.bootstrap.close_if_supported()`. REST and
  MCP shutdown now call that facade instead of reading private manager store
  fields, while worker/scheduler-like resources, MCP publisher shutdown,
  companion stores, and aclose-only clients use the shared stop/close helpers.
  Static public-surface checks now guard both shutdown paths. REST shutdown
  consolidation now calls `run_shutdown_consolidation()` instead of inspecting
  engine state or running/canceling cycles inline.
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
- Evidence-bundle smoke evidence:
  A disposable operator run generated `/private/tmp/engram-evidence-showcase/results.json`
  from the deterministic showcase benchmark for `temporal_override` and then ran
  `uv run engram evaluate --smoke --require-evaluation-signals
  --benchmark-artifact /private/tmp/engram-evidence-showcase/results.json
  --require-benchmark-evidence --min-benchmark-scenarios 1
  --min-benchmark-pass-rate 1.0 --evidence-bundle
  /private/tmp/engram-brain-loop-evidence.json --format json`. The command
  passed and wrote an `engram_brain_loop_evidence_bundle` with all six
  `evaluation_signals` measured, benchmark status `measured`, one available
  `engram_full` showcase scenario, pass rate `1.0`, false recall `0.0`, and one
  transcript hash. This proves the packaging path works, but it is still
  smoke-sized evidence, not the production-grade benchmark run needed for goal
  completion.
- Quick benchmark evidence:
  A follow-up local operator run generated
  `/private/tmp/engram-evidence-showcase-quick-20260518/results.json` with
  deterministic showcase `quick` mode, seed `7`, and the `engram_full` baseline
  across all four quick scenarios. The gated evaluation command required
  `--min-benchmark-scenarios 4 --min-benchmark-pass-rate 1.0` and wrote
  `/private/tmp/engram-brain-loop-evidence-quick-20260518.json`. The bundle
  passed with all six `evaluation_signals` measured, benchmark status
  `measured`, four available scenarios, pass rate `1.0`, false recall `0.0`,
  four transcript hashes, and a recorded `engram_full` fairness contract. This
  is stronger than the one-scenario packaging proof, but still local
  deterministic evidence rather than a production completion claim.
- Full deterministic benchmark evidence:
  A larger local operator run generated
  `/private/tmp/engram-evidence-showcase-full-20260518/results.json` with
  deterministic showcase `full` mode, seeds `7, 19, 31`, and the `engram_full`
  baseline across all 13 scenario transcripts. The gated evaluation command
  required `--min-benchmark-scenarios 39 --min-benchmark-pass-rate 1.0` and
  wrote `/private/tmp/engram-brain-loop-evidence-full-20260518.json`. The bundle
  passed with all six `evaluation_signals` measured, benchmark status
  `measured`, 39 available scenario runs, 39 passed, pass rate `1.0`, false
  recall `0.0`, 13 transcript hashes, and a recorded `engram_full` fairness
  contract. This is the strongest benchmark-labeled local evidence so far, but
  it is still not a substitute for live AI-harness adoption evidence.

## Completion Evidence And Remaining Gaps

1. Facade audit evidence (closed):
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
   The final facade audit now also has a whole-runtime static scan that rejects
   direct `manager._*`, `graph_manager._*`, or `_manager._*` access outside
   `server/engram/graph_manager.py`; focused facade-boundary coverage passes
   with 91 tests. The REST evaluation report also reads consolidation cycles/calibration
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
   The broad non-Docker/non-external-Helix backend gate now passes with 3320
   tests, 43 skips, and 236 deselections after these route-orchestration
   slices, the Python 3.13 event-loop test harness cleanup, the doctor
   readiness failure-path guard, the date-stable Helix dashboard analytics
   fixture, the shared companion-store bootstrap follow-up, explicit
   notification/scheduler dependency cleanup, and the public smoke cue-feedback
   facade, plus the REST/MCP shutdown stop/close facade cleanup, shutdown
   consolidation helper including failure logging coverage, REST chat SSE
   runtime extraction, REST chat response-surface extraction, MCP memory
   authority/onboarding prompt contract, `claim_authority()` callable/adoption
   contract, dashboard WebSocket auth route-boundary extraction, REST health
   route-boundary extraction, REST consolidation trigger scheduling extraction,
   REST consolidation status pressure/config extraction, static guards,
   adoption transcript stdin validation, and self-reported file-memory bypass
   classification.
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
   The route-local orchestration review is now guarded rather than open-ended:
   atlas warning logging and response wrapping now live in
   `build_api_atlas_json_response()`, and the remaining route branches are
   explicitly limited to chat JSON-vs-stream response wrapping plus WebSocket
   auth/session try/excepts. The static guard now
   discovers every decorated REST API route and fails if a route is missing from
   `PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES`, and it rejects nested function
   definitions inside decorated API route handlers and MCP public
   tools/resources/prompts. Dashboard WebSocket event and command loops now live
   in `run_dashboard_websocket_session()` instead of nested functions inside
   `dashboard_ws()`, and dashboard WebSocket tenant auth now lives in
   `server/engram/api/websocket_auth.py` instead of direct route-local
   `resolve_tenant_from_scope` plus query-token parsing.

2. Dashboard verification evidence (current):
   The lifecycle/evaluation UI has a refreshed default gate after the current
   REST/MCP adoption, route-boundary, adoption-template, and benchmark-evidence
   work: `pnpm test -- --run` passes with 214 tests and 1 skipped live-native
   test, and `pnpm build` passes with the existing large-chunk warning. The
   live native dashboard smoke also passes
   against a seeded PyO3 REST server when both app and auth default groups are
   set to `native_brain`. The config contract now allows future
   unauthenticated REST runs to omit `ENGRAM_AUTH__DEFAULT_GROUP_ID` when they
   want it to follow `ENGRAM_DEFAULT_GROUP_ID`; this is covered by config tests,
   not by a separate live dashboard smoke yet. The patched rerun after native
   evidence update normalization also shuts down without the previous
   `update_evidence` decode errors. Treat this as the current dashboard
   packaging snapshot unless further UI/API contracts change.

3. Native/full-mode boundary decision (closed):
   PyO3 native is the completion path for the full graph/vector backend. Docker
   Helix/full-mode remains an explicit compatibility/integration lane, not a
   blocker for this goal's no-Docker local operator readiness. If a future
   release wants Docker/full as a ship gate, run it as a separate external-service
   acceptance pass instead of mixing it back into the local non-Docker gate.

4. Live AI-harness adoption evidence:
   The MCP prompt, setup guidance, `claim_authority()` protocol,
   `validate_agent_protocol_calls()`, `engram adoption`, stdio MCP-client test,
   `--require-live-evidence`, `--template`, live-evidence metadata schema,
   placeholder rejection, and copied Claude failure regression now cover the
   known file-memory bypass failure mode and prevent a bare handcrafted JSONL
   transcript or untouched template from standing in for live client evidence.
   A later logged-in Claude Code stream-json prompt run against the REST-mounted
   HTTP MCP endpoint passed `engram adoption --require-live-evidence` with
   observed `claim_authority`, `get_context`, `recall`, and `remember`.
   AutoCapture hook generation now writes adoption traces that the verifier can
   merge with Claude stream-json logs, filter by session id, reject when stale
   trace records conflict with the current client session, and assert the
   expected live client label with `--require-client`. The remaining adoption
   hardening is broader client diversity: repeat the same verifier against
   Cursor, Windsurf, or another MCP harness before treating cross-harness
   adoption as release-complete.

5. Evaluation confidence (blocking):
   The report and labels exist, and the report now exposes structured
   `evaluation_signals` for cue usefulness, projection yield, recall quality,
   false recall, triage calibration, and consolidation effect. The projected/
   consolidated smoke verifier fails if those records are missing or unmeasured.
   The normal `engram evaluate --require-evaluation-signals` CLI path applies
   the same hard gate to JSON exports, live lite reports, and native PyO3
   reports outside the smoke harness. `--from-json` also accepts
   already-rendered brain-loop report JSON so saved report artifacts can be
   verified directly, and `--min-evaluation-signal-evidence N` can require more
   than one smoke-sized evidence record per measured signal for benchmark or
   release gates. `--benchmark-artifact results.json
   --require-benchmark-evidence` can also attach a deterministic showcase
   benchmark artifact and fail if the `engram_full` baseline lacks enough
   scenarios, pass rate, fairness contract, or transcript hashes. Markdown
   reports render this Benchmark Evidence section for operator review.
   `--human-label-artifact human-labels.json
   --require-human-label-evidence` adds a separate release/staging gate for
   real human-reviewed harness data. `--human-label-template` prints the
   artifact schema, starter examples, and validation command. The filled
   artifact must declare `humanLabeled: true`, a non-synthetic source, client
   label, capture time, and labeler, and it fails if an untouched placeholder
   template or smoke, benchmark, showcase, fixture, deterministic, simulated, or
   synthetic data is presented as production evidence. Loaded human-label
   artifacts include a SHA-256 digest in the evidence summary and Markdown
   report so archived bundles can be traced back to the exact reviewed file.
   `--adoption-report adoption-report.json --require-adoption-evidence` attaches
   the matching adoption validation JSON, requires it to be a passed live-client
   report, and cross-checks client/session metadata against the human-label
   artifact when both are present.
   `--evidence-bundle brain-loop-evidence.json` archives the report, attached
   benchmark/human-label/adoption evidence, source paths, and gate thresholds as
   one reproducible JSON artifact after requested gates pass. The
   native surface manifest tracks the Helix variant as an operator hard gate.
   `engram doctor` now surfaces the same
   evaluation-signal readiness in its disposable smoke report, so the local
   diagnostic path shows whether the six operator signals are measured instead
   of only reporting smoke totals and coverage-gap counts.
   The evidence-bundle path has been exercised end to end with disposable lite
   smoke plus deterministic showcase artifacts: first a one-scenario packaging
   proof at `/private/tmp/engram-brain-loop-evidence.json`, then a four-scenario
   quick-mode gate at
   `/private/tmp/engram-brain-loop-evidence-quick-20260518.json`, and now a
   full deterministic 39-scenario gate at
   `/private/tmp/engram-brain-loop-evidence-full-20260518.json` requiring pass
   rate `1.0` for `engram_full`.
   Production completion still needs an actual release/staging
   `human-labels.json` artifact collected from a real harness run if the bar
   includes human-labeled recall sessions rather than deterministic showcase
   scenarios. The gate mechanics now exist; the real artifact has not been
   collected in this repo.

6. Completion packaging:
   The previous dirty milestone scope has been split into committed and pushed
   slices. Recent adoption commits include generated hook trace coverage,
   hook-installer verifier guidance, and template validation commands for both
   single live-wrapper transcripts and Claude stream-json plus AutoCapture
   traces. `/private/tmp` evidence artifacts remain local-only and should not be
   added to git. The current broad backend gate recorded for this audit remains
   `3320 passed, 43 skipped, 236 deselected`; focused adoption/evaluation tests
   and `git diff --check` were run for the later adoption-template and
   hook-trace slices. Before any future packaging, run `git status --short
   --branch` and stage only the next intentional scope.

## Next Concrete Work

The route-orchestration and facade-boundary audit is now guarded enough to stop
treating it as the main completion blocker. The next goal-critical work is:

1. If another live harness is available, run the same adoption verifier against
   Cursor, Windsurf, or a second MCP client using `claim_authority()` and
   `engram adoption --require-live-evidence --require-client <client>`.
   Capture whether the client bootstraps empty Engram state, recalls before
   answering, routes durable cross-context facts into Engram, and treats
   project-local files as scratch/conventions instead of a reason to bypass
   Engram.
2. If the project wants a release gate beyond the local deterministic milestone,
   collect a real `human-labels.json` artifact from a staging or production
   harness plus the matching `engram adoption --format json` report, then run
   `engram evaluate --require-human-label-evidence
   --require-adoption-evidence` against both.
   Start with `engram evaluate --human-label-template` so the artifact follows
   the expected schema.
   The current local benchmark bundle is
   `/private/tmp/engram-brain-loop-evidence-full-20260518.json` and already
   satisfies the deterministic 39-scenario milestone.
3. Keep the repo packaging current: after each small slice, update
   `docs/CURRENT_HANDOFF.md` and this audit only when the real completion state
   changes, then commit/push the bounded scope.
