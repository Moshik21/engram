# Brain Runtime Audit

Date: 2026-05-11

This audit anchors the long-running goal of turning Engram into a coherent
"one brain per person" memory runtime for AI agents.

The target product contract is:

```text
Capture -> Cue -> Project -> Recall -> Consolidate
```

## Current Read

The core loop exists, but it is spread across `GraphManager`, REST handlers,
MCP tools, the worker, consolidation phases, retrieval formatting, and dashboard
state. The first integration pass moved several contracts back into sync:
granular episode status, projection state, cue summaries, cue-backed recall,
16-phase consolidation, quest-mode dashboard compilation, and Helix search
support.

The next risk is architectural drift. Most behavior is reachable, but the
runtime does not yet make the lifecycle explicit as a shared service boundary.

The first P0 recall-contract slice is now in place. REST recall, MCP recall,
and knowledge-chat recall use `engram.retrieval.presenter` to turn
`GraphManager.recall()` dictionaries into one shared semantic contract, then
surface-specific API, MCP, or chat payloads.

The second P0 public-contract slice is also in place. REST and MCP
observe/remember responses use `engram.ingestion.presenter` to share capture,
cue, projection-mode, projection-status, and adjudication semantics while
preserving legacy surface keys such as `episodeId` and `episode_id`.

The first P1 runtime-service extraction is now in place. `GraphManager` remains
the compatibility facade, but `GraphManager.store_episode()` delegates raw
episode creation, queued events, deterministic cue generation/indexing, initial
projection-state updates, and decision-graph capture side effects to
`engram.ingestion.capture_service.EpisodeCaptureService`.

One-shot episode ingestion now has its own boundary as well.
`GraphManager.ingest_episode()` remains the REST/MCP/benchmark compatibility
API, but `engram.ingestion.episode_ingestion.EpisodeIngestionService` owns the
store-then-project workflow, proposed entity/relationship forwarding,
attachment and conversation-date forwarding, and the legacy behavior that
returns the captured episode ID even when projection fails after recording
failure state.

Offline capture replay now has a Capture-stage service boundary too.
`engram.ingestion.offline_replay.OfflineReplayService` owns queue draining,
short-content and dedup skips, active `group_id` store calls, failed-entry
accounting, and replay counts. The route-facing
`build_api_offline_replay_surface()` helper now owns service construction and
the `status: replayed` REST acknowledgement payload, so
`/api/knowledge/replay-queue` keeps only tenant lookup, manager dependency
lookup, and JSON wrapping.

Capture deduplication is now explicit as well. The API module keeps
`_DEDUP_CACHE` and `_dedup_check()` as compatibility handles for tests and
monkeypatches, while `engram.ingestion.dedup.CaptureDedupCache` owns content
hashing, TTL duplicate detection, stale eviction, and max-entry cleanup for
auto-observe and replay.

The second P1 runtime slice is now in place. Episode projection-state updates
and cue projection metadata for the touched manager, worker, triage, replay, and
capture paths route through `engram.ingestion.projection_state.sync_projection_state`.
The cue-feedback promotion path in `GraphManager` now uses that same helper when
hot cue recall schedules an episode for projection, keeping episode status,
episode projection state, and cue projection metadata in one shared update
contract.

The follow-up projection-state audit tightened two remaining production runtime
paths. Worker system-discourse skips and project-bootstrap artifact suppression
now route through `sync_projection_state()` as well. The remaining direct
`update_episode()` calls inspected in production code write emotional encoding
context, semantic tier counters, or storage-adapter serialization fields rather
than episode projection state.

The third P1 runtime slice is now in place. `GraphManager.project_episode()` is
now a compatibility facade over
`engram.ingestion.projection_service.EpisodeProjectionService`, which owns the
Project stage orchestration for duplicate/system-discourse skips, projection
planning, evidence or legacy extraction, graph apply/index work, completion
events, and retry/dead-letter handling.

The fourth P1 runtime slice has started tightening that service boundary.
`engram.ingestion.projection_execution.LegacyProjectionExecutor` now owns the
legacy extractor -> entity apply -> bootstrap edge -> relationship write path
and returns a typed execution outcome to the projection service.

The fifth P1 runtime slice continued that tightening.
`engram.ingestion.projection_execution.EvidenceProjectionExecutor` now owns
evidence bundle construction, optional edge-adjudication request storage,
commit/defer decisions, committed evidence materialization, and
deferred/committed evidence persistence. `EpisodeProjectionService` keeps the
Project lifecycle orchestration and delegates both concrete execution paths.

The sixth P1 runtime slice added the Project-stage lifecycle result contract.
`engram.ingestion.projection_execution.ProjectionLifecycleResult` is returned
by `EpisodeProjectionService.project_episode()` and the `GraphManager` facade
for projected and skipped outcomes. It preserves event-safe fields such as
status, outcome, projection state, counts, duration, reason, retry count, and
execution path while existing callers may still ignore the return value.

The first Consolidate-stage boundary is now in place.
`engram.consolidation.lifecycle` builds a typed `ConsolidationCyclePlan` and
normalizes phase/cycle lifecycle event payloads through
`ConsolidationPhaseLifecycleResult` and `ConsolidationCycleLifecycleResult`.
This established selected phase planning and event shape as a shared contract
before later slices extracted phase execution and event publication.

The next Consolidate-stage boundary is now in place.
`engram.consolidation.phase_runner.ConsolidationPhaseRunner` owns one phase
execution, direct audit-record persistence, newly appended decision-trace/label
persistence, and merge/prune removed-node discovery. `ConsolidationEngine` still
owns the cycle loop, cancellation, capability validation, non-fatal phase
failure handling, and post-cycle learning.

The following Consolidate-stage boundary moved event publication out of the
engine. `engram.consolidation.events.ConsolidationEventPublisher` now owns
cycle start/end, phase start/complete/fail, graph delta, and learning update
events. It uses the lifecycle result contracts for cycle and phase payloads
while preserving existing event names and legacy keys.

The latest Consolidate-stage boundary moved post-cycle learning out of the
engine. `engram.consolidation.learning.ConsolidationLearningService` now owns
distillation example generation/persistence, calibration history collection,
calibration snapshot generation/persistence, and artifact counts returned to the
event publisher.

The following Consolidate-stage boundary moved successful post-cycle
finalization out of the engine. `engram.consolidation.finalization`
now owns pinned context refresh after completed consolidation cycles. The engine
still owns the cycle loop, cancellation, selected phase ordering, and when
successful finalization is allowed to run.

The next Consolidate-stage boundary moved runtime capability preflight out of
the engine. `engram.consolidation.capabilities.ConsolidationCapabilityValidator`
now owns selected-phase required-method checks for graph, activation, and search
stores while preserving the same missing-method failure contract for cycles.

The first P2 dashboard lifecycle slice is now in place.
`dashboard/src/components/LifecyclePanel.tsx` makes Brain Loop the default
observatory view and maps existing stats, episodes, cue summaries, recall
results, activation, scheduler state, and consolidation cycles into
Capture -> Cue -> Project -> Recall -> Consolidate.

The next P2 slice added the first backend summary contract for that surface.
`server/engram/api/lifecycle.py` exposes `GET /api/lifecycle/summary`, returning
semantic stage health, totals, recent episodes, active recall context, pressure,
and latest consolidation-cycle state for the full loop. The dashboard now loads
that contract through `dashboard/src/store/lifecycleSlice.ts`; WebSocket episode,
graph, activation snapshot, and consolidation events refresh the summary so the
Brain Loop is no longer only derived from disconnected client collections.

That lifecycle summary is now a shared runtime contract instead of a REST-only
dashboard endpoint. `server/engram/lifecycle_summary.py` builds the
Capture/Cue/Project/Recall/Consolidate summary, REST delegates to it, and MCP
exposes `get_lifecycle_summary` so headless clients can see the same brain-loop
state as the dashboard.

The Consolidate stage inside that summary now reuses the shared consolidation
presenter. `consolidate.latestCycle` carries the same cycle-level `error`,
phase-level `error`, and phase duration fields as REST consolidation
status/history/detail, MCP, and CLI responses instead of maintaining a duplicate
serializer in the lifecycle builder.

Shared lifecycle Consolidate health now treats phase-level failures as
attention signals even when the containing cycle is recorded as `completed`.
The backend summary checks phase `status: error` and non-empty phase `error`,
and the Brain Loop dashboard, headless lifecycle Markdown, and doctor Markdown
display the first phase issue so non-fatal consolidation failures do not look
ready.

`engram doctor` also treats lifecycle attention as diagnostic signal now. A
loaded snapshot with any stage in `attention` marks the `lifecycle_snapshot`
check as `warn` and carries stage statuses plus the Consolidate issue in
metadata, instead of reporting a clean pass just because the snapshot read
succeeded.

The direct consolidation CLI also distinguishes successful cycles with warnings
from clean cycles. A cycle recorded as `completed` still exits successfully, but
if any phase has an error the human summary says `completed with warnings` and
stderr names the first phase issue.
That warning label now lives in the shared consolidation presenter, so MCP
`trigger_consolidation` reports `cycle with warnings` for the same completed
phase-error case.
The same presenter now serializes `phase_issue` on cycle summaries, giving
REST, MCP, lifecycle, doctor, CLI, and dashboard consumers a direct first
phase-level issue field while preserving the full phase result list.
The P3 brain-loop evaluation report consumes the same field for
`latest_cycle.phase_issue`, and the dashboard Evaluate panel shows it in the
Consolidate latest-issue row when there is no top-level cycle error.
The dedicated dashboard Consolidation panel also shows `phase_issue` in the
cycle list, so completed-with-warning cycles are visible before opening cycle
detail. Its phase timeline now renders phase `error` text under the phase name,
so detail view explains the warning without requiring raw JSON inspection.
The typed `consolidation.completed` lifecycle event payload now includes the
same `phase_issue`, keeping WebSocket/event consumers aligned with REST, MCP,
CLI, lifecycle, doctor, and dashboard reads.
Dashboard quest-mode WebSocket handling consumes that field as well: a completed
cycle with `phase_issue` now records a warning quest event instead of a clean
completion celebration.

The brain-loop evaluation report now exposes cycle-level consolidation error
text as well. Its `consolidate.latest_cycle.error` field complements the
existing phase `errors` list and `error_count`, so P3 reports no longer count a
failed cycle without showing the top-level failure reason.

The Markdown renderer for that report now prints the same latest cycle error on
the Consolidate summary line, keeping CLI/operator output aligned with the JSON
contract.

The dashboard Evaluation panel now consumes that field. Its Consolidate card
shows a compact `latest error` diagnostic when `latest_cycle.error` is present,
and the dashboard API client tests preserve the raw field through evaluation
report mapping.

The primary Brain Loop dashboard now consumes it too. The Lifecycle panel's
Consolidate stage appends the latest cycle error to the stage summary when
present, using wrapping text so longer operator failures stay inside the card.

The headless lifecycle Markdown snapshot now consumes it as well.
`engram lifecycle --format markdown` prints the latest consolidation cycle error
on the Consolidate line when one is present, matching lifecycle JSON and the
dashboard Brain Loop.

The Recall stage in that lifecycle summary now includes prospective-memory
state. Active intentions, refresh-context intention count, after-consolidation
pinned-context count, cached pinned results, needs-refresh count, and latest
refresh timestamp are reported from the same `list_intentions` source that REST
and MCP use. The CLI Markdown renderer and dashboard Recall stage both show
active intention/pinned context counts beside the rest of Recall state.

The summary also has a headless CLI surface. `engram lifecycle` resolves the
configured engine mode, including Helix native PyO3, and prints the same shared
lifecycle contract as Markdown or JSON without requiring REST, MCP, or the
dashboard to be running.

`engram doctor` now embeds that same lifecycle summary in its diagnostics. The
doctor report includes a `lifecycle_snapshot` check plus `lifecycle_summary`
output by default, and `--no-lifecycle` keeps the old config-only behavior when
an operator wants to avoid opening the local brain DB.

The external agent/client contract now points at the same lifecycle surface.
README lists `/api/lifecycle/summary`, MCP `get_lifecycle_summary`, and the
evaluation endpoints, while the OpenClaw/Engram skill tells agents when to
inspect the lifecycle snapshot before heavier diagnostics. The same pass
removed old phase/tool-count wording from the public install and skill docs.

The following P2 slice made the Brain Loop cards navigational. Capture opens the
Feed drilldown, Cue and Project open Stats, Recall opens Knowledge, and
Consolidate opens the Consolidation drilldown while selecting the latest known
cycle when available. This keeps the lifecycle surface as the primary map while
preserving technical panels as drilldowns.

The next P2 slice added target-panel context for the first drilldowns.
`lifecycleDrilldownStage` is stored in dashboard preferences state, ordinary
view navigation clears it, and Brain Loop card navigation sets it after opening
the target view. Feed applies Capture context as an active-capture status
filter. Stats applies Cue or Project context by focusing the matching section
and scrolling it into view when needed.

The final P2 drilldown-context slice added Recall context inside Knowledge.
When the Recall card opens Knowledge, the memory pulse strip becomes a focused
`Recall Context` surface and remains visible even if no active entities are
currently loaded. This gives every Brain Loop stage a concrete drilldown target
without turning quest mode or lower-level panels into the primary explanation.

The first P3 dashboard evaluation drilldown is now in place.
`dashboard/src/components/EvaluationPanel.tsx` renders the local brain-loop
report for Cue, Project, Recall, Continuity, and Consolidate, backed by
`dashboard/src/store/evaluationSlice.ts` and
`GET /api/evaluation/brain-loop/report`. The sidebar exposes it as Evaluate,
and lifecycle-affecting WebSocket events refresh the report.

The cue-usefulness part of that report is now more operator-visible. The
Markdown brain-loop report includes cue-to-projection conversion, and the
Evaluate drilldown shows surfaced cues, selected rate, used rate, projection
conversion, and near-miss rate from the shared evaluation contract.

Consolidation calibration quality is visible in the same first-pass evaluation
surface. The Consolidate card now shows top-phase calibration accuracy and
expected calibration error alongside cycle, affected-item, error, and snapshot
counts.

The headless Markdown report now carries the same calibration signal. Its
Consolidate section appends the top labeled calibration phase's accuracy and
expected calibration error when snapshots are present.

The next P3 slice brought MCP into the same evaluation loop. `engram mcp` now
initializes `SQLiteEvaluationStore`, exposes `record_recall_evaluation`,
`record_session_continuity_evaluation`, and `get_evaluation_report`, and shares
REST/MCP write acknowledgement semantics through
`engram.evaluation.presenter`.

The live P3 evaluation smoke now passes with the lite backend and dashboard
connected. A seeded observed episode plus recall/session-continuity labels
flowed through the REST report endpoint and rendered in the Evaluate drilldown
with measured recall and continuity signals.

The operator-facing P3 label path is now in place. The Evaluate drilldown can
write recall and session-continuity labels through the same REST endpoints used
by direct API callers, and successful writes refresh the local brain-loop
report.

The next P3 signal slice added missed-recall measurement. Recall labels now
carry optional `recall_needed` / `recallNeeded` state through the SQLite
evaluation store, REST, MCP, the shared presenter, CLI/report builder, smoke
labels, and the dashboard Evaluate form. Reports expose memory-need recall,
missed-recall rate, need-label counts, needed count, and missed count so Engram
can measure recall omissions, not only precision for triggered recall.

The evaluation report now has an explicit signal-readiness contract for the
goal's production-quality checks. `server/engram/evaluation/brain_loop_report.py`
returns `evaluation_signals` for cue usefulness, projection yield, recall
quality, false recall, triage calibration, and consolidation effect. Each signal
reports status, evidence count, current metric, and a concrete gap when it is not
measured. The Markdown report prints the same readiness map, REST exposes it
through `/api/evaluation/brain-loop/report`, MCP returns it through
`get_evaluation_report`, and the dashboard Evaluate panel renders it as Signal
Readiness so operators can see which evidence is still missing instead of
inferring from raw metric cards. Fast MCP tests and populated native PyO3 MCP
parity now assert the six-signal map is measured when the report has matching
cue/project/recall/calibration/consolidation evidence.
The projected/consolidated smoke verifier now treats those readiness records as
real gates too: `assert_smoke_report()` fails if any required signal is missing,
unmeasured, evidence-free, or metric-free. This keeps smoke success tied to the
same six evaluation signals named by the goal.
The normal evaluation CLI can now enforce the same readiness contract outside
the deterministic smoke harness. `engram evaluate --require-evaluation-signals`
exits non-zero when a JSON export, live lite report, or native PyO3 report is
missing any required measured signal, evidence count, or metric. `--from-json`
accepts raw stats/sample exports and already-rendered brain-loop report JSON, so
saved benchmark or native report artifacts can be verified directly.
For release/benchmark gates, `--min-evaluation-signal-evidence N` raises the
threshold so one smoke-sized evidence record per signal cannot satisfy the
hard gate.
The same command can now attach and gate a deterministic showcase benchmark
artifact with `--benchmark-artifact results.json --require-benchmark-evidence`.
That evidence check verifies the `engram_full` baseline summary, scenario count,
pass-rate threshold, fairness contract, and transcript hashes before treating a
saved report as benchmark-backed. The Markdown report renders the attached
benchmark evidence too, so the operator artifact is readable without inspecting
raw JSON. The CLI now also has a separate real harness gate:
`--human-label-artifact human-labels.json --require-human-label-evidence`.
`--human-label-template` prints the required artifact shape, starter examples,
and validation command. When an adoption report already exists,
`--human-label-template --adoption-report adoption-report.json` pre-fills the
client, capture timestamp, and session metadata that the release gate later
cross-checks, and it now rejects failed or non-live-gated adoption reports
before generating that label-collection template. The gate requires explicit
`humanLabeled: true`
metadata, a non-synthetic source, client label, capture timestamp, and human
reviewer, and it rejects untouched placeholder templates plus smoke, benchmark,
showcase, fixture, deterministic, simulated, or synthetic sources as release
evidence. Human-label artifacts loaded from disk are summarized with a SHA-256
digest, so Markdown reports and evidence bundles can be tied to the exact
reviewed file. `--adoption-report adoption-report.json
--require-adoption-evidence` attaches the matching `engram adoption --format
json` report, requires the adoption report to pass with live-client metadata,
and cross-checks client/session metadata against human-label evidence when both
are attached. `--require-release-evidence` composes measured evaluation signals,
human-label evidence, and adoption evidence into one operator gate.
`--evidence-bundle` writes the same report plus source paths, benchmark
evidence, human-label evidence, adoption evidence, and gate thresholds as a
single archiveable JSON artifact.
The native surface manifest also tracks
`engram evaluate --mode helix --require-evaluation-signals` as an operator hard
gate beside the native smoke and doctor commands.
The no-bind native dashboard smoke fixture now mirrors that contract. Its
native-shaped evaluation payload includes measured `evaluation_signals`, cue
feedback evidence, calibration quality, and consolidation effect data; the smoke
asserts the dashboard Signal Readiness panel renders `6/6 measured`.

The projected/consolidated P3 smoke is now repeatable and first-class.
`engram evaluate --smoke` boots local stores against disposable storage, seeds
queued episodes, runs the real triage consolidation phase, stores
recall/session-continuity labels, builds the shared brain-loop report, and
fails if projection yield, consolidation cycle, or calibration coverage gaps
remain. Bare `--smoke` stays lite for the fallback path, while
`--smoke --mode helix` runs the same smoke against native PyO3 Helix without
Docker. The reusable implementation lives in
`server/engram/evaluation/smoke.py`, and
`server/scripts/projected_consolidated_smoke.py` remains a compatibility wrapper.

The first operator diagnostic command is now in place. `engram doctor` loads
config, checks the configured SQLite path, resolves runtime mode, optionally
checks the REST `/health` endpoint, and runs the disposable
`Capture -> Cue -> Project -> Recall -> Consolidate` smoke. When diagnostics
resolve to Helix, the smoke now runs against native PyO3 Helix and reports
`mode: helix`; bare smoke remains the lite fallback.

Native CLI inspection can now target an explicit PyO3 data directory.
`engram lifecycle --mode helix --helix-data-dir ...` and
`engram doctor --mode helix --helix-data-dir ...` both inspect that native brain
directory for the lifecycle snapshot. Doctor still runs its
projected/consolidated smoke against disposable native storage so diagnostics do
not mutate the inspected brain.

Runtime startup now accepts the same explicit native data-directory override.
`engram serve --mode helix --helix-data-dir ...` and
`engram mcp --mode helix --helix-data-dir ...` set native Helix transport plus
`ENGRAM_HELIX__DATA_DIR` before booting, keeping REST, MCP, lifecycle, doctor,
and evaluate aligned around the preferred no-Docker PyO3 path.
The Makefile native shortcuts now delegate to those same commands when
`NATIVE_DATA_DIR=...` is supplied.

The preferred local operator path is now Helix native PyO3. Lite still matters
as the disposable smoke/demo fallback, but local commands should assume
`ENGRAM_MODE=helix` plus `ENGRAM_HELIX__TRANSPORT=native` is the main way to get
full HelixDB graph/vector/BM25 behavior without Docker.

That native-first stance now has a static public-surface manifest.
`engram.quality.native_surface_manifest` classifies every current REST route,
MCP tool, MCP resource, MCP prompt, dashboard native smoke, and operator native
smoke against PyO3-native parity evidence. The static tests compare the
manifest to the actual FastAPI route table and MCP decorators so new public
surfaces cannot appear without an explicit native-path classification. The
manifest verifier also checks that native runtime evidence names actual parity
helpers/tests and that dashboard/operator/static evidence points at existing
repo artifacts.

A separate completion-readiness audit now exists at
`docs/design/brain-runtime-completion-audit.md`. Its current verdict is not
complete: the remaining blockers are stronger evaluation confidence, live
AI-harness adoption evidence, and completion packaging for the dirty worktree.
The final facade audit now has core delegate coverage plus a whole-runtime scan
against direct private `GraphManager` field access. The Docker/full-mode boundary
is now decided: PyO3 native is the full-backend completion path, while Docker
Helix/full-mode remains a separate compatibility/integration lane.
The default dashboard verification part of that list has been refreshed again
after the current REST/MCP adoption and route-boundary work: `pnpm test -- --run`
passes with 214 tests and 1 skipped live-native test, and `pnpm build` passes
with the existing large-chunk warning. The bind-dependent live native dashboard
smoke now passes too when REST is started with both `ENGRAM_DEFAULT_GROUP_ID`
and `ENGRAM_AUTH__DEFAULT_GROUP_ID` set to `native_brain`. Starting the same
seeded native brain without those defaults produced the expected failure shape:
dashboard smoke reached REST but saw `groupId=default` instead of
`native_brain`. The later default-group config slice makes top-level
`ENGRAM_DEFAULT_GROUP_ID` flow into the auth fallback when the explicit auth
override is omitted.

The local evaluation report now follows the resolved runtime mode as well.
`engram evaluate --mode helix` reads graph stats and consolidation cycles from
Helix while retaining SQLite for local evaluation labels. Native smoke now
uses `engram evaluate --smoke --mode helix` to verify a populated PyO3 brain,
and bare `engram evaluate --smoke` remains the lite fallback. Helix graph stats
now count episodes, cue coverage, projection attempts, and projected episode
entity yield so native Brain Loop and evaluation snapshots are not silently
empty.

Projection freshness is now a measured P3 signal on the same native path.
Helix graph stats compute average capture-to-projection lag from episode
`created_at` and `last_projected_at` timestamps, and the shared Markdown report
plus dashboard Evaluate card show projection latency and processing duration
beside Project-stage yield and failure rate.

Projection backlog pressure is now measured in that same Project-stage report.
The shared evaluation contract derives tracked count, projected rate, and
backlog rate from projection state counts, so native/lite reports can show
whether Capture/Cue is outpacing Project instead of only showing eventual
projection totals.

Recall gate latency is now measured in the Recall-stage report. The shared
evaluation contract carries recall-need analyzer and graph-probe average/p95
latency from the existing runtime controller metrics, the Markdown report
prints analyzer/probe p95, and the dashboard Evaluate card surfaces both p95
timings beside labeled recall quality.

Recall gate control state is now measured in the same Recall-stage report. The
shared evaluation contract carries runtime surfaced/used/dismissed interaction
counts, selected/confirmed/corrected counts, graph override count,
adaptive-threshold enabled state, and active recall thresholds. Markdown prints
the control summary, and the dashboard Evaluate view surfaces the posture in a
Recall Gate card.

The projected/consolidated evaluation smoke now exercises those Recall gate
signals directly. It runs one real memory-need analysis plus surfaced recall
interaction before building the report from `GraphManager.get_graph_state()`,
so the smoke proves gate latency/control fields are runtime-populated instead
of only fixture-populated.

That smoke path has been checked against the preferred Helix native PyO3 mode
as well. The disposable native run for `native_brain` produced one trigger
analysis, analyzer p95 latency, keyword trigger-family contribution, surfaced
runtime feedback, and no coverage gaps.

The smoke verifier now treats those Recall gate fields as required coverage.
`assert_smoke_report()` fails if gate analysis, a trigger, analyzer latency,
surfaced recall feedback, or the `gate_recall_checks` counter is missing, and
the smoke footer prints that gate-check count.

The evaluation coverage gaps now distinguish labeled recall samples from actual
runtime Recall Gate coverage. A report with recall labels but no runtime gate
analyses surfaces `recall gate needs runtime analyses`; a report with gate
analysis but no analyzer latency surfaces `recall gate latency needs analyzer
samples`.

The latest runtime Recall Gate metrics snapshot is now persisted through the
local evaluation store and merged into CLI, REST, and MCP evaluation reports
only when current in-memory stats are weaker. The in-process native PyO3 smoke
still proves runtime gate fields are populated, and reopened native live reports
now recover analyzer latency, trigger counts, and surfaced feedback from that
snapshot instead of losing runtime coverage after process reopen.

The no-bind native dashboard smoke fixture now carries that same Recall gate
payload shape. The dashboard API client verifies native-shaped analyzer latency,
trigger count, surfaced recall feedback, and threshold mapping without needing
an approved local REST bind. The fixture lane also renders the Evaluation panel
from that native-shaped report and checks the Recall Gate card.

Consolidation effect rate is now measured in the Consolidate-stage report. The
shared evaluation contract derives affected/processed rates overall and per
phase, the Markdown report prints the cycle effect percentage, and the dashboard
Evaluate card surfaces the signal beside cycle count, affected count,
calibration accuracy, and ECE.

Adjudication pressure is now measured inside the same Consolidate-stage report.
The shared evaluation contract summarizes evidence/edge adjudication phase
runs, affected/unaffected counts, effect rate, and errors so operators can see
whether ambiguous evidence is being resolved rather than hidden inside aggregate
consolidation effect.

Live open adjudication work is now measured from storage stats too. SQLite/lite
and PyO3 native Helix emit `adjudication_metrics` with open evidence/request
counts by status, and the shared brain-loop report marks Consolidate
`attention` when unresolved evidence or edge adjudication requests are waiting
even before a phase run records an error or low effect rate.

Evidence/adjudication materialization now has its own service boundary.
`engram.ingestion.adjudication_service.EvidenceAdjudicationService` owns open
adjudication presentation, evidence-id materialization mapping, stored evidence
materialization for consolidation, evidence storage-row serialization, and
client/server adjudication resolution. `GraphManager` stays as the compatibility
facade used by REST, MCP, projection execution, and consolidation phases, but no
longer carries that Project-to-Consolidate contract directly.

Post-cycle finalization now participates in the Consolidate lifecycle event
contract. The pinned-context refresh finalizer already ran after successful
cycles, but `ConsolidationEngine` discarded the result. Completed-cycle events
now include `finalization.refreshedPinnedContexts`, so dashboard or WebSocket
consumers can distinguish a cycle that merely ran phases from one that also
refreshed pinned context.

MCP evaluation reports now use the same consolidation source. The MCP runtime
opens a consolidation audit store for the resolved mode, so
`get_evaluation_report` includes Helix native cycles instead of only seeing
cycle context when the graph store happens to expose a SQLite `_db`.

MCP lifecycle reports now use that consolidation source too.
`get_lifecycle_summary` passes the active MCP consolidation store into the
shared lifecycle builder, so Helix native cycle counts and latest-cycle details
show up in headless lifecycle snapshots.

MCP consolidation controls now have native coverage too. The populated native
surface test calls `get_consolidation_status` and dry-run
`trigger_consolidation` against the active PyO3 graph, verifying completed
latest-cycle status, completed dry-run status, populated graph stats, phase
results, summary totals, and that the triggered dry-run becomes the latest
status cycle through the active MCP consolidation store.

The MCP graph-stats resource now has native coverage as its own headless
surface. The populated native test calls `graph_stats_resource()` separately
from the graph-state tool and verifies active PyO3 episode/entity counts,
TestMemory type counts, cue coverage, projected cue counts, projection state
counts, and projection yield.

MCP evaluation-label writes now have native coverage. The populated native test
records recall-quality and session-continuity samples through
`record_recall_evaluation` and `record_session_continuity_evaluation`, then
requires the MCP evaluation report to see the extra native-brain samples.

REST dashboard read surfaces now have native coverage. The populated native test
verifies `/api/stats`, `/api/activation/snapshot`,
`/api/activation/{entity_id}/curve`, `/api/graph/neighborhood`,
`/api/graph/at`, and `/api/episodes` against the active PyO3 brain, including
graph counts, top-connected entities, growth timeline, activation curve data,
graph neighborhood/temporal edges, and cue-bearing episode listing.

Native episode listing now covers status filtering and cursor pagination as
well. The populated REST test filters the observed cue-bearing episode with
`status=queued`, verifies the endpoint returns only queued episodes, and checks
the episode projection state matches the serialized cue projection state while
leaving the cue policy free to route that cue as `cued`, `scheduled`, or
`queued`. It also reads two one-item pages and verifies the cursor advances to a
different episode.

Native atlas persistence now matches the Helix schema. `HelixAtlasStore` writes
`region_label` and deletes old materialized snapshot children through available
find + hard-delete queries instead of non-existent bulk-delete query names. The
populated native REST test covers atlas refresh, history, snapshot lookup, and
region drilldown against PyO3 Helix, plus a same-snapshot-ID upsert check that
proves stale region/member children are removed.

The dashboard WebSocket now has native group-scope coverage. A PyO3 native test
opens `/ws/dashboard`, verifies ping/pong, forwards native-brain event-bus
messages, ignores wrong-group events, and checks resync stays scoped to
`native_brain`.
The lite WebSocket contract is aligned with that behavior as well: the endpoint
docstring now says it subscribes to the authenticated tenant group's event bus,
and a non-default `tenant_brain` regression proves regular TestClient sockets
do not receive raw `default` group events.

OIDC tenant fallback now follows the same configured-brain contract. Missing
OIDC group claims fall back to `AuthConfig.default_group_id` through
`OIDCValidator.validate_token()` and middleware instead of producing a literal
`default` group when the operator configured a non-default local brain.

Helix cue usefulness is no longer a stats-only placeholder. Native
`EpisodeCue` nodes now persist the same cue metadata and feedback counters used
by SQLite/Falkor, and `HelixGraphStore.get_stats()` aggregates cue hits,
surfaced/selected/used/near-miss counts, policy score, and projection attempts
from those nodes. This keeps the preferred PyO3 path from reporting cue
coverage while silently zeroing cue usefulness. A native no-Docker doctor pass
initialized the updated schema successfully, and the rebuilt PyO3 native engine
now exposes 171 routes after adding key-based cue updates, graph-embedding
vector deletion, and open-status evidence/adjudication queries. The
projected/consolidated smoke now records a surfaced cue feedback check before
projection, and brain-loop reports flag `cue usefulness needs surfaced cue
feedback` when cue rows exist but no cue interaction feedback has been
persisted.
The cue contract now also has a static schema-drift guard: the server Helix
schema, native `helixdb-cfg/db/schema.hx`, and generated PyO3 query source must
carry the same `EpisodeCue` fields plus the key-based cue feedback update route.
The same guard now covers `Entity` provenance fields, so PyO3 native preserves
`source_episode_ids`, evidence counts, and evidence span bounds instead of
silently dropping projected lineage metadata.

EventBus hook scheduling now avoids creating fire-and-forget coroutine tasks
when `publish()` is called from sync contexts without a running event loop. This
keeps TestClient and CLI publishes quiet while preserving async runtime hooks.

The remaining native parity UTC deprecation warnings are now gone. MCP session
activity timestamps and GraphManager freshness labels use the shared
`utc_now()` helper, preserving Engram's naive-UTC storage convention without
calling deprecated `datetime.utcnow()`.

Native runtime ownership is now explicit. Helix-backed stores distinguish owned
and borrowed shared `HelixClient` instances, the factory gives the graph store
ownership of the shared native client, and MCP lifespan shutdown stops worker
resources, removes/closes the Redis publisher, and closes evaluation,
consolidation, search, activation, and graph stores. This matters for the
preferred PyO3 path because the in-process native engine should be released by
the runtime that created it rather than leaked behind CLI, REST, or MCP exits.
The native engine itself is cached process-wide because the PyO3 extension
currently keeps the LMDB environment process-owned; repeated same-process opens
reuse the cached engine and process exit releases it.

Native evaluation smoke now covers a populated PyO3 brain. The smoke writes
three episodes, projects all three through triage, persists a consolidation
cycle and calibration snapshot, records recall/continuity labels, and proves
that reopening the same native data directory through `engram evaluate --mode
helix` preserves linked-entity projection yield for a non-default `group_id`.
The smoke also reapplies its deterministic activation overrides after profile
post-init so the standard/rework profile cannot silently reduce
`triage_extract_ratio` or re-enable background runtime features during the
operator gate.

The populated native brain now reaches public REST and MCP surfaces as well.
`tests/test_native_surface_parity.py` seeds PyO3 Helix through the canonical
smoke, reopens that same native data directory through FastAPI, checks
`/api/lifecycle/summary`, `/api/evaluation/brain-loop/report`, and
`/api/knowledge/recall`, then points MCP lifecycle, evaluation, and recall tools
at the same runtime objects. That test also guards the non-default `group_id`
path by keeping `native_brain` as the active group. Recall now passes the active
group into episode-linked entity lookups so native episode context does not fall
back to `default` during recall formatting.

The same native REST pass now checks `/health` before the deeper lifecycle
surfaces. It verifies the reopened PyO3 runtime reports `status=healthy`,
`mode=helix`, and a healthy graph store, giving the preferred no-Docker path a
basic operator readiness contract before recall, evaluation, and mutation
surfaces run.

The REST admin benchmark loader now has bounded native coverage too. The
surface parity test patches `CorpusGenerator` to a tiny fixture so the
regression path does not load the full benchmark corpus, then calls
`/api/admin/load-benchmark`, verifies the endpoint rewrites stale fixture
`group_id` values to `native_brain`, writes the fixture through the live PyO3
graph store, and reads it back through public entity search and fact lookup.

That native REST/MCP surface test now covers evaluation writes too. It posts a
recall-quality label and a session-continuity label through REST while the PyO3
brain is active, includes stale/wrong group fields in those payloads, verifies
the write acknowledgements use `native_brain`, and then checks the MCP
evaluation report can see the same persisted sample counts.

The populated native REST test now covers consolidation read surfaces too.
`/api/consolidation/status`, `/api/consolidation/history`, and
`/api/consolidation/cycle/{id}` read the completed PyO3 smoke cycle from the
active Helix-backed consolidation store, including phase and detail collections.

REST consolidation trigger now has native coverage too. The populated native
test calls `/api/consolidation/trigger?dry_run=true` after the core lifecycle
assertions, verifies the trigger payload uses `native_brain`, and waits for the
dry-run manual cycle to appear completed in native history.

The populated native REST test now covers notification reads and dismissal too.
It seeds active and wrong-group notifications in the app notification store,
verifies `/api/knowledge/notifications` pending/since reads stay scoped to
`native_brain`, dismisses through `/api/knowledge/notifications/dismiss`, and
checks dismissed notifications leave pending while retaining dismissal metadata
in recent reads.
Notification dismissal itself is now group-scoped: the store-level dismiss
methods accept an optional `group_id`, REST passes the active tenant group, and
the populated native test includes a wrong-group ID in the dismiss request to
prove one brain cannot dismiss another brain's notification.
The dashboard WebSocket `dismiss_notification` command uses the same connected
group now, with a lite WebSocket regression and the native dashboard WebSocket
parity test protecting the route.
The Redis event bridge now preserves the same channel group boundary. The
publisher only forwards events for the configured bridge group, and the
subscriber republishes Redis messages under its subscribed group instead of
trusting a missing or stale serialized `group_id`. Mismatched serialized groups
are dropped so Redis cannot route lifecycle/dashboard events across brains.
The PyO3 Helix graph store's internal-ID caches now honor the same one-brain
boundary. Entity and episode resolver caches key Helix IDs by
`(group_id, external_id)`, and the old bare external-ID caches are retained only
while an external ID maps to one Helix node. Focused no-Helix tests cover
duplicate external entity and episode IDs across groups, including
`get_episode_entities(group_id=...)` using the group-scoped episode cache.
The in-memory activation snapshot path is group-aware now too. When hot
activation state is flushed back to graph entity metadata,
`MemoryActivationStore.snapshot_to_graph()` passes the entity's recorded
`group_id` into `GraphStore.update_entity()` and normalizes epoch
`last_accessed` values to storage-compatible naive UTC datetimes. Ungrouped
demo/test activation still writes to the `default` brain.
The Redis/full activation store now preserves and verifies that same group
boundary around its sorted-set acceleration path. `set_activation()` and
`batch_set()` keep the existing hash `group_id` when consolidation-style writes
replace activation state, `record_access()` refreshes the group sorted-set
index immediately, `clear_activation()` removes the indexed member, and
`get_top_activated(group_id=...)` rechecks each fast-path candidate hash's
current group before returning it. Fake-Redis tests cover the contract without
requiring Docker or a live Redis service.
Episode-entity linking now carries the active brain group through the graph
write path as well. `GraphStore.link_episode_entity()` accepts an optional
`group_id`, Helix uses that group to resolve episode/entity internal IDs through
group-scoped caches, Falkor matches both nodes under the same group, and the
projection apply, replay, and benchmark corpus loaders pass the active episode
group. This removes another native fallback where non-default PyO3 brains could
depend on raw-ID uniqueness or `default` resolution during Capture/Project
linking.
The matching read path now applies the same group boundary. SQLite and Falkor
`get_episode_entities(group_id=...)` filter linked entities by group as well as
episodes, and Helix filters returned entity nodes by group after resolving the
active episode. This keeps projection yield, replay vocab linking, and
consolidation graph methods from counting legacy or malformed cross-group
HAS_ENTITY edges as part of the active brain.
The auth-exempt health probe no longer uses a raw unscoped graph stats call.
`/health` now probes `graph_store.get_stats(group_id=config.default_group_id)`,
which keeps no-auth operational checks aligned with the configured local/native
brain such as `native_brain` rather than silently falling back to raw
`default`.
Helix unscoped entity and episode reads now match the storage protocol too.
When callers omit `group_id`, `HelixGraphStore.find_entities()` uses all-group
Helix queries for name, type, combined name/type, and unfiltered searches, and
`get_episodes()` uses an all-group episode query. Neither path silently queries
the raw `default` group anymore. The grouped recall/runtime paths still pass an
explicit brain group, but this removes another PyO3/native parity trap where a
utility or admin path could see different results on Helix than on SQLite or
FalkorDB.
The same all-group behavior now covers the optional dashboard/analytics reads:
Helix stats, cursor-paginated episode lists, top-connected entities, growth
timeline, and entity type counts use all groups when no `group_id` is supplied.
Grouped calls remain scoped, and stats cue lookup now uses each episode's own
group when aggregating across all groups.
Unscoped Helix graph-neighborhood reads no longer fall back to raw `default`
either. `get_neighbors()` and `get_active_neighbors_with_weights()` resolve an
external entity ID across all groups only when that ID is unique; duplicate
external IDs across brains are treated as ambiguous unless the caller supplies
the active `group_id`.
The same unique-or-ambiguous rule now covers unscoped episode link helpers.
`get_episode_entities()` and `link_episode_entity()` resolve episode/entity
internal IDs across all groups only when the external IDs are unique, so legacy
or utility callers no longer attach/link data by probing the raw `default`
brain.
Native consolidation audit cycle caches are group-scoped now too.
`HelixConsolidationStore.update_cycle()` resolves the target cycle inside the
active brain group instead of using an unscoped cycle-id lookup, and duplicate
cycle IDs across groups are treated as ambiguous in the bare cache. Cleanup now
works from group-scoped cached cycles instead of assuming a known `default`
group.
Helix search fallback now preserves that same brain boundary. If grouped vector
queries return no rows because an older schema/transport lacks filtered vector
routes, entity, episode, and cue search fall back to over-fetched unfiltered
vector reads before Python-side `group_id` filtering. Native PyO3 episode-chunk
recall now skips server-side `Embed()` routes and uses client-side embeddings,
matching the indexing path where native stores explicit vectors.
SQLite/lite hybrid search now follows the same optional-group contract. FTS5
already treated omitted `group_id` as all groups; the vector side now does too
for entity, episode, cue, entity-embedding, graph-embedding, and similarity
reads instead of silently narrowing unscoped calls to the raw `default` group.
Redis/full vector embedding lookups now follow that contract as well.
`RedisSearchIndex.get_entity_embeddings()` and `compute_similarity()` use the
explicit group key when supplied and scan matching entity embedding keys across
groups when it is omitted. The same pass corrected the lookup field to
`embedding`, matching what Redis indexing writes.
A static group-scope contract now guards that class of regression across
production storage, retrieval, and consolidation code by rejecting optional
`group_id` fallbacks that silently narrow to the raw `default` group.

The showcase evaluation adapter now follows the same one-brain contract. Each
Engram-backed showcase baseline derives an explicit `showcase_<adapter>` brain
group and passes it through observe, remember, projection, prospective-memory
intention writes/deletes, context assembly, recall, and relationship-name
resolution. This keeps benchmark/evaluation runs from silently proving only the
raw `default` group path while the product runtime is expected to operate as
one explicit brain per person/session.
The other benchmark defaults have been aligned as well: LoCoMo uses
`locomo_benchmark`, recall-need fixtures use `memory_need_benchmark`, and the
echo-chamber simulation uses `echo_chamber_benchmark`. Focused tests assert
those defaults flow into episode creation/indexing, retrieval probes, graph
probe calls, entity lookups, and activation writes. A benchmark-package scan no
longer finds raw `group_id="default"` defaults in production benchmark code.

Explicit recall packet analysis is group-scoped now too. REST
`/api/knowledge/recall`, chat-tool recall execution, and MCP `recall` already
ran retrieval under the active brain group, but their packet-side
`analyze_memory_need()` calls could rely on the analyzer default. Those calls
now pass the same tenant/tool/session `group_id` used for retrieval and
threshold lookup, with focused REST/chat/MCP tests covering the contract.
The chat memory-need helper now enforces the same rule: `_analyze_chat_memory_need()`
requires an explicit `group_id`, and helper tests pass a tenant brain so future
callers cannot accidentally reintroduce raw `default` analyzer state.

Consolidation replay now carries that one-brain contract through its remaining
deferred projection path. Replay linked-entity reads pass the active cycle
`group_id`, and group-aware extractors receive both `episode_id` and `group_id`
when reprocessing cue-only episodes, so native narrow replay cannot rebuild
evidence metadata under the raw `default` brain.

The remaining projection-yield and semantic-transition linked-entity reads are
also group-scoped now. Worker feedback, triage feedback, and semantic coverage
all read episode links with the active brain `group_id`, keeping yield
calibration and episode maturation inside the same one-brain boundary as the
projection that produced the links.

A static group-scope contract now protects that boundary. The test scans
production calls to group-aware graph/activation accessors and fails if a call
omits an explicit `group_id` keyword or the positional argument slot that carries
the active brain group.

`engram evaluate` now uses the configured one-brain default on operator paths.
When `--group-id` is omitted, both `engram evaluate --smoke` and
`engram evaluate --from-json` fall back to `EngramConfig.default_group_id`
instead of raw `default`, so native PyO3 smoke/report checks follow the same
configured brain as serve, doctor, lifecycle, and health.

The populated native surface test now covers project topology too. It bootstraps
a temporary project through REST, includes stale/wrong group fields in the
payload, then verifies artifact search returns the README hit from the active
`native_brain`. Bootstrap adds a cue-only capture episode, so the lifecycle
expectation tracks Capture/Cue growth without increasing projected-memory
counts. The same bootstrapped project is now checked through MCP
`bootstrap_project` as an idempotent `already_bootstrapped` project, then
searched through MCP `search_artifacts`, covering the headless agent artifact
path against the same PyO3 native brain. REST `/api/knowledge/runtime` and MCP
`get_runtime_state` now verify the same native runtime mode and artifact
freshness state for that project. This exposed a native-specific artifact gap:
Helix entity attributes could be double-encoded after GraphManager attribute
updates, and artifact search could miss content/claim hits when vector/BM25
search returned no useful rows. Native reads now tolerate JSON-string
attributes, and artifact search has a bounded lexical fallback over
bootstrapped Artifact summaries, snippets, and claims.

The populated native surface test now covers epistemic routing too. REST
`/api/knowledge/route` and MCP `route_question` classify the same
project-grounded reconcile question against the active PyO3 project context,
verifying the evidence plan points to artifacts and discourages raw fact search
for project-truth questions.

MCP route-question auto-observe now has native coverage too. The surface test
temporarily enables tool-call recall, calls `route_question` with a long
project-grounded question, and verifies the middleware stores a `tool_piggyback`
episode plus cue in the active PyO3 `native_brain`.

The populated native surface test now covers prospective memory intentions too.
REST `/api/knowledge/intentions` and MCP `intend`/`list_intentions`/
`dismiss_intention` create, list, soft-disable, and list disabled intentions in
the same active PyO3 `native_brain`.

The populated native surface test now covers edge adjudication resolution too.
It creates pending adjudication work items in PyO3 Helix, resolves them through
REST `/api/knowledge/adjudicate` and MCP `adjudicate_evidence`, and verifies the
persisted rejected request state in the active `native_brain`.

The populated native surface test now covers conversation persistence too. It
creates a conversation through REST, includes stale/wrong group fields in the
payload, appends user and assistant messages, then verifies conversation list
and message reads come back from the active `native_brain`. This keeps the
no-Docker PyO3 path covered for conversation storage.

That conversation coverage now includes update/delete semantics. The REST
parity test renames the conversation, deletes it, verifies missing/deleted
native conversations return 404, and checks the PyO3 Helix message nodes are
removed through existing find + hard-delete queries instead of a non-existent
bulk delete query.

The same populated native test now covers the chat stream path without calling
an external model. It mocks the Anthropic client, posts to `/api/knowledge/chat`
with stale/wrong group fields in the payload, verifies the SSE finish event
returns a conversation id, reconstructs the streamed assistant text, and polls
the native conversation endpoint until the fire-and-forget user/assistant turn
is persisted in Helix.

The populated native surface test now covers forgetting too. It creates explicit
test entities in `native_brain`, exercises `/api/knowledge/forget` and the MCP
`forget` tool, then verifies the forgotten entities disappear from search and
their activation state is cleared. REST payloads include stale/wrong group
fields so the test continues to prove tenant ownership comes from the active
native brain, not caller-supplied body fields.

The populated native surface test now covers explicit feedback too. It creates
explicit native entities, records REST feedback that creates a `PREFERS` edge,
records MCP feedback that creates an `AVOIDS` edge, and verifies both edges hang
off the active `native_brain` `UserPreference` profile. This exposed a feedback
publisher mismatch: `record_explicit_feedback()` was awaiting the real sync
`EventBus.publish()` with the old two-argument shape. It now publishes
group-scoped events to the real bus while preserving async test-double support.

The populated native surface test now covers identity-core mutation too. MCP
`mark_identity_core` marks and unmarks an explicit native entity, then reads the
flag back from the active PyO3 Helix graph under `native_brain`.

The populated native surface test now covers context assembly too. It calls REST
`/api/knowledge/context` with a topic hint and verifies the dashboard/API
camelCase shape, then calls MCP `get_context` and verifies the raw manager shape
against the same active `native_brain`. This keeps the agent-facing context
surface tied to the same native brain as recall, lifecycle, and evaluation.

MCP notification piggybacking now has native coverage too. The surface test
temporarily enables notification surfacing, seeds active and wrong-group
notifications, calls MCP `get_context`, and verifies only the active PyO3
`native_brain` notification appears in `memory_notifications`.

MCP auto-recall piggybacking now has native coverage too. The surface test
creates an explicit `native_brain` anchor entity, temporarily enables
tool-call auto-recall, calls MCP `search_entities`, and verifies the response
attaches `recalled_context` from `recall_lite` over the active PyO3 graph.

MCP prompt guidance now names the shared brain-loop contract. The system prompt
explicitly describes `Capture -> Cue -> Project -> Recall -> Consolidate`, so
headless agents see the same lifecycle language that REST, dashboard, docs, and
tests use for observe/remember/recall/consolidation behavior.

The populated native surface test now covers direct entity/fact lookup too. It
creates explicit native entities plus a `USES` relationship, verifies REST
`/api/entities/search` and `/api/knowledge/facts`, then checks MCP
`search_entities` and `search_facts` over a second explicit native
relationship. This keeps fact lookup tied to the same PyO3 brain rather than
only testing recall's higher-level result path.

The same explicit native relationship now covers graph-detail reads too. REST
entity detail and neighbor endpoints verify the outgoing `USES` fact and
neighborhood edge, while MCP `get_graph_state`, entity profile, and neighbor
resources read the same relationship from the active PyO3 graph.

REST entity mutation now has native coverage too. The surface parity test
patches a standalone TestMemory entity through `/api/entities/{id}`, reads the
updated entity back from PyO3 Helix, soft-deletes it through the same REST route,
and verifies detail/search plus activation no longer surface it.

The populated native surface test now covers MCP write tools too. After the
stable lifecycle/evaluation/recall assertions, it calls MCP `remember` and
`observe`, verifies their shared write lifecycle contract, and reads the created
episodes plus the observe cue back from the active PyO3 Helix graph under
`native_brain`.

The populated native surface test now covers REST text observe too. REST
`/api/knowledge/observe` verifies the Capture -> Cue lifecycle write contract,
conversation-date preservation, and episode/cue persistence in the active PyO3
Helix graph under `native_brain`.

The populated native surface test now covers REST auto-observe too.
`/api/knowledge/auto-observe` verifies hook-style Capture -> Cue writes,
session/conversation metadata, cue persistence, and duplicate-content
`dedup_skipped` lifecycle semantics in the active PyO3 `native_brain`.

The populated native surface test now covers attachment capture too. REST
`/api/knowledge/observe-image` and `/api/knowledge/observe-file`, plus MCP
`observe_image` and `observe_file`, verify image/file lifecycle metadata and
read the persisted attachment episodes plus cues back from the active PyO3
Helix graph under `native_brain`.

The dashboard now has an opt-in native smoke harness. The test at
`dashboard/src/test/nativeDashboardSmoke.test.tsx` is skipped in the default
Vitest lane, but with `VITE_ENGRAM_DASHBOARD_NATIVE_SMOKE=1` and `VITE_API_URL`
pointed at a populated native REST server, it fetches lifecycle, evaluation, and
recall through the dashboard API client and renders the Lifecycle panel from
that live contract. With approved local bind access, the live smoke now passes
against a PyO3 native REST server seeded by `engram evaluate --smoke --mode
helix`; the test fetches lifecycle, evaluation, and recall from
`127.0.0.1:8102` and renders native `native_brain` data.

After the adjudication-pressure evaluation contract, the live dashboard native
smoke was re-run against a freshly seeded PyO3 data directory. It still passed,
and the REST logs showed lifecycle/evaluation/recall/dashboard reads without
the previous optional `search_graph_embed_vectors` HNSW index errors.

That harness now also has a no-bind fixture lane. The default Vitest path mocks
native-shaped lifecycle, evaluation, recall, episode listing, and consolidation
status/history API payloads, runs them through the dashboard API client, and
renders the Lifecycle, Consolidation, and Memory Feed panels with
`native_brain` data. It also asserts native episode cue/projection-state
preservation for Capture/Cue. This does not replace the live native REST smoke,
but it keeps the dashboard contract verifiable in normal sandboxed runs.

`NativeTransport` now treats Helix native's missing-HNSW-entrypoint response for
optional graph-embedding vector searches as an empty result. This keeps Recall's
graph-structural path graceful when graph embeddings have not been trained yet,
instead of logging errors during normal PyO3 native operation.

Native same-process reopen stability now has coverage. The surface parity test
seeds a populated PyO3 brain once, then creates and shuts down the FastAPI app
three times against the same native data directory. Each reopen checks
lifecycle, evaluation, and recall, which guards the process-wide native engine
cache and shared-client shutdown ownership without requiring a bound local
socket.

Native write/read load now has bounded coverage too. The surface parity test
writes five additional memories through REST `remember`, runs repeated recall
queries against both seeded and newly remembered content, and then verifies that
lifecycle and evaluation totals remain coherent on the populated PyO3 brain. It
uses ASGI transport, so it exercises the public REST contracts without depending
on local socket binding.

That bounded native load test now includes REST offline replay as well. It
patches the drained queue to return one queued entry with a stale/wrong
`group_id`, calls `/api/knowledge/replay-queue`, and verifies recall plus
lifecycle totals from the active `native_brain`. The assertion intentionally
counts the replayed entry as captured/cued, not immediately projected, because
offline replay routes through `store_episode()` rather than the synchronous
`remember` projection path.

Native multi-batch load/reopen now has coverage too. The surface parity suite
writes 12 additional memories through REST `remember` across three batches,
checks lifecycle totals and recall after each batch, shuts the ASGI runtime down,
reopens the same PyO3 native data directory, and verifies lifecycle, evaluation,
and recall still describe the expanded `native_brain` coherently. This is still
a bounded pytest soak rather than a true long-running stress test.

The native path now also has an operator load smoke outside pytest.
`engram evaluate --smoke --mode helix` accepts `--smoke-load-count` and
`--smoke-recall-rounds`, adding deterministic load episodes before projection
and then running post-projection recall checks. A 120-load native run exposed
the triage batch boundary: one triage-only cycle projected 100 of 123 episodes.
The smoke runner now repeats deterministic triage cycles until the expected
projected count is reached and records `smoke.cycle_count`/`smoke.cycle_ids`.
The verified command with `--smoke-load-count 120 --smoke-recall-rounds 5`
returned 123 captured/cued/projected episodes, two completed consolidation
cycles, no coverage gaps, and 30 recall checks against native PyO3 Helix.
The same smoke now accepts `--smoke-min-duration-seconds` plus
`--smoke-pause-seconds` so operators can run hour-scale native recall soaks
against the populated PyO3 brain without inventing a separate command. The JSON
report records `smoke.duration_recall_checks`,
`smoke.duration_elapsed_seconds`, and the requested duration/pause settings.
The first real hour-scale run passed against native PyO3 Helix with disposable
storage: 9 captured/cued/projected episodes, one completed triage consolidation
cycle, no coverage gaps, 10,362 sustained recall checks, and 3600.599 seconds of
reported Recall soak time.

The group-scoped episode-entity lookup contract is now backend-wide. The native
recall work required `GraphManager.recall()` to pass `group_id` into
`get_episode_entities()` so non-default native brains do not fall back to
`default`. That exposed a SQLite/Falkor protocol mismatch in the non-Helix gate,
so the graph-store protocol, SQLite store, FalkorDB store, and shared storage
contract now all accept and verify optional group-scoped episode-entity lookups.

REST offline replay now preserves the same one-brain boundary. The
`/api/knowledge/replay-queue` endpoint drains local queued entries under the
current request tenant and ignores any `group_id` embedded in the offline
payload, preventing stale or injected queue metadata from writing memories into
another brain.

Project bootstrap now has explicit non-default group coverage for that same
boundary. Project paths stay as topology/context within a brain: Project and
Artifact entities, bootstrap episodes, and cue-only projection-state sync all
remain under the active `group_id`.

Fresh SQLite/lite databases now include `skipped_meta` and `skipped_triage`.
Those columns were already part of the projection-state update contract, and
adding them to the base schema plus idempotent migrations keeps lite doctor and
smoke runs aligned with the cue/projection state helper.

Clean dashboard smoke now has a server-side guard. `server.auto_observe_enabled`
defaults on for normal hook capture, but can be set false with
`ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false` for local UI smokes and demos. When
disabled, `/api/knowledge/auto-observe` returns a skipped capture contract with
`reason=disabled` and does not write an episode.

## Dirty Worktree Snapshot

The current worktree already contains substantial uncommitted work from the
first pass.

Modified areas:

- Dashboard API, shell, sidebar, episode/feed components, WebSocket hook, store
  types, preferences, and tests.
- Backend knowledge API, MCP server, config, `GraphManager`, retrieval pipeline
  and scoring, triage scorer, consolidation engine/scheduler/models/phases, and
  Helix schema/search.
- Backend tests for consolidation, graph embedding, maturation, schema
  formation, and triage.

Untracked areas:

- Quest-mode dashboard files under `dashboard/src/views`, `questSlice`, RPG
  constants, and `FeedbackButtons`.
- `server/engram/consolidation/phases/calibrate.py` and associated tests.
- LongMemEval data/results and SDK scripts.
- Helix generated config/schema under `helixdb-cfg/db`.
- `.mcp.json` and `scripts/clawhub-publish.sh`.

Do not stage or revert these implicitly. Treat them as active user/first-pass
work unless a later task narrows the scope.

## Runtime Map

### 1. Capture

Capture enters through REST and MCP:

- REST: `POST /api/knowledge/observe`, `auto-observe`, `observe-image`,
  `observe-file`, and `replay-queue`.
- MCP: `observe`, `observe_image`, and `observe_file`.
- Shared internal path: `GraphManager.store_episode()`.
- Explicit service: `EpisodeCaptureService`.

`store_episode()` creates an `Episode` with `EpisodeStatus.QUEUED` and
`EpisodeProjectionState.QUEUED`, publishes `episode.queued`, and optionally
creates an `EpisodeCue` when `cue_layer_enabled` is active.

REST and MCP observe responses now expose this as a shared public lifecycle:
capture stored, lifecycle stage `cue`, projection mode `background`, projection
status `queued`.

### 2. Cue

Cue generation is deterministic and lives around:

- `engram.extraction.cues.build_episode_cue`
- `EpisodeCue`
- `GraphStore.upsert_episode_cue`
- optional `SearchIndex.index_episode_cue`

Cues carry projection state, cue text, salience, mention spans, contradiction
hints, hit/use counters, policy score, attempts, and timestamps. Recall can
return `result_type="cue_episode"` before full projection has happened.

The first-pass cue fix is important: triage-skipped episodes now keep cue
projection state aligned with the episode's projection state.

### 3. Project

Projection enters through three paths:

- REST/MCP `remember()` calls `GraphManager.ingest_episode()`.
- `EpisodeWorker` calls `GraphManager.project_episode()` for high-confidence or
  scheduled episodes.
- `TriagePhase` calls `GraphManager.project_episode()` for selected queued
  episodes.

`EpisodeProjectionService` now owns the Project stage behavior previously
embedded directly in `GraphManager.project_episode()`:

- duplicate-content skip
- system-discourse skip
- projection-state transitions
- projection planning
- evidence extraction and commit policy
- edge adjudication request creation
- legacy projector fallback
- entity/relationship apply
- indexing
- emotional encoding context
- prospective memory checks
- projection graph events
- retry/dead-letter handling

This is an explicit service boundary now, but its dependency surface is still
wide because the first extraction intentionally reused existing GraphManager
collaborators. The legacy apply path is now split behind
`LegacyProjectionExecutor`, and the evidence hot path is split behind
`EvidenceProjectionExecutor`. Projection outcomes now have a typed
`ProjectionLifecycleResult`, making successful and skipped lifecycle outcomes
assertable without GraphManager internals.

`GraphManager.project_episode()` remains the stable public facade for REST, MCP,
worker, triage, and tests.

REST and MCP remember responses now expose synchronous projection as a shared
public lifecycle: capture stored, lifecycle stage `project`, projection mode
`synchronous`, projection status `attempted`. This does not claim projection
success, because `EpisodeIngestionService` intentionally preserves the
compatibility contract that returns the captured episode ID after
`project_episode()` records failure state.

### 4. Recall

Recall uses `GraphManager.recall()` and `engram.retrieval.pipeline.retrieve()`.

The retrieval pipeline builds entity candidates, episode candidates, and
cue-backed candidates, then applies query routing, activation, spreading,
working memory, optional planner/decomposition, graph expansion, scoring,
reranking/diversity, and special result merging.

`GraphManager.recall()` converts scored results into runtime dictionaries:

- `entity`
- `episode`
- `cue_episode`

The raw dictionary assembly now lives in
`engram.retrieval.result_builder.RecallResultBuilder`. `GraphManager.recall()`
still owns retrieval orchestration, cue/access feedback, working-memory updates,
and Recall-stage side-effect sequencing, but tier-aware episode truncation, cue
payloads, score breakdowns, relationship payloads, and raw
`entity`/`episode`/`cue_episode` result shapes are no longer hand-built inline.

Entity-linked episode traversal and temporal contiguity now live in
`engram.retrieval.episode_traversal.RecallEpisodeTraversal`. The service appends
synthetic episode results while preserving the active `group_id`, duplicate
filtering, merged-episode filtering, and expansion score metadata. This leaves
`GraphManager.recall()` responsible for Recall-stage sequencing while moving
read-only graph expansion behind a narrower retrieval boundary.

Near-miss lookup and payload construction now live in
`engram.retrieval.near_miss.RecallNearMissBuilder`. The helper formats entity
near misses, checks cue/episode eligibility, skips merged episodes, and builds
the cue near-miss payload through the same cue payload contract. `GraphManager`
still records cue near-miss feedback through `_record_cue_hit()`, so the
feedback side effect has not moved.

Retrieval priming updates now live in
`engram.retrieval.priming.RecallPrimingUpdater`. The updater handles top-N
entity filtering, one-hop neighbor lookup, TTL calculation, max-neighbor
clamping, and boost writes into the caller-owned priming buffer. `GraphManager`
still owns the buffer and decides when priming participates in a Recall-stage
turn.

Relevance-confidence application is now wrapped by
`engram.retrieval.confidence.RecallConfidenceApplier`. The lower-level
`apply_relevance_confidence()` scorer remains available for direct tests, while
the wrapper owns the feature flag, empty-result guard, error isolation, and
call into the scorer. `GraphManager.recall()` now only sequences the confidence
step.

Recall-query conversation fingerprint recording now lives in
`engram.retrieval.context.RecallConversationFingerprintRecorder`. The recorder
owns the feature flag, missing-context no-op, provider embedding adapter, and
`recall_query:<source>` tagging while using non-fingerprinting ingest mode so
recall queries can enter conversation history without overwriting the live
conversation fingerprint. `GraphManager.recall()` only decides when that
recording should be attempted.

Recall working-memory writes now live in
`engram.retrieval.working_memory.RecallWorkingMemoryUpdater`. The updater owns
the disabled-buffer no-op and write-through to entity, episode, and query
working-memory entries. `GraphManager.recall()` still decides when recalled
items and the query should enter working memory, but the write policy no longer
lives inline in the facade loop.

Recall entity interaction telemetry now lives in
`engram.retrieval.feedback.RecallInteractionRecorder`. The recorder owns
publishing `recall.interaction` events and writing recall-need interaction
samples for entity results. `GraphManager.recall()` still decides which recalled
entity has an interaction, but the telemetry/learning side effect is no longer
inline in the facade loop.

True Recall-stage entity access recording now lives in
`engram.retrieval.feedback.RecallEntityAccessRecorder`. `GraphManager` still
decides whether a recall or feedback operation should count as a true access,
but the activation write, `activation.access` event payload, and labile
reconsolidation-window marking now live together in the retrieval feedback
module.

Recall-stage cue feedback now lives in
`engram.retrieval.feedback.RecallCueFeedbackRecorder`. `GraphManager` still
decides when recalled cue episodes and cue near-misses should record feedback,
but cue counter updates, cue hit/near-miss/policy/promotion events, recall-need
cue samples, and hot-cue projection scheduling now live together in the
retrieval feedback module and route promotion through the shared
`sync_projection_state()` helper.

Explicit post-response memory feedback now lives in
`engram.retrieval.feedback.RecallMemoryInteractionApplier`.
`GraphManager.apply_memory_interaction()` still exposes the compatibility API
used by REST/chat feedback, but dedupe, cue feedback dispatch, entity access,
Thompson-sampling positive/negative feedback, `recall.interaction` publication,
and recall-need interaction samples now live behind the retrieval feedback
boundary.

Explicit preference feedback reinforcement now lives in
`engram.retrieval.preference_feedback.PreferenceFeedbackRecorder`.
`GraphManager.record_explicit_feedback()` still exposes the REST/MCP
compatibility API, but UserPreference profile creation, PREFERS/AVOIDS edge
creation or strengthening, domain lookup, and `feedback.recorded` publication
now live behind the retrieval feedback boundary used by calibration and
preference-directed recall scoring.

Memory forgetting/correction now lives in
`engram.retrieval.forgetting.MemoryForgettingService`.
`GraphManager.forget_entity()` and `GraphManager.forget_fact()` still expose
the REST/MCP compatibility APIs, but entity soft-delete plus activation clear,
fact endpoint resolution, relationship invalidation, and forget response
formatting now live outside the facade.

Direct entity/fact lookup now lives in
`engram.retrieval.lookup.EntityFactLookupService`.
`GraphManager.resolve_entity_name()`, `GraphManager.search_entities()`, and
`GraphManager.search_facts()` still expose compatibility APIs for REST, MCP,
and internal callers, but read-only entity search, activation-score decoration,
fact search/name resolution, and epistemic fact filtering now live outside the
facade.

Agent context loading now lives in
`engram.retrieval.context_builder.MemoryContextBuilder`.
`GraphManager.get_context()` and its older helper APIs still expose the REST/MCP
and test compatibility surface, but tiered identity/project/recent/intention/
pinned-context assembly, deterministic briefing, token budgeting,
project-neighbor injection, and context access-event publication now live outside
the facade.

Prospective-memory intention management now lives in
`engram.retrieval.prospective.ProspectiveMemoryService`.
`GraphManager.create_intention()`, `list_intentions()`, `dismiss_intention()`,
`delete_intention()`, `migrate_flat_intentions()`, `_update_intention_fire()`,
and `update_intention_meta()` still expose the REST/MCP/consolidation
compatibility surface, but graph-embedded intention creation, v1 flat-table
fallback, TRIGGERED_BY edges, active filtering, soft/hard dismiss behavior,
fire-count updates, metadata updates, and lifecycle event publication now live
outside the facade.

Graph-state reads now live in `engram.retrieval.graph_state.GraphStateService`.
`GraphManager.get_graph_state()` still exposes the REST/MCP/lifecycle/evaluation
compatibility API, but graph stats enrichment, top-activated entity
materialization, active/dormant counts, recall/epistemic metrics attachment,
entity-type filtering, and optional relationship-edge expansion now live outside
the facade.

Epistemic question routing now lives in
`engram.retrieval.epistemic_route.EpistemicRouteService`.
`GraphManager.route_question()` and `_build_epistemic_route()` still expose the
REST/MCP/evidence-gathering compatibility surface, but memory-need analysis
integration, graph-probe use, surface capability derivation, question-frame
construction, evidence-plan construction, answer-contract application, payload
formatting, and route metrics recording now live outside the facade.

Epistemic evidence gathering now lives in
`engram.retrieval.epistemic_evidence.EpistemicEvidenceService`.
`GraphManager.gather_epistemic_evidence()` still exposes the chat/API
compatibility surface, but route-guided memory/artifact/runtime source queries,
project bootstrap before artifact reads, memory/artifact/runtime claim
construction, claim-state summary generation, two-pass answer-contract
reconciliation, stale-artifact miss detection, execution metrics, and
`EpistemicBundle` assembly now live outside the facade.

Runtime-state reads now live in
`engram.retrieval.runtime_state.RuntimeStateService`.
`GraphManager.get_runtime_state()` still exposes the REST/MCP/epistemic-evidence
compatibility surface, but effective mode/config reporting, feature flags,
project artifact freshness counts, latest observed artifact timestamp selection,
recall/epistemic metrics attachment, and generated-at timestamps now live
outside the facade.

Decision graph materialization now lives in
`engram.ingestion.decision_materializer.DecisionMaterializer`.
Conversation capture and artifact bootstrap still call through `GraphManager`
compatibility wrappers, but committed conversation-decision extraction,
conversation-record artifact upsert, artifact-claim decision linking, Decision
entity upsert/reinforcement, supersession edges, and idempotent relationship
creation now live outside the facade.

Consolidation cycle completion now lives in
`engram.consolidation.completion.ConsolidationCycleCompletionService`.
`ConsolidationEngine` still owns phase-loop orchestration, cancellation,
capability validation, and non-fatal phase failures, but final duration
stamping, final store update, post-cycle learning event publication,
successful-cycle finalization, and the final `consolidation.completed` event now
live outside the engine facade.

Structure-aware entity indexing now lives in
`engram.ingestion.entity_indexer.StructureAwareEntityIndexer`.
Projection post-processing, adjudication materialization, decision
materialization, and project bootstrap still call the `GraphManager`
compatibility wrapper, but predicate-weighted relationship context expansion and
enriched search-index payload construction now live outside the facade.

Artifact search and read formatting now live in
`engram.retrieval.artifacts.ArtifactSearchService`.
REST, MCP, runtime state, and epistemic evidence still call through
`GraphManager` compatibility wrappers, but project artifact listing, optional
bootstrap-before-read, search-index/fallback lookup, lexical claim fallback,
stale detection, and `ArtifactHit` construction now live outside the facade.

Project bootstrap writes now live in
`engram.ingestion.project_bootstrap.ProjectBootstrapService`.
REST, MCP, artifact search, and epistemic evidence still call through
`GraphManager.bootstrap_project()`, but project entity create/refresh, bootstrap
file expansion, artifact entity upsert, cue-only bootstrap episode capture,
PART_OF links, artifact-decision materialization calls, and bootstrap lifecycle
events now live outside the facade.

Lightweight entity-probe recall now lives in
`engram.retrieval.entity_probe.EntityProbeRecallService`.
`GraphManager.recall_lite()` and `GraphManager.recall_medium()` remain the MCP
and compatibility entry points, but mention extraction, session-cache handling,
FTS candidate probing, optional embedding rerank, top-fact rendering,
confidence/freshness labels, and token-budget packing now live outside the
facade.

Recall current-state result selection now lives in
`engram.retrieval.result_selection`. `GraphManager.recall()` still sequences
final result shaping, but the rule that current/currently/now queries prefer
entity state over raw episode or cue history now has direct tests and lives with
retrieval selection logic.

Recall request policy now lives in `engram.retrieval.request_policy`.
`GraphManager.recall()` still calls the retrieval pipeline, but near-miss
fetch-window sizing, primary/near-miss splitting, and ranking-feedback learning
decisions for passive versus true-usage interactions now live in pure
retrieval helpers with direct tests.

Primary Recall result materialization now lives in
`engram.retrieval.primary_results.RecallPrimaryResultMaterializer`.
`GraphManager.recall()` still sequences retrieval and post-processing, but
primary entity, episode, and cue result assembly, merged-episode filtering,
working-memory writes, entity access recording, cue feedback dispatch,
relationship fetches, and entity interaction telemetry now live behind a
retrieval-side materialization boundary.

Recall near-miss materialization now lives in
`engram.retrieval.near_miss.RecallNearMissMaterializer`. `GraphManager.recall()`
still stores `_last_near_misses` for downstream API/MCP surfaces, but entity
near-miss formatting, cue near-miss context lookup, cue near-miss feedback,
refreshed cue payload lookup, and near-miss payload assembly now live behind
the retrieval near-miss boundary.

Recall post-processing now lives in
`engram.retrieval.post_process.RecallPostProcessor`. `GraphManager.recall()`
still owns the compatibility API and calls the retrieval pipeline, but the
post-primary sequence for entity-linked episode expansion, temporal expansion,
current-state result selection, query working-memory writes, priming updates,
near-miss materialization, relevance-confidence scoring, and recall-query
fingerprint recording is now one retrieval-side service boundary.

Recall-stage orchestration now lives in
`engram.retrieval.service.RecallService`. `GraphManager.recall()` remains the
public compatibility method, but request policy, retrieval pipeline invocation,
primary/near-miss splitting, primary result materialization, post-processing,
and the final results/near-misses contract now execute behind one tested Recall
service boundary.

REST, MCP, and knowledge-chat now route those dictionaries through
`engram.retrieval.presenter`. The presenter owns the shared semantic contract
for score, score breakdown, packet inclusion, cue fields, relationships, and
result metadata while still preserving surface-specific naming conventions.

### 5. Consolidate

The current engine contract is 16 phases:

```text
triage -> merge -> calibrate -> infer -> evidence_adjudication ->
edge_adjudication -> replay -> prune -> compact -> mature -> semanticize ->
schema -> reindex -> graph_embed -> microglia -> dream
```

Backend phase order and scheduler tiering now share
`engram.consolidation.phase_registry`, so the engine tests and tiered scheduler
are guarded against divergent phase-name lists. `ConsolidationEngine` also
validates constructed runtime phase order against that registry on startup.
Concrete runtime phase construction now lives in
`engram.consolidation.phase_catalog.build_consolidation_phases`, keeping the
16-phase object assembly in a dedicated Consolidate-stage catalog while the
engine retains the cycle loop.
The dashboard has a small mirror of that order in
`dashboard/src/constants/consolidation.ts`; consolidation panel fixtures build
from it, and quest-mode phase descriptions are typed against it so the frontend
does not quietly drop adjudication or future phases. A backend contract test
reads that dashboard constant and compares it to
`engram.consolidation.phase_registry.CONSOLIDATION_PHASE_ORDER`.

`ConsolidationEngine` owns phase orchestration, capability validation,
cancellation, cycle finalization, and non-fatal phase failure handling, but it
no longer owns concrete phase construction.

The engine now uses `engram.consolidation.lifecycle` for the initial selected
phase plan and for phase/cycle event payloads. This makes the Consolidate stage
partly explicit. It also delegates concrete phase execution and audit-record
persistence to `engram.consolidation.phase_runner.ConsolidationPhaseRunner`,
which returns the `PhaseResult`, direct records, and merge/prune removed-node
ids. Event publication routes through
`engram.consolidation.events.ConsolidationEventPublisher`, including cycle
events, phase events, graph deltas, and learning update notifications.
Post-cycle distillation and calibration artifact generation routes through
`engram.consolidation.learning.ConsolidationLearningService`.
Requested phase names are now validated by the lifecycle plan before a cycle
starts, so CLI, scheduler, or tool-call typos cannot silently complete a
zero-work cycle against an unknown phase. The documented
`python -m engram.consolidation --phases ...` path reports a clean operator
error for unknown phases instead of leaking a traceback.
The CLI now also treats engine-returned failed or cancelled cycles as operator
failures: it still prints the structured JSON result, but the human summary no
longer says `complete`, and the process exits nonzero. That JSON result now
includes cycle-level and phase-level `error` fields.
MCP `trigger_consolidation` now exposes the same cycle/phase error fields and
uses failed-cycle summary wording when the engine returns a failed cycle.
MCP `get_consolidation_status` now also reads the active consolidation store:
it keeps `is_running=false` for synchronous MCP execution while returning
`latest_cycle` through the shared presenter, including cycle and phase errors.
REST consolidation status, history, and detail reads expose the same cycle and
phase failure fields, so cycle `error`, phase `error`, and phase `duration_ms`
are consistently available to dashboard and API clients.
The dashboard consolidation slice now preserves `latest_cycle` from REST status
refreshes by merging it into the cycle list, so cycle and phase errors are not
dropped while history refresh is still pending.
The Consolidation panel now surfaces cycle-level errors in the cycle list, so a
failed cycle's top-level failure reason is visible before opening detail.
REST, MCP, and CLI now route cycle/phase result formatting through
`engram.consolidation.presenter`, which owns the shared serializer, affected and
processed totals, and failed-cycle description semantics while preserving each
surface's existing naming convention.
That presenter now also owns the structured operator summary payload, so REST
status/history/detail, MCP trigger/status, CLI JSON, lifecycle summary, and
dashboard types all carry the same `summary.total_processed`,
`summary.total_affected`, and `summary.description` contract.
The dashboard API client trigger type now matches the REST trigger response:
`status`, `group_id`, and `dry_run`, without a fake `cycle_id`.

`ConsolidationScheduler` maps those phases into tiers:

- Hot: `triage`
- Warm: `merge`, `calibrate`, `infer`, `evidence_adjudication`,
  `edge_adjudication`, `compact`, `mature`, `semanticize`, `reindex`,
  `microglia`
- Cold: `replay`, `prune`, `schema`, `graph_embed`, `dream`

Manual, pressure, and flat scheduled cycles can still run all phases.

## Drift And Gaps

1. `GraphManager` is now a compatibility facade over several explicit services,
   but it is still the easiest place for new lifecycle logic to accumulate. Keep
   extracting only concrete Capture, Project, Recall, or Consolidate contracts
   when a smaller tested boundary appears; avoid a wholesale rewrite.

2. REST, MCP, chat, CLI, and dashboard surfaces now share the core
   remember/observe, recall, lifecycle, and evaluation contracts. The remaining
   risk is future drift on new surfaces, so new API or tool responses should
   reuse the shared presenters/builders instead of reformatting lifecycle state
   locally.

3. The public phase-count drift is closed for the current 16-phase engine
   contract. Backend registry validation and dashboard consolidation fixtures
   now guard the current order whenever phases are added, removed, renamed, or
   rescheduled.

4. The dashboard is now lifecycle-first through the Brain Loop surface and
   stage drilldowns. Quest/world-map mode should stay an alternate presentation
   or drilldown, not the primary product explanation of Engram's memory loop.

5. Local verification is much cleaner: the broad non-Docker/non-Helix backend
   gate currently passes with 3357 tests, 43 skips, and 236 external-service
   tests deselected after REST auto-observe request parsing moved behind the
   Capture-stage surface boundary and the native surface manifest learned to
   classify the advertised FastMCP `/mcp` transport path from the root-mounted
   app. Earlier broad-gate fixes made the Helix dashboard analytics unit
   fixture date-stable, guarded the doctor readiness failure path, centralized
   shared companion-store bootstrap creation, made notification/scheduler
   dependencies explicit, and moved smoke cue feedback onto the public manager
   facade. REST and MCP shutdown now use the same runtime-resource stop/close
   boundary too, with shutdown consolidation orchestration behind a
   helper and static guards against local store-close or engine-cycle drift.
   REST knowledge-chat SSE orchestration is in chat runtime code instead of the
   route, and MCP instructions plus `claim_authority(project_path, user_message,
   file_memory_present)` now claim Engram's portable-memory authority and tell
   agents to bootstrap an empty project runtime instead of routing around it.
   PyO3 native has focused parity plus a one-hour operator Recall soak. Docker
   Helix/full-mode is a separate compatibility/integration lane, and multi-hour
   native endurance remains an explicit optional gate, not an assumption.

6. The operator-facing P3 evaluation loop now measures cue usefulness,
   projection yield/backlog/freshness, recall gate latency/control posture,
   recall quality, false/missed recall, continuity, consolidation effect rate,
   adjudication phase pressure, live open adjudication/evidence work pressure,
   and calibration quality. The Recall gate fields are
   covered by REST/CLI/dashboard contracts, projected/consolidated smoke,
   native PyO3 smoke, smoke verifier, and no-bind native dashboard fixtures.
   Coverage gaps now flag label-only reports that lack runtime gate analyses or
   analyzer latency, so evaluation reports cannot confuse saved labels with live
   Recall Gate coverage. The latest runtime gate metrics snapshot is also
   persisted and reloaded by CLI, REST, and MCP reports, so reopened native live
   reports keep the in-process smoke proof. Coverage gaps also distinguish
   saved calibration snapshots from quality-scored calibration evidence, so
   unscored calibration history cannot satisfy the Consolidate quality gate.
   `engram evaluate --require-evaluation-signals` now makes the six-signal
   readiness map a hard CLI gate for JSON/live/native reports outside smoke,
   including saved brain-loop report artifacts passed back through
   `--from-json`. The native surface manifest now tracks the Helix variant as
   an operator hard gate.
   LongMemEval/benchmark artifacts remain deeper benchmark evidence, not yet
   the everyday operator gate.

## Next Build Queue

### P0 - Stabilize The Contract

- Done: add a shared runtime/presenter module for recall result formatting used
  by REST, MCP, knowledge chat, and tests.
- Done: add contract tests proving REST, MCP, and chat derive equivalent
  semantic recall payloads for `entity`, `episode`, and `cue_episode`.
- Done: extend the same contract pattern to `remember` and `observe` responses
  so capture/projection semantics do not drift between REST and MCP.
- Done: remove a random GraphSAGE normalization test flake so the lite backend
  gate remains a meaningful development signal.
- Done: extract capture/cue storage from `GraphManager.store_episode()` into
  `EpisodeCaptureService`.
- Done: add a shared projection-state/cue-state synchronization helper used by
  the touched manager, worker, triage, replay, and capture paths.
- Done: extract projection execution from `GraphManager.project_episode()` into
  `EpisodeProjectionService` while keeping `GraphManager` as the facade.
- Done: extract the legacy projection apply path into `LegacyProjectionExecutor`
  with focused tests for successful apply and retryable extractor errors.
- Done: extract the evidence projection hot path into
  `EvidenceProjectionExecutor` with focused tests for commit/defer persistence
  and missing commit-policy failures.
- Done: add `ProjectionLifecycleResult` so Project-stage outcomes have a typed
  contract for events, tests, and future API/MCP surfaces.
- Done: carry `group_id` on `ProjectionPlan` and forward it to group-aware
  extractors, so the legacy projection/narrow-extractor path no longer builds
  internal evidence bundles under the raw `default` brain when the active
  episode belongs to another group.
- Done: pass the consolidation replay cycle `group_id` into linked-episode
  entity reads and group-aware extractor calls, so deferred/native narrow replay
  stays in the active brain during reprocessing.
- Done: pass active `group_id` into worker projection-yield feedback, triage
  projection-yield feedback, and semantic-transition coverage linked-entity
  reads.
- Done: add a static group-scope contract for production calls to group-aware
  graph/activation accessors.
- Done: make REST offline replay queue ingestion tenant-scoped so queued
  payload metadata cannot override the current `group_id`.
- Done: add `engram.consolidation.lifecycle` for the Consolidate-stage phase
  plan and phase/cycle lifecycle event result contract.
- Done: make that Consolidate-stage phase plan reject unknown requested phase
  names before a cycle starts, preventing silent no-op cycles from CLI,
  scheduler, or tool-call typos.
- Done: make the consolidation CLI exit nonzero for failed/cancelled cycles
  instead of reporting failed engine cycles as complete operator runs.
- Done: include cycle-level and phase-level errors in consolidation CLI JSON so
  tooling can inspect failed cycles without parsing stderr.
- Done: align MCP `trigger_consolidation` with the same cycle/phase error
  contract and failed-cycle summary wording.
- Done: align REST consolidation status/history/detail read contracts around
  one cycle serializer with cycle error, phase error, and phase duration.
- Done: correct the dashboard consolidation trigger response type to match the
  REST backend contract.
- Done: extract `engram.consolidation.presenter` so REST, MCP, and CLI
  consolidation outputs share one cycle/phase serialization, total, and
  failed-cycle description contract.
- Done: add `engram.consolidation.phase_registry` so backend phase order,
  scheduler tiers, engine phase-order tests, and engine runtime construction
  share one 16-phase contract.
- Done: add `engram.consolidation.phase_catalog` so concrete runtime phase
  construction lives outside the engine loop while still validating against the
  shared registry.
- Done: add a dashboard-side consolidation phase-order mirror so panel
  fixtures and quest-mode descriptions use one frontend 16-phase contract.
- Done: add a backend/dashboard phase-order contract test so the TypeScript
  mirror cannot quietly diverge from the Python registry.
- Done: add `engram.consolidation.phase_runner` for one-phase execution,
  audit-record persistence, new context decision persistence, and merge/prune
  removed-node discovery.
- Done: add `engram.consolidation.events` so cycle events, phase events, graph
  deltas, and learning update notifications use one Consolidate-stage event
  publishing boundary.
- Done: add `engram.consolidation.learning` so post-cycle distillation examples,
  calibration history collection, and calibration snapshots live outside the
  engine loop.
- Keep backend `engram.consolidation.phase_registry` and dashboard
  `CONSOLIDATION_PHASE_ORDER` aligned whenever the engine phase list changes.
- Classify or mark remaining Helix-dependent tests so `uv run pytest -m "not
  requires_docker and not requires_helix"` is meaningful.

### P1 - Extract Brain Loop Services

- Introduce explicit services for:
  - Done: capture/cue storage
  - Done: one-shot episode ingestion
  - Done: offline capture replay
  - Done: capture deduplication
  - Done: projection execution facade/service boundary
  - Done: recall presentation
  - Done: recall raw result assembly
  - Done: recall episode traversal
  - Done: recall near-miss lookup/payload formatting
  - Done: recall priming updates
  - Done: recall relevance-confidence application wrapper
  - Done: recall conversation fingerprint recording
  - Done: recall working-memory writes
  - Done: recall entity interaction telemetry
  - Done: recall true entity access and labile-window marking
  - Done: recall cue feedback and hot-cue scheduling
  - Done: recall explicit post-response feedback applier
  - Done: recall current-state result selection
  - Done: recall request policy
  - Done: recall primary result materialization
  - Done: recall near-miss materialization
  - Done: recall post-processing orchestration
  - Done: recall service orchestration
  - Done: agent context assembly
  - Done: prospective-memory intention management
  - Done: graph-state read model
  - Done: epistemic question routing
  - Done: epistemic evidence gathering
  - Done: runtime-state read model
  - Done: decision graph materialization
  - Done: consolidation cycle completion
  - Done: consolidation phase catalog construction
  - Done: structure-aware entity indexing
  - Done: artifact search/read service
  - Done: project bootstrap service
  - Done: lightweight entity-probe recall
  - Done: consolidation lifecycle orchestration boundaries
  - Done: episode worker runtime-store, batching, scoring, routing, and event
    parsing boundaries
- Keep `GraphManager` as a compatibility facade while tests are migrated.
- Done: move projection-state and cue-state synchronization into one shared
  helper for the touched worker, triage, replay, and projection paths.
- Done: route GraphManager cue-feedback promotion through the same helper so hot
  cue recall cannot drift episode and cue projection state.
- Done: route worker system-discourse skips and project-bootstrap artifact
  suppression through the same helper; the remaining inspected production
  `update_episode()` calls are non-projection metadata or storage writes.
- Consolidation lifecycle orchestration is now split into lifecycle contracts,
  event publishing, phase execution/audit persistence, post-cycle learning
  artifact generation, post-cycle finalization/pinned-context refresh, and
  preflight capability validation.
  `ConsolidationEngine` remains the compatibility cycle-loop facade.

### P2 - Make The Dashboard Lifecycle-First

- Done: add a lifecycle surface that shows Capture, Cue, Project, Recall, and
  Consolidate as one continuous loop.
- Done: map episode status, projection state, cue metrics, recall usage, and
  consolidation phase activity into that surface.
- Done: add a backend/API lifecycle summary contract where the UI previously derived
  stage health from separate collections.
- Done: extract that summary into a shared runtime builder and expose it through
  MCP `get_lifecycle_summary` for non-dashboard clients.
- Done: make lifecycle `consolidate.latestCycle` reuse the shared
  consolidation presenter so cycle and phase error fields stay aligned with
  REST/MCP/CLI consolidation outputs.
- Done: add `engram lifecycle` as a headless CLI surface for that same summary.
- Done: embed the same lifecycle summary in `engram doctor` diagnostics with a
  `--no-lifecycle` escape hatch.
- Done: document the shared lifecycle endpoint/tool for external agents and
  update stale public phase/tool counts.
- Done: refresh the lifecycle summary from episode, graph, activation snapshot,
  and consolidation WebSocket events.
- Done: wire lifecycle stage cards into existing drilldowns.
- Done: add stage-specific filter or anchor context for Capture, Cue, and
  Project drilldowns.
- Done: add Recall-stage context inside Knowledge where the existing panel can
  support it cleanly.
- Keep technical graph/feed/consolidation panels as drilldowns.

### P3 - Build The Evaluation Loop

- First slice done: `server/engram/evaluation/brain_loop_report.py` defines a
  pure local report builder and Markdown renderer, and
  `server/scripts/brain_loop_report.py` runs it against SQLite/lite mode or a
  JSON stats export.
- Persisted sample slice done: `server/engram/evaluation/store.py` stores
  labeled recall outcomes and session-continuity labels in SQLite, and the
  local report CLI reads those samples by default in lite mode.
- REST write/report slice done: `server/engram/api/evaluation.py` records
  recall labels and session-continuity labels through API endpoints, and returns
  the same local brain-loop report contract as the CLI.
- First-class CLI slice done: `engram evaluate` now prints the local brain-loop
  report, while `server/scripts/brain_loop_report.py` remains as a
  compatibility wrapper.
- Dashboard drilldown slice done: `dashboard/src/components/EvaluationPanel.tsx`
  exposes the same report contract as an operator-facing Evaluate view.
- MCP label/report slice done: `engram mcp` exposes recall-label,
  session-continuity-label, and report tools backed by the same local evaluation
  store and report builder.
- Live smoke done: the lite backend served the evaluation report to the
  dashboard Evaluate drilldown with seeded episode, recall-label, and
  session-continuity-label data.
- Operator label capture done: the dashboard Evaluate drilldown can submit
  recall-quality and session-continuity labels without direct REST/MCP calls.
- Missed-recall signal done: recall-quality labels can also mark whether recall
  was needed, and REST/MCP/CLI/dashboard reports expose memory-need recall,
  missed-recall rate, and supporting label counts.
- Projected/consolidated smoke done: `engram evaluate --smoke` verifies that
  lite-mode capture/cue, triage-driven projection, recall and continuity labels,
  a persisted consolidation cycle, and calibration snapshots produce a report
  with no coverage gaps.
- Native projected/consolidated smoke done: `engram evaluate --smoke --mode
  helix` verifies the same loop against PyO3 Helix without Docker, and reopened
  live reports preserve projection yield from the native data directory.
- Projection freshness signal done: Helix native stats compute
  `avg_time_to_projection_ms`, and the shared evaluation report/dashboard show
  projection latency plus processing duration beside Project-stage yield.
- Projection backlog signal done: the shared Project-stage report derives
  tracked count, projected rate, and backlog rate from projection state counts,
  and the dashboard Evaluate card surfaces backlog pressure.
- Recall gate-latency signal done: the shared Recall-stage report carries
  recall-need analyzer and graph-probe average/p95 latency from runtime
  controller metrics, and REST/CLI/dashboard surfaces show analyzer/probe p95.
- Recall gate-control signal done: the shared Recall-stage report carries
  runtime interaction outcome counts, graph override count, adaptive-threshold
  state, and active thresholds, and the dashboard Evaluate view surfaces that
  posture in a Recall Gate card.
- Recall gate smoke coverage done: `engram evaluate --smoke` runs a real
  memory-need analysis plus surfaced recall interaction and builds the report
  through `GraphManager.get_graph_state()` so latency/control fields are
  runtime-populated.
- Native Recall gate smoke coverage done: `engram evaluate --smoke --mode
  helix` verifies those runtime-populated latency/control fields on disposable
  PyO3 native storage for `native_brain`.
- Recall gate smoke verifier done: `assert_smoke_report()` now fails when the
  smoke report lacks gate analysis, a trigger, analyzer latency, surfaced
  feedback, or the gate-check counter.
- Recall gate coverage-gap contract done: the shared report now flags
  label-only recall evaluation as missing runtime gate analyses and flags gate
  analysis without analyzer latency, while the native PyO3 smoke keeps the
  in-process runtime proof separate from persisted labels.
- Recall gate runtime-metric persistence done: the evaluation store now saves
  the latest runtime metrics snapshot, and CLI/REST/MCP reports merge it when
  current stats have weaker gate coverage, with REST save-side and fallback
  regression coverage plus MCP fallback coverage. Reopened native live reports
  now keep the PyO3 smoke's analyzer latency, trigger count, and
  surfaced-feedback proof. Snapshot history is bounded to the latest 25 rows per
  group so repeated report reads do not turn evaluation telemetry into
  unbounded local DB growth. The evaluation store also tracks SQLite connection
  ownership now, so lite-mode report stores initialized with a borrowed graph DB
  connection do not close the shared graph store during cleanup. The SQLite
  consolidation store now follows the same ownership rule for borrowed lite-mode
  graph DB connections, and the SQLite conversation store closes standalone
  connections while leaving borrowed graph DB connections open. SQLite feedback
  storage now follows the same borrowed-connection rule for implicit feedback
  telemetry, and SQLite atlas storage has explicit regression coverage for the
  same ownership contract. Lite search storage now follows the same pattern:
  FTS and vector stores close owned standalone connections, preserve borrowed
  graph DB connections, and hybrid search delegates close to its component
  stores/providers. A shared borrowed-connection contract test now covers
  atlas, consolidation, conversation, evaluation, feedback, FTS, and vector
  storage so future close-path edits keep the lite shared-DB runtime intact.
  Runtime entrypoints now use `server/engram/storage/bootstrap.py` for that
  same shared-DB contract. REST startup creates atlas, consolidation,
  evaluation, and conversation stores through that module; MCP startup,
  lifecycle CLI, evaluation CLI, and projected/consolidated smoke share its
  consolidation/evaluation store factories. Search-index and lite companion
  store initialization use `initialize_*_for_graph()` helpers instead of
  repeating `graph_store._db` checks. MCP consolidation trigger fallback,
  lifecycle summary fallback reads, and graph-health SQLite metrics also route
  borrowed consolidation-store creation and private DB probing through the same
  bootstrap module.
  `GraphManager.close_runtime_resources()` now uses the same module's
  `close_if_supported()` helper so MCP and REST shutdown close owned runtime
  stores through the manager facade instead of reaching into private manager
  fields. The same bootstrap module now owns `stop_if_supported()` for worker,
  scheduler, subscriber, and pressure-accumulator shutdown.
  Background worker startup now has the same explicit dependency shape:
  `server/engram/ingestion/worker_runtime.py` defines
  `EpisodeWorkerRuntimeStores`, REST/MCP startup pass graph, activation, and
  search stores into `EpisodeWorker`, and `GraphManager` exposes
  `get_episode_worker_runtime_stores()` only as a compatibility accessor for
  direct worker construction.
  Adjacent auto-capture turn batching now has its own Cue-stage helper too:
  `server/engram/ingestion/worker_batching.py` owns primary content merge,
  primary cue rebuild/re-index, and merged-away cue retirement/re-index while
  `EpisodeWorker` keeps queue consumption, deterministic scoring, and
  projection routing.
  Worker deterministic scoring now has its own helper too:
  `server/engram/ingestion/worker_scoring.py` owns heuristic scoring,
  multi-signal scorer access, goal boost lookup, and projection-yield feedback;
  `EpisodeWorker` delegates scoring and outcome recording while keeping event
  routing and Project-stage dispatch.
  Worker projection routing now has its own helper too:
  `server/engram/ingestion/worker_routing.py` owns duplicate projection guards,
  system-discourse cue-only skips, skip/defer projection-state sync, and the
  boolean project-now routing contract while `EpisodeWorker` keeps event
  consumption, batch timing, and Project-stage dispatch.
- Native dashboard Recall gate fixture done: the no-bind native dashboard smoke
  verifies analyzer latency, trigger count, surfaced recall feedback, and
  threshold mapping from native-shaped evaluation payloads, then renders the
  Evaluation panel's Recall Gate card.
- Consolidation effect-rate signal done: the shared Consolidate-stage report
  derives overall and per-phase affected/processed rates, and REST/CLI/dashboard
  surfaces show that effect signal beside cycle and calibration metrics.
- Calibration-quality coverage gap done: the shared evaluation report now
  requires at least one quality-scored calibration phase before the Consolidate
  calibration-quality gate is considered covered; unscored snapshots remain
  visible but produce a coverage gap, a `needs_quality` calibration status, and
  an `attention` state for the Consolidate stage. Completed cycles without
  saved calibration snapshots also keep Consolidate in `attention` until the
  calibration evidence exists. Markdown/CLI output now spells the
  `needs_quality` case out as needing labeled decisions.
- Doctor diagnostics now list individual brain-loop smoke coverage gaps in
  Markdown output instead of only printing the gap count, keeping local operator
  checks tied to the same concrete evaluation gap names.
- The static group-scope contract now covers every production Python module for
  silent `group_id or "default"` fallback patterns, keeping public surfaces and
  future runtime services under the same one-brain-per-person guard as
  storage/retrieval/consolidation.
- Dashboard calibration-quality state done: the Evaluate panel now treats
  `needs_quality` as a distinct operator state and avoids rendering unscored
  calibration snapshots as fake accuracy/ECE evidence. The dashboard API client
  contract now also preserves that backend status during normalization.
- Evaluation Consolidate attention status done: recent cycle-level or
  phase-level consolidation issues now make the P3 Consolidate stage report
  `attention` instead of reporting `ready` while also carrying `error_count` or
  `latest_cycle.phase_issue`.
- Dashboard Consolidation warning state done: completed cycles with a
  `phase_issue` now render as warning state in the cycle/detail UI instead of
  using the same green visual treatment as clean completed cycles.
- Native dashboard warning fixture done: the no-bind PyO3 dashboard fixture now
  carries `phase_issue` through lifecycle, evaluation, consolidation
  status/history, and cycle detail payloads, then selects the cycle in the
  Consolidation panel to verify the warning detail state without a REST bind.
- Native REST/MCP surface parity done: the populated PyO3 smoke data now flows
  through REST lifecycle/evaluation/recall endpoints and MCP
  lifecycle/evaluation/recall tools in one focused integration test.
- Native REST consolidation-read parity done: REST `/api/consolidation/status`,
  `/api/consolidation/history`, and `/api/consolidation/cycle/{id}` read the
  completed native smoke cycle from the active Helix-backed consolidation store.
- Native REST consolidation-trigger parity done: REST
  `/api/consolidation/trigger?dry_run=true` starts a manual dry-run cycle for
  `native_brain` and that cycle appears completed in native history.
- Native REST notification parity done: REST `/api/knowledge/notifications` and
  `/api/knowledge/notifications/dismiss` keep pending/since/dismiss behavior
  scoped to the active PyO3 `native_brain`, including refusal to dismiss a
  wrong-group notification ID supplied in the active group's request.
- Dashboard WebSocket notification dismiss scope done: the
  `dismiss_notification` command now dismisses only within the connected
  tenant group, and native dashboard WebSocket parity remains green.
- Lite WebSocket tenant subscription scope done: the endpoint wording now
  matches the resolved tenant-group subscription behavior, and a non-default
  `tenant_brain` test proves wrong-group `default` events are ignored.
- OIDC missing-group fallback done: missing OIDC group claims now resolve to
  `AuthConfig.default_group_id`, preserving non-default PyO3/operator brain
  routing for OIDC-enabled sessions.
- Helix cue usefulness parity done: native `EpisodeCue` persistence now carries
  cue metadata, feedback counters, policy score, projection attempts, and
  timestamps, and native stats aggregate those fields instead of reporting
  zeroed cue feedback. The smoke gate now also requires surfaced cue feedback,
  the rebuilt native PyO3 route set includes the key-based cue update path, and
  a static schema guard prevents the server/native/generated cue schemas from
  drifting again.
- Native entity provenance parity done: native `Entity` persistence now carries
  source episode IDs, evidence counts, and evidence span bounds through create,
  read, and update, and the same static guard covers the generated PyO3 query
  source plus `delete_graph_embed_vector`.
- Native graph-embedding cleanup parity done: `HelixSearchIndex` now reads
  graph-embedding vectors from Helix `data` payloads, and a PyO3 native
  regression covers sync, read, clear, and read-empty behavior through the
  `delete_graph_embed_vector` route.
- Native graph-embedding phase replacement parity done: `GraphEmbedPhase` now
  has a PyO3 native regression proving full retrain clears stale native graph
  vectors before syncing the new trained vectors.
- Native open adjudication/evidence status parity done: Helix now exposes
  status-filtered evidence/adjudication queries, `HelixGraphStore` reads the
  same open statuses as lite mode, and a PyO3 native regression proves deferred
  evidence plus deferred/error adjudication requests round-trip through the
  preferred no-Docker backend.
- Native/lite open adjudication pressure visibility done: both strategic local
  paths now expose `adjudication_metrics` from stats, and the brain-loop report
  folds live open evidence/request counts into Consolidate attention state and
  Markdown output.
- Native artifact lookup parity done: REST project bootstrap and artifact
  search now feed MCP `bootstrap_project` and `search_artifacts` in the same
  populated PyO3 runtime.
- Native runtime-state parity done: REST `/api/knowledge/runtime` and MCP
  `get_runtime_state` report the same Helix mode and project artifact freshness
  over the populated native brain.
- Native MCP consolidation-control parity done: MCP `get_consolidation_status`
  and dry-run `trigger_consolidation` execute against the active PyO3 graph and
  return latest-cycle plus completed dry-run phase/summary data.
- MCP consolidation trigger active-store persistence done: `trigger_consolidation`
  now reuses the MCP runtime's active consolidation store for native Helix
  instead of falling back to SQLite-only audit storage, and native parity proves
  the triggered dry-run becomes the latest MCP consolidation status cycle.
- Evidence/adjudication service boundary done:
  `EvidenceAdjudicationService` now owns open adjudication presentation, stored
  evidence materialization, committed-id mapping, evidence storage-row
  serialization, and adjudication resolution, with `GraphManager` delegating the
  public compatibility APIs used by REST, MCP, projection execution, and
  consolidation phases.
- Native route parity done: REST `/api/knowledge/route` and MCP
  `route_question` share the same project-grounded evidence-plan behavior in
  the populated native runtime.
- Native MCP route auto-observe parity done: MCP `route_question` can
  auto-capture long tool-call input as a `tool_piggyback` episode plus cue in
  the active PyO3 `native_brain`.
- Native intention parity done: REST `/api/knowledge/intentions` and MCP
  `intend`/`list_intentions`/`dismiss_intention` create, list, create
  `refresh_context`/`after_consolidation` pinned-context intentions,
  acknowledge trigger/refresh metadata, soft-disable, hard-delete, and list
  disabled prospective-memory intentions in the active PyO3 `native_brain`.
  README, `skills/engram-memory/SKILL.md`, and the MCP prompt now document the
  same creation fields, refresh-context pinned query mode, and REST listing
  metadata.
- Native adjudication parity done: REST `/api/knowledge/adjudicate` and MCP
  `adjudicate_evidence` resolve pending native adjudication work items and
  persist rejected request state in the active PyO3 `native_brain`.
- Native forget parity done: REST `/api/knowledge/forget` and the MCP `forget`
  tool both soft-delete explicit native entities and clear activation in the
  active `native_brain`.
- Memory forgetting service boundary done:
  `MemoryForgettingService` now owns entity soft-delete plus activation clear
  and fact relationship invalidation, with `GraphManager.forget_entity` and
  `GraphManager.forget_fact` delegating as REST/MCP compatibility APIs.
- Native feedback parity done: REST `/api/knowledge/feedback` and the MCP
  `feedback` tool create `PREFERS`/`AVOIDS` preference edges under the active
  native `UserPreference` profile, and feedback event publishing now supports
  the real group-scoped sync event bus.
- Preference feedback service boundary done:
  `PreferenceFeedbackRecorder` now owns UserPreference profile creation,
  PREFERS/AVOIDS edge create/strengthen behavior, domain lookup, and
  `feedback.recorded` publication, with `GraphManager.record_explicit_feedback`
  delegating as the REST/MCP compatibility API.
- Native identity-core parity done: MCP `mark_identity_core` updates and clears
  the identity-core flag on an explicit entity in PyO3 Helix.
- Native context parity done: REST `/api/knowledge/context` and MCP
  `get_context` both assemble non-empty topic-biased context from the active
  `native_brain`.
- Native MCP notification piggyback parity done: MCP `get_context` surfaces
  `memory_notifications` for active PyO3 `native_brain` notifications while
  excluding wrong-group notifications.
- Native MCP auto-recall piggyback parity done: MCP `search_entities` can attach
  `recalled_context` from `recall_lite` over an explicit PyO3 native anchor
  entity in the active `native_brain`.
- Native entity/fact lookup parity done: REST `/api/entities/search`,
  REST `/api/knowledge/facts`, MCP `search_entities`, and MCP `search_facts`
  all find explicit native entities and relationships in the active
  `native_brain`.
- Direct entity/fact lookup service boundary done:
  `EntityFactLookupService` now owns read-only entity search, activation-score
  decoration, fact lookup/name resolution, and epistemic fact filtering, with
  `GraphManager` delegating the compatibility APIs used by REST, MCP, and
  internal callers.
- Agent context builder service boundary done:
  `MemoryContextBuilder` now owns tiered identity/project/recent/intention/
  pinned-context context assembly, deterministic briefing cache behavior, token
  budgeting, project-neighbor injection, and context access events, with
  `GraphManager.get_context()` and older helper APIs delegating as compatibility
  surfaces.
- Prospective-memory intention service boundary done:
  `ProspectiveMemoryService` now owns graph-embedded intention creation, v1
  flat-table fallback, TRIGGERED_BY edges, active filtering, soft/hard dismiss
  behavior, fire-count updates, metadata updates, and intention lifecycle events,
  with `GraphManager` delegating the compatibility APIs used by REST, MCP,
  consolidation finalization, and lifecycle summaries.
- Graph-state read service boundary done:
  `GraphStateService` now owns graph stats enrichment, top-activated entity
  materialization, active/dormant counts, recall/epistemic metrics attachment,
  entity-type filtering, optional relationship-edge expansion, entity profile
  resources, REST entity detail views, and dashboard graph neighborhood/
  temporal graph payloads, with `GraphManager` delegating the graph-state
  facades used by REST, MCP, lifecycle, evaluation, and dashboard consumers.
- Epistemic route service boundary done:
  `EpistemicRouteService` now owns memory-need analysis integration,
  graph-probe use, surface capability derivation, question-frame construction,
  evidence-plan construction, answer-contract application, route payload
  formatting, and route metrics recording, with `GraphManager.route_question()`
  and `_build_epistemic_route()` delegating for REST, MCP, and evidence
  gathering.
- Epistemic evidence service boundary done:
  `EpistemicEvidenceService` now owns route-guided memory/artifact/runtime source
  queries, project bootstrap before artifact reads, memory/artifact/runtime claim
  construction, claim-state summary generation, two-pass answer-contract
  reconciliation, stale-artifact miss detection, execution metrics, and
  `EpistemicBundle` assembly, with `GraphManager.gather_epistemic_evidence()`
  delegating for chat/API callers.
- Lightweight entity-probe recall service boundary done:
  `EntityProbeRecallService` now owns the `recall_lite`/`recall_medium` mention
  extraction, graph candidate probing, session-cache reuse, optional similarity
  rerank, top-fact rendering, freshness/confidence labeling, and token packing,
  with `GraphManager` delegating the MCP/chat compatibility APIs.
- Consolidation phase catalog boundary done:
  `build_consolidation_phases()` now owns concrete 16-phase runtime
  construction and registry-order validation, with `ConsolidationEngine`
  delegating phase assembly while retaining run-loop, cancellation, capability,
  event, and completion orchestration.
- Episode ingestion service boundary done:
  `EpisodeIngestionService` now owns one-shot store-then-project sequencing,
  proposal forwarding, attachment/conversation metadata forwarding, and the
  projection-failure-swallow compatibility behavior, with `GraphManager`
  delegating `ingest_episode()` for REST, MCP, benchmarks, and older tests.
- Offline replay service boundary done:
  `OfflineReplayService` now owns REST offline capture replay queue draining,
  active-brain store calls, short/dedup/error skips, and replay counts, with the
  route-facing `build_api_offline_replay_surface()` helper now owning service
  construction and the REST replay acknowledgement payload.
- Capture dedup service boundary done:
  `CaptureDedupCache` now owns auto-observe/replay content hashing, TTL duplicate
  checks, stale eviction, and max-entry cleanup, while the API keeps compatibility
  handles for existing dedup tests and monkeypatches.
- Native surface manifest done:
  `NativeSurface` entries now classify every public REST route and MCP tool,
  resource, and prompt against native parity evidence, with static tests that
  compare the manifest to the live FastAPI route table and MCP decorators.
  The verifier also checks runtime evidence against real parity helper/test
  function names and checks non-runtime evidence paths against the repo.
- GraphManager facade-boundary guard done:
  `tests/test_graph_manager_facade_boundaries.py` now statically verifies the
  core lifecycle facades delegate to extracted services instead of silently
  reabsorbing Capture, Project, Recall, runtime-state, context, graph-state,
  or epistemic route/evidence orchestration. The guard now also covers the
  remaining service-backed compatibility adapters for evidence adjudication,
  artifact search, decision materialization, lookup, forgetting, prospective
  memory, context, graph-state, and recall access/interaction updates. It now
  also scans runtime modules for direct `manager._*`, `graph_manager._*`, or
  `_manager._*` access outside `server/engram/graph_manager.py`.
- REST/MCP presenter-boundary guard done:
  `tests/test_public_surface_presenter_boundaries.py` now statically verifies
  REST and MCP observe/remember/recall/chat recall surfaces use the shared
  ingestion and retrieval presenters instead of local response formatting.
- Consolidation presenter-boundary guard done:
  `tests/test_consolidation_presenter_boundaries.py` now statically verifies
  REST, MCP, and CLI consolidation status/history/detail/trigger surfaces use
  the shared consolidation presenter for cycle and phase payloads.
- Native graph detail/resource parity done: REST entity detail/neighborhood and
  MCP `get_graph_state` plus entity profile/neighbor resources read the same
  explicit native relationship from PyO3 Helix.
- Native REST entity mutation parity done: REST `/api/entities/{id}` PATCH and
  DELETE update then soft-delete a standalone native entity, with detail/search
  and activation proving it no longer surfaces after deletion.
- Native MCP write parity done: MCP `remember` and `observe` write to PyO3
  Helix, return the shared lifecycle write contract, and persist episodes/cues
  under the active `native_brain`.
- Native REST observe parity done: REST `/api/knowledge/observe` writes to
  PyO3 Helix, preserves conversation date, and persists episode/cue state under
  the active `native_brain`.
- Native REST auto-observe parity done: REST `/api/knowledge/auto-observe`
  writes hook-style captures to PyO3 Helix and returns the duplicate-content
  `dedup_skipped` capture lifecycle contract for repeated content.
- Native attachment capture parity done: REST image/file observe endpoints and
  MCP image/file observe tools persist attachments and cues in the active
  PyO3 brain.
- Native repeated-reopen surface stability done: the REST lifecycle/evaluation/
  recall surfaces are now checked across three FastAPI startup/shutdown cycles
  over the same native data directory.
- Native bounded write/read load done: REST `remember`, repeated recall, and
  lifecycle/evaluation totals are now checked against a populated PyO3 brain
  without Docker or a bound socket.
- Native multi-batch ASGI load/reopen done: 12 additional REST `remember` writes
  across three batches survive native runtime shutdown/reopen with lifecycle,
  evaluation, and recall still coherent.
- Native operator load smoke done: `engram evaluate --smoke --mode helix` now
  accepts deterministic load/repeated-recall options and was verified against
  native PyO3 Helix with 123 projected episodes across two triage-only cycles
  and 30 recall checks.
- Native sustained recall-soak control done: the same operator smoke accepts
  `--smoke-min-duration-seconds` and `--smoke-pause-seconds` for hour-scale
  PyO3 native recall stress loops and records duration recall counts in the
  report.
- Native hour-scale recall soak done: the PyO3 operator smoke passed with
  3600.599 seconds of sustained Recall, 10,362 duration recall checks, 9
  captured/cued/projected episodes, one completed triage consolidation cycle,
  and no coverage gaps.
- Native current operator gate refreshed: `engram evaluate --smoke --mode helix`
  passed on a disposable PyO3 data directory with 3 projected episodes, one
  completed triage consolidation cycle, zero coverage gaps, and all six
  evaluation signals measured. Reopening the same native data with `engram
  evaluate --mode helix --require-evaluation-signals` passed, and `engram
  doctor --mode helix --skip-server` reported the lifecycle snapshot ready
  across Capture, Cue, Project, Recall, and Consolidate plus a disposable helix
  smoke with 6/6 evaluation signals measured.
- Consolidation finalization event contract done: completed-cycle lifecycle
  events now carry `finalization.refreshedPinnedContexts` from the post-cycle
  pinned-context refresh service.
- Public installer native path aligned: `bash -s -- helix` now reaches
  `engramctl setup --mode helix`, and that setup path writes local lifecycle
  config for `ENGRAM_MODE=helix` plus `ENGRAM_HELIX__TRANSPORT=native` instead
  of behaving like an interactive lite install or falling through to Docker
  full setup. Lite install docs now point users toward Helix native as the
  first growth path.
- Public Helix install now has a native support gate. The one-click installer
  requests `engram[local,native]` for `helix`, and `engramctl setup --mode helix`
  verifies native mode with `engram doctor --mode helix --skip-server
  --no-smoke --no-lifecycle`. If `helix_native` is not present in the package
  environment, setup fails with source-build remediation instead of writing a
  config that will fail later at store initialization.
- Local update is now native-aware too. `engramctl update` reads local
  `ENGRAM_MODE`, upgrades `engram[local,native]` for Helix-native installs, and
  reruns the native verification guard before restarting a previously running
  local server.
- Native transport guard tightened: `make up-native` and `make mcp-native` now
  force `ENGRAM_HELIX__TRANSPORT=native` even without a `NATIVE_DATA_DIR`, and
  explicit native mode resolution checks `helix_native` before returning Helix
  mode. Missing PyO3 support now fails at mode resolution with a build/install
  remediation instead of later during graph/search store initialization.
- Dashboard native smoke harness done: the opt-in Vitest smoke renders the
  Lifecycle panel from live native REST data when local bind access is approved.
- Dashboard native fixture smoke done: the default Vitest lane now validates
  native-shaped lifecycle/evaluation/recall/episode/consolidation API payloads
  and renders the Lifecycle, Consolidation, and Memory Feed panels without a
  REST bind, while the live smoke remains opt-in.
- Local doctor done: `engram doctor` wraps config, SQLite path, mode resolution,
  optional REST health, and brain-loop smoke checks into one operator-facing
  diagnostic report.
- Clean dashboard smoke done: a separate Chrome profile with extensions
  disabled loaded the Brain Loop and Evaluate views against the seeded lite
  backend, submitted recall and session-continuity labels, and refreshed the
  report without unrelated auto-observe writes entering the DB.
- The report combines:
  - cue coverage and cue-to-projection conversion
  - projection yield, backlog pressure, freshness/latency, and failure/dead-letter rate
  - recall precision/false recall/continuity lift
  - surfaced -> selected -> used telemetry
  - consolidation phase yield, effect rate, and calibration snapshots
- Next: keep using the shared projection-state helper for any new projection
  transition paths and continue extracting lifecycle orchestration boundaries
  without a wholesale `GraphManager` rewrite.

## Verification Notes

Current verification from this audit pass:

- `cd server && uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_cycle_context.py tests/test_triage_phase.py
  tests/test_rework_integration.py -q`: 41 passed, 2 skipped.
- `cd server && uv run pytest -m "not requires_docker and not requires_helix"
  -q`: 2498 passed, 43 skipped, 236 deselected, with existing warnings.
- `cd server && uv run ruff check engram/extraction/resolver.py
  tests/test_auto_observe.py tests/test_benchmark.py
  tests/test_consolidation_merge.py tests/test_evidence_storage.py
  tests/test_graph_health.py tests/test_episode_retrieval.py
  tests/test_microglia.py tests/test_typed_edge_weights.py
  tests/test_extraction_factory.py tests/test_mcp_tools.py`: passed.
- `cd dashboard && pnpm test -- --run src/test/ConsolidationPanel.test.tsx`:
  8 passed. Existing React `act(...)` warnings remain in the loading-state test.
- `cd dashboard && pnpm run build`: passed. Existing Vite large chunk warning
  remains, especially around `three-core`.
- `cd server && uv run pytest tests/test_recall_presenter.py
  tests/test_knowledge_api.py::TestRecall
  tests/test_knowledge_api.py::TestChatRecallHelpers::test_execute_tool_recall_formats_cue_episode
  tests/test_autorecall.py::TestRecallSetsLastTime -q`: 15 passed. Existing
  `datetime.utcnow()` deprecation warnings remain in autorecall fixtures.
- `cd server && uv run ruff check engram/retrieval/presenter.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_recall_presenter.py
  tests/test_knowledge_api.py tests/test_autorecall.py`: passed.
- `cd server && uv run pytest tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py::TestObserve tests/test_knowledge_api.py::TestRemember
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_autorecall.py::TestObserveWithAutoRecall
  tests/test_autorecall.py::TestRememberWithAutoRecall -q`: 17 passed, 2 skipped.
- `cd server && uv run pytest
  tests/test_auto_observe.py::test_auto_observe_endpoint
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_auto_observe.py::test_auto_observe_short_content_skipped -q`: 3 passed.
- `cd server && uv run ruff check engram/ingestion/presenter.py
  engram/api/knowledge.py engram/mcp/server.py tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py tests/test_mcp_tools.py tests/test_autorecall.py`:
  passed.
- `cd server && uv run pytest
  tests/test_gnn.py::TestGraphSAGEInference::test_output_normalized -q`:
  1 passed.
- `cd server && uv run ruff check tests/test_gnn.py`: passed.
- `cd server && uv run pytest tests/test_capture_service.py
  tests/test_episode_cues.py tests/test_cqrs_split.py::TestStoreEpisode
  tests/test_auto_observe.py::test_auto_observe_endpoint
  tests/test_auto_observe.py::test_auto_observe_dedup
  tests/test_auto_observe.py::test_auto_observe_short_content_skipped -q`:
  13 passed, 2 skipped.
- `cd server && uv run pytest tests/test_memory_write_presenter.py
  tests/test_knowledge_api.py::TestObserve tests/test_knowledge_api.py::TestRemember
  tests/test_mcp_tools.py::TestJSONResponses
  tests/test_autorecall.py::TestObserveWithAutoRecall
  tests/test_autorecall.py::TestRememberWithAutoRecall
  tests/test_capture_service.py -q`: 20 passed, 2 skipped.
- `cd server && uv run pytest
  tests/test_mcp_tools.py::TestSearchFacts::test_search_hides_epistemic_facts_by_default
  tests/test_mcp_tools.py::TestSearchFacts::test_question_form_does_not_materialize_decision_entity
  tests/test_mcp_tools.py::TestSearchFacts::test_explicit_commitment_materializes_decision_entity
  -q`: 3 passed.
- `cd server && uv run ruff check engram/ingestion/capture_service.py
  engram/graph_manager.py tests/test_capture_service.py tests/test_episode_cues.py
  tests/test_cqrs_split.py tests/test_auto_observe.py`: passed.
- `cd server && uv run pytest tests/test_projection_state_sync.py
  tests/test_capture_service.py tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_cqrs_split.py::TestProjectEpisode
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_formats_cue_results_and_tracks_hits
  tests/test_episode_retrieval.py::TestGraphManagerRecallEpisodes::test_recall_promotes_hot_cue_to_scheduled_projection
  tests/test_recall_feedback.py -q`: 60 passed, 6 skipped.
- `cd server && uv run ruff check engram/ingestion/projection_state.py
  engram/ingestion/capture_service.py engram/graph_manager.py engram/worker.py
  engram/consolidation/phases/triage.py engram/consolidation/phases/replay.py
  tests/test_projection_state_sync.py tests/test_capture_service.py
  tests/test_episode_worker.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py tests/test_cqrs_split.py
  tests/test_episode_retrieval.py tests/test_recall_feedback.py`: passed.
- `cd server && uv run pytest tests/test_episode_retrieval.py
  tests/test_recall_feedback.py tests/test_projection_state_sync.py
  tests/test_api_endpoints.py::TestLifecycleSummary -q`: 32 passed, 8 skipped,
  with existing AsyncMock coroutine warnings in `test_episode_retrieval.py`.
- `cd server && uv run ruff check engram/graph_manager.py
  engram/ingestion/projection_state.py tests/test_episode_retrieval.py
  tests/test_recall_feedback.py tests/test_projection_state_sync.py`: passed.
- `cd server && uv run pytest tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py -q`: 22 passed.
- `cd server && uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_cycle_context.py
  tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`: 118 passed, 22 skipped, with existing
  `datetime.utcnow()` warnings.
- `cd server && uv run ruff check engram/consolidation/engine.py
  engram/consolidation/phase_runner.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`: passed.
- `cd server && uv run pytest tests/test_consolidation_events.py
  tests/test_consolidation_engine.py tests/test_consolidation_phase_runner.py
  -q`: 25 passed.
- `cd server && uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_cycle_context.py tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`: 121 passed, 22 skipped, with existing
  `datetime.utcnow()` warnings.
- `cd server && uv run ruff check engram/consolidation/engine.py
  engram/consolidation/events.py engram/consolidation/phase_runner.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`: passed.
- `cd server && uv run pytest tests/test_consolidation_learning.py
  tests/test_consolidation_engine.py tests/test_consolidation_events.py
  tests/test_consolidation_phase_runner.py -q`: 28 passed.
- `cd server && uv run pytest tests/test_consolidation_engine.py
  tests/test_consolidation_learning.py tests/test_consolidation_events.py
  tests/test_consolidation_phase_runner.py tests/test_consolidation_cycle_context.py
  tests/test_triage_phase.py
  tests/test_consolidation_replay.py::TestReplayDeferredExtraction
  tests/test_rework_integration.py tests/test_consolidation_merge.py
  tests/test_consolidation_infer.py tests/test_evidence_adjudication.py
  tests/test_graph_embed_phase.py -q`: 124 passed, 22 skipped, with existing
  `datetime.utcnow()` warnings.
- `cd server && uv run ruff check engram/consolidation/engine.py
  engram/consolidation/learning.py engram/consolidation/events.py
  engram/consolidation/phase_runner.py tests/test_consolidation_learning.py
  tests/test_consolidation_events.py tests/test_consolidation_phase_runner.py
  tests/test_consolidation_engine.py`: passed.
- `cd dashboard && pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx`: 20 passed, with existing React
  `act(...)` and SVG casing warnings.
- `cd dashboard && pnpm test -- --run`: 191 passed, with existing React
  `act(...)`, canvas, and SVG casing warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `cd dashboard && pnpm test -- --run src/test/components.test.tsx`: 42 passed,
  with existing React `act(...)` and jsdom canvas warnings.
- `cd dashboard && pnpm test -- --run`: 200 passed, with existing React
  `act(...)`, SVG casing, and jsdom canvas warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `cd dashboard && pnpm exec vite --host 127.0.0.1 --clearScreen false`:
  Chrome opened the Evaluate route and showed the manual `Load Evaluation`
  fallback without first-viewport overlap. The backend was not running, so API
  500/proxy failures were expected for this UI-only smoke.
- `cd server && uv run pytest tests/test_mcp_tools.py::TestJSONResponses
  tests/test_api_endpoints.py::TestEvaluation tests/test_brain_loop_report.py
  tests/test_evaluation_store.py -q`: 13 passed, 2 skipped, with existing
  `datetime.utcnow()` warnings.
- `cd server && uv run ruff check engram/mcp/server.py
  engram/api/evaluation.py engram/evaluation/presenter.py
  engram/evaluation/__init__.py tests/test_mcp_tools.py
  tests/test_api_endpoints.py tests/test_brain_loop_report.py
  tests/test_evaluation_store.py`: passed.
- `ENGRAM_MODE=lite
  ENGRAM_SQLITE__PATH=/private/tmp/engram-live-eval-smoke.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0 uv run engram serve
  --host 127.0.0.1 --port 8100`, plus `curl` calls to seed one observed
  episode, one recall evaluation label, one session-continuity label, and
  `GET /api/evaluation/brain-loop/report`: passed. The report returned 1
  episode, cue coverage 100%, Recall `measured`, memory-need precision 100%,
  false recall 33.3%, useful packet rate 66.7%, Continuity `measured`, session
  continuity lift 0.5, open-loop recovery 100%, and expected coverage gaps for
  unprojected/unconsolidated data.
- `cd dashboard && pnpm exec vite --host 127.0.0.1 --clearScreen false` against
  the running API: Chrome opened the Evaluate drilldown at
  `http://127.0.0.1:5173/`, rendered the live report values, and showed no
  obvious first-viewport overlap.
- `cd dashboard && pnpm test -- --run src/test/apiClient.test.ts
  src/test/store.test.ts src/test/components.test.tsx`: 84 passed, with
  existing React `act(...)` and jsdom canvas warnings.
- `cd dashboard && pnpm test -- --run`: 205 passed, with existing React
  `act(...)`, SVG casing, and jsdom canvas warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `ENGRAM_MODE=lite
  ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-label-smoke.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0 uv run engram serve
  --host 127.0.0.1 --port 8100`, plus
  `cd dashboard && pnpm exec vite --host 127.0.0.1 --clearScreen false`: Chrome
  submitted a Recall label through the Evaluate UI, the form reached `stored`,
  and the report refreshed to Recall `measured` with 1 label. Direct REST calls
  on the same backend stored recall and session-continuity labels, and
  `GET /api/evaluation/brain-loop/report` returned Recall sample count 2,
  memory-need precision 100%, useful packet rate 66.7%, false recall 33.3%,
  Continuity sample count 1, session continuity lift 0.4, and open-loop
  recovery 100%.
- `cd server && uv run python scripts/projected_consolidated_smoke.py`: passed.
  The report returned 3 captured/cued episodes, 3 projected episodes, 2 linked
  entities, Recall and Continuity `measured`, 1 completed triage consolidation
  cycle, 1 measured calibration snapshot, and no coverage gaps.
- `cd server && uv run python -m engram evaluate --smoke --group-id
  operator_brain --format json`: passed with the same no-gap
  projected/consolidated report contract and `group_id` set to
  `operator_brain`.
- `cd server && ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-config.db uv run
  python -m engram doctor --mode lite --skip-server --format json`: passed.
  Config, SQLite parent, mode resolution, and brain-loop smoke checks were
  `pass`; server was skipped.
- `cd server && ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-config.db uv run
  python -m engram doctor --mode lite --skip-server --no-smoke --format
  markdown`: passed and rendered the Markdown check list.
- `cd server && uv run pytest tests/test_doctor.py -q`: 3 passed.
- `cd server && uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_brain_loop_report.py tests/test_evaluation_store.py -q`: 7 passed.
- `cd server && uv run ruff check engram/doctor.py engram/evaluation/cli.py
  engram/evaluation/smoke.py engram/evaluation/__init__.py engram/__main__.py
  scripts/projected_consolidated_smoke.py tests/test_doctor.py
  tests/test_projected_consolidated_smoke.py`: passed.
- `cd server && uv run pytest tests/test_cli_main.py tests/test_lifecycle_cli.py
  tests/test_doctor.py tests/test_setup.py -q`: 28 passed.
- `cd server && uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_lifecycle_cli.py tests/test_doctor.py -q`: 16 passed.
- `cd server && uv run pytest tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_evaluation_report_reads_active_consolidation_store
  -q`: 3 passed.
- `cd server && uv run pytest tests/test_helix_stats.py
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_shutdown_closes_runtime_resources
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_evaluation_report_uses_saved_samples
  -q`: 6 passed.
- `cd server && uv run ruff check engram/storage/helix/graph.py
  engram/storage/helix/search.py engram/storage/helix/atlas.py
  engram/storage/helix/consolidation.py
  engram/storage/helix/conversations.py engram/storage/factory.py
  engram/main.py engram/mcp/server.py tests/test_helix_stats.py
  tests/test_mcp_tools.py`: passed.
- `cd server && ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native
  ENGRAM_HELIX__DATA_DIR=/private/tmp/engram-native-path-eval-data uv run
  python -m engram evaluate --mode helix --no-saved-samples --format json`:
  passed. The command initialized `helix_native` and returned the expected
  empty local report contract without Docker.
- `cd server && uv run python -m engram evaluate --smoke --mode helix
  --sqlite-path /private/tmp/engram-native-smoke-labels-20260512.db
  --helix-data-dir /private/tmp/engram-native-smoke-data-20260512 --replace
  --group-id native_brain --format json`: passed. The command initialized
  `helix_native`, captured/cued/projected 3 episodes, persisted 1 completed
  triage consolidation cycle, saved a measured calibration snapshot and
  recall/continuity labels, and returned no coverage gaps.
- `cd server && uv run python -m engram evaluate --mode helix --sqlite-path
  /private/tmp/engram-native-smoke-labels-20260512.db --helix-data-dir
  /private/tmp/engram-native-smoke-data-20260512 --group-id native_brain
  --format json`: passed. Reopening the same native data preserved 3 episodes,
  3 cues, 3 projected memories, linked-entity projection yield, 1 consolidation
  cycle, measured recall/continuity labels, and no coverage gaps.
- `cd server && uv run engram evaluate --smoke --mode helix --helix-data-dir
  /private/tmp/engram-native-goal-20260519-data --sqlite-path
  /private/tmp/engram-native-goal-20260519-labels.db --replace --format json`:
  passed. The command initialized `helix_native`, projected 3 episodes,
  completed 1 triage consolidation cycle, returned no coverage gaps, and
  measured all 6 required evaluation signals.
- `cd server && uv run engram evaluate --mode helix --helix-data-dir
  /private/tmp/engram-native-goal-20260519-data --sqlite-path
  /private/tmp/engram-native-goal-20260519-labels.db
  --require-evaluation-signals --format json`: passed. Reopening the same
  native data and label store preserved 3 episodes, 3 cues, 3 projected
  memories, 1 consolidation cycle, zero coverage gaps, and all 6 evaluation
  signals measured.
- `cd server && uv run engram doctor --mode helix --helix-data-dir
  /private/tmp/engram-native-goal-20260519-data --skip-server --format json`:
  passed. Doctor loaded a ready Capture -> Cue -> Project -> Recall ->
  Consolidate lifecycle snapshot from the native data directory, intentionally
  skipped only the REST server check, and passed a disposable helix brain-loop
  smoke with 6/6 evaluation signals measured.
- `cd server && uv run pytest tests/test_projected_consolidated_smoke.py
  tests/test_helix_stats.py tests/test_brain_loop_report.py
  tests/test_lifecycle_cli.py tests/test_doctor.py -q`: 20 passed.
- `cd server && uv run ruff check engram/evaluation/cli.py
  engram/evaluation/smoke.py engram/storage/helix/graph.py
  engram/storage/helix/native_transport.py engram/storage/sqlite/graph.py
  tests/test_projected_consolidated_smoke.py tests/test_helix_stats.py
  tests/test_doctor.py`: passed.

Live browser caveat:

- The Chrome smoke environment emitted a large stream of
  `/api/knowledge/auto-observe` requests during the UI smoke. That traffic made
  the Continuity UI submit fail after the page reported offline, while direct
  REST continuity writes still succeeded. Future live dashboard smokes should
  use a clean browser context and set
  `ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false`.
- Clean dashboard smoke rerun:
  `ENGRAM_MODE=lite ENGRAM_SQLITE__PATH=/private/tmp/engram-dashboard-clean-smoke-disabled-20260512.db
  ENGRAM_EMBEDDING__PROVIDER=noop ENGRAM_MCP_ENABLED=0
  ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false uv run engram serve --host
  127.0.0.1 --port 8100` plus
  `pnpm exec vite --host 127.0.0.1 --clearScreen false`.
  - Result: Brain Loop rendered 3 episodes, 3 cues, 3 projected memories, 1
    consolidation cycle, and all five lifecycle stage cards. Evaluate rendered
    the shared brain-loop report, submitted one Recall label and one Continuity
    label through the UI, and refreshed to Recall sample count 2 / Continuity
    sample count 2. `GET /api/evaluation/brain-loop/report` returned no
    coverage gaps, Recall precision 100%, useful packet rate 60%, Continuity
    lift 0.8, open-loop recovery 100%, and temporal correctness 100%.
  - Guard check: external `/api/knowledge/auto-observe` calls still arrived, but
    the endpoint returned skipped `reason=disabled`; lifecycle summary stayed at
    3 seeded episodes.
- `cd server && uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  -q`: 3 passed.
- `cd server && uv run ruff check engram/lifecycle_summary.py
  engram/api/lifecycle.py engram/mcp/server.py tests/test_mcp_tools.py`:
  passed.
- `cd server && uv run pytest tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  -q`: 5 passed.
- `cd server && uv run ruff check engram/lifecycle_cli.py
  engram/lifecycle_summary.py engram/api/lifecycle.py engram/mcp/server.py
  engram/__main__.py tests/test_lifecycle_cli.py tests/test_mcp_tools.py`:
  passed.
- `cd server && ENGRAM_SQLITE__PATH=/private/tmp/engram-lifecycle-cli-smoke-20260512.db
  uv run python -m engram lifecycle --format json --episodes 1 --cycles 1`:
  passed outside the sandbox after the first sandboxed attempt hit the existing
  `uv` cache permission boundary.
- `cd server && uv run pytest tests/test_doctor.py tests/test_lifecycle_cli.py
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_mcp_tools.py::TestJSONResponses::test_mcp_get_lifecycle_summary_uses_shared_contract
  -q`: 9 passed.
- `cd server && uv run ruff check engram/doctor.py engram/lifecycle_cli.py
  engram/lifecycle_summary.py engram/api/lifecycle.py engram/mcp/server.py
  engram/__main__.py tests/test_doctor.py tests/test_lifecycle_cli.py
  tests/test_mcp_tools.py`: passed.
- `cd server && ENGRAM_SQLITE__PATH=/private/tmp/engram-doctor-lifecycle-config-20260512.db
  uv run python -m engram doctor --mode lite --skip-server --no-smoke --format
  json`: passed outside the sandbox after the first sandboxed attempt hit the
  existing `uv` cache permission boundary.
- README/docs/skill text audit for old phase-count and tool-count public wording:
  no stale public phase/tool count matches.
- `rg "/api/lifecycle/summary|get_lifecycle_summary|/api/evaluation/brain-loop/report"
  README.md skills/engram-memory/SKILL.md`: README and the Engram skill expose
  the lifecycle/evaluation contracts.
- `git diff --check`: passed.
- `cd dashboard && pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx`: 21 passed, with existing React
  `act(...)` and SVG casing warnings.
- `cd dashboard && pnpm test -- --run`: 192 passed, with existing React
  `act(...)`, canvas, and SVG casing warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `cd dashboard && pnpm exec vite --host 127.0.0.1`: Brain Loop rendered in
  Chrome with all five clickable stage cards visible, and clicking Capture
  opened the Feed drilldown. The backend was not running, so API 500/proxy
  failures were expected for this UI-only smoke.
- `cd dashboard && pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/components.test.tsx src/test/mediumComponents.test.tsx
  src/test/store.test.ts`: 93 passed, with existing React `act(...)`, canvas,
  and SVG casing warnings.
- `cd dashboard && pnpm test -- --run`: 195 passed, with existing React
  `act(...)`, canvas, and SVG casing warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `cd dashboard && pnpm exec vite --host 127.0.0.1 --clearScreen false`: Chrome
  reached the Stats drilldown from a lifecycle card. The backend was not
  running, so the Stats panel had no live data and API 500/proxy failures were
  expected for this UI-only smoke.
- `cd dashboard && pnpm test -- --run src/test/components.test.tsx
  src/test/LifecyclePanel.test.tsx`: 42 passed, with existing React `act(...)`
  and canvas warnings.
- `cd dashboard && pnpm test -- --run`: 196 passed, with existing React
  `act(...)`, canvas, and SVG casing warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.
- `cd dashboard && pnpm exec vite --host 127.0.0.1 --clearScreen false`: Chrome
  opened Knowledge from the Recall card and showed the focused
  `Recall Context` strip with an empty active-entity state. The backend was not
  running, so API 500/proxy failures were expected for this UI-only smoke.
- `cd dashboard && pnpm exec vite --host 127.0.0.1`: visual smoke passed in
  Safari for the Brain Loop first viewport. The backend was not running, so the
  visible API 500 was expected and this did not verify live API data.
- `cd server && uv run pytest tests/test_api_endpoints.py::TestLifecycleSummary
  -q`: 2 passed.
- `cd server && uv run pytest tests/test_api_endpoints.py::TestStats
  tests/test_api_endpoints.py::TestLifecycleSummary
  tests/test_api_endpoints.py::TestEpisodes -q`: 14 passed.
- `cd server && uv run pytest tests/test_api_endpoints.py -q`: 35 passed.
- `cd server && uv run ruff check engram/api/lifecycle.py engram/main.py
  tests/test_api_endpoints.py`: passed.
- `cd dashboard && pnpm test -- --run src/test/LifecyclePanel.test.tsx
  src/test/mediumComponents.test.tsx src/test/store.test.ts
  src/test/useWebSocket.test.ts src/test/apiClient.test.ts`: 61 passed, with
  existing React `act(...)` and SVG casing warnings.
- `cd dashboard && pnpm test -- --run`: 191 passed, with existing React
  `act(...)`, canvas, and SVG casing warnings.
- `cd dashboard && pnpm run build`: passed, with the existing Vite large chunk
  warning.

During this pass, the lite backend gate initially exposed 52 Helix connection
errors leaking through `not requires_helix` and 4 non-Helix failures. The Helix
tests are now marked/deselected, stale test expectations were updated, and a
numeric-identifier validation bug was fixed so code-like IDs can reach the
identifier normalization path. A later lite-gate run exposed one unrelated
random GraphSAGE normalization test failure; the test now uses deterministic
nonzero activations and the lite gate passes again. The capture/cue extraction
increased the lite gate to 2382 passing tests, the projection/cue-state sync
slice increased it to 2385 passing tests, the consolidation lifecycle/runner
slices brought it to 2396 passing tests, and the consolidation event publisher
slice brought it to 2399 passing tests. The consolidation learning service
slice brought it to 2402 passing tests with the same 41 skips and 230
deselections. The native surface/load slice plus backend-wide
`get_episode_entities(group_id=...)` contract alignment brought the lite gate
to 2443 passing tests with 41 skips and 234 deselections. The recall raw result
builder extraction brought it to 2446 passing tests with the same 41 skips and
234 deselections. The recall episode traversal extraction brought it to 2449
passing tests with the same 41 skips and 234 deselections. The recall near-miss
helper extraction brought it to 2453 passing tests with the same 41 skips and
234 deselections. The recall priming updater extraction brought it to 2457
passing tests with the same 41 skips and 234 deselections. The recall
confidence applier extraction brought it to 2459 passing tests with the same 41
skips and 234 deselections. The recall conversation fingerprint recorder
extraction brought it to 2461 passing tests with the same 41 skips and 234
deselections. Native doctor smoke-mode alignment brought it to 2462 passing
tests with the same 41 skips and 234 deselections. Explicit native data-dir
alignment for lifecycle and doctor brought it to 2464 passing tests with the
same 41 skips and 234 deselections. Runtime startup native data-dir alignment
for REST and MCP brought it to 2466 passing tests with the same 41 skips and
234 deselections. The Recall post-processor and Recall service extraction
refresh brought the broad non-Docker/non-Helix backend gate to 2493 passing
tests with 43 skips and 234 deselections. The native atlas/WebSocket,
conversation update/delete, MCP auto-recall, consolidation-trigger, MCP prompt,
admin benchmark-loader, and episode-list filter/cursor parity slices brought
that gate to 2501 passing tests with 43 skips and 236 deselections; the warning
count was 11,474 before benchmark-corpus UTC cleanup. Replacing
`datetime.utcnow()` / `datetime.utcfromtimestamp()` in
`engram.benchmark.corpus` with Engram's naive UTC helper and timezone-aware
timestamp conversion preserved the same 2501/43/236 result and reduced the
broad warning count to 500. A second production UTC-helper pass moved
prospective-memory expiry/cooldown checks, temporal hint defaults,
graph-embedding storage timestamps, and dream-association TTLs onto the same
helpers; the broad gate still passes at 2501/43/236 and now reports 462
warnings. Updating the schema-formation test entity factory to use the same
helper removed the largest remaining fixture warning block, keeping the broad
gate at 2501/43/236 and reducing the count to 106 warnings. Remaining warnings
were test-fixture datetime uses plus the known async-mark warnings. Cleaning
the next replay/prune/maturation/microglia/predicate fixture cluster kept the
broad gate at 2501/43/236 and reduced the remaining warning count to 20. The
final consolidation graph, MCP facts, prospective memory, structural merge,
structure-aware embedding, and proactive recall cleanup removed the remaining
datetime and async-mark warnings; the broad non-Docker/non-Helix backend gate
now passes at 2501/43/236 with zero warnings reported.
Later lifecycle/native parity, evaluation, and consolidation contract slices
expanded that same broad local gate to 2580 passing tests with 43 skips and 236
external-service deselections after requested-phase validation, phase-registry
runtime validation, the backend/dashboard phase-order contract, the
consolidation CLI/MCP/REST failed-cycle operator contracts, the shared
consolidation presenter extraction, the projection-plan group metadata
contract, Redis event bridge group routing, and evaluation CLI default-group
alignment.
Native graph-embedding cleanup/replacement, native open evidence/adjudication
status queue parity, and native/lite open adjudication stats/report visibility
expanded the same broad non-Docker/non-Helix backend gate to 2630 passing tests
with 43 skips and 236 external-service deselections. The latest dashboard gate
also passes at 214 tests with 1 skipped test, and the dashboard build still
passes with only the existing large-chunk warning.
The MCP consolidation trigger active-store contract expanded the same backend
gate to 2631 passing tests with 43 skips and 236 external-service deselections.
The evidence/adjudication service-boundary extraction preserved that same broad
gate result: 2631 passing tests, 43 skips, and 236 external-service
deselections; the latest run completed in 112.33 seconds.
The preference feedback service-boundary extraction preserved the same broad
gate result: 2631 passing tests, 43 skips, and 236 external-service
deselections; the latest run completed in 112.67 seconds.
The memory forgetting service-boundary extraction preserved the same broad gate
result: 2631 passing tests, 43 skips, and 236 external-service deselections;
the latest run completed in 109.58 seconds.
The direct entity/fact lookup service-boundary extraction preserved the same
broad gate result: 2631 passing tests, 43 skips, and 236 external-service
deselections; the latest run completed in 110.70 seconds.
The agent context builder service-boundary extraction preserved the same broad
gate result: 2631 passing tests, 43 skips, and 236 external-service
deselections; the latest run completed in 108.87 seconds.
The prospective-memory intention service-boundary extraction preserved the same
broad gate result: 2631 passing tests, 43 skips, and 236 external-service
deselections; the latest run completed in 111.13 seconds.
The graph-state read service-boundary extraction preserved the same broad gate
result: 2631 passing tests, 43 skips, and 236 external-service deselections; the
latest run completed in 112.33 seconds.
The epistemic route service-boundary extraction preserved the same broad gate
result: 2631 passing tests, 43 skips, and 236 external-service deselections; the
latest run completed in 115.08 seconds.
The epistemic evidence service-boundary extraction preserved the same broad gate
result: 2631 passing tests, 43 skips, and 236 external-service deselections; the
latest run completed in 115.05 seconds.
The runtime-state service-boundary extraction preserved the same broad gate
result: 2631 passing tests, 43 skips, and 236 external-service deselections; the
latest run completed in 113.47 seconds.
The decision materializer service-boundary extraction preserved the broad gate
with its new focused regression included: 2632 passing tests, 43 skips, and 236
external-service deselections; the latest run completed in 111.64 seconds.
The consolidation cycle completion service-boundary extraction preserved the
broad gate with its new focused completion tests included: 2634 passing tests,
43 skips, and 236 external-service deselections; the latest run completed in
112.19 seconds.
The structure-aware entity indexer service-boundary extraction preserved the
broad gate with its new focused indexer test included: 2635 passing tests, 43
skips, and 236 external-service deselections; the latest run completed in 114.49
seconds.
The artifact search/read service-boundary extraction preserved the broad gate
with its new focused artifact service test included: 2636 passing tests, 43
skips, and 236 external-service deselections; the latest run completed in 113.89
seconds.
The project bootstrap service-boundary extraction preserved the broad gate:
2636 passing tests, 43 skips, and 236 external-service deselections; the latest
run completed in 114.68 seconds.
The lightweight entity-probe recall service-boundary extraction preserved the
broad gate with new `recall_medium` rerank coverage included: 2638 passing
tests, 43 skips, and 236 external-service deselections; the latest run
completed in 112.33 seconds.
The consolidation phase catalog extraction preserved the broad gate with its
new phase-catalog order regression included: 2639 passing tests, 43 skips, and
236 external-service deselections; the latest run completed in 115.19 seconds.
The episode ingestion service-boundary extraction preserved the broad gate with
new one-shot ingestion service tests included: 2641 passing tests, 43 skips, and
236 external-service deselections; the latest run completed in 112.58 seconds.
The offline replay service-boundary extraction preserved the broad gate with
new replay service tests included: 2643 passing tests, 43 skips, and 236
external-service deselections; the latest run completed in 112.22 seconds.
The capture dedup service-boundary extraction preserved the broad gate with new
dedup cache service tests included: 2646 passing tests, 43 skips, and 236
external-service deselections; the latest run completed in 112.42 seconds.
The native surface manifest slice added static route/tool coverage accounting
and preserved the native runtime evidence path. Focused manifest tests passed,
the main populated PyO3 REST/MCP parity smoke passed alongside the manifest,
and the broad non-Docker/non-Helix gate now passes with 2650 tests, 43 skips,
and 236 external-service deselections.
The GraphManager facade-boundary guard then added 12 static checks for the core
service delegates, and the broad non-Docker/non-Helix gate now passes with 2662
tests, 43 skips, and 236 external-service deselections.
The REST/MCP presenter-boundary guard added 12 more public-surface checks, and
the broad non-Docker/non-Helix gate now passes with 2674 tests, 43 skips, and
236 external-service deselections.
The consolidation presenter-boundary guard added 6 checks for REST/MCP/CLI
cycle/status/history/detail output, and the broad non-Docker/non-Helix gate now
passes with 2680 tests, 43 skips, and 236 external-service deselections.
The dashboard completion-readiness refresh then passed full Vitest with 214
tests and 1 skipped native-live test, and the production build passed with the
existing large-chunk warning.
The live native dashboard smoke then passed against a seeded PyO3 REST server
on `127.0.0.1:8102` after aligning app/auth default group to `native_brain`.
The initial temporary server exited cleanly, with residual native
`update_evidence` decode-error logs during shutdown consolidation.
Helix native evidence updates now normalize optional string payloads before
calling the PyO3 query layer, so `commit_reason=None` and `committed_id=None`
are sent as empty strings instead of JSON null. Focused regression coverage
matches the shutdown-noise shape from the live native dashboard smoke, and the
broad non-Docker/non-Helix gate then passed with 2681 tests, 43 skips, and 236
external-service deselections.
The patched live native dashboard smoke was re-run against the same seeded
PyO3 REST path and still passed. Shutdown no longer logged native
`update_evidence` decode errors; only the existing graph-embedding
not-persisted/too-few-entities warnings remained.

The native/default-group config contract is now explicit. When
`ENGRAM_DEFAULT_GROUP_ID` is non-default and `ENGRAM_AUTH__DEFAULT_GROUP_ID` is
omitted, `EngramConfig` carries the top-level brain ID into
`auth.default_group_id`, so unauthenticated REST follows the same one-brain
group without an extra auth env var. Explicit auth default-group overrides still
win, including an explicit `default`. Focused config tests cover constructor and
environment loading, README and Helix install docs now show the auth env var as
a commented override, and the broad non-Docker/non-Helix gate now passes with
2687 tests, 43 skips, and 236 external-service deselections.

The GraphManager facade-boundary guard has now been expanded from the core 12
facade checks to 61 static checks. It covers both the main lifecycle APIs and
the service-backed compatibility adapters that REST, MCP, projection execution,
consolidation, and older tests still call. Focused verification passed, and the
broad non-Docker/non-Helix gate now passes with 2725 tests, 43 skips, and 236
external-service deselections.

The first REST/MCP route-orchestration cleanup found the MCP
`mark_identity_core` tool writing directly through `manager._graph`. That
mutation now lives in `engram.retrieval.identity_core.IdentityCoreService`,
`GraphManager.mark_identity_core()` is the compatibility facade, and the MCP
tool calls the facade. Focused service/facade/public-surface verification
passed, and the broad non-Docker/non-Helix gate now passes with 2730 tests,
43 skips, and 236 external-service deselections.

The next REST/MCP route-orchestration cleanup moved MCP
`trigger_consolidation` out of route-local engine construction. The new
top-level `engram.consolidation_trigger.ConsolidationTriggerService` owns ad
hoc `ConsolidationEngine` construction/execution and graph-stats capture,
`GraphManager.trigger_consolidation_cycle()` is the compatibility facade, and
the consolidation trigger helper now owns active audit-store resolution plus the
lite shared-DB fallback. The MCP tool keeps JSON wrapping and session-state
lookup, then presents the returned cycle. Focused service, MCP trigger,
presenter-boundary, and public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2735 tests, 43 skips, and 236
external-service deselections. The latest route-facing store-resolution check
passed with 165 tests.

The following route-orchestration cleanup moved MCP entity graph resources out
of direct graph/activation reads. `GraphStateService` now owns entity profile
and one-hop neighbor resource views, `GraphManager.get_entity_profile()` and
`get_entity_neighbors()` are compatibility facades, and the MCP resources call
those facades. Focused service/facade/public-surface checks passed, and the
broad non-Docker/non-Helix gate now passes with 2742 tests, 43 skips, and 236
external-service deselections.

REST and MCP recall-need graph-probe helpers now use
`GraphManager.get_recall_need_graph_probe()` instead of constructing
`GraphProbe` from `manager._graph` and `_activation` in the transport layer.
Focused facade/public-surface checks passed, and the broad non-Docker/non-Helix
gate now passes with 2744 tests, 43 skips, and 236 external-service
deselections.

REST and MCP intention-list presentation now lives in
`ProspectiveMemoryService.list_intention_views()`. API and MCP handlers keep
their existing response shapes by calling `GraphManager.list_intention_views()`
instead of reading `manager._cfg` and `_activation` in the transport layer to
compute warmth. Focused service/facade/public-surface checks passed, and the
broad non-Docker/non-Helix gate now passes with 2750 tests, 43 skips, and 236
external-service deselections.

MCP `intend` now gets the effective default threshold through
`GraphManager.effective_intention_threshold()`, backed by
`ProspectiveMemoryService.effective_activation_threshold()`, instead of reading
`manager._cfg.prospective_activation_threshold` in the transport layer. Focused
facade/public-surface checks passed, and the broad non-Docker/non-Helix gate
passes with 2753 tests, 43 skips, and 236 external-service deselections.

REST/MCP live conversation context access now has a route-facing runtime
boundary. `engram.retrieval.context.ConversationRuntimeService` owns active
conversation context exposure, live-turn embedding lookup, turn counts, session
entity names, recent turns, and live-turn ingestion. `GraphManager` exposes
compatibility facades for those operations, so REST chat/route helpers and MCP
auto-recall/route helpers no longer read `manager._conv_context` or
`manager._search`. Focused service/facade/public-surface/API/MCP checks passed,
and the broad non-Docker/non-Helix gate now passes with 2775 tests, 43 skips,
and 236 external-service deselections.

The next route-orchestration cleanup moved public REST route policy out of
transport-local config reads. `engram.public_surface_policy.PublicSurfacePolicyService`
now owns explicit recall packet policy, chat tool recall interaction semantics,
memory-need graph-probe enablement, chat runtime flags, post-response recall
feedback enablement, chat retry safety-net enablement, and client-adjudication
response visibility. `GraphManager` exposes those as compatibility facades, so
REST remember/recall/chat helpers no longer read `manager._cfg`. MCP lifecycle
summary reads now also use `GraphManager.get_lifecycle_graph_store()` and
`get_activation_config()` instead of private manager graph/config fields, with
`GraphStateService` owning the lifecycle graph-store exposure. Focused
policy/API/lifecycle/facade/public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2797 tests, 43 skips, and 236
external-service deselections.

REST entity detail and mutation routes now have service-backed manager
boundaries. `GraphStateService.get_entity_detail()` owns the
`/api/entities/{id}` detail view, including activation fallback and fact
materialization, while `engram.retrieval.entity_mutation.EntityMutationService`
owns profile updates and soft-delete plus activation clearing. The
`/api/entities` route now preserves its search/detail/PATCH/DELETE response
shapes without reading `manager._graph`, `_activation`, or `_cfg`. Focused
service/API/facade/public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2809 tests, 43 skips, and 236
external-service deselections.

The REST admin benchmark-loader route now has a service boundary as well.
`engram.benchmark_loader.BenchmarkLoadService` owns benchmark corpus generation,
active `group_id` rewriting, and loading into the active graph, activation, and
search stores. `GraphManager.load_benchmark_corpus()` is the manager
compatibility facade, and `build_api_benchmark_load_surface()` is now the
route-facing helper, so `/api/admin/load-benchmark` preserves its payload
without calling the manager directly or reading `manager._graph`, `_activation`,
or `_search`. Focused service/facade/public-surface checks passed, and the
broad non-Docker/non-Helix gate now passes with 2812 tests, 43 skips, and 236
external-service deselections. The latest route-facing benchmark helper check
passed with 238 tests.

REST graph neighborhood and temporal graph endpoints now use the same
graph-state boundary. `GraphStateService` owns dashboard node/edge formatting,
activation fallback, neighborhood pruning, auto-center selection, and
point-in-time relationship traversal. `GraphManager.get_graph_neighborhood()`
and `get_temporal_graph()` are the route-facing compatibility facades, so
`/api/graph/neighborhood` and `/api/graph/at` preserve their JSON shapes
without reading `manager._graph`, `_activation`, or `_cfg` in the transport
layer. Route-facing helpers in `server/engram/retrieval/graph_state.py` now
also own manager dispatch, missing-entity payloads, temporal timestamp parsing,
and invalid-timestamp payloads for those REST routes. Focused graph-state/API/
facade/public-surface checks passed, and the broad non-Docker/non-Helix gate now
passes with 2821 tests, 43 skips, and 236 external-service deselections. The
latest route-facing graph-state surface check passed with 126 tests.
The entity-neighbor convenience route now uses that same route-facing helper
directly instead of importing and calling the graph route function, so
`/api/entities/{entity_id}/neighbors` no longer couples one public route to
another transport handler. Focused entity-neighbor, graph-state surface, and
public-surface checks passed with 149 tests.

REST atlas snapshot/history/region routes now have a route-facing atlas surface
boundary. `server/engram/retrieval/atlas_surface.py` owns atlas representation
metadata, snapshot region/bridge/stat serialization, history row shaping,
region/snapshot lookup error payloads, and atlas service dispatch. The graph
route now preserves tenant lookup, dependency lookup, logging, and HTTP wrapping
for `/api/graph/atlas`, `/api/graph/atlas/history`, and
`/api/graph/regions/{region_id}` without reassembling atlas payloads locally.
Focused atlas helper/API/public-surface checks passed with 140 tests.

MCP recall-response enrichment now has a route-facing state boundary.
`engram.retrieval.response_state.RecallResponseStateService` owns triggered
intention serialization, near-miss payload copying, recall-item access-count
lookup, and surprise-connection formatting. `GraphManager` exposes facades for
those transient views, so MCP `recall` and piggyback middleware no longer read
`manager._activation`, `_last_near_misses`, `_surprise_cache`, or
`_triggered_intentions` in the transport layer. Focused response-state/MCP/
facade/public-surface checks passed, the REST/MCP route-private scan is now
clean, and the broad non-Docker/non-Helix gate now passes with 2832 tests,
43 skips, and 236 external-service deselections.

REST dashboard stats now has the same graph-state boundary. `GraphStateService`
owns `/api/stats` top-activated camel-case formatting plus top-connected and
growth timeline reads, and `GraphManager.get_dashboard_stats()` is the
manager compatibility facade. The route-facing graph-state helper now owns the
REST dispatch shape, so the stats route preserves its dashboard JSON without
calling the manager method or reaching into the graph store directly. Focused
graph-state/API/facade/public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2835 tests, 43 skips, and 236
external-service deselections.

REST activation monitor reads now use the same graph-state boundary.
`GraphStateService` owns top-activation snapshot formatting and ACT-R decay
curve payload construction, and `GraphManager.get_activation_snapshot()` plus
`get_activation_curve()` are the manager compatibility facades. Route-facing
graph-state helpers now own the REST dispatch shape for snapshot reads and
curve missing-entity 404 payload/status mapping, so the activation route
preserves snapshot/curve JSON without reading app state, graph store,
activation store, config, or `compute_activation` directly in the transport
layer. Focused graph-state/API/facade/public-surface checks passed, the latest
activation response-surface gate passed with 150 tests, and the broad
non-Docker/non-Helix gate now passes with 2845 tests, 43 skips, and 236
external-service deselections.

REST episode dashboard reads now use `GraphStateService` too.
`GraphStateService.list_episode_summaries()` owns `/api/episodes` paginated
episode/cue payload construction, and `GraphManager.list_episode_summaries()`
is the manager compatibility facade. The route-facing graph-state helper now
owns the REST dispatch shape, so the route preserves source/status filters,
cursor pagination, projection fields, cue counters, timestamp formatting, and
item totals without calling the manager method, reading the graph store, or
formatting episode/cue state in the transport layer. Focused graph-state/API/
facade/public-surface checks passed, the route-local store/formatting scan is
clean, and the broad non-Docker/non-Helix gate now passes with 2848 tests, 43
skips, and 236 external-service deselections. The latest route-facing dashboard
read helper check covered stats, activation snapshot, activation curve, episode
list, and public-surface guards with 173 tests.

REST and MCP lifecycle summary reads now use a route-facing lifecycle service.
`LifecycleSummaryService` owns the call into the shared
`build_lifecycle_summary()` contract, and `GraphManager.get_lifecycle_summary()`
is the compatibility facade for `/api/lifecycle/summary` and MCP
`get_lifecycle_summary`. Both transports preserve the
`Capture -> Cue -> Project -> Recall -> Consolidate` payload without directly
passing graph/config facades or calling the builder from route code. Focused
API/MCP/facade/public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2851 tests, 43 skips, and 236
external-service deselections.

Dashboard WebSocket activation-monitor snapshots now use the existing
graph-state activation surface helper. The `subscribe.activation_monitor` loop
calls `build_api_activation_snapshot_surface()` and forwards its `topActivated`
payload instead of calling the manager directly, reading app-state
graph/activation/config stores, or recomputing activation in
`server/engram/api/websocket.py`. Focused WebSocket/auth/public-surface checks
passed, the socket-local activation-store scan is clean, and the broad
non-Docker/non-Helix gate now passes with 2853 tests, 43 skips, and 236
external-service deselections. The latest WebSocket helper guard passed with
160 tests.

REST and MCP notification surfacing now share a notification surface boundary.
`NotificationSurfaceService` owns REST notification list/dismiss payloads and
MCP `memory_notifications` piggyback payloads, so
`server/engram/api/knowledge.py` and `server/engram/mcp/server.py` no longer
read the notification store or format notification payloads directly. The REST
notification routes now use route-facing surface helpers for list/dismiss
response envelopes and missing-service fallbacks too. Focused notification,
piggyback, MCP, and public-surface checks passed, and the broad
non-Docker/non-Helix gate now passes with 2858 tests, 43 skips, and 236
external-service deselections. The latest focused response-envelope helper gate
passed with 169 tests.

The dashboard WebSocket notification-dismiss command now uses that same
notification surface boundary. `dismiss_notification` calls
`NotificationSurfaceService.dismiss_notifications()` with the connected brain
group instead of reading `_app_state["notification_store"]` directly from
`server/engram/api/websocket.py`. Focused WebSocket/auth/public-surface checks,
the widened notification/WebSocket/static suite, Ruff, and `git diff --check`
passed; the broad non-Docker/non-Helix gate now passes with 2859 tests, 43
skips, and 236 external-service deselections.

Dashboard WebSocket command/event payload shaping now has a route-facing helper
too. `server/engram/api/websocket_surface.py` owns event payload flattening,
`pong` responses, `resync` replay envelopes, activation snapshot WebSocket
envelopes, and connected-group notification dismiss dispatch. The WebSocket
route keeps auth, event-bus subscription task lifecycle, and JSON transport
only. Focused WebSocket-surface, WebSocket, auth, public-surface, and Ruff
checks passed with 176 tests, and the broad non-Docker/non-Helix gate now
passes with 3155 tests, 43 skips, and 236 external-service deselections.

The dashboard WebSocket auth setup now uses the existing API config dependency
too. `server/engram/api/websocket.py` calls `get_config().auth`, preserving the
previous `AuthConfig()` fallback if app config is unavailable, and the public
surface guard now asserts the socket route does not import `_app_state`
directly. Focused WebSocket/auth/public-surface checks and Ruff passed, and the
broad non-Docker/non-Helix gate now passes with 2861 tests, 43 skips, and 236
external-service deselections.

REST knowledge-chat rate limiting now uses the API dependency boundary as well.
`server/engram/api/knowledge.py` calls `get_rate_limiter()` before the chat
runtime policy and no longer imports `_app_state` directly.
`server/engram/retrieval/chat_runtime.py` now owns
`check_api_chat_rate_limit()`, so the route does not execute
`rate_limiter.check()` directly either. The public-surface guard now covers both
`knowledge.py` and `websocket.py` for direct app-state reads and rejects
route-local `rate_limiter.*` dispatch through the broader store/service guard.
Focused chat/public-surface checks, the widened knowledge/public-surface suite,
and Ruff passed; the broad non-Docker/non-Helix gate now passes with 2862 tests,
43 skips, and 236 external-service deselections.

REST health now uses the same dependency boundary pattern. `/health` calls
`get_graph_store()` for the graph probe, `get_config().default_group_id` for the
one-brain stats scope, and `get_mode()` for the response mode instead of
importing `_app_state` directly. The public-surface guard now covers
`health.py`, `knowledge.py`, and `websocket.py` for route-local app-state reads.
Focused health/public-surface checks, the widened API/public-surface suite, and
Ruff passed; the broad non-Docker/non-Helix gate now passes with 2864 tests, 43
skips, and 236 external-service deselections.

REST health response assembly now has a route-facing helper too.
`server/engram/api/health_surface.py` owns graph-store health probing,
default-brain stats checks, service status aggregation, and the public
`HealthResponse`; `/health` keeps dependency lookup and return wiring. The
public-surface guard now forbids `health_check()` from calling `get_stats`
directly. Focused health-surface, health endpoint, public-surface, and Ruff
checks passed with 165 tests, and the broad non-Docker/non-Helix gate now
passes with 3155 tests, 43 skips, and 236 external-service deselections.

The remaining `/health` dependency fallback is now out of the route too.
`server/engram/api/health_runtime.py` owns optional graph-store resolution and
default-group fallback before calling the shared health surface. `health_check()`
is now a one-call route wrapper, and the public-surface guard rejects direct
`get_graph_store()`, `get_config()`, `get_mode()`, `get_stats()`, or
`build_api_health_surface()` usage in the route. Focused health/runtime/static
checks passed with 191 tests.

REST consolidation trigger scheduling is now behind the consolidation trigger
boundary as well. `build_api_consolidation_trigger_response_surface()` owns the
running-cycle check, public trigger payload, and `BackgroundTasks.add_task()`
scheduling for `run_api_consolidation_cycle()`. `trigger_consolidation()` now
keeps tenant and engine lookup plus JSON wrapping, and the public-surface guard
rejects direct task scheduling or direct trigger/run helper usage in the route.
Focused consolidation-trigger/static checks passed with 198 tests.

REST consolidation status pressure/config selection now lives in the same
boundary. `build_api_consolidation_status_response_surface()` chooses the
activation config used for pressure reporting before delegating to the shared
status surface. `consolidation_status()` keeps tenant/dependency lookup plus
JSON wrapping, and the public-surface guard rejects direct activation-config
extraction or direct status-surface calls in the route. Focused
consolidation-trigger/static checks passed with 201 tests.

The remaining public REST/WebSocket route control flow is now explicitly
bounded. `tests/test_public_surface_presenter_boundaries.py` allows only chat
JSON-vs-stream response wrapping and WebSocket auth/session try/excepts inside
decorated FastAPI routes after atlas warning logging and response wrapping moved
behind `build_api_atlas_json_response()`. Any new route-local `if`, `try`, loop,
match, or expression branch must either be justified in the guard or moved to a
route-facing helper. The focused public-surface plus atlas suite passes with 197
tests after this guard.

That app-state guard now covers every API route module, not just the three
routes cleaned in this pass. `tests/test_public_surface_presenter_boundaries.py`
discovers `server/engram/api/*.py`, excludes only `__init__.py` and `deps.py`,
and fails if any route module imports `_app_state` directly. The route scan now
shows `_app_state` only in `server/engram/api/deps.py`. The latest broad
non-Docker/non-Helix gate now passes with 3316 tests, 43 skips, and 236
external-service deselections after the evaluation-signal CLI gate, doctor
readiness reporting, Python 3.13 event-loop test harness cleanup, and Helix
dashboard analytics date-stability fix, the shared companion-store bootstrap
follow-up, explicit notification/scheduler dependency cleanup, the public smoke
cue-feedback facade, and REST/MCP shutdown stop/close plus consolidation-helper
cleanup with static guards.

The REST evaluation report no longer reaches into consolidation engine private
store state or dispatches engine methods directly from the route.
`server/engram/evaluation/report_service.py` now exposes
`build_api_brain_loop_evaluation_surface()`, which loads recent cycles plus
calibration snapshots through `ConsolidationEngine.get_recent_evaluation_context()`
and then calls the shared report builder with REST snapshot labeling.
`server/engram/api/evaluation.py` only resolves dependencies and returns the
payload. The public-surface guard rejects route-local `engine._*` access
alongside `manager._*` access and now statically rejects direct REST API
`engine.*` dispatch. Focused evaluation/consolidation/static checks and Ruff
passed; a later route-orchestration broad gate now passes with 3213 tests, 43
skips, and 236 external-service deselections. The current broader gate passes
with 3316 tests after the doctor readiness, Helix analytics fixture,
companion-store bootstrap, notification/scheduler dependency, public smoke
feedback, and REST/MCP shutdown facade updates.

The same public-surface guard now covers direct REST route store/service method
dispatch. REST route modules may still resolve stores and services through the
dependency layer, but the route body must pass them into a surface/runtime
helper instead of calling `store.*`, `evaluation_store.*`, `conv_store.*`,
`graph_store.*`, `notification_surface.*`, or generic `service.*` methods
directly. This keeps Capture, Recall, evaluation, notification, and conversation
work inside named lifecycle helpers instead of drifting back into FastAPI
handlers. The same static suite now discovers every decorated REST API route and
requires a `PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES` entry for each one, so new
public routes cannot bypass the named boundary map. The focused public-surface
suite passes with 165 tests.

REST route functions are now also guarded by direct-await shape. The public API
routes may directly await route-facing helpers such as `build_*`,
`check_api_chat_rate_limit()`, `resolve_chat_conversation()`,
`resolve_dashboard_websocket_tenant()`,
`close_dashboard_websocket_auth_failure()`, `run_dashboard_websocket_session()`,
and `run_api_consolidation_cycle()`, but direct awaits of arbitrary imported
runtime functions will fail the public-surface suite. The focused public-surface
suite passes with 167 tests.

Dashboard WebSocket session orchestration is now out of the route-local nested
functions too. `run_dashboard_websocket_session()` owns event forwarding,
command handling, activation-monitor task lifecycle, bus subscription cleanup,
and WebSocket disconnect cancellation cleanup. `dashboard_ws()` keeps auth,
accept, dependency lookup, and one route-facing runtime call. The static guard
now also fails if a decorated API route handler defines nested functions, so new
transport handlers cannot hide lifecycle loops inside the route body. Focused
WebSocket, WebSocket-surface, and public-surface checks pass with 185 tests.

Dashboard WebSocket auth is now behind a route-facing helper too.
`server/engram/api/websocket_auth.py` owns auth config fallback, header tenant
resolution, and browser query-token bearer fallback before socket acceptance.
`dashboard_ws()` now delegates tenant resolution and auth failure close
behavior, and the public-surface guard rejects direct `resolve_tenant_from_scope`,
`Headers`, or query-param auth parsing in the route body. Focused
WebSocket/static/Ruff checks passed.

The public-surface guard now covers decorated MCP public surfaces too. MCP
startup/shutdown and store construction are still allowed to initialize and
close runtime state, but MCP tools/resources/prompts cannot directly dispatch
`store.*`, `session.*`, `_evaluation_store.*`, or `_consolidation_store.*`
methods. They must keep session/global dependency lookup plus JSON wrapping in
`mcp/server.py` and send lifecycle work through the named surface helpers. The
focused public-surface suite passes with 166 tests.

The same guard now rejects nested functions inside decorated MCP public
surfaces. This gives MCP tools/resources/prompts the same handler-shape
constraint as REST/WebSocket routes: public handlers stay as dependency lookup,
route-specific validation, and JSON transport wrappers rather than hiding
runtime callbacks in the decorated function body.

Decorated MCP public surfaces are also guarded by direct-await shape. MCP
tools/resources/prompts may directly await `build_*` helpers and
`resolve_mcp_consolidation_trigger_store()`, but direct awaits of arbitrary
runtime functions will fail the public-surface suite. The focused
public-surface suite passes with 168 tests.

Consolidation audit reads now have a single read-side boundary.
`server/engram/consolidation/audit_reader.py` owns latest/recent cycle reads,
cycle-detail audit record fanout, and evaluation-context snapshot collection.
REST consolidation status/history/detail routes call public
`ConsolidationEngine` reader facades, and cycle-detail payload assembly lives in
`serialize_cycle_detail()` instead of `server/engram/api/consolidation.py`.
MCP evaluation reports, MCP consolidation status, CLI lifecycle summaries, and
MCP lifecycle summaries also use `ConsolidationAuditReader`, so those surfaces
no longer need synthetic `consolidation_engine._store` objects. Focused
lifecycle/consolidation/API/MCP/static checks, Ruff, and `git diff --check`
passed.

REST consolidation control/read response assembly now has a route-facing public
surface too. `server/engram/consolidation_trigger.py` owns trigger conflict and
acknowledgement payloads, background manual-cycle execution, status
pressure/latest-cycle shaping, history cycle-list payloads, and cycle-detail
404/detail payloads. `server/engram/api/consolidation.py` keeps tenant lookup,
dependency lookup, background-task registration, and HTTP wrapping. Focused
consolidation-trigger, REST consolidation API, public-surface,
consolidation-presenter, and Ruff checks passed.

Knowledge-chat rich UI event shaping now lives outside the REST route.
`server/engram/retrieval/chat_events.py` converts raw recall/fact results into
route-neutral chat tool events and converts summarized chat recall output back
into the raw recall shape used by post-response feedback and UI event
selection. It also owns Anthropic tool-result message shaping and accumulation
of recall/fact tool JSON outputs for later rich-memory UI events, so the REST
route no longer parses tool JSON to rebuild memory result state. The same
presenter now also owns AI SDK synthetic tool payload-pair construction, while
`server/engram/api/knowledge.py` keeps only SSE wrapping for those payloads.
Focused chat event, recall presenter, public-surface, full knowledge API, and
Ruff checks passed; the latest chat event/route/static check passed with 172
focused tests.

Knowledge-chat tool execution payloads now live outside the REST route too.
`server/engram/retrieval/chat_tools.py` owns recall/search_entities/search_facts
tool dispatch payloads for the Anthropic chat loop, including chat recall packet
shaping, entity/fact LLM payloads, fact deduplication, unknown-tool responses,
and the non-streaming Anthropic tool-use loop/result accumulation.
`server/engram/api/knowledge.py` keeps Anthropic client construction, SSE
framing, while the chat tool schema, Anthropic text-block extraction, and
JSON-string compatibility wrapper used by legacy tests now live in
`server/engram/retrieval/chat_tools.py` too.
Focused chat-tool, chat API, chat-event, public-surface, and Ruff checks passed.

Knowledge-chat recall feedback and retry policy now live outside the REST route
too. `server/engram/retrieval/chat_feedback.py` owns used/dismissed memory
interaction application, generic memory-free response detection, retry gating,
and retry system-prompt construction, while `server/engram/retrieval/chat_tools.py`
owns retry provider execution. `server/engram/api/knowledge.py` keeps client
construction and stream framing as transport/client behavior.
Focused chat-feedback, chat-tool, full knowledge API, chat-event,
public-surface, Ruff, and `git diff --check` gates passed.

Knowledge-chat response-turn orchestration now lives outside the REST route as
well. `server/engram/retrieval/chat_runtime.py` owns chat memory-need analysis,
memory-guidance text, live conversation hydration, assistant-turn recording,
recent-turn extraction, chat runtime policy lookup, chat epistemic-evidence
dispatch, baseline context dispatch, system-prompt assembly, sliding-window
message assembly, tool-use loop invocation, retry policy application, recall
feedback, route-neutral rich tool stream payloads, and route-neutral text stream
payloads. The same runtime module owns rate-limit execution and 429 payload
selection through `check_api_chat_rate_limit()`. `server/engram/api/knowledge.py`
keeps rate-limiter dependency lookup, conversation helper invocation, Anthropic
client construction, SSE wrapping, and persistence scheduler invocation. Focused
chat-runtime/feedback/tool, full knowledge API, chat-event, public-surface,
Ruff, and `git diff --check` gates passed. The latest response-turn
orchestration check passed with 174 focused tests. A later broad
non-Docker/non-Helix gate now passes with 3213
tests, 43 skips, and 236 external-service deselections.
After this slice, the direct manager-dispatch scan across `server/engram/api/*.py`
is clean; the remaining direct matches are MCP auto-recall, recall middleware,
and live-turn piggyback compatibility paths. A later MCP auto-recall helper
slice moved those remaining full auto-recall, session-prime, and middleware
auto-observe dispatches into retrieval runtime helpers too, so the same scan
across `server/engram/api/*.py` and `server/engram/mcp/server.py` now returns
no matches except MCP shutdown resource closing, and the public-surface guard
now enforces that boundary.

REST/MCP explicit recall result and packet assembly now share a retrieval
boundary too. `server/engram/retrieval/recall_surface.py` owns the explicit
Recall-stage manager call, recall packet analysis, memory packet assembly, and
API/MCP recall item presentation. It also owns MCP entity-name/access-count
resolution plus near-miss/surprise side-channel enrichment for explicit recall.
MCP explicit recall query timing, recall-session flags, and recall middleware
invocation now live there too. REST still returns `items`, camelCase packets,
and `query`; MCP keeps manager/session lookup, config fallback, tool signature,
and JSON wrapping. Focused knowledge API, MCP JSON-response, autorecall, chat,
public-surface, Ruff, and `git diff --check` gates passed. The latest MCP
explicit recall tool-surface check passed with 168 tests.

Recall-control manager compatibility helpers now have one home.
`server/engram/retrieval/control.py` owns sync/async recall-need threshold
resolution and memory-need analysis recording. REST, MCP, chat runtime, chat
tool execution, and explicit recall surface code now share those adapters
instead of carrying private helper copies. Focused recall-control, knowledge
API, MCP JSON-response, autorecall, chat, public-surface, Ruff, and
`git diff --check` gates passed.

REST/MCP artifact search now shares retrieval-side result assembly.
`server/engram/retrieval/artifacts.py` owns artifact hit loading and item
serialization for both public surfaces. REST keeps the existing `projectPath`
response key; MCP keeps `project_path`. The MCP artifact-search tool-surface
helper now also owns recall middleware invocation, leaving the MCP transport
with manager lookup, callback injection, and JSON wrapping. Focused
artifact-search, artifact service, REST artifact endpoint, MCP artifact,
public-surface, Ruff, and `git diff --check` gates passed.

REST/MCP deterministic question routing now shares retrieval-side route surface
assembly. `server/engram/retrieval/epistemic_route.py` owns route history
normalization and the manager `route_question` call. REST keeps HTTP response
wrapping. The MCP question-route tool-surface helper now owns recall middleware
invocation with `auto_observe=True`, leaving the MCP transport with session
entity lookup, callback injection, and JSON wrapping. Focused route-surface,
REST epistemic endpoint, MCP JSON-response, public-surface, Ruff, and
`git diff --check` gates passed.

REST/MCP prospective-memory intention surfaces now share retrieval-side
assembly. `server/engram/retrieval/prospective.py` owns intention create, list,
and dismiss manager calls plus API/MCP acknowledgement shapes. REST keeps HTTP
wrapping; MCP keeps JSON wrapping. REST intention validation/not-found payload
bodies, REST create/dismiss status mapping, and MCP create/dismiss error
payloads now live in the same helper module too. Focused
prospective-surface, public-surface, full knowledge API, full MCP tool, Ruff,
and `git diff --check` gates passed. The latest prospective/chat error-payload
helper gate passed with 132 tests; the latest route-facing response-surface
check covering intentions, entities, and conversations passed with 165 tests;
the latest MCP intention response-surface gate passed with 165 tests and 2
skips.

REST/MCP forget surfaces now share retrieval-side target dispatch.
`server/engram/retrieval/forgetting.py` owns the public forget helpers and
fact-field normalization. REST keeps its entity-first behavior when both an
entity and fact are supplied; MCP keeps its exactly-one-target validation. REST
missing-target payload and 400/404 response mapping now also live in the
route-facing helper. Focused forget-surface, REST forget, MCP forget,
public-surface, Ruff, and `git diff --check` gates passed. The latest REST
forget response-surface check passed with 138 tests.

REST/MCP explicit preference feedback now shares retrieval-side validation and
manager dispatch. `server/engram/retrieval/preference_feedback.py` owns public
rating validation, the `record_explicit_feedback` manager call, REST error
payloads, and MCP invalid-rating error payloads. REST keeps HTTP status mapping.
Focused feedback-surface, feedback recorder, full knowledge API, full MCP tool,
public-surface, Ruff, and `git diff --check` gates passed. The latest REST
preference-feedback response-surface check passed with 135 tests.

REST/MCP project bootstrap and runtime-state calls now share route-facing
surface helpers. `server/engram/ingestion/project_bootstrap.py` owns the public
bootstrap manager call and REST skipped-status mapping while the existing
`ProjectBootstrapService` still owns artifact capture, cue-only bootstrap
episodes, and graph writes. `server/engram/retrieval/runtime_state.py` owns the
public runtime-state manager call while `RuntimeStateService` still owns the
runtime/config/artifact freshness read model. Focused project-runtime surface,
REST bootstrap/runtime, MCP runtime, public-surface, and Ruff checks passed.

REST/MCP public entity/fact lookup now shares route-facing lookup helpers too.
`server/engram/retrieval/lookup.py` still owns the deeper
`EntityFactLookupService`, and now also owns REST entity/fact search payload
shaping plus MCP entity/fact search payload shaping and missing-query
validation. REST keeps camelCase `items`; MCP keeps raw lookup results. The MCP
entity/fact lookup tool-surface helpers now own recall middleware invocation for
those read tools, leaving the MCP transport with manager lookup, callback
injection, and JSON wrapping. Focused lookup-surface, REST facts, MCP
entity/fact search, MCP middleware, public-surface, and Ruff checks passed.

REST/MCP public agent-context assembly now shares route-facing context helpers.
`server/engram/retrieval/context_builder.py` still owns the deeper
`MemoryContextBuilder`, and now also owns REST context payload shaping and MCP
raw context manager access. REST keeps camelCase count/token fields; MCP keeps
the raw `get_context` shape. The MCP context tool-surface helper now owns
recall/notification middleware invocation, leaving the MCP transport with
manager lookup, callback injection, and JSON wrapping. Focused context-surface,
tiered context, REST context/runtime, MCP context middleware, public-surface,
and Ruff checks passed.

REST/MCP adjudication resolution now shares ingestion-side surface helpers.
`server/engram/ingestion/adjudication_surface.py` owns the public
client-adjudication manager dispatch and API/MCP outcome shaping for resolved
edge-adjudication work items. REST keeps camelCase IDs; MCP keeps snake_case IDs.
Focused adjudication-surface, REST adjudicate, MCP adjudicate, public-surface,
and Ruff checks passed.

REST/MCP public Capture writes now share route-facing capture helpers.
`server/engram/ingestion/capture_surface.py` owns public conversation-date
parsing, attachment construction, raw observation storage dispatch, and
Capture -> Project ingest dispatch. REST observe/image/file/remember now route
memory-write presentation and adjudication request loading through that helper
too, while REST routes keep tenant/dependency lookup and JSON wrapping. MCP
write tools route through the same module for session activity updates,
live-turn recording, adjudication-request loading, memory-write presentation,
and recall middleware invocation as well, while `server/engram/mcp/server.py`
keeps manager/session lookup, JSON wrapping, and tool signatures. REST
auto-observe routes enablement, short-content skip, dedup skip, raw observation
storage, and memory-write presentation through the capture surface while the
route keeps dependency lookup and JSON wrapping. Focused capture-surface,
memory-write presenter, REST observe/remember/adjudication, REST auto-observe,
MCP write/adjudication, public-surface, and Ruff checks passed.

REST offline replay also has a manager-facing route helper now.
`server/engram/ingestion/offline_replay.py` owns the route-to-manager store
facade for replaying queued Capture entries, so `/api/knowledge/replay-queue`
no longer passes `manager.store_episode` through the REST handler body. Focused
offline replay, REST replay queue, public-surface, and Ruff checks passed.

REST entity detail/update/delete now has a route-facing public-surface helper.
`server/engram/retrieval/entity_surface.py` owns entity detail manager dispatch,
sparse update payload construction, delete dispatch, 404 status mapping, and
the shared REST not-found payload, while `GraphStateService` and
`EntityMutationService` remain the deeper service owners. Focused entity-surface,
REST entity detail/mutation, public-surface, and Ruff checks passed. The latest
entity response-surface check passed with 143 tests.

MCP graph-state tool and graph/entity resources now have route-facing public
surface helpers. `server/engram/retrieval/graph_state.py` now owns MCP graph
tool dispatch, graph stats resource shaping, entity profile resource dispatch,
and entity-neighbor resource dispatch on top of the existing `GraphStateService`
read model. The same module now owns REST graph neighborhood/temporal route
dispatch, entity-neighbor route dispatch, not-found payloads, and timestamp
validation payloads. Focused MCP graph-state surface, graph-state
service/resource, MCP graph-state, public-surface, and Ruff checks passed.

MCP identity-core and consolidation controls now have route-facing public surface
helpers too. `server/engram/retrieval/identity_core.py` owns MCP identity-core
manager dispatch, and `server/engram/consolidation_trigger.py` owns MCP trigger
dispatch, consolidation status reads, and cycle-summary shaping. The MCP
transport keeps JSON wrapping and active consolidation-store selection. Focused
identity-core, consolidation trigger/status, MCP control, public-surface,
consolidation-presenter, and Ruff checks passed.

Lifecycle summary route-facing helpers now cover both REST and MCP.
`server/engram/lifecycle_summary.py` owns the REST runtime-context manager call
shape for `/api/lifecycle/summary`, plus active audit-store reader construction,
inactive-engine placeholder wiring, and limit clamping for MCP lifecycle reads.
The REST route keeps dependency lookup and JSON wrapping; the MCP tool keeps
JSON wrapping and session state lookup. Focused API lifecycle, lifecycle
summary, public-surface, and Ruff checks passed, including 146 tests for the
latest REST helper slice.

Knowledge-chat conversation persistence now has its own helper boundary too.
`server/engram/retrieval/chat_persistence.py` validates existing conversation
IDs against the active `group_id`, creates missing conversations with the
request group and session date, schedules best-effort completed-turn
persistence, persists completed user/assistant turns, and tags unique recalled
entity IDs. It also owns the chat conversation not-found payload body. The REST
chat route still owns SSE wrapping, Anthropic client construction, and HTTP
status mapping, but no longer owns conversation persistence rules, the
background persistence task body, or that not-found response body. Focused
chat-persistence, conversation
ownership, chat, public-surface, and full knowledge API checks passed, and Ruff
passed.

REST conversation CRUD now uses a group-scoped persistence helper.
`server/engram/retrieval/conversation_persistence.py` owns conversation listing,
creation, message reads/appends, title updates, deletes, and not-found
translation for the active `group_id`; it now also owns the REST response
envelopes for list/create/message/update/delete acknowledgements and the shared
not-found body/status mapping. `server/engram/api/conversations.py` keeps tenant
lookup, request parsing, and JSON wrapping, but no longer directly encodes
conversation store calls, payload bodies, or 404 branches. Focused
conversation-persistence, conversation API, public-surface, and Ruff checks
passed. The latest conversation response-surface check passed with 157 tests.

REST/MCP post-write adjudication request loading now has one helper.
`server/engram/ingestion/adjudication_surface.py` owns the compatibility lookup
for episode adjudication work items after remember writes, including sync/async
manager facades, client-enabled surfacing gates, and missing/malformed
responses. REST and MCP remember surfaces still own their transport response
shapes, but the adjudication request loading contract is shared before the
common memory-write presenters run. Focused
adjudication-surface, REST remember, MCP JSON-response, public-surface, and
Ruff checks passed.

REST/MCP live conversation facade access now uses shared retrieval helpers.
`server/engram/retrieval/context.py` owns the defensive sync/async/type checks
around manager conversation context, live-turn embedding function, turn counts,
session entity names, recent turns, and live-turn ingestion. REST chat and MCP
recall piggybacking keep their local compatibility wrapper names, but those
wrappers now delegate to the shared retrieval-side facade helpers. Focused
conversation-runtime, chat-context, MCP piggyback, public-surface, and Ruff
checks passed.

REST/MCP brain-loop evaluation reports now share report assembly as well.
`server/engram/evaluation/report_service.py` reads graph state through the
manager facade, persists or reloads runtime Recall metrics snapshots through
the evaluation store, loads saved recall/session labels, and calls the shared
`build_brain_loop_report()` contract. MCP active audit-store reads, cycle-limit
clamping, and calibration snapshot loading now live in that report service too.
REST engine-derived cycle context loading also lives behind the route-facing
`build_api_brain_loop_evaluation_surface()` helper, with a static guard against
direct REST route `engine.*` dispatch. Focused evaluation report service, REST
evaluation, MCP JSON-response, public-surface, and Ruff checks passed.

REST/MCP evaluation label writes now share a service boundary too.
`server/engram/evaluation/label_service.py` builds and persists recall-quality
and session-continuity samples for the active `group_id`, preserves count
clamping for MCP inputs, and now owns the REST/MCP write acknowledgement payloads
through route-facing helpers. Focused label service, REST evaluation, MCP
JSON-response, public-surface, and Ruff checks passed.

After these route-orchestration slices, the broad backend non-Docker/non-Helix
gate passes with 3316 tests, 43 skips, and 236 external-service deselections
after the shared companion-store bootstrap follow-up, explicit
notification/scheduler dependency cleanup, and the public smoke cue-feedback
facade, plus REST/MCP shutdown stop/close and consolidation-helper cleanup with
static guards.

Shared storage bootstrap initialization now has a named helper boundary.
`server/engram/storage/bootstrap.py` owns the lite shared-DB lookup plus store
and search-index initialization helpers. It also owns atlas, consolidation,
evaluation, and conversation store creation for REST startup plus
consolidation/evaluation companion-store creation for MCP startup, lifecycle
CLI, evaluation CLI, and projected/consolidated smoke. Those entrypoints now
share lite borrowed-DB and Helix shared-client handling instead of repeating
private graph-store SQLite connection and Helix-client checks. Borrowed
in-memory consolidation fallback creation now lives there as well for MCP
trigger resolution, lifecycle summary fallback reads, and graph-health SQLite
metrics. Focused storage-bootstrap, borrowed-connection, lifecycle CLI,
consolidation CLI, projected/consolidated smoke, REST startup, auto-observe,
native manifest, runtime shutdown-facade, and Ruff checks passed; the latest
companion-store follow-up added focused storage, REST startup/API,
CLI/doctor/smoke, MCP tool import, graph-health, and live lifecycle/evaluation
CLI smoke checks.

Episode worker runtime-store access now has the same named dependency boundary.
`server/engram/ingestion/worker_runtime.py` defines `EpisodeWorkerRuntimeStores`,
REST and MCP startup pass graph/search/activation stores explicitly to the
worker, and `GraphManager.get_episode_worker_runtime_stores()` exists only as
the compatibility accessor for direct worker construction. `EpisodeWorker`
still uses `GraphManager.project_episode()` as the Project-stage facade, but it
no longer reaches through private manager graph/search/activation fields for
Capture/Cue batching, projection-state sync, multi-signal scoring, goal boosts,
cue rebuild, or merged-cue retirement. Focused worker, auto-observe, rework,
facade-boundary, group-scope, Ruff, and broad non-Docker/non-Helix checks
passed.

Episode worker adjacent-turn batching is now a named ingestion helper as well.
`server/engram/ingestion/worker_batching.py` owns auto-capture batch merge,
primary episode content update, primary cue rebuild/re-index, and merged-away
episode/cue retirement. `EpisodeWorker` still decides when to flush, score, and
route the primary episode, but the Cue-stage mutation and indexing contract no
longer lives inside the worker loop. Focused worker-batching, worker,
auto-observe, rework, facade-boundary, group-scope, Ruff, and broad
non-Docker/non-Helix checks passed.

Episode worker deterministic scoring is now a named ingestion helper.
`server/engram/ingestion/worker_scoring.py` owns heuristic triage scoring,
multi-signal scorer access, goal boost lookup, and projection-yield feedback.
`EpisodeWorker` now delegates scoring and outcome recording while keeping event
consumption, duplicate guards, routing, and the `GraphManager.project_episode()`
Project-stage facade call. Focused worker-scoring, worker-batching, worker,
auto-observe, rework, facade-boundary, group-scope, Ruff, and broad
non-Docker/non-Helix checks passed.

Episode worker projection routing is now a named ingestion helper.
`server/engram/ingestion/worker_routing.py` owns duplicate projection guards,
system-discourse cue-only skips, skip/defer projection-state sync, and the
boolean project-now routing contract. `EpisodeWorker` now keeps event
consumption, batch timing, and Project-stage dispatch. Focused worker-routing,
worker-scoring, worker-batching, worker, auto-observe, rework, facade-boundary,
group-scope, Ruff, and broad non-Docker/non-Helix checks passed.

Episode worker event parsing and compact content loading are now a named
ingestion helper. `server/engram/ingestion/worker_events.py` owns raw EventBus
payload parsing for `episode.queued` and `episode.projection_scheduled`,
normalizes the worker event shape, and expands compact auto-capture payloads
from the graph store before batching. `EpisodeWorker` now keeps subscription
lifecycle, queue/batch timing, and Project-stage dispatch without embedding
raw payload keys or route-specific event shape. Focused worker-event,
worker-routing/scoring/batching, worker, auto-observe, rework, facade-boundary,
group-scope, Ruff, and broad non-Docker/non-Helix checks passed; the latest
broad gate passes with 3357 tests, 43 skips, and 236 external-service
deselections after the Helix dashboard analytics fixture was made date-stable,
the doctor readiness failure path was guarded, the shared companion-store
bootstrap follow-up landed, notification/scheduler dependencies were made
explicit, smoke cue feedback moved onto the public manager facade, and REST/MCP
shutdown stop/close facade cleanup plus static guards landed. The latest gate
also includes REST knowledge-chat SSE runtime extraction, the MCP
authority/onboarding prompt contract, the `claim_authority()` callable
contract, dashboard WebSocket auth route-boundary extraction, REST chat
response-surface extraction, REST health route-boundary extraction, adoption
transcript stdin validation, and self-reported file-memory bypass
classification, the REST auto-observe request-surface boundary, and FastMCP
`/mcp` manifest classification through the root-mounted app.

REST shutdown now shares the same runtime-resource boundary as MCP shutdown.
`server/engram/main.py` stops subscriber/worker/pressure/scheduler resources
through `stop_if_supported()`, closes consolidation/evaluation/atlas/conversation
and aclose-only clients through `close_if_supported()`, then delegates owned
search/activation/graph store cleanup to `GraphManager.close_runtime_resources()`
when the manager is available. The direct store-close path remains only as a
startup-failure fallback. MCP Redis publisher shutdown also uses
`close_if_supported()` after removing the event-bus hook. `tests/test_public_surface_presenter_boundaries.py`
now statically checks both shutdown paths use the shared stop/close boundary and
rejects REST-local direct `stop`/`close`/`aclose` calls plus MCP direct
`stop`/`close` calls. Focused
shutdown/bootstrap/MCP/public-surface checks passed, the API/MCP
startup-shutdown suite passed, and the broad non-Docker/non-Helix gate now
includes the new REST shutdown regressions, stop-helper tests, and static
guards.

Shutdown consolidation orchestration is now out of `server/engram/main.py`.
`run_shutdown_consolidation()` in `server/engram/consolidation_trigger.py`
owns the shutdown decision to cancel a running engine, skip disabled
consolidation, or run a final `trigger="shutdown"` cycle. `main._shutdown()`
now passes the engine/config/logger into that helper, and the public-surface
static guard rejects direct `is_running`, `cancel`, or `run_cycle` usage in the
FastAPI shutdown function. Focused helper/main/static tests passed, including
dynamic coverage that `main._shutdown()` passes the active engine/config/logger
into the helper, and the broad non-Docker/non-Helix gate now includes helper
coverage for run, skip, cancel, and logged-failure paths.

REST knowledge-chat SSE transport orchestration is now out of
`server/engram/api/knowledge.py`. `stream_api_chat_sse_events()` in
`server/engram/retrieval/chat_runtime.py` owns start/step/text/finish/error SSE
framing, Anthropic client construction, response-turn execution, and best-effort
conversation persistence scheduling. The remaining REST chat response setup now
lives in `build_api_chat_stream_response_surface()`, which owns rate-limit
responses, optional conversation-store handling, conversation not-found payloads,
session entity lookup, and SSE stream construction. The route keeps tenant and
dependency lookup plus JSON/streaming response wrapping. `tests/test_chat_runtime_stream.py`
covers success/error stream envelopes plus rate-limit and not-found response
surfaces, and the public-surface guard rejects inline `_sse`, nested
event-stream generators, direct Anthropic construction, chat rate-limit helpers,
conversation resolution, `run_chat_response_turn()`, and
`schedule_chat_turn_persistence()` in the REST route.

MCP adoption now has an explicit authority contract. `ENGRAM_SYSTEM_PROMPT` says
Engram owns portable cross-context user facts, preferences, corrections, durable
decisions, relationships, goals, commitments, and long-tail recall, while
project-local files own repo conventions and current-task scratch notes. It also
tells agents that an empty runtime (`artifactCount: 0`, `lastObservedAt: null`,
or zero recall/evaluation stats) is an onboarding state and should trigger
`bootstrap_project(project_path)` when a project path is available. README
automatic-memory behavior and `tests/test_mcp_prompts.py` now cover that
agent-adoption contract. The new MCP `claim_authority(project_path, user_message,
file_memory_present)` tool makes that contract callable: it returns Engram-owned vs project-local memory
responsibilities, onboarding state, recommended bootstrap/context/recall
actions, the current runtime-state payload, and the shared brain-loop lifecycle.
It now also accepts `user_message` and `file_memory_present` and returns an
`agent_protocol` with required tools before answering plus capture routing. The
deterministic covered failure mode is a connected-but-empty Engram runtime with
file-local memory visible: the protocol requires bootstrap/context/recall before
answering and routes high-signal cross-context facts to Engram `remember`.
`validate_agent_protocol_calls()` now gives real MCP clients and thin harnesses a
transcript validator for that contract, including missing/out-of-order
pre-answer tools, missing Engram capture, unexpected Engram writes for
project-local scratch, and visible file memory substituting for Engram.
`tests/test_mcp_authority_client_adoption.py` runs an actual stdio MCP client
against `engram mcp`, follows the `claim_authority()` protocol, and validates
the resulting transcript. That live client path also found and closed an
onboarding drift: missing or stale project artifacts now produce
`needs_project_bootstrap` and require `bootstrap_project`, even if other runtime
metrics make the graph itself look active.
The MCP system prompt now explicitly says to follow
`agent_protocol.required_tools_before_answer` and the returned `capture`
decision, while `engram setup` and README print the same adoption checklist for
Claude Code, Cursor, Windsurf, and similar clients. This moves the authority
contract out of hidden internals and into the surfaces real agents and installers
see.
`engram adoption --authority claim-authority.json --calls mcp-calls.jsonl` now
wraps the same validator as an operator-facing transcript check, so recorded
real-client sessions can be scored without manually inspecting tool logs.
`claim_authority()` also returns this verifier command and the JSONL transcript
schema inside `agent_protocol.verification`, making the adoption contract
self-describing for harnesses that want to prove compliance.
Adoption validation reports also include a `release_evidence` handoff section
with the prefilled human-label template command and final
`engram evaluate --require-release-evidence` command, so a passed live-client
artifact directly names the next release packaging step instead of leaving that
sequence only in prose. `engram adoption --report-out adoption-report.json`
writes that JSON validation artifact directly for the release gate, and the
generated live-harness templates include the same flag in their validation
commands. `engram evaluate --human-label-template-out human-label-template.json`
now writes the fillable JSON label template artifact while preserving the
normal JSON/Markdown output. Evidence bundles now include a top-level
`source_sha256` map for report, benchmark, human-label, adoption, and sample
files, so archived release evidence can be traced back to exact source
artifacts.
The release gate is now stricter by default: adoption evidence must come from a
report validated with `--require-live-evidence`, and
`--require-release-evidence` defaults to 10 human recall labels plus 3
session-continuity labels instead of the 1/1 local-check threshold. The
adoption report handoff mirrors that gate by staying blocked until
`--require-live-evidence` was used.
For completion-grade live harness evidence, the same verification block now
also returns a `live_evidence_command` and JSON wrapper schema requiring
`client` plus `capturedAt` metadata. `engram adoption --require-live-evidence`
adds a `missing_live_harness_evidence` failure when the tool-call transcript
passes but lacks current client/session evidence, so a handcrafted JSONL sample
cannot substitute for Claude/Cursor/Windsurf proof.
The verifier accepts common real-log forms too: prefixed tool names such as
`mcp__engram__recall`, nested `tool` / `function` / `tool_call` records, and
`stage` as a phase alias. It also accepts explicit plaintext/Markdown harness
notes with `before_answer`/`capture` headings plus common
`Before answer`/`pre-answer` aliases and Engram tool lines, so copied
Claude/Cursor/Windsurf session notes can be validated without manual JSON
conversion. Malformed copied notes now return a structured
`invalid_calls_transcript` report instead of surfacing a parser exception, and
the CLI accepts `--calls -` to read copied transcript notes from stdin.
Copied chat notes that do not expose raw tool calls but include the agent's own
admission that it ignored Engram or treated file-local memory as primary now
become failed adoption transcripts instead of parse failures. The copied Claude
transcript that motivated this slice is now covered as a regression: Engram is
reachable, reports an empty runtime, and the agent admits file memory stayed
primary, so `engram adoption` classifies the transcript as a failed adoption.
The returned protocol metadata also marks whether capture is required, and
project-local scratch examples no longer show a fake capture tool.
It is covered by `tests/test_project_runtime_surfaces.py`,
`tests/test_mcp_tools.py`, `tests/test_public_surface_presenter_boundaries.py`,
and the native surface manifest; README now lists 27 MCP tools.

MCP recall middleware is now a thin named-adapter path too.
`_recall_middleware()` passes `_ingest_live_tool_turn()` into
`run_mcp_recall_middleware()` instead of defining a nested runtime callback in
`mcp/server.py`. The static boundary suite asserts that the middleware delegates
through the named helper and contains no nested runtime callbacks, keeping
live-turn ingestion behind the retrieval middleware boundary.

MCP auto-recall policy helpers now live in retrieval runtime code.
`server/engram/retrieval/auto_recall.py` owns the cooldown/topic deduplication
class, compact recall-query extraction, per-tool recall gating, and first-call
session-prime planning used by MCP. It also owns the MCP recall middleware
side-effect plan for auto-observe, live-turn ingestion, and notification
fallbacks when recall is disabled. It now also owns
`build_lite_auto_recall_surface()`, the lite/medium dispatch plus
entity-probe surface compaction that `_auto_recall_lite()` attaches as
`recalled_context`, `build_full_auto_recall_surface()`, the full recall
need-analysis/cooldown/topic-shift/packet dispatch plus score-filtered
entity/cue/packet surface that `_auto_recall_full()` attaches as
`recalled_context`, `build_session_prime_surface()`, the first-call context
prime dispatch used by `_session_prime()`, `store_mcp_auto_observe_turn()`, the
middleware auto-observe storage boundary, `drain_mcp_triggered_intentions()`,
the triggered-intention manager-facade drain for MCP recall enrichment, and
`apply_mcp_recall_enrichment()`, the additive response attachment contract for
session context, recalled context, triggered intentions, and memory
notifications. MCP piggyback notification state lookup now uses the pure
`build_mcp_notifications_surface()` presenter with a notification surface
service supplied by `api/deps.py`.
`run_mcp_recall_middleware()` now owns the middleware plan execution too,
including middleware auto-observe, read-tool live-turn ingestion, first-call
session prime, lite auto-recall, triggered-intention draining, notification
lookup, and additive response enrichment.
`server/engram/mcp/server.py` keeps compatibility wrappers for existing tool
tests and still owns tool-specific fetching, session/global dependency lookup,
JSON wrapping, and transport behavior. Focused autorecall, piggyback,
notification, recall-lite/MCP recall-selection, MCP response-enrichment,
public-surface, and Ruff checks passed.

Notification presentation and scheduler temporal scans now use explicit
dependencies instead of shared runtime modules reading app state. The optional
notification-service lookup stays in `server/engram/api/deps.py`, while
`server/engram/notifications/surface.py` remains a pure MCP notification
presenter. `ConsolidationScheduler` receives the active graph store from REST
startup and passes it into the temporal scanner. Public-surface checks now guard
that those shared runtime modules do not import or read `_app_state` directly.

The projected/consolidated smoke cue-feedback path now writes through
`GraphManager.apply_memory_interaction()` and verifies the public cue result,
instead of calling manager private graph or cue-hit helpers. This keeps the
Recall feedback verifier on the same facade that runtime consumers use.

Not covered in this pass:

- Docker/full-mode compatibility/integration execution.
- Multi-hour or overnight native stress execution. The one-hour PyO3 native
  Recall soak is now covered, but this pass did not attempt a longer endurance
  run.
