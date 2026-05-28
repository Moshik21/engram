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
  inherit slow embedding/vector latency from a loaded native store. On the
  local FastEmbed/native Helix path, that threshold is a soft background-lane
  threshold rather than a cancellation point, so successful vector writes can
  finish and clear the cue-index outbox after capture has already acknowledged.
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
- Startup storage/runtime probes now avoid deep graph paths by default:
  `/api/storage` returns cached last-known counts and paths unless `live=true`
  is requested, AXI home uses `/api/knowledge/runtime/fast`, and
  `engram axi storage` / `engramctl storage` are the explicit live diagnostic
  paths. Memory operation p95 now uses nearest-rank percentile math so small
  samples cannot report p95 below a slower average.
- MCP now keeps its initialized runtime process-local by default instead of
  tearing down GraphManager/storage on every streamable HTTP lifespan close.
  Deep `/api/knowledge/runtime` is cache-first and budgeted by default, with
  `live=true` as the explicit deep refresh. Runtime, storage, and MCP init
  surfaces now expose first-stage timing keys (`runtime_state`, `storage_counts`,
  `storage_paths`, and `mcp_init`) to make the next bottleneck measurable.
- MCP startup now defers evaluation and consolidation store initialization until
  the report/label/consolidation tools actually need those stores. The lazy
  initialization timings are recorded separately as
  `mcp_evaluation_store_lazy_init` and `mcp_consolidation_store_lazy_init` so
  cheap tools do not pay that startup cost silently.
  A live isolated lite MCP probe with noop embeddings initialized in about 62ms;
  `get_runtime_state` and `observe` left both stores uninitialized, while
  `get_evaluation_report` initialized them lazily and surfaced both lazy timing
  keys.
  The same isolated PyO3 native Helix probe initialized in about 145ms with 171
  native routes loaded; `get_runtime_state` still left evaluation and
  consolidation stores uninitialized.
- AXI CLI project/topic routing now preserves global `--project` and `--topic`
  values when they appear before `context`, `recall`, or `doctor` subcommands.
  This closes a dogfood false latency path where the intended topic-specific
  context fallback could be bypassed by argparse defaults, making the command
  depend on CWD inference. After reinstalling the local CLI, fresh
  global-before-subcommand probes from `/tmp` returned useful packets without
  degradation: AXI context rebuilt project-file packets in `562.4966ms`, AXI
  recall returned three project packets in `636.2156ms`, and AXI doctor passed.
  The startup validator now warns when a hook trace records `project=/`, and the
  matrix propagates validation warnings instead of flattening them to pass. With
  those stricter checks, the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-134521` with
  `11 pass, 2 warn, 0 fail, 0 skip`; both warnings are the same Codex
  SessionStart root-project trace. After that matrix restart, AXI context
  returned five project-file packets in `699.2828ms`, AXI recall returned three
  project packets in `1132.7942ms`, MCP context returned useful project-file
  packets in `129.0879ms`, and MCP recall hit cache in `3.2646ms`. Managed Codex
  and Claude Code AXI hooks now use `engram axi hook-run`, which reads hook stdin
  JSON `cwd` instead of shell `$PWD`; both local hook configs were rewritten with
  that command after reinstall. Manual `hook-run` smoke with stdin
  `{"cwd":"/Users/konnermoshier/Engram"}` returned a healthy packet with
  `brain.project=/Users/konnermoshier/Engram`; the validator warning should clear
  only after a fresh real Codex SessionStart trace replaces the older `project=/`
  row.
  The validator now also compares each installed hook config mtime against the
  latest SessionStart trace timestamp, so it warns when startup evidence predates
  the current hook command. Current skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`: Codex and Claude Code both need fresh
  SessionStart traces after the hook-run reinstall, and Codex still has the older
  root-project row. Fresh live read probes remain healthy: AXI context returned
  three loaded-store packets in `78.3556ms`, AXI recall was `cache_satisfied` in
  `1.5127ms`, MCP context returned five project-file packets in `123.034ms`, and
  MCP recall was `cache_satisfied` in `2.4123ms`.
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
  Post-matrix runtime stayed healthy on LaunchAgent PID `74501`. Fresh
  post-matrix probes stayed bounded without degradation: AXI context returned
  five project-file fallback packets in `1205.4917ms`, AXI recall returned three
  project-file packets in `1559.5697ms`, MCP context returned five project-file
  packets in `142.7577ms`, and MCP recall was `cache_satisfied` in `54.5339ms`.
  A follow-up fallback-quality pass fixed stale and cross-project project-file
  cache behavior found in live dogfood. Project-file fallback now uses a larger
  bounded topic scan, adjacent line-window scoring for wrapped evidence, capped
  historical term scoring, and `docs/CURRENT_HANDOFF.md` priority for current
  evidence. Explicit recall's context-packet fallback now filters cached
  project-file packets by `project_path`, so an Engram recall cannot be
  satisfied by MachineShopScheduler project-file packets. After reinstall/restart,
  AXI context for `startup matrix 20260527 tiecheck gold` returned the current
  `20260527-140202` `CURRENT_HANDOFF.md` packet in `537.2311ms`; repeat AXI
  context hit cache in `0.1081ms`; AXI recall was `cache_satisfied` in
  `18.1105ms`; MCP context hit the same packet in `0.0629ms`; and MCP recall was
  `cache_satisfied` in `24.3204ms`. Focused context/recall surface tests passed
  with `72 passed`, and ruff passed on the touched retrieval files/tests. The
  refreshed full lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-142608` with
  `11 pass, 2 warn, 0 fail, 0 skip`; after that restart, LaunchAgent PID `9661`
  was healthy, cold-ish AXI context returned the current handoff packet in
  `952.857ms`, and repeat AXI recall hit cache in `1.8494ms`.
  The follow-up cache-quality pass fixed stale fallback-row masking and noisy
  adjacent-line summaries. Project-file fallback packets now carry
  `version=2`; exact context cache hits require the current version, and
  explicit recall drops old unversioned project-file fallback packets before
  considering cache satisfaction. Adjacent line scoring now joins only true
  wrapped-continuation lines, so summaries stop blending unrelated previous
  evidence. After reinstall/restart, first-hit AXI recall for
  `startup matrix 20260527 tiecheck diamond` rebuilt current Engram evidence in
  `1249.1747ms`, and first-hit AXI context for
  `native PyO3 dogfood performance continuation cleanline 20260527` rebuilt a
  clean handoff packet in `931.882ms`. Repeats then hit cache: AXI context
  `0.2474ms`, AXI recall `cache_satisfied` in `74.5458ms`, MCP context
  `0.1637ms`, and MCP recall `cache_satisfied` in `2.4684ms`. Focused
  retrieval tests passed with `75 passed`, ruff passed, skip-slow validation
  reports `11 pass, 1 warn, 0 fail, 2 skip`, and the refreshed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260527-144207` with
  `11 pass, 2 warn, 0 fail, 0 skip`. The remaining warning class is still hook
  evidence freshness/root-cwd, not the read-path cache behavior.
  The next live packet-quality pass tightened usefulness rather than raw
  latency: lowercase wrapped evidence lines now join to the preceding unfinished
  line, and packet snippets truncate on word boundaries with `...` instead of
  cutting a token. Fresh AXI context for
  `startup matrix 20260527 tiecheck diamond project_file_recall_fallback
  continuationproof2` rebuilt the current handoff packet in `785.6527ms`, with
  the summary starting at the `startup matrix ...` evidence line and ending at
  `native PyO3...` instead of the prior `native PyO3 dogfood p` cut. Repeats
  hit the fast path: AXI context `0.1051ms`, AXI recall `cache_satisfied` in
  `52.3059ms`, MCP context `0.0807ms`, and MCP recall `cache_satisfied` in
  `1.1909ms`. Focused retrieval tests passed with `77 passed`, ruff passed, and
  `git diff --check` is clean. The refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-145536` with
  `11 pass, 2 warn, 0 fail, 0 skip`; the remaining warnings are still only
  stale/root SessionStart hook evidence.
  The next continuation-window follow-up made evidence-line selection match the
  summary fix. Matching now requires a direct term hit on the current line,
  walks bounded previous continuation chains, and trims unrelated prior
  sentences from wrapped previous lines. Fresh AXI context for
  `evidence project_file_recall_fallback wrappedwindow liveproof 20260527
  chainfixed2` rebuilt the current handoff packet in `747.8033ms`; MCP context
  returned evidence lines beginning with the full `startup matrix ...` line and
  `After reinstall...`, with no `evidence in...` or `can satisfy...` starts.
  Repeats hit cache: AXI context `0.179ms`, AXI recall `cache_satisfied` in
  `3.8688ms`, MCP context `1.5832ms`, and MCP recall `cache_satisfied` in
  `1.0518ms`. Focused retrieval tests passed with `80 passed`, ruff passed, and
  `git diff --check` is clean. The refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-151148` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `48671`, and the remaining warnings are still only stale/root
  SessionStart hook evidence.
- Storage diagnostics now keep cached counts warm with write-through deltas
  from Capture, Project, and known consolidation mutations, so default
  `/api/storage` can show growth without scanning Helix. Live storage refreshes
  remain the reconciliation path.
  Capture cue vector indexing is persisted to a SQLite cue-index outbox and then
  queued in the background after the cue is written, so observe/remember
  acknowledgements no longer wait on embeddings and restart-time replay can
  recover unfinished cue indexing work.
  Explicit recall responses now include diagnostic stage timings for recall
  search, packet assembly, MCP presentation, and lower-level manager stages
  such as graph expansion/retrieve/materialize/post-process when available.
  If the deep explicit-recall path times out, REST/MCP now attempt a bounded
  fast fallback over cue and episode search plus already-cached explicit
  recall packets, so degraded responses can still carry useful context instead
  of returning only an empty timeout payload.
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
- 2026-05-22 native dogfood rerun after reinstalling the local worktree and
  updated permissions: `python3 scripts/dogfood_startup_validation.py --json`
  passed native config, health, port listener, LaunchAgent parity, project MCP,
  storage path, `engramctl status`, `engramctl storage`, `engramctl doctor`,
  the live MCP catalog including `remember`, Codex MCP config, Claude Code MCP
  config, AXI hook traces, and OpenClaw MCP config. The OpenClaw check now
  parses pretty JSON even when `npx` prints a warning before the payload.
- After hardening LaunchAgent shutdown waits, an explicit
  `engramctl stop && engramctl start` sequence passed on the loaded native
  dogfood runtime in about 18.5 seconds. The full lifecycle matrix then passed:
  `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260522-161920/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`, including warmed, stopped, restarted, and
  stale-PID validation states.
- Cache-first context now works on the loaded native dogfood store. After
  clearing the packet cache, the first REST context call for the Engram project
  returned useful context in about 0.63 seconds and wrote 7 packets; the next
  REST context call hit cache in about 0.019 seconds. `engram axi context`
  returned the cached packet view in about 1.35 seconds. REST/MCP explicit
  recall still degrades on the deep recall path, but now returns cached project
  packets in about 1.7 seconds instead of an empty timeout payload.
- After the AXI presenter fix, `engram axi recall "Engram performance dogfood
  runtime" --timeout 10 --json` also prints the compact cached packet bodies
  when deep recall degrades. The live native dogfood run returned
  `status=degraded`, `skipReason=recall_timeout`, `packet_count=3`, and
  redacted cached project packets in about 2.7 seconds end-to-end through the
  CLI.
- 2026-05-23 follow-up hardening fixed the remaining cold/split-manager packet
  cache miss. Long topic hints that already include the project name now load
  stable project context first, so a cold MCP
  `get_context(project_path=/Users/konnermoshier/Engram, topic_hint="Engram
  native PyO3 dogfood performance goal continuation")` returned useful project
  context in about 0.93 seconds instead of an empty degraded timeout. The next
  MCP call hit packet cache in about 0.08 seconds.
- Packet-cache reads now sync fresh persistent entries on cache miss/recent
  fallback, so REST/AXI and MCP GraphManager instances can share warmed packet
  cache entries. After warming through MCP, REST recall returned
  `status=degraded`, `skipReason=recall_timeout`, `packetCount=3`, and cached
  project packets in about 1.76 seconds; `engram axi recall` returned the same
  three compact cached packets in about 2.86 seconds end-to-end.
- Earlier installed dogfood runs still had deep recall degradation under the
  explicit 1200ms search budget; the later 2026-05-26 latency pass moved that
  budget to a configurable 900ms search stage and capped timeout-rescue
  fallback at 150ms, making the degraded path useful and measurably shorter
  rather than empty.
- The confirmed lifecycle matrix was rerun after these changes and the local
  permission cleanup:
  `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260523-192255/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-25 loaded-store hardening added a bounded project-artifact context
  fallback that does not trigger bootstrap, plus a 50ms per-entity enrichment
  cap so relationship/detail lookups cannot consume the whole context budget.
  Project bootstrap refresh now skips unchanged artifact decision
  rematerialization and unchanged `PART_OF` relationship checks.
- After reinstall/restart on the native dogfood store, a sequential live probe
  showed cold REST context rebuilding project packets in about 1.05-1.26s,
  subsequent REST context cache hits in about 4-33ms, REST recall still
  degrading on the 1200ms deep search budget but returning cached packets in
  about 1.7s, and REST bootstrap returning `already_bootstrapped` in about
  19-187ms. AXI and MCP context/recall also returned the warmed cached packet
  payloads.
- Remaining performance work: deep explicit recall still times out under the
  loaded-store search budget, cold artifact packet ranking can still be too
  generic for narrow performance topics, and stale full-project bootstrap
  refresh is unit-covered but not yet live-proven because the timed-out dogfood
  bootstrap advanced the local project timestamp before this pass completed.
- The dogfood startup validator passed against the current runtime, and the
  lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260525-160226/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-26 continuation added a graph-free project-file context fallback for
  the last cold-cache timeout case: when deep graph context times out and no
  packet-cache entry is available, REST/MCP context now synthesize redacted
  `project_file` packets from bounded local project files and write them into
  the normal `project_home` packet cache for the next call. This path is covered
  by `test_mcp_context_surface_uses_project_files_after_timeout_when_cache_cold`.
- After reinstall/restart, the exact continuation MCP context that previously
  returned an empty timeout instead returned graph-backed project context and
  warmed cache. The next REST/AXI context calls hit `project_home` cache, and
  REST recall still degraded on deep search but returned cached packet titles
  (`SQLite`, `README.md`, `Makefile`) instead of an empty payload. A fresh
  Engram MCP `observe` call still timed out at the 120s client boundary before
  this reinstall, so capture latency remains part of the dogfood performance
  backlog.
- The current runtime then passed
  `python3 scripts/dogfood_startup_validation.py --json`, and
  `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260526-072233/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- After the local permission update, a live MCP `observe` sample stored
  `ep_c8fcc74220dd` and queued projection, but still spent about 11 seconds in
  `capture_store`. A post-restart AXI context call proved the graph-free
  project-file fallback on the loaded store, and MCP explicit recall still
  degraded under the explicit budget while returning cached project-file packets
  instead of an empty payload. The recall-quality sample was recorded with
  `source=codex_dogfood`; one duplicate cached Helix packet was labeled noisy.
  The validator passed again, and the lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-073541/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A follow-up pass found that live capture latency was not only cue indexing:
  `create_episode` and `upsert_episode_cue` could both stall on the loaded native
  store, and the worker was projecting the same observed episode twice by
  consuming both the raw queued event and the cue-scheduled event. Cue
  persistence is now bounded by `capture_cue_store_timeout_ms`, cue storage
  continues in the background when it misses that bound, and the worker ignores
  raw queued projection work when cue-layer routing is enabled while still
  preserving the system-discourse skip guard.
- Recent packet-cache fallbacks now de-duplicate packet payloads by provenance
  before degraded recall presents them. After reinstall/restart, a live REST
  observe sample returned with `captureStore≈1008ms` and
  `cueStoreTimeout≈1002ms`, MCP context hit warmed project cache in about 0.3ms,
  and MCP/AXI recall still degraded under the explicit search budget while
  returning three distinct cached project-file packets. The recall-quality
  sample was recorded with `source=codex_dogfood`.
- The current runtime then passed
  `python3 scripts/dogfood_startup_validation.py --json`, and
  `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260526-101234/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A cache-satisfaction pass now lets explicit recall skip deep search when
  recent cached project/identity packets strongly match the query. After
  warming context, `engram axi recall "native PyO3 Helix install" --timeout 10
  --json` returned `status=ok`, `skipReason=cache_satisfied`, three packets,
  and `durationMs≈25`; MCP recall reported `query_time_ms≈2.2` and
  `budget.duration_ms≈1.9` with the same `cache_satisfied` lifecycle. The
  current runtime then passed the startup validator, and the lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-104900/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`.
- Topic-specific context now rejects irrelevant stable project-home cache before
  it can short-circuit a useful answer. When stable cached packets do not match
  the topic, MCP context warms bounded local project-file packets under the
  exact topic instead of returning generic README/Makefile context. Live
  evidence on the loaded native store: AXI context for
  `"cache relevance miss loaded store PyO3 recall docs"` returned five
  project-file packets in about 1.95s CLI wall; follow-up AXI recall returned
  `status=ok`, `skipReason=cache_satisfied`, `durationMs=3.1908`, and three
  packets; hot MCP `get_context` returned from packet cache with
  `duration_ms=0.0809`, and MCP `recall` reported `query_time_ms=1.8`.
- The same reinstall/restart exposed that the loaded 4G native store can exceed
  the previous 90s `engramctl start` readiness window even though the process
  later becomes healthy. Local Helix startup waits and the startup-matrix health
  wait now default to 180s so a slow vector-integrity warmup is not reported as
  a failed start. The updated lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-114939/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A subsequent MCP write-path pass found two more latency sources. First, MCP
  observe/remember were waiting on conversation live-turn fingerprinting and
  recall middleware side effects after capture; those side effects are now
  bounded at the write surface and continue/degrade independently. Second,
  REST-mounted MCP started its own EpisodeWorker/cue-outbox/Redis publisher in
  the same process as the REST runtime, which could duplicate projection work;
  embedded MCP now marks background runtime ownership external and skips those
  loops while standalone MCP behavior is unchanged. Live MCP `observe` after
  reinstall returned in 2.8s wall with `capture_store=32ms`,
  `cue_store_timeout=1001ms`, `live_turn=78.608ms`, and
  `recall_middleware_timeout=250.915ms`; MCP init logged
  `mcp_background_managed_externally` instead of `mcp_worker_start`.
- `/api/consolidation/status` now bounds the latest-cycle read and returns a
  degraded status packet if the loaded store is slow. Live evidence returned in
  about 0.56s with `latest_cycle_status=timeout`,
  `degraded=true`, and `skip_reason=latest_cycle_timeout`.
- The confirmed lifecycle matrix then exposed a health-starvation issue during
  optional consolidation cross-encoder scoring: a missing local cross-encoder
  model was retried repeatedly during consolidation and could make `/health`
  miss its 3s validation timeout. Cross-encoder initialization failures are now
  negative-cached per process, so consolidation falls back after the first
  unavailable-model result. After reinstall/restart, the full validator passed
  with `14 pass, 0 warn, 0 fail, 0 skip`, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-122954/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`.
- Cache-satisfied explicit recall now scores aggregate packet coverage, so a
  relevant set of cached project packets can skip loaded-store recall even when
  no single packet covers the entire query. Live evidence: after context warmed
  project-file packets for
  `"native PyO3 dogfood cue store timeout deep recall"`, AXI recall returned
  `status=ok`, `skipReason=cache_satisfied`, `durationMs=1.3717`, and about
  0.32s CLI wall. MCP recall returned the same three cached packets with
  `skip_reason=cache_satisfied`, `query_time_ms=0.6`, and no degraded state.
- Storage count refreshes are now non-sticky when Helix counts are slow. A
  live `?live=true` storage refresh that misses its caller timeout keeps the
  count read running in the background, but that background refresh has its own
  30s cap and exposes `countsRefreshStatus` so operator tools can avoid
  launching duplicate count scans. Live evidence: a 0.5s live storage refresh
  returned `cached_timeout` with `countsRefreshStatus=running`, `engramctl
  storage` returned in about 0.21s while the refresh was in flight, and after
  30s the runtime logged `background storage count refresh timed out after 30.0
  seconds` with default `/api/storage` back to `countsRefreshStatus=idle`.
- Empty degraded recall now produces an explicit diagnostic packet. REST, AXI,
  and MCP forced-miss probes for
  `"zzzzquasarflux xylofract wugplinth nonmatching"` returned zero results,
  one `recall_diagnostic` packet, `skipReason=recall_timeout`, and stage timing
  evidence instead of an empty response. The fast cue/episode fallback also now
  filters unrelated hits before returning them as timeout rescue results; live
  REST evidence reported `fallbackStatus=filtered`, while AXI/MCP reported
  `fallbackStatus=timeout` when the fallback itself missed its cap. After
  warming project context with `engram axi context --project
  /Users/konnermoshier/Engram`, the useful query
  `"native PyO3 dogfood cue store timeout deep recall"` returned
  `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs=3.1484`.
- Helix fast fallback now uses BM25-only cue/episode search instead of the
  normal hybrid vector path. GraphManager prefers
  `search_episode_cues_fast` / `search_episodes_fast` when a backend declares
  them, while keeping the existing search methods as fallback. Live evidence on
  the loaded native store: the forced-miss REST probe returned
  `fallbackStatus=miss`, `recallFallback=5.0037ms`, and
  `recallRetrieveCancelled=1200.5598ms`; the MCP probe returned
  `recall_fallback=0.6211ms` and
  `recall_retrieve_cancelled=1201.9258ms`. This moved the timeout-rescue path
  from hundreds of milliseconds / timeout to single-digit milliseconds and
  confirmed the remaining miss cost sits inside deep retrieval.
- Deep retrieval substages are now separately bounded and named in diagnostics:
  stats, primary search, activation/graph pools, planner search,
  episode/cue/chunk search, activation state, entity-name fallback, spreading,
  entity attributes, and graph-structural similarity. After reinstall/restart,
  the loaded-store forced-miss REST probe returned `status=ok`, no degraded
  timeout, `durationMs=801.8291`, and `recallSearch=742.4659ms`; the key
  bounded misses were `recallPrimarySearchTimeout=301.5487ms` and
  `recallEntityMatchTimeout=74.9526ms`. A noisier prior sample exposed
  `recallSpreadTimeout=250.8645ms`, which is now explicit instead of hidden
  inside the outer retrieve timer. The matching AXI forced-miss smoke returned
  `status=ok`, `durationMs=617.67`, zero packets/results, and no degraded
  timeout.
- Fast fallback is now a real timeout rescue instead of an always-paid
  preflight. After reinstall/restart and packet-cache clear, cold project-topic
  recall for `"loaded-store recall context performance dogfood"` returned
  `status=ok` with no degraded timeout: AXI `durationMs=591.3883` and
  `fallbackStatus=not_run`; REST `durationMs=810.7498`,
  `recallSearch=810.687ms`, and `fallbackStatus=not_run`. Once context warmed
  project-file packets, recall hit the packet cache across surfaces: AXI
  `cache_satisfied` in `durationMs=1.7864`, MCP context cache hit in
  `duration_ms=0.0687`, and MCP recall `cache_satisfied` in
  `duration_ms=1.6463`. The lifecycle matrix evidence for this pass is
  `/private/tmp/engram-dogfood-startup-20260526-140146/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The same live dogfood turn exposed capture-write contention from deferred cue
  storage. Before the fix, MCP observe spent about 59.4s in `capture_store`,
  REST observe later reported `captureStore=31616ms`, and AXI observe timed out
  at 15s. Deferred cue persistence is now serialized so background cue writes
  cannot occupy all native storage workers. After reinstall/restart, REST
  observe returned with `captureStore=58ms` and `cueStore=32ms`; AXI observe
  completed in about 0.39s wall; MCP observe returned with
  `capture_store=23ms`, `cue_store=55ms`, and the intended
  `recall_middleware_timeout` side-effect cap. Focused tests passed with
  `255 passed, 2 skipped`, and startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip`.
- Repeated observe soak exposed one more capture tax: each capture still waited
  about 1000ms when serialized cue persistence was already busy. Capture now
  queues cue persistence immediately when the cue-store lane is occupied.
  After reinstall/restart, six sequential REST observes returned in about
  70-119ms with no cue-store timeouts; a concurrent mixed observe/recall soak
  returned six observes in about 47-153ms, with queued cue writes reporting
  `cueStoreQueued=0.0` and the active cue write reporting `cueStore=74ms`.
  Project recall remained `cache_satisfied` during the same run with
  `durationMs=1.4748`. A fresh no-evidence recall still degraded at the outer
  recall budget, but returned a `recall_diagnostic` packet; remaining latency
  work should trace the hidden retrieve tail that still consumes that true-miss
  path.
- Cue persistence acknowledgement is now separated from slower cue follow-up
  work. The bounded capture wait now ends when the cue is durably persisted or
  queued, while projection-state sync and cue vector indexing continue in
  serialized background lanes. After reinstall/restart, six sequential REST
  observes returned in about 100-391ms with no cue-store timeout; a concurrent
  observe/recall soak returned five queued observes in about 159-187ms and one
  active cue write in about 487ms with `cueStore=307ms`, again without
  `cueStoreTimeout`. MCP `observe` returned in about 0.57s with
  `capture_store=67ms`, `cue_store=54ms`, and only the expected
  `recall_middleware_timeout≈251ms`. Explicit recall still has a loaded-store
  miss tail around 0.8s on this store, but it now returns bounded `status=ok`
  responses with `fallbackStatus=not_run` instead of empty degraded timeouts.
  Focused backend tests passed with `258 passed, 2 skipped`, startup validation
  passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was skipped, and
  the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-143550/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Project-file fallback now emits topic-matched snippets from a bounded
  50k-character scan window instead of ranking long docs by body text while
  caching only their early headings. Explicit recall also returns filtered
  context packets when live recall succeeds but finds no direct results. After
  reinstall/restart and packet-cache clear, AXI context for
  `"capture_store cue_store recall_middleware_timeout serialized cue persistence dogfood evidence"`
  surfaced late evidence snippets including `capture_cue_store_timeout_ms`,
  `capture_store=23ms`, and `cue_store=55ms`. The follow-up AXI recall returned
  `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs=1.4963`; MCP recall returned the same cache-satisfied shape with
  `duration_ms=1.4921`. This turned the prior 600ms empty response / 1.4s
  degraded retry into a useful packet-cache hit. Focused tests passed with
  `87 passed`, ruff and `git diff --check` passed, and startup validation
  passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was skipped.
- Explicit recall now has a configurable
  `recall_budget_explicit_search_ms` search-stage budget, defaulting to 900ms,
  and timeout-rescue fallback is capped by
  `recall_fast_fallback_timeout_ms`, defaulting to 150ms. Zero-score
  non-semantic candidate pools short-circuit before activation/spreading work.
  After reinstall/restart on the 4G native PyO3 dogfood store, fresh REST
  forced-miss probes returned diagnostic packets in about 1.09-1.12s wall
  (`durationMs≈1057-1087`, `maxSearchMs=900`,
  `recallFallback≈151ms`) instead of the earlier 1.35-1.86s miss tail under the
  1200ms search budget. A real no-evidence project query returned
  `status=ok` in about 0.60s with `fallbackStatus=not_run`. AXI warmed recall
  remained `cache_satisfied` with `durationMs=0.7747`, and MCP warmed recall
  remained `cache_satisfied` with `duration_ms=0.8048`. MCP forced-miss recall
  returned one diagnostic packet with `duration_ms=1054.6106`,
  `recall_search_ms=902.0106`, and `recall_fallback_ms=152.539`. Focused tests
  for recall surface, budgets, candidate pools, and retrieval passed with
  `27 passed`; the broader focused backend suite passed with
  `159 passed, 2 skipped`; ruff and `git diff --check` passed; startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped.
- Follow-up tuning lowered the default explicit search budget to 650ms and the
  timeout-rescue fallback cap to 50ms, bounded reranker/MMR/post-processing
  stages, and made cache satisfaction separator-aware for compound query terms
  such as `cue_store`. After reinstall/restart on the 4G native dogfood store,
  a forced-miss AXI recall returned one diagnostic packet with
  `durationMs=709.785`, `maxSearchMs=650`, and `fallbackStatus=timeout`; REST
  forced miss returned `durationMs=704.1293` in about 730ms wall. The warmed
  project query
  `"capture_store cue_store recall_middleware_timeout serialized cue persistence dogfood evidence"`
  skipped live search: AXI recall returned `status=ok`,
  `skipReason=cache_satisfied`, three packets, and `durationMs=1.097`; MCP
  recall returned `duration_ms=0.9786`; REST recall returned
  `durationMs=0.9048` in about 3ms wall. The installer also now preserves an
  existing custom `ENGRAM_HELIX__DATA_DIR` across `setup --mode helix`, which
  prevents restamping a dogfood store back to the empty default native path.
  Focused backend tests passed with `193 passed, 2 skipped`, ruff and
  `git diff --check` passed, and startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip` when doctor was skipped.
- A degraded-recall usefulness pass made the timeout path return recent
  identity/project packets when strict query filtering misses. Strict matching
  still controls the `cache_satisfied` fast skip; this fallback is only used
  after live recall degrades. After reinstall/restart on the same 4G store, the
  warmed broad project query
  `"native PyO3 dogfood runtime performance hardening packet cache loaded store current goal"`
  returned `cache_satisfied` on AXI (`durationMs=1.1821`), REST
  (`durationMs=1.123`, about 19ms wall), and MCP (`duration_ms=0.6021`). A
  deliberately unrelated query
  `"zzzzheliodogfood impossibleterm 20260526 recent fallback proof"` still
  degraded under the 650ms search budget, but REST, AXI, and MCP each returned
  three recent `project_home` packets instead of a diagnostic-only payload; MCP
  reported `duration_ms=700.989`. MCP context for the same warmed topic returned
  from packet cache in `duration_ms=0.0654`. Focused backend tests passed with
  `194 passed, 2 skipped`, ruff and `git diff --check` passed, startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-155840` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Cold degraded recall now has a bounded project-file fallback when the packet
  cache is empty. Recall surfaces accept `project_path`, AXI recall exposes
  `--project`, and the no-argument MCP/REST fallback conservatively uses the
  server working directory only when it looks like a project. The recall-specific
  project-file scan is capped separately from full context fallback at 40
  candidates and 12k chars per topic scan. After reinstall/restart and
  `engram axi packet-cache clear`, MCP recall for
  `"Engram dogfood Codex real sessions evidence performance hardening current goal cold bounded fallback 20260526"`
  returned three `project_home` packets with `duration_ms=751.294` and
  `project_file_recall_fallback=46.122ms`. REST cold no-project recall returned
  three project packets with `durationMs=928.1492` and
  `projectFileRecallFallback=222.1487`; AXI explicit-project recall returned
  three project packets with `durationMs=741.9536`. Focused backend tests passed
  with `197 passed, 2 skipped`, ruff and `git diff --check` passed, startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-161716` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The cold project-file fallback now reads bounded file prefixes instead of
  loading whole files before slicing. This matters on the Engram repo because
  docs such as `docs/CURRENT_HANDOFF.md` are hundreds of KB and were making the
  fallback scan itself expensive. After reinstall/restart and packet-cache
  clear, live Codex MCP recall for
  `"Engram dogfood real Codex sessions replay evidence commands memory value performance current goal cold prefix read 20260526"`
  returned three project packets with `duration_ms=763.5911` and
  `project_file_recall_fallback=57.7283ms`, down from the prior
  `project_file_recall_fallback≈1979.5ms`. REST cold no-project recall returned
  three project packets with `durationMs=760.8558` and
  `projectFileRecallFallback=54.9111`; AXI explicit-project recall returned
  three project packets with `durationMs=739.5595`. Focused backend tests passed
  with `197 passed, 2 skipped`, ruff and `git diff --check` passed, and startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped. The confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-162643` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Real Codex dogfood evidence is now prepared for the May 26 transcript at
  `/Users/konnermoshier/.codex/sessions/2026/05/26/rollout-2026-05-26T12-12-39-019e65b4-0ea8-7dd0-ab47-4b7d4366ef5c.jsonl`
  with AXI trace input from `/Users/konnermoshier/.engram/axi-hook-runs.jsonl`.
  `engram dogfood prepare` wrote
  `/private/tmp/engram-dogfood-review-20260526-codex-ms`, replayed 7 redacted
  user turns across `off,startup,cached,gated_lite,gated_medium,deep`, and
  measured 46 AXI trace entries (`codex=26`, `claude-code=20`, `home=38`,
  `context=8`, average `1012.1304ms`, p95 `2520ms`, max `4936ms`,
  `timeout_count=1`). This trace spans earlier offline/error/degraded states, so
  it is useful adoption evidence but not a clean post-fix latency benchmark. The
  review bundle is intentionally still `needs_labels` with 7 unreviewed turns,
  0 recall labels, and 0 session labels; no human-label lift is claimed yet.
- Dogfood trace evidence can now be filtered with `--trace-since` and
  `--trace-project-only`, so post-fix latency numbers do not get mixed with old
  offline/error/degraded hook rows. A filtered MachineShopScheduler bundle at
  `/private/tmp/engram-dogfood-review-20260526-codex-ms-filtered` kept 5 of 49
  trace rows after `2026-05-26T19:13:00Z` for
  `/Users/konnermoshier/MachineShopScheduler`: all 5 were healthy startup
  `home` rows, with average `42.8ms`, p95 `84ms`, max `84ms`, and no degraded
  or timed-out rows. A current Engram trace-only replay at
  `/private/tmp/engram-dogfood-review-20260526-current-axi-trace/dogfood-replay.json`
  kept 3 post-fix Codex follow-up rows after `2026-05-26T23:41:55Z`: `home`,
  `context`, and `recall`, all healthy/ok with no degraded/timeouts. The first
  filtered slice showed AXI `context` at `1877ms` for a cold project-file packet
  rebuild, while follow-up AXI recall was cache-satisfied with
  `durationMs=4.1945`; MCP context then hit packet cache in `duration_ms=0.041`,
  and MCP recall hit cache in `duration_ms=1.7387`.
- The context fallback instrumentation now reports the actual project-file
  fallback cost and includes it in `budget.duration_ms`, instead of recording
  `project_file_fallback=0.0` before the fallback work happened. After
  reinstall/restart on the dogfood runtime, REST cold context after packet-cache
  clear returned 5 project packets in `69.58ms` wall with
  `durationMs=52.5488`, `projectFileFallback=51.7395`, and
  `cacheRelevanceMiss=0.8093`. A repeated AXI cold context trace for the Engram
  project returned in `66ms`, showing the earlier `1877ms` context trace was not
  representative of the current post-fix path. Focused context and dogfood
  replay tests passed with `67 passed`, ruff and `git diff --check` passed,
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-164948` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Capture now uses an agent-bounded default cue wait:
  `capture_cue_store_timeout_ms=250` instead of `1000`. Raw episode durability
  remains synchronous, but a slow cue upsert no longer holds a live agent for a
  full second before falling into the background lane. A focused regression test
  covers the default timeout by forcing slow cue persistence, returning capture
  under 500ms, then proving cue persistence drains afterward. After
  reinstall/restart, REST observe probes returned in `110.5ms`, `87.5ms`, and
  `55.3ms`; a diagnostic REST observe reported `captureStore=16ms`,
  `cueStore=49ms`, and `cueIndexEnqueue=2ms`. MCP `observe` reported
  `capture_store=17ms`, `cue_store=39ms`, `live_turn=77.6472ms`, and the
  remaining write-side `recall_middleware_timeout=253.0875ms`; AXI observe
  completed in about `0.30s` wall. Focused capture tests passed with
  `27 passed`, ruff passed, startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip` when doctor was skipped, and the confirmed
  lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-170217` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- MCP write-side enrichment now has a `75ms` inline budget instead of `250ms`.
  This keeps fast `recall_lite` enrichment available for `observe`/`remember`,
  while preventing slower session-prime or auto-recall work from adding a fixed
  quarter-second tax to every write. After reinstall/restart, MCP `observe`
  reported `capture_store=33ms`, `cue_store=55ms`,
  `live_turn_timeout=78.6785ms`, and
  `recall_middleware_timeout=77.4614ms`; a repeated call reported
  `capture_store=21ms`, `cue_store=56ms`,
  `live_turn_timeout=80.5909ms`, and
  `recall_middleware_timeout=77.2381ms`. Focused capture-surface tests passed
  with `15 passed`, the capture-focused pair passed with `28 passed`, ruff
  passed, startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip`, and
  the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-171106` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Foreground capture starvation was still possible when installed AutoCapture
  hooks started consolidation from a session-end event. The live regression was
  visible without synthetic load: MCP `observe` reported `capture_store=2556ms`
  and `cue_store_timeout=252ms`, and direct REST observe probes reported
  `captureStore=1893ms` followed by a `16346ms` `captureStore` sample while the
  runtime reported `consolidation.status.is_running=true`. Session-end
  AutoCapture now captures only the session-end marker; consolidation remains a
  runtime-scheduled/background concern instead of a user-hook side effect.
  `install_hooks()` now refreshes stale managed Engram AutoCapture scripts while
  preserving custom user scripts, and the local
  `/Users/konnermoshier/.engram/hooks/session-end.sh` was refreshed to remove
  `/api/consolidation/trigger`. After reinstall/restart, eight REST observes
  stayed at `55-82ms` wall with `captureStore=13-29ms`, `cueStore=31-45ms`,
  and no cue timeouts. MCP `observe` returned with `capture_store=19ms`,
  `cue_store=40ms`, `live_turn_timeout=75.7847ms`, and
  `recall_middleware_timeout=76.332ms`. A session-end hook smoke posted only
  `/api/knowledge/auto-observe`, AXI context returned useful project packets in
  about `0.39s` wall, AXI recall was `cache_satisfied` with
  `durationMs=0.7873`, MCP context hit packet cache with `duration_ms=0.0573`,
  and MCP recall stayed `cache_satisfied` with `duration_ms=3.452`. Focused
  tests passed with `33 passed`, ruff passed, and startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip` when doctor was skipped. The confirmed
  lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-172521` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The 2026-05-28 recall-latency pass closed the post-restart no-evidence
  explicit recall tail. Explicit recall can now treat same-project home packets
  and identity packets as weak cached context after fast preflight misses or
  times out; when those packets exist, preflight is capped to the shorter fast
  fallback timeout and the deep recall path is skipped. Project-scoped recall
  also syncs the persistent packet cache before checking recent context packets,
  so a fresh process can reuse persisted project-home packets instead of paying
  the deep loaded-store tail. Live evidence after reinstall/restart:
  `zzpersist noartifact yonderplasm quibbleflux 20260528 final true miss tail`
  returned three project packets in `100.1129ms` with
  `preflight_timeout_context_packet_fallback`, no degradation, and no budget
  miss. After the lifecycle matrix restarted the runtime again, `zzaftermatrix
  ...` still returned useful project context in `101.9488ms`. The final live
  value report measured read-path p95 `65.7748ms`, cache hit rate `1.0`, and
  zero read budget misses/degraded reads/timeouts. `python3
  scripts/dogfood_startup_validation.py --json` passed, and
  `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260528-071304` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Focused retrieval tests passed with
  `94 passed, 2 skipped`, ruff passed, and `git diff --check` is clean.
- A follow-up real Codex turn after commit `1b98a19` confirmed the cache/fallback
  behavior holds outside the immediate implementation turn. MCP `get_context`
  returned useful project packets in `249.4908ms` with no degradation, MCP
  `recall` hit cache in `1.4862ms`, AXI same-topic context hit project-file
  cache rescue in `2.5496ms`, AXI same-topic recall was `cache_satisfied` in
  `32.0827ms`, and a fresh no-evidence recall returned a project packet in
  `101.3667ms`. A brand-new context-miss topic then hit cache rescue in
  `4.3832ms`; MCP `get_context` hit cache in `0.2735ms`, and MCP `recall` was
  `cache_satisfied` in `45.9587ms`. The live value report still showed zero
  read budget misses, degraded reads, or timeouts.
- A third resumed Codex/restart sample separated one-time project-file cache
  creation from persistent cache rescue. A fresh topic-specific MCP context
  probe for write-path capture/observe work returned useful packets but spent
  `581.6269ms` overall with `project_file_fallback=550.3093ms`; that call
  created the stable `project_home` cache entry. After a real
  `engramctl stop && engramctl start` restart to PID `35144`, the first fresh
  AXI context topic hit project-file cache rescue in `2.3597ms`, AXI recall
  returned three packets in `106.2332ms` with
  `preflight_timeout_context_packet_fallback`, MCP `get_context` hit packet
  cache in `0.0617ms`, and MCP `recall` was `cache_satisfied` in `89.6023ms`.
  The post-restart value window reported read-path p95 `106.2332ms`, cache hit
  rate `1.0`, and zero read budget misses, degraded reads, or timeouts.
- The next soft-wait pass fixed the hidden reason a later fresh MCP context
  probe could still pay a whole slow project-file scan. Loaded-store context
  preflight now waits only its soft budget and no longer blocks until the
  project-file scan wins; when that scan is still pending, stable same-project
  project-file packets can rescue the response and the scan caches the exact
  topic afterward. Focused regression coverage forces the slow-preflight plus
  slow-scan case and verifies `project_file_cache_rescue`. Live reinstall/restart
  on LaunchAgent PID `40680` showed the first fresh MCP topic with no stable
  sidecar entry still rebuilt in `937.6488ms`, then AXI fresh context used
  `project_file_cache_rescue` in `10.3801ms`, exact repeat context hit cache in
  `0.0413ms`, and fresh MCP context stayed bounded without degradation at
  `104.0038ms` and `138.8205ms`. After the final no-project guardrail
  reinstall/restart, PID `41982` is healthy and a fresh AXI context probe used
  `project_file_cache_rescue` in `2.238ms`.
- A continuation pass verified the current committed dogfood runtime rather
  than finding another degradation to patch. HEAD `78aa7ed` was clean and
  pushed; startup validation passed; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260528-074024` with
  `13 pass, 0 warn, 0 fail, 0 skip`. After the matrix restart to PID `43378`,
  AXI context returned project-file packets in `38.2785ms`, AXI recall found a
  real cue packet in `11.8581ms`, a forced miss returned a project packet in
  `102.2185ms`, MCP `get_context` returned project-file packets in
  `143.7264ms`, and MCP `recall` was `cache_satisfied` in `2.2772ms`. The value
  report's read path still showed a single matrix-era MCP context p95 of
  `581.6838ms`, but no read budget misses, degraded reads, or timeouts.

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
LABELS="/private/tmp/engram-dogfood-review-20260527-active-codex/dogfood-labels.json"
REPLAY="/private/tmp/engram-dogfood-review-20260527-active-codex/dogfood-replay.json"
EVIDENCE="/private/tmp/engram-dogfood-review-20260527-active-codex/human-labels.json"
```

The bundle has 80 labelable turns, 2 reviewed recall labels, 1 reviewed session
label, 78 skipped turns, and an exported `human-labels.json`. The reviewed
session label records the operator-approved verdict for this local continuity
sample: baseline `1.0`, memory `1.0`, open-loop expected/recovered, and no
measurable Engram lift. To continue reviewing more turns later, run:

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
uv run engram dogfood export-evidence --labels "$LABELS" --out "$EVIDENCE" --source native_dogfood_harness --client Codex --captured-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --labeler codex-human-review
uv run engram dogfood closeout --labels "$LABELS" --human-label-artifact "$EVIDENCE" --sqlite-path "$HOME/.engram/engram.db" --mode helix --require-ready
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram dogfood finalize --labels "$LABELS" --replay-report "$REPLAY" --human-label-artifact "$EVIDENCE" --sqlite-path "$HOME/.engram/engram.db" --source native_dogfood_harness --client Codex --captured-at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --labeler codex-human-review --mode helix
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
  `/private/tmp/engram-dogfood-review-20260527-active-codex/dogfood-labels.json`
  reports `status=ready_for_native_memory_value`, 2 reviewed recall samples,
  1 reviewed session sample, and measured human-label evidence from
  `/private/tmp/engram-dogfood-review-20260527-active-codex/human-labels.json`.
- `human-labels.json` is marked `humanLabeled=true`, `source=native_dogfood_harness`,
  `client=Codex`, `labeler=codex-human-review`, and contains 2 recall samples
  plus 1 session sample.
- `engram dogfood finalize` completed with `status=finalized`, imported the
  reviewed labels, exported evidence, checked closeout, and ran native
  memory-value evaluation.
- Native PyO3
  `engram evaluate --memory-value --require-memory-value --require-human-label-evidence`
  passes against the reviewed dogfood evidence with
  `memory_value.status=measured`.
- AXI `engram axi value --server-url http://127.0.0.1:8100 --timeout 20 --json`
  reports `status=measured`.

Latency dogfood evidence:

- 2026-05-26 native PyO3 follow-up made the fallback/read side and write side
  cheaper under real Codex use. Project-file fallback packets are now cached
  in-process only, avoiding durable SQLite writes for one-off topic packets.
  After packet-cache clear, REST context rebuilt useful project packets in
  `60.2ms` wall (`durationMs=58.0146`, `projectFileFallback=56.3851`), and MCP
  context rebuilt useful project packets with `duration_ms=49.2701` in the MCP
  budget. Timed-out live storage count refreshes now cancel instead of running
  a background graph scan after the operator response returns; after
  `/api/storage?live=true&timeoutSeconds=1` reported `countsRefreshStatus=idle`,
  six REST observes stayed between `62.05-92.67ms` wall with
  `captureStore=19-51ms` and `cueStore=33-50ms`. Mounted MCP now reuses the
  REST app's `GraphManager` rather than lazily creating a second native runtime;
  the post-matrix MCP observe on PID `54432` returned in `0.2473s` wall with
  `capture_store=19ms`, `cue_store=29ms`, and no cue timeout. Validation:
  focused tests `53 passed`, startup validation `13 pass, 0 warn, 0 fail,
  1 skip`, lifecycle matrix `/private/tmp/engram-dogfood-startup-20260526-175129`
  with `13 pass, 0 warn, 0 fail, 0 skip`, ruff passed, and `git diff --check`
  passed.
- 2026-05-26 bounded fallback follow-up tightened cold project-file fallback
  variance. The prior same-topic REST/MCP probes exposed fallback samples in
  the `623-930ms` range and one MCP context budget miss despite useful packets.
  The fallback now pre-ranks path-relevant files and scans a smaller topic
  window, so it keeps `docs/memory-value-latency-plan.md` and
  `docs/dogfood-startup-validation-goal.md` useful without reading every
  fallback candidate deeply. After reinstall/restart, five REST context calls
  with packet-cache clear stayed between `54.91-59.94ms` wall, MCP context
  returned useful packets with `duration_ms=64.5478`, AXI recall was
  `cache_satisfied` in `0.5715ms`, MCP recall was `cache_satisfied` in
  `0.8831ms`, AXI value reported `p95_added_latency_ms=250.6712`, and
  post-matrix MCP context on PID `57400` was `duration_ms=52.287`. Validation:
  focused tests `54 passed`, startup validation `13 pass, 0 warn, 0 fail,
  1 skip`, lifecycle matrix `/private/tmp/engram-dogfood-startup-20260526-180521`
  with `13 pass, 0 warn, 0 fail, 0 skip`, ruff passed, and `git diff --check`
  passed.
- 2026-05-26 wrapper follow-up closed the remaining live MCP fallback miss. The
  public fallback wrapper still passed `topic_scan_chars=50_000`, bypassing the
  helper's new `16_000` default; one live MCP context sample after restart
  reported `project_file_fallback=1494.5866ms` and a budget miss. After the
  wrapper default was made bounded and the runtime was reinstalled, the same
  topic returned useful packets after packet-cache clear with
  `duration_ms=40.1795` and `project_file_fallback=39.1043`. AXI recall was
  `cache_satisfied` in `1.3291ms`, MCP recall was `cache_satisfied` in
  `0.838ms`, post-matrix AXI value reported `p95_added_latency_ms=74.0474` over
  4 measured operations with no budget misses, and post-matrix MCP context on
  PID `60090` was `duration_ms=56.5303` with
  `project_file_fallback=52.9783`. Validation: focused tests `55 passed`,
  startup validation `13 pass, 0 warn, 0 fail, 1 skip`, lifecycle matrix
  `/private/tmp/engram-dogfood-startup-20260526-181432` with
  `13 pass, 0 warn, 0 fail, 0 skip`, ruff passed, and `git diff --check`
  passed.
- 2026-05-26 explicit-recall fallback prebuild reduced the cold degraded recall
  tail without caching local-file packets over successful loaded-store recall.
  Before the patch, cold REST recall after packet-cache clear returned useful
  project packets but took `2706.5797ms` wall with `durationMs=2703.8135`
  because `projectFileRecallFallback=1993.9967` ran after the 650ms
  loaded-store timeout. The fallback now builds uncached project-file packets in
  a side task while loaded-store recall runs, then caches and records them only
  if the live recall actually degrades. After reinstall/restart, the same cold
  REST path returned 3 useful packets in `705.2311ms` wall with
  `durationMs=702.7804`, `projectFileRecallFallbackWait=0.198`, and
  `projectFileRecallFallback=41.8739`. AXI cold recall returned 3 useful
  packets with `durationMs=706.3645`; MCP cold recall returned 3 useful packets
  with `duration_ms=746.481`, `project_file_recall_fallback_wait=0.1875`, and
  `project_file_recall_fallback=282.2759`. The response can still be marked
  `recall_timeout` when the loaded-store search stage exhausts 650ms, but the
  timeout path is now useful and bounded. Validation: focused tests
  `80 passed`, startup validation `13 pass, 0 warn, 0 fail, 1 skip`, lifecycle
  matrix `/private/tmp/engram-dogfood-startup-20260526-183141` with
  `13 pass, 0 warn, 0 fail, 0 skip`, post-matrix AXI value
  `p95_added_latency_ms=90.4855` with no budget misses over 4 fresh operations,
  ruff passed, and `git diff --check` passed.
- 2026-05-26 empty-success recall follow-up made the no-evidence path useful
  instead of merely fast. Cold explicit recall that completes under budget with
  zero loaded-store items now consumes the already-started project-file rescue
  packet task instead of returning an empty `ok` payload. The rescue task now
  ranks the full project-file candidate path list before reading and limits
  explicit-recall fallback reads to 16 files with a `6000` character topic
  window, reducing GIL pressure on MCP turns. After reinstall/restart, cold
  no-project REST recall returned 3 useful packets with `durationMs=705.5507`,
  `projectFileRecallFallbackWait=0.1941`, and
  `projectFileRecallFallback=8.4883`; the repeated REST recall hit cache with
  `durationMs=0.6293`. AXI cold recall returned 3 useful packets with
  `durationMs=721.2843`, and MCP cold recall returned 3 useful packets with
  `duration_ms=704.7271`, `project_file_recall_fallback_wait=0.1659`, and
  `project_file_recall_fallback=16.8017`. Validation: focused tests
  `82 passed`, startup validation `13 pass, 0 warn, 0 fail, 1 skip`, lifecycle
  matrix `/private/tmp/engram-dogfood-startup-20260526-184734` with
  `13 pass, 0 warn, 0 fail, 0 skip`, post-matrix AXI value
  `p95_added_latency_ms=92.414` with no budget misses over 4 fresh operations,
  ruff passed, and `git diff --check` passed.
- 2026-05-26 loaded-store recall tail follow-up bounded the candidate-hit path
  that was still degrading after project-file fallback became useful. When graph
  preflight probes time out, explicit recall now caps primary search to `150ms`,
  skips secondary graph-heavy enrichers, avoids noop-reranker graph reads, caps
  primary materialization graph reads to `15ms`, and skips near-miss
  materialization in that slow-graph state. Before the materialization caps, the
  same cold REST recall degraded at `durationMs=704.792` with
  `recallRetrieve=371.8813` and `recallMaterializeCancelled=278.1135`. After
  reinstall/restart, cold REST recall returned `ok` with 3 project packets in
  `durationMs=383.8734`, with `recallRetrieve=297.0664`,
  `recallMaterialize=85.4673`, and `recallPostProcess=0.3265`. Repeated REST
  recall was `cache_satisfied` in `durationMs=43.9772`; cold AXI recall returned
  `ok` with 3 packets in `durationMs=269.5134`; local MCP recall without a
  project path returned `ok` in `duration_ms=227.1599` but no project packets.
  Validation: focused tests `194 passed, 2 skipped`, startup validation
  `13 pass, 0 warn, 0 fail, 1 skip`, lifecycle matrix
  `/private/tmp/engram-dogfood-startup-20260526-193243` with
  `13 pass, 0 warn, 0 fail, 0 skip`, AXI value
  `p95_added_latency_ms=382.8965` with no budget misses over 13 measured
  operations, ruff passed, and `git diff --check` passed.
- 2026-05-26 direct loaded-store fallback follow-up made the loaded-store path
  useful when deep recall finds candidates but fails to materialize them under
  the graph-read budget. Helix BM25 now has fast episode/cue record methods,
  `GraphManager.fast_recall_fallback()` can materialize those rows without
  graph scans, scoreless ordered BM25 rows get rank-based scores, and
  REST/MCP run fast fallback on empty successful recalls before falling back to
  project files. After reinstall/restart, cold REST recall for the current
  Codex stuck-check query returned `ok`, `itemCount=5`, `packetCount=3`,
  `durationMs=657.0303`, `fallbackStatus=hit`, and
  `recallEmptySuccessFallback=46.0678`. Cold AXI recall returned `ok` with
  `result_count=5`, `packet_count=3`, and `durationMs=595.7207`; cold MCP
  recall returned `ok` with `result_count=1`, `packet_count=1`, and
  `duration_ms=472.8815`. Validation: focused tests
  `94 passed, 23 skipped`, startup validation `14 pass, 0 warn, 0 fail,
  0 skip`, and ruff passed on touched files.
- 2026-05-26 planner, similarity, and post-process latency follow-up tightened
  the cold loaded-store path again. After graph preflight timeout, primary
  search now caps at `100ms`, semantic similarity backfill is skipped when
  primary search already timed out, and planner work is skipped/deferred for
  zero-semantic special episode/cue candidates. Recall post-processing now has
  explicit `recallConfidence` and `recallFingerprintRecord` timings,
  relevance-confidence scoring is capped at `50ms`, and returned episode/chunk
  text is not re-embedded by default because existing semantic similarity is
  already available from search. After reinstall/restart on the same native
  dogfood store, clean cold REST recall for
  `"deep recall performance loaded-store bottleneck Codex dogfood q2"` returned
  `ok`, `itemCount=5`, `packetCount=3`, `durationMs=136.3629`,
  `fallbackStatus=hit`, `recallPostProcess=0.2971`,
  `recallConfidence=0.0424`, and `recallEmptySuccessFallback=1.7816`. Clean
  cold AXI recall for the same query returned `ok`, `result_count=5`,
  `packet_count=3`, and `durationMs=377.448`; clean cold MCP recall returned
  `ok`, `result_count=5`, `packet_count=3`, `duration_ms=224.3714`,
  `recall_post_process=1.4265`, and `recall_confidence=0.0727`. Startup
  validation passed with all checks passing against LaunchAgent PID `87157`.
  Validation: focused recall/AXI/storage tests `189 passed, 23 skipped`,
  focused confidence/post-process tests `28 passed`, ruff passed, and
  `git diff --check` passed. Post-probe `engram axi value` still showed one
  budget miss in the recent mixed sample (`budget_miss_rate=0.0714`,
  `p95_added_latency_ms=767.4577`), so the broader dogfood goal remains active
  and needs more clean-session samples before closeout.
- 2026-05-26 project-file fallback prefix-cache follow-up reduced the remaining
  cold context cache-warming cost. Before the patch, topic-specific project-file
  context after packet-cache clears was useful but could spend `744-1907ms` in
  `projectFileFallback`; the same helper was fast in isolation, pointing to
  repeated synchronous file scanning in the long-lived runtime. The fallback now
  caches local file prefixes by path, mtime, size, and read limit, so packet-cache
  clears and new query variants reuse bounded project snippets. After
  reinstall/restart on LaunchAgent PID `89509`, cold REST context for a long
  Engram dogfood topic returned 5 project-file packets with
  `durationMs=44.1913` and `projectFileFallback=43.5274`; a second cold topic
  returned `durationMs=24.7345` and `projectFileFallback=24.0498`. Hot REST
  context returned in `0.0246ms`; hot REST recall was `cache_satisfied` in
  `0.5346ms`; hot MCP context returned in `0.0216ms`; hot MCP recall was
  `cache_satisfied` in `0.6661ms`. Cold loaded-store q2 recall stayed healthy:
  REST `durationMs=343.6238`, AXI `durationMs=332.4911`, and MCP
  `duration_ms=333.363`, all `ok` with 5 results and 3 packets. Startup
  validation passed all checks against PID `89509`. Validation: focused
  context/recall/AXI tests `110 passed`, ruff passed, and `git diff --check`
  passed. Post-validation `engram axi value` improved but still showed one
  recent mixed-sample budget miss (`budget_miss_rate=0.0385`,
  `p95_added_latency_ms=343.4741`), so closeout still needs more clean dogfood
  session evidence.
- 2026-05-26 session-prime and live-value follow-up made the current dogfood
  report reflect the installed runtime instead of stale saved cost samples.
  MCP session-prime now uses cached identity/project packets only and records
  a cheap `cache_miss` skip when no packets are ready, avoiding a duplicate full
  `manager.get_context()` call inside the `250ms` startup budget. The REST
  evaluation report accepts `liveCost=True`, and `engram axi value` uses that
  live-cost path with `cost.source=live_runtime`. After reinstall/restart,
  immediate AXI value returned `operation_count=0` and `budget_miss_rate=0`;
  after post-matrix MCP/AXI probes, MCP session-prime returned cached project
  packets in `1.9447ms` with `budget_miss=false`, MCP recall was
  `cache_satisfied` in `0.7062ms`, AXI recall was `cache_satisfied` in
  `0.3862ms`, and final AXI value reported `operation_count=15`,
  `p95_added_latency_ms=129.8354`, `cache_hit_rate=0.8571`,
  `budget_miss_rate=0`, and `source=live_runtime`. Startup validation passed,
  and the solo lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-214331` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Focused tests passed with `131 passed`
  and 6 existing AsyncMock coroutine warnings in `tests/test_autorecall.py`;
  ruff passed.
- 2026-05-26 explicit packet-cache sharing follow-up removed a cross-surface
  cache gap. Explicit recall packet payloads now use one shared
  `explicit_recall` cache scope instead of source-specific scopes, while
  operation metrics still attribute latency to `axi_recall`, `api_recall`, or
  `mcp_recall`. Auto-recall also includes the shared explicit scope in its
  packet-cache preflight, so a warmed explicit recall can satisfy MCP
  middleware without paying medium recall. After reinstall/restart and cache
  clear, cold AXI recall for
  `"Engram AXI packet cache performance dogfood recall middleware cached patch"`
  built 3 packets in `durationMs=379.5135`; MCP recall reused them with
  `skip_reason=cache_satisfied` in `duration_ms=3.3138`, and REST reused them
  with `skipReason=cache_satisfied` in `durationMs=2.2597`. The fresh live
  cost report showed `api_recall.cache_hit_rate=1.0`,
  `mcp_recall.cache_hit_rate=1.0`, shared `explicit_recall.cache_hit_rate=0.6667`,
  `medium.avg_added_latency_ms=1.048` with `skip_reason=cache_satisfied`, and
  `auto_recall_packet.cache_hit_rate=1.0`. `engram axi value` reported
  `source=live_runtime`, `operation_count=8`, `p95_added_latency_ms=379.3022`,
  `cache_hit_rate=0.8333`, and `budget_miss_rate=0`; startup validation passed
  with `14 pass, 0 warn, 0 fail, 0 skip`. Focused tests passed with
  `116 passed` and 6 existing AsyncMock coroutine warnings; ruff and
  `git diff --check` passed.
- After the same live runtime surfaced slow capture/context interference, cue
  vector indexing was moved behind a `1000ms` rework-profile quiet period. The
  cue-index outbox still makes the work durable immediately, but best-effort
  native vector writes wait until the live capture burst is quiet so they do not
  race the agent's next context/recall turn. Before the patch, MCP `observe`
  showed `capture_store=2960ms`, `cue_store_timeout=252ms`, and
  `recall_middleware_timeout=76.1255ms`; MCP `get_context` returned useful
  packets but spent `project_file_fallback=1672.5267ms` and
  `cache_relevance_miss=205.6315ms`. After reinstall/restart, REST `observe`
  returned with `captureStore=34ms`, immediate REST context after packet-cache
  clear returned useful project packets with `durationMs=61.983` and
  `projectFileFallback=61.081`, and a follow-up REST observe returned with
  `captureStore=19ms`. MCP `observe` returned in `153.8472ms` wall with
  `capture_store=39ms`, `cue_store=51ms`, `live_turn=56.6796ms`, and
  `recall_middleware=2.0812ms`; MCP session-prime loaded cached project packets
  in `1.1508ms`, and MCP `get_context` hit packet cache in `5.0142ms`. The
  cue-index outbox stayed at the pre-existing `failed=26`. Startup validation
  passed with `14 pass, 0 warn, 0 fail, 0 skip`, and the confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260526-223823` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Focused tests passed with `108 passed`;
  ruff and `git diff --check` passed.
- The next live follow-up found one remaining write-side spike: SQLite
  cue-index outbox enqueue was still synchronous even though the vector write
  itself had moved to the quiet background lane. The enqueue path now uses
  `asyncio.to_thread`, so the live Capture/Cue turn records durable cue-index
  work without blocking the event loop. After reinstall/restart on PID `10142`,
  five REST observes measured `captureStore=22/8/7/7/6ms`,
  `cueStore=28-35ms`, and `cueIndexOutboxEnqueue=1-3ms`. Three MCP observe
  probes measured `capture_store=26/10/8ms`, `cue_store=44/29/25ms`,
  `cue_index_outbox_enqueue=3/1/1ms`, and `recall_middleware=1.3-1.8ms`.
  MCP recall returned `cache_satisfied` in `1.578ms`; the fresh live cost
  report showed `auto_recall_packet.p95_added_latency_ms=1.448`,
  `budget_miss_rate=0`, and no MCP observe/context/recall degraded samples.
  The cue-index outbox stayed at the pre-existing `failed=26`. Focused tests
  passed with `108 passed`; ruff and `git diff --check` passed; startup
  validation passed all checks; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-225237` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The post-matrix cold-context follow-up found the remaining read-side spike in
  packet-cache persistence rather than project-file fallback. The first live
  MCP context after the lifecycle matrix spent `1602.966ms` with
  `project_file_fallback=1600.8288ms`; direct REST probes then isolated a
  separate sidecar miss cost where `cacheRelevanceMiss` could take about
  `974ms` before local fallback. In-process fallback builds were only about
  `20-35ms`, so the packet-cache sidecar is now non-blocking for agent turns:
  SQLite opens use a short timeout, transient locked/busy errors are treated as
  cache misses instead of disabling persistence, and hot in-memory entries stay
  usable while the sidecar is locked. After reinstall/restart on PID `13990`,
  three cold REST context probes returned
  `durationMs=31.6594/17.3439/21.6871` and
  `cacheRelevanceMiss=1.0287/0.7544/0.7145ms`; AXI context returned in about
  `0.32s` wall; MCP `get_context` returned useful project packets in
  `33.9049ms`; MCP session-prime hit packet cache in `1.3005ms`; MCP `observe`
  reported `capture_store=26ms`, `cue_store=45ms`, and
  `recall_middleware=2.0325ms`; MCP `recall` was `cache_satisfied` in
  `2.0937ms`. Fresh live cost reported `operation_count=15`,
  `p95_added_latency_ms=137.8351`, `budget_miss_rate=0`, and no degraded
  API/AXI/MCP context or recall samples. Focused tests passed with
  `143 passed`; ruff and `git diff --check` passed; startup validation passed
  all checks against PID `13990`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-230849` with
  `13 pass, 0 warn, 0 fail, 0 skip`. After that matrix restart, the runtime was
  healthy on PID `15399`; cold REST context returned useful packets in
  `61.7303ms` with `cacheRelevanceMiss=3.5983ms`; cold MCP context returned
  useful packets in `351.8508ms`, with `cacheRelevanceMiss=2.6909ms` and the
  remaining cost in local project-file fallback.
- The next pass closed that restart gap by making project-file fallback packets
  durable and project-stable. The fallback cache now writes both the exact topic
  key and the stable project key with `persist=True`, so a local fallback can
  seed later related topics across process restarts. Packet-cache hot persistent
  hits also stop writing hit-count metadata on every read; persistence sync is
  opportunistic and bounded. After reinstall/restart on PID `18399`, a seeded
  REST context fallback took `55.34ms`; after another restart, the first related
  REST context hit persisted `project_home` packets in `0.036ms` with no
  `projectFileFallback` stage. MCP `get_context` for the same topic hit
  `project_home` in `0.033ms`, MCP session-prime was `1.6562ms`, and MCP
  `observe` reported `recall_middleware=2.1446ms`. Focused tests passed with
  `144 passed`; ruff and `git diff --check` passed; startup validation passed
  all checks against PID `18827`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-232931` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix, a new topic still used
  bounded fallback in `52.6259ms`, then a related follow-up hit persisted
  `project_home` packets in `0.0365ms`.
- A follow-up Codex turn exposed a narrower first-tool startup spike: MCP
  session-prime was still allowed to sync the persistent SQLite sidecar while
  searching for recent packets, and one live call spent `770.252ms` in
  `packet_cache`. Session-prime now uses in-memory cache reads only
  (`sync_persistent=False`) for exact and recent packet lookups; normal context
  and recall paths still sync persistence when they need to. After
  reinstall/restart on PID `22311`, the first MCP `get_context` session-prime
  loaded cached project packets in `0.1164ms`, the main MCP context call used
  bounded project-file fallback in `49.6486ms`, follow-up REST context hit
  `project_home` in `0.1377ms`, AXI recall was `cache_satisfied` in
  `1.3637ms`, MCP recall was `cache_satisfied` in `2.1577ms` with
  `packet_cache=0.7901ms`, and MCP `observe` reported
  `recall_middleware=1.89ms`. The live cost report showed
  `operation_count=16`, `p95_added_latency_ms=199.2932`,
  `budget_miss_rate=0`, `degraded_rate=0`, and `cache_hit_rate=0.9167`.
  Focused tests passed with `183 passed` plus the pre-existing AsyncMock
  warnings; ruff and `git diff --check` passed; startup validation passed all
  checks against PID `22311`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-234715` with
  `13 pass, 0 warn, 0 fail, 0 skip`. After the matrix restart, PID `23587`
  stayed healthy and first post-matrix MCP session-prime was `0.1323ms`.
- 2026-05-27 follow-up: the next live Codex continuation showed
  `mcp_session_prime` still fixed but exposed intermittent raw capture
  contention. Pre-fix evidence: MCP `observe` spent `capture_store=3023ms`;
  a six-sample REST observe burst had two `captureStore` outliers at `2820ms`
  and `3275ms`. The fix makes `liveCost=true` brain-loop reports skip the
  expensive graph-state scan and use runtime memory-operation metrics directly,
  and it re-anchors cue-vector quiet time after durable capture/cue writes so
  slow raw capture time is not misread as idle time. After reinstall/restart,
  MCP session-prime was `0.1296ms`; MCP `observe` stored
  `ep_31b0f81edcdd` with `capture_store=47ms`, `cue_store=40ms`, and
  `recall_middleware=6.3583ms`; an immediate eight-sample REST observe burst
  returned in `53-90ms` wall with `captureStore=19-31ms`; and live cost
  reported `operation_count=15`, `p95_added_latency_ms=187.0021`,
  `api_observe.p95=79.8831`, zero budget misses, zero degraded cost samples,
  and `graph_state` explicitly skipped as `live_cost_runtime_only`. AXI recall
  found the new cue packet in `durationMs=385.9009`, while MCP recall was
  `cache_satisfied` in `duration_ms=0.3854`. Focused tests passed with
  `26 passed`, adjacent API/report/capture/storage tests passed with
  `91 passed`, ruff and `git diff --check` passed, startup validation passed
  all checks, and the lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-000715` with
  `12 pass, 0 warn, 0 fail, 0 skip` using `--skip-stale-pid`.
- 2026-05-27 AXI startup polish: AXI context, recall, home, doctor, and trace
  paths now infer the current project from the working directory when it has
  normal project markers. This keeps no-`--project` agent startup from falling
  back to generic context. After reinstall/restart on PID `32331`, sequential
  no-`--project` AXI context then recall from `/Users/konnermoshier/Engram`
  returned project-file packets and then `cache_satisfied` recall with
  `durationMs=24.0405`; `engram axi --json` reported the inferred project path
  and project-scoped next commands. Running the same context command from
  `/Users/konnermoshier` left project context empty, which verifies the
  inference is repo-scoped rather than unconditional. Focused tests passed with
  `82 passed`, and ruff passed on the touched AXI/cache files.
- 2026-05-27 fallback telemetry polish: explicit recall now labels empty
  recall rescue paths as `context_packet_fallback` or
  `project_file_recall_fallback` instead of leaking `not_run`, `miss`, or
  `timeout` from earlier fallback stages. After reinstall/restart, AXI
  no-evidence recall returned `status=ok`,
  `fallbackStatus=context_packet_fallback`, three project packets, and
  `durationMs=266.6227`; REST no-evidence recall reported
  `fallbackStatus=project_file_recall_fallback` in about `408ms` wall; and
  warmed context followed by AXI recall returned `cache_satisfied` in
  `durationMs=0.8506`. Focused tests passed with `82 passed`, ruff and
  `git diff --check` passed, skip-slow startup validation passed, and the full
  confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-010214` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 AXI trace usefulness telemetry: AXI traces now remain content-free
  while recording `cacheHit`, `packetCount`, `resultCount`, `fallbackStatus`,
  `skipReason`, `budgetMiss`, and `degraded` fields. Dogfood replay summaries
  aggregate packet/result totals and fallback-status counts, and hook status
  surfaces the same fields for `last_followup`. After reinstalling the local
  tool with `helix-native` and restarting on PID `40221`, a real Codex follow-up
  recall wrote `cacheHit=true`, `fallbackStatus=cache_satisfied`,
  `packetCount=3`, `resultCount=0`, `budgetMiss=false`, and `durationMs=5`.
  `engram axi doctor --hooks codex claude-code --require-hook-run
  --require-followup --json` reported those fields under Codex `last_followup`;
  warmed `engram axi value --json` reported `operation_count=2`,
  `cache_hit_rate=1.0`, `p95_added_latency_ms=0.2162`, and zero budget misses.
  A filtered dogfood replay over the real AXI trace since
  `2026-05-27T08:00:00Z` kept three Engram-project follow-up records,
  summarized `packet_count=6`, `result_count=5`, and counted fallback statuses
  `cache_satisfied=1` and `hit=1`.
  Focused AXI/dogfood replay tests passed with `70 passed`, ruff passed, and
  skip-slow startup validation passed against the native PyO3 LaunchAgent.
- 2026-05-27 native storage diagnostics hardening: live storage counts were
  still starting the expensive Helix native `get_stats()` path, which scans
  entities, episodes, and cues. Pre-fix `/api/storage?live=true&timeoutSeconds=5`
  spent `5002-5015ms` in `storage_counts`, and a concurrent capture burst still
  hit `captureStore=943ms`. Live native storage now uses cached/write-through
  counts and refreshes only disk paths, reporting
  `countsStatus=cached_native_live_skipped` with
  `countsRefreshSkippedReason=helix_native_counts_use_cached_write_through`.
  After reinstall/restart on PID `42603`, live storage returned in `2.1-2.3ms`,
  the concurrent capture burst stayed at `captureStore=15-118ms`, `engramctl
  storage` completed in `0.052s`, MCP `observe` reported `capture_store=40ms`
  and `live_turn=56.3673ms`, and live value reported `operation_count=3`,
  `p95_added_latency_ms=144.21`, and zero budget misses. Focused
  storage/capture/report tests passed with `47 passed`, storage/installer tests
  passed with `33 passed`, ruff passed, and skip-slow startup validation passed.
- 2026-05-27 explicit recall cache/usefulness follow-up: project-file fallback
  packets no longer satisfy explicit recall preflight by themselves. A cached
  packet must carry loaded-store source ids or `episode:` / `entity:` /
  `relationship:` provenance before it can skip live recall. Successful deep
  recall with low visible query overlap now runs the existing fast episode/cue
  fallback and prefers its exact loaded-store hit. After reinstall/restart on
  PID `46546`, `Post-fix validation note storage diagnostics skip count scans
  capture writes responsive` returned five loaded-store items headed by
  `ep_890639326f32` in `durationMs=332.6329`, and `User checked whether the
  Engram performance-goal run was stuck after a continuation` returned five
  loaded-store items headed by `ep_4be71b058394` in `durationMs=220.5227`.
  Both probes reported `skipReason=null` and `fallbackStatus=hit`, proving the
  project-file cache no longer masks fresh cueable episodes. Focused
  recall-surface tests passed with `31 passed`, and ruff passed on the touched
  recall files.
- 2026-05-27 fast-preflight timeout split: the explicit recall preflight now
  has a separate `recall_fast_preflight_timeout_ms=250` budget while timeout
  rescue keeps `recall_fast_fallback_timeout_ms=100`. This preserves the
  tighter post-timeout rescue cap but gives MCP's first loaded-store cue pass
  enough room for long compound queries. After reinstall/restart on PID
  `50407` and packet-cache clears, `Engram native PyO3 dogfood runtime
  performance fast preflight loaded-store recall cache AXI startup Codex`
  returned five loaded-store results with `fallbackStatus=fast_preflight_hit`
  through REST in `durationMs=9.2032`, AXI in `durationMs=8.8308`, and live MCP
  in `duration_ms=14.5175`. Focused recall/cache/AXI tests passed with
  `119 passed`; ruff, `git diff --check`, and skip-slow startup validation
  passed.
- 2026-05-27 loaded-store context preflight: topic-specific context cache
  misses now run the bounded cue/episode preflight before project-file fallback
  and cache successful cue packets for the exact topic. This keeps `get_context`
  fast while preferring loaded-store memory over local docs when the cue layer
  has relevant state. After reinstall/restart on PID `52393` and packet-cache
  clears, `did you get stuck Engram dogfood performance status broad human
  update` returned three `loaded_store_context` cue packets through REST context
  in `durationMs=23.6572`, live MCP context in `duration_ms=26.7368`, and cold
  AXI context with `packet_cache.scopes.loaded_store_context=3`; AXI recall on
  the same topic returned five results with `fallbackStatus=fast_preflight_hit`
  in `durationMs=19.0012`. Live value reported `p95_added_latency_ms=68.557`,
  `budget_miss_rate=0`, and `useful_packet_rate=0.8889`. Focused
  context/recall/cache/AXI tests passed with `142 passed`; ruff,
  `git diff --check`, and skip-slow startup validation passed.
- 2026-05-27 fast BM25 fallback query compaction: a live broad follow-up query
  exposed `trace` as a high-fanout operator term. `AXI` and `startup matrix`
  were fast alone, but `trace`, `AXI trace`, and the full dogfood follow-up
  query could spend the preflight cap before falling back to project-file
  packets. Helix fast BM25 fallback now drops high-fanout terms such as `trace`,
  prefers specific terms such as `loaded store preflight bottleneck packet cache
  startup matrix`, and keeps broad terms only when nothing more specific is
  present. After reinstall/restart on PID `53913`, the full `Engram native PyO3
  dogfood performance loaded-store context preflight next bottleneck
  packet-cache hot behavior startup matrix AXI trace` query returned five
  loaded-store cue results through REST in `durationMs=30.2404` and live MCP in
  `duration_ms=7.2961`, both with `fallbackStatus=fast_preflight_hit`; `AXI
  trace` returned four cue results in `durationMs=21.5398`. Focused tests
  passed with `145 passed`; ruff, `git diff --check`, skip-slow startup
  validation, and the confirmed lifecycle matrix passed. Matrix evidence:
  `/private/tmp/engram-dogfood-startup-20260527-025753` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 context fallback latency and relevance follow-up: after the trace
  fix, a fresh context probe exposed a different startup-path risk. A cold
  topic miss could spend roughly `254ms` in loaded-store context preflight
  before project-file fallback, and synthetic miss probes showed unrelated
  loaded-store packets could be accepted because generated `why_now` text
  repeated the query. Context now has a separate
  `context_fast_preflight_timeout_ms=100` budget, project-file topic fallback
  is bounded to 12 ranked files and 8 KB per file, and context relevance no
  longer scores generated `why_now` fields. After reinstalling the local
  `server/` package into the uv tool and restarting on LaunchAgent PID `59395`,
  the synthetic miss `xqzvplm brontide nonesuch cymophane vellichor 20260527`
  returned five `project_file_fallback` packets in `durationMs=8.8183`
  (`cacheRelevanceMiss=1.9336`, `projectFileFallback=6.8847`), while a relevant
  loaded-store context query still returned three packets in `durationMs=27.7934`.
  A clean AXI value window after context, recall, and observe probes reported
  `operation_count=5`, `p95_added_latency_ms=64.2905`, `budget_miss_rate=0`,
  and `cache_hit_rate=0.5`. Focused context/recall/Helix tests passed with
  `81 passed, 21 skipped`; ruff passed on touched backend/test files; and
  skip-slow startup validation passed against the native PyO3 LaunchAgent. The
  confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-032321` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 MCP observe/context hot-path follow-up: `build_mcp_observe_write_surface`
  now runs live-turn ingestion and recall middleware concurrently inside the
  bounded MCP write-side window, native PyO3 startup warms both capture and cue
  storage routes, and every MCP observe writes a non-persistent
  `session_recent` packet before side effects run. Context cache lookup checks
  `session_recent` immediately after `identity_core`, which gives Codex an
  immediate useful packet for the just-recorded turn while the graph projection
  pipeline catches up. After reinstall/restart on PID `65651`, startup logged
  warmup timings of `capture_store_warmup=11ms`, `cue_store_warmup=29ms`, and
  `capture_store_warmup_cleanup=98ms`. The first live MCP observe still showed
  a first-write spike (`0.5068s` wall, `capture_store=303ms`, `cue_store=77ms`,
  `cue_index_outbox_enqueue=42ms`), but steady MCP observe samples returned in
  `0.1726s` and `0.1406s`. The immediate follow-up MCP context call hit
  `session_recent` with `duration_ms=0.0359`, `cache_fallback=0.0359`, and
  `packet_cache.scopes.session_recent=1`, instead of scanning project files or
  waiting for loaded-store projection. Live value reported `cache_hit_rate=1.0`
  and `budget_miss_rate=0`, with p95 still dominated by the first observe
  spike. Focused backend tests passed with `115 passed, 21 skipped`; ruff and
  `git diff --check` passed; skip-slow startup validation passed with
  `12 pass, 2 skip`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-035942` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 REST hook parity follow-up: REST `api_auto_observe` now seeds the
  same non-persistent `session_recent` packet cache as MCP observe, with packet
  trust source `api_auto_observe`. Topic-specific context cache responses now
  select relevant packets before rendering; when a fresh `session_recent`
  packet matches the query, unrelated project-home fallback packets are left out
  of the response. After reinstall/restart on PID `69900`, a live REST
  auto-observe hook probe returned in `0.05s` wall with `captureStore=10ms`,
  `cueStore=34ms`, and `cueIndexOutboxEnqueue=1ms`. The immediate follow-up
  `engram axi context --topic "20260527-moonstone filtered recent packet narrow
  Codex AXI context query"` returned only the fresh recent observation packet
  from `packet_cache.scopes.session_recent=1`, with source `api_auto_observe`,
  in about `0.30s` CLI wall. A fresh AXI value window reported
  `p95_added_latency_ms=45.1793`, `cache_hit_rate=1.0`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `138 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`.
- 2026-05-27 broad context enrichment follow-up: context packet-cache selection
  now treats a session-recent-only hit as an immediate fallback, not as a full
  topic-specific answer when bounded loaded-store preflight is available. This
  preserves narrow hook/session behavior while allowing broader context asks to
  include older matching cue packets. After reinstall/restart on PID `71606`,
  the live broad query `"loaded-store recall performance packet cache broad
  context topaz older matching cue packets"` returned three
  `loaded_store_context` cue packets through AXI context in about `0.32s` wall.
  Follow-up AXI recall was `cache_satisfied` with `durationMs=1.1824`. MCP
  context returned the fresh `session_recent` packet plus three cached
  project/cue packets in `duration_ms=0.0517`, and MCP recall was
  `cache_satisfied` with `duration_ms=1.319`. Fresh value reported
  `p95_added_latency_ms=113.3016`, `cache_hit_rate=0.6`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `140 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`; and the confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260527-042226` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 project-file masking follow-up: broad topic context no longer lets
  durable cache hits that only have local fallback provenance block a bounded
  loaded-store preflight. Cached packets from `project_file`, `mcp_observe`, and
  `api_auto_observe` remain useful for narrow/recent turns, but broad project
  memory asks now require loaded-store provenance such as `cue:`, `episode:`,
  `entity:`, or `relationship:` before the cache can satisfy the response by
  itself. After reinstall/restart on PID `73970`, AXI context for
  `"Engram native PyO3 dogfood performance loaded-store recall context packet
  cache Codex evidence next bottleneck"` returned three `loaded_store_context`
  cue packets in about `0.34s` wall, and follow-up AXI recall returned
  `cache_satisfied` with `durationMs=0.1675`. MCP context for the same query
  returned loaded cue packets plus cached project-file packets in
  `duration_ms=0.0521`, and MCP recall stayed `cache_satisfied` with
  `duration_ms=1.319`. Fresh AXI value reported
  `p95_added_latency_ms=20.9004`, `cache_hit_rate=0.6667`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `140 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`.
- 2026-05-27 AXI context diagnostics polish: `engram axi context --json` now
  carries the REST/MCP context `budget`, `lifecycle`, and `diagnostics`
  metadata through the compact AXI presenter. This makes the agent-facing CLI
  useful for tracing loaded-store context cost and degraded status without
  needing a separate raw REST probe. After reinstall/restart on PID `75584`,
  cold AXI context for `"what should we work on next to make Engram faster for
  Codex without losing useful memory"` returned three `loaded_store_context` cue
  packets with `budget.durationMs=12.0913`,
  `diagnostics.stageTimingsMs.loadedStoreContextPreflight=10.9676`, and no
  budget miss; follow-up AXI recall was `cache_satisfied` with
  `durationMs=0.6887`. Fresh AXI value reported
  `p95_added_latency_ms=12.0913`, `cache_hit_rate=0.6667`, and
  `budget_miss_rate=0`. AXI presenter tests passed with `40 passed`; ruff and
  `git diff --check` passed; skip-slow startup validation passed with
  `12 pass, 2 skip`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-044127` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 MCP project-path recall adoption follow-up: live Codex recall
  exposed an adoption weakness rather than a storage failure. Without a
  project path, one broad MCP recall spent about `732ms`, found no loaded-store
  results, and rescued with project-file fallback packets. AXI recall with the
  same query and explicit `--project /Users/konnermoshier/Engram` hit the
  loaded cue store directly in `durationMs=13.4652`. The MCP prompt now tells
  agents to use `recall(query, project_path=...)` when a project path is
  available and to carry the same project path across context, routing,
  artifacts, and recall. After reinstall/restart on PID `78269`, the installed
  uv-tool prompt contained the new guidance. With packet cache clear, AXI recall
  for `"Engram native PyO3 dogfood performance current state next bottleneck
  AXI context diagnostics real Codex sessions packet cache budget misses"`
  returned five loaded-store cue results in `durationMs=22.2766`; AXI context
  returned three `loaded_store_context` cue packets in `durationMs=19.7424`
  with `loadedStoreContextPreflight=16.9041`; and warmed MCP recall in this
  already-running Codex session returned cue packets with `duration_ms=0.1982`
  and `cache_satisfied`. Focused MCP prompt/tool/recall tests passed with
  `68 passed, 2 skipped`; ruff and `git diff --check` passed; skip-slow startup
  validation passed with `12 pass, 2 skip`. A fresh Codex restart is still
  needed to verify that the live tool schema exposes the new optional
  `project_path` argument to the harness.
- 2026-05-27 session-recent recall preflight follow-up: explicit recall now
  checks `session_recent` context packets alongside `identity_core` and
  `project_home` before running deeper search. This closes a live-agent gap
  where MCP `observe` made the just-captured turn immediately available to
  `get_context`, but a follow-up `recall` without `project_path` could still
  miss that fresh packet and fall back to project files. After reinstall/restart
  on PID `79521` and packet-cache clear, live MCP `observe` for
  `20260527-amber` stored `ep_d1004c18fb42` with `capture_store=268ms`,
  `cue_store=69ms`, and `recall_middleware=52.5167ms`; follow-up no-project MCP
  recall for the same phrase returned one `recent_observation` packet from
  `_cache_scope=session_recent`, `fallback_status=cache_satisfied`, and
  `duration_ms=1.3854`. Focused recall/MCP tests passed with
  `69 passed, 2 skipped`; ruff and `git diff --check` passed; skip-slow startup
  validation passed with `12 pass, 2 skip`. Fresh AXI value reported
  `budget_miss_rate=0`, `cache_hit_rate=0.6667`, and p95
  `459.2958ms`, with the window dominated by the live observe sample rather than
  recall/context latency.
- 2026-05-27 rolling `session_recent` follow-up: the previous patch made recall
  eligible to use session-recent packets, but the capture side still overwrote
  the single untargeted `session_recent` cache entry on every observe. Observe
  now keeps a rolling five-packet non-persistent session cache, newest first,
  de-duplicated by episode/provenance, so later turns do not erase immediately
  useful earlier session state. After reinstall/restart on PID `81126` and
  packet-cache clear, live MCP observes stored `20260527-orchid`
  (`capture_store=11ms`, `cue_store=31ms`) and `20260527-lapis`
  (`capture_store=89ms`, `cue_store=51ms`). No-project MCP recall for the older
  `orchid` phrase returned two `session_recent` packets with orchid ranked
  first, `cache_satisfied`, and `duration_ms=0.9169`; recall for `lapis`
  returned lapis first in `duration_ms=1.4209`. A traced AXI follow-up for the
  orchid query wrote current Codex hook evidence with `cacheHit=true`,
  `fallbackStatus=cache_satisfied`, `packetCount=2`, and `duration_ms=8`, and
  `engram axi doctor --hooks codex --require-hook-run --require-followup`
  passed. Focused capture/recall/context/packet-cache tests passed with
  `97 passed`; ruff and `git diff --check` passed; skip-slow startup validation
  passed with `12 pass, 2 skip`. Fresh AXI value reported
  `budget_miss_rate=0`, `cache_hit_rate=0.8571`, and p95 `207.9191ms`.
- AXI hook status now exposes a compact follow-up trend so Codex dogfood
  latency evidence does not depend on manual `tail` of
  `~/.engram/axi-hook-runs.jsonl`. `engram axi hooks status codex --json` and
  `engram axi doctor --hooks codex --require-hook-run --require-followup --json`
  include `followup_summary` with recent context/recall counts, duration
  avg/p95/max, cache hits, fallback status counts, degraded/timeout counts, and
  the five newest redaction-safe records. Live output after reinstall showed
  latest recall at `8ms`, `cacheHit=true`, `cache_satisfied`, and
  `packetCount=2`, while keeping the earlier `509ms` context-packet fallback
  and `512ms` project-file fallback in the same report. Focused AXI tests passed
  with `35 passed`; ruff passed on the touched AXI files.
- `followup_summary.latest_healthy_streak` now separates current non-degraded
  hook behavior from older failures. After reinstalling the CLI and appending a
  fresh traced context/recall pair, the newest Codex follow-ups were context
  `31ms`, recall `31ms`, and the earlier fixed recall `8ms`; the streak
  included 16 ok records with no degraded rows or timeouts. The same live run
  showed direct AXI context returning from `session_recent` cache in
  `durationMs=0.056` and explicit AXI recall returning `cache_satisfied` in
  `durationMs=0.7595`. Focused hook/CLI tests passed with `36 passed`.
- Live MCP evidence from the same Codex session now follows the intended
  bounded path. The first `get_context` on the active dogfood topic missed cache
  but returned useful project-file packets in `duration_ms=500.8099` with no
  degradation. MCP `recall` then used fast preflight and finished with
  `query_time_ms=55.8`, five cue results, three packets, and
  `fallback_status=fast_preflight_hit`. Repeated `get_context` first produced
  loaded-store cue packets in `31.8175ms`, then hit packet cache directly in
  `0.0698ms` with six packets.
- The startup validator now checks MCP schema drift directly. `MCP live tool
  catalog` fails unless `recall.project_path` is present and a read-only
  `recall(project_path=...)` probe returns without budget miss, degradation, or
  timeout. A full live validator run passed `14/14` with
  `recall_has_project_path=true` and a `cache_satisfied` probe in
  `query_time_ms=48.6`. The confirmed startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-052815` passed
  `13 pass, 0 warn, 0 fail, 0 skip`; both warmed validations inside it passed
  `14/14`, and their project-path recall probes returned project-file fallback
  packets in `654.5ms` before restart and `591.6ms` after restart, with no
  budget miss or degradation. The runtime was restored healthy afterward on
  LaunchAgent PID `86144`.
- Real-session dogfood collection now has a recent-first path. `engram dogfood
  scan --sort recent` surfaces active resumed Codex transcripts by modification
  time; after reinstall it put the active Engram goal transcript first with 80
  labelable turns. A redacted review bundle at
  `/private/tmp/engram-dogfood-review-20260527-active-codex` pairs that
  transcript with AXI trace rows since `2026-05-27T12:00:00Z`. The measured
  trace slice has five Codex follow-up rows (`context=2`, `recall=3`), all
  `ok`, no degraded rows or timeouts, average `119.8ms`, p95/max `509ms`,
  four cache hits, six packets, and fallback counts `cache_satisfied=2`,
  `context_packet_fallback=1`. The bundle was then reviewed with 2 recall
  labels, 1 session label, and 78 skipped turns; the session label records
  baseline `1.0`, memory `1.0`, open-loop expected/recovered, and no measurable
  Engram lift for that reviewed local sample.
- AXI `value` now exposes a compact mode breakdown plus separate `read_path` and
  `write_path` cost summaries. The first live run after reinstall preserved the
  aggregate `p95_added_latency_ms=8200.8437`, but showed the spike was
  `api_auto_observe`; the read path covered nine recall/context/packet
  operations with max mode p95 `279.4478ms`, `cache_hit_rate=0.5`, and no budget
  misses. This keeps the agent-facing value packet honest about capture latency
  without making startup/follow-up recall look slower than it is.
- Fast runtime packets now expose startup-safe packet-cache summary data, which
  keeps AXI home aligned with the real cache state after context/recall warming.
  After reinstall/restart, AXI context warmed three relevant loaded-store cue
  packets in `24.1565ms`, AXI home reported two fresh cache entries and
  `packet_cache.status=warm`, AXI recall was `cache_satisfied` in `0.5949ms`,
  MCP get_context hit packet cache in `0.0649ms`, and MCP recall returned three
  packets in `query_time_ms=2.0` with no degradation or budget miss. A clean
  post-restart AXI value window showed read-path p95 `24.1565ms`,
  `cache_hit_rate=0.875`, and zero degraded/timeouts/budget misses.
- `engram doctor` now bounds lifecycle snapshot and smoke phases independently,
  and the startup matrix now preserves doctor warnings from preambled JSON
  output. The latest confirmed matrix at
  `/private/tmp/engram-dogfood-startup-20260527-061148` completed with
  `11 pass, 2 warn, 0 fail, 0 skip`; the warnings are the bounded loaded-store
  lifecycle snapshot timeouts, while REST/MCP doctor checks, disposable Helix
  smoke, both warmed validations, stopped-state detection, restart, and stale-PID
  simulation all completed.
- Installed `engramctl doctor` now treats the loaded-store lifecycle snapshot as
  an explicit deep diagnostic for native Helix rather than part of the default
  startup readiness gate. Live `engramctl doctor --format json` returned
  `status=pass` with `lifecycle_snapshot=skipped`; REST health, MCP reachability,
  and disposable Helix smoke still passed. The confirmed matrix at
  `/private/tmp/engram-dogfood-startup-20260527-062433` then completed with
  `13 pass, 0 warn, 0 fail, 0 skip`, and `engramctl status` restored the 4G
  native PyO3 LaunchAgent runtime healthy on PID `97708`.
- Active Codex review evidence now closes the loop against the installed editable
  native PyO3 tool. The review bundle at
  `/private/tmp/engram-dogfood-review-20260527-active-codex` moved from
  `needs_labels` to finalized with 2 reviewed recall labels, 1 reviewed session
  label, 78 skipped turns, and exported human-label evidence SHA-256
  `68fa77851b5a8b6bf20e3946955ccbef87fbb9f90df3781b7214912ef0d09ade`.
  The replay window covered five Codex follow-up trace rows (`context=2`,
  `recall=3`) with average `119.8ms`, p95/max `509ms`, cache hit rate `0.8`,
  and zero degraded operations, timeouts, or budget misses. `engram dogfood
  finalize` now replays idempotently with `status=finalized`, native evaluation
  exit `0`, and `memory_value.status=measured`; the measured benefit report has
  7 recall samples, 2 session samples, useful packet rate `0.9`,
  memory-need precision/recall `1.0`, false recall rate `0.1`, and open-loop
  recovery rate `1.0`. This is release-useful human-label evidence, but still a
  small local Codex sample rather than broad multi-session proof.
- The next live pass tightened usefulness under real Codex traffic rather than
  only latency. A specific query for the just-fixed dogfood finalize path first
  exposed two issues: context could treat a weak `session_recent` packet as
  enough, and recall could return cross-project `MachineShopScheduler` episodes
  when the current Codex tool schema could not pass `project_path`. The runtime
  now refuses weak one-token session-recent matches before project-file rescue,
  treats `mcp_observe`/`api_auto_observe` packets as session-local rather than
  loaded-store enrichment, applies project-path preference to fast preflight and
  context preflight results, and stores the latest MCP `get_context` project path
  on the session so schema-limited `recall()` calls inherit it. Live proof after
  restart: AXI recall for `dogfood finalize idempotent INSERT OR REPLACE
  graph_stats_timeout human label artifact` returned two Engram hits in
  `19.4049ms` with no degradation and no cross-project hits. In the same live
  Codex MCP session, cold `get_context(project_path=...)` stayed under budget
  with project-file packets, repeat context used loaded-store preflight in
  `10.4572ms`, and `recall()` without a callable `project_path` argument
  inherited the session path and returned two Engram hits in
  `query_time_ms=34.0`. The post-pass AXI value packet showed read-path
  p95 `238.3133ms`, cache hit rate `1.0`, and zero read-path budget misses,
  degraded operations, or timeouts. The confirmed startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-074345` passed
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Project-scoped dogfood scanning now uses Codex tool `workdir` evidence, not
  only the transcript `session_cwd`, so active Engram work launched from
  `/Users/konnermoshier` is not filtered out of real evaluation prep. The live
  scan found 20 Engram candidates, selected the active transcript via
  `project_match_source=tool_workdir`, and prepared
  `/private/tmp/engram-dogfood-review-20260527-project-workdir` with 80
  labelable user turns and 10 project-filtered AXI trace records. The bundle
  remains `needs_labels`, which keeps the memory-value evidence gate honest
  until those turns are reviewed.
- A small real review pass is now imported for that bundle: 6 reviewed recall
  turns, 1 reviewed session-continuity sample, and selected modes split between
  `cached=3` and `off=3` so the evidence includes both memory-helpful turns and
  self-contained/current-truth turns where recall should stay out of the way.
  The exported artifact is
  `/private/tmp/engram-dogfood-review-20260527-project-workdir/human-labels.json`.
  `engram evaluate --format json --require-memory-value` now passes with
  13 recall samples and 3 session samples: useful packet rate `0.9231`, false
  recall rate `0.0769`, open-loop recovery `1.0`, session-continuity lift
  `0.0333`, p95 added latency `1082.8418ms`, cache hit rate `0.6364`, and
  budget miss rate `0.0`. `--require-evaluation-signals` is still incomplete
  because cue usefulness and projection yield need data.
- Fresh post-import runtime probes show the current happy path is bounded:
  AXI context returned loaded-store packets in `78.9281ms`, AXI recall returned
  three packets via `fast_preflight_hit` in `58.7302ms`, MCP context hit packet
  cache in `0.0757ms`, and MCP recall hit cache in `0.104ms`. The next latency
  target is the cold `mcp_context` path, which still dominates aggregate p95 at
  `1082.8418ms`; historical medium-mode recall timeouts remain visible in the
  aggregate store and should be driven down by better cache warming and fewer
  unnecessary loaded-store misses.
- The loaded-store evaluation report path is now bounded and cache-backed.
  A raw native `graph_store.get_stats("default")` against the 4.0G dogfood store
  measured `51728.62ms`, confirming that Python-side Helix stats aggregation is
  too expensive for synchronous request paths. The REST report now shields graph
  stats reads, keeps successful snapshots in `GraphStateService`, and returns
  cached stats with fresh runtime cost counters. After cold warmup, the live
  report returned in about `0.52s` with `1381` episodes, `1288` cues, `901`
  entities, `8109` relationships, `849` projected episodes, projection yield
  `1.6243`, and no degradation entries. This is a mitigation, not the final
  architecture: Helix needs cheap aggregate/materialized stats so cold reports
  do not require a 50s background scan.
- Startup warmup is now bounded separately from storage diagnostics. A slow
  native capture warmup previously delayed server readiness while cue storage
  took `91525ms`; `capture_startup_warmup_timeout_ms` now defaults to `2000` so
  the API can start while the internal create/delete probe finishes best-effort.
  Live restart verified both paths: one restart continued after the 2s bound and
  stayed healthy, and a later restart completed warmup normally with
  `capture_store_warmup=27ms`, `cue_store_warmup=43ms`, and cleanup `115ms`.
- AXI/MCP current-state probes after the evaluation-cache patch: AXI context for
  `native PyO3 dogfood project workdir labels evidence` returned project-file
  fallback context in `1548.5852ms` with no degradation; first AXI recall was a
  cold miss and timed out at `2489.841ms`, but repeat AXI recall and REST recall
  returned under budget at about `494ms`/`484ms` with three context packets via
  `context_packet_fallback`. Live MCP recall reported `duration_ms=483.3426`
  with packet fallback and no degradation. The value gate still passes with
  13 recall samples, 3 session samples, useful packet rate `0.9231`, false
  recall rate `0.0769`, p95 added latency `699.349ms`, cache hit rate `0.6522`,
  and timeout rate `0.0682`.
- A follow-up packet-cache-clear probe found one remaining empty-payload edge:
  after restart, explicit-project AXI recall for the active Engram query stayed
  bounded at `589.3109ms` but returned zero packets because the cold project-file
  fallback outlived the fixed 100ms wait. The empty-success fallback now waits
  against the remaining recall wall budget, capped at 1.25s. After restart on
  PID `51738`, the same Engram query used loaded-store preflight directly
  (`result_count=5`, `packet_count=3`, `durationMs=94.2722`), and a forced
  empty-cache temp-project probe returned a packet-bearing project-file fallback
  (`packet_count=1`, `fallbackStatus=project_file_recall_fallback`,
  `durationMs=963.7207`) instead of an empty payload. After clearing that temp
  probe cache, real Engram project context rebuilt five project-file packets in
  `716.9267ms`, and follow-up AXI recall returned three Engram packets in
  `514.82ms` via `project_file_recall_fallback`. The warm evaluation report
  still recovered after background stats warmup with `1401` episodes, `901`
  entities, `8109` relationships, projection yield `1.6091`, and cue usefulness
  `needs_feedback`.
- Server-backed evaluation now avoids the direct native CLI cold-stats path.
  `engram evaluate --server-url http://127.0.0.1:8100 --require-memory-value`
  reads the running REST report and passed against the dogfood service with
  `1405` episodes, `901` entities, `8109` relationships, projection yield
  `1.6054`, and measured memory value. The strict
  `--require-evaluation-signals --require-memory-value` gate now fails only on
  the real missing feedback signal (`cue_usefulness:needs_feedback`). This means
  the remaining release gate work is cue-feedback evidence, not report latency or
  projection-yield aggregation. `engramctl storage` also now prints count source
  metadata so native write-through counts are not mistaken for historical graph
  totals.
- Native lifecycle stop now has bounded orphan-listener cleanup for the dogfood
  path. When a half-started LaunchAgent restart leaves an older `engram serve`
  process owning port `8100`, `engramctl stop` finds the listener with `lsof`,
  terminates only Engram-looking `serve` command lines, and refuses unrelated
  processes. Manual stop/start proof left the runtime offline after stop and then
  restarted with one LaunchAgent-owned listener. The lifecycle matrix at
  `/private/tmp/engram-dogfood-startup-20260527-110425` passed with
  `13 pass, 0 warn, 0 fail, 0 skip`; live MCP context returned 5 packets with no
  degradation and recall returned 3 fallback packets in about `497ms`.
- The follow-up stats pass removes the post-restart server-backed evaluation
  timeout that still showed `graph_state_timeout` and zero graph totals after a
  fresh native dogfood start. Helix stats refresh now prefers four bulk
  generated routes for cues and projected episode/entity links, while retaining
  the older per-episode fallback when a native build lacks those handlers. The
  generated PyO3 route map was rebuilt and reinstalled; live startup logs now
  show `helix_native` with `routes=180`. Server-backed evaluation on
  `http://127.0.0.1:8100` now returns measured graph totals immediately enough
  for the normal 10s CLI path: `1414` episodes, `901` entities, `8109`
  relationships, Capture `ready`, Cue `attention`, and Project `active`. Current
  AXI/MCP probe evidence after the restart: AXI context returned three
  loaded-store packets in `37.7618ms`; AXI recall returned five results and
  three packets in `48.2276ms`; MCP `get_context` returned three loaded-store
  packets in `82.0965ms`; MCP `recall` returned cached packets in `0.7431ms`.
  The post-fix startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-112804` passed with
  `13 pass, 0 warn, 0 fail, 0 skip`.
  A final report-service guard now marks runtime-only graph stats as
  `graph_state_unavailable` so a cold fallback cannot masquerade as a real empty
  graph. After the final restart on PID `14146`, server-backed evaluation
  immediately returned `1416` episodes, `901` entities, `8109` relationships,
  and no report degradation. The remaining observed recall issue is bounded and
  useful rather than empty: AXI recall exceeded its search budget at
  `1069.5837ms` but returned three fallback packets; MCP context/recall stayed
  under budget at `22.1129ms` and `1.4698ms`.
- The follow-up AXI repeat-recall pass removes that repeated project-file
  fallback timeout for matching cached packets. AXI recall now prints REST
  diagnostics so a timeout shows whether time was spent in packet cache, fast
  preflight, deep recall search, or project-file fallback. GraphManager's fast
  recall fallback now runs record-backed cue and episode searches concurrently
  enough that a stalled cue-record lookup cannot block a ready episode-record
  hit. Cached packets produced by the bounded project-file fallback
  (`trust.source=project_file`) can satisfy the next matching explicit recall
  from packet cache; generic file-context packets still do not short-circuit
  loaded-store recall. Live evidence after reinstall/restart on PID `31237`:
  the previously degraded query
  `"Windsurf Cursor adoption cross harness authority shadow memory routing 20260527"`
  returned `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs=43.9192`; a second cached project-file query returned
  `durationMs=1.2747`. Fresh live-cost value showed `axi_recall` p95
  `43.9192ms`, `cache_hit_rate=1.0`, and zero timeouts, degraded operations,
  or budget misses. Full live-cost report context still degraded with
  `evaluation_context_timeout` plus `live_cost_runtime_only`, so the next
  performance pass should target report/context aggregation rather than repeat
  AXI recall.
- The report/context aggregation pass now keeps live-cost reports out of
  consolidation-context scans, starts native graph-stats warmup as a REST startup
  background task, and cancels/replaces stale graph-stats and consolidation
  context warmups after 30 seconds. Native Helix stats now have four count routes
  (`count_entities_by_group`, `count_episodes_by_group`,
  `count_relationships_by_group`, `count_cues_by_group`) and the reinstalled
  PyO3 route map reports `routes=184`. Direct native count-route proof returned
  `901` entities, `1427` episodes, `3963` relationships, and `1332` cues.
  Running-server proof now shows both normal and `liveCost=true`
  `/api/evaluation/brain-loop/report` calls returning populated graph totals and
  no report degradation. AXI recall for current dogfood status returned
  `budgetMiss=false`, `skipReason=cache_satisfied`, and `durationMs=54.7065`;
  topic-specific AXI context rebuilt five project-file packets in
  `durationMs=534.6486` without loaded-store graph reads. The residual measured
  latency spike is now explicitly isolated to project-file fallback samples in
  `engram axi value`, not graph-state report loading.
- Repeat topic-specific project-file context now becomes a real cache hit.
  Project-file fallback packets carry exact topic/project markers when cached,
  so a matching follow-up context call can use the packet cache without paying
  loaded-store preflight or another file scan; generic project-home/file packets
  still remain conservative. Live proof after reinstall: the first AXI context
  call for a fresh repeat-cache topic built five packets in `1142.4237ms`, the
  second identical AXI context call hit cache in `0.1268ms`, MCP `get_context`
  hit the same cache in `0.1611ms`, and MCP `recall` returned three cached
  packets in `2.1714ms`.
- The same real Codex continuation surfaced a write-path tail: MCP `observe`
  stored successfully but spent `36503ms` in raw `capture_store`, which made the
  live write-path p95 `36593.9551ms`. Raw episode persistence now has
  `capture_store_timeout_ms` (default `1000`) so live capture can acknowledge
  and let the raw write, event publishing, cue persistence, and projection
  scheduling finish in the background when native Helix stalls. After reinstall,
  REST observe returned with `captureStore=122ms` then `8ms`, and real MCP
  `observe` returned in about `1.28s` wall with `capture_store=415ms`,
  `cue_store=123ms`, and bounded `live_turn`/`recall_middleware` side effects.
- The startup-matrix doctor failure after that write-path change was a stats
  truth issue, not a projection failure. Native Helix projected the disposable
  smoke episodes, but `get_stats()` returned the count-only fast packet with
  empty projection metrics, so evaluation reported
  `projection yield cannot be measured until episodes are projected`. Exact
  stats are now the default for `get_stats()`, recall pool sizing requests the
  fast count route with `exact=False`, and smoke uses synchronous raw/cue capture
  so deterministic checks do not inherit live capture deferral. Installed
  `engram evaluate --smoke --mode helix --format json` returned no coverage gaps
  with `projected_count=3`, `linked_entity_count=3`, and one consolidation cycle;
  `engramctl doctor --format json` passed; and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260527-130309` with
  `13 pass, 0 warn, 0 fail, 0 skip`. After the matrix restart, AXI/MCP repeat
  reads hit cache (`axi context` `0.1597ms`, MCP context `0.0261ms`, MCP recall
  `3.184ms`) while the runtime stayed healthy on LaunchAgent PID `19164`.
- 2026-05-27 repeat-cache latency follow-up: resident packet-cache reads now avoid
  per-call SQLite sidecar syncs on context, explicit recall, and auto-recall hot
  paths; context prebuilds project-file fallback while loaded-store preflight
  runs; and exact project-file fallback packets can satisfy repeated context and
  recall even when generic file summaries do not contain unusual query terms.
  After reinstall/restart, synthetic miss query
  `xafnorb quexilate zumbrel frobnicate mintcase exactcache5` rebuilt AXI context
  in `618.2917ms` (`cacheRelevanceMiss=2.5779ms`,
  `projectFileFallback=564.7988ms`), repeated context from cache in `0.047ms`,
  and returned AXI recall `cache_satisfied` in `0.7253ms` and `0.585ms`. Focused
  context/recall tests passed with `81 passed`, ruff passed, and
  `git diff --check` is clean. Skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`; the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-153357` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on LaunchAgent
  PID `57186`. The remaining warnings are still stale/root Codex and Claude Code
  SessionStart hook evidence, not runtime failures. On that post-matrix runtime,
  the same synthetic topic rebuilt project-file context in `41.1342ms`, repeated
  from cache in `0.6055ms`, and AXI recall was `cache_satisfied` in `0.5183ms`.
- 2026-05-27 weak-relevance follow-up: cached context packets now reject lone
  date/id matches, broad-only matches require multiple supporting hits when the
  query has high-signal terms, and explicit recall no longer lets generated
  `why_now` text satisfy the query. After reinstall/restart, weak synthetic query
  `qvanta noexisting loadedstore miss tail 20260527 probeB` no longer returned
  stale loaded-store dogfood packets; AXI context reported
  `cache_relevance_miss` and produced project-file fallback packets in
  `44.7505ms` (`projectFileFallback=42.4623ms`). AXI recall for the same topic
  was `cache_satisfied` in `0.6213ms` from exact project-file fallback cache. A
  fresh recall-first probe `qvanta noexisting loadedstore miss tail 20260527
  probeC` ran bounded recall in `228.2368ms`, found no memory results, and
  returned three project-file packets with `fallbackStatus=context_packet_fallback`.
  Live value after the probe set reports `0%` budget misses, `0%` degradation,
  `75%` read-path cache hit rate, and p95 `223.127ms` over five read-path
  samples. Focused context/recall tests passed with `83 passed`, ruff passed,
  and `git diff --check` is clean. Skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`; the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-154316` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `59654`. The remaining warnings are still stale/root Codex and
  Claude Code SessionStart hook evidence, not runtime failures.
- 2026-05-27 write-path latency follow-up: MCP write live-turn fingerprinting now
  has a short side-effect wait while continuing in the background, and write-tool
  auto-recall is cache-only. Writes still cache fresh session packets and can
  surface already-warm context, but a cache miss no longer runs a medium recall
  probe on the write response path. Before this pass, live value showed
  `mcp_observe` p95 `178.0555ms`, `api_auto_observe` p95 `314.8745ms`, and stale
  `medium` recall timeouts. After reinstall/restart on PID `63522`, live MCP
  observe probe `obsF` returned in `85.9ms` wall time with `capture_store=11ms`,
  `cue_store=39ms`, `live_turn_timeout=11.6503ms`, and
  `recall_middleware=0.4302ms`; live value showed write-path p95 `65.566ms` and
  cache-miss `medium` auto-recall skipped in `0.0775ms`. AXI context for
  `write auto recall cache-only short live-turn timeout obsF` returned
  loaded-store cue packets in `30.4921ms`, and AXI recall was `cache_satisfied`
  from the fresh `mcp_observe` recent packet in `0.3212ms`. Focused backend tests
  passed with `129 passed, 2 skipped`, ruff passed, and `git diff --check` is
  clean. Skip-slow validation reports `11 pass, 1 warn, 0 fail, 2 skip`; the
  confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-155526` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `64398`. A post-matrix MCP observe probe `obsG` kept
  `recall_middleware=0.3129ms` and `live_turn_timeout=12.5198ms`; live value
  after that probe showed read-path p95 `2.444ms`, `0%` read/write degradation,
  and write-path p95 still dominated by matrix `api_auto_observe` at
  `155.1062ms`.
- 2026-05-27/28 AXI trace refresh guidance follow-up: startup validation now
  separates startup proof from follow-up proof. Manual `agent-followup` traces
  can prove context/recall commands work, but they no longer masquerade as fresh
  SessionStart evidence. When stale/root SessionStart evidence remains, the
  validator tells the operator to start a new interactive Codex or Claude Code
  session from the target project. A nested `codex exec` run from
  `/Users/konnermoshier/Engram` proved real MCP adoption by calling Engram
  `get_context` and returning `ENGRAM_SESSIONSTART_PROBE`, but it did not emit a
  SessionStart hook row. Focused startup-validation tests for stale/root
  guidance passed, ruff passed, and `git diff --check` is clean. The remaining
  `13 pass, 1 warn, 0 skip` startup validation warning is honest stale/root
  SessionStart evidence, not a runtime, MCP, or AXI follow-up failure.
- 2026-05-27 agent raw-capture wait follow-up: MCP observe and REST
  auto-observe now pass a per-write `capture_store_timeout_ms=250`, while the
  global explicit-write default stays at `1000ms` for reliability. After
  reinstall/restart on LaunchAgent PID `67368`, live MCP observe returned with
  `capture_store=169ms`, `cue_store_timeout=251ms`, `live_turn_timeout=13.097ms`,
  and `recall_middleware=1.3739ms`; live REST auto-observe deferred raw capture
  at `captureStoreTimeout=252ms`. `engram axi value --json` reported write-path
  p95 `440.135ms`, read-path p95 `0.1838ms`, `0%` degradation, and `0%` budget
  misses. Focused capture/validation tests passed with `166 passed, 2 skipped`,
  ruff passed, and `git diff --check` is clean.
- 2026-05-27 runtime-fast prefix-warmup follow-up: the startup-safe
  `/api/knowledge/runtime/fast` path now schedules a non-blocking in-memory
  project-file prefix warmup for the project path. This keeps startup probes
  graph-free and non-capturing while reducing the first real
  topic-specific context miss after AXI session-start. After reinstall/restart
  on LaunchAgent PID `69492`, AXI home triggered the warmup; fresh AXI context
  for `project prefix warmup liveproof citrine 20260527` returned five
  project-file packets in `durationMs=183.2692`, with
  `projectFileFallback=137.6598`, then repeated in `0.0474ms`; AXI recall was
  `cache_satisfied` in `0.7942ms`. Fresh MCP `get_context` for
  `project prefix warmup mcp liveproof beryl 20260527` returned useful
  project-file packets in `duration_ms=127.3182`, with
  `project_file_fallback=29.3808`, and MCP recall was `cache_satisfied` in
  `1.1467ms`. `engram axi value --json` reported live read-path p95
  `217.9204ms`, cache hit rate `0.8`, and zero budget misses/degraded
  operations/timeouts over the small post-restart sample. Focused
  runtime/context tests passed with `139 passed`; ruff and `git diff --check`
  passed.
- 2026-05-27 project-file executor isolation follow-up: live MCP context still
  produced a fresh-topic `project_file_fallback=806.2496ms` even though local
  profiling showed the warmed fallback builder itself takes about `25ms`. The
  local rescue path was queueing behind the default executor, so project-file
  context fallback, recall fallback, and runtime-fast prefix warmup now use a
  small dedicated executor. After reinstall/restart on LaunchAgent PID `71087`,
  fresh MCP `get_context` returned five project-file packets in `149.254ms` with
  `project_file_fallback=141.8561`, and MCP recall was `cache_satisfied` in
  `0.5977ms`. Fresh AXI context returned loaded-store cue packets in
  `77.4707ms`, and AXI recall was `cache_satisfied` in `0.5423ms`. Fresh
  `engram axi value --json` reported read-path p95 `149.6347ms`, cache hit rate
  `0.7857`, and zero budget misses/degraded operations/timeouts. Focused
  runtime/context tests passed with `182 passed`; ruff and `git diff --check`
  passed.
- 2026-05-27 AXI project-context follow-up: `engram axi context --project ...`
  now skips loaded-store preflight with or without an explicit topic and uses
  cached/project-file context. AXI stays in the startup-safe project packet lane,
  while explicit memory lookup remains available through `engram axi recall` and
  MCP `get_context`. After reinstall/restart on LaunchAgent PID `74526`,
  packet-cache clear plus a cold topic-specific AXI context showed no loaded
  store preflight (`cacheRelevanceMiss=0.3322ms`) but paid one cold prefix scan
  at `projectFileFallback=285.6067ms`; the next fresh topic returned in
  `57.0956ms` with `projectFileFallback=32.1867ms`, followed by a cache hit in
  `0.0461ms`. Startup validation reports `13 pass, 1 warn`, with the remaining
  warning limited to stale/root real SessionStart proof. Final focused
  runtime/context tests passed with `184 passed`; ruff and `git diff --check`
  passed.
- 2026-05-27 duplicate project-file scan follow-up: summary matching and
  evidence-claim extraction now share one topic-match scan per candidate file.
  This targets the fresh-topic fallback path without changing loaded-store
  recall/context semantics. Local profiling dropped the Engram project-file
  fallback builder from roughly `18-22ms` to `11-16ms`. After reinstall/restart
  on LaunchAgent PID `75796`, AXI home warmed the project; a fresh AXI context
  built project-file packets in `22.6326ms`, exact repeat hit cache in
  `0.0393ms`, MCP `get_context` reported
  `project_file_fallback=24.8746ms`, and MCP recall hit cache in `1.0761ms`.
  Live value reported read-path p95 `175.694ms`, cache hit rate `0.625`, and
  zero budget misses/degraded operations/timeouts. Startup validation stayed at
  `13 pass, 1 warn`; the warning remains stale/root real SessionStart proof.
  Focused runtime/context tests passed with `185 passed`; ruff and
  `git diff --check` passed.
- 2026-05-27 MCP context preflight diagnostics: successful loaded-store context
  now reports search and packet-assembly timings separately, and project-file
  fallback responses include `loaded_store_context_preflight` when loaded-store
  preflight misses. After reinstall/restart on LaunchAgent PID `76806`, a
  repeated useful goal-continuation context hit packet cache in `0.0772ms`; a
  fresh loaded-store miss returned useful project-file packets without
  degradation in `103.96ms`, split as
  `loaded_store_context_preflight=99.7659ms` and
  `project_file_fallback=23.6974ms`. AXI context for a comparable fresh project
  topic returned in `69.46ms` with `projectFileFallback=21.6849ms`. One earlier
  post-restart MCP context sample showed a transient cold project-file build at
  `1228.0362ms`, which remains visible in live value p95. Startup validation
  stayed at `13 pass, 1 warn`; the warning remains stale/root real SessionStart
  proof. Focused runtime/context tests passed with `185 passed`; ruff and
  `git diff --check` passed.
- 2026-05-27 MCP context soft wait: quick loaded-store context hits still win,
  but when project-file context is already ready, MCP no longer waits the full
  `context_fast_preflight_timeout_ms=100` on loaded-store misses. The new
  `context_fast_preflight_soft_wait_ms` default is `75ms`, and late loaded-store
  work continues in the background so it can still populate cache. After
  reinstall/restart on LaunchAgent PID `77735`, AXI home warmed the project. A
  useful goal-continuation MCP context returned loaded-store cue packets in
  `85.4594ms` with `loaded_store_context_search=52.4431ms`; a fresh miss
  returned useful project-file packets in `25.7279ms`, with
  `project_file_fallback=23.1409ms` and
  `loaded_store_context_preflight=17.2334ms`. MCP recall for the same miss hit
  cache in `0.8398ms`. Live value reported read-path p95 `85.4594ms`, cache hit
  rate `0.7143`, and zero budget misses/degraded reads/timeouts. Startup
  validation stayed at `13 pass, 1 warn`; the warning remains stale/root real
  SessionStart proof. Focused runtime/context tests passed with `186 passed`;
  ruff and `git diff --check` passed.
- 2026-05-27 MCP persistent project-file rescue: the remaining variance was the
  first fresh context miss after restart, where in-memory cache could be cold
  and project-file fallback could still take hundreds of milliseconds. A
  pre-fix live MCP sample spent `project_file_fallback=750.092ms`. The rescue
  path now returns current-version same-project project-file packets from
  persistent packet cache when the fresh project-file scan is still pending,
  while leaving the initial strict cache lookup in-memory only. The slow scan
  keeps running and refreshes the exact topic cache in the background. Unit
  coverage verifies both the immediate rescue and background cache refresh.
  After reinstall/restart on LaunchAgent PID `80961`, the first fresh live MCP
  miss `persistent rescue first post restart miss zibble norvax klym 20260527`
  returned loaded-store cue packets in `30.344ms` instead of falling back, split
  as `loaded_store_context_search=21.6139ms` and packet assembly `0.0417ms`.
  AXI recall hit cache in `0.6234ms`, AXI context hit project cache in
  `0.049ms`. After full startup validation, live value reports read-path p95
  `80.397ms`, read cache hit rate `0.8`, and zero budget misses/degraded
  reads/timeouts. Full startup validation reported `13 pass, 1 warn, 0 skip`:
  doctor and live MCP catalog passed, and the only warning remains stale/root
  SessionStart proof. Focused runtime/context tests passed with
  `558 passed, 13 skipped`; ruff and `git diff --check` passed.
- 2026-05-28 interactive Codex SessionStart proof: a real Codex TUI session
  launched from `/Users/konnermoshier/Engram` accepted the managed read-only AXI
  hook and wrote a current startup trace:
  `timestamp=2026-05-28T00:24:34.184814Z`, `operation=hook-run`,
  `project=/Users/konnermoshier/Engram`, `durationMs=11`, `status=healthy`.
  The startup validator now accepts both legacy `home` startup traces and the
  current `hook-run` trace shape. A Claude Code print-mode probe then emitted
  `operation=hook-run`, `project=/Users/konnermoshier/Engram`, `durationMs=12`,
  and `status=healthy` before its prompt-argument error. Full startup validation
  now passes AXI hook/tracing evidence for both clients. The installed AXI home
  packet now uses the active trace client for capture suggestions
  (`--source claude-code`, `--source codex`, or generic `--source axi`) instead
  of hard-coding Codex. Focused AXI/startup-validation tests passed with
  `39 passed`, ruff passed, and live value reports read-path p95 `104.0759ms`,
  read cache hit rate `0.8545`, and zero read budget misses/degraded
  reads/timeouts.
- 2026-05-28 resumed live runtime baseline: the installed local runtime was
  restarted with `engramctl stop && engramctl start` and came back healthy on
  LaunchAgent PID `86463`. AXI home stayed graph-free/startup-safe. A fresh AXI
  context query built five project-file packets in `49.1589ms` with
  `projectFileFallback=35.2116ms`; AXI recall for the same topic returned five
  loaded-store episode results and three packets in `235.963ms` with
  `fallbackStatus=fast_preflight_hit`; MCP `get_context` then hit cache in
  `0.0496ms`; MCP `recall` was cache-satisfied in `0.2021ms`. Full startup
  validation passed all checks, and the lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-173419` with
  `13 pass, 0 warn, 0 fail, 0 skip`, leaving the runtime healthy on PID `87404`.
  Post-matrix probes confirmed the no-empty-timeout behavior: forced-miss AXI
  recall returned a relevant historical diagnostic episode in `19.1024ms`,
  fresh AXI context fallback returned useful packets in `16.7976ms`, and broad
  AXI recall hit cache in `0.4516ms`. Final live value reports read-path p95
  `83.526ms`, read cache hit rate `0.6667`, and zero read budget
  misses/degraded reads/timeouts. This pass found no current recall/context
  timeout to patch; the remaining release hygiene is source cleanliness and
  longer real-session evidence.
- 2026-05-28 real Codex continuation sample: after commit/push at clean
  checkpoint `e59be43`, Engram recovered the active native dogfood performance
  goal and project context for a resumed Codex session. The run recorded
  session-continuity sample `esc_35ecade2bbf7` and recall-quality sample
  `ers_7aa46915657c`; shell probes confirmed the runtime was still healthy on
  native PyO3 LaunchAgent PID `87404` and that HEAD matched the pushed commit.
  The post-sample value packet reports continuity lift `0.075`, useful packet
  rate `0.6429`, memory-need precision `0.9286`, read-path p95 `87.0364ms`,
  read cache hit rate `0.5714`, and zero read budget misses/degraded
  reads/timeouts. This is useful evidence for the real-session window, but the
  durable goal remains open until longer Codex dogfood continuity is observed.
- 2026-05-28 post-restart cache-rescue sample: after the next resumed Codex
  turn created a stable same-project packet with one cold MCP context scan
  (`duration_ms=581.6269`, `project_file_fallback=550.3093ms`), a real restart
  to LaunchAgent PID `35144` preserved that packet cache. First post-restart
  AXI context returned via project-file cache rescue in `2.3597ms`; AXI recall
  returned three packets in `106.2332ms`; MCP `get_context` hit packet cache in
  `0.0617ms`; MCP `recall` was `cache_satisfied` in `89.6023ms`; and
  `engram axi value --json` reported read-path p95 `106.2332ms`, cache hit
  rate `1.0`, and zero read budget misses/degraded reads/timeouts. Recall
  evaluation sample `ers_ab1decfc5cfe` records this as useful real-session
  evidence.
- 2026-05-28 soft-wait rescue fix: a later installed-runtime check showed the
  MCP helper still waited for project-file scan completion after loaded-store
  soft wait, which could hide stable cache rescue on slow scans. The helper now
  returns after the loaded-store soft wait and lets the project-file payload
  path rescue from stable same-project cache when the scan is still pending.
  After reinstall/restart to PID `40680`, first fresh MCP context with no
  stable sidecar seed rebuilt in `937.6488ms`; once seeded, AXI fresh context
  used `project_file_cache_rescue` in `10.3801ms`, exact repeat context hit
  cache in `0.0413ms`, and fresh MCP contexts were bounded at `104.0038ms` and
  `138.8205ms` with no budget misses or degradation. Final reinstall/restart on
  PID `41982` stayed healthy and returned fresh AXI context through
  `project_file_cache_rescue` in `2.238ms`.
- 2026-05-28 continuation checkpoint: HEAD `78aa7ed` remained clean/pushed and
  the dogfood runtime passed startup validation plus
  `/private/tmp/engram-dogfood-startup-20260528-074024` (`13 pass, 0 warn,
  0 fail, 0 skip`). Post-matrix PID `43378` stayed healthy. AXI context was
  `38.2785ms`, AXI recall found a cue packet in `11.8581ms`, forced miss recall
  returned a project packet in `102.2185ms`, MCP `get_context` was
  `143.7264ms`, MCP `recall` was `cache_satisfied` in `2.2772ms`, and recall
  evaluation sample `ers_28654b6d8385` records the run.
- 2026-05-28 observe/projection cache-invalidation fix: a resumed Codex run at
  HEAD `5d3554d` showed that broad graph packet-cache invalidation after normal
  capture/projection could mark stable `project_home` file packets stale. The
  next MCP `get_context` still returned useful project packets, but paid
  `715.1811ms` total with `project_file_fallback=709.6484ms` because the stable
  rescue row had been invalidated. `MemoryPacketCache.invalidate(...)` now has a
  preserve mode for entries whose packets are all `trust.source=project_file`,
  and `GraphManager.invalidate_memory_packet_cache(...)` uses that mode for
  graph/episode mutations. After reinstall/restart to PID `45085`, AXI context
  seeded project-file cache rescue in `2.2862ms`; live MCP observe stored
  `ep_4c0605de51da`; background projection ingested it without invalidating the
  `project_home` file rows; MCP `get_context` returned useful packets in
  `83.4848ms`; AXI context hit `project_file_cache_rescue` in `3.608ms`; AXI
  recall was `cache_satisfied` in `2.3008ms`; and live value reported read-path
  p95 `83.8738ms`, read cache hit rate `0.7778`, and zero read budget misses,
  degradation, or timeouts.
- 2026-05-28 post-`1029cf7` reinstall/restart sample: the dogfood runtime was
  reinstalled from the clean checkout and restarted to LaunchAgent PID `48229`.
  AXI home stayed healthy; AXI context used project-file cache rescue in
  `2.8982ms`; repeat AXI recall was `cache_satisfied` in `1.9462ms`; and a
  forced no-evidence AXI recall returned three project packets in `102.0141ms`
  with `preflight_timeout_context_packet_fallback` and no degraded/empty
  timeout. MCP `get_context` returned useful project packets in `161.2856ms`;
  MCP `recall` was `cache_satisfied` in `5.6553ms`; validation passed with 27
  MCP tools, `remember`, and `recall.project_path`; and the confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260528-075531`
  (`13 pass, 0 warn, 0 fail, 0 skip`). Post-matrix PID `52284` stayed healthy.
  Live value reported read-path p95 `73.4648ms`, read cache hit rate `1.0`, and
  zero read budget misses, degradation, or timeouts. Recall-quality sample
  `ers_145ab372177d` records the session evidence.
- 2026-05-28 recall-priority follow-up: the next resumed Codex session exposed
  packet-cache scope starvation. A fresh `session_recent` observation
  `ep_971656d12ca9` surfaced once, then repeat recall could be satisfied by
  newer `project_home` packets because the context fallback fetched
  `session_recent`, `identity_core`, and `project_home` through one global
  recency-limited query. The fallback now reads those scopes separately before
  dedupe and relevance filtering. After reinstall/restart to PID `62723`, live
  marker `ep_0352d83b5ece` remained the first packet for repeated AXI recalls
  after two newer project-home warmups: `3.2818ms` and `2.2598ms`, both
  `cache_satisfied`, with no degradation or budget miss. Focused tests passed
  with `113 passed`, ruff passed, startup validation passed all 14 checks, and
  post-validation live value reported read-path p95 `81.5933ms`, read cache hit
  rate `0.95`, and zero read budget misses, degradation, or timeouts.

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
