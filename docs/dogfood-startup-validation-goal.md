# Dogfood Startup Validation Goal

## Goal Objective

Make Engram's local native dogfood path boringly reliable across Codex, Claude
Code, OpenClaw, and shell-only agents by turning the recent startup/config/MCP
issues into a repeatable validation matrix, then fixing every confirmed
breakage until a user can install, start, connect, inspect, restart, and resume
Engram without hidden state, duplicate config, stale processes, missing tools,
or unclear storage growth.

This goal is not complete when one manual run passes. It is complete only when
the workflow is documented, automated where practical, and verified from a
fresh or intentionally reset local state.

## Why This Exists

Recent dogfood exposed multiple related failures that all sit around the same
release boundary:

- Codex AXI startup worked, but project path selection depended on the launch
  directory.
- `remember` existed in Engram but was not visible until targeted tool
  discovery loaded it.
- Claude Code had duplicate Engram MCP definitions across user and project
  scopes.
- Claude Code's MCP health changed from failed to connected depending on
  whether the runtime was fully warm.
- `engramctl start/stop/status` and the local LaunchAgent path drifted in the
  native dogfood setup.
- A stale listener/PID state made port `8100` look occupied while health failed.
- Native startup spends time in vector integrity verification, which can make
  short client probes report false failures.
- Storage growth was visible only after adding explicit `engramctl storage`
  checks.

Treat these as one startup/adoption reliability problem, not separate one-off
bugs.

## Validation Matrix

Run the matrix on macOS first, with native PyO3 Helix as the primary path and
Lite mode only as a fallback smoke path.

| Area | Checks |
| --- | --- |
| Runtime lifecycle | `engramctl start`, `status`, `storage`, `doctor`, `stop`, restart after stop, restart after stale listener cleanup |
| Supervisor parity | LaunchAgent loaded/unloaded state, PID file behavior, port listener ownership, health readiness |
| Native startup | vector integrity phase, health latency, MCP readiness after warmup, no duplicate background processes |
| Codex | MCP tools exposed, `remember` visible, `get_context` works, AXI startup trace recorded, follow-up context trace recorded |
| Claude Code | no conflicting MCP scopes, project HTTP MCP connected, AXI hook installed if requested, startup/follow-up evidence recorded |
| OpenClaw | MCP config written, global CLI or `npx -y openclaw` fallback verified, docs match installed config, AXI fallback documented and smoke-tested where possible |
| Tool catalog | expected MCP tool count, `remember`, `observe`, `recall`, `get_context`, `bootstrap_project`, `claim_authority`, `route_question` exposed |
| Storage visibility | paths, sizes, counts, growth since startup, old data dirs called out before deletion |
| Failure handling | offline runtime, half-started runtime, stale PID file, port occupied, MCP probe timeout, duplicate client config |

## Current Checkpoint

2026-05-28 native PyO3 dogfood performance hardening pass:

- Full startup validation passed after the final reinstall/restart.
- The confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260528-101318` with `13 pass,
  0 warn, 0 fail, 0 skip`.
- Native Helix now marks completed vector integrity verification in storage
  metadata and skips the expensive migration-only scan on later dogfood starts.
  The first patched restart still verified the 4.0G store; the following start
  skipped the scan and stayed under the matrix timeout.
- The installer and `engramctl` update paths now reinstall `helix-native`
  explicitly when using the bundled/local native source, so PyO3 source changes
  are not hidden by the uv tool cache.
- MCP `observe` is bounded on the agent path: the final live samples returned
  `capture_store_timeout` at about `101ms` with no budget miss or degradation.
- `session_recent` packets now survive normal graph invalidation and can satisfy
  exact marker/date/id context queries. Final live probes returned the two fresh
  `rubymarker`/`emeraldmarker` observations through MCP context in `0.3724ms`,
  AXI context in `0.4443ms`, and AXI recall in `6.9803ms`.

2026-05-22 native PyO3 dogfood validation passes on the warmed local runtime:

- `python3 scripts/dogfood_startup_validation.py --json` passes native config,
  health, listener, LaunchAgent parity, project MCP config, storage path,
  `engramctl status`, `engramctl storage`, `engramctl doctor`, MCP tool catalog
  including `remember`, Codex config, Claude Code config, AXI traces, and
  OpenClaw config.
- After hardening LaunchAgent shutdown waits, a direct
  `engramctl stop && engramctl start` sequence passed in about 18.5 seconds.
- `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260522-161920/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`, covering warmed, stopped, restarted, and
  stale-PID states.
- 2026-05-23 rerun after packet-cache hardening and permission cleanup produced
  `/private/tmp/engram-dogfood-startup-20260523-192255/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-25 loaded-store context/recall hardening was reinstalled into the
  local native dogfood runtime. Cold REST context rebuilt project packets in
  about 1.05-1.26s, hot context cache hits returned in about 4-33ms, degraded
  REST/AXI/MCP recall returned cached packets instead of empty timeout payloads,
  and REST/MCP bootstrap now returns `already_bootstrapped` on the refreshed
  local store. Stale full-refresh bootstrap still needs a future live proof.
- The same runtime passed `python3 scripts/dogfood_startup_validation.py --json`
  and `python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle` produced
  `/private/tmp/engram-dogfood-startup-20260525-160226/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-26 continuation added a graph-free project-file fallback for
  cold-cache context timeouts. After reinstall/restart, the continuation MCP
  `get_context` probe returned useful project context instead of an empty
  timeout, REST/AXI context hit warmed `project_home` cache, and degraded recall
  returned cached packets. The validator passed again, and the lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-072233/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`. A fresh MCP `observe` call still timed
  out at the 120s client boundary before this reinstall, so capture latency
  remains a follow-up item.
- After the local permission update, the installed runtime was checked again.
  MCP `observe` now stores and queues projection successfully, but the live
  sample still spent about 11 seconds in `capture_store`. MCP/AXI context proved
  the graph-free project-file fallback on the loaded store, and MCP explicit
  recall still degraded under budget while returning cached project-file packets
  instead of an empty payload. The validator passed again, and the lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260526-073541/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`.
- A later 2026-05-26 pass bounded cue persistence behind raw episode capture,
  de-duplicated recent packet-cache fallbacks, and stopped the worker from
  racing raw queued events against cue-scheduled projection when the cue layer is
  enabled. After reinstall/restart, REST observe returned in the client window
  with `captureStore≈1008ms` and `cueStoreTimeout≈1002ms`; MCP context hit cache
  in about 0.3ms; MCP/AXI recall still degraded under deep search but returned
  three distinct cached project-file packets. The validator passed, and the
  lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-101234/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A follow-up cache-satisfaction pass made explicit recall skip deep search when
  recent cached project/identity packets strongly match the query. After warming
  context, `engram axi recall "native PyO3 Helix install" --timeout 10 --json`
  returned `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs≈25`; MCP recall reported `query_time_ms≈2.2` with the same
  `cache_satisfied` lifecycle. The validator passed again, and the lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260526-104900/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`.
- A later cache-relevance pass found that topic-specific context could still be
  satisfied by generic stable project-home packets. The MCP context surface now
  rejects stable-cache packets that do not match the specific topic and warms
  local project-file packets without loaded-store reads. Live evidence:
  `engram axi context --topic "cache relevance miss loaded store PyO3 recall docs"`
  returned five project-file packets in about 1.95s CLI wall, follow-up
  `engram axi recall` returned `status=ok`, `skipReason=cache_satisfied`,
  three packets, and `durationMs=3.1908`; hot MCP `get_context` returned from
  packet cache with `duration_ms=0.0809`, and MCP `recall` reported
  `query_time_ms=1.8`. The same restart surfaced a false-negative
  `engramctl start` readiness result on the loaded 4G native store, so the
  local Helix startup wait and startup-matrix health wait now default to 180s.
  The updated lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-114939/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The next 2026-05-26 performance pass bounded MCP write-side live-turn and
  recall-middleware side effects, bounded `/api/consolidation/status` latest
  cycle reads, and made embedded REST-mounted MCP skip its own background
  worker/cue-outbox/Redis publisher because the REST runtime already owns those
  process-local loops. Live evidence after reinstall: MCP `observe` stored
  `ep_2d9c43d681af` in 2.8s wall with `capture_store=32ms`,
  `cue_store_timeout=1001ms`, `live_turn=78.608ms`, and
  `recall_middleware_timeout=250.915ms`; MCP init logged
  `mcp_background_managed_externally` with no `mcp_worker_start`;
  `/api/consolidation/status` returned in about 0.56s with
  `skip_reason=latest_cycle_timeout` instead of hanging.
- The same matrix run exposed that optional consolidation cross-encoder model
  initialization could repeatedly fail on a missing local model and starve
  `/health`. Cross-encoder availability is now negative-cached per process so
  consolidation falls back after the first failure instead of retrying the same
  load repeatedly. After reinstall/restart,
  `python3 scripts/dogfood_startup_validation.py --repo /Users/konnermoshier/Engram --server-url http://127.0.0.1:8100`
  passed with `14 pass, 0 warn, 0 fail, 0 skip`, and
  `python3 scripts/dogfood_startup_matrix.py --repo /Users/konnermoshier/Engram --confirm-lifecycle`
  produced `/private/tmp/engram-dogfood-startup-20260526-122954/matrix-report.md`
  with `13 pass, 0 warn, 0 fail, 0 skip`.
- Cache-satisfied recall now considers a small set of cached packets together,
  not only the best single packet. The live query
  `"native PyO3 dogfood cue store timeout deep recall"` previously warmed useful
  context in about 1.6s but follow-up AXI recall still degraded after about
  2.8s. After the aggregate packet-coverage change and reinstall, the same
  follow-up AXI recall returned `status=ok`,
  `skipReason=cache_satisfied`, `durationMs=1.3717`, and about 0.32s CLI wall;
  MCP recall returned `skip_reason=cache_satisfied` with `query_time_ms=0.6`.
- Storage diagnostics now keep timed-out live count reads running in the
  background with their own 30s cap, so operator storage commands do not keep
  paying the same loaded-store timeout. Live evidence: explicit
  `/api/storage?live=true&timeoutSeconds=0.5` returned in about 0.52s with
  `countsStatus=cached_timeout` and `countsRefreshStatus=running`; a concurrent
  `engramctl storage` returned in about 0.21s with cached counts and full path
  visibility; after the 30s background cap elapsed, `/api/storage` showed
  `countsRefreshStatus=idle` and the log reported
  `background storage count refresh timed out after 30.0 seconds`.
- Empty degraded recall is now explicit instead of silent. The live forced-miss
  query `"zzzzquasarflux xylofract wugplinth nonmatching"` returned zero
  results and one `recall_diagnostic` packet on REST, AXI, and MCP, with
  `skipReason=recall_timeout`, `fallbackStatus=filtered` or `timeout`, and
  stage timing evidence. The fast fallback now filters unrelated cue/episode
  hits before they can mask a timeout. After warming project context with
  `engram axi context --project /Users/konnermoshier/Engram`, the useful query
  `"native PyO3 dogfood cue store timeout deep recall"` returned
  `status=ok`, `skipReason=cache_satisfied`, `durationMs=3.1484`, and three
  project packets.
- Fast fallback no longer pays the normal hybrid search path on Helix. The
  Helix search backend exposes BM25-only `search_episode_cues_fast` and
  `search_episodes_fast`, and GraphManager prefers those for the timeout-rescue
  fallback. After reinstall/restart, the same forced-miss query returned
  `fallbackStatus=miss` with REST `recallFallback=5.0037ms` and MCP
  `recall_fallback=0.6211ms`, instead of spending hundreds of milliseconds or
  timing out. The diagnostic packet now also includes
  `recall_retrieve_cancelled_ms`, which proved the remaining 1200ms cost is
  inside deep retrieval, not fallback or packet cache.
- Deep retrieval now has bounded substages for stats, primary search,
  activation/graph pools, planner search, episode/cue/chunk search, activation
  state loading, entity-name fallback, spreading, entity attributes, and
  graph-structural similarity. After reinstall/restart, the live forced-miss
  query `"zzzzquasarflux xylofract wugplinth nonmatching"` returned
  `status=ok`, `durationMs=801.8291`, `recallSearch=742.4659ms`, and no
  degraded timeout. The important stage evidence was
  `recallPrimarySearchTimeout=301.5487ms`,
  `recallEntityMatchTimeout=74.9526ms`, and `fallbackStatus=miss`, proving the
  empty loaded-store recall path now exits with bounded no-evidence semantics.
  The matching AXI smoke returned `status=ok`, `durationMs=617.67`, and no
  degraded timeout.
- A follow-up pass stopped paying the fast fallback cost before every normal
  deep recall. Fast fallback now runs only as a timeout rescue. After
  reinstall/restart and clearing packet cache, the cold AXI recall for
  `"loaded-store recall context performance dogfood"` returned `status=ok`,
  `durationMs=591.3883`, `fallbackStatus=not_run`, zero packets/results, and
  no degraded timeout. The same cold REST recall returned `status=ok`,
  `durationMs=810.7498`, `recallSearch=810.687ms`, and no degraded timeout.
  After AXI/MCP context warmed project-file packets, AXI recall returned
  `cache_satisfied` in `durationMs=1.7864`; MCP context returned from cache in
  `duration_ms=0.0687`, and MCP recall returned `cache_satisfied` in
  `duration_ms=1.6463`. The lifecycle matrix then produced
  `/private/tmp/engram-dogfood-startup-20260526-140146/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The same dogfood run exposed a separate capture-write risk after several
  deferred cue writes accumulated: one MCP observe spent about 59.4s in
  `capture_store`, and a REST observe later reported `captureStore=31616ms`.
  Deferred cue persistence now runs serially so timed-out cue writes cannot
  saturate native storage workers and starve later raw captures. After
  reinstall/restart, REST observe returned with `captureStore=58ms` and
  `cueStore=32ms`; AXI observe completed in about 0.39s wall; MCP observe
  returned with `capture_store=23ms`, `cue_store=55ms`, and only the expected
  `recall_middleware_timeout` side-effect cap. Focused tests then passed with
  `255 passed, 2 skipped`, and startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip` (`engramctl doctor` skipped by flag).
- A subsequent soak found that repeated observes still paid roughly the full
  1000ms cue-store timeout when a serialized cue write was already in flight.
  Capture now queues cue persistence immediately when the cue-store lane is
  busy instead of waiting for another timeout. After reinstall/restart, six
  sequential REST observes returned in about 70-119ms with `captureStore`
  between 5-21ms after the first sample and no cue-store timeout. A concurrent
  mixed soak returned six observes in about 47-153ms; five queued cue storage
  immediately with `cueStoreQueued=0.0`, while the one active cue write finished
  with `cueStore=74ms`. Project recall stayed hot with `cache_satisfied`
  (`durationMs=1.4748`) during the same concurrent run. A fresh no-evidence
  query still reached the outer recall timeout, but returned one
  `recall_diagnostic` packet instead of an empty timeout payload; the next
  recall target is the remaining hidden retrieve tail behind that diagnostic.
- A final 2026-05-26 capture-boundary pass found that successful cue upserts
  could still be reported as `cue_store_timeout` because capture waited for the
  whole cue follow-up task, including projection-state sync. Capture now
  acknowledges once the cue is persisted or queued, serializes background cue
  indexing, and tracks later cue work in the background. After reinstall/restart
  on the 4G native dogfood store, six sequential REST observes returned in
  about 100-391ms with no cue-store timeout. A concurrent mixed soak returned
  five queued observes in about 159-187ms and one active cue write in about
  487ms with `cueStore=307ms`, again with no cue-store timeout. MCP `observe`
  returned in about 0.57s with `capture_store=67ms`, `cue_store=54ms`,
  `live_turn=128ms`, and only the expected `recall_middleware_timeout≈251ms`.
  Explicit recall is now bounded rather than empty: true no-evidence and
  loaded-store project queries returned `status=ok` in about 0.8s with
  `fallbackStatus=not_run`, though loaded-store miss retrieval remains the next
  latency target. Focused backend tests passed with `258 passed, 2 skipped`,
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-143550/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A follow-up packet usefulness pass fixed a gap where project-file fallback
  scored long docs by their body but cached only early headings, so later recall
  could not match terms from late dogfood evidence. Project-file fallback now
  extracts topic-matched lines from a bounded 50k-character scan window, and
  explicit recall can return filtered context packets after a successful but
  empty live recall. Live evidence after reinstall/restart and packet-cache
  clear: AXI context for
  `"capture_store cue_store recall_middleware_timeout serialized cue persistence dogfood evidence"`
  surfaced late snippets such as `capture_cue_store_timeout_ms`,
  `capture_store=23ms`, and `cue_store=55ms`; the follow-up AXI recall returned
  `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs=1.4963`. MCP recall on the same warmed cache returned
  `skip_reason=cache_satisfied`, three packets, and `duration_ms=1.4921`.
  Focused tests passed with `87 passed`, ruff and `git diff --check` passed,
  and startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when
  doctor was skipped.
- A later recall-tail pass shortened the remaining explicit no-evidence miss
  path. Explicit recall now uses configurable
  `recall_budget_explicit_search_ms=900` by default, caps timeout-rescue
  fallback with `recall_fast_fallback_timeout_ms=150`, records working-memory
  and episode/cue candidate counts, and short-circuits zero-score
  non-semantic pools before graph-heavy scoring. After reinstall/restart on
  the 4G native PyO3 dogfood store, fresh REST forced-miss probes returned
  diagnostic packets in about 1.09-1.12s wall with `maxSearchMs=900` and
  `recallFallback≈151ms`; the real no-evidence project query returned
  `status=ok` in about 0.60s. AXI warmed recall stayed cache-satisfied with
  `durationMs=0.7747`, MCP warmed recall stayed cache-satisfied with
  `duration_ms=0.8048`, and MCP forced-miss recall returned a diagnostic packet
  with `duration_ms=1054.6106` instead of the prior 1.35-1.86s miss tail.
  Focused tests passed with `27 passed`, the broader focused backend suite
  passed with `159 passed, 2 skipped`, ruff and `git diff --check` passed, and
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped.
- The next pass tightened dogfood latency defaults to
  `recall_budget_explicit_search_ms=650` and
  `recall_fast_fallback_timeout_ms=50`, added bounded reranker/MMR and recall
  post-processing stages, and made cached-packet satisfaction match separator
  variants such as `cue_store` versus "cue store". After reinstall/restart on
  the 4G native dogfood store, AXI forced miss returned one diagnostic packet
  in `durationMs=709.785`, REST forced miss returned `durationMs=704.1293` in
  about 730ms wall, and MCP forced miss returned `duration_ms=711.188`.
  Warmed project recall for
  `"capture_store cue_store recall_middleware_timeout serialized cue persistence dogfood evidence"`
  returned `cache_satisfied` without live search on all three surfaces: AXI
  `durationMs=1.097`, MCP `duration_ms=0.9786`, and REST `durationMs=0.9048`
  in about 3ms wall. `setup --mode helix` now preserves an existing custom
  `ENGRAM_HELIX__DATA_DIR`, preventing operator reinstalls from accidentally
  switching away from the dogfood store. Focused tests passed with
  `193 passed, 2 skipped`, ruff and `git diff --check` passed, and startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped.
- Degraded explicit recall now has a recent identity/project packet fallback
  when strict query filtering misses. This keeps the strict cache-satisfaction
  gate intact while avoiding diagnostic-only payloads under real agent traffic.
  After reinstall/restart, the warmed broad project query
  `"native PyO3 dogfood runtime performance hardening packet cache loaded store current goal"`
  returned `cache_satisfied` on AXI (`durationMs=1.1821`), REST
  (`durationMs=1.123`, about 19ms wall), and MCP (`duration_ms=0.6021`). A
  deliberately unrelated degraded query returned three recent `project_home`
  packets on AXI, REST, and MCP instead of only a diagnostic; MCP reported
  `duration_ms=700.989`. MCP context for the warmed topic returned from packet
  cache in `duration_ms=0.0654`. Focused tests passed with
  `194 passed, 2 skipped`, ruff and `git diff --check` passed, startup
  validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor was
  skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-155840` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- Cold degraded recall now also falls back to local project-file packets when
  the packet cache is empty. `recall` accepts `project_path`, `engram axi recall`
  exposes `--project`, and the no-argument MCP/REST fallback uses the server
  working directory only when it looks like a project. The recall-specific
  project scan is capped at 40 candidates and 12k chars per topic scan so this
  path stays useful without becoming another loaded-store-sized delay. After
  reinstall/restart and `engram axi packet-cache clear`, MCP no-arg recall for
  `"Engram dogfood Codex real sessions evidence performance hardening current goal cold bounded fallback 20260526"`
  returned three `project_home` packets with `duration_ms=751.294` and
  `project_file_recall_fallback=46.122ms`. REST cold no-project recall returned
  three project packets with `durationMs=928.1492`; AXI explicit-project recall
  returned three project packets with `durationMs=741.9536`. Focused tests
  passed with `197 passed, 2 skipped`, ruff and `git diff --check` passed,
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-161716` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The cold project-file fallback now reads bounded file prefixes instead of
  loading whole files before slicing. On the Engram repo, this moved live Codex
  MCP recall for
  `"Engram dogfood real Codex sessions replay evidence commands memory value performance current goal cold prefix read 20260526"`
  from `project_file_recall_fallback≈1979.5ms` to
  `project_file_recall_fallback=57.7283ms`, with total
  `duration_ms=763.5911` and three returned project packets. REST cold
  no-project recall returned three project packets with `durationMs=760.8558`
  and `projectFileRecallFallback=54.9111`; AXI explicit-project recall returned
  three project packets with `durationMs=739.5595`. Focused tests passed with
  `197 passed, 2 skipped`, ruff and `git diff --check` passed, and startup
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
- Capture now defaults `capture_cue_store_timeout_ms` to `250ms` instead of
  `1000ms`, so live agents do not pay a full second when cue persistence is
  slow after the raw episode is durable. The regression test forces a slow cue
  store under the default config and verifies capture returns under 500ms while
  cue persistence still drains in the background. After reinstall/restart,
  sequential REST observes returned in `110.5ms`, `87.5ms`, and `55.3ms`; a
  diagnostic REST observe reported `captureStore=16ms`, `cueStore=49ms`, and
  `cueIndexEnqueue=2ms`. MCP `observe` reported `capture_store=17ms`,
  `cue_store=39ms`, `live_turn=77.6472ms`, and the still-separate
  `recall_middleware_timeout=253.0875ms`; AXI observe completed in about
  `0.30s` wall. Focused capture tests passed with `27 passed`, ruff passed,
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-170217` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The MCP write-side enrichment budget is now `75ms` instead of `250ms`, so
  `observe`/`remember` still allow very fast inline `recall_lite` enrichment
  but do not pay the full quarter-second when session prime or auto-recall is
  slow. After reinstall/restart, MCP `observe` reported
  `capture_store=33ms`, `cue_store=55ms`,
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
- Live dogfood then exposed a foreground capture starvation path caused by the
  installed AutoCapture session-end hook starting `/api/consolidation/trigger`
  on the same local runtime. Before the fix, an MCP `observe` reported
  `capture_store=2556ms` and `cue_store_timeout=252ms`; direct REST observe
  probes then reported `captureStore=1893ms` followed by a `16346ms`
  `captureStore` sample while consolidation status showed `is_running=true`.
  Session-end AutoCapture now records only the session-end marker and lets the
  runtime scheduler own consolidation, and `install_hooks()` refreshes stale
  first-party Engram AutoCapture scripts while preserving custom user scripts.
  The installed `/Users/konnermoshier/.engram/hooks/session-end.sh` was
  refreshed and confirmed to contain no `/api/consolidation/trigger` call.
  After reinstall/restart, eight REST observes stayed between `55-82ms` wall
  with `captureStore=13-29ms`, `cueStore=31-45ms`, and no cue timeouts. MCP
  `observe` returned with `capture_store=19ms`, `cue_store=40ms`,
  `live_turn_timeout=75.7847ms`, and `recall_middleware_timeout=76.332ms`.
  The refreshed session-end hook smoke posted only `/api/knowledge/auto-observe`
  and no consolidation trigger. AXI context for the same topic returned useful
  project-file packets in about `0.39s` wall, AXI recall was `cache_satisfied`
  with `durationMs=0.7873`, MCP context hit packet cache with
  `duration_ms=0.0573`, and MCP recall stayed `cache_satisfied` with
  `duration_ms=3.452`. Focused tests passed with `33 passed`, ruff passed, and
  startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip` when doctor
  was skipped. The confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-172521` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-26 latency follow-up removed three dogfood stalls from the native
  path. Project-file fallback packet writes now stay process-local instead of
  persisting one-off topic packets to the SQLite sidecar; after clearing packet
  cache, REST context rebuilt useful project packets in `60.2ms` wall
  (`durationMs=58.0146`, `projectFileFallback=56.3851`) and MCP context rebuilt
  useful packets with `duration_ms=49.2701` in the MCP budget. Timed-out live
  storage count refreshes now cancel instead of continuing as background graph
  scans; after `/api/storage?live=true&timeoutSeconds=1` returned
  `countsRefreshStatus=idle`, six REST observes stayed between `62.05-92.67ms`
  wall with `captureStore=19-51ms` and `cueStore=33-50ms`. Mounted streamable
  HTTP MCP now attaches to the already-started REST `GraphManager` instead of
  lazily creating a second native runtime; post-matrix MCP observe on PID
  `54432` returned in `0.2473s` wall with `capture_store=19ms`,
  `cue_store=29ms`, and no cue timeout. Startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip`, the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-175129` with
  `13 pass, 0 warn, 0 fail, 0 skip`, focused tests passed with `53 passed`, and
  ruff plus `git diff --check` passed.
- 2026-05-26 bounded fallback follow-up reduced cold topic fallback variance by
  pre-ranking path-relevant project files and lowering the default topic scan
  window. Before the patch, a controlled topic could spend `623-930ms` inside
  `projectFileFallback`, and one MCP context sample exceeded the `2000ms`
  budget while still returning useful packets. After reinstall/restart, five
  REST context calls with packet-cache clear stayed between `54.91-59.94ms`
  wall with `projectFileFallback=52.584-56.6592ms`, MCP context returned useful
  packets with `duration_ms=64.5478`, AXI recall was `cache_satisfied` in
  `0.5715ms`, MCP recall was `cache_satisfied` in `0.8831ms`, and AXI value
  reported `p95_added_latency_ms=250.6712` over 10 measured operations. Focused
  tests passed with `54 passed`, ruff passed, startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip`, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-180521` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime PID `57400` stayed
  healthy, MCP context was `duration_ms=52.287`, and MCP observe recorded
  `capture_store=14ms`, `cue_store=32ms`.
- 2026-05-26 follow-up fixed the public project-file fallback wrapper, which
  still defaulted topic scans to `50_000` chars even though the lower-level
  helper had moved to `16_000`. The live MCP regression was visible immediately
  after restart: one topic-specific context call reported
  `project_file_fallback=1494.5866ms` and a budget miss. After reinstalling the
  wrapper fix and clearing packet cache, the same topic returned useful packets
  with `duration_ms=40.1795` and `project_file_fallback=39.1043`; AXI recall was
  `cache_satisfied` in `1.3291ms`, MCP recall was `cache_satisfied` in
  `0.838ms`, and post-matrix AXI value reported `p95_added_latency_ms=74.0474`
  over 4 measured operations with no budget misses. The confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260526-181432` with
  `13 pass, 0 warn, 0 fail, 0 skip`; post-matrix runtime PID `60090` stayed
  healthy, and MCP context was `duration_ms=56.5303` with
  `project_file_fallback=52.9783`. Focused tests passed with `55 passed`, ruff
  passed, startup validation passed with `13 pass, 0 warn, 0 fail, 1 skip`, and
  `git diff --check` passed.
- 2026-05-26 explicit-recall fallback prebuild removed the remaining long
  degraded recall tail. Before the patch, a cold REST recall after packet-cache
  clear returned useful project-file packets but still took `2706.5797ms` wall
  with `durationMs=2703.8135` because `projectFileRecallFallback=1993.9967`
  ran only after loaded-store recall timed out. Project-file rescue packets now
  build in a side task while loaded-store recall runs, are cached only if the
  live recall actually degrades, and are consumed immediately on timeout. After
  reinstall/restart, the same cold REST path returned 3 useful packets in
  `705.2311ms` wall with `durationMs=702.7804`,
  `projectFileRecallFallbackWait=0.198`, and
  `projectFileRecallFallback=41.8739`. AXI cold recall returned 3 useful
  packets with `durationMs=706.3645`; MCP cold recall returned 3 useful packets
  with `duration_ms=746.481`, `project_file_recall_fallback_wait=0.1875`, and
  `project_file_recall_fallback=282.2759` while the build overlapped the
  loaded-store search. The calls still mark `recall_timeout` when the 650ms
  loaded-store search budget is exhausted, but they no longer produce empty or
  multi-second timeout payloads. Focused tests passed with `80 passed`, ruff and
  `git diff --check` passed, startup validation passed with
  `13 pass, 0 warn, 0 fail, 1 skip`, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-183141` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime PID `63694` stayed
  healthy, and AXI value reported `p95_added_latency_ms=90.4855` with no budget
  misses over 4 fresh operations.
- 2026-05-26 empty-success recall follow-up made cold explicit recall useful
  even when loaded-store search finishes under budget with zero items, and
  tightened recall fallback scanning so the side task does not stretch MCP
  turns. Empty successful recalls now consume the already-started project-file
  rescue task instead of returning a blank `ok` payload. The project-file
  fallback now ranks the full candidate path list cheaply, then reads only the
  top 16 candidates with a `6000` character topic window for explicit recall.
  After reinstall/restart, cold no-project REST recall returned 3 useful
  project packets in `durationMs=705.5507` with
  `projectFileRecallFallback=8.4883`; a repeated REST recall was
  `cache_satisfied` in `durationMs=0.6293`. AXI cold recall returned 3 useful
  packets with `durationMs=721.2843`, and MCP cold recall returned 3 useful
  packets with `duration_ms=704.7271`,
  `project_file_recall_fallback_wait=0.1659`, and
  `project_file_recall_fallback=16.8017`. Focused tests passed with
  `82 passed`, ruff and `git diff --check` passed, startup validation passed
  with `13 pass, 0 warn, 0 fail, 1 skip`, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260526-184734` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime PID `67784` stayed
  healthy, and AXI value reported `p95_added_latency_ms=92.414` with no budget
  misses over 4 fresh operations.
- 2026-05-26 loaded-store recall tail follow-up stopped secondary graph stages
  from consuming the explicit recall budget after graph preflight had already
  timed out. Primary search now caps to `150ms` after a stats/graph-expansion
  timeout, graph expansion, graph pool, goal priming, cross-domain seed typing,
  spreading, entity-attribute boosts, GC-MMR, and near-miss materialization are
  skipped in that slow-graph state, noop reranking no longer materializes graph
  docs, and primary materialization graph reads cap to `15ms` in the same state.
  Before the final caps, the same cold REST recall degraded at
  `durationMs=704.792` with `recall_timeout`, `recallRetrieve=371.8813`, and
  `recallMaterializeCancelled=278.1135`. After reinstall/restart, cold REST
  recall for `next bottleneck loaded-store recall` returned `ok` with 3 project
  packets in `durationMs=383.8734`; `recallRetrieve=297.0664`,
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
- 2026-05-26 direct loaded-store fallback follow-up addressed the remaining
  empty-materialization tail. Helix BM25 now exposes fast episode/cue record
  methods so timeout rescue can build cue/episode recall results without
  scanning `find_episodes_by_group`, all-zero ordered BM25 rows receive
  rank-based scores, the cold rescue budget is `100ms`, and REST/MCP run the
  fast loaded-store fallback when deep recall returns `ok` with zero items and
  no cache/context packets. After reinstall/restart, cold REST recall for the
  current Codex stuck-check query returned `ok`, `itemCount=5`,
  `packetCount=3`, `durationMs=657.0303`, `fallbackStatus=hit`, and
  `recallEmptySuccessFallback=46.0678` with rank scores
  `[1.0, 0.5, 0.3333, 0.25, 0.2]`. Cold AXI recall returned `ok` with
  `result_count=5`, `packet_count=3`, and `durationMs=595.7207`. Cold MCP
  recall returned `ok` with `result_count=1`, `packet_count=1`, and
  `duration_ms=472.8815`. Validation: focused tests
  `94 passed, 23 skipped`, startup validation `14 pass, 0 warn, 0 fail,
  0 skip`, and ruff passed on the touched recall, graph manager, Helix search,
  and focused test files.
- 2026-05-26 planner, similarity, and post-process follow-up removed the next
  loaded-store cold-recall tail. After graph preflight timeout, primary search
  now caps at `100ms`, semantic similarity backfill is skipped when primary
  search already timed out, and planner work is skipped/deferred when the
  candidate pool only has zero-semantic special episode/cue candidates. Recall
  post-processing now exposes `recallConfidence` and
  `recallFingerprintRecord` timings, caps relevance-confidence scoring at
  `50ms`, and reuses existing semantic similarity by default instead of
  re-embedding returned episode/chunk text. After reinstall/restart on the same
  4.0 GB native dogfood store, clean cold REST recall for
  `"deep recall performance loaded-store bottleneck Codex dogfood q2"` returned
  `ok`, `itemCount=5`, `packetCount=3`, `durationMs=136.3629`,
  `fallbackStatus=hit`, `recallPostProcess=0.2971`,
  `recallConfidence=0.0424`, and `recallEmptySuccessFallback=1.7816`.
  Clean cold AXI recall for the same query returned `ok`, `result_count=5`,
  `packet_count=3`, and `durationMs=377.448`. Clean cold MCP recall returned
  `ok`, `result_count=5`, `packet_count=3`, `duration_ms=224.3714`,
  `recall_post_process=1.4265`, and `recall_confidence=0.0727`. Startup
  validation passed with all checks passing against LaunchAgent PID `87157`,
  including native storage visibility, MCP `remember` catalog exposure, Codex,
  Claude Code, AXI hook, and OpenClaw config checks. Focused tests passed with
  `189 passed, 23 skipped`; the narrower confidence/post-process tests passed
  with `28 passed`; ruff and `git diff --check` passed. Post-probe
  `engram axi value` still showed one budget miss in the recent mixed sample
  (`budget_miss_rate=0.0714`, `p95_added_latency_ms=767.4577`), so the broader
  dogfood goal remains active even though the clean post-process probe is fixed.
- 2026-05-26 project-file fallback prefix-cache follow-up addressed the next
  cache-warming rough edge: after packet-cache clears, topic-specific local
  project context sometimes spent `744-1907ms` in `projectFileFallback` even
  though the same helper was fast in isolation. Project-file fallback now keeps
  an mtime/size-aware in-process prefix cache for local project docs so new
  query variants and packet-cache clears can reuse bounded snippets without
  rereading the same files. After reinstall/restart on LaunchAgent PID `89509`,
  cold REST context for a long Engram dogfood topic returned 5 project-file
  packets with `durationMs=44.1913` and `projectFileFallback=43.5274`; a second
  cold topic returned `durationMs=24.7345` and
  `projectFileFallback=24.0498`. Hot REST context returned in `0.0246ms`,
  hot REST recall was `cache_satisfied` in `0.5346ms`, hot MCP context returned
  in `0.0216ms`, and hot MCP recall was `cache_satisfied` in `0.6661ms`. Cold
  loaded-store q2 recall stayed healthy: REST `durationMs=343.6238`, AXI
  `durationMs=332.4911`, and MCP `duration_ms=333.363`, all `ok` with 5
  results and 3 packets. Startup validation passed all checks against PID
  `89509`. Validation: focused context/recall/AXI tests `110 passed`, ruff
  passed, and `git diff --check` passed. Post-validation `engram axi value`
  improved but still showed one recent mixed-sample budget miss
  (`budget_miss_rate=0.0385`, `p95_added_latency_ms=343.4741`), so the broader
  dogfood goal remains active.
- 2026-05-26 session-prime and value-report follow-up removed the remaining
  mixed-sample startup penalty. MCP session prime is now truly cache-only:
  it loads cached identity/project packets or records a cheap `cache_miss`
  skip instead of calling the full `manager.get_context()` path inside the
  `250ms` startup budget. AXI value now requests live runtime cost with
  `liveCost=True`, so `engram axi value` no longer replays stale persisted
  operation windows after a clean restart. After reinstall/restart, immediate
  AXI value reported `source=live_runtime`, `operation_count=0`, and
  `budget_miss_rate=0` instead of the previous saved miss. Post-matrix MCP
  context returned useful project packets with `duration_ms=129.8165` and
  `budget_miss=false`; MCP session-prime attached cached project packets in
  `1.9447ms` with `max_packets=2`, `budget_miss=false`, and `degraded=false`;
  MCP recall was `cache_satisfied` in `0.7062ms`, and AXI recall was
  `cache_satisfied` in `0.3862ms`. Final `engram axi value` reported
  `source=live_runtime`, `operation_count=15`, `p95_added_latency_ms=129.8354`,
  `cache_hit_rate=0.8571`, and `budget_miss_rate=0`. Startup validation passed
  all checks against PID `94072`; the solo lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-214331` with
  `13 pass, 0 warn, 0 fail, 0 skip`; the final LaunchAgent runtime was healthy
  on PID `96664`. Focused tests passed with `131 passed` and 6 existing
  AsyncMock coroutine warnings in `tests/test_autorecall.py`; ruff passed.
- 2026-05-26 continuation traced the next live capture spike to cue-vector
  indexing cancellation on the local FastEmbed/native Helix path. The
  `capture_cue_vector_index_timeout_ms` threshold is now a soft background-lane
  threshold: capture still acknowledges after raw episode/cue persistence, but
  the cue-index task is shielded from cancellation so a slow-but-successful
  local vector write can finish and clear the SQLite cue-index outbox instead
  of leaving failed retry rows. After local reinstall/restart on PID `495`,
  direct REST observes reported `captureStore=8-36ms`, `cueStore=36-50ms`, and
  `cueIndexEnqueue=1-3ms`; MCP `observe` stored `ep_b4b112e7ed20` with
  `capture_store=30ms`, `cue_store=51ms`, `live_turn_timeout=74.5353ms`, and
  `recall_middleware=53.5501ms`. The cue-index outbox stayed at the pre-existing
  `failed=26` instead of growing, and `engram axi value` on the fresh runtime
  reported `source=live_runtime`, `operation_count=8`, `p95_added_latency_ms=213.0971`,
  and `budget_miss_rate=0`. Installed-user startup validation passed all 14
  checks against the same runtime, including storage visibility, MCP tool
  catalog, Codex/Claude Code config, AXI hooks, and OpenClaw config. Focused
  capture tests passed with `14 passed`; ruff passed on the touched capture
  files.
- 2026-05-26 continuation follow-up made packet-cache warming useful across
  agent surfaces. Explicit recall packet payloads now use one shared
  `explicit_recall` cache scope instead of separate `axi_recall`, `api_recall`,
  and `mcp_recall` namespaces, while the measured operation source still stays
  per caller. MCP auto-recall also considers that shared explicit-recall scope
  before running medium recall. After reinstall/restart on LaunchAgent PID
  `4695` and `engram axi packet-cache clear`, cold AXI recall for
  `"Engram AXI packet cache performance dogfood recall middleware cached patch"`
  built 3 packets in `durationMs=379.5135`. The same query then returned
  `cache_satisfied` through MCP in `duration_ms=3.3138` and through REST in
  `durationMs=2.2597`. The middleware auto-recall sample executed from cache in
  `1.048ms` with `skip_reason=cache_satisfied`, `auto_recall_packet` hit in
  `1.03ms`, and there were no medium recall timeouts or degraded samples in the
  fresh run. `engram axi value` reported `source=live_runtime`,
  `operation_count=8`, `p95_added_latency_ms=379.3022`, `cache_hit_rate=0.8333`,
  and `budget_miss_rate=0`. Startup validation passed with `14 pass, 0 warn,
  0 fail, 0 skip`. Focused tests passed with `116 passed` and 6 existing
  AsyncMock coroutine warnings; ruff and `git diff --check` passed.
- After the same live runtime surfaced slow capture/context interference, cue
  vector indexing was moved behind a `1000ms` rework-profile quiet period. Cue
  indexing still persists durable work to the SQLite outbox immediately, but
  best-effort vector writes now wait until the live capture burst has been quiet
  before touching the native Helix vector path. Before the patch, an MCP
  `observe` sample reported `capture_store=2960ms`, `cue_store_timeout=252ms`,
  and `recall_middleware_timeout=76.1255ms`; an MCP `get_context` sample
  returned useful project packets but spent `project_file_fallback=1672.5267ms`
  and `cache_relevance_miss=205.6315ms`. After reinstall/restart on LaunchAgent
  PID `7032`, REST `observe` returned with `captureStore=34ms`, immediate REST
  context after packet-cache clear returned useful project packets with
  `durationMs=61.983` and `projectFileFallback=61.081`, and a follow-up REST
  observe returned with `captureStore=19ms`. MCP `observe` returned in
  `153.8472ms` wall with `capture_store=39ms`, `cue_store=51ms`,
  `live_turn=56.6796ms`, and `recall_middleware=2.0812ms`; MCP session-prime
  loaded cached project packets in `1.1508ms`, and MCP `get_context` hit packet
  cache in `5.0142ms`. The cue-index outbox stayed at the pre-existing
  `failed=26`. Startup validation passed with `14 pass, 0 warn, 0 fail,
  0 skip`, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-223823` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Focused tests passed with `108 passed`;
  ruff and `git diff --check` passed.
- A follow-up on the same loaded native store found that synchronous SQLite
  cue-index outbox writes could still block the live event loop. The outbox
  enqueue now runs through the async threadpool, preserving durable cue-index
  recovery while keeping the agent turn clear. After reinstall/restart on
  LaunchAgent PID `10142`, five REST observe probes returned
  `captureStore=22/8/7/7/6ms`, `cueStore=28-35ms`, and
  `cueIndexOutboxEnqueue=1-3ms`. Live MCP probes then stored
  `ep_4df1441683d9`, `ep_d19482822af6`, and `ep_af320c530be7` with
  `capture_store=26/10/8ms`, `cue_store=44/29/25ms`,
  `cue_index_outbox_enqueue=3/1/1ms`, `recall_middleware=1.3-1.8ms`, and no
  degraded recall middleware. MCP explicit recall was `cache_satisfied` in
  `1.578ms`, and auto-recall packet cache p95 stayed around `1.45ms`. The
  cue-index outbox stayed at the pre-existing `failed=26`, and the runtime
  remained healthy in native PyO3 mode on the 4.0G dogfood store. Focused
  tests passed with `108 passed`; ruff and `git diff --check` passed; startup
  validation passed all checks; the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-225237` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- The next post-matrix probe found the remaining cold-context cost was not the
  local project-file scan itself. The pre-fallback packet-cache relevance check
  could spend about a second opening the persistent SQLite packet-cache sidecar
  while another local operation had the DB busy. Packet-cache persistence is now
  opportunistic for agent-turn reads: sidecar lookups use a short SQLite timeout,
  transient locks do not permanently disable persistence, and in-memory hits are
  preserved when the sidecar is temporarily locked. After reinstall/restart on
  LaunchAgent PID `13990`, three cold REST context probes returned
  `durationMs=31.6594/17.3439/21.6871` with
  `cacheRelevanceMiss=1.0287/0.7544/0.7145ms`; MCP `get_context` returned
  useful project packets in `33.9049ms`, MCP session-prime hit packet cache in
  `1.3005ms`, MCP `observe` stored `ep_e911aefd02cb` with
  `capture_store=26ms`, `cue_store=45ms`, and `recall_middleware=2.0325ms`;
  MCP `recall` was `cache_satisfied` in `2.0937ms`. Fresh live cost reported
  `operation_count=15`, `p95_added_latency_ms=137.8351`,
  `budget_miss_rate=0`, and no degraded API/AXI/MCP context or recall samples.
  Focused tests passed with `143 passed`; ruff and `git diff --check` passed;
  startup validation passed all checks against PID `13990`; the confirmed
  lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260526-230849`
  with `13 pass, 0 warn, 0 fail, 0 skip`. After that matrix restart, the
  runtime was healthy on PID `15399`; a cold REST context probe still returned
  useful packets in `61.7303ms` with `cacheRelevanceMiss=3.5983ms`; a cold MCP
  context probe returned useful packets in `351.8508ms`, with the cost in local
  project-file fallback rather than packet-cache sidecar lookup.
- The follow-up cache persistence pass made project-file fallback packets
  survive restarts and share across related topics. Fallback packets are now
  cached under both the exact topic key and the stable project key, persisted to
  the packet-cache sidecar, and hot persistent hits no longer write hit-count
  metadata on every agent turn. After reinstall/restart on LaunchAgent PID
  `18399`, a seeded REST context fallback took `55.34ms`; after a second
  restart, the first related REST context hit persisted `project_home` packets
  in `0.036ms` with no `projectFileFallback` stage. MCP `get_context` for the
  same topic hit `project_home` in `0.033ms`, MCP session-prime was
  `1.6562ms`, and MCP `observe` stored `ep_113bf3c057a1` with
  `recall_middleware=2.1446ms`. Focused tests passed with `144 passed`; ruff
  passed; startup validation passed all checks against PID `18827`; and the
  confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-232931` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix, the runtime was healthy on
  PID `19925`; a deliberately new REST topic still used bounded local fallback
  in `52.6259ms`, and a related follow-up hit persisted `project_home` packets
  in `0.0365ms`.
- The next live Codex turn showed that MCP session-prime could still pay a
  one-off persistent-cache sync on the first tool call after restart:
  `mcp_session_prime` spent `770.252ms` in `packet_cache` even though later
  auto-recall packet hits were about `2ms`. Session-prime now uses only
  in-memory packet-cache entries (`sync_persistent=False`) because startup
  already preloads the persistent sidecar. After reinstall/restart on
  LaunchAgent PID `22311`, the first MCP `get_context` session-prime loaded
  cached project packets in `0.1164ms`; the main context call used bounded
  local fallback in `49.6486ms`; follow-up REST context hit `project_home` in
  `0.1377ms`; AXI recall was `cache_satisfied` in `1.3637ms`; MCP recall was
  `cache_satisfied` in `2.1577ms` with `packet_cache=0.7901ms`; and MCP
  `observe` stored `ep_b81d207e3672` with `capture_store=70ms`,
  `cue_store=40ms`, and `recall_middleware=1.89ms`. The fresh live cost report
  showed `operation_count=16`, `p95_added_latency_ms=199.2932`,
  `budget_miss_rate=0`, `degraded_rate=0`, and `cache_hit_rate=0.9167`.
  Focused tests passed with `183 passed` plus the pre-existing AsyncMock
  warnings; ruff and `git diff --check` passed. Startup validation passed all
  checks against PID `22311`, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260526-234715` with
  `13 pass, 0 warn, 0 fail, 0 skip`. After that matrix restart, PID `23587`
  stayed healthy and the first post-matrix MCP session-prime was `0.1323ms`
  with no budget miss.
- The next continuation isolated a new capture-store contention pattern rather
  than a session-prime regression. Before the fix, a live MCP `observe` spent
  `capture_store=3023ms`, and a six-sample REST burst had `captureStore`
  outliers at `2820ms` and `3275ms`. The root causes were low-priority
  foreground probes starting expensive graph-state reads for `liveCost=true`
  reports, plus cue-vector indexing treating slow raw capture time as idle
  time. Live-cost reports now use runtime metrics without launching graph-state
  scans, and cue-vector quiet time is re-anchored after durable capture/cue
  writes. After reinstall/restart, MCP `get_context` session-prime was
  `0.1296ms`, MCP `observe` stored `ep_31b0f81edcdd` with
  `capture_store=47ms`, `cue_store=40ms`, and `recall_middleware=6.3583ms`,
  and an immediate eight-sample REST burst returned in `53-90ms` wall with
  `captureStore=19-31ms`. The fresh live-cost report showed
  `operation_count=15`, `p95_added_latency_ms=187.0021`,
  `api_observe.p95=79.8831`, `mcp_session_prime=0.1296ms`, no budget misses,
  no degraded cost samples, and an explicit `graph_state` skip reason of
  `live_cost_runtime_only`. AXI context returned project-file packets, AXI
  recall found the new cue packet in `durationMs=385.9009`, and MCP recall was
  `cache_satisfied` in `duration_ms=0.3854`. Focused tests passed with
  `26 passed`, adjacent API/report/capture/storage tests passed with
  `91 passed`, ruff and `git diff --check` passed, startup validation passed
  all checks against PID `26326`, and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-000715` with
  `12 pass, 0 warn, 0 fail, 0 skip` (`--skip-stale-pid`).
- AXI now infers the current project path from the working directory when
  normal project markers are present, so agents do not need to pass `--project`
  from a repo root. After reinstall/restart on LaunchAgent PID `32331`,
  `engram axi --json` reported
  `brain.project=/Users/konnermoshier/Engram`; `engram axi context --topic
  "startup resident packet cache project home native PyO3 recall sequential
  verification" --json` returned five project-file packets without an explicit
  `--project`; and the sequential follow-up `engram axi recall ... --json`
  returned `status=ok`, `skipReason=cache_satisfied`, three packets, and
  `durationMs=24.0405`. A deliberately parallel context/recall probe can still
  race the cache-warm write, so startup hooks should run context before relying
  on recall cache satisfaction. Running from `/Users/konnermoshier`, which is
  not detected as a project directory, correctly leaves project context empty.
- Recall fallback telemetry now distinguishes the packet path that rescued an
  otherwise empty response. After reinstall/restart on LaunchAgent PID `35530`,
  unrelated no-evidence AXI recall returned `status=ok`,
  `fallbackStatus=context_packet_fallback`, three project packets, and
  `durationMs=266.6227`; a warmed AXI context followed by recall returned
  `fallbackStatus=cache_satisfied` and `durationMs=0.8506`. REST recall on the
  same no-evidence shape reported `fallbackStatus=project_file_recall_fallback`
  in about `408ms` wall instead of inheriting a stale `timeout` label. Focused
  tests passed with `82 passed`, ruff and `git diff --check` passed, startup
  validation passed in skip-slow mode, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260527-010214` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix `engramctl status` reported a
  healthy native PyO3 runtime on PID `36477`.
- AXI trace evidence now carries redaction-safe usefulness metadata instead of
  only status and wall time. The trace writer records cache hit, packet count,
  result count, fallback status, skip reason, and budget-miss/degraded flags
  without serializing packet or transcript content; hook status and startup
  validation compact the same fields. After reinstalling with the local
  `helix-native` PyO3 add-on and restarting on LaunchAgent PID `40221`, a real
  Codex follow-up recall wrote `cacheHit=true`, `fallbackStatus=cache_satisfied`,
  `packetCount=3`, `resultCount=0`, `budgetMiss=false`, and `durationMs=5`.
  `engram axi doctor --hooks codex claude-code --require-hook-run
  --require-followup --json` surfaced those fields under Codex `last_followup`;
  `engram axi value --json` on the warmed runtime reported `operation_count=2`,
  `cache_hit_rate=1.0`, `p95_added_latency_ms=0.2162`, and zero budget misses.
  A dogfood replay over the real AXI trace since `2026-05-27T08:00:00Z` kept
  three Engram-project follow-up records, summarized `packet_count=6`,
  `result_count=5`, and fallback statuses `cache_satisfied=1` and `hit=1`.
  Focused AXI/dogfood replay tests passed with `70 passed`, ruff passed on the
  touched files, and skip-slow startup validation passed with Codex, Claude Code,
  and OpenClaw config checks green.
- Native storage diagnostics no longer launch disruptive Helix graph count scans
  on the live local path. Before the change, `/api/storage?live=true&timeoutSeconds=5`
  blocked for the full `5002-5015ms` count window and a concurrent capture burst
  still had a `captureStore=943ms` write while that scan was running. Live
  storage now returns cached/write-through counts plus fresh path sizes with
  `countsStatus=cached_native_live_skipped` and
  `countsRefreshSkippedReason=helix_native_counts_use_cached_write_through`;
  after reinstall/restart on LaunchAgent PID `42603`, two live storage probes
  returned in `2.1-2.3ms`, a concurrent capture burst stayed at
  `captureStore=15-118ms`, and `engramctl storage` completed in `0.052s`.
  MCP `observe` after the fix reported `capture_store=40ms`,
  `cue_store=48ms`, and `live_turn=56.3673ms`. Focused storage/capture/report
  tests passed with `47 passed`, storage/installer tests passed with `33 passed`,
  ruff passed, and skip-slow startup validation passed against the native PyO3
  LaunchAgent.
- Explicit recall now requires loaded-store backing before cached packets can
  satisfy the preflight gate. Project-file fallback packets can still rescue an
  empty/degraded recall, but they no longer mask fresh cued episodes by
  returning `cache_satisfied` before live search. Successful deep recall with
  low visible query overlap now runs the existing fast episode/cue fallback as a
  quality rescue. After reinstall/restart on LaunchAgent PID `46546`, `Post-fix
  validation note storage diagnostics skip count scans capture writes
  responsive` returned loaded-store `ep_890639326f32` first in
  `durationMs=332.6329`, and `User checked whether the Engram performance-goal
  run was stuck after a continuation` returned loaded-store `ep_4be71b058394`
  first in `durationMs=220.5227`; both reported `skipReason=null` and
  `fallbackStatus=hit`. Focused recall-surface tests passed with `31 passed`
  and ruff passed on the touched recall files.
- Explicit recall fast preflight now has its own bounded timeout. The rescue
  fallback still keeps the tighter `recall_fast_fallback_timeout_ms=100`, but
  the preflight gets `recall_fast_preflight_timeout_ms=250` so long compound
  MCP recall queries can return loaded cue hits instead of timing out and
  dropping to project-file fallback. After reinstall/restart on LaunchAgent PID
  `50407` and clearing packet cache, the same long Codex dogfood query returned
  five loaded-store results with `fallbackStatus=fast_preflight_hit` through
  REST in `durationMs=9.2032`, AXI in `durationMs=8.8308`, and live MCP in
  `duration_ms=14.5175`. Focused recall/cache/AXI tests passed with
  `119 passed`, ruff passed on the touched files, `git diff --check` passed,
  and skip-slow startup validation passed against the native PyO3 LaunchAgent.
- Topic-specific context cache misses now try loaded-store cue packets before
  project-file fallback. This closes the case where `get_context` stayed fast
  but bypassed the brain for human/status-style topics. After reinstall/restart
  on LaunchAgent PID `52393` and clearing packet cache, `did you get stuck
  Engram dogfood performance status broad human update` returned three
  `loaded_store_context` cue packets through REST context in `durationMs=23.6572`,
  live MCP context in `duration_ms=26.7368`, and cold AXI context with
  `packet_cache.scopes.loaded_store_context=3`; AXI recall on the same topic
  returned five results with `fallbackStatus=fast_preflight_hit` in
  `durationMs=19.0012`. Live value after the probes reported
  `p95_added_latency_ms=68.557`, `budget_miss_rate=0`, and
  `useful_packet_rate=0.8889`. Focused context/recall/cache/AXI tests passed
  with `142 passed`, ruff and `git diff --check` passed, and skip-slow startup
  validation passed against the native PyO3 LaunchAgent.
- Fast Helix BM25 fallback now compacts high-fanout operator queries before
  they hit native BM25. The measured bad term was `trace`: `AXI` and
  `startup matrix` were fast alone, but `trace`, `AXI trace`, and the full
  dogfood follow-up query could burn the preflight cap and fall back to
  project-file packets. The fast path now drops high-fanout terms such as
  `trace`, prefers specific terms such as `loaded store preflight bottleneck
  packet cache startup matrix`, and uses broad terms only when a query has
  nothing more specific. After reinstall/restart on LaunchAgent PID `53913`,
  the full `Engram native PyO3 dogfood performance loaded-store context
  preflight next bottleneck packet-cache hot behavior startup matrix AXI trace`
  query returned five loaded-store cue results through REST in
  `durationMs=30.2404` and live MCP in `duration_ms=7.2961`, both with
  `fallbackStatus=fast_preflight_hit`; `AXI trace` returned four cue results in
  `durationMs=21.5398`. Focused tests passed with `145 passed`, ruff and
  `git diff --check` passed, skip-slow startup validation passed, and the
  confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-025753` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime was healthy on
  LaunchAgent PID `54961`.
- Context fallback is now both faster and less prone to false loaded-store
  hits. A post-matrix probe showed a synthetic context miss could spend about
  `254ms` in loaded-store preflight before doing project-file fallback, and an
  unrelated loaded-store packet could pass relevance because generated
  `why_now` text repeated the query. Context preflight now has its own
  `context_fast_preflight_timeout_ms=100` default instead of sharing explicit
  recall's `250ms` budget; topic project-file fallback now scans at most 12
  ranked files and 8 KB per file; and context packet relevance ignores
  generated `why_now` fields. After reinstalling the local `server/` package
  into the uv tool and restarting on LaunchAgent PID `59395`, a synthetic miss
  for `xqzvplm brontide nonesuch cymophane vellichor 20260527` returned five
  `project_file_fallback` packets in `durationMs=8.8183`, with
  `cacheRelevanceMiss=1.9336` and `projectFileFallback=6.8847`, instead of a
  false loaded-store hit. A relevant loaded-store context query still returned
  three packets in `durationMs=27.7934`. Fresh AXI value after the clean probes
  reported `p95_added_latency_ms=64.2905`, `budget_miss_rate=0`, and
  `cache_hit_rate=0.5`. Focused tests passed with `81 passed, 21 skipped`;
  ruff passed on touched backend/test files; and skip-slow startup validation
  passed against the native PyO3 LaunchAgent. The confirmed lifecycle matrix
  then produced `/private/tmp/engram-dogfood-startup-20260527-032321` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 MCP observe/context hot-path pass: MCP observe now runs live-turn
  and recall-middleware side effects concurrently under the existing 75ms cap,
  native startup warms capture and cue write routes, and MCP observe seeds a
  non-persistent `session_recent` packet so immediate follow-up context can use
  the just-captured turn without waiting for graph projection. After
  reinstall/restart on LaunchAgent PID `65651`, startup logged native warmup
  timings of `capture_store_warmup=11ms`, `cue_store_warmup=29ms`, and
  `capture_store_warmup_cleanup=98ms`. The first live MCP observe still spiked
  to `0.5068s` wall with `capture_store=303ms`, `cue_store=77ms`, and
  `cue_index_outbox_enqueue=42ms`, so first-write latency remains a real
  follow-up. Steady MCP observe samples returned in `0.1726s` and `0.1406s`.
  Immediate MCP `get_context` for the same probe hit `session_recent` with
  `duration_ms=0.0359` and `packet_cache.scopes.session_recent=1`, avoiding the
  slower project-file fallback. Live value showed `cache_hit_rate=1.0`,
  `budget_miss_rate=0`, and p95 still dominated by the first observe spike.
  Focused tests passed with `115 passed, 21 skipped`; ruff and
  `git diff --check` passed; skip-slow startup validation passed with
  `12 pass, 2 skip`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-035942` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- 2026-05-27 REST hook parity follow-up: successful `api_auto_observe` calls now
  seed the same non-persistent `session_recent` packet cache as MCP observe, and
  topic-specific context responses filter cached packets before rendering so a
  fresh recent hook packet does not drag unrelated project-file fallback packets
  into a narrow follow-up. After reinstall/restart on LaunchAgent PID `69900`,
  a REST auto-observe hook probe returned in `0.05s` wall with
  `captureStore=10ms`, `cueStore=34ms`, and `cueIndexOutboxEnqueue=1ms`.
  Immediate `engram axi context` for the unique hook phrase returned exactly
  one packet from `packet_cache.scopes.session_recent=1`, with trust source
  `api_auto_observe`, in about `0.30s` CLI wall. Fresh `engram axi value`
  reported `p95_added_latency_ms=45.1793`, `cache_hit_rate=1.0`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `138 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`.
- 2026-05-27 broad context enrichment follow-up: topic-specific context no
  longer lets a lone `session_recent` packet short-circuit the bounded
  loaded-store preflight when the runtime can enrich the response. Narrow
  unique hook/session topics still render only the fresh recent packet, but
  broader loaded-store topics now pull older matching cue packets into context
  before returning. After reinstall/restart on PID `71606`, the live broad
  probe `"loaded-store recall performance packet cache broad context topaz
  older matching cue packets"` returned three `loaded_store_context` cue
  packets through AXI context in about `0.32s` wall, and the follow-up AXI
  recall returned `cache_satisfied` in `durationMs=1.1824`. MCP context then
  returned the fresh `session_recent` packet plus three cached project/cue
  packets in `duration_ms=0.0517`, and MCP recall returned
  `cache_satisfied` in `duration_ms=1.319`. Fresh value reported
  `p95_added_latency_ms=113.3016`, `cache_hit_rate=0.6`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `140 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`; and the confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260527-042226` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A same-day project-file masking follow-up tightened that enrichment rule for
  durable `project_home` cache hits too. Context no longer treats cached
  packets sourced only from `project_file`, `mcp_observe`, or
  `api_auto_observe` as enough to skip loaded-store preflight when the caller
  asked for broad project memory. After reinstall/restart on PID `73970`, the
  broad AXI context query
  `"Engram native PyO3 dogfood performance loaded-store recall context packet
  cache Codex evidence next bottleneck"` returned three `loaded_store_context`
  cue packets in about `0.34s` wall; follow-up AXI recall was
  `cache_satisfied` with `durationMs=0.1675`. MCP context for the same query
  returned loaded cue packets plus cached project-file packets in
  `duration_ms=0.0521`, which is acceptable because loaded-store provenance was
  present instead of masked. Fresh AXI value reported
  `p95_added_latency_ms=20.9004`, `cache_hit_rate=0.6667`, and
  `budget_miss_rate=0`. Focused backend tests passed with
  `140 passed, 24 skipped`; ruff and `git diff --check` passed; skip-slow
  startup validation passed with `12 pass, 2 skip`.
- The next AXI tracing polish pass made `engram axi context --json` preserve
  the server's `budget`, `lifecycle`, and `diagnostics` fields instead of
  hiding context timing behind a generic `status=ok`. This keeps the live
  performance loop operable from the agent-facing CLI without switching to raw
  REST/MCP output. After reinstall/restart on PID `75584`, cold AXI context for
  `"what should we work on next to make Engram faster for Codex without losing
  useful memory"` returned three `loaded_store_context` cue packets with
  `budget.durationMs=12.0913`, `budget.budgetMiss=false`, and
  `diagnostics.stageTimingsMs.loadedStoreContextPreflight=10.9676`; follow-up
  AXI recall was `cache_satisfied` with `durationMs=0.6887`. Fresh AXI value
  reported `p95_added_latency_ms=12.0913`, `cache_hit_rate=0.6667`, and
  `budget_miss_rate=0`. AXI presenter tests passed with `40 passed`; ruff and
  `git diff --check` passed; skip-slow startup validation passed with
  `12 pass, 2 skip`; and the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-044127` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- A follow-up MCP adoption pass closed the project-path gap exposed by live
  Codex recall. The MCP `recall` tool already accepted `project_path`, but the
  system prompt still taught the generic `recall(query)` pattern, so project
  harnesses could miss the fast project-scoped cue path and fall back to local
  project files. The prompt now tells agents to call
  `recall(query, project_path=...)` when a project path is available and to
  carry the same `project_path` used for `get_context`, `route_question`, or
  `search_artifacts`. The installed uv-tool prompt was verified after
  reinstall/restart on PID `78269`. With packet cache clear, AXI recall for
  `"Engram native PyO3 dogfood performance current state next bottleneck AXI
  context diagnostics real Codex sessions packet cache budget misses"` returned
  five loaded-store cue results in `durationMs=22.2766`; AXI context returned
  three `loaded_store_context` cue packets in `durationMs=19.7424` with
  `loadedStoreContextPreflight=16.9041`; and the live MCP recall in this
  existing Codex session reused warmed cue packets with
  `duration_ms=0.1982`, `cache_satisfied`. Focused MCP prompt/tool/recall tests
  passed with `68 passed, 2 skipped`; ruff and `git diff --check` passed; and
  skip-slow startup validation passed with `12 pass, 2 skip`. The current Codex
  session still exposes the older MCP schema until the client reloads tools, so
  the new `project_path` recall argument must be validated in the next fresh
  harness session.
- The next live-session recall pass made explicit recall consume
  `session_recent` packets, matching the context fast path. Before this change,
  MCP `observe` wrote a fresh non-persistent session packet that `get_context`
  could use immediately, but explicit `recall` only checked `identity_core` and
  `project_home` context packets before deeper search. After reinstall/restart
  on PID `79521` and clearing packet cache, a live MCP `observe` for
  `20260527-amber` stored `ep_d1004c18fb42` with `capture_store=268ms`,
  `cue_store=69ms`, and `recall_middleware=52.5167ms`; a no-`project_path` MCP
  recall for the same phrase returned the `recent_observation` packet from
  `_cache_scope=session_recent`, `fallback_status=cache_satisfied`, and
  `duration_ms=1.3854`, with no project-file fallback. Focused recall/MCP tests
  passed with `69 passed, 2 skipped`; ruff and `git diff --check` passed;
  skip-slow startup validation passed with `12 pass, 2 skip`. Fresh AXI value
  reported no budget misses and `cache_hit_rate=0.6667`, with p95
  `459.2958ms` dominated by the deliberate live observe sample.
- A rolling-session follow-up fixed the next live trace gap: `session_recent`
  used one untargeted packet-cache key, so each observe replaced the previous
  recent packet. The observe surfaces now preserve a rolling five-packet
  non-persistent session cache under that key, newest first, and strip
  `_cache_scope` before re-caching. After reinstall/restart on PID `81126` and
  packet-cache clear, two live MCP observes stored `20260527-orchid`
  (`capture_store=11ms`, `cue_store=31ms`) and `20260527-lapis`
  (`capture_store=89ms`, `cue_store=51ms`). No-`project_path` MCP recall for
  the older `orchid` phrase returned both recent packets from
  `_cache_scope=session_recent`, with the orchid packet ranked first,
  `cache_satisfied`, and `duration_ms=0.9169`; recall for `lapis` returned the
  same two packets with lapis first in `duration_ms=1.4209`. A traced AXI
  follow-up for `orchid` recorded Codex hook evidence with `cacheHit=true`,
  `fallbackStatus=cache_satisfied`, `packetCount=2`, and `duration_ms=8`, so
  `engram axi doctor --hooks codex --require-hook-run --require-followup`
  passed with current fixed behavior instead of the stale project-file fallback
  trace. Focused capture/recall/context/packet-cache tests passed with
  `97 passed`; ruff and `git diff --check` passed; skip-slow startup validation
  passed with `12 pass, 2 skip`; fresh AXI value reported
  `budget_miss_rate=0`, `cache_hit_rate=0.8571`, and p95 `207.9191ms`.
- AXI hook status now makes the real Codex before/after trace inspectable
  without manual JSONL tails. `engram axi hooks status codex --json` and
  `engram axi doctor --hooks codex --require-hook-run --require-followup --json`
  include `followup_summary` for recent context/recall follow-ups: operation and
  status counts, duration avg/p95/max, cache-hit rate, packet/result totals,
  fallback counts, degraded/timeout counts, and the five newest compact records.
  Live Codex evidence after reinstall showed latest recall
  `cacheHit=true`, `fallbackStatus=cache_satisfied`, `packetCount=2`, and
  `duration_ms=8`; the same summary preserved earlier fallback records at
  `509ms` and `512ms` plus older errors, making the improvement auditable from
  the supported CLI surface. Focused AXI hook/CLI tests passed with `35 passed`;
  ruff passed on the touched AXI files.
- The hook summary now also reports `latest_healthy_streak`, a newest-first
  contiguous slice of non-degraded follow-up records. After reinstalling the
  local CLI again and writing a fresh traced context/recall pair, `engram axi
  hooks status codex --json` showed the newest traced context at `31ms`,
  previous traced recall at `31ms`, and the older fixed recall at `8ms`; the
  streak covered 16 ok records with `degraded_count=0`, `timeout_count=0`, and
  `cache_hit_rate=0.375`, while the all-history summary still preserved older
  failures. Direct AXI context for the Engram project returned from
  `session_recent` packet cache in `durationMs=0.056`, and direct AXI recall
  returned `cache_satisfied` in `durationMs=0.7595`. Focused hook/CLI tests
  passed with `36 passed`.
- The live MCP surface now shows the same bounded behavior. A first Codex
  `get_context` for the active dogfood topic missed the existing packet cache
  but returned five useful project-file packets in `duration_ms=500.8099`
  (`project_file_fallback=355.7379`, no timeout or degradation). MCP `recall`
  for the same topic used fast preflight, returned five cue results plus three
  packets, and finished with `query_time_ms=55.8`, `budget.duration_ms=3.9472`,
  and `fallback_status=fast_preflight_hit`. Repeating `get_context` then
  produced loaded-store cue packets in `duration_ms=31.8175`, and the next
  repeated call hit packet cache directly with six project/cue packets in
  `duration_ms=0.0698`.
- Startup validation now asserts the exact MCP schema and call path that was
  stale in this live Codex session: the live MCP catalog must expose
  `recall.project_path`, and a read-only `recall(project_path=...)` probe must
  return without budget miss, degradation, or timeout. The strengthened full
  validator passed with `14 pass, 0 warn, 0 fail, 0 skip`; its MCP evidence
  showed 27 tools, `remember=true`, `recall_has_project_path=true`, and
  `recall_probe` `status=ok`, `fallback_status=cache_satisfied`,
  `query_time_ms=48.6`. A confirmed lifecycle startup matrix then passed at
  `/private/tmp/engram-dogfood-startup-20260527-052815` with
  `13 pass, 0 warn, 0 fail, 0 skip`, covering warmed checks, stopped-state
  detection, restart, post-restart warmed checks, and stale-PID remediation.
  Both warmed validations inside the matrix passed `14/14` and proved
  `recall_has_project_path=true`; their read-only recall probes returned useful
  project-file fallback packets in `query_time_ms=654.5` before restart and
  `591.6` after restart, with no budget miss, degradation, or timeout. After
  the matrix, `engramctl status` reported the restored LaunchAgent runtime
  healthy on PID `86144`.
- Dogfood transcript discovery now supports `engram dogfood scan --sort recent`
  so resumed Codex sessions surface by modification time instead of being hidden
  behind old high-turn transcripts. After reinstall, recent scan placed the
  active Codex transcript
  `/Users/konnermoshier/.codex/sessions/2026/05/11/rollout-2026-05-11T11-10-31-019e183b-c702-7711-8e6c-844739b05658.jsonl`
  first with 80 labelable turns. A redacted review bundle was prepared at
  `/private/tmp/engram-dogfood-review-20260527-active-codex` using AXI trace
  rows since `2026-05-27T12:00:00Z` for `/Users/konnermoshier/Engram`.
  The replay measured 80 user turns and five Codex follow-up trace rows
  (`context=2`, `recall=3`), all `ok`, with no degradation/timeouts,
  average `119.8ms`, p95/max `509ms`, four cache hits, six packets, and
  fallback counts `cache_satisfied=2`, `context_packet_fallback=1`. The bundle
  was then reviewed with 2 recall labels and 1 session label, leaving 78
  skipped turns. The session label matches the operator-approved verdict:
  baseline `1.0`, memory `1.0`, open loop expected/recovered, and no measurable
  Engram lift for that reviewed sample.
- AXI value reporting now keeps read-path and write-path latency separate instead
  of flattening all memory operations into one ambiguous p95. A live
  `engram axi value --json --full` run after reinstall still showed aggregate
  `p95_added_latency_ms=8200.8437`, but the mode breakdown attributed that spike
  to `api_auto_observe`. The read path covered nine context/recall/packet
  operations with max mode p95 `279.4478ms`, `cache_hit_rate=0.5`, and no budget
  misses; the write path carried the `8200.8437ms` observe p95 with four skipped
  dedup rows. This keeps agent startup/follow-up cost visible without hiding
  capture cost.
- Fast runtime packets now include startup-safe packet-cache summary data, so AXI
  home no longer reports `packet_cache.status=cold` after context/recall have
  warmed usable packets. After reinstall/restart, `/api/knowledge/runtime/fast`
  reported the persistent packet-cache path, AXI context warmed three relevant
  loaded-store cue packets in `24.1565ms`, AXI home then reported two fresh
  cache entries and `packet_cache.status=warm`, AXI recall was
  `cache_satisfied` in `0.5949ms`, MCP get_context hit packet cache in
  `0.0649ms`, and MCP recall returned three packets in `query_time_ms=2.0`
  with no degradation or budget miss. A clean post-restart AXI value window
  showed read-path p95 `24.1565ms`, `cache_hit_rate=0.875`, and zero
  degraded/timeouts/budget misses.
- `engram doctor` lifecycle and smoke phases are now bounded separately. The
  loaded 4G native store can still exceed the lifecycle snapshot cap, but doctor
  now returns `status=warn` instead of hanging; REST/MCP checks and disposable
  Helix smoke pass. The startup matrix now preserves doctor warning status from
  preambled JSON instead of treating a zero exit code as a clean pass. The latest
  confirmed matrix at `/private/tmp/engram-dogfood-startup-20260527-061148`
  completed with `11 pass, 2 warn, 0 fail, 0 skip`; both warnings are the bounded
  `lifecycle snapshot timed out after 10s` doctor checks, while both warmed
  validations still passed `14 pass, 0 warn, 0 fail, 0 skip` and the runtime was
  restored healthy on LaunchAgent PID `94726`.
- Installed `engramctl doctor` now skips the loaded-store lifecycle snapshot by
  default for native Helix readiness, while lower-level `engram doctor` keeps the
  bounded lifecycle phase for explicit deep inspection. Live
  `engramctl doctor --format json` returned `status=pass` with
  `lifecycle_snapshot=skipped` and REST, MCP, and disposable Helix smoke passing.
  The confirmed startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-062433/matrix-report.md`
  completed with `13 pass, 0 warn, 0 fail, 0 skip`, and the runtime was restored
  healthy afterward on LaunchAgent PID `97708`.
- Dogfood closeout now completes for the active Codex review bundle at
  `/private/tmp/engram-dogfood-review-20260527-active-codex`. The exported
  human-label artifact is
  `/private/tmp/engram-dogfood-review-20260527-active-codex/human-labels.json`
  with SHA-256
  `68fa77851b5a8b6bf20e3946955ccbef87fbb9f90df3781b7214912ef0d09ade`.
  `engram dogfood finalize` is idempotent after replay, returning
  `status=finalized`, `ready=true`, imported labels, exported evidence, closeout
  `ready_for_native_memory_value`, and native evaluation
  `memory_value.status=measured`. The measured memory-value window has
  `recall_sample_count=7`, `session_sample_count=2`, useful packet rate `0.9`,
  memory-need precision/recall `1.0`, false recall rate `0.1`, average added
  latency `119.8ms`, p95 added latency `509ms`, cache hit rate `0.8`, and zero
  budget misses, degraded operations, or timeouts. This is valid small-sample
  evidence, not a broad multi-session lift claim.
- A live relevance hardening pass found the next real Codex bottleneck after
  startup was no longer initialization but low-quality cache/fallback behavior.
  The query `dogfood finalize idempotent INSERT OR REPLACE graph_stats_timeout
  human label artifact` first hit an unrelated `session_recent` packet, and
  explicit AXI recall degraded in `1827.6422ms` while still returning that weak
  packet. The fix keeps `mcp_observe`/`api_auto_observe` packets from counting
  as loaded-store enrichment just because they have episode provenance, ignores
  weak one-token session-recent matches before project-file rescue, biases
  explicit recall and fast preflight toward the supplied project path, filters
  loaded-store context preflight to project hits when available, and lets MCP
  `recall()` inherit the last `get_context(project_path=...)` path for clients
  whose live schema cannot yet pass `project_path`. After reinstall/restart, AXI
  recall for the same query returned two Engram hits in `19.4049ms` with no
  `MachineShopScheduler` noise. In the current Codex MCP session, cold
  `get_context(project_path=...)` stayed under budget with project-file packets,
  the repeat context path used loaded-store preflight in `10.4572ms`, and MCP
  `recall()` without a callable `project_path` argument inherited the session
  path and returned two Engram hits in `query_time_ms=34.0` with no budget miss,
  degradation, or timeout. `engram axi value --json --full` then reported
  read-path p95 `238.3133ms`, cache hit rate `1.0`, and zero read-path
  degraded operations, timeouts, or budget misses. The startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-074345/matrix-report.md`
  completed with `13 pass, 0 warn, 0 fail, 0 skip`.
- The dogfood candidate scanner now handles Codex sessions launched from a
  parent directory by inspecting redacted tool-call `workdir` metadata in
  addition to `session_cwd`. Before this, `engram dogfood scan --project
  /Users/konnermoshier/Engram --project-only` missed the active Engram session
  because its transcript-level cwd was `/Users/konnermoshier`. After the fix,
  the same project-scoped scan found 20 Engram candidates and selected the
  active transcript with `project_match_source=tool_workdir`, 80 labelable user
  turns, and `session_cwd=/Users/konnermoshier`. A fresh private bundle at
  `/private/tmp/engram-dogfood-review-20260527-project-workdir` contains those
  80 labelable turns plus 10 project-filtered AXI trace records since
  `2026-05-27T08:00:00Z`; review status is correctly `needs_labels` until real
  human/agent labels are filled.
- The same bundle now has a small real review pass instead of placeholder
  labels: 6 reviewed recall turns, 1 session-continuity label, selected modes
  split between `cached=3` and `off=3`, and exported human-label evidence at
  `/private/tmp/engram-dogfood-review-20260527-project-workdir/human-labels.json`.
  Finalize imported the labels and the measured AXI trace costs into the local
  evaluation store. `engram evaluate --format json --require-memory-value`
  passed with memory value measured: 13 recall samples, 3 session samples,
  useful packet rate `0.9231`, false recall rate `0.0769`, open-loop recovery
  `1.0`, session-continuity lift `0.0333`, p95 added latency `1082.8418ms`,
  cache hit rate `0.6364`, and budget miss rate `0.0`. The stricter
  `--require-evaluation-signals` gate still reports
  `cue_usefulness:needs_data` and `projection_yield:needs_data`, so this is
  stronger dogfood value/cost evidence, not a completed release audit.
- Post-import live probes stayed bounded and useful. AXI context for
  `native PyO3 dogfood project workdir labels evidence` returned loaded-store
  context packets in `78.9281ms`; AXI recall returned three packets via
  `fast_preflight_hit` in `58.7302ms`; MCP `get_context(project_path=...)`
  hit packet cache in `0.0757ms`; MCP `recall()` hit cache in `0.104ms`. The
  current remaining performance hotspot in aggregate value metrics is no longer
  startup; it is cold `mcp_context` p95 around `1082.8418ms`, plus historical
  medium-mode degraded recall traces that should continue shrinking as packet
  warming improves.
- Follow-up live evaluation on the 4.0G native dogfood store exposed two
  startup/reporting hot spots and one remaining aggregate limitation. Native
  capture warmup once blocked startup for about 91.5s in cue storage, so startup
  now bounds that warmup at `capture_startup_warmup_timeout_ms=2000` and lets the
  create/delete probe continue best-effort in the background. The live restart
  verified the bounded path: the server logged `Native capture warmup exceeded
  2.0s; continuing startup` and returned healthy; a later restart with fast
  warmup completed normally in `27ms/43ms/115ms`.
- Live brain-loop reports now keep expensive evaluation reads off the request
  path. Cold `/api/evaluation/brain-loop/report?liveCost=false` returns within
  the 2s graph-state budget and warms graph stats in the background; warm
  follow-up returned in about `0.52s` with `1381` episodes, `1288` cues, `901`
  entities, `8109` relationships, `849` projected episodes, and projection
  yield measured at `1.6243` with no degradation entries. Consolidation context
  gets the same background-cache treatment so calibration and consolidation
  evidence do not disappear after a cold slow read.
- Direct `engram evaluate --require-evaluation-signals --require-memory-value`
  still opens its own native runtime and can report `cue_usefulness:needs_data`
  and `projection_yield:needs_data` when its independent 2s stats read times
  out. The live server report is the accurate loaded-runtime view for this
  dogfood state; a follow-up should add a server-backed evaluation CLI mode or
  materialized Helix aggregate stats before using the direct CLI as the release
  gate on large native stores.
- Follow-up cold fallback inspection found and closed one more empty-payload
  edge. After a restart, empty-cache AXI recall for the active Engram query first
  returned bounded `status=ok` in `589.3109ms` but with zero packets because the
  project-file fallback needed more than the fixed 100ms wait. Successful empty
  recall now waits against the remaining recall wall budget, capped at 1.25s.
  After reinstall/restart on PID `51738`, the same Engram query hit loaded-store
  preflight instead (`status=ok`, `result_count=5`, `packet_count=3`,
  `durationMs=94.2722`). A forced empty-cache temp-project probe returned a
  packet-bearing project-file fallback (`packet_count=1`,
  `fallbackStatus=project_file_recall_fallback`, `durationMs=963.7207`) instead
  of an empty payload. The temp probe cache was then cleared and re-warmed with
  the real Engram project: AXI context rebuilt five project-file packets in
  `716.9267ms`, and follow-up AXI recall returned three Engram project packets
  via `project_file_recall_fallback` in `514.82ms`. Focused tests cover the cold
  empty-success branch.
- Post-restart warm report proof still matches the intended cache-backed
  evaluation behavior. The first cold report returned quickly with
  `graph_state_timeout` while warming graph stats in the background; the warm
  follow-up returned measured totals with `1401` episodes, `901` entities,
  `8109` relationships, projection yield `1.6091`, cue usefulness
  `needs_feedback`, and no degradation payload.
- The next live pass added a server-backed evaluation CLI path for large native
  stores. `engram evaluate --server-url http://127.0.0.1:8100` now reads the
  running REST report instead of opening a second native runtime. Against the
  dogfood service, `--require-memory-value` passed with `1405` episodes, `901`
  entities, `8109` relationships, projection yield `1.6054`, and measured memory
  value. The stricter `--require-evaluation-signals --require-memory-value` gate
  now fails only on real missing cue feedback (`cue_usefulness:needs_feedback`),
  not on graph stats or projection-yield timeouts. The startup matrix after this
  pass produced `/private/tmp/engram-dogfood-startup-20260527-101557` with
  `13 pass, 0 warn, 0 fail, 0 skip`.
- `engramctl storage` now prints the source of count data. On native Helix it
  reports `cached_native_live_skipped; helix_native_counts_use_cached_write_through`
  beside the startup/write-through counts, so the operator does not confuse those
  bounded counters with full historical graph totals while still seeing the 4.0G
  data path and growth since startup.
- A follow-up lifecycle pass closed the remaining stop-path failure: a previous
  matrix could leave an old `engram serve` listener on port `8100` while a newer
  LaunchAgent process failed to bind. `engramctl stop` now checks listener PIDs
  with `lsof`, terminates only command lines that look like Engram's local
  `serve` process, and refuses non-Engram listeners. Manual proof removed the
  stale listener, left the API offline, then restarted with exactly one
  LaunchAgent-owned listener. The confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-110425/matrix-report.md` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Its live MCP catalog check exposed
  `remember`, warmed context with 5 packets and no degradation, and validated
  recall with 3 fallback packets, `fallback_status=context_packet_fallback`, and
  `query_time_ms=497.2`.
- The next live performance pass removed the remaining cold server-backed
  evaluation stats timeout on the loaded 4.0G native dogfood store. After a
  restart, `engram evaluate --server-url http://127.0.0.1:8100
  --require-memory-value` still surfaced a `graph_state_timeout` fallback with
  zero graph totals, even though memory-value metrics were preserved. The fix
  adds bulk Helix stats routes for cues and projected episode/entity links
  (`find_cues_by_group`, `find_cues_all`,
  `get_projected_episode_entities_by_group`,
  `get_projected_episode_entities_all`) and keeps Python fallback compatibility
  for older native route maps. The bundled PyO3 query source now rebuilds with
  180 routes, and the live LaunchAgent restart on PID `3324` logged
  `routes=180`. Post-restart, server-backed evaluation returned measured totals
  with `1414` episodes, `901` entities, `8109` relationships, Capture `ready`,
  Cue `attention`, and Project `active` instead of `needs_data`; AXI context
  returned three loaded-store packets in `37.7618ms`, AXI recall returned five
  results with three packets in `48.2276ms`, MCP `get_context` returned three
  loaded-store packets in `82.0965ms`, and MCP `recall` hit cache in `0.7431ms`.
  The post-fix startup matrix at
  `/private/tmp/engram-dogfood-startup-20260527-112804` passed with
  `13 pass, 0 warn, 0 fail, 0 skip`.
  A final report-service guard now marks runtime-only graph stats as
  `graph_state_unavailable` instead of letting a cold fallback look like a real
  empty brain. After the final restart on PID `14146`, server-backed evaluation
  immediately returned `1416` episodes, `901` entities, `8109` relationships,
  and no report degradation. AXI context returned five project-file fallback
  packets in `367.6213ms`; AXI recall still timed out its search budget but
  returned three fallback packets instead of an empty payload; MCP context
  returned loaded-store packets in `22.1129ms` and MCP recall hit cache in
  `1.4698ms`.
- A follow-up AXI latency pass made repeated project-file fallback recall
  cache-satisfied instead of repeatedly paying the deep loaded-store timeout.
  The first trace showed the hard query still spending `250ms` in fast preflight
  and `651ms` in deep search before project-file packets rescued the response.
  The fix exposes those diagnostics through `engram axi recall --json`, races
  record-backed cue/episode fast searches so one stalled peer cannot consume the
  whole preflight budget, and lets packets explicitly marked
  `trust.source=project_file` satisfy the next matching explicit recall from
  cache. After reinstall/restart on PID `31237`, the same AXI query returned
  `status=ok`, `skipReason=cache_satisfied`, three project-file packets, and
  `durationMs=43.9192`; another cached project-file recall returned in
  `durationMs=1.2747`. Fresh live-cost value reported read-path p95
  `43.9192ms`, `cache_hit_rate=1.0`, and zero timeouts, degraded operations, or
  budget misses. The remaining live report degradation is
  `evaluation_context_timeout` / `live_cost_runtime_only`, not repeat AXI
  recall.
- The next report-startup pass removed that live report degradation on the
  loaded native dogfood store. Live-cost reports now skip consolidation context
  scans, graph-stats warmup starts in the native REST startup background lane,
  stale graph/consolidation warmup tasks are replaced after 30 seconds, and
  PyO3 Helix exposes direct count routes for entity, episode, relationship, and
  cue totals. After reinstalling both `engram` and `helix-native`, startup logs
  showed `routes=184`; direct native count routes returned `901` entities,
  `1427` episodes, `3963` relationships, and `1332` cues. Running-server normal
  and `liveCost=true` report calls now return populated totals with no
  degradation. AXI recall for the active dogfood status returned
  `durationMs=54.7065`, `budgetMiss=false`, and `skipReason=cache_satisfied`;
  AXI context rebuilt topic-specific project packets in `durationMs=534.6486`.
  The quick startup validator passed with `12 pass, 0 warn, 0 fail, 2 skip`
  (`engramctl doctor` and live MCP catalog intentionally skipped).
- A repeat-cache follow-up fixed topic-specific project-file context rebuilds.
  Before this patch, a matching second context call still paid loaded-store
  preflight plus file fallback because cached project-file packets were treated
  as needing enrichment. Exact topic/project fallback packets now satisfy repeat
  context calls from packet cache, while generic stable project packets still
  fall through conservatively. Live proof after reinstall: first AXI context for
  `repeat project-file fallback cache live 20260527 exact marker residual latency`
  built five project-file packets in `1142.4237ms`; the second identical AXI
  context call returned from cache in `0.1268ms`; MCP `get_context` hit cache in
  `0.1611ms`; MCP `recall` returned three cached packets in `2.1714ms`; and the
  quick startup validator still passed with `12 pass, 0 warn, 0 fail, 2 skip`.
- A live MCP write-path follow-up found the next real agent-turn blocker:
  `observe` succeeded but spent `36503ms` in raw `capture_store`. Raw episode
  persistence is now bounded by `capture_store_timeout_ms` and deferred in the
  background on timeout, preserving the immediate session packet while letting
  durable storage, event publication, cue storage, and projection scheduling
  complete when Helix catches up. After reinstall/restart, REST observe returned
  with `captureStore=122ms` then `8ms`, and real MCP `observe` returned in about
  `1.28s` wall with `capture_store=415ms`, `cue_store=123ms`, and only bounded
  MCP side-effect timeouts around `103ms`.
- The final matrix follow-up fixed a doctor smoke false negative introduced by
  fast native stats. The disposable Helix smoke projected episodes, but native
  `get_stats()` used the count-only fast packet and reported empty projection
  metrics, so doctor failed with
  `projection yield cannot be measured until episodes are projected`. Exact
  `get_stats()` is now the default again; recall uses `exact=False` for the cheap
  pool-sizing count path; and deterministic smoke forces synchronous raw/cue
  capture instead of live capture deferral. Installed `engram evaluate --smoke
  --mode helix --format json` returned no coverage gaps with `3` projected
  episodes, `3` linked episode entities, and one consolidation cycle. Live
  `engramctl doctor --format json` passed, and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260527-130309` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix `engramctl status` reported the
  native LaunchAgent healthy on PID `19164`, repeat AXI context hit cache in
  `0.1597ms`, MCP `get_context` hit in `0.0261ms`, and MCP `recall` hit in
  `3.184ms`.
- A follow-up live AXI CLI probe found a routing issue rather than a new Helix
  latency regression: global `--project` / `--topic` values were erased when a
  subcommand also declared the same destination and the global flags appeared
  before the subcommand. That made `engram axi --project ... --topic ... context`
  eligible to fall back to CWD inference instead of the intended topic-specific
  fast path. The duplicate subcommand options now suppress absent defaults so
  global values survive, while subcommand-local flags still override. After
  reinstalling the local CLI, fresh global-before-subcommand probes launched from
  `/tmp` returned useful packets without degradation: AXI context rebuilt five
  project-file packets in `562.4966ms`, AXI recall returned three project packets
  in `636.2156ms`, and AXI doctor passed for
  `/Users/konnermoshier/Engram`. The startup validator now warns when a hook
  trace records `project=/`, and the matrix now propagates validation warnings
  instead of flattening them to pass. With those stricter checks, the confirmed
  lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-134521` with
  `11 pass, 2 warn, 0 fail, 0 skip`; both warnings are the same Codex
  SessionStart root-project trace, while stop/start/stale-PID lifecycle checks
  still pass. After that restart, `engramctl status` reported LaunchAgent PID
  `59643`; AXI context from `/tmp` returned five project-file packets in
  `699.2828ms`, AXI recall returned three project packets in `1132.7942ms`, MCP
  `get_context` returned useful project-file packets in `129.0879ms`, and MCP
  `recall` hit cache in `3.2646ms`, all without degradation or budget misses.
  Managed Codex and Claude Code AXI hooks now use `engram axi hook-run`, which
  reads the hook stdin JSON `cwd` instead of shell `$PWD`; both local hook configs
  were rewritten with that command after reinstall. Manual `hook-run` smoke with
  stdin `{"cwd":"/Users/konnermoshier/Engram"}` returned a healthy packet with
  `brain.project=/Users/konnermoshier/Engram`. The validator warning should clear
  only after a fresh real Codex SessionStart trace replaces the older `project=/`
  row.
- The validator now also compares each installed hook config mtime against the
  latest SessionStart trace timestamp, so it warns when startup evidence predates
  the current hook command. Current skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`: Codex and Claude Code both need fresh
  SessionStart traces after the hook-run reinstall, and Codex still has the older
  root-project row. Fresh live read probes remain healthy: AXI context returned
  three loaded-store packets in `78.3556ms`, AXI recall was `cache_satisfied` in
  `1.5127ms`, MCP `get_context` returned five project-file packets in
  `123.034ms`, and MCP `recall` was `cache_satisfied` in `2.4123ms`.
  `engram axi doctor --hooks codex claude-code --require-hook-run
  --require-followup --json` now shares the same freshness rule and fails with
  `stale_session_start_run` for both clients instead of accepting pre-reinstall
  startup evidence. Codex's hook payload also reports `last_run_project_root=true`.
  Focused regression gates passed for AXI CLI/hooks and startup-warning coverage
  (`46 passed`), ruff passed on the touched AXI/validation files, and
  `git diff --check` is clean.
- The refreshed full lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-140202` with
  `11 pass, 2 warn, 0 fail, 0 skip`; `engramctl doctor` and the live MCP catalog
  passed, and both warnings are still the expected stale/root hook-run evidence.
  Post-matrix runtime stayed healthy on LaunchAgent PID `74501`. Fresh
  post-matrix probes stayed bounded without degradation: AXI context returned
  five project-file fallback packets in `1205.4917ms`, AXI recall returned three
  project-file packets in `1559.5697ms`, MCP `get_context` returned five
  project-file packets in `142.7577ms`, and MCP `recall` was `cache_satisfied`
  in `54.5339ms`.
- A follow-up fallback-quality pass fixed stale and cross-project project-file
  cache behavior found in live dogfood. Project-file fallback now uses a larger
  bounded topic scan, adjacent line-window scoring for wrapped evidence, capped
  historical term scoring, and `docs/CURRENT_HANDOFF.md` priority for current
  evidence. Explicit recall's context-packet fallback now filters cached
  project-file packets by `project_path`, so an Engram recall cannot be
  satisfied by MachineShopScheduler project-file packets. After reinstall/restart,
  AXI context for `startup matrix 20260527 tiecheck gold` returned the current
  `20260527-140202` `CURRENT_HANDOFF.md` packet in `537.2311ms`; repeat AXI
  context hit cache in `0.1081ms`; AXI recall was `cache_satisfied` in
  `18.1105ms`; MCP `get_context` hit the same packet in `0.0629ms`; and MCP
  `recall` was `cache_satisfied` in `24.3204ms`. Focused context/recall surface
  tests passed with `72 passed`, and ruff passed on the touched retrieval
  files/tests. The refreshed full lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-142608` with
  `11 pass, 2 warn, 0 fail, 0 skip`; after that restart, LaunchAgent PID `9661`
  was healthy, cold-ish AXI context returned the current handoff packet in
  `952.857ms`, and repeat AXI recall hit cache in `1.8494ms`.
- The follow-up cache-quality pass fixed stale fallback-row masking and noisy
  adjacent-line summaries. Project-file fallback only joins real continuation
  lines now, packets are tagged with project-file fallback `version=2`, exact
  context cache hits require the current version, and explicit recall drops old
  unversioned project-file fallback packets instead of letting them satisfy an
  Engram project query. After reinstall/restart on PID `21404`, AXI recall for
  `startup matrix 20260527 tiecheck diamond` rebuilt current Engram evidence in
  `1249.1747ms` with `project_file_recall_fallback`; AXI context for
  `native PyO3 dogfood performance continuation cleanline 20260527` rebuilt a
  clean current-handoff packet in `931.882ms`. Repeating the same surfaces then
  hit cache: AXI context `0.2474ms`, AXI recall `cache_satisfied` in
  `74.5458ms`, MCP `get_context` `0.1637ms`, and MCP `recall`
  `cache_satisfied` in `2.4684ms`. Focused context/recall tests passed with
  `75 passed`, ruff passed on touched retrieval files/tests, skip-slow
  validation reports `11 pass, 1 warn, 0 fail, 2 skip`, and the refreshed
  lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260527-144207`
  with `11 pass, 2 warn, 0 fail, 0 skip`. Post-matrix runtime is healthy on
  LaunchAgent PID `23794`; the only remaining warnings are stale/root
  SessionStart traces until fresh real Codex and Claude Code sessions run the
  installed `hook-run` command.
- A live packet-quality follow-up fixed the remaining mid-sentence and mid-word
  summary rough edges in project-file fallback packets. Lowercase wrapped
  evidence lines now join to the previous line when that previous line does not
  end a sentence, and reusable packet snippets truncate at word boundaries with
  `...` instead of cutting tokens. After reinstall/restart, fresh AXI context for
  `startup matrix 20260527 tiecheck diamond project_file_recall_fallback
  continuationproof2` rebuilt the current handoff packet in `785.6527ms`, with
  the summary starting at the `startup matrix ...` line and ending at
  `native PyO3...` instead of `native PyO3 dogfood p`. Repeat calls hit cache:
  AXI context `0.1051ms`, AXI recall `cache_satisfied` in `52.3059ms`, MCP
  `get_context` `0.0807ms`, and MCP `recall` `cache_satisfied` in `1.1909ms`.
  Focused context/recall surface tests passed with `77 passed`, ruff passed, and
  `git diff --check` is clean. The refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-145536` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `37314`, and the only remaining warnings are still stale/root
  SessionStart traces.
- The next continuation-window follow-up made evidence-line selection match the
  summary fix. Project-file matching now ignores lines with no direct query-term
  hit, walks a bounded chain of previous continuation lines, and trims unrelated
  prior sentences before using a wrapped previous line. After reinstall/restart,
  fresh AXI context for
  `evidence project_file_recall_fallback wrappedwindow liveproof 20260527
  chainfixed2` rebuilt the current handoff packet in `747.8033ms`; MCP
  `get_context` returned cached evidence lines starting with the full
  `startup matrix ...` line and `After reinstall...`, with no `evidence in...`
  or `can satisfy...` starts. Repeats hit cache: AXI context `0.179ms`, AXI
  recall `cache_satisfied` in `3.8688ms`, MCP `get_context` `1.5832ms`, and MCP
  `recall` `cache_satisfied` in `1.0518ms`. Focused context/recall surface
  tests passed with `80 passed`, ruff passed, and `git diff --check` is clean.
  The refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-151148` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `48671`, and the only remaining warnings are still stale/root
  SessionStart traces.
- The next latency follow-up removed avoidable SQLite sidecar syncs from hot
  context/recall cache reads and made exact project-file fallback packets usable
  for repeated context and recall even when the fallback summaries are generic.
  Context now prebuilds project-file fallback packets while loaded-store
  preflight runs. After reinstall/restart, synthetic miss query
  `xafnorb quexilate zumbrel frobnicate mintcase exactcache5` rebuilt AXI context
  in `618.2917ms` (`cacheRelevanceMiss=2.5779ms`,
  `projectFileFallback=564.7988ms`), then hit context cache in `0.047ms`; AXI
  recall for the same topic was `cache_satisfied` in `0.7253ms` and `0.585ms`.
  Focused context/recall tests passed with `81 passed`, ruff passed, and
  `git diff --check` is clean. Skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`; the refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-153357` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `57186`, and the only remaining warnings are still stale/root
  SessionStart traces. On that post-matrix runtime, the same synthetic topic
  rebuilt project-file context in `41.1342ms`, repeated from cache in
  `0.6055ms`, and AXI recall was `cache_satisfied` in `0.5183ms`.
- The next relevance/usefulness follow-up made context packet cache acceptance
  require a stronger match than a lone date/id token and made explicit recall
  ignore generated `why_now` text before deciding a cached packet satisfies the
  query. After reinstall/restart, weak synthetic query
  `qvanta noexisting loadedstore miss tail 20260527 probeB` no longer returned
  stale loaded-store dogfood packets; AXI context reported
  `cache_relevance_miss` and produced project-file fallback packets in
  `44.7505ms` (`projectFileFallback=42.4623ms`). AXI recall for the same topic
  was `cache_satisfied` in `0.6213ms` from exact project-file fallback cache. A
  fresh recall-first probe `qvanta noexisting loadedstore miss tail 20260527
  probeC` ran bounded recall in `228.2368ms`, found no memory results, and
  returned three project-file context packets with
  `fallbackStatus=context_packet_fallback`. Focused context/recall tests passed
  with `83 passed`, ruff passed, and `git diff --check` is clean. Live value
  after the probe set reports `0%` budget misses, `0%` degradation, `75%`
  read-path cache hit rate, and p95 `223.127ms` over five read-path samples.
  Skip-slow validation reports `11 pass, 1 warn, 0 fail, 2 skip`; the refreshed
  lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260527-154316`
  with `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `59654`, and the only remaining warnings are still stale/root
  SessionStart traces.
- The next write-path latency follow-up split MCP write side-effect budgets and
  made write-tool auto-recall cache-only. `observe`/`remember` still cache fresh
  session packets and can surface already-warm context, but a cache miss no
  longer runs a medium recall probe on the write response path. Before this
  pass, live value showed `mcp_observe` p95 `178.0555ms`,
  `api_auto_observe` p95 `314.8745ms`, and stale `medium` recall timeouts.
  After reinstall/restart on PID `63522`, live MCP observe probe `obsF` returned
  in `85.9ms` wall time with `capture_store=11ms`, `cue_store=39ms`,
  `live_turn_timeout=11.6503ms`, and `recall_middleware=0.4302ms`; live value
  showed write-path p95 `65.566ms` and cache-miss `medium` auto-recall skipped
  in `0.0775ms`. AXI context for
  `write auto recall cache-only short live-turn timeout obsF` returned
  loaded-store cue packets in `30.4921ms`, and AXI recall was `cache_satisfied`
  from the fresh `mcp_observe` recent packet in `0.3212ms`. Focused backend
  tests passed with `129 passed, 2 skipped`, ruff passed, and
  `git diff --check` is clean. Skip-slow validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`; the refreshed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260527-155526` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `64398`. A post-matrix MCP observe probe `obsG` kept
  `recall_middleware=0.3129ms` and `live_turn_timeout=12.5198ms`; live value
  after that probe showed read-path p95 `2.444ms`, `0%` read/write degradation,
  and write-path p95 still dominated by matrix `api_auto_observe` at
  `155.1062ms`.
- The next operator-friction follow-up separated startup proof from manual
  follow-up proof. Stale/root SessionStart warnings now tell the operator to
  start a new interactive Codex or Claude Code session from the target project,
  and explicitly state that manual `agent-followup` traces do not refresh
  SessionStart evidence. A nested `codex exec` run from `/Users/konnermoshier/Engram`
  proved real MCP adoption by calling Engram `get_context` and returning
  `ENGRAM_SESSIONSTART_PROBE`, but it did not emit a SessionStart hook row, so
  the validator correctly keeps the stale/root startup warning. Focused
  startup-validation tests passed for stale/root guidance, ruff passed, and
  `git diff --check` is clean. Current full startup validation remains
  `13 pass, 1 warn, 0 skip`; the remaining warning is still only stale/root real
  SessionStart evidence awaiting fresh interactive Codex and
  Claude Code sessions.
- A follow-up interactive Codex TUI probe from `/Users/konnermoshier/Engram`
  produced a real current SessionStart hook trace:
  `timestamp=2026-05-28T00:24:34.184814Z`, `operation=hook-run`,
  `origin=session-start-hook`, `project=/Users/konnermoshier/Engram`,
  `durationMs=11`, `status=healthy`. The validator now accepts current
  `hook-run` startup traces as well as legacy `home` traces, so Codex startup
  evidence is no longer stale/root. A Claude Code print-mode probe also ran the
  managed SessionStart hook before its prompt-argument error and wrote
  `timestamp=2026-05-28T00:28:48.891213Z`, `operation=hook-run`,
  `project=/Users/konnermoshier/Engram`, `durationMs=12`, `status=healthy`.
  Full startup validation now passes AXI hook/tracing evidence for both clients.
  The installed AXI home packet now uses the active trace client for capture
  suggestions (`--source claude-code`, `--source codex`, or generic
  `--source axi`) instead of hard-coding Codex. Focused AXI/startup-validation
  tests passed with `39 passed`, ruff passed, and live value reports read-path
  p95 `104.0759ms`, read cache hit rate `0.8545`, and zero read budget
  misses/degraded reads/timeouts.
- The resumed dogfood performance pass restarted the installed local runtime:
  `engramctl stop && engramctl start` returned a healthy native PyO3 LaunchAgent
  on PID `86463`. Full startup validation passed with all checks green. AXI home
  remained startup-safe; AXI context for
  `live dogfood loaded-store context performance restart 20260528` returned five
  project-file packets in `49.1589ms`; AXI recall for the same query returned
  loaded-store episode packets in `235.963ms` with
  `fallbackStatus=fast_preflight_hit`; MCP `get_context` hit cache in
  `0.0496ms`; MCP `recall` was cache-satisfied in `0.2021ms`. The lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260527-173419` with
  `13 pass, 0 warn, 0 fail, 0 skip`, leaving the runtime healthy on PID `87404`.
  Post-matrix hard probes did not produce empty timeout payloads: forced-miss
  AXI recall returned a relevant historical diagnostic episode in `19.1024ms`,
  fresh AXI context fallback returned useful packets in `16.7976ms`, and broad
  AXI recall hit cache in `0.4516ms`. Final live value reports read-path p95
  `83.526ms`, read cache hit rate `0.6667`, and zero budget misses/degraded
  reads/timeouts. The source checkpoint is now clean after commit/push at
  `e59be43`.
- A real Codex continuation sample on the clean `e59be43` checkpoint recorded
  session-continuity sample `esc_35ecade2bbf7` and recall-quality sample
  `ers_7aa46915657c`. Engram recovered the active dogfood performance goal and
  useful project context while shell probes confirmed the current runtime:
  native PyO3 LaunchAgent PID `87404`, clean source, and no current degraded
  recall/context path. `engram axi value --json` after the samples reports
  continuity lift `0.075`, useful packet rate `0.6429`, memory-need precision
  `0.9286`, read-path p95 `87.0364ms`, read cache hit rate `0.5714`, and zero
  read budget misses/degraded reads/timeouts. The broader goal remains active
  because this is one real Codex continuation sample, not a long-window
  completion audit.
- The same continuation lowered the agent-facing raw capture wait for MCP
  observe and REST auto-observe by passing a per-write
  `capture_store_timeout_ms=250`, while keeping the global explicit-write default
  at `1000ms`. After reinstall/restart on LaunchAgent PID `67368`, live MCP
  observe returned with `capture_store=169ms`, `cue_store_timeout=251ms`,
  `live_turn_timeout=13.097ms`, and `recall_middleware=1.3739ms`; live REST
  auto-observe deferred raw capture at `captureStoreTimeout=252ms`.
  `engram axi value --json` reported write-path p95 `440.135ms`, read-path p95
  `0.1838ms`, `0%` degradation, and `0%` budget misses. Focused
  capture/validation tests passed with `166 passed, 2 skipped`, ruff passed, and
  `git diff --check` is clean.
- The next startup-safe read warmup pass made `/api/knowledge/runtime/fast`
  schedule a non-blocking in-memory project-file prefix warmup for the supplied
  project path. This keeps AXI session-start read-only and graph-free, while
  avoiding the first real topic-specific context call paying cold local file
  prefix reads. After reinstall/restart on LaunchAgent PID `69492`, AXI home for
  `/Users/konnermoshier/Engram` triggered the warmup; fresh AXI context returned
  five useful project-file packets in `durationMs=183.2692` with
  `projectFileFallback=137.6598`, repeated in `0.0474ms`, and AXI recall was
  `cache_satisfied` in `0.7942ms`. Fresh MCP `get_context` returned useful
  project-file packets in `duration_ms=127.3182` with
  `project_file_fallback=29.3808`, and MCP recall was `cache_satisfied` in
  `1.1467ms`. Installed-user startup validation reports
  `11 pass, 1 warn, 0 fail, 2 skip`; the warning is still limited to stale/root
  SessionStart proof pending fresh Codex and Claude Code sessions. Focused
  runtime/context tests passed with `139 passed`, ruff passed, and
  `git diff --check` is clean.
- The next project-file executor isolation pass addressed the resumed Codex
  probe where MCP context reported `project_file_fallback=806.2496ms` while the
  same fallback builder profiled locally at about `25ms` after prefix warmup.
  Project-file context fallback, explicit-recall fallback, and runtime-fast
  prefix warmup now run on a small dedicated executor instead of the default
  executor shared with native Helix/storage/background work. After
  reinstall/restart on LaunchAgent PID `71087`, AXI home triggered warmup; fresh
  MCP `get_context` returned five useful project-file packets in `149.254ms`
  with `project_file_fallback=141.8561`, and MCP recall was `cache_satisfied` in
  `0.5977ms`. Fresh AXI context returned loaded-store cue packets in
  `77.4707ms`, and AXI recall was `cache_satisfied` in `0.5423ms`. Fresh value
  reported read-path p95 `149.6347ms`, cache hit rate `0.7857`, and zero budget
  misses/degraded operations/timeouts. Installed-user startup validation stayed
  at `11 pass, 1 warn, 0 fail, 2 skip`; the warning is still only stale/root
  SessionStart proof. Focused runtime/context tests passed with `182 passed`,
  ruff passed, and `git diff --check` is clean.
- The same pass then tightened AXI's project-context lane:
  `engram axi context --project ...` now skips loaded-store preflight with or
  without an explicit topic and uses cached/project-file context. This keeps AXI
  in the startup/home -> compact project packet lane while leaving explicit
  long-tail memory lookup to `engram axi recall` and MCP `get_context`. After
  reinstall/restart on LaunchAgent PID `74526`, packet-cache clear plus a cold
  topic-specific AXI context showed no loaded-store preflight
  (`cacheRelevanceMiss=0.3322ms`) but paid one cold prefix scan at
  `projectFileFallback=285.6067ms`; the next fresh topic returned in
  `57.0956ms` with `projectFileFallback=32.1867ms`, and exact repeat hit cache
  in `0.0461ms`. Startup validation reports `13 pass, 1 warn`; the only warning
  remains stale/root real SessionStart proof. The final focused suite passed
  with `184 passed`, ruff passed, and `git diff --check` is clean.
- The next fresh-context pass reduced duplicate work inside project-file packet
  building. Summary matching and evidence-claim extraction now reuse one
  `_project_file_matching_lines(...)` pass per candidate file. Local profiling
  dropped the Engram fallback builder from roughly `18-22ms` to `11-16ms`.
  After reinstall/restart on LaunchAgent PID `75796`, AXI home warmed the
  project; a fresh AXI context built project-file packets in `22.6326ms`, and
  exact repeat hit cache in `0.0393ms`. MCP `get_context` showed
  `project_file_fallback=24.8746ms`; total `duration_ms=175.392ms` is now
  mostly the intentional 100ms loaded-store preflight miss budget plus
  transport/presentation overhead. MCP recall hit cache in `1.0761ms`.
  `engram axi value --json` reported read-path p95 `175.694ms`, cache hit rate
  `0.625`, and zero budget misses/degraded operations/timeouts. Startup
  validation stayed at `13 pass, 1 warn`, with the same stale/root SessionStart
  warning. The focused runtime/context suite passed with `185 passed`; ruff and
  `git diff --check` passed.
- The next instrumentation pass made MCP context miss costs explicit. Successful
  loaded-store context responses now report separate search and packet-assembly
  timings, and project-file fallback responses include
  `loaded_store_context_preflight` when the loaded-store preflight missed. After
  reinstall/restart on LaunchAgent PID `76806`, a repeated useful
  goal-continuation context hit packet cache in `0.0772ms`; a fresh loaded-store
  miss returned useful project-file packets without degradation in `103.96ms`
  with `loaded_store_context_preflight=99.7659ms` and
  `project_file_fallback=23.6974ms`. AXI context for a comparable fresh project
  topic returned in `69.46ms` with `projectFileFallback=21.6849ms`. One earlier
  post-restart MCP context sample showed a transient cold project-file build at
  `1228.0362ms`, which remains visible in live value p95. Startup validation
  stayed at `13 pass, 1 warn`, with the same stale/root SessionStart warning.
  The focused runtime/context suite passed with `185 passed`; ruff and
  `git diff --check` passed. Recall evaluation sample `ers_5fba1e65db42`
  recorded useful real-Codex goal-continuation context.
- The next MCP context policy pass added a soft wait in front of loaded-store
  preflight misses. Quick loaded-store hits still win, but when project-file
  context is already ready, MCP can return useful project packets before the
  full `context_fast_preflight_timeout_ms=100` expires. The default
  `context_fast_preflight_soft_wait_ms` is `75ms`; late loaded-store work
  continues in the background and can still cache cue packets. After
  reinstall/restart on LaunchAgent PID `77735`, AXI home warmed the project. A
  useful goal-continuation MCP context returned loaded-store cue packets in
  `85.4594ms` with `loaded_store_context_search=52.4431ms`; a fresh miss
  `softwait miss qxjv norel zaffron plinket 20260527` returned useful
  project-file packets in `25.7279ms`, with
  `project_file_fallback=23.1409ms` and
  `loaded_store_context_preflight=17.2334ms`. MCP recall for the same miss hit
  cache in `0.8398ms`. Live value reported read-path p95 `85.4594ms`, cache hit
  rate `0.7143`, and zero budget misses/degraded reads/timeouts. Startup
  validation stayed at `13 pass, 1 warn`; the warning remains stale/root real
  SessionStart proof. Focused runtime/context tests passed with `186 passed`;
  ruff and `git diff --check` passed.
- The follow-up cache-rescue pass handles the case where a fresh MCP context
  miss reaches project-file fallback while the in-memory cache is cold but the
  persistent packet cache already has stable same-project packets. The initial
  strict cache lookup stays in-memory, but the rescue lookup may sync
  persistent packet cache if the project-file scan is still pending. Unit
  coverage verifies that cached same-project project-file packets can return
  immediately and that the slow fresh scan still refreshes the exact topic cache
  in the background. Before the final tweak, one post-reinstall fresh MCP miss
  spent `project_file_fallback=750.092ms`; after reinstall/restart on
  LaunchAgent PID `80961`, the first fresh live MCP miss
  `persistent rescue first post restart miss zibble norvax klym 20260527`
  returned loaded-store cue packets in `30.344ms` instead, with
  `loaded_store_context_search=21.6139ms`. AXI recall on the same topic hit
  cache in `0.6234ms`, and AXI context hit project cache in `0.049ms`. After
  full startup validation, live value reports read-path p95 `80.397ms`, read
  cache hit rate `0.8`, and zero budget misses/degraded reads/timeouts. Full
  startup validation reported `13 pass, 1 warn, 0 skip`: doctor and live MCP
  catalog passed, and the only warning remains stale/root real SessionStart
  proof. Focused runtime/context tests passed with `558 passed, 13 skipped`;
  ruff and `git diff --check` passed.
- The 2026-05-28 loaded-store recall hardening pass handles the corresponding
  explicit-recall miss path. Same-project home packets and identity packets can
  carry an otherwise no-evidence recall after fast preflight misses or times
  out, and project-scoped explicit recall now syncs the persistent packet cache
  before assuming no project packet is available. This keeps post-restart
  dogfood recall from falling into the full deep-search tail when local project
  context already exists. Live evidence after final reinstall/restart:
  `zzpersist noartifact yonderplasm quibbleflux 20260528 final true miss tail`
  returned three project packets in `100.1129ms` with
  `preflight_timeout_context_packet_fallback`, no degradation, and no budget
  miss. After the lifecycle matrix restarted the runtime again, the fresh
  `zzaftermatrix ...` miss still returned project context in `101.9488ms`.
  Final `engram axi value --json` reported read-path p95 `65.7748ms`, cache hit
  rate `1.0`, and zero read budget misses/degraded reads/timeouts. Full startup
  validation passed with 27 MCP tools, `remember` present, and
  `recall.project_path`; the refreshed matrix produced
  `/private/tmp/engram-dogfood-startup-20260528-071304` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Focused retrieval tests passed with
  `94 passed, 2 skipped`, ruff passed, and `git diff --check` is clean.
- The next resumed Codex turn provided a second real-session performance sample
  without code changes. MCP `get_context` returned useful project packets in
  `249.4908ms` with no degradation, and MCP `recall` hit cache in `1.4862ms`.
  AXI context for the same topic hit project-file cache rescue in `2.5496ms`,
  AXI recall was `cache_satisfied` in `32.0827ms`, and a fresh no-evidence
  recall returned a project packet in `101.3667ms`. A brand-new context-miss
  topic then hit cache rescue in `4.3832ms`; MCP `get_context` on that topic
  hit cache in `0.2735ms`, and MCP `recall` was `cache_satisfied` in
  `45.9587ms`. `engram axi value --json` still reported zero read budget
  misses, degraded reads, or timeouts. This moves the goal forward as real
  Codex dogfood evidence rather than another isolated startup proof.
- The following resumed turn intentionally probed the apparent next bottleneck:
  a fresh topic-specific MCP context request for write-path capture/observe
  work. That first call returned useful project packets but paid a cold
  project-file scan (`duration_ms=581.6269`,
  `project_file_fallback=550.3093ms`), and created the stable same-project
  cache entry. After `engramctl stop && engramctl start` restarted the runtime
  to PID `35144`, the first new AXI context topic returned through
  project-file cache rescue in `2.3597ms`, the matching AXI recall returned
  three project packets in `106.2332ms` with
  `preflight_timeout_context_packet_fallback`, MCP `get_context` hit packet
  cache in `0.0617ms`, and MCP `recall` was `cache_satisfied` in `89.6023ms`.
  Post-restart `engram axi value --json` reported read-path p95
  `106.2332ms`, cache hit rate `1.0`, and zero read budget misses, degraded
  reads, or timeouts. This confirms stable project packets survive restart
  once warmed; the remaining cold edge is the first-ever stable project packet
  build when no same-project home entry exists yet.
- A follow-up soft-wait fix tightened the MCP context path exposed by the same
  real-session work. The loaded-store context helper no longer blocks on the
  project-file scan after its soft preflight wait; if the scan is still pending,
  the project-file payload builder can immediately rescue from same-project
  cached packets while the scan refreshes the exact topic in the background.
  Regression coverage now forces slow loaded-store preflight plus slow
  project-file scan and verifies `project_file_cache_rescue`. After reinstalling
  the local package and restarting to LaunchAgent PID `40680`, the first fresh
  MCP topic without a usable stable sidecar entry rebuilt in `937.6488ms`.
  Once stable cache existed, fresh AXI context used
  `project_file_cache_rescue` in `10.3801ms`, exact repeat context hit cache in
  `0.0413ms`, and fresh MCP topics stayed bounded without degradation at
  `104.0038ms` and `138.8205ms`. After the final no-project guardrail
  reinstall/restart, the runtime is healthy on PID `41982` and a fresh AXI
  context probe used `project_file_cache_rescue` in `2.238ms`.
- The next continuation verified the committed runtime without another code
  change. HEAD `78aa7ed` was clean and pushed; `engramctl status` reported
  native PyO3 Helix healthy; startup validation passed all checks; and
  `python3 scripts/dogfood_startup_matrix.py --repo /Users/konnermoshier/Engram --confirm-lifecycle`
  produced `/private/tmp/engram-dogfood-startup-20260528-074024` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix status was healthy on PID
  `43378`. AXI context returned five project-file packets in `38.2785ms`, AXI
  recall found a cue packet in `11.8581ms`, a forced no-evidence recall returned
  a project packet in `102.2185ms`, MCP `get_context` returned five project-file
  packets in `143.7264ms`, and MCP `recall` was `cache_satisfied` in `2.2772ms`.
  The value report retained one matrix-era MCP context p95 sample at
  `581.6838ms`, but zero read budget misses, degraded reads, or timeouts.
- The next real Codex continuation at HEAD `5d3554d` found that normal
  observe/projection pressure could invalidate stable project-file cache rescue
  rows. The first fresh MCP `get_context` after capture/projection returned
  useful packets but paid `715.1811ms` total with
  `project_file_fallback=709.6484ms`. Broad graph/episode cache invalidation now
  preserves entries whose packets are all `trust.source=project_file`, while
  explicit clears and non-file packet invalidation still clear mutable packet
  views. After reinstall/restart to PID `45085`, AXI context seeded cache rescue
  in `2.2862ms`; live MCP observe stored `ep_4c0605de51da`; projection ingested
  it without invalidating `project_home` project-file rows; post-observe MCP
  `get_context` returned useful packets in `83.4848ms`; AXI context used
  `project_file_cache_rescue` in `3.608ms`; AXI recall was `cache_satisfied` in
  `2.3008ms`; and live value read-path p95 was `83.8738ms` with zero budget
  misses, degradation, or timeouts.
- A post-commit reinstall/restart pass verified HEAD `1029cf7` as the running
  dogfood path. The local package was reinstalled from
  `/Users/konnermoshier/Engram/server[local,native]`, the LaunchAgent restarted
  to PID `48229`, AXI context used project-file cache rescue in `2.8982ms`,
  repeat AXI recall was `cache_satisfied` in `1.9462ms`, and a forced
  no-evidence AXI recall returned three project packets in `102.0141ms` instead
  of an empty timeout payload. MCP `get_context` returned
  useful packets in `161.2856ms`; MCP `recall` was `cache_satisfied` in
  `5.6553ms`; validation passed with a 27-tool live MCP catalog including
  `remember` and `recall.project_path`; and the confirmed lifecycle matrix
  produced `/private/tmp/engram-dogfood-startup-20260528-075531` with
  `13 pass, 0 warn, 0 fail, 0 skip`. Post-matrix runtime stayed healthy on PID
  `52284`, and live value reported read-path p95 `73.4648ms`, read cache hit
  rate `1.0`, and zero read budget misses, degradation, or timeouts.
- A later real Codex continuation found a recall-priority edge in the cached
  context fallback. Fresh session packet `ep_971656d12ca9` could surface once,
  then repeat recall for the same resumed-goal query was masked by newer
  `project_home` packets because the fallback fetched all scopes through one
  globally recency-limited packet-cache query. The fallback now reads
  `session_recent`, `identity_core`, and `project_home` separately before
  dedupe and relevance filtering. After reinstall/restart to LaunchAgent PID
  `62723`, fresh marker `ep_0352d83b5ece` stayed first for repeated AXI recalls
  even after two newer project-home context warmups: `3.2818ms` and `2.2598ms`,
  both `cache_satisfied`, with no degradation or budget miss. Focused
  regression tests passed with `113 passed`, ruff passed, and full startup
  validation passed all 14 checks against PID `62723`, including the 27-tool
  live MCP catalog with `remember` and `recall.project_path`. Post-validation
  live value reported read-path p95 `81.5933ms`, read cache hit rate `0.95`,
  and zero read budget misses, degradation, or timeouts.
- The next project-file context usefulness pass fixed first-use stale rescue
  behavior. A specific AXI context query could return old stable
  `project_file_cache_rescue` packets while a fresh project-file scan was still
  running. The context path now waits one bounded
  `context_fast_preflight_soft_wait_ms` window for the scan before returning
  stable rescue packets. After reinstall/restart to PID `70588`, AXI context
  for `fresh observations are no longer starved by project-home cache recency
  live softwait firstuse 20260528` returned current `docs/CURRENT_HANDOFF.md`
  evidence first in `1083.3078ms` with no degradation or budget miss, then the
  exact repeat hit cache in `0.0475ms`. MCP `get_context` for the same topic
  hit cache in `0.069ms`, MCP `recall` was `cache_satisfied` in `23.2ms`, and
  forced no-evidence AXI recall still returned project context in `100.7934ms`
  with `preflight_timeout_context_packet_fallback`. A stricter rescue filter
  now requires stable project-file rescue packets to match the specific topic;
  live AXI context for `stable project-file rescue packets relevant topic
  soft-wait filter 20260528` returned only relevant current handoff/AXI-plan
  packets in `130.8898ms` with no budget miss or degradation. Full startup
  validation passed all 14 checks against PID `74441`; the confirmed lifecycle
  matrix produced `/private/tmp/engram-dogfood-startup-20260528-082753` with
  `13 pass, 0 warn, 0 fail, 0 skip`; and post-matrix value on PID `75670`
  showed read-path p95 `73.0892ms`, cache hit rate `1.0`, and zero read budget
  misses, degradation, or timeouts.
- The recent project-file reuse follow-up fixed the next live MCP latency edge.
  A resumed Codex `get_context` sample paid `project_file_fallback=568.0876ms`,
  while the isolated local fallback builder measured about `19ms`; the issue was
  cache reuse/rebuild policy, not parser cost. Context now reuses recent
  same-project current-version project-file packets before launching
  loaded-store preflight or exact project scans, but only after topic relevance
  filtering and with a specific-token guard for date/hyphen/long-token queries.
  After reinstall/restart to PID `81443`, AXI context for `stable project-file
  rescue packets relevant topic soft-wait filter 20260528 final3 nearby reuse`
  hit `packet_cache.scopes.project_file_recent_reuse=2` in `0.0451ms`; MCP
  `get_context` for the same family hit `project_file_recent_reuse=2` in
  `0.1312ms`. Startup validation passed against PID `81443`; the final
  lifecycle matrix produced `/private/tmp/engram-dogfood-startup-20260528-085123`
  with `13 pass, 0 warn, 0 fail, 0 skip`, and post-matrix PID `82520` stayed
  healthy. The first fresh post-matrix topic rebuilt bounded project context in
  `99.5414ms`, then the nearby topic hit recent reuse in `0.0447ms`.
- The cache-specificity hardening follow-up fixed noisy project-file
  cache-satisfaction. A live probe showed that a query with the new marker
  `orchid` could be satisfied by older project-file packets that only matched
  generic dogfood/context terms plus `20260528`. Recent project-file context
  reuse now requires all distinctive non-generic query tokens to match, and
  explicit recall uses the same guard before non-exact project-file packets can
  report `cache_satisfied`. After reinstall/restart to PID `84563`, AXI context
  for `live dogfood loaded-store recall context trace orchid2 20260528
  specificity probe` rebuilt project-file packets instead of using stale recent
  reuse (`durationMs=1020.0549`, no degradation or budget miss). Matching AXI
  recall no longer reported `cache_satisfied`; it returned bounded
  project-file fallback packets with `skipReason=null`, `durationMs=1139.2205`,
  and no degradation. Exact repeat context hit cache in `0.0413ms`, and MCP
  `recall` for the exact MCP topic was `cache_satisfied` in `2.4848ms`.
  Focused retrieval/cache tests passed with `119 passed`; startup validation
  passed against PID `84563`; the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260528-090414` with `13 pass, 0 warn,
  0 fail, 0 skip`; and post-matrix PID `85767` stayed healthy.
- The immediate-rescue follow-up removed the remaining first-request soft-wait
  cost when relevant same-project project-file packets are already available. A
  cold first AXI context probe previously returned in `129.2005ms` because it
  still paid `projectFileFallbackSoftWait=75.0566` before returning a rescue
  packet. Context now checks topic-relevant same-project `project_file_cache_rescue`
  packets before waiting on loaded-store preflight or project-file fallback
  soft wait, and leaves the fresh exact-topic project-file scan to finish and
  update cache in the background. After reinstall/restart to PID `97485`, AXI
  context for `early rescue mcp axial garnetmarker 20260528 project file
  fallback` returned through `project_file_cache_rescue=1` in `46.3217ms` with
  `projectFileFallbackSoftWait=0.0`; MCP `get_context` for `early rescue mcp
  sapphiremarker 20260528 project file fallback` returned through the same
  rescue path in `5.8787ms` with no loaded-store preflight wait; and MCP
  `recall` for that topic was `cache_satisfied` in `43.1145ms`. Focused
  retrieval/cache tests passed with `119 passed`; ruff and `git diff --check`
  passed; the installer doctor passed; startup validation passed against PID
  `97485`; the confirmed lifecycle matrix produced
  `/private/tmp/engram-dogfood-startup-20260528-092158` with `13 pass, 0 warn,
  0 fail, 0 skip`; and post-matrix PID `99293` stayed healthy. Post-matrix
  value reported read-path p95 `4.3265ms`, read cache hit rate `1.0`, and zero
  read budget misses, degradation, or timeouts.

## Work Plan

1. Capture the current installed-user state before edits.
   Record `.mcp.json`, Claude Code MCP scopes, Codex MCP config, AXI hook trace,
   LaunchAgent state, process/listener state, `engramctl status`, and
   `engramctl storage`.

2. Build a dogfood validation script or documented command runner.
   It should run non-destructive checks first and print a compact pass/fail
   report with exact next actions. Keep destructive cleanup behind explicit
   flags.

   Current runner:

   ```bash
   python3 scripts/dogfood_startup_validation.py
   python3 scripts/dogfood_startup_validation.py --skip-slow
   python3 scripts/dogfood_startup_matrix.py --confirm-lifecycle
   ```

   The full run includes `engramctl doctor` and a live MCP tool-catalog probe.
   `--skip-slow` is for quick startup/config inspection when the runtime is
   warming or when a shell agent needs a fast status packet. The matrix runner
   records exact commands and raw outputs for warmed, stopped, restarted, and
   stale-PID states into `/tmp/engram-dogfood-startup-*` by default.

3. Fix `engramctl` lifecycle parity.
   `start`, `stop`, and `status` must agree with the configured local
   supervisor. If LaunchAgent is the active native dogfood path, the CLI should
   not rely only on a PID file created by `nohup`.

   The local-mode CLI recognizes an existing macOS LaunchAgent at
   `~/Library/LaunchAgents/dev.engram.local.plist` by default, or
   `ENGRAM_LAUNCH_AGENT_LABEL` / `ENGRAM_LAUNCH_AGENT_PLIST` when overridden.
   If no LaunchAgent exists, it falls back to the PID-file `nohup` path.

4. Harden readiness probes.
   Distinguish "process not listening", "HTTP health not ready", "native graph
   warmup still running", "MCP handshake timed out", and "tool catalog missing
   expected tools". Avoid reporting false config failures while native startup
   is still warming.

5. Add MCP client config audits.
   Detect duplicate Claude Code scopes, stale stdio entries, project/user
   endpoint mismatches, missing Codex server entries, and missing OpenClaw
   config. Print exact cleanup commands before changing anything.
   OpenClaw config should prefer a global `openclaw` command, fall back to
   `npx -y openclaw` when npm/npx is available, and still write the same
   streamable HTTP MCP registry entry.

6. Add tool-catalog verification.
   Assert the installed MCP server exposes the expected tool set, including
   `remember`, and add a focused regression test for the catalog or manifest so
   prompt instructions cannot mention unavailable tools.

7. Verify AXI startup traces.
   Confirm session-start hook records and follow-up context/recall records for
   Codex and Claude Code. Classify failures as hook-not-installed,
   hook-not-run, runtime-offline, follow-up-missing, or project-path-mismatch.

8. Re-run from clean-ish states.
   Validate at least: warmed runtime, cold native runtime, stopped runtime,
   stale PID/listener simulation, duplicate Claude config simulation, and
   restarted Codex/Claude Code sessions.

9. Update public docs.
   Align README and install docs around the actual happy path, the diagnostic
   commands, expected storage paths, and known warmup behavior.

10. Commit only intentional repo changes.
    Do not commit machine-local config such as `~/.claude.json`; document those
    as operator-state changes in the final report.

## Acceptance Criteria

- `engramctl doctor` or the new dogfood validation command reports a clear pass
  for native PyO3 Helix startup.
- `engramctl status` and `engramctl storage` agree with the actual running
  process, active supervisor, port, data directory, and storage counts.
- Claude Code has exactly one Engram MCP definition for the tested project and
  reports `Status: connected`.
- Codex can discover and call `mcp__engram__.remember` without relying on a
  lucky broad discovery query.
- AXI startup and follow-up trace evidence exists for Codex and Claude Code
  when AXI hooks are installed.
- Project path behavior is explicit: starting from home, repo root, and nested
  directories either resolves correctly or reports the limitation clearly.
- Stale PID/listener and half-started runtime states are detected with specific
  remediation.
- OpenClaw docs/config remain MCP-first, with AXI as a documented shell fallback.
- No release doc implies Docker is required for the main native path.
- The final report lists what was fixed, what remains, and the exact commands
  that passed.

## Non-Goals

- Do not redesign Engram memory semantics in this goal.
- Do not make AXI replace MCP.
- Do not require Docker for the primary user path.
- Do not purge real user data without an explicit operator command.
- Do not treat third-party client auth failures unrelated to Engram as Engram
  release blockers.
