# AXI Interface Plan

Date: 2026-05-20

Status: implementation in progress. The first REST-backed CLI slice is in place:
`engram axi`, `context`, `recall`, `storage`, `doctor`, `observe`, `remember`,
`bootstrap`, JSON output, output budgets, request timeouts, and
degraded-offline behavior. The home packet intentionally stays read-only and
progressive: it reports runtime/storage/context availability and points to
`engram axi context` instead of loading full context every session start. Codex
and Claude Code read-only hook print/install support is now implemented through
`engram axi hooks` and `engramctl connect <client> --axi`. Local Codex and
Claude Code configs have been dogfooded with read-only startup hooks that pin
the installed Engram executable instead of depending on agent startup `PATH`.
OpenClaw remains MCP-first with documented AXI shell fallback until a stable
OpenClaw hook mechanism is confirmed. Real restarted-session adoption evidence
remains open.

This plan defines what Engram needs in order to add an AXI-style agent
interface. The target is not to replace MCP, REST, or the dashboard. The target
is to give AI harnesses a low-friction, token-efficient way to discover, load,
and use Engram at session start and during shell-driven work.

## Research Inputs

Primary sources reviewed:

- AXI homepage: <https://axi.md/>
- AXI skill: <https://raw.githubusercontent.com/kunchenguid/axi/main/.agents/skills/axi/SKILL.md>
- AXI JS SDK README: <https://raw.githubusercontent.com/kunchenguid/axi/main/packages/axi-sdk-js/README.md>
- `gh-axi` reference README: <https://raw.githubusercontent.com/kunchenguid/gh-axi/main/README.md>
- TOON spec: <https://toonformat.dev/reference/spec>
- Atuin AI agent hook docs: <https://docs.atuin.sh/cli/guide/agent-hooks/>
- Claude Code hooks reference: <https://code.claude.com/docs/en/hooks>
- Codex hooks reference: <https://developers.openai.com/codex/hooks>

Key takeaways:

- AXI is an agent-facing CLI design pattern. Its premise is that a carefully
  designed CLI can preserve much of MCP's discoverability and reliability while
  reducing schema and JSON overhead.
- AXI treats stdout as agent context, so compact output is a first-class
  requirement. Defaults should answer "what should the agent do next?" instead
  of dumping full state.
- A top-level no-argument command should return a compact "home" packet:
  identity, live status, current context, and next commands.
- Use progressive disclosure. Lists should carry small schemas; detail commands
  expose larger bodies; `--full`, `--fields`, `--limit`, and `--json` are escape
  hatches.
- TOON is the preferred AXI output format. It is line-oriented and especially
  compact for arrays of uniform objects. The spec is still evolving, so Engram
  should start with a small, internal output boundary rather than making core
  runtime logic depend on a young serializer.
- Hooks are the adoption lever. AXI reference tools use session-start hooks so
  the agent sees useful state without first deciding to call a tool. Current
  real-world hook conventions include Claude Code's `~/.claude/settings.json`
  and Codex's `~/.codex/hooks.json`.

## Why AXI Fits Engram

Engram's adoption problem is not only "is the MCP server connected?" The live
client work already showed that a connected memory server can still be ignored
if the agent has no compact, trustworthy reason to use it. Engram needs a small
agent-visible surface that says:

- this brain is running or why it is not
- which storage mode is active
- whether the graph is fresh, empty, stale, or bootstrapped
- what context is relevant to the current workspace or topic
- which next command is safe and useful
- how to capture or recall without reading a full tool catalog

MCP remains valuable for structured tool calls, client integrations, and rich
contracts. AXI should sit beside it as the shell-native agent experience:

```text
MCP: typed protocol for tool-capable clients
REST: runtime API and dashboard backend
Dashboard: human observability
engramctl: install, lifecycle, startup, client config
engram axi: compact agent-facing CLI context and capture surface
```

## Current Repo Baseline

Relevant existing surfaces:

- `server/engram/__main__.py` exposes the Python `engram` CLI with `setup`,
  `serve`, `mcp`, `lifecycle`, `evaluate`, `authority`, `adoption`, `doctor`,
  `health`, `update`, and `version`.
- `installer/engramctl` owns public install, startup, storage visibility, client
  connection, OpenClaw skill install, quickstart, and lifecycle commands.
- `server/engram/api/knowledge.py` already exposes REST endpoints for
  `observe`, `remember`, `recall`, `context`, `bootstrap`, `route`,
  `artifacts/search`, and `runtime`.
- `server/engram/api/storage.py` exposes storage diagnostics that `engramctl
  storage` already formats for humans.
- `server/engram/mcp/server.py` exposes the memory tools agents need:
  `observe`, `remember`, `recall`, `search_entities`, `search_facts`, `forget`,
  `get_context`, `bootstrap_project`, `route_question`, `search_artifacts`,
  `claim_authority`, `get_runtime_state`, intentions, identity core, and
  evaluation helpers.
- `server/engram/setup.py` already has Claude Code hook templates for
  auto-capture. `engramctl connect` already writes MCP config for Codex,
  Claude Code, Cursor, Windsurf, Claude Desktop, and OpenClaw.
- `docs/CURRENT_HANDOFF.md` makes Helix native PyO3 the preferred no-Docker
  runtime path. AXI should assume that path as the default public dogfood path.

Conclusion: do not create a separate standalone tool first. Add AXI as a
subcommand under the existing `engram` CLI and wire hook installation through
`engramctl`.

## Product Contract

AXI should make Engram easier for agents to use, not easier to pollute.

Default behavior must be read-only:

- `engram axi` must not write memories.
- session-start hooks must not capture conversation content by default.
- storage or runtime failure must degrade to a compact status packet, not hang.

Write behavior must be explicit:

- `engram axi observe --stdin`
- `engram axi remember --stdin`
- `engram axi bootstrap /path`
- optional capture hooks installed only with a visible opt-in flag.

The default command should be safe to run in every new agent session.

Current dogfood detail: AXI home uses `/api/knowledge/runtime/fast`, which now
schedules a non-blocking in-memory project-file prefix warmup for the provided
project path. The startup packet remains read-only, graph-free, and
non-capturing, but the first follow-up `engram axi context --project ...` is less
likely to pay cold local file reads. Live proof after reinstall/restart on
LaunchAgent PID `69492`: AXI home triggered the warmup, fresh AXI context
returned useful project-file packets in `durationMs=183.2692`, repeat context
hit cache in `0.0474ms`, and AXI recall was `cache_satisfied` in `0.7942ms`.
The follow-up executor-isolation pass moved project-file context/recall fallback
and prefix warmup off the default executor, so loaded native storage work cannot
starve the local-file rescue path. After reinstall/restart on PID `71087`, fresh
AXI context returned loaded-store cue packets in `77.4707ms`, and AXI recall was
`cache_satisfied` in `0.5423ms`.
AXI project context now stays in that same startup-safe lane with or without an
explicit topic: `engram axi context --project ...` skips loaded-store preflight
and uses cached/project-file context, while explicit long-tail memory searches
remain available through `engram axi recall`. After reinstall/restart on PID
`74526`, a warmed fresh topic-specific AXI context returned in `57.0956ms`
with `projectFileFallback=32.1867ms`, and the exact repeat hit cache in
`0.0461ms`. Startup validation stayed at `13 pass, 1 warn`; the only warning is
stale/root real SessionStart proof.
The next live pass removed a duplicate scan inside the project-file packet
builder, so summary matching and evidence claims reuse one topic-match pass per
candidate file. After reinstall/restart on PID `75796`, a warmed fresh AXI
context built project-file packets in `22.6326ms`, and the exact repeat hit
cache in `0.0393ms`.
The MCP-side follow-up also made the richer context path easier to tune:
project-file fallback responses now expose the loaded-store preflight wait when
that preflight misses. After reinstall/restart on PID `76806`, a fresh MCP miss
returned useful project-file packets in `103.96ms`, split as
`loaded_store_context_preflight=99.7659ms` and
`project_file_fallback=23.6974ms`.
The next MCP policy pass added `context_fast_preflight_soft_wait_ms` with a
`75ms` default, so quick loaded-store hits still win but loaded-store misses can
fall back as soon as project-file context is ready. After reinstall/restart on
PID `77735`, useful loaded-store MCP context returned in `85.4594ms`, while a
fresh miss returned project-file packets in `25.7279ms` and recall on that same
topic hit cache in `0.8398ms`.
The follow-up persistent-cache rescue keeps the initial AXI/MCP context cache
lookup in-memory, but if a project-file scan is still pending it may sync
persistent packet cache for current-version same-project project-file packets.
That gives agents a fast project packet instead of waiting on a cold local file
scan, while the scan continues and refreshes the exact topic cache. After
reinstall/restart on PID `80961`, the first fresh MCP miss after restart was
served by loaded-store context in `30.344ms`; AXI recall on the same topic was
cache-satisfied in `0.6234ms`, AXI context hit project cache in `0.049ms`, and
post-validation live value read-path p95 was `80.397ms` with zero budget
misses/degraded reads/timeouts.
The follow-up soft-wait fix keeps that rescue from being masked by a slow fresh
project-file scan. MCP loaded-store context preflight now returns after the
configured soft wait instead of blocking until either loaded-store or the file
scan finishes; when the scan is still pending, stable same-project packets can
serve `project_file_cache_rescue` while the exact-topic scan completes in the
background. After reinstall/restart on PID `40680`, the first fresh MCP context
with no stable sidecar entry rebuilt in `937.6488ms`; after that seed existed,
fresh AXI context used `project_file_cache_rescue` in `10.3801ms` and exact
repeat context hit cache in `0.0413ms`. Fresh MCP contexts on the same warmed
filesystem returned bounded project-file packets in `104.0038ms` and
`138.8205ms` without degradation. After the final no-project guardrail
reinstall/restart, PID `41982` stayed healthy and fresh AXI context used
`project_file_cache_rescue` in `2.238ms`.
The next real Codex continuation found that normal capture/projection could
erase this stable project-file rescue path by broadly invalidating packet-cache
entries after graph mutations. Broad graph/episode invalidation now preserves
entries composed only of `trust.source=project_file` packets; explicit cache
clears still clear them, and cue/entity/relationship packet views can still be
invalidated when their source ids are affected. After reinstall/restart on PID
`45085`, AXI context used project-file cache rescue in `2.2862ms`, a live MCP
observe/projection cycle left the `project_home` file rows fresh, MCP
`get_context` returned useful packets in `83.4848ms`, AXI context hit rescue in
`3.608ms`, and AXI recall was `cache_satisfied` in `2.3008ms`.

## Command Shape

Use a dedicated top-level subcommand:

```bash
engram axi
```

This preserves existing `engram` no-argument help and avoids disrupting human
operator muscle memory. The AXI home packet lives at `engram axi` with no
subcommand.

Recommended commands:

```text
engram axi
engram axi context [--topic TEXT] [--project PATH] [--budget N] [--timeout SECONDS]
engram axi recall QUERY [--limit N] [--budget N] [--timeout SECONDS]
engram axi observe --stdin [--source SOURCE] [--conversation-date ISO]
engram axi remember --stdin [--source SOURCE] [--conversation-date ISO]
engram axi bootstrap PATH [--include GLOB ...]
engram axi storage [--json]
engram axi doctor [--hooks codex claude-code] [--require-hook-run] [--require-followup] [--json]
engram axi hooks install codex|claude-code [--capture] [--dry-run]
engram axi hooks print codex|claude-code
engram axi hooks status codex|claude-code
engram axi hooks uninstall codex|claude-code [--dry-run]
```

Global flags:

```text
--server-url URL      default http://127.0.0.1:8100
--timeout SECONDS    default depends on command
--budget N           approximate output-token budget
--json               machine-readable JSON instead of compact text
--full               include full detail where available
--no-color           never emit ANSI
```

The `engram axi` command should prefer the running REST API. If the API is not
running, it should inspect local install files when possible and return a
degraded packet with the exact next command to start Engram.

## Home Packet

The no-argument AXI command is the most important behavior. It is what session
hooks should run.

Example healthy output:

```text
bin: ~/.local/bin/engram
description: Long-term memory brain for AI agents
status: healthy
mode: helix
transport: native
server: http://127.0.0.1:8100
storage:
  data_dir: ~/.helix/engram-native
  size: 2.1 GB
brain:
  group: default
  artifact_status: ready
  recall_status: ready
  lifecycle: Capture -> Cue -> Project -> Recall -> Consolidate
context:
  freshness: recent
  project: /Users/konnermoshier/Engram
  summary: Engram is the portable cross-harness memory authority; use recall when prior context could change the answer.
next[3]{cmd,reason}:
  engram axi context --project "$PWD" --budget 800 --timeout 5,Load compact workspace context
  engram axi recall "query" --limit 5 --timeout 5,Search long-tail memory
  engram axi observe --stdin --source <client|axi>,Capture explicit user-approved notes
```

Example degraded output:

```text
bin: ~/.local/bin/engram
description: Long-term memory brain for AI agents
status: offline
server: http://127.0.0.1:8100
install:
  variant: lite
  config: ~/.engram/.env
next[2]{cmd,reason}:
  engramctl start,Start the local Engram runtime
  engramctl status,Inspect configured mode, paths, and logs
```

Output rules:

- Default output is compact text in a TOON-compatible style.
- `--json` returns stable JSON for scripts and tests.
- Large fields must be truncated with a visible total length and a `--full`
  escape hatch.
- Default command output should fit under roughly 800 tokens.
- `context` should fit under the caller's budget, defaulting to 600-800 tokens.
- Errors should include a concrete next command, not a generic stack trace.

## REST Mapping

The first implementation should be an HTTP client over the local REST runtime.
That avoids initializing a second graph runtime from every hook and keeps the
agent path aligned with dashboard and MCP behavior.

Mapping:

```text
engram axi                       GET  /health, /api/knowledge/runtime/fast, /api/storage
engram axi context               GET  /api/knowledge/context
engram axi recall                GET  /api/knowledge/recall?q=...
engram axi observe --stdin       POST /api/knowledge/observe
engram axi remember --stdin      POST /api/knowledge/remember
engram axi bootstrap             POST /api/knowledge/bootstrap
engram axi storage               GET  /api/storage?live=true&timeoutSeconds=10
engram axi doctor                call existing doctor runner, plus REST probes
```

Direct store initialization should be a later fallback only if the running API
is absent and a read-only local packet is still useful.
The no-argument home packet uses short bounded probes and must not load full
context or write memories; it only points agents to `engram axi context` and
`engram axi recall` when deeper memory is useful.

## Output Boundary

Add a small AXI output layer, not formatting scattered across commands:

```text
server/engram/axi/
  __init__.py
  cli.py              argparse command registration and dispatch
  client.py           bounded REST client
  surfaces.py         command payload normalization
  toon.py             small TOON-compatible renderer for Engram's supported shapes
  budgets.py          truncation and field selection helpers
  hooks.py            hook config payload generation
```

Do not put AXI formatting inside MCP presenters. MCP should keep its JSON
contract. AXI can reuse REST payloads, but it should have its own compact
presenters.

TOON dependency decision:

- Start with a minimal internal renderer for the limited shapes Engram needs:
  mappings, nested mappings, arrays of uniform objects, primitives, and
  conservative string quoting.
- Keep all internal logic as Python dictionaries.
- Add a dependency only after verifying a Python TOON implementation tracks the
  current spec closely enough and does not complicate the install path.
- Treat TOON as an output contract, not a storage format.

## Hook Strategy

AXI's highest leverage is session-start context injection. The first hook should
be read-only.

Codex:

```bash
engram axi hooks install codex
```

Should merge a managed command into `~/.codex/hooks.json` that runs something
like:

```bash
engram axi hook-run --budget 800 --timeout 3
```

Codex uses matcher groups in `~/.codex/hooks.json`: `hooks.SessionStart[]`
entries contain a `matcher` and nested command `hooks[]`. The managed command
uses `type: "command"`, a seconds-based `timeout`, and `statusMessage`. Legacy
flat managed entries should be removed during merge so old experimental shapes
cannot linger beside the active hook. In Codex TUI dogfood, `codex exec` did not
exercise `SessionStart`; the interactive TUI displayed hook review, then ran the
trusted `SessionStart` hook at the first submitted prompt.

Claude Code:

```bash
engram axi hooks install claude-code
```

Should merge a managed `SessionStart` hook into `~/.claude/settings.json`. This
can coexist with the current auto-capture hooks, but should remain separately
managed so users can enable read-only context without capture.

OpenClaw:

```bash
engram axi hooks install openclaw
```

Should be planned after confirming OpenClaw's current hook or extension
mechanism. Until then, OpenClaw remains MCP-first with the published
`engram-brain` skill.

Capture hooks:

- Not default.
- Require `--capture`.
- Capture policy must be explicit in generated config.
- Prefer session-end or user-approved capture over every-turn capture unless
  the harness provides clean event boundaries and privacy controls.

Hook installer requirements:

- Preserve user config.
- Mark managed blocks clearly.
- Be idempotent.
- Provide `--dry-run`.
- Print exact files touched.
- Never start the Engram server from the hook by default.
- Bound execution time to 3-5 seconds for session-start.

## Relationship To MCP

AXI should reduce the chance that agents ignore Engram. It should not reduce
MCP quality.

Use MCP when:

- the harness supports tools well
- the agent needs structured writes or graph operations
- the client can follow the memory authority protocol
- rich response schemas matter

Use AXI when:

- session-start context should be injected before tool choice
- the harness has shell access but poor MCP ergonomics
- the agent needs a compact status/context packet
- a user wants a cross-harness memory affordance that looks the same in Codex,
  Claude Code, OpenClaw, Cursor, Windsurf, or other shell-capable clients

The combined adoption story should be:

```text
AXI hook primes the agent with compact context.
MCP tools handle structured memory operations when available.
AXI commands remain available as a shell-native fallback.
```

## Implementation Phases

### Phase 1: Read-Only AXI CLI

Deliver:

- `engram axi` home packet.
- `engram axi context`.
- `engram axi recall`.
- `engram axi storage`.
- `--json`, `--budget`, `--timeout`, and `--full` flags.
- Compact degraded output when REST is offline.

Acceptance:

- `engram axi` returns in under 2 seconds when the server is healthy.
- `engram axi` returns in under 2 seconds when the server is offline.
- The default home packet stays below the target token budget.
- Context and recall respect truncation and include escape hatches.
- Tests cover healthy, offline, timeout, malformed REST response, truncation,
  and JSON output.

Suggested tests:

```text
tests/axi/test_axi_cli_home.py
tests/axi/test_axi_context.py
tests/axi/test_axi_recall.py
tests/axi/test_axi_toon.py
tests/axi/test_axi_budgeting.py
```

### Phase 2: Explicit Capture Commands

Deliver:

- `engram axi observe --stdin`.
- `engram axi remember --stdin`.
- source metadata defaults for `codex`, `claude-code`, `openclaw`, and manual.
- structured success and degraded error output.
- offline queue decision: either reject cleanly or reuse the existing capture
  queue. Do not silently drop input.

Acceptance:

- Writes require `--stdin` or an explicit argument.
- The command prints capture lifecycle state without dumping stored content.
- Auth-enabled runtimes have a documented token path.
- Tests cover successful write, auth failure, offline behavior, empty stdin,
  and oversized input.

### Phase 3: Hook Installation

Deliver:

- `engram axi hooks print codex|claude-code`.
- `engram axi hooks status codex|claude-code`.
- `engram axi hooks install codex --dry-run`.
- `engram axi hooks install codex`.
- `engram axi hooks install claude-code`.
- `engram axi hooks uninstall codex|claude-code`.
- `engramctl connect codex --axi` and `engramctl connect claude-code --axi`
  wrappers, or a separate `engramctl connect-axi` if that is cleaner.
- `engram axi doctor --hooks codex claude-code --require-hook-run --require-followup`
  release/dogfood gate.

Acceptance:

- Existing user hook config is preserved.
- Running install twice is idempotent.
- Read-only hook is the default.
- `--capture` is opt-in and clearly reported.
- Hook status reports missing, ready, and attention states without writing
  config, and includes the latest metadata-only hook run when available.
- Doctor can include requested hook checks and fails if an expected startup hook
  is missing, capturing, or no longer read-only.
- With `--require-hook-run`, doctor also fails until each requested hook has
  metadata-only run evidence.
- With `--require-followup`, doctor also fails until each requested hook has
  metadata-only evidence that the agent used AXI for `context` or `recall`.
- Tests use temporary home directories and verify exact JSON/TOML merges.

### Phase 4: OpenClaw And Generic Harness Support

Deliver:

- Confirm OpenClaw's current hook or extension mechanism.
- If supported, add `engram axi hooks install openclaw`.
- If not supported, update the OpenClaw skill to tell agents to use
  `engram axi` as the shell fallback beside MCP.
- Add generic documentation for shell-capable harnesses:

```bash
engram axi --project "$PWD" --budget 800 --timeout 3
```

Acceptance:

- OpenClaw users have a documented AXI path even if hooks are not available.
- The OpenClaw skill does not depend on Claude-specific hook semantics.
- Existing MCP install instructions remain valid.

### Phase 5: Dogfood And Adoption Evidence

Deliver:

- Local Codex dogfood with session-start AXI enabled.
- Claude Code dogfood with read-only AXI enabled.
- Compare default MCP-first session behavior against AXI-primed session
  behavior on the same prompts.
- Capture timing, token size, whether the agent called recall, and whether it
  captured useful memory.

Acceptance:

- AXI home packet is visibly shown to the agent at session start.
- Agent uses Engram without the user reminding it.
- No capture occurs from the read-only hook.
- Metrics show AXI startup cost and latency are low enough to keep enabled.

Evidence, 2026-05-20:

- Installed-user Codex path verified with `engramctl connect codex --axi`.
  The generated `~/.codex/hooks.json` uses the current matcher-group
  `SessionStart` schema with a nested managed `engram-axi-context` command,
  `read_only: true`, `capture: false`, a seconds-based bounded hook process
  timeout, `statusMessage`, and an absolute command targeting the installed
  `engram` executable. The command still uses `--timeout 3` for each REST
  request so startup cannot hang on one slow endpoint.
- Installed-user Claude Code path verified with
  `engramctl connect claude-code --axi --project <repo>`. The managed AXI hook
  was merged alongside an existing user hook without replacing it.
- The read-only AXI home packet returned healthy native Helix state from the
  dogfood runtime with artifact-backed project context, storage counts, and the
  resolved native data directory. `engram axi context --project <repo>
  --budget 800` also returned structured context under the default request
  timeout.
- After the dogfood runtime became busy, the installed `engram axi --timeout 3`
  home packet still returned in five local runs under 2 seconds
  (1.041-1.736s). Slow health/runtime/storage probes degrade the packet instead
  of blocking startup or falsely requiring writes.
- A release-style wheel build was checked and includes the full `engram/axi/`
  package, after adding an explicit Hatch wheel package target.
- `engram axi hooks status codex|claude-code` provides a no-write adoption
  audit so users and agents can confirm the managed startup hook is installed,
  read-only, and non-capturing before restarting a client.
- `engram axi doctor --hooks codex claude-code --require-hook-run` promotes
  that adoption audit into a release gate: runtime probes, requested hook
  readiness, and latest run evidence must all pass.
- Managed startup hook commands include `--trace-file ~/.engram/axi-hook-runs.jsonl`
  `--trace-client <client>`, and `--trace-origin session-start-hook`. The trace
  is metadata-only: timestamp, client, origin, operation, status, exit code,
  duration, server, project, budget, and timeout. It intentionally excludes
  prompts, retrieved context, memory bodies, and transcript content.
- Early local trace rows were manually seeded by invoking the exact Codex and
  Claude Code hook commands. After adding `trace-origin`, the strict dogfood
  gate correctly rejected those rows with `missing_session_start_origin`. This
  keeps manual CLI checks from satisfying restarted-client adoption proof.
- Real client dogfood then passed the strict gate:
  `engram axi doctor --hooks codex claude-code --require-hook-run --require-followup`
  returned `status: pass` after a trusted interactive Codex TUI session produced
  a `session-start-hook` home row and an `agent-followup` context row, and a
  Claude Code print-mode session produced the same startup plus follow-up
  evidence by running the suggested AXI context command through Bash.
- Hook-produced home packets include metadata-only trace flags on their
  suggested `context` and `recall` commands. The strict follow-up gate only
  accepts successful `context` or `recall` rows with
  `trace-origin=agent-followup`; failed rows, startup rows, and manual rows do
  not satisfy it.
- AXI context now applies the token budget before optional project expansion:
  identity, topic recall, project neighbors, and recency are capped, and
  project lookup/topic recall are timeboxed. Live native PyO3 dogfood returned
  `engram axi context --project "$PWD" --budget 800 --timeout 5` in 1.9-4.9s
  instead of timing out. A 3s follow-up timeout still timed out once on the busy
  dogfood graph, so startup hooks keep the home packet at 3s while recommended
  follow-up context calls use 5s when proving adoption.
- AXI home probes health/runtime/storage in parallel with a bounded 2.5s
  per-call cap. A busy native health probe can still mark the packet degraded,
  but the packet keeps the agent-facing `context` and `recall` next actions
  visible. A warmed live native dogfood packet returned healthy state, storage
  counts, and project artifact readiness in about 1.1s.
- REST shutdown consolidation is timeboxed so local native dogfood restarts do
  not get trapped in an opportunistic merge cycle during stop/restart.
- Capture remains disabled by default. `--capture` is metadata-only in the
  startup hook path and requires explicit user action before any write command
  is emitted.
- Later Codex dogfood found one real SessionStart trace with `project=/`,
  meaning the shell `$PWD` path can be wrong in some sessions. Managed hooks now
  call `engram axi hook-run`, read the client hook JSON from stdin, and use its
  `cwd` as the startup project path when available. The startup validator still
  warns on existing filesystem-root startup traces until a new trusted client
  SessionStart row replaces them. Follow-up `context`/`recall` commands with
  explicit `--project "$PWD"` still returned useful packets without degradation.
- The strict doctor gate now rejects pre-install hook evidence too:
  `engram axi doctor --hooks codex claude-code --require-hook-run
  --require-followup --json` currently fails with `stale_session_start_run` for
  both clients, and Codex also reports `last_run_project_root=true`.
- A refreshed full lifecycle matrix at
  `/private/tmp/engram-dogfood-startup-20260527-140202` passed runtime, doctor,
  MCP catalog, and lifecycle checks with `11 pass, 2 warn, 0 fail, 0 skip`; the
  remaining warnings are the expected stale/root hook-run evidence.
- Bounded project-file fallback now prefers current handoff evidence over older
  append-only matrix mentions and filters cached project-file packets by
  `project_path` before explicit recall can use them. Live AXI/MCP dogfood for
  `startup matrix 20260527 tiecheck gold` returned the current
  `20260527-140202` `CURRENT_HANDOFF.md` packet, then repeated from cache in
  sub-millisecond context and low-double-digit-millisecond recall paths.
- A later refreshed lifecycle matrix at
  `/private/tmp/engram-dogfood-startup-20260527-142608` passed with
  `11 pass, 2 warn, 0 fail, 0 skip`; the remaining warnings are still only stale
  client SessionStart evidence.
- Project-file fallback cache rows now carry `version=2`; AXI/MCP context exact
  hits require the current version, and explicit recall ignores stale
  unversioned project-file fallback rows before cache satisfaction. The same
  pass tightened adjacent-line summary scoring so only true wrapped evidence is
  joined. Live dogfood after reinstall rebuilt current Engram evidence for
  `startup matrix 20260527 tiecheck diamond` in `1249.1747ms`, rebuilt a clean
  context packet for
  `native PyO3 dogfood performance continuation cleanline 20260527` in
  `931.882ms`, then hit cache on repeats: AXI context `0.2474ms`, AXI recall
  `74.5458ms`, MCP context `0.1637ms`, and MCP recall `2.4684ms`.
- The latest full lifecycle evidence is
  `/private/tmp/engram-dogfood-startup-20260527-144207` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `23794`, and the remaining warnings are still only stale/root
  SessionStart evidence awaiting fresh real client sessions.
- Packet summaries now handle wrapped/lowercase evidence and long snippets more
  cleanly. Fresh AXI context for
  `startup matrix 20260527 tiecheck diamond project_file_recall_fallback
  continuationproof2` rebuilt the current handoff packet in `785.6527ms` with a
  whole-line summary and word-boundary truncation; repeats hit cache through AXI
  context `0.1051ms`, AXI recall `52.3059ms`, MCP context `0.0807ms`, and MCP
  recall `1.1909ms`.
- The latest full lifecycle evidence is now
  `/private/tmp/engram-dogfood-startup-20260527-145536` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `37314`, and the remaining warnings are still stale/root
  SessionStart evidence.
- Evidence-line selection now uses the same continuation quality bar: direct
  term hits are required, bounded previous continuation chains are joined, and
  unrelated prior sentences are trimmed before a wrapped previous line is used.
  Live dogfood for
  `evidence project_file_recall_fallback wrappedwindow liveproof 20260527
  chainfixed2` rebuilt current handoff evidence in `747.8033ms`, then hit cache
  through AXI context `0.179ms`, AXI recall `3.8688ms`, MCP context
  `1.5832ms`, and MCP recall `1.0518ms`.
- The latest full lifecycle evidence is now
  `/private/tmp/engram-dogfood-startup-20260527-151148` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `48671`, and the remaining warnings are still stale/root
  SessionStart evidence.
- Hot AXI/MCP reads now avoid per-call SQLite sidecar syncs on resident packet
  cache paths, and exact project-file fallback packets can satisfy repeated
  context/recall even when generic fallback summaries do not echo unusual query
  terms. Live AXI for
  `xafnorb quexilate zumbrel frobnicate mintcase exactcache5` rebuilt fallback
  packets in `618.2917ms`, repeated context from cache in `0.047ms`, and returned
  `cache_satisfied` recall in `0.7253ms` then `0.585ms`.
- The latest full lifecycle evidence is now
  `/private/tmp/engram-dogfood-startup-20260527-153357` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `57186`, and the remaining warnings are still stale/root
  SessionStart evidence. On that post-matrix runtime, the same synthetic topic
  rebuilt project-file context in `41.1342ms`, repeated from cache in
  `0.6055ms`, and AXI recall was `cache_satisfied` in `0.5183ms`.
- Packet-cache relevance is now stricter on weak synthetic misses: context
  rejects lone date/id matches, and explicit recall ignores generated `why_now`
  text before deciding cache satisfaction. Live AXI for
  `qvanta noexisting loadedstore miss tail 20260527 probeB` reported
  `cache_relevance_miss` and built project-file fallback packets in `44.7505ms`;
  a fresh recall-first `probeC` ran bounded recall in `228.2368ms`, found no
  memory results, and fell back to three project-file packets instead of stale
  loaded-store packets.
- The latest full lifecycle evidence is now
  `/private/tmp/engram-dogfood-startup-20260527-154316` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `59654`, and the remaining warnings are still stale/root
  SessionStart evidence.
- Write-tool auto-recall is now cache-only, and MCP write live-turn
  fingerprinting uses a short response-path wait while continuing in the
  background. Live MCP observe `obsF` returned in `85.9ms` wall time with
  `recall_middleware=0.4302ms`; AXI context for the same topic returned
  loaded-store cue packets in `30.4921ms`, and AXI recall hit the fresh
  `mcp_observe` packet in `0.3212ms`.
- The latest full lifecycle evidence is now
  `/private/tmp/engram-dogfood-startup-20260527-155526` with
  `11 pass, 2 warn, 0 fail, 0 skip`; post-matrix runtime is healthy on
  LaunchAgent PID `64398`, and the remaining warnings are still stale/root
  SessionStart evidence.
- Startup-validator follow-up guidance now distinguishes manual follow-up traces
  from real SessionStart proof. Manual `agent-followup` context/recall rows can
  prove AXI is callable, but stale/root SessionStart warnings require a fresh
  interactive client session from the target project. A real Codex TUI session
  from `/Users/konnermoshier/Engram` produced
  `operation=hook-run`, `origin=session-start-hook`,
  `project=/Users/konnermoshier/Engram`, `durationMs=11`, and `status=healthy`;
  a Claude Code print-mode probe then produced the same current shape with
  `durationMs=12`. The validator accepts this current `hook-run` shape and now
  passes startup/follow-up evidence for both clients. AXI home now uses the
  active trace client for capture suggestions instead of hard-coding Codex.
- Agent write surfaces now use a shorter raw-capture wait: MCP observe and REST
  auto-observe pass per-write `capture_store_timeout_ms=250`, while explicit
  writes keep the global `1000ms` default. After reinstall/restart on PID
  `67368`, live MCP observe returned with `capture_store=169ms` and
  `cue_store_timeout=251ms`, REST auto-observe deferred at
  `captureStoreTimeout=252ms`, and value telemetry showed write-path p95
  `440.135ms` with no degradation or budget misses.

Still needed:

- Preserve the trace fixture shape in docs/tests so future hook schema changes
  cannot silently regress Codex or Claude Code startup evidence.
- Keep the Codex and Claude Code `hook-run` fixture shape covered as client
  hook schemas evolve.

### Phase 6: Release Polish

Deliver:

- README quickstart update.
- `docs/install/helix.md` update.
- `docs/install/openclaw.md` update.
- `engramctl quickstart` optional prompt for AXI hooks.
- install bundle includes any new AXI hook templates.
- release notes explain MCP vs AXI clearly.

Acceptance:

- A new user can install native Helix, start Engram, enable Codex or Claude Code
  AXI context, and see storage paths without Docker.
- No release docs imply AXI replaces MCP.
- No hook is installed without explicit user action.

## Security And Privacy

Risks:

- Session hooks can leak private context into a harness unexpectedly.
- Capture hooks can store transcripts the user did not intend to persist.
- A compact output packet can still expose sensitive memories.
- A server-starting hook can create confusing background processes.

Controls:

- Read-only hook by default.
- Explicit `--capture` flag for write hooks.
- Budgeted context by default.
- Do not include raw memory bodies in home output.
- Respect existing auth config.
- Keep hook output explainable and inspectable.
- Provide uninstall or disable instructions in `engram axi hooks --help`.

## Release Blockers

Do not call AXI ready until all of these are true:

- Home packet, context, recall, and storage are implemented with tests.
- Default session-start hook is read-only and idempotent.
- `--json` output is stable enough for tests and scripts.
- Token budgets and timeouts are enforced.
- Offline behavior is deterministic and helpful.
- Hook installation has dry-run coverage.
- Codex and Claude Code paths are locally dogfooded.
- OpenClaw docs explain the current AXI posture.

## Recommended First Slice

The first implementation slice should be:

```text
Build `engram axi` as a read-only, REST-backed, budgeted CLI surface that
prints a compact home packet plus `context`, `recall`, and `storage` commands,
using internal TOON-compatible formatting and no hook installation yet.
```

This slice proves the agent-facing interface without touching user config or
capture behavior. Once the output feels right in real Codex sessions, hook
installation becomes a smaller and safer follow-up.
