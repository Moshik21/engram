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
  engram axi observe --stdin --source codex,Capture explicit user-approved notes
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
engram axi                       GET  /health, /api/knowledge/runtime, /api/storage
engram axi context               GET  /api/knowledge/context
engram axi recall                GET  /api/knowledge/recall?q=...
engram axi observe --stdin       POST /api/knowledge/observe
engram axi remember --stdin      POST /api/knowledge/remember
engram axi bootstrap             POST /api/knowledge/bootstrap
engram axi storage               GET  /api/storage
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
engram axi --project "$PWD" --budget 800 --timeout 3
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

Still needed:

- Preserve the trace fixture shape in docs/tests so future hook schema changes
  cannot silently regress Codex or Claude Code startup evidence.
- Re-run the strict doctor gate after installer or hook changes, because the
  proof is intentionally metadata-only and tied to the local hook config plus
  `~/.engram/axi-hook-runs.jsonl`.

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
