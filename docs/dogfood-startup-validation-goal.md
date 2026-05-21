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
| OpenClaw | MCP config written, docs match installed config, AXI fallback documented and smoke-tested where possible |
| Tool catalog | expected MCP tool count, `remember`, `observe`, `recall`, `get_context`, `bootstrap_project`, `claim_authority`, `route_question` exposed |
| Storage visibility | paths, sizes, counts, growth since startup, old data dirs called out before deletion |
| Failure handling | offline runtime, half-started runtime, stale PID file, port occupied, MCP probe timeout, duplicate client config |

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
