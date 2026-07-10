# Multi-client golden loop verify

**Goal:** Claude Code, Grok Build, and OpenClaw all land on the **public** MCP
surface + session-promote habit — no secret better path.

Public tools (freeze): `get_context`, `recall`, `observe`, `remember`, `intend`,
`forget`, `claim_authority` (+ onboard: `bootstrap_project`, `get_runtime_state`).

AXI mirrors the same loop (`engram axi context|recall|observe|remember`) — see
`server/engram/axi/__init__.py` `PUBLIC_AXI_COMMANDS`.

---

## One-shot connect matrix

| Client | Connect | Hooks / promote | Surface env |
|--------|---------|-----------------|-------------|
| Claude Code | `engramctl connect claude-code` | `--axi` + PreCompact/SessionEnd | `ENGRAM_MCP_SURFACE=public` (default) |
| Grok Build | `engramctl connect grok-build` | skill pack + session-promote | public |
| OpenClaw | `engramctl connect openclaw` + `install-openclaw` | skill `engram-brain` + promote skill | public |
| Codex | `engramctl connect codex --axi` | AXI session-start hooks | public |

Installer defaults `ENGRAM_MCP_SURFACE=public` in env. Full/operator is opt-in.

---

## Session-promote habit (every client skill pack)

1. Ship / install `skills/engram-session-promote/SKILL.md`
2. Main memory skill must link to it (OpenClaw: `skills/engram-memory`)
3. Optional: `hooks/session-promote-nudge.sh` on SessionEnd
4. Cap **0–5** `remember()` per compaction window

Client-specific skill stubs live under `skills/client-packs/`:

- `claude-code.md` — Claude Code / Claude Desktop
- `grok.md` — Grok Build
- `openclaw.md` — OpenClaw (points at engram-memory skill)

---

## Verify checklist (per client)

```bash
# 1. Surface freeze
export ENGRAM_MCP_SURFACE=public
# MCP tool list must include get_context/recall/observe/remember
# and must NOT include search_entities / get_evaluation_report

# 2. Continuity smoke
cd server && uv run engram continuity --smoke

# 3. Doctor golden loop (helix installs)
engramctl doctor   # adds --require-golden-loop for helix mode

# 4. Cold path
# In a FRESH session: get_context → expect ≥1 Decision if brain has promotions
# recall "strategy decision" → Decision hit

# 5. AXI parity (shell clients)
engram axi context --project "$PWD" --budget 800
engram axi recall "decision" --limit 5
# No engram axi search_entities (does not exist)
```

---

## Anti-patterns

- Pointing agents at `ENGRAM_MCP_SURFACE=full` “because more tools”
- Using AXI as a privileged write path with different semantics
- Skipping session-promote and dumping full transcripts via `remember`
