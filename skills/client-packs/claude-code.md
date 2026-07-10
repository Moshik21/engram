# Engram — Claude Code / Claude Desktop client pack

Public surface only. Product metric: cold Decision continuity.

## Connect

```bash
engramctl connect claude-code --axi
# or
engramctl connect claude-desktop
engramctl doctor
```

## Every session

1. `claim_authority` when project file memory is present
2. `get_context` once before substantive answers
3. `recall` when people/projects/prior decisions appear
4. `observe` only for harness-invisible context
5. `remember` only for high-signal facts with proposals (0–5 / compact window)

## Session end

Invoke skill **engram-session-promote** (`skills/engram-session-promote/SKILL.md`).
Cap ≤5 promotions. Never dump recaps.

## Do not use as primary path

`search_entities`, `search_facts`, eval tools, full surface — operator/debug only
via `ENGRAM_MCP_SURFACE=operator|full`.

## AXI fallback (shell)

```bash
engram axi context --project "$PWD" --budget 800
engram axi recall "decision" --limit 5
```
