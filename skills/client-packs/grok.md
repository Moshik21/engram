# Engram — Grok Build client pack

Public surface only. Product metric: cold Decision continuity.

## Connect

```bash
engramctl connect grok-build
engramctl doctor
```

Default MCP surface is `ENGRAM_MCP_SURFACE=public` (golden loop tools only).

## Every session

1. `get_context` once before substantive answers
2. `recall` when prior context could change the response
3. `remember` high-signal Decisions/Preferences/People with proposals
4. `observe` for uncertain-value context only when harness auto-capture misses it

## Session end

Invoke skill **engram-session-promote** (`skills/engram-session-promote/SKILL.md`).
Prefer 0 promotions when nothing is durable.

## Hooks

Install PreCompact / session-promote nudge when available:

```bash
engram hooks   # or hooks/install-precompact.sh + session-promote-nudge.sh
```

## Do not

- Expand to full MCP “for more tools”
- Treat open_work as success/failure
- Use LongMemEval as the weekly score
