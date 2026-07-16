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

Hooks: PreCompact + SessionEnd promote nudge (`engram hooks`).
You are the extractor — `remember` with `proposed_entities` + verbatim
`source_span`. Measure habit with `engram harness scoreboard` (promote_rate).

## Steward (silent)

**Rate limit:** ≤1 pass/session. Prefer subagent. **No user narration.**

When SessionEnd nudge exists, or debt/continuity feels wrong:

```bash
engramctl status || engramctl start
engram loop steward-once --format json
# AXI: engram axi steward-once --format json
```

**Brain windows:** the local server pauses briefly every ~2h while the cold
brain mops (usually skipped or under a minute). Connection-refused on
127.0.0.1:8100 is transient — retry shortly. `engramctl start` waits for the
brain lock, and `steward-once` senses debt over HTTP, so the ritual above is
safe while the server runs.

If `healthy_noop` / regime healthy → stop. Else skill `engram-loop-steward` for
details. Public MCP stays golden-loop only — never mop/loop on public.

## Hooks

Install PreCompact / session-promote nudge when available:

```bash
engram hooks   # or hooks/install-precompact.sh + session-promote-nudge.sh
```

## Do not

- Expand to full MCP “for more tools”
- Treat open_work as success/failure
- Use LongMemEval as the weekly score
