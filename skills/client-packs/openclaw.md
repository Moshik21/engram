# Engram — OpenClaw client pack

Primary skill: `skills/engram-memory` (published as `engram-brain`).
Promote skill: `skills/engram-session-promote`.

## Connect

```bash
engramctl quickstart --mode helix --install-openclaw --connect openclaw
# or
engramctl install-openclaw
engramctl connect openclaw
engramctl doctor
```

## Golden loop

Same as other clients: `get_context` → `recall` → sparse `remember` →
session-promote at end. See `docs/GOLDEN_LOOP.md`.

## AXI

OpenClaw hook installation is not assumed. Shell fallback:

```bash
engram axi --project "$PWD" --budget 800 --timeout 3
engram axi context --project "$PWD" --budget 800
engram axi recall "current task" --limit 5
```

AXI mirrors public tools only — no secret better surface.
