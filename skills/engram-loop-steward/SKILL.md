---
name: engram-loop-steward
description: Silent Loop Steward — sense memory quality and debt, apply short-lived allowlisted shell adjustments, optional mop, verify. Harness subconscious; not a user-facing chat ritual.
version: 0.1.0
user-invocable: true
---

# Engram Loop Steward (harness subconscious)

**You are not the consolidator.** Engram’s shell still owns sleep. You bias it.

**Silence:** Do not narrate phases, budgets, or mop details to the end user unless
they are dogfooding Engram itself. Prefer zero user-visible text.

## When to run

- Session end / SessionEnd nudge file present
- `should_mop` or deferred climbing
- `get_context` / `recall` miss a known Decision or feel degraded/timeout
- Server/MCP flaky after warmup

**Rate limit:** ≤1 steward pass per session (or 4h). Prefer a **subagent**.  
**Do not** run every message or narrate to the user.

## Preferred procedure (one-shot)

```bash
# Ensure shell is up first (prerequisite)
engramctl status || engramctl start

# Single operator command — sense → propose → apply if needed
engram loop steward-once --format json
# AXI alias:
engram axi steward-once --format json

# Dry-run (no write):
engram loop steward-once --dry-run --format json

# Optional mop after apply:
engram loop steward-once --mop --budget 200 --format json
```

If output has `healthy_noop: true` or `regime: healthy` → **stop**.  
If `applied: true` → optional verify with `engram continuity --against-live`.

**Public MCP:** never call loop/mop tools. Operator MCP may use `loop_steward_once`.

## Manual procedure (if one-shot unavailable)

### 1. Sense

- Lived: note whether `get_context` / `recall` returned durable Decisions
- Operator:
  ```bash
  engram hygiene report --format json
  engram continuity --against-live   # if server up
  ```

### 2. Classify regime

| Observations | Regime |
|--------------|--------|
| deferred high / should_mop / continuity OK | `debt_heavy` |
| auto-observe flood / cue_only climbing | `intake_heavy` |
| continuity miss or latency/degraded | `latency_degraded` |
| server/MCP down | `offline` |
| all quiet | `healthy` → **stop** (no apply) |

### 3. Propose LoopAdjustment JSON

```bash
engram hygiene report --format json > /tmp/debt.json
engram loop propose-from-report --debt-json /tmp/debt.json --format json
```

Allowlisted only. Example `debt_heavy`:

```json
{
  "version": 1,
  "group_id": "default",
  "regime": "debt_heavy",
  "reason": "hygiene should_mop; deferred multi-k; continuity pass",
  "ttl_hours": 12,
  "created_by": "harness:steward",
  "max_risk": "low",
  "budgets": {
    "evidence_drain": 2000,
    "already_exists": 500,
    "stale_reject": 500,
    "cue_hygiene": 500,
    "adjudication_limit": 400
  },
  "phase_boost": ["evidence_adjudication", "prune"],
  "phase_defer": ["graph_embed", "dream"],
  "intake": {
    "auto_extract_min_score": 0.85,
    "pattern_junk_reject": true
  },
  "expected": {
    "continuity_must_pass": true
  }
}
```

Hard rules: `max_risk` must be `low`; `ttl_hours` 1–48; never invent phase names;
never disable `pattern_junk_reject`; never touch public MCP tools.

### 4. Apply

```bash
engram loop apply --file /path/to/adj.json --format json
# offline dogfood:
engram loop apply --file adj.json --skip-continuity-check
engram loop status --format json
```

### 5. Act (optional)

```bash
engram hygiene mop --budget 500   # shell uses steward budgets when higher
```

Never reimplement merge/infer/dream in the agent.

### 6. Verify

```bash
engram hygiene report --format json
engram continuity --against-live
engram loop status
```

If continuity regresses, `engram loop clear` and stop.

### 7. Expire

TTL snaps back automatically. Do not leave permanent `.env` snowflakes.

## Surfaces

| Use | Surface |
|-----|---------|
| Sense / apply / mop | Operator CLI (`engram loop`, `hygiene`, `continuity`) |
| User chat | Public golden loop only — no loop tools |
| Subagent | Preferred so main chat stays “conscious” |

## Related

- Protocol: `docs/design/loop-steward-protocol.md`
- Build plan: `docs/design/loop-steward-build-plan.md`
- Self-regulation body: `docs/design/memory-loop-self-regulation.md`
- Session promote (conscious meaning): `skills/engram-session-promote/SKILL.md`
