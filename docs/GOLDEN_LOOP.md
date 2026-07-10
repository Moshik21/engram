# Engram Golden Loop (Public Product Contract)

**Status:** Product surface freeze for multi-agent continuity  
**Last updated:** 2026-07-10

Engram’s public job is not LongMemEval. It is:

> A fresh agent on a different session surfaces **≥1 high-signal prior Decision**
> without opening a handoff doc.

## The only six tools agents need

| Tool | Role |
|------|------|
| `get_context` | Session start — durable Decisions/Preferences first |
| `recall` | When prior context could change the answer |
| `observe` | Passive raw capture (or let harness auto-capture) |
| `remember` | Sparse high-signal promotion (you are the extractor) |
| `intend` | Prospective memory / pinned context |
| `forget` | Repair wrong or stale facts |

Everything else is internal, experimental, or eval-only until this loop is boring.

## Three layers

1. **Passive capture** — harness `auto_observe` / `observe` (complete, cheap)
2. **Sparse promotion** — `remember` with `proposed_entities` + `proposed_relationships`, **0–5 per compaction window**
3. **Deliberate consolidation** — offline merge/adjudicate/prune (not always-on LLM ETL)

## Remember rules

- Types: Decision, Preference, Person, Correction, Goal, Commitment
- Always pass `source_span` that appears in `content`
- Pass `model_tier` and, after compact, optional `compaction_id`
- Reject session recaps (“what we did today”)

Example:

```text
remember(
  content="LongMemEval is not Engram north star. Continuity is the metric.",
  model_tier="sonnet",
  proposed_entities=[{
    "name": "LongMemEval is not Engram north star",
    "entity_type": "Decision",
    "source_span": "LongMemEval is not Engram north star",
    "summary": "Product metric is multi-agent continuity."
  }],
  proposed_relationships=[{
    "subject": "Engram",
    "predicate": "DECIDED",
    "object": "LongMemEval is not Engram north star",
    "source_span": "LongMemEval is not Engram north star"
  }]
)
```

Client-promoted high-signal entities are marked **identity_core** so prune/merge
pressure does not erase them casually.

## Compaction window

- Budget is **per compaction window**, not multi-day MCP lifetime
- Claude PreCompact hook stamps `~/.engram/promotion-window.json`
- Install: `bash hooks/install-precompact.sh` (or copy `hooks/pre-compact.sh`)

## Session-end sparse promote

Before you leave a long session, run skill **`engram-session-promote`**
(`skills/engram-session-promote/SKILL.md`):

1. Propose **0–5** portable Decision/Preference/Person facts (prefer 0 over noise)
2. `remember()` with proposals + verbatim `source_span`
3. Do **not** dump recaps

Optional harness hook `hooks/session-promote-nudge.sh` writes
`~/.engram/session-promote-nudge.md` on SessionEnd (installed via `engram hooks`).

## Continuity smoke (release gate)

```bash
cd server
uv run pytest tests/test_continuity_golden_path.py -v
# or
uv run engram continuity --smoke
```

## Non-goals (for now)

- Always-on LLM extraction over every episode
- LongMemEval as the product north star
- Growing consolidation phases without ROI
- More MCP tools before agents use these six well

See also: `docs/product/CONTINUITY_PRODUCT_PLAN.md`
