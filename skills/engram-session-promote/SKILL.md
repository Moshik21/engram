---
name: engram-session-promote
description: End-of-session sparse promotion — propose at most 5 high-signal Decisions/Preferences/People for Engram remember(), never dump recaps.
version: 0.1.0
user-invocable: true
---

# Engram Session Promote (sparse)

Run this **at session end**, before context compaction, or when the user says
"promote what matters" / "save durable decisions."

Product metric is multi-agent continuity — not exhaustively storing the chat.

## Hard rules

1. **Cap: 0–5** `remember()` calls per **compaction window** (not multi-day session).
2. Prefer **0** when nothing is durable. Silence is fine.
3. **Never** promote session recaps ("what we did today", transcript dumps).
4. Types only: `Decision`, `Preference`, `Person`, `Correction`, `Goal`, `Commitment`.
5. You are the extractor: always pass `proposed_entities` + `proposed_relationships`
   with a verbatim `source_span` that appears in `content`.
6. Pass `model_tier` (opus/sonnet/haiku). After PreCompact, pass `compaction_id`
   if the harness exposed one (or rely on `~/.engram/promotion-window.json`).

## Procedure

1. Scan this conversation for **portable** facts that another agent would need
   cold (without a handoff doc).
2. Rank by durability:
   - Identity / explicit preferences / corrections
   - Product decisions with lasting consequences
   - Named people the user cares about
   - Active goals / commitments with owners
3. Keep the top **≤5**. Drop noise, temporary task status, and file paths that
   are only true inside this repo checkout.
4. For each kept fact, call **one** `remember()` with short `content` that
   contains the span:

```text
remember(
  content="<1–3 sentences including the exact decision text>",
  model_tier="sonnet",
  proposed_entities=[{
    "name": "<atomic decision or person name>",
    "entity_type": "Decision|Preference|Person|Correction|Goal|Commitment",
    "source_span": "<verbatim substring of content>",
    "summary": "<one-line why it matters later>"
  }],
  proposed_relationships=[{
    "subject": "<entity or project>",
    "predicate": "DECIDED|PREFERS|RELATED_TO|CORRECTS|COMMITS_TO",
    "object": "<other entity or decision name>",
    "source_span": "<verbatim substring of content>"
  }]
)
```

5. Read the response:
   - `committed_entities` / `committed_relationships` → promotion landed
   - `identity_core: true` → protected for merge/prune
   - `status: rejected` + `promotion_window_cap` → stop; remaining facts wait
     for the next compaction window
6. Optionally call `get_context` once and confirm ≥1 promoted Decision surfaces.
7. Tell the user briefly: **N promoted / M skipped** (no essay).

## What not to promote

- Long chat summaries or "session notes"
- Transient bugs already fixed in-repo
- Raw log lines, stack traces, or secrets
- Cadence / work-ticket noise unless the user explicitly wants it portable
- More than five items "just in case"

## When uncertain

Use `observe()` for bulk/uncertain context, or skip.  
**If uncertain whether it is durable — do not remember it.**

## Related

- Golden loop: `docs/GOLDEN_LOOP.md`
- Main memory skill: `skills/engram-memory/SKILL.md`
- PreCompact window reset: `hooks/pre-compact.sh`
