# Engram Continuity Product Plan

**Last updated:** 2026-07-10  
**North star metric:** *Fresh session / other agent surfaces ≥1 high-signal prior Decision without opening a handoff doc.*  
**Not the north star:** LongMemEval (optional regression only).

Use this checklist to track deliverables. Mark items `[x]` when verified in code + dogfood.

---

## Thesis

Engram is a **portable multi-agent continuity layer**, not chat RAG scoring.

Three layers:

1. Passive capture (cheap, complete)
2. Sparse agent promotion (expensive meaning, 0–5 / compaction window)
3. Deliberate consolidation (offline hygiene)

Public tool freeze: `get_context` · `recall` · `observe` · `remember` · `intend` · `forget` (+ `claim_authority` for authority routing).

---

## Slice 1 — Continuity polish

| ID | Deliverable | Status |
|----|-------------|--------|
| 1.1 | Client-promoted high-signal entities marked **identity_core** | [x] `apply.py` + promotion policy |
| 1.2 | Durable pack in **`get_context`** (graph before project files) | [x] `context_builder.py` |
| 1.3 | Deterministic **briefing** for durable facts (`format=briefing`) | [x] `_render_durable_briefing` |
| 1.4 | Parallel type listing for sub-second-ish session-start context | [x] `asyncio.gather` type probes |
| 1.5 | Dogfood: cold `get_context` shows strategy Decisions | [x] live REST verified 2026-07-10 |

---

## Slice 2 — Agent adoption / distribution

| ID | Deliverable | Status |
|----|-------------|--------|
| 2.1 | Public **GOLDEN_LOOP** contract doc | [x] `docs/GOLDEN_LOOP.md` |
| 2.2 | PreCompact hook in repo + install script | [x] `hooks/pre-compact.sh`, `hooks/install-precompact.sh` |
| 2.3 | `engram hooks` installs PreCompact | [x] `setup.py` `_HOOKS_CONFIG` |
| 2.4 | Skill documents sparse promotion + proposals | [x] `skills/engram-memory/SKILL.md` |
| 2.5 | Compaction window file read by `remember` | [x] `promotion.py` `load_external_compaction_id` |
| 2.6 | OpenClaw/Grok one-pager points at golden loop | [x] skill + GOLDEN_LOOP.md |

---

## Slice 3 — Quality without LongMemEval

| ID | Deliverable | Status |
|----|-------------|--------|
| 3.1 | Continuity smoke: promote 3 → cold context/recall | [x] `evaluation/continuity.py` |
| 3.2 | Pytest golden path suite | [x] `tests/test_continuity_golden_path.py` |
| 3.3 | CLI: `engram continuity --smoke` | [x] `__main__.py` |
| 3.4 | CI job for continuity suite (lite) | [~] snippet in `docs/product/CI_CONTINUITY_JOB.md` (local `ci.yml` dirty; needs PAT with `workflow` scope to push) |
| 3.5 | LongMemEval remains optional / not gate | [x] CI does not require LME |

---

## Slice 4 — Graph hygiene as product

| ID | Deliverable | Status |
|----|-------------|--------|
| 4.1 | Shared `is_decision_statement_noise` filter | [x] `promotion.py` |
| 4.2 | Rescue + context + entity search demote scrap | [x] recall_surface, context_builder, lookup |
| 4.3 | Identity-core on promoted Decisions reduces prune risk | [x] apply path |
| 4.4 | Operator clarity: open_work ≠ product success | [x] documented in this plan + CURRENT_HANDOFF |
| 4.5 | Prefer Person/Preference density in demos | [ ] dogfood practice (not code) — promote real People sparingly |

---

## Slice 5 — Later (after 1–4 are boring)

| ID | Deliverable | Status |
|----|-------------|--------|
| 5.1 | Multi-brain / multi-device sync design | [ ] design only until continuity CI stays green 2 weeks |
| 5.2 | Federated / policy intelligence | [ ] vision (`docs/vision/`) — no build until 5.1 |
| 5.3 | Sub-second durable context on 17G dogfood brain | [~] process cache 45s TTL + hard 1s pack budget in code; live p95 still optional dogfood |
| 5.4 | Session-end promotion subagent (propose ≤N) | [ ] optional harness skill |
| 5.5 | Dashboard: approved vs deferred framing | [ ] UI copy — open_work secondary metric |
| 5.6 | **Identity-core merge protection** | [x] block bad merges; prefer identity_core survivor |
| 5.7 | **Captain protect CLI** | [x] `engram captain protect "Decision name"` |

---

## Engineering already shipped (foundation)

| Item | Commit / area |
|------|----------------|
| Sparse promotion + window caps | `promotion.py`, capture_surface |
| Span-verified client proposal commit | client_proposals, commit_policy |
| Decision name validation (≤24 words) | resolver, apply |
| Explicit recall rescue + ranking | recall_surface |
| Project-file cache not a false win | `_packets_satisfy_explicit_query` |
| Auto-capture cue-only floor | worker_routing |
| Explicit recall budgets 4s / 1.5s search | config, budgets |
| Durable-context process cache + 1s hard pack budget | `context_builder.py` |
| Remember returns committed entity/relationship ids + identity_core | presenter + capture_surface |

---

## Release checklist (operator)

- [ ] `engramctl status` healthy
- [ ] `uv run engram continuity --smoke` PASS
- [ ] `uv run pytest tests/test_continuity_golden_path.py -q` PASS
- [ ] PreCompact installed (`hooks/install-precompact.sh` or `engram hooks`)
- [ ] Cold Grok/Claude: `get_context` shows ≥1 Decision entity packet
- [ ] Cold: `recall` strategy query hits LongMemEval / sparse promotion Decision
- [ ] No reliance on LongMemEval for ship/no-ship

---

## Metric dashboard (weekly dogfood)

| Signal | Good | Bad |
|--------|------|-----|
| Cold Decision hit rate | ≥1 correct Decision / week of dogfood | Only project files |
| False Decision scrap | 0 decision_statement in top-5 | Cadence scrap in top-5 |
| Promote rate | 0–5 / compact window | 0 forever or dump every turn |
| open_work_count | Background hygiene | Treated as product KPI |

---

## Non-goals (parked)

- [ ] Always-on LLM ETL over every episode
- [ ] LongMemEval as primary scoreboard
- [ ] Expanding consolidation phase count without ROI
- [ ] New MCP tools before golden six are habit

---

## How to mark progress

1. Implement / dogfood the row  
2. Flip `[ ]` → `[x]` in this file  
3. One-line note in `docs/CURRENT_HANDOFF.md` START HERE if operator-facing  

For slice 5 items, keep designs under `docs/product/` or `docs/vision/` until continuity CI is green on main for **14 days**.
