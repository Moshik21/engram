# M5.2 Fresh-Agent Suite — baseline (gate B6)

**Date: 2026-07-22 (run against the live shell, http://127.0.0.1:8100)**
**Harness: `server/engram/evaluation/fresh_agent.py` · CLI: `engram battery --fresh-agent --against-live`**
**Grader: deterministic containment (same scorer as M5.1); `--judge ollama` is a stub for future local-LLM grading.**

## Method

Simulated FRESH-context agent with only Engram tools, fully local, no LLM:

1. One `get_context` at session start (amortized; counted in totals).
2. `recall` with the verbatim battery question (top-3).
3. On miss, ONE reformulated recall. Deterministic rule: drop stopwords /
   question words, keep remaining key noun-phrase tokens in original order
   ("what is the flip condition for usage ranking" → "flip condition usage
   ranking"). Identical reformulations are not retried.
4. "Answer" by extraction: containment of `expected_tokens` groups in the
   surfaced text (per-result containment, M5.1 scorer imported).

**Control arm** (what an agent gets with Engram degraded — the fallback
lane's packets): the same questions scored against the project files
`CLAUDE.md` + `docs/CURRENT_HANDOFF.md` (13,165 chars, 0 tool calls; a
group must land wholly inside one file). The pure no-tool control is 0/10
by construction and not reported separately.

## Baseline result

**Engram arm 7/10 vs project-file arm 6/10 → answerability lift +1** (B6:
memory-agent > no-memory/degraded control — PASS, narrowly).

- Tool calls: 15 total (1 get_context @ 2,334 chars + 14 recalls; 4/10
  questions needed the reformulation retry).
- Token cost (chars surfaced, engram arm): 27,648.
- Duration: 34.7 s.

| question | engram | projfile | calls | chars | reformulated query |
|---|---|---|---|---|---|
| flip-condition | HIT | miss | 1 | 384 | |
| recall-outage | miss | miss | 2 | 2001 | broke deep recall dogfood brain |
| ts-kill | HIT | HIT | 1 | 4133 | |
| north-star | HIT | HIT | 1 | 1407 | |
| deleted-phases | HIT | HIT | 2 | 9391 | consolidation phases deleted (hit on retry) |
| durable-lane | HIT | HIT | 1 | 384 | |
| fastembed-outage | miss | miss | 2 | 3545 | FastEmbed outage root cause |
| vector-write-path | HIT | miss | 1 | 1430 | |
| bm25-breaker | miss | HIT | 2 | 2022 | BM25 circuit breaker |
| founder-identity | HIT | HIT | 1 | 617 | |

## Reading

- Engram uniquely answers: flip-condition, vector-write-path (recent
  operational facts absent from project files).
- Project files uniquely answer: bm25-breaker — an answer-locality miss
  (M1.3 target).
- Both miss: recall-outage, fastembed-outage — the historical
  coarse-backfill locality problem (M1.3) plus one reformulation that a
  smarter agent would phrase differently; the deterministic rule is
  intentionally not tuned per-question.
- The +1 lift is honest but thin because the control's packets (CLAUDE.md)
  are unusually answer-rich for this battery; the lift should widen as
  M1.3/M2 land and should be re-run at each checkpoint.

## Reproduce

```bash
cd server && uv run engram battery --fresh-agent --against-live [--json] \
  [--project-file PATH ...] [--judge containment]
```

Lite CI mode (harness logic, injected tools, no server):
`server/tests/test_fresh_agent.py` (9 tests).

| Checkpoint | engram | projfile | lift |
|---|---|---|---|
| Baseline 2026-07-22 | 7/10 | 6/10 | +1 |
