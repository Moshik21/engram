# Engram Standing Goal — the ledger and the discipline

**Status:** live, open-ended. Not a sprint. Not a checklist to burn down.
**Owner:** whichever harness agent is holding the session. Judgement is delegated (Konner, 2026-07-24).
**Opened:** 2026-07-24, after a day that closed 8 tickets and opened 14 more.

---

## 0. What this document is, and what it is not

It is **not** a backlog. The items below will churn — several will be wrong within a week,
and the ones that matter most are probably not written down yet.

It is the **ordering logic and the operating discipline**. Those are the durable parts. An
agent picking this up cold should be able to read §2 and §3, look at §4, and know what to do
next without asking. If you find yourself asking "which of these should I do?", §3 has failed
and you should fix §3 rather than guess.

**Judgement is delegated.** Do not stop to ask permission for ordinary engineering calls.
Ask only when §2 says to, or when the answer would change what gets built rather than how.

---

## 1. The standing objective

Engram is a fully-local memory layer whose real consumer is a coding agent mid-task. The
objective is not "close the tickets". It is:

> **Make Engram worth its tokens to the agent holding it.**

That bar is currently **not met**, and we have a number. In the fresh-agent measurement
(2026-07-24):

| | answers | chars surfaced |
|---|---|---|
| agent **with Engram** | 5 | 79,066 |
| agent with **just the repo files** | **6** | **13,275** |

Six times the context for one fewer answer. Alongside `agentRecallCount: 0` over a long
session — the agent never reached for it unprompted.

Everything in §4 is subordinate to that. If a work item cannot be connected to that number,
it is hygiene, and hygiene is fine, but do not let it crowd out §7.

---

## 2. The operating discipline (non-negotiable)

These are not style preferences. Each one was bought with a wrong conclusion.

### 2.1 A metric must be either accurate or absent, never plausible-but-wrong
Fourteen instruments were caught fabricating numbers on 2026-07-23/24 — see
`INSTRUMENT_AUDIT.md`. A wrong metric is strictly worse than a missing one: an absent metric
prompts investigation, a wrong metric *ends* investigation with a false answer. There is a
contract test (`tests/test_metric_honesty_contract.py`); it has a ratchet that may only
shrink. Do not add a ledger entry to silence it without fixing something.

**Corollary:** never trust a reported number without reading the code that computes it. The
"fewer relationships than entities" argument was built on a count extrapolated from a
10-entity sample.

### 2.2 Every mechanism needs a consumer, and every fix needs a probe that can fail
The dominant bug class is **computed-but-silently-inert**: code that runs and whose result is
discarded. It has appeared at least a dozen times, including in the mechanism the product is
named for (ACT-R scoring, applied to a channel whose budget is `0`).

So: ship a positive probe, then **prove the probe can fail** — neuter the mechanism, show it
red, restore, show it green, and report both outputs. A test that passes when the feature is
dead is worse than no test.

This has caught real failures twice. A metric-honesty scanner went *green and vacuous* when
neutered — only its canary noticed. A spreading probe passed with the mechanism completely
dead because the synthetic graph was too small to reproduce the overrun.

### 2.3 `engram battery` is not a gate
Two independent proofs: identical code has scored 0,0,0 warm and 2,1,1 cold in one session
and 3–5 in another; and `battery.py:98-114` requires all answer tokens inside **one** top-3
result, so a correct two-source answer scores MISS *by construction*.

Use `engram meter` (`evaluation/meter.py`). It reports per-question hit rate with variance,
attributes each answer to the rescue lane that produced it, derives the minimum N to resolve
a 1-answer effect, and **refuses to emit a headline when the run set cannot support one.**
Respect the refusal. Do not go looking for a number it declined to give you.

### 2.4 Any A/B must defeat the packet cache, and say how
`retrieval/packet_cache.py:117` keys on `group_id:scope:digest(topic):digest(project_path)` —
**no build, config, or arm component** — with a 300 s TTL and SQLite persistence that
survives restarts. Arm B can be served arm A's packets and report "no difference" for a
change of any size, with a clean low-variance number.

A tidy null result that did not explicitly defeat this cache is worthless. Vary queries
between arms, space beyond the TTL, or verify per-probe — and state which you did.

### 2.5 The live dogfood brain is not an A/B substrate
Measured: per-run totals `[4,4,4,5,5,7,7]`, sd 1.345, **N ≥ 105 runs/arm** to resolve one
answer — and the flips were a *regime change* (a rescue lane vanishing), not dispersion. No N
fixes a regime change. Use a lite planted-corpus clone, per `GRAPH_THESIS.md` §5.

### 2.6 Warm and cold are different machines; isolated and live are different machines
A build measured warm-only and shipped a change with a 2000× cold cliff. Config docstrings
have twice stated isolated-microbenchmark figures as live ones (a "29 ms p95" that measures
36–99 ms live; a "79–90 ms" that exceeds 300 ms under real recall). Say which regime you
measured, always.

### 2.7 Fixtures assembled from imagination inherit your blind spots
A prose-fragment predicate killed **6/6 live Correction entities** — including this project's
own north star — while its suite passed 47/47, because every "must survive" fixture entry was
Title-Cased and not one was a lowercase declarative. **Seed fixtures from live data.**

### 2.8 `UNKNOWN` beats a confident wrong answer
Especially about dead code. In this repo, static analysis mislabels load-bearing code as dead
at roughly **91:1** (§6). A wrong `ABANDONED` deletes a feature; an honest `UNKNOWN` costs a
follow-up.

### 2.9 Never `git add -A`; never a bare `git commit -a`
`git add <path>` is scoped but `git commit` commits the whole index. An agent swept another
workflow's staged files into its commit that way. **Always** `git commit -m "..." -- path/one
path/two`, and run `git status --short` first.

### 2.10 Do not run a measurement subprocess with a fake `HOME`
`~/.engram/.env` sets `FASTEMBED_CACHE_PATH`. An agent sandboxed `HOME`, failed to find the
embedding model, and reported a phantom outage. If you sandbox `HOME` you are measuring a
different machine.

### 2.11 Verify live config via `GET /api/knowledge/runtime`
Never by instantiating `EngramConfig()` from `server/`. The LaunchAgent sources
`~/.engram/.env` into real process env vars, which outrank every dotenv file in
pydantic-settings; a CLI run instead picks up `server/.env` at highest file precedence. Same
directory, same venv, **opposite answers** (`quiet` vs `standard`). `engram doctor` shares the
flaw and cannot detect it.

### 2.12 Do not run `tests/test_loop_ritual.py` casually
It shells out to `scripts/dogfood_loop_steward.sh`, which runs `engram continuity
--against-live` — a live latency benchmark **plus a POST write to the dogfood brain** —
whenever `/health` responds. It has already fired unintentionally and re-opened the BM25
breaker for 300 s mid-measurement.

### 2.13 One workflow owns the shell at a time
Concurrent workflows contaminated each other's latency numbers twice, cost a full measurement
cycle, and produced a regime change that a later instrument had to diagnose. Static/read-only
work may run in parallel. Anything touching the shell, `pipeline.py`, `config.py`, or
`activation/**` must be serialised.

### 2.14 Never delete working code until the replacement is verified and Konner has approved
Standing instruction. Applies with double force here — see §2.8.

---

## 3. Ordering rule

Work the highest tier that has an unblocked item. Within a tier, prefer the item that
unblocks the most other items.

- **T0 — Measurement integrity.** Anything that makes other work unmeasurable or produces
  false conclusions. *Nothing downstream is valid while a T0 is open.* This tier outranks
  everything including live bugs, because a live bug fixed against a lying instrument is not
  known to be fixed.
- **T1 — Actively harmful.** Live service degradation, data or result loss, crashes.
- **T2 — Silent-inert.** A shipped mechanism whose output nothing consumes. **These are
  unfinished features, not chores** — each one is capability already paid for and not
  collected. Historically the highest value-per-hour tier.
- **T3 — Capability and decision.** New work, and experiments that decide direction.
- **T4 — Strategic.** §7. Not a ticket. Outranks T3 on importance, but is gated on T0.

**Escape hatch:** if you have worked three items without the §1 number moving, stop and go to
§7. Do not grind the ledger.

### 3.1 Keep going. Do not wait to be told.

This is a **standing** goal, not a request queue. Judgement is delegated. When a piece of work
lands, **do not stop, report, and wait** — take the next item by §3 and start it. Reporting is
something you do *while* the next thing runs, not instead of running it.

Concretely:
- A workflow finishing is **not** a stopping point. It is an intake event (§5) and a trigger to
  start the next tier.
- Do not ask "shall I continue?" or "what next?". §3 answers both. If it genuinely cannot,
  that is a defect in §3 — fix §3 and keep moving.
- **Parallelise across tiers when the files do not collide.** T0 and T1 can run at once if they
  touch disjoint paths; §2.13 only serialises the *shell* and shared hot files, not all work.
- Push at boundaries (§6) as you go, so stopping is never required to make progress durable.
- The only legitimate stops: a §2 rule says ask; the work would be unsafe or irreversible; or
  §7 needs a decision that is genuinely Konner's rather than an engineering call.

*Added 2026-07-24 after the harness agent halted between workflows and had to be prompted
twice. The doc told it what to work on and never told it to keep working.*

---

## 4. The ledger

Live task IDs refer to the session task board; the numbers survive here even when the board
resets. Status as of 2026-07-24 evening.

### T0 — measurement integrity
| # | item | note |
|---|---|---|
| 29 | packet cache key has no build/config component | §2.4. Fix the key or add a measurement bypass; test that keys cannot collide across configs |
| 24 | effective config is launcher-dependent | §2.11. `engram doctor` shares the flaw |
| 27 | `config.hx.json` says 50 GB / ef_search 512; Rust hardcodes 20 GB / 768 | three copies of a file the runtime ignores. Make the loser impossible, not merely unused |
| 31 | **spreading completion is process-level bimodal: 0% / 6% / 100% / 100% on identical code** | `390866b`, four careful measurements. Cause is executor contention, not traversal cost: `recallSearch` runs 1082–1500 ms and saturates the 4-worker native executor, so a spreading read queues behind it — and `recallEntityAttributes` (independent stage, independent budget) times out at ~75 ms *in lockstep*. **The kill rig's `SPREAD_COMPLETION_MIN = 0.80` pre-flight therefore passes or fails on which server process it hits.** Makes ticket 2 (concurrency) a T0 dependency of ticket 3 |
| 32 | benchmark harness injects ~450 candidates where production injects 32 | `benchmark/methods.py:588-601` has no cap; `spread_candidate_injection_max` has one read site (`pipeline.py:1522`). A 12× pool divergence that breaks the exact property `d8bf60f` was built to establish. Any A/B through the harness is void until fixed |
| 33 | kill-rig provenance omits the two knobs that decide whether spreading traverses at all | `graph_kill_rig/runner.py:62-77` `_PROVENANCE_FIELDS` lacks `retrieval_spread_traversal_budget_ms` and `retrieval_spread_max_reads`; `max_reads=0 + budget=0` exactly reproduces pre-fix behaviour, so a result recorded today is indistinguishable from one recorded before the fix |
| — | *(done)* instrument audit + contract test | `14acb1e` |
| — | *(done)* `engram meter` | `e22163f` |

### T1 — actively harmful
| # | item | note |
|---|---|---|
| 25 | spreading starvation | **in flight** (`wq05nbnpx`). Fix works (10% → 86.7%) but leaked a recall budget into the offline dream phase (−44% reach) and lost rows on cold recalls |
| 22 | `GET /api/episodes?limit=200` takes the shell down | de-risked (~28 MB of ~89 MB), **not fixed** — needs a bounded page route in `schema.hx` |
| 19 | `recall_graph_gate` returns `None` in 0.11 ms after probe timeout | silent wrong answer to a bounded caller. Needs a typed miss (TIMEOUT vs NOT_FOUND) |
| 26 | `_resolve_episode_helix_id` full group scan on cache miss | 533 ms cold vs 0.263 ms warm — a 2000× cliff, and the gather that uses it is unbounded against a 4-worker executor |
| 34 | `slowest_read` is a running max that never decays — one outlier collapses the traversal | `bfs.py:89`, `ppr.py:192`, `actr.py:70`. Measured on a 0.5 ms/read store with a 50 ms budget: no outlier → 37 reads / 293 reached; **inject a single 25 ms read → 2 reads / 17 reached**, 94% of reach gone with 23 ms of budget unspent. Live per-read latency spans 0.1–150 ms, so this is not hypothetical. Silent — no gate notices |

### T2 — silent-inert (unfinished features)
| # | item | note |
|---|---|---|
| 20 | `if not candidates:` early-return skips Step 5.5 | on the **default** tier. Third independent cause of the reranker's false-negative history |
| 28 | `pool_total_limit` has zero read sites | shadowed on the exact recall-depth lever. Line 889 does the same pattern correctly |
| 7 | cue loop: registration fixed, but never recorded a use | producer needs a cue surfaced *and* echoed in the same process; ~8.4k episodes predate cues |
| 21 | some episodes unretrievable by their own verbatim text | index/embedding gap. No ranker can fix it. Caps the two-stage ceiling at 7/10 |
| 23 | graph consumer built, **flags OFF** | arming is a decision, not a chore. Blocked on 25 |
| — | `evidence_client_proposals_enabled` is a no-op kill switch | 11 references, all writes/assertions, zero reads; `extraction/apply.py` acts unconditionally |
| — | `apply_chat_recall_feedback` reachable only from the dashboard | so the "used" gate is dead for every MCP user specifically |
| — | `spread_candidate_injection_max=32` saturates on 100% of recalls | frozen guess sitting at its ceiling, discarding ~450 entities per request |

### T3 — capability and decision
| # | item | note |
|---|---|---|
| 3 | the graph kill experiment | rig built (`b14a6d4`). Arm C — *no graph, one extra recall round* — has never been run. Blocked on 25 + a planted corpus |
| 2 | M3 concurrency: parallelise episode/cue/chunk lanes | GIL confirmed released (`lib.rs:144`). Bound the fan-out; see ticket for measured constraints |
| 13 | P8 compaction-swap — memory as the agent's swap space | the only item that creates capability rather than repairing leakage |
| 12 | P7 presence curve — should session-start exist at all? | reframe the briefing as a design experiment, not a filter |
| 9 | P4 small-brain / fresh-install behaviour | everything was tuned on a pathological 9,343-episode corpus |
| 4 | RF flip, `ef_search` rebuild, `reindex_sweep` re-test, feeder flip, website sync | note the ef_search item targets a dead file — see 27 |

### T4 — strategic
See §7.

### Recorded disagreements (do not smooth these over)

Two careful verifiers, same commit, opposite conclusions. Both did real work; neither is
dismissible. Resolving these is worth more than another fix.

- **Does restored `ensure_fresh` buy anything?** V1 measured `is_bridge_edge` returning `None`
  on **358/358** live calls with `_assignments` empty, and gave a structural reason:
  `label_propagation` labels only `entity_ids` and restricts adjacency with `if nid in labels`
  (`community.py:155`), while BFS asks about *(node, **discovered** neighbour)* (`bfs.py:134`)
  — so the two can never meet. V2 measured `is_bridge_edge` resolving **10–23%** of calls and
  called the restore justified. If V1's mechanism argument holds, the commit restored the cost
  without the capability. **Unresolved.** Decide it with the mechanism, not another sample.
- **Did the completion win survive?** Now understood as ticket 31 — both were right about
  their own regime. Recorded here because the shape recurs: when two honest measurements of
  one system disagree, suspect a *regime*, not an error.

### Commit-message corrections owed
`390866b`'s message states the completion win "did not survive honest measurement (0/20)". That
is true of one regime and **false as a general claim** — 51/51 and 45/45 were independently
reproduced on the same commit. The code is right; the message overstates. Correct it in a
follow-up note rather than rewriting pushed history.

---

## 5. Intake protocol — how this stays alive

New findings arrive constantly, usually as a byproduct of fixing something else. That is the
normal case, not an interruption.

**When you find something, write it down immediately in the ledger**, in this shape:

```
| — | one-line claim | file:line · how you verified · what would falsify it |
```

Rules:
1. **Evidence at `file:line` or it does not go in.** "I think X is broken" is not an entry.
2. **State the tier.** If you cannot decide between T1 and T2, ask: *does this produce a wrong
   answer to a user, or merely fail to produce a right one?* Wrong answer is T1.
3. **Do not fix it now** unless it is T0 or blocking you. Write it down and continue. Scope
   creep is how a two-file fix becomes a twelve-file commit nobody can review.
4. **A finding that contradicts an entry here supersedes it.** Edit the entry, note the date,
   keep the old claim visible with a strikethrough or a "CORRECTED:" line. This document has
   already been wrong (see §6) and will be again; the correction history is worth more than a
   clean-looking page.
5. **Findings that invalidate past conclusions get flagged loudly**, not quietly patched. When
   the packet-cache trap was found, the honest statement was "this is a confirmed trap for
   future A/Bs *and does not by itself void the two-stage result* — that needs a specific
   check." Preserve that precision.

---

## 6. Push cadence and repo hygiene

**Push to GitHub at every natural boundary** — when a workflow lands, when a fix is verified,
before starting anything that touches the same files, and at the end of any session. Do not
accumulate. Nine commits sat unpushed for hours today; four workflows' uncommitted edits sat
in one shared tree, which is how work gets swept or lost.

- `env -u GH_TOKEN git push origin main`
- Explicit pathspec commits, always (§2.9).
- Commit messages state **what was measured**, not what was intended. If a change has no
  agent-visible effect at shipped defaults, **say so in the message** rather than implying a
  win.
- Preserve irreplaceable work outside the tree before risky operations. `.git/attic/` holds a
  complete recovered patch; `git diff` does **not** include untracked files, so a patch
  generated that way silently omits new modules.
- Never `git stash drop`/`pop` blind — a stash may be the only copy of another agent's work.

**Docs are part of the repo.** `INSTRUMENT_AUDIT.md`, `CODE_CENSUS.md`, `GRAPH_THESIS.md`,
`RECALL_PERFORMANCE_PLAN.md`, and this file are the institutional memory. Keep them corrected
rather than pristine.

**Known correction to my own earlier claim:** the code census found the repo has ~153 safely
deletable lines out of 290,829 (**0.05%**), against ~14,000 lines that looked dead to
competent static analysis and were load-bearing — and 45 unfinished features vs 13 abandoned
ones. **Engram does not have a dead-code problem; it has a finishing problem.** The remedy is
a consumer test at merge time (§2.2), not a prune. `CODE_CENSUS.md` §4 is a trap map of things
a future sweep would plausibly mistake for dead — consult it before deleting anything.

---

## 7. The strategic overhang

These are not tickets and must not be turned into tickets. They outrank the ledger on
importance and are gated on §2 being true.

**Q1. Why does Engram lose to the filesystem?** (§1). The hypothesis worth falsifying first:
the repo control wins because its content is *dense and current* while Engram's is *verbose
and stale* — which would make compression and recency the lever, not retrieval depth, and
would reorder most of §4. Tickets 10/11 (P5/P6) live here. **This is the next thing after
ticket 25.**

**Q2. Should the graph tier exist?** The consumer is built but unarmed; the deciding
experiment has a rig and pre-registered kill thresholds. Arm C is the question nobody has
asked: does the graph buy a *capability*, or a *round-trip the agent could take itself*?

**Q3. What is the extraction architecture, given no Ollama and no external API?** The only
intelligence sources are the harness agent (free input — it already has the content in
context) and embedded ONNX models. Working conclusion: **entities local, relations agent,
local proposes and agent disposes** (multiple choice, not free generation — it bounds
hallucination, tokens, and variance at once). Unresolved: agent cooperation cannot be forced,
and if agents skip an optional field the yield is zero *silently* — the exact bug class in
§2.2. Note the client-proposal path already runs, with a kill switch that does nothing.

**Q4. Should memory be the agent's swap space?** (ticket 13). The only item on the board that
creates capability rather than recovering it.

**Q5. Is `n=1` a problem?** Every claim about "what an agent wants" comes from one agent
introspecting in one atypical session. Treat §1 and §7 conclusions as provisional until a
second agent, ideally a weaker one, reproduces them.

---

## 8. Log

Append one line per landed change. Keep it terse; the detail belongs in the commit.

- **2026-07-24** — `d7c764e` pushed: graph-signal consumer (flags off), instrument audit +
  contract test, RF gate reader, briefing squatter filter **and its correction** (the first
  version killed 6/6 live Corrections), cue backfill tool, bounded episode hydration,
  `GRAPH_THESIS.md`.
- **2026-07-24** — `d8bf60f`, `b14a6d4`, `e22163f`: benchmark harness now exercises production
  scoring; three-arm kill rig with a pre-flight that refuses; `engram meter`.
- **2026-07-24** — `CODE_CENSUS.md`: 0.05% deletable, 91:1 false-to-true, 3.5:1 unfinished vs
  abandoned.
- **2026-07-24** — `390866b`: spreading traversal bounded, recall budget kept inside recall.
  Both blockers cleared and independently confirmed — offline reach restored **byte-identical**
  to pre-fix across six cases including read-order digest (the broken version had lost 78% of
  reads), and the cold-path row loss reversed (was −1 row; now +3/−1, totals 89→92, and the
  blocker's own timeout signature inverted). Nine mechanisms neutered one at a time, each
  confirmed RED — **two passed on the first attempt, so the tests were tightened until they
  failed.** AUDIT-14 defeated with a per-query nonce plus a positive control: repeating one
  exact query returned in 4.8 ms with no spread keys versus 2238.9 ms on its first run, proving
  the cache was live and the nonce is what avoided it. Opened tickets 31–34 (§4) and two
  recorded disagreements.
