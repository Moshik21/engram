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
**CORRECTED 2026-07-24 (ticket #29 landed).** The key *used to be*
`group_id:scope:digest(topic):digest(project_path)` — no build, config or arm component,
300 s TTL, SQLite persistence across restarts — so arm B could be served arm A's packets
and report "no difference" for a change of any size with a clean low-variance number.

It is now `pc2:<fingerprint>:<group>:<scope>:<topic>:<project>`, where the fingerprint
covers the activation config (minus five cache-plumbing fields), the runtime mode, the
version and a digest of `retrieval/**` + `pipeline.py`. Foreign-fingerprint and pre-`pc2`
rows are never loaded from the sidecar, which matters because `recent_packets()` serves
the in-memory map without rebuilding a key.

**The rule survives the fix.** Arms that differ by neither config nor code — a planted
corpus, an uncommitted edit outside the digested tree — still share a fingerprint. So:
compare the `fingerprint` line the two arm reports print (identical = not isolated), set
`ENGRAM_PACKET_CACHE_NAMESPACE` per arm, or run `recall_packet_cache_enabled=False` and
confirm `bypassed=yes` on the report. **State which you did.** And repeats inside the TTL
are still replays, not samples — that half is unchanged.

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
| 29 | ~~packet cache key has no build/config component~~ | **DONE 2026-07-24.** Key is now `pc2:<fingerprint>:<group>:<scope>:<topic>:<project>`; fingerprint = whole `ActivationConfig` minus 5 cache-plumbing fields (TTL/path/max-entries/enabled/persistence — they change how long a packet is kept, never what it contains) + runtime mode + version + digest of `retrieval/**` + `pipeline.py`. Whole-config beat a "retrieval-relevant" allowlist on the asymmetry: 190/577 activation fields are read under `retrieval/`, an allowlist rots *silently*, and entries live 300 s so over-inclusion costs ≤1 TTL window of warmth while under-inclusion costs a clean wrong null. The subtler half was persistence: `recent_packets()` serves the in-memory map **without rebuilding a key**, so a key-only fix would still have leaked arm A's packets through the degraded-fallback lane — foreign-fingerprint rows are now never loaded, pre-`pc2` rows are purged. Verifiable, not hoped for: `stats.packetCache` reports `fingerprint`/`ttl_seconds`/`enabled`/`identity`, and `engram meter` reads it (server TTL replaces the compile-time guess — AUDIT-14's unshipped half), prints the fingerprint on every report, and accepts fast repeats **only** when it can verify the bypass. `tests/test_packet_cache_ab_isolation.py` (22 tests) proved red on 6 neuters; one neuter caught a **vacuous test in this same file** (two in-process caches cannot share regardless of the key) which was deleted rather than kept green |
| 24 | effective config is launcher-dependent | §2.11. `engram doctor` shares the flaw |
| 27 | ~~`config.hx.json` says 50 GB / ef_search 512; Rust hardcodes 20 GB / 768~~ | **DONE 2026-07-24.** All three copies pinned to the effective `queries.rs` literals with a leading `_authority` key; phantom `vector_config.db_max_size` removed. `tests/test_native_config_authority.py` fails on divergence in *either* direction, on the copies differing, on a key the Rust structs cannot honour, or on a fourth copy appearing. Proved red on: pre-fix JSON restored, single-copy edit-without-rebuild, `queries.rs` moved with JSON left behind, empty struct-allowlist, and a tautological `_effective()`. **Residual:** the Rust still does not read the JSON — closing that needs `make build-native`, deliberately deferred. Ticket 4's ef_search item is **not free and not reload-only**: it is `queries.rs` + all three JSONs + a rebuild |
| — | **8 of 16 Helix schema-contract tests had been skipping, silently** | **DONE 2026-07-24.** `tests/test_helix_schema_contract.py:13` pointed at `helixdb-cfg/.helix/dev/helix-repo-copy/helix-container/src/queries.rs` — a path that does not exist (the artifact is one dir up) inside a gitignored tree, so it could never exist in CI either. Every assertion about the generated PyO3 bindings was green and vacuous. Repointed at the tracked `native/helix-repo/helix-python/src/queries.rs`, both `pytest.skip` branches replaced with hard assertions; 16/16 now run, proved red by renaming `Entity.name` in the generated Rust. **Generalisation:** any `pytest.skip` guarding a *git-tracked* path is a latent instance |
| — | `HelixDBConfig.config_path` has zero read sites | `config.py:76` **[RV repo-wide]**, while `docker-compose.helix.yml:49` sets `ENGRAM_HELIX__CONFIG_PATH=/etc/helixdb`. The knob that names the directory holding the dead config is itself dead. Same shape as `CODE_CENSUS` B19. **T2** — falsified by finding any reader |
| — | `EmbeddingConfig.hnsw_*` reads as the HNSW knobs but governs Redis only | `config.py:117-119`. `hnsw_m`/`hnsw_ef_construction` have exactly one reader each — `storage/vector/redis_search.py:73-74`, `FT.CREATE`, FULL mode only. `hnsw_ef_runtime` has **zero** readers **[RV]**. `hnsw_ef_construction = 200` is the same stale 200 that was in `config.hx.json`: the declared surfaces were synced to each other, never to the runtime. **T2** — falsified by finding a native reader |
| 31 | **spreading completion is process-level bimodal: 0% / 6% / 100% / 100% on identical code** | `390866b`, four careful measurements. Cause is executor contention, not traversal cost: `recallSearch` runs 1082–1500 ms and saturates the 4-worker native executor, so a spreading read queues behind it — and `recallEntityAttributes` (independent stage, independent budget) times out at ~75 ms *in lockstep*. **The kill rig's `SPREAD_COMPLETION_MIN = 0.80` pre-flight therefore passes or fails on which server process it hits.** Makes ticket 2 (concurrency) a T0 dependency of ticket 3 |
| 32 | benchmark harness injects ~450 candidates where production injects 32 | `benchmark/methods.py:588-601` has no cap; `spread_candidate_injection_max` has one read site (`pipeline.py:1522`). A 12× pool divergence that breaks the exact property `d8bf60f` was built to establish. Any A/B through the harness is void until fixed |
| 33 | kill-rig provenance omits the two knobs that decide whether spreading traverses at all | `graph_kill_rig/runner.py:62-77` `_PROVENANCE_FIELDS` lacks `retrieval_spread_traversal_budget_ms` and `retrieval_spread_max_reads`; `max_reads=0 + budget=0` exactly reproduces pre-fix behaviour, so a result recorded today is indistinguishable from one recorded before the fix |
| — | **no native store can be created on this machine right now — every native fixture fails at construction** | `helix_native.HelixEngine(data_dir=<fresh dir>)` → `RuntimeError: Engine init failed: IO error: No space left on device (os error 28)`, with **65.5 GB free** (`diskutil info /System/Volumes/Data`). Reproduced 3/3 on empty scratch dirs and via `tests/test_native_entity_provenance.py`, which fails at the fixture, not at an assertion — so it is not a skip and does not announce itself as an environment problem. `lock.mdb` is written, `data.mdb` never is. **Cause UNKNOWN**, narrowed: `storage_core/mod.rs:78-88` maps `db_max_size_gb` GB and `helix-python/src/queries.rs:105` sets `Some(20)`, yet a **100 GiB sparse `ftruncate` in the same directory succeeds**, so this is not a plain sparse-allocation refusal and 20 GB alone should fit. Consequence: any lane needing a throwaway native brain is blocked and falls back to HTTP :6969 (also down), and the fresh-install smoke gate cannot run. Falsified by freeing disk and re-running the one-liner |
| — | *(done)* instrument audit + contract test | `14acb1e` |
| — | *(done)* `engram meter` | `e22163f` |

### T1 — actively harmful
| # | item | note |
|---|---|---|
| 25 | spreading starvation | **in flight** (`wq05nbnpx`). Fix works (10% → 86.7%) but leaked a recall budget into the offline dream phase (−44% reach) and lost rows on cold recalls |
| 22 | `GET /api/episodes?limit=200` takes the shell down | de-risked (~28 MB of ~89 MB), **not fixed** — needs a bounded page route in `schema.hx` |
| 19 | `recall_graph_gate` returns `None` in 0.11 ms after probe timeout | silent wrong answer to a bounded caller. Needs a typed miss (TIMEOUT vs NOT_FOUND) |
| 26 | ~~`_resolve_episode_helix_id` full group scan on cache miss~~ | **MOSTLY DONE 2026-07-24** (`562653c`). The cliff was never the 533 ms — it was that the 533 ms was **paid again on every recall and thrown away every time**. `wait_for` cancels the coroutine before its caching loop runs while the executor job completes anyway, so the pre-fix stage cached *nothing*: three consecutive runs, 2000-episode fixture, 40 ms stage budget → timed out 3/3, **0 episodes ever given graph signal**, 4/8/12 scans, 6337 ms of worker time, 100% discarded. Now: one scan per group per 5 s, shared by concurrent callers and `shield`ed, so recall 1 pays the cliff and recalls 2–3 complete with 60/60 covered off 1 scan / 747 ms. Also fixed: an id the scan does not find is remembered for the TTL — previously a deleted / cross-group / ticket-21-gap episode re-paid a full scan on **every** call, so "0.263 ms warm" never applied to it. `tests/test_episode_helix_id_cold_scan.py` (9 tests), red on 5 separate neuters. **STILL OPEN:** a single cold call still costs a full scan. The real fix is the missing `find_episode_by_episode_id` route — the Entity twin is `schema.hx:261` — which another lane owns; with it, this scan becomes a fallback. Keep as T2 until that lands |
| — | the fan-out bound in `episode_graph_signal` does **not** reduce worker burn — corrected claim | Four arms, isolated: adding `asyncio.Semaphore(8)` cut executor submissions 60 → 8 but left jobs actually run at **4 vs 4** and worker time at **3042 ms vs 3043 ms**, because `run_in_executor` cancels its own still-pending future when the gather wrapper is cancelled. The queued surplus was never going to run. What the bound *does* buy is head-of-line blocking on the shared executor — **ticket 31's mechanism**: one independent read submitted 5 ms into the stage waited **40.5 ms unbounded vs 5.2 ms bounded** (median of 5). Recorded because the wrong rationale was written into a code comment first and only measurement caught it. Peak in-flight is now counted, not inferred: `recall_episode_graph_signal_inflight_max` |
| 34 | `slowest_read` is a running max that never decays — one outlier collapses the traversal | `bfs.py:89`, `ppr.py:192`, `actr.py:70`. Measured on a 0.5 ms/read store with a 50 ms budget: no outlier → 37 reads / 293 reached; **inject a single 25 ms read → 2 reads / 17 reached**, 94% of reach gone with 23 ms of budget unspent. Live per-read latency spans 0.1–150 ms, so this is not hypothetical. Silent — no gate notices |

### T2 — silent-inert (unfinished features)
| # | item | note |
|---|---|---|
| 20 | `if not candidates:` early-return skips Step 5.5 | on the **default** tier. Third independent cause of the reranker's false-negative history |
| 28 | ~~`pool_total_limit` has zero read sites~~ **FIXED** | `candidate_pool.py:76` now derives the cap from `cfg.pool_total_limit * scale`; the shadow expression is deleted, so there is no loser to fall back to. Two defects, not one: the knob was inert (20/80/400/1000 all produced 85), **and** the shadow value was the *sum of the four pools it caps*, i.e. ≥ their union by construction — a depth cap that could only ever trim entity-query-unique candidates. Probe: `tests/retrieval/test_pool_total_limit_contract.py` (14 tests, ratchets over every key the function returns). **CONFIG EDIT OWED (T0 owns `config.py`):** set `pool_total_limit` default `80 → 85` to keep the DEFAULT lane byte-identical at scale 1.0. Routed lanes intentionally tighten (TEMPORAL 125→85, DIRECT_LOOKUP 115→85, ASSOCIATIVE 105→85 at the live 847-entity corpus): query-type multipliers redistribute depth, they no longer enlarge the cap. Unmeasured live — the ticket's own note says raising the pool measured NEGATIVE, so the direction is supported but n=1 |
| — | `pool_entity_query_limit` is the only pool limit that does not scale with corpus size | `candidate_pool.py:913` reads it via `limits.get("pool_entity_query_limit", cfg.…)`, but `compute_dynamic_limits` never returns that key — the dict branch is unreachable and the fallback fires 100% of the time. Not a lie (the knob works), but at 50k entities its six siblings grow 7× and this one stays at 20. Falsified by adding the key to the returned dict and seeing the entity-query pool widen. **T2** |
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
- **2026-07-24** — ticket 27 (T0): the declared HelixDB config now cannot silently disagree with
  the effective one. Three `config.hx.json` copies pinned to `queries.rs fn config()`, phantom
  `vector_config.db_max_size` removed, `_authority` marker required as the first key.
  `tests/test_native_config_authority.py` (8 tests) is the deliverable — proved red on five
  separate neuters, **one of which (a tautological `_effective()` reading the JSON instead of the
  Rust) passed the first canary, which was then tightened until it failed.** Adjacent: 8 of 16
  Helix schema-contract tests had been skipping on a nonexistent gitignored path since they were
  written; repointed at the tracked `queries.rs`, skips replaced with assertions, 16/16 run.
  No live measurement taken and no rebuild performed — the Rust still does not read the JSON.
- **2026-07-24** — `562653c`, ticket 26 (T1): the episode→helix-id scan is paid once per group per
  5 s instead of once per cache miss, and is `shield`ed so a stage timeout stops discarding it.
  The headline was not the 533 ms — it was that the pre-fix stage **timed out on every recall,
  gave 0 episodes graph signal, and cached nothing**, so the same 533 ms was re-burned forever
  (measured: 3/3 runs timed out, 12 scans, 6337 ms of worker time, 100% discarded; after: 1 scan,
  747 ms, recalls 2–3 complete at 60/60). Nine tests, red on five neuters. **A claim was caught
  and corrected before shipping:** the fan-out semaphore does *not* reduce worker burn (4 vs 4
  jobs, 3042 vs 3043 ms) — its measured benefit is head-of-line blocking, 40.5 ms → 5.2 ms, which
  is ticket 31's mechanism. Substrate was isolated with a stub transport and every assertion is a
  count, because **no native fixture could be built: engine init fails ENOSPC at 65.5 GiB free**
  (new T0 row). Still open: the missing `find_episode_by_episode_id` route, owned by another lane.
  Correction landed in `RECALL_PERFORMANCE_PLAN.md`: its M1 "free, reload-only" ef_search win
  targets a dead file and is neither free nor reload-only. AUDIT-15/16 added.
- **2026-07-24** — ticket 29 (T0): the packet-cache A/B trap is closed at the source. Key carries
  an identity fingerprint (config + mode + build + `retrieval/**` source digest); the sidecar no
  longer loads foreign-fingerprint or pre-`pc2` rows, which was the half that mattered because
  `recent_packets()` bypasses `build_key` entirely. `engram meter` now *verifies* rather than
  assumes: it reads the live TTL/fingerprint/enabled off `/api/knowledge/runtime/fast`, enforces
  the server's TTL, prints the fingerprint on every report, and skips the spacing refusal only
  when a bypass is confirmed in the measured process. Six neuters, six specific REDs — **one of
  them caught a vacuous test in my own new file** (two in-process caches cannot share regardless
  of the key), which was deleted rather than left green. **Not closed:** arms differing by neither
  config nor code still share a fingerprint — use `ENGRAM_PACKET_CACHE_NAMESPACE`. No live
  measurement taken (lane discipline); every claim is from source and unit tests.
