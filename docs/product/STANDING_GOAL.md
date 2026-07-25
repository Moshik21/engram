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
| — | **no native store can be created on this machine right now — every native fixture fails at construction** | `helix_native.HelixEngine(data_dir=<fresh dir>)` → `RuntimeError: Engine init failed: IO error: No space left on device (os error 28)`, with **65.5 GB free** (`diskutil info /System/Volumes/Data`). Reproduced 3/3 on empty scratch dirs and via `tests/test_native_entity_provenance.py`, which fails at the fixture, not at an assertion — so it is not a skip and does not announce itself as an environment problem. `lock.mdb` is written, `data.mdb` never is. **STILL LIVE 2026-07-24 evening** — reproduced again on a fresh scratch dir with **66.6 GB container free** (`RuntimeError: Engine init failed: IO error: No space left on device (os error 28)`), so it is not transient. New datum from the ticket-21 lane: a **plain `heed3` reader** (same crate, same `lmdb-master3-sys 0.2.5`) opens the *existing* 8.2 GB live store read-only with a 64 GB `map_size` without complaint, so the failure is specific to **env creation**, not to mapping a large size. **A LIVE failure with the same signature was caught the same evening:** `~/.engram/brain-status.json` after the 04:20–04:32 UTC mop window reads `"ok": false, "paused_shell": true, "error": "brain child exited rc=-10 without parseable result: ''"` — **rc=-10 is SIGBUS**, which on an mmap'd LMDB is the classic "touched a mapped page the filesystem could not allocate". So the ENOSPC is not only a fixture blocker: the cold brain is dying on it in production, silently (the shell restarted and `/health` went 000→200 without anyone being told a mop window was lost). **Cause UNKNOWN**, narrowed: `storage_core/mod.rs:78-88` maps `db_max_size_gb` GB and `helix-python/src/queries.rs:105` sets `Some(20)`, yet a **100 GiB sparse `ftruncate` in the same directory succeeds**, so this is not a plain sparse-allocation refusal and 20 GB alone should fit. Consequence: any lane needing a throwaway native brain is blocked and falls back to HTTP :6969 (also down), and the fresh-install smoke gate cannot run. Falsified by freeing disk and re-running the one-liner |
| — | *(done)* instrument audit + contract test | `14acb1e` |
| — | *(done)* `engram meter` | `e22163f` |

### T1 — actively harmful
| # | item | note |
|---|---|---|
| 25 | spreading starvation | **in flight** (`wq05nbnpx`). Fix works (10% → 86.7%) but leaked a recall budget into the offline dream phase (−44% reach) and lost rows on cold recalls |
| 22 | `GET /api/episodes?limit=200` takes the shell down | de-risked (~28 MB of ~89 MB), **not fixed** — needs a bounded page route in `schema.hx` |
| 19 | ~~`recall_graph_gate` returns `None` in 0.11 ms after probe timeout~~ **FIXED** | `7dda96d`. Refusals raise `GraphGateTimeoutError` (same contract as `native_transport.py:54` `NativeQueryError`, not a second vocabulary); `recall_graph_gate_refusals` counts them. Reproduced first, through the real pipeline with a real `asyncio.wait_for` expansion timeout: the gate answered `('value', None)` for an episode that **exists**, byte-identical to its answer for one that does not, and **3/3** documents reached the cross-encoder as empty text while the store held content for all three — coverage **0.0 through the gate vs 1.0 direct** — after which `scored.sort()` reordered real results by the rank of those empty strings. Caller audit, one decision each: rerank **degrades** (keeps pre-rerank order, `recall_reranker_skipped_probe_timeout`); temporal ×4, durable-slot reserve **stop + record**; entity-match and entity-query pools **pre-check + record**; working-memory pool stops expanding but **keeps its entities** (the blanket `except` would have returned `[]` and deleted the pool the falsy sentinel used to preserve); durable feeder serves partial ids and does not cache a refused listing. Four neuters proved RED, including "remove the Step 1.8 pre-check → `GraphGateTimeoutError` escapes `retrieve()`", which is what makes the audit test able to fail. Output unchanged at shipped defaults except the reranker. **Not measured live** — the shell was owned by another lane |
| 26 | ~~`_resolve_episode_helix_id` full group scan on cache miss~~ | **MOSTLY DONE 2026-07-24** (`562653c`). The cliff was never the 533 ms — it was that the 533 ms was **paid again on every recall and thrown away every time**. `wait_for` cancels the coroutine before its caching loop runs while the executor job completes anyway, so the pre-fix stage cached *nothing*: three consecutive runs, 2000-episode fixture, 40 ms stage budget → timed out 3/3, **0 episodes ever given graph signal**, 4/8/12 scans, 6337 ms of worker time, 100% discarded. Now: one scan per group per 5 s, shared by concurrent callers and `shield`ed, so recall 1 pays the cliff and recalls 2–3 complete with 60/60 covered off 1 scan / 747 ms. Also fixed: an id the scan does not find is remembered for the TTL — previously a deleted / cross-group / ticket-21-gap episode re-paid a full scan on **every** call, so "0.263 ms warm" never applied to it. `tests/test_episode_helix_id_cold_scan.py` (9 tests), red on 5 separate neuters. **STILL OPEN:** a single cold call still costs a full scan. The real fix is the missing `find_episode_by_episode_id` route — the Entity twin is `schema.hx:261` — which another lane owns; with it, this scan becomes a fallback. Keep as T2 until that lands |
| — | the fan-out bound in `episode_graph_signal` does **not** reduce worker burn — corrected claim | Four arms, isolated: adding `asyncio.Semaphore(8)` cut executor submissions 60 → 8 but left jobs actually run at **4 vs 4** and worker time at **3042 ms vs 3043 ms**, because `run_in_executor` cancels its own still-pending future when the gather wrapper is cancelled. The queued surplus was never going to run. What the bound *does* buy is head-of-line blocking on the shared executor — **ticket 31's mechanism**: one independent read submitted 5 ms into the stage waited **40.5 ms unbounded vs 5.2 ms bounded** (median of 5). Recorded because the wrong rationale was written into a code comment first and only measurement caught it. Peak in-flight is now counted, not inferred: `recall_episode_graph_signal_inflight_max` |
| — | **captured assistant responses are silently truncated at 2 000 characters, mid-word, with no marker anywhere** | `setup.py:861` writes the installed hook line `RESPONSE="${RESPONSE:0:2000}"`, then prefixes `[assistant\|<project>] ` — which is exactly why **129 live episodes are 2 019 chars** (2 000 + `[assistant\|Engram] `) and **419 are 2 033** (2 000 + a 33-char header). Measured by reading every Episode node in the live store: **454 episodes** land in the 1 985–2 048 **byte** window and **1 522** in 4 033–4 096 (the second cluster is `project_bootstrap.py:265` `raw_content[:max_chars]` at 4 000, source `auto:bootstrap`) — ~21 % of a 9 403-episode corpus sitting against a cap. `ep_293a18033a09`'s stored content ends `"…on lite/SQLite test brains those rea"`. Nothing records that a truncation happened: no ellipsis, no `content_truncated` flag, no counter — so the tail is not merely unretrievable, it is **not known to be missing**, and every "the episode contains X" scan is scanning a prefix. This is write-side data loss, so **T1**, not T2. Falsified by capturing a >2 000-char response and finding the tail in the store. **Not fixed here:** the effective file is `~/.engram/hooks/capture-response.sh`, a user config file, and raising the cap changes capture cost/latency — that is a product decision, not a chore |
| 34 | ~~`slowest_read` is a running max that never decays~~ **FIXED** | Estimator is now an EWMA seeded at 0 in `activation/read_budget.py` (one `ReadBudget`, three strategies). Reproduced first: 0.5 ms/read store, 50 ms budget, `max_reads=64` → clean 64 reads / 498 reached; **one 25 ms read → 6 reads / 34 reached** (93% of reach, 19.5 ms of 50 unspent); cold 30 ms FIRST read → 1 read / **0 reached**. After: **36 / 274** and **28 / 210**, both with <0.5 ms unspent. Clean store unchanged. Visibility: `recall_spread_budget_unspent_ms` + `recall_spread_reads` + `recall_spread_stop_<reason>`. Three neuters proved RED |

### T2 — silent-inert (unfinished features)
| # | item | note |
|---|---|---|
| 20 | `if not candidates:` early-return skips Step 5.5 | on the **default** tier. Third independent cause of the reranker's false-negative history |
| 28 | ~~`pool_total_limit` has zero read sites~~ **FIXED** | `candidate_pool.py:76` now derives the cap from `cfg.pool_total_limit * scale`; the shadow expression is deleted, so there is no loser to fall back to. Two defects, not one: the knob was inert (20/80/400/1000 all produced 85), **and** the shadow value was the *sum of the four pools it caps*, i.e. ≥ their union by construction — a depth cap that could only ever trim entity-query-unique candidates. Probe: `tests/retrieval/test_pool_total_limit_contract.py` (14 tests, ratchets over every key the function returns). **CONFIG EDIT OWED (T0 owns `config.py`):** set `pool_total_limit` default `80 → 85` to keep the DEFAULT lane byte-identical at scale 1.0. Routed lanes intentionally tighten (TEMPORAL 125→85, DIRECT_LOOKUP 115→85, ASSOCIATIVE 105→85 at the live 847-entity corpus): query-type multipliers redistribute depth, they no longer enlarge the cap. Unmeasured live — the ticket's own note says raising the pool measured NEGATIVE, so the direction is supported but n=1 |
| — | `pool_entity_query_limit` is the only pool limit that does not scale with corpus size | `candidate_pool.py:913` reads it via `limits.get("pool_entity_query_limit", cfg.…)`, but `compute_dynamic_limits` never returns that key — the dict branch is unreachable and the fallback fires 100% of the time. Not a lie (the knob works), but at 50k entities its six siblings grow 7× and this one stays at 20. Falsified by adding the key to the returned dict and seeing the entity-query pool widen. **T2** |
| — | a spreading traversal cancelled by the stage wall clock DISCARDS every completed read | `pipeline.py` Step 4 wraps the traversal in `asyncio.wait_for`, which cancels; a cancelled coroutine never reaches its return, so k−1 finished adjacency reads are thrown away when read k overruns. Found working ticket 34, which asked about the *first* read — that part is off by one (on read 1 there is nothing to discard) but the general case is real and **no estimator can fix it**: a surprise 150 ms read is unpredictable from a 0.5 ms history under any statistic. Now at least *visible* — `recall_spread_reads` survives the cancel while `recall_spread_reached` is 0, proved by `test_a_discarded_traversal_still_reports_what_was_discarded`. The fix is a caller-owned frontier (pass `bonuses`/`hop_distances` in, as `traversal_stats` already is) so a cancelled traversal degrades to shallow instead of to nothing. **Deliberately not smuggled into ticket 34's commit** — it changes the timeout path, which is the exact place `390866b` lost cold-recall rows. **T2** |
| — | the durable-feeder TTL cache can never hit across recalls | `candidate_pool.py:316` keys on `(id(graph_store), group_id)`, but `pipeline.py:587` wraps the store in a **fresh** `GatedGraphStore` every recall, so the key is a new object identity each time. Measured: two recalls against one underlying store → **2** `get_identity_core_entities` listings; the same wrapper reused twice → **1**, so the cache works and is merely keyed on something per-request. Two consequences: the 5 s TTL buys nothing (every recall re-lists), and a recycled CPython `id()` could serve one store's listing to another. Falsified by keying on the underlying store (`getattr(store, "underlying", store)`) and seeing the second listing disappear. Found while auditing ticket 19. **T2** |
| 7 | ~~cue loop: registration fixed, but never recorded a use~~ **THE PROCESS-BOUNDARY HALF IS FIXED; THE GATE IS NOT YET KNOWN TO MOVE** | The surfaced half lived in a **non-persistent in-process ring** (`retrieval/feedback.py` `SurfacedUsageBuffer`), so `record_observed_usage_events` could fire only when surfacing and echo landed in ONE shell lifetime — architecturally impossible for an agent session that spans restarts. Now backed by a SQLite sidecar (`retrieval/usage_surface_store.py`, `surfaced-usage.sqlite3`, next to the packet-cache sidecar), self-arming from whatever config reaches the hot path rather than from a startup call nobody would remember. **The echo mask and the dedup marks are persisted with the payload, and that is the load-bearing half**: a durable cue without its mask makes a post-restart verbatim parrot read as novel reuse, i.e. the durability fix would *manufacture* the number the gate reads. Definition chosen: *restated, not parroted* — the agent's own next capture contains a ≥2-token/≥10-char phrase from the cue at a position no 5-gram of the surfaced payload covers. Rejected: "explicitly acknowledged" (needs agent opt-in; the one emitter that ever did is dashboard-only, which is why the counter read zero) and bare token overlap (counts parroting). New bound, stated because durability removed the implicit one: a surfacing is eligible for **30 min** (= the dedup window, so a fired cue cannot refire without being re-surfaced). Second call site added: `remember` ran no citation scan at all, so the highest-signal turn was the one that could not record a use. Probe: `tests/test_usage_surface_durability.py` (20 tests) drives surface → **discard the buffer** → new buffer → counter, RED on 8 neuters — **two neuters (mask-not-persisted, capture-never-binds) passed the first probe and the tests were tightened until they failed.** **DO NOT read this as "the flip is unblocked."** Still unverified: that live recall surfaces `cue_episode` results at all, that an agent's next capture reuses a cue phrase inside the window, and that the helix `update_episode_cue` usage trailer round-trips live (no native fixture — ticket #35 ENOSPC). Also unchanged: ~8.4k episodes predate cues |
| 21 | ~~some episodes unretrievable by their own verbatim text~~ **DIAGNOSED — the premise was wrong** | **CORRECTED 2026-07-24.** It is **not** an index/embedding gap. Read-only LMDB forensics on the live brain (heed3 reader, `READ_ONLY\|NO_LOCK`, never wrote a byte; tool in the session scratchpad at `bm25read/`): all three "unretrievable" episodes are **present and correctly indexed**. `ep_293a18033a09` — Episode node present, `group_id="default"`, BM25 doc_length 317 / 223 unique terms, live `EpisodeVec` + `CueVec` + 2 `EpisodeChunk` vectors all `deleted=0`; replaying the real BM25 scorer over the real inverted index puts it at **global rank 2 / rank 1 among Episode-label docs** for its own opening sentence and **global rank 4 / episode-rank 1** for `"what is the flip condition for usage ranking"`. `ep_103d89337b0c` (dl 323) and `ep_bb61718b8e60` (dl 262) are the same shape. So hypotheses (a) missing/corrupt embeddings, (b) BM25 doc-id collision, (d) group/scope mismatch are **all falsified**. The mechanism is the RRF lane veto — next row. Ticket 21's own claim "no ranker can fix it" is right for the wrong reason: no ranker can, but the **fusion** can |
| — | **the BM25 lane cannot contribute a single document to any hybrid search — it is fetched 3× deep and discarded by construction** | `search.py` `search` (entities, :1600), `search_episodes` (:1684), `search_episode_cues` (:1788) are byte-identical in shape: `bm25_fetch_limit = limit * 3`, `vec_fetch_limit = limit` (whenever `group_id` is set, i.e. always in production), `_rrf_fusion(fts, vec, 0.3, 0.7)` then `[:limit]`. RRF weights rank *r* at `w/(60+r+1)`, so the **worst** vector-lane item scores `0.7/(60+limit)` and the **best** BM25-only item scores `0.3/61 = 0.004918`; `0.7/(60+limit) > 0.3/61` for every `limit ≤ 82`. The shipped episode-lane limit is `episode_retrieval_max(5) × 3 × 2` (`retrieval_strategy` defaults to `passage_first`, `pipeline.py:905`) = **30**, and `episode_retrieval_max` is `Field(le=20)` so the knob cannot reach the crossover. Lane *overlap* — the normal case — widens the band further. Net: the slowest native call in the stage (the one the BM25 circuit breaker exists for) runs at 3× depth and can only ever **reorder** what the ANN lane already found. Probe: `tests/retrieval/test_episode_search_lane_fusion.py` (18 tests), RED on two independent neuters (equal lane weights → 9 failures; a 3-slot BM25 reserve → 10). **Fix deliberately NOT shipped** — it changes recall composition on the hot path and this lane could not measure live. **T2** |
| — | **`SearchBM25<Label>(q, k)` takes the global top-k across EVERY node label and only THEN filters by label** | `native/helix-repo/.../ops/bm25/search_bm25.rs:56-95` — `s.search(txn, query, k)` is label-agnostic (one inverted index, one `total_docs`, one `avgdl` for the whole graph) and the label test is a `filter_map` **after** `results.truncate(limit)`. Generated caller: `helix-python/src/queries.rs:5787` `search_bm25("Episode", &data.query, data.k)`. Measured on the live index: **1 343 243 BM25 docs**, of which **1 182 202 (88 %) are `ConsolDecisionTrace`** and only **9 403 (0.7 %) are Episodes**; replaying `k=150` for `"flip condition usage ranking"` returned **60 Episodes out of 150 slots** (the rest EpisodeCue/Evidence/Entity/Consol*), and for the opening-sentence query 100/150. So 33–60 % of every BM25 fetch is spent on labels the caller discards, the loss is query-dependent and unbounded, and a query whose terms are common in consolidation audit rows could return **zero** episodes at any k. Second-order: `avgdl = 70.7` is set by those 1.18 M audit rows while Episodes average **339.8**, so every episode carries a ~4.8× BM25 length penalty *because internal bookkeeping nodes are in the user-facing text index*. Falsified by pre-filtering by label inside `search`, or by keeping Consol\* nodes out of BM25. **Blocked here — `native/**` is owned by another lane. T2** |
| — | `HNSWConfig::new` clamps `ef` to `10..=512`, so the declared `ef_search: 768` can never take effect | `native/helix-repo/.../vector_core/vector_core.rs:64-70` — `let ef = ef.unwrap_or(768).clamp(10, 512)`. `queries.rs:105` sets `ef_search: Some(768)`. Ticket 27 pinned the three `config.hx.json` copies to the `queries.rs` literal `768`, which is the *declared* value — the **effective** value is 512. Ticket 4's "ef_search rebuild" item should target 512, and the authority test should assert the clamped value, not the literal. **T0-adjacent (a declared number that is not the effective one); recorded here, blocked on `native/**`** |
| 23 | graph consumer built, **flags OFF** | arming is a decision, not a chore. Blocked on 25 |
| — | ~~`evidence_client_proposals_enabled` is a no-op kill switch~~ **FIXED** | `03cba85`. One read site (`GraphManager._client_proposals_accepted`); both places that asked "were proposals supplied?" now ask it. Reproduced first, end to end on a lite store: with the flag **False** an agent-proposed Decision still committed — `flag OFF must not commit an agent-proposed fact; found ['keep Engram fully local']`. **Two defects, not one:** the harness scoreboard computed `is_proposal_path` as *"the proposal path ran OR proposals were supplied"*, which stop being the same question once the flag can decline them — with the gate in but the scoreboard untouched, a **narrow** extraction reported `client_proposal_commits=1, client_proposal_rejects=2, client_proposal_share=0.5`. `tests/test_client_proposal_kill_switch.py` (7 tests, 3 controls), RED on two neuters — **the first probe passed on neuter 2 and had to be tightened until it failed.** Unchanged at shipped defaults (flag defaults True; every profile that sets it sets it True). Not measured live |
| — | `apply_chat_recall_feedback` reachable only from the dashboard | **BUILT + PROVEN, BLOCKED at the commit boundary.** The detector was never the missing part: `record_observed_usage_events` already runs on the observe fast path, already finds echoed reliance, and already writes a `used`-TIER ACCESS EVENT. It just never became a `used` INTERACTION, so the recall-need controller — rolling `used_count` **and** the adaptive-threshold learner at `control.py:347` that reads `interaction_counts["used"] + ["confirmed"]` — could never see it from a non-dashboard surface. Fix is `GraphManager.record_echoed_memory_usage` + 6 lines at the `capture_surface.py` seam; deliberately NOT routed through `apply_memory_interaction`, which would write the used-tier access event a **second** time (neuter C caught exactly that). Probe RED on three neuters, GREEN restored. **Blocked:** `ingestion/capture_surface.py` is being rewritten by the ticket-#7 lane; the hunk was applied, verified, and overwritten within ~10 minutes. Ready-to-apply handoff in `.git/attic/LANE3B-OWED-README.md`. **`dismissed` is declared unsupported here, not forgotten:** a capture has no bounded response to partition, so "surfaced and not echoed in this one observe" would fire on memories the agent used and simply did not capture — plausible-and-wrong beats nothing only in the wrong direction (§2.1) |
| — | **a client-proposed EDGE is dropped at materialization on the live profile, while the scoreboard counts it as a commit** | Four arms, isolated lite stores, `consolidation_profile=quiet` (the live value) vs `standard`. An edge whose endpoints the agent did **not** also propose as entities: `quiet` → **0 edges, even when both endpoint entities already exist in the graph**; `standard` → 2 edges. Mechanism: `apply.py:285` resolves endpoints only through this episode's `entity_map` (built from *this episode's committed entity candidates*) and never against the existing graph; the only rescue is `_auto_create_endpoint`, gated on `graph_auto_create_endpoints`, which is `False` by default (`config.py:2982`) and set True **only** by the `standard` profile (`config.py:3250`). The evidence layer commits the relationship (`client_proposal_span_verified`) and `client_proposal_commits` counts that DECISION, so the counter reads healthy while zero edges land. **This is the answer to "128 proposals, no new predicates" — see the row below.** Falsified by proposing an edge-only annotation on `quiet` and seeing an edge appear. **T1** (the metric is wrong, not merely missing) |
| — | on `standard`, the same rescue creates DUPLICATE endpoints instead of resolving them | Same four arms: `endpoints_in_graph=True, entities_proposed=False` on `standard` returned **4** entities for 2 names. `_auto_create_endpoint` (`apply.py:235`) mints a fresh `ent_<uuid>` unconditionally — it never calls `resolve_entity_fast`, which the entity path does. So the profile that makes edges land also silently forks every endpoint it rescues. **T2** |
| — | "no new semantic predicates from client proposals" is a DESIGNED INVARIANT, not a defect | `ALLOWED_CLIENT_PREDICATES` (`extraction/promotion.py:37`) is a closed 34-item frozenset; anything outside it is capped to 0.40 (`client_proposals.py:274`) and hard-rejected `predicate_not_allowed` (`commit_policy.py:203`). The agent **cannot** introduce a predicate the vocabulary does not already contain. The graph-thesis investigation was measuring the wrong thing: new predicates were never possible, so their absence is not evidence of an inert path. The real leak is the row above (edges dropped at materialization). Live scoreboard corroborates the shape: `client_proposal_rejects: 41` and `predicate_not_allowed_rejects: 41` — **every single reject was a predicate reject** |
| — | **the test suite writes to the operator's LIVE harness scoreboard** | `harness_metrics.py:91 harness_metrics_path()` falls back to `~/.engram/harness-metrics.json` and **nothing in `tests/conftest.py` overrides it** [RV]. Measured: running five extraction test files moved live `client_proposal_commits` **1769 → 1818**, `narrow_extractions` 1227 → 1241, `external_extractor_skipped` 1340 → 1374. So `engram harness` and the "128 client proposals in a 14-day window" figure are pytest-contaminated and cannot separate live agent traffic from fixtures. One-line fix: an autouse `conftest.py` fixture setting `ENGRAM_HARNESS_METRICS_PATH` (done locally in `test_client_proposal_kill_switch.py`; **repo-wide fix not applied — `conftest.py` is shared**). **T0** |
| — | `graph_kill_rig/runner.py:340` reaches `manager._search`, and the facade boundary test fails on it | `tests/test_graph_manager_facade_boundaries.py::test_runtime_code_does_not_reach_through_graph_manager_private_fields` is RED at HEAD, extra item `('engram/evaluation/graph_kill_rig/runner.py', 'run_rig', 'manager', '_search')`. Introduced by committed `93bbbf7` (ticket 33); `runner.py` is unmodified in the worktree, so this is **not** a working-tree artifact. A red boundary test trains everyone to ignore boundary tests. **T2**, owned by the kill-rig lane |
| — | `resolve_usage_surface_path` stringifies whatever it is handed and creates the directory | The ticket-#7 sidecar wrote `server/MagicMock/mock._cfg.recall_packet_cache_path/surfaced-usage.sqlite3` into the repo root when a test passed a `MagicMock` manager. It is not Mock-specific: any cfg whose path field is not a real path yields a garbage **relative** directory under CWD instead of a refusal or a degrade. Found incidentally; belongs to the ticket-#7 lane's uncommitted `retrieval/usage_surface_store.py`. **T2** |
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
§2.2. ~~Note the client-proposal path already runs, with a kill switch that does nothing.~~
**UPDATED 2026-07-24 (`03cba85`).** The kill switch now works. More importantly, the
"relations agent" half of that conclusion is **currently not delivered on the live profile**:
an agent-proposed edge commits as evidence and is then dropped at materialization unless the
agent *also* proposes both endpoints as entities in the same call, because endpoint resolution
only consults this episode's `entity_map` and the auto-create rescue is `standard`-only. And
the predicate allowlist is closed at 34 items, so "relations agent" can only ever re-use
existing vocabulary — never extend it. Both are design decisions that were never stated as
such. **Decide them explicitly before building more of Q3 on top:** (a) should endpoint
resolution look up the existing graph, (b) is a closed predicate set the intended ceiling on
agent-supplied meaning?

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
- **2026-07-24** — ticket 34 (T1): the predictive bail added in `390866b` was itself an instance
  of the bug class it shipped inside. Its cost estimate was a **running max that never decayed**,
  so one slow read refused every read after it. Reproduced before touching anything (0.5 ms/read
  store, 50 ms budget, `max_reads=64`): clean **64 reads / 498 reached**; one 25 ms read →
  **6 / 34** with 19.5 ms of 50 unspent; a cold 30 ms **first** read → **1 / 0**, 17.9 ms unspent.
  Replaced with an EWMA (α=0.3) seeded at 0.0 in `activation/read_budget.py` — one `ReadBudget`
  shared by BFS/PPR/ACT-R, which had carried three identical copies of the wrong answer. After:
  **36 / 274** and **28 / 210**, both spending the budget to within 0.5 ms; the clean store is
  unchanged. The argument is the cost asymmetry, written into the module: under-predicting costs
  ONE read of overshoot (the 75 ms stage cap over a 50 ms traversal budget is exactly that
  headroom), over-predicting costs the whole traversal. **A guarantee was deliberately traded and
  is documented, not smoothed over:** the running max never overshot after read 1; the EWMA
  overshoots by at most one read's cost. `test_spreading_reaches_graph.py`'s assertion was
  rewritten to the new bound rather than deleted. **Correction to the ticket:** its claim that a
  cold *first* read discards completed reads is off by one — on read 1 there are none; the discard
  is real for a slow read at position k>1 and no estimator can predict a surprise outlier, so that
  is a property of the stage's cancel-and-discard contract, not of the estimator (new T2 row).
  **Correction to my own first design:** `predicted_cost` is NOT the collapse signature — a
  healthy traversal stops that way constantly (63 reads / 314 reached / **0.07 ms** unspent). The
  discriminator is `recall_spread_budget_unspent_ms`; the collapse read 19.5 ms. Three tests
  asserting the stop reason flaked before that was understood. Three neuters, each RED:
  running-max estimator (3 failures), metric computed-and-discarded (3), publish-at-close instead
  of per-read (3). No live measurement taken — lane discipline; static + unit only.
- **2026-07-24** — `03cba85`, the client-proposal kill switch (T2): `evidence_client_proposals_enabled`
  now has a read site. Reproduced first — with the switch **off**, an agent-proposed Decision still
  committed to the graph. The second half was the scoreboard: gated behaviour with an ungated
  `is_proposal_path` made a **narrow** extraction report `client_proposal_share=0.5`. Probe RED on
  two neuters; **the first version passed on the scoreboard neuter and was tightened until it
  failed.** Adjacent, and larger than the ticket: the **live** harness scoreboard is
  pytest-contaminated (five test files moved `client_proposal_commits` 1769 → 1818 — no conftest
  override exists), and a four-arm isolated measurement answered the graph-thesis open question —
  client-proposed **edges are dropped at materialization on the `quiet` profile even when both
  endpoints already exist in the graph**, while `client_proposal_commits` counts them as commits,
  and "no new predicates" turns out to be a *designed* invariant of a closed 34-item allowlist
  rather than evidence of an inert path. Lane 3B (the `used` interaction from non-dashboard
  surfaces) is built and proved RED-on-three-neuters but **blocked** — its seam file is owned by the
  in-flight ticket-#7 lane; handoff in `.git/attic/LANE3B-OWED-README.md`. No live measurement.
- **2026-07-24** — ticket 21 (T2): **the premise was wrong and the ticket is now diagnosed.** The
  three "unretrievable" episodes are not missing from any index. Read-only LMDB forensics on the
  live 8.2 GB native store (a throwaway `heed3` reader opened `READ_ONLY | NO_LOCK`, so it never
  wrote a byte to `data.mdb` and never touched the reader table; the live shell kept running)
  replayed the real BM25 scorer over the real inverted index: `ep_293a18033a09` is **global rank 2 /
  rank 1 among Episode-label docs** for its own opening sentence and **global rank 4 / episode-rank
  1** for `"what is the flip condition for usage ranking"`, with a live `EpisodeVec`, `CueVec` and 2
  `EpisodeChunk` vectors, all `deleted=0`, `group_id="default"`. Hypotheses (a) missing/corrupt
  embeddings, (b) BM25 doc-id collision, (d) group/scope mismatch: **all falsified from live data.**
  The mechanism is an **RRF lane veto** in `search.py`: all three hybrid lanes fetch BM25 at `3 ×
  limit` but vectors at exactly `limit`, fuse at 0.3/0.7, and truncate to `limit` — and
  `0.7/(60+limit) > 0.3/61` for every `limit ≤ 82`, while the shipped episode-lane limit is **30**
  and `episode_retrieval_max` is capped at 20 by its own `Field`. So the slowest native call in the
  stage runs at 3× depth and **cannot introduce a single document the ANN lane missed** — it can
  only reorder. `tests/retrieval/test_episode_search_lane_fusion.py` (18 tests) pins it, RED on two
  independent neuters (equal lane weights → 9 failures; a 3-slot BM25 reserve → 10), GREEN restored
  with the source byte-identical (sha checked). **The fix was deliberately not shipped:** it changes
  recall composition on the hot path and this lane was not allowed to measure live.
  Three findings opened along the way, each with `file:line`: captured assistant responses are
  **silently truncated at 2 000 chars mid-word** (`setup.py:861`; 454 live episodes against the
  2 048-byte wall, 1 522 against 4 096 from `project_bootstrap.py:265`) with **no marker, flag or
  counter** — write-side data loss, so **T1**; `SearchBM25<Label>` takes a **global** top-k over all
  1 343 243 BM25 docs (88 % of them `ConsolDecisionTrace`, 0.7 % Episodes) and filters by label only
  *after* truncation, measured at 60/150 usable slots on the live index, with `avgdl` dragged to
  70.7 by audit rows against an Episode mean of 339.8; and `HNSWConfig::new` **clamps `ef` to 512**,
  so ticket 27's pinned `ef_search: 768` is declared-but-not-effective. Ticket 35's ENOSPC
  reproduced again at 66.6 GB free — but the same crate opens the *existing* store fine, so the
  fault is in env **creation**, not in mapping a large size.
