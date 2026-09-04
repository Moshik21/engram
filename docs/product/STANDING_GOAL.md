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
| — | **CORRECTION to 31: the split is driven by QUERY TEXT, and one arm of it is a probe that lies.** `7229f89` | `31` blamed executor contention. A case-only live A/B (same six words, one letter's case changed, warm helix shell) reproduces an all-or-nothing split with no process change at all: `summarise the engram brain schedule` → `graph_expand` 3.1 ms, gate open, **spread reach 197**; `summarise the Engram brain schedule` → `graph_expand_timeout` 77.3 ms, gate ARMED, **spread reach 0** and 4 stages skipped. Same for `anthropic`/`Anthropic` (reach 497 → 0). Mechanism: `_extract_query_terms` returns `[]` for an all-lowercase query so the stage reads nothing, but ONE capitalised token sends it into a serial cascade under a 75 ms cap; the timeout is a `PROBE_TIMEOUT_KEYS` entry, so it refuses every secondary graph read for the rest of the request. `recall_stats` finished in 220–373 ms and the primary search in 7–14 ms on every one of those, so **the store was never the problem**. Third case, worse: `tell me about helixdb storage limits` extracts ZERO terms, issues ZERO reads, and still recorded a 76.5 ms `graph_expand_timeout` — a starved event loop reported as an over-budget graph. Fixed at the source (bounded fan-out + own deadline + `graph_expand_starved` when zero reads were ISSUED). **31 must be re-measured controlling for capitalisation before its process-level explanation is trusted.** `recallEntityAttributes` was not "timing out in lockstep" at HEAD — it records `*_skipped_probe_timeout` in 0 ms, i.e. it is the shared gate, not a shared executor |
| — | **the nonce used to defeat the packet cache is itself an A/B trap** | Appending `"(ref L2A)"` to 13 live queries made `_extract_query_terms` yield the term `"L2A)"`, which sent `find_entity_candidates` into the store and armed the graph gate on **9/9** pipeline recalls. The same queries without a nonce: **0/10**. So a per-query nonce perturbs graph expansion, term extraction, query classification and BM25 — any lane using one to satisfy §2.4 is measuring a different query class. Prefer `ENGRAM_PACKET_CACHE_NAMESPACE` or distinct natural queries. Falsified by finding a nonce form that `_extract_query_terms` ignores (all-lowercase, no punctuation-attached capitals) |
| 32 | ~~benchmark harness injects ~450 candidates where production injects 32~~ **FIXED `7f21150` (2026-07-24)** | `benchmark/methods.py:588-601` has no cap; `spread_candidate_injection_max` has one read site (`pipeline.py:1522`). A 12× pool divergence that breaks the exact property `d8bf60f` was built to establish. Any A/B through the harness is void until fixed **Struck 2026-09-04:** `7f21150` moved the cap into one shared rule, `retrieval/spread_injection.py:26-47` `select_spread_injections` (returns the pre-cap `discovered` count, so a bound cap is visible instead of reading as "32 was enough"); both call sites import it — `benchmark/methods.py:40,604` and `retrieval/pipeline.py:1927-1930` — so the harness and production cannot diverge on the cap by construction. |
| 33 | ~~kill-rig provenance omits the two knobs that decide whether spreading traverses at all~~ **FIXED `93bbbf7` (2026-07-24)** | `graph_kill_rig/runner.py:62-77` `_PROVENANCE_FIELDS` lacks `retrieval_spread_traversal_budget_ms` and `retrieval_spread_max_reads`; `max_reads=0 + budget=0` exactly reproduces pre-fix behaviour, so a result recorded today is indistinguishable from one recorded before the fix **Struck 2026-09-04:** `graph_kill_rig/runner.py:69-87` `_ARM_CRITICAL_FIELDS` lists `retrieval_spread_traversal_budget_ms` and `retrieval_spread_max_reads`, and the provenance record is the WHOLE `ActivationConfig` via `_config_snapshot` (`runner.py:279`, written per arm at `:423`), so a curated list going stale can no longer hide a knob; `_UNKNOWN_CRITICAL_FIELDS` (`:93-100`) raises at import on a renamed knob instead of recording `None`. |
| — | ~~**no native store can be created on this machine right now — every native fixture fails at construction**~~ **RESTATED 2026-09-04: native env creation is INTERMITTENT on this machine — 3/3 failed 2026-07-24, 2/2 passed 2026-09-04 (scratch native store, raw engine + product path); cause UNKNOWN; positive creation probe kept** | **CORRECTED 2026-09-04:** the blocker as stated is falsified — a fresh-dir `helix_native.HelixEngine(data_dir=<scratch>)` created a store **2/2** today, once through the raw engine and once through the product transport path (`ASSESSMENT_2026-09-04.md:77`), on ~50 GiB free. So "every native fixture fails" is no longer true on this box. Nothing below explains why it failed 3/3 in July and passes now: the map size is unchanged at 20 GiB (`helix-python/src/queries.rs:105` `db_max_size_gb: Some(20)`), so the **cause stays UNKNOWN** and the row stays OPEN as intermittent rather than closed. The **positive creation probe** (open a `HelixEngine` on an empty scratch dir, assert `data.mdb` appears) is kept as the pre-flight for any lane that needs a throwaway native brain — run it before trusting the lane in EITHER direction, because a 3/3 failure and a 2/2 success have now both been observed on the same machine with the same binary. The 2026-07-24 record follows, unchanged: `helix_native.HelixEngine(data_dir=<fresh dir>)` → `RuntimeError: Engine init failed: IO error: No space left on device (os error 28)`, with **65.5 GB free** (`diskutil info /System/Volumes/Data`). Reproduced 3/3 on empty scratch dirs and via `tests/test_native_entity_provenance.py`, which fails at the fixture, not at an assertion — so it is not a skip and does not announce itself as an environment problem. `lock.mdb` is written, `data.mdb` never is. **STILL LIVE 2026-07-24 evening** — reproduced again on a fresh scratch dir with **66.6 GB container free** (`RuntimeError: Engine init failed: IO error: No space left on device (os error 28)`), so it is not transient. New datum from the ticket-21 lane: a **plain `heed3` reader** (same crate, same `lmdb-master3-sys 0.2.5`) opens the *existing* 8.2 GB live store read-only with a 64 GB `map_size` without complaint, so the failure is specific to **env creation**, not to mapping a large size. **A LIVE failure with the same signature was caught the same evening:** `~/.engram/brain-status.json` after the 04:20–04:32 UTC mop window reads `"ok": false, "paused_shell": true, "error": "brain child exited rc=-10 without parseable result: ''"` — **rc=-10 is SIGBUS**, which on an mmap'd LMDB is the classic "touched a mapped page the filesystem could not allocate". So the ENOSPC is not only a fixture blocker: the cold brain is dying on it in production, silently (the shell restarted and `/health` went 000→200 without anyone being told a mop window was lost). **Cause UNKNOWN**, narrowed: `storage_core/mod.rs:78-88` maps `db_max_size_gb` GB and `helix-python/src/queries.rs:105` sets `Some(20)`, yet a **100 GiB sparse `ftruncate` in the same directory succeeds**, so this is not a plain sparse-allocation refusal and 20 GB alone should fit. Consequence: any lane needing a throwaway native brain is blocked and falls back to HTTP :6969 (also down), and the fresh-install smoke gate cannot run. Falsified by freeing disk and re-running the one-liner |
| — | *(done)* instrument audit + contract test | `14acb1e` |
| — | *(done)* `engram meter` | `e22163f` |
| — | **the 9/14 meter read was mostly the brain reading this session's own work log** | 2026-09-04 (T0, Step 1 of the 'proceed in order' plan). A 14-question rig drawn from June–August episodes this session never discussed (`tests/rigs/recall_meter_rig_clean_2026-09.json`, source ids recorded) scores **3–4/14 warm** on the live corpus and **[4,3,3]** on the pre-work snapshot; the old rig's 9/14 included questions whose answers this session had captured. Treat 3–4/14 as the honest number for unseen questions. The meter's own guard held: it marked the runs `unresolved` (31 % / 17 % degraded probes) and my first read-out of `mean 1.667` had skipped the refusal. Every cold run (first after a ≥5 min gap or a restart) scored **0** — all 14 questions fell to `context_packet_fallback` at 2.5–4.5 s — so the instrument now needs a warm/cold split per run, not a mean. |
| — | **the store's working set no longer fits this machine: swap 3.3/4 GB, shell 3.3 GB resident after restart, an INDEXED single-key lookup at 29 s** | 2026-09-04. `nativeQueryStats` maxima of 33 s on entity BM25 and 29 s on `find_entity_exact_name_indexed` in the same window are one process-wide stall, not a slow index. Any A/B run on this machine while another 8 GB store is mapped (my side server on :8101 did exactly that) is invalid — §2.6 now has a third machine: *swapping*. |
| — | **a unit test wrote its cursor into the live `~/.engram/hygiene-state.json` (group `g`, episode `ep1`)** — **FIXED `c833c83`** | 2026-09-04. A mop wiring test without the module's `engram_home` fixture; the state path is `ENGRAM_HOME`-relative and nothing set it. `tests/conftest.py` now gives every test a throwaway `ENGRAM_HOME` (autouse); the live file was repaired by hand. Any live number read after a test run on this box before today may have been steered by test state. |
| — | ~~**the vector backfill's presence probe called 7 017 of 8 654 episodes missing; a native census counts 8 074 vector rows and the ANN serves them**~~ **FIXED `773755b`** | 2026-09-04. `get_episode_embeddings` returns `data: []` for every row written before the data-field projection, so 'has floats' ≠ 'has a row'. The drain re-embedded 50 already-indexed episodes per window (no duplicates: the row count moved only by the 44 the machinery de-index removed). Presence is now row-based. |
| — | **CORRECTION to the 'cold-brain LaunchAgent silent since 2026-08-15' row above: launchd fires it every 2 h; the runs were `skipped: on battery power` by design, and my kickstart collided with my own store-opening census** | 2026-09-04. The verbatim plist command ran clean from a terminal (`Brain run ok`). Row stays for the record; the open question is only whether the battery skip is the right default on a laptop that is mostly on battery. |

### T1 — actively harmful
| # | item | note |
|---|---|---|
| 25 | ~~spreading starvation~~ **LANDED at HEAD; row was stale** | 2026-09-02 audit: `retrieval_spread_max_reads` (work bound), `spread_candidate_injection_max`, `recall_spread_reached` probe and the EWMA `ReadBudget` are all in `activation/spreading.py` / `read_budget.py`; `spread_activation` documents that only latency-owing callers pass bounds and `consolidation/phases/dream.py:88` passes none, so the dream-reach leak is closed by construction. The `spreading-fix-wip` stash predates these commits and is superseded (left in place, not deleted). Not re-verified: "lost rows on cold recalls" — needs a cold-store measurement, §2.6. |
| 22 | ~~`GET /api/episodes?limit=200` takes the shell down~~ **FIXED** | `38129d1` (2026-09-02). Five keyset page routes in `schema.hx`; the store falls back to the full scan with a WARNING when the native binary predates them — which was the live state until `make build-native` (28 s per listing, measured). |
| — | **the local embedding model, not the store, was the shell's memory: one 40k-char episode = 7.4 GB** | `embeddings/provider.py` `FastEmbedProvider.DEFAULT_MAX_CHARS` (`1cae312`, 2026-09-02). Measured standalone: engine open 10 MB, full episode list 536 MB, but nomic-embed (8192-token window, ONNX arena never returned) at 64 × 4.5k chars = 11 GB / 76 s. The 962-item startup outbox drain took the shell to a 17 GB footprint on 16 GB RAM, 14.3 GB swap, native queries timing out at 20 s, then Jetsam SIGKILL (launchd exit -9, no log line). Capped at 2000 chars (= the chunk window) / batch 16: 2.1 GB, plateau 3.4–4.5 GB live. Falsifier: a shell footprint above ~5 GB with the cap on. |
| — | **explicit recall was degraded on 10/10 live calls: 3.75 s of pre-pipeline probes on a 4 s wall, pipeline cancelled at 250 ms, 2.5 s rescue after** | `retrieval/recall_surface.py` (`d71dba7`, `65f4230`) + `retrieval/pipeline.py` (`6b8bc3c`), 2026-09-02, 10-query REST probe, same queries before/after. Before: 0/10 ok, median wall 6698 ms, every item from the packet cache. After: 10/10 ok, median 1665 ms, 50 items from the pipeline. Four levers: the rescue wrapper doubled every cap it was handed; the second pre-pipeline durable rescue was a duplicate of the first and is **deleted (§2.14 — Konner to confirm)**; probes that time out back off 30 s; the entity-count probe (sqrt(n/1000) pool scale, ×1.06 here) cost 1.5 s per recall and is memoised on the store. Falsifier: `recall_stats_timeout` or `durable_entity_first` ≥ 400 ms on consecutive live recalls. |
| — | **background indexing starves recall: the drain's HNSW inserts queue ahead of the recall's reads on the shared 4-worker pool** | `ingestion/capture_service.py` + `retrieval/recall_activity.py` (`989a259`). Episode materialisation 1.2 s per recall under the drain for lookups that are ms standalone. The drain now awaits `wait_idle()` (bounded 2 s) before each item. Partial: a continuous stream of recalls still lets it proceed every 2 s by design. Chunks are embedded in one batched call (`e1afd09`): drain rate ~5/min → ~18/min. |
| — | **16 cue nodes SIGBUS the process on `update_cue`; the cold brain has died on the first of them 203 times since 2026-08-15, and the shell would die the same way on feedback to any of them** | `scratchpad/bad_cues.txt` (ids), `cue_sweep.log`: no-op `update_cue` over all 10 745 cues, 10 729 ok, 16 crash rc=138 (SIGBUS), reads of the same nodes succeed. Every bad node carries 7–80 KB of span JSON (overflow pages); good nodes reach 10.7 KB, so size alone is not it. Reproduced standalone with `helix_native` alone (no brain code). Compacting copy under test. **OPEN.** Falsifier: the copy updates cleanly (→ overflow-chain damage in the source file) or crashes identically (→ a decode bug in the update path). |
| — | **every label-scan read leaks its scanned bytes natively: `find_cue_by_episode` +8.5 MB per call, `find_episode_by_episode_id` +25–50 MB per call, linear, never released** | `scratchpad/leak_probe.py` 2026-09-02, standalone `helix_native`, sequential calls: 300 cue lookups = +2.5 GB, 300 episode lookups = +10 GB; Python tracemalloc traced 68 MB, so it is Rust/LMDB-side. Both routes compile to `n_from_type(..).filter_ref(..)` full label scans (`queries.rs:5601`, `:8027`); the `INDEX episode_id` declared on `N::EpisodeCue` is never used (`n_from_index` is imported and unused). Consequences: the cold-brain mop child reached 11 GB in 75 s doing nothing but cue lookups (`fp6.log`: +100–500 MB per 5 s window with only `find_cue_by_episode` in flight) and is SIGKILLed by Jetsam before it finishes a window — the *second* reason the brain has never completed; the shell performs the same lookup on every capture. **LEAK SITE FOUND, ROUTE MITIGATED, RUST FIX OPEN.** `helix-db/src/utils/properties.rs:95` writes heap-owning `Value::String`s (`protocol/value.rs:1496` `.to_owned()`) into the bump arena; `Node`/`ImmutablePropertiesMap` are `Copy` with no `Drop`, and bumpalo never runs destructors, so every decoded node's string bytes are orphaned when the arena drops. Applies to every `n_from_type` scan. `a8542fa`: `find_cue_by_episode_indexed` (`N<EpisodeCue>({episode_id: ep_id})`, the first route in the schema to use `n_from_index`) — 305/305 sampled cues resolve identically, 0.1–0.4 ms vs 79 ms, footprint flat over 300 calls. **The cold brain then completed its first window since 2026-08-15 (ok, 18.6 s, 12 evidence items adjudicated).** Still leaking per decoded row on every other scan route (`find_episode_by_episode_id` has no INDEX to switch to: `N::Episode` indexes group_id/status/session_id only, and adding one does not backfill). Falsifier for the mitigation: a shell footprint that still climbs ~8 MB per capture. |
| — | ~~16 cue nodes SIGBUS…~~ **REPAIRED LIVE 2026-09-03** (`engram backup ovfix --apply`: 391 headers, write probe OK, quarantine retired, brain window ok afterwards; pristine backup at `~/.helix/backups/engram-native-dogfood-axi.20260903T135931Z`) — history: | `7c64b0d`. Overflow pages of the 16 records carry stale headers (page numbers and txn stamps 87k–1.6M against a store counter of 48k); LMDB master3 `IS_MUTABLE` (`mdb.c:1124`) treats a stamp ≥ current txn as the txn's own dirty page and overwrites it in place inside the read-only map. Cause of the stale headers: master3's compacting copy rewrites leaf/branch pages and copies overflow pages verbatim (`mdb.c:11147` region), then restarts the counter — the same defect reproduces on every record of a fresh `engine.compact()` output, so **`engram backup compact --apply` would swap in an unwritable brain (T1)** and its "verified before swap" gate reads counts only, never writes. Lifting the meta txnid is NOT a safe repair on its own: the stale headers also carry the chain page count LMDB trusts on free, and the patched copy became unopenable (`MDB_PAGE_NOTFOUND`) after a handful of writes. `~/.engram/cue-quarantine.txt` blocks writes to the 16 episode ids in `update_episode_cue` (brain and shell). Proper repair = logical rebuild of those 16 records (delete/recreate through routes that never touch the stale chain), or a compaction that renumbers overflow headers. |
| — | ~~the BM25 circuit breaker was persisted OPEN for the whole session~~ **FIXED 2026-09-03** (`f9fe125`) | Mechanism: every half-open probe was cancelled by the lane timeout before it could report (cancelled young = no information), so nothing ever closed it, while native BM25 measured idle answers in 18–700 ms — and it is the lane that FINDS the fact for the meter's questions (`Thompson sampling` 2/2 rows with the token, `BM25 circuit breaker` 3/3, `usage_ranking_enabled` 3/3, `FastEmbed outage` 2/2) where the vector lane returns other projects' noise (recent 200 episodes: 12 Engram, 78 firstmate, 51 shielded-bid …). The probe now runs shielded and records its true wall time; budget 500→1000 ms, cancel strike 100→400 ms, retry 300→60 s. Live: `CLOSED (probe=851ms)` at 08:56:55 after a day open. Side finding: pytest runs persist breaker state for tmp brains into the user's real `~/.engram/bm25-breaker-state.json` (dozens of stale keys) — harmless, sloppy, not fixed. |
| — | ~~recall dropped found episodes at the 300 ms materialise cap~~ **FIXED** (`2d864d5`) | ticket 34's per-id native route is a full Episode label scan (~80 ms measured) and ran FIRST, so the helix-id cache never helped; several per recall on 4 workers = 250–300 ms and `recallMaterializeEpisodeTimeout`. Now cache → one shared warm scan per 300 s (was 5 s) → per-id route only for ids the warm does not know. Live: materialise 251 ms → 0 ms; warm recall 183 ms end to end. |
| — | ~~a one-row preflight page replaced the deep pipeline~~ **FIXED** (`4d34853`) | `why was Thompson sampling removed` → one bare cue row, `fast_preflight_hit`, pipeline never ran. A preflight hit must be a full page to short-circuit; a short page answers only when the pipeline is empty. `engram meter --project-path` added: the agent always sends it, the meter never did. |
| — | **the meter moved 1/12 → 6/11 (stable 6,6,6 over three runs) on 2026-09-03, all from plumbing, none from ranking changes** | `engram meter --against-live --runs 3 --project-path /Users/konnermoshier/Engram`. Sequence, each verified live between runs: recall no longer drops materialised episodes (`2d864d5`); a one-row preflight page no longer replaces the pipeline (`4d34853`); the BM25 breaker's probe survives the caller's timeout (`f9fe125`); BM25 lanes drop function words, 10–50× cheaper per lane (`9616362`); a caller's cancellation only strikes past the lane's own budget (`8f10b71`) — that last one was the flapping: every OPEN was `cancelled=True` at 402 ms from the preflight's 400 ms timeout. Meter runs with project_path because the agent always sends it. Cache-cleared, 301 s-gapped rerun (the honest number): **5, 6, 6 of 14, mean 5.67, sd 0.58**, breaker CLOSED throughout, after `5909467` (no entity scan for the health probe; entity ids warm once). Three of the remaining misses were answered in ~15 ms by a full preflight page (`fast_preflight_hit`) that the meter scores 0 — the preflight is a latency hedge that still out-ranks the pipeline by arriving first — but a single run with `RECALL_FAST_PREFLIGHT_ENABLED=0` scored 5/14 (north-star lost, recall-outage gained), so it is net-neutral and NOT the blocker. **Coverage check (T0, standalone over 11,977 episodes + 10,747 cues + 951 entities): every expected-token group has matching rows except `type_rank` (0) and two of the three alternatives of one pair** — e.g. 8 episodes contain both `Thompson` and `noise`, 13 contain `usage_ranking_enabled` and `False`, 92 contain `ts_` and `deleted` — and none of them reaches top-3. The remaining gap is RANKING/FUSION on a clean instrument (ticket 21's "RRF vetoes the BM25 lane" is the prime suspect: the keyword lane finds the rows, the fused top-3 does not carry them). |
| — | **ticket 21 proven on a real miss, and the reserve was inert by construction** | `why was Thompson sampling removed`, traced lane by lane offline (2026-09-03): BM25 held the answer at ranks 3–4; the reserve seated those rows on the 15-row page at positions 14–15 with fused score 0.30 against 0.65–1.0 for eleven vector rows that were ALL other projects' captures; `pipeline.py` keeps the top `episode_retrieval_max` by score → the seat never reached a consumer. Fixed (`11dab64`): seats are positions inside the cut with that position's score (the old test pinned the inert contract and was replaced); `recall_other_project_multiplier` (0.6) demotes episodes whose capture header names another project — episodes have no project field, the hook header is the marker. Offline the pipeline's top-3 then carried the answer. Live, the preflight lane was answering first in 15 ms with a full page that lacked it, so recall is now pipeline-first with the preflight as a latency hedge (`2ae5c65`). Meter (gapped, cache-cleared): 6,6,6 → run 1 of the next pass **7/14**, then runs 2–3 collapsed to 1: the BM25 breaker RE-OPENED on a probe that took **7 047 ms** — a native keyword call queued 7 s behind something on the 4-worker pool. Three collapses today share this shape; capture indexing is ruled out (15 events all day). Always-on per-route native queue/exec counters now ship on `GET /api/knowledge/runtime` (`nativeQueryStats`, `605001e`) so the next collapse names its route. |
| — | **the mid-run collapse is COLD PAGES, not queueing: an 8 GB store on a 16 GB machine that swaps loses its posting-list pages in a five-minute idle gap** | `nativeQueryStats` + `vm_stat` sampled through meter runs 11–12 (2026-09-03): Python-side queue ≈ 0 even on the per-store executors; single native calls (`search_episodes_bm25`, `find_entities_exact_name`, `search_entities_bm25_filtered`) reached 1.3–4.7 s with 435 MB of page-ins per 20 s, and the same queries answer in 5–90 ms warm. The breaker then opened on those cold calls, its pre-armed state carried across restarts, and every half-open probe (the lane's first call after a rest = cold) "failed" at 1.3 s — the lane could never warm because it was off: meter12 1,1,1. Fixed the cycle (`probe grace 3×`, this commit); the shared executor (`594b43d`) and the identity-core memo stand on their own evidence. After the probe grace (`17427a1`): **meter 7, 7, 7 of 14 (sd 0)**, breaker CLOSED at 10:57 and never re-opened across three gapped, cache-cleared runs — the first time the keyword lane survived a full pass. Day's trajectory on the same instrument: 1/12 → 5.67 → 6 → 7 → 7,7,7 again with exact-name by INDEX (`efd375f`, flip-condition 4.2 s → 1.3 s) and the durable-first probe exact-only (`e2c4a3e`). Then **8, 8, 8** (`8a21bc4`): tracing the three remaining singles lane by lane showed the vector lane's top 15 for `flip-condition` were ALL this operator's short chat prompts ("is it running again", "keep going on all."), which embed close to any short question and carry no facts; `recall_short_episode_floor_chars` (300) weighs a candidate by its body length below the floor (min 0.3). flip-condition 0 → 0.67, recall-outage 0 → 0.33. Conjunction split for pairs measured offline: 0/3 groups whole and split on all four — the pairs are bound by their singles (`vector-write-path`, `recall-outage`), not by the join. Remaining: those two singles (vocabulary mismatch: the answer rows use different words than the question; bootstrap doc episodes dominate when the project name is in the query). **Open:** the store's working set does not fit beside everything else on this Mac; levers are the ONNX arena (shell footprint 0.8→4.5 GB), page-locality of BM25 postings, and what else the machine runs. |
| — | **cue hygiene leaves demoted cues searchable: blank rows reach the agent** | `consolidation/cue_hygiene.py` sets `cue_text=""` on demotion and never touches the cue VECTOR, so `search_cue_vectors_filtered` keeps returning them and both materialisers built rows with no visible text. Live 2026-09-03: the pair question `why was Thompson sampling removed and what is the flip condition…` got three empty cue rows as a "full" preflight page, while the offline pipeline's top-3 held both answers (rows [A] and [BC]). Read side fixed (`1e591d5`: a blank cue falls through to its episode). **Write side still open (T2):** demotion should delete or unindex the cue vector (`HelixSearchIndex.delete_cue_vector` exists; `run_cue_hygiene` receives only the graph store), and the number of blank-text cues in the brain has not been counted. |
| — | **the project-name query prefix is load-bearing; the pair questions need a different mechanism** | Offline trace of the exact live query for `pair-ts-flip` (2026-09-03) showed the prefix "Engram " lifting a bootstrap doc that repeats the name to BM25 #1 and filling the vector lane with this session's own `[assistant|Engram]` captures, pushing the row holding both flip-condition facts from fused #2 to #5. Gating the prefix off (`0b7c420`) measured **4,4,4** against 7,7,7 — it lost ts-kill, deleted-phases and fastembed-outage and fixed no pair — so it was reverted (`81740e1`). Lesson for §2.7: one traced query is a mechanism, not a policy; the meter decides. Pairs are two questions joined by "and" whose halves compete for one top-3; the honest next experiment is conjunction split (recall each half, interleave), measured on the same instrument. Side finding: blank cue rows fixed on the read side (`1e591d5`). |
| — | **34% of the corpus was stale bootstrap edit history: 695 files as 4,081 snapshot episodes** | Term-by-term BM25 probe on `recall-outage` (2026-09-03): 22 of the keyword lane's top 40 were `[project-bootstrap|…]` rows; for `vector-write-path` the same doc sat at ranks 1–2 twice. Bootstrap stored a new snapshot per content-hash change and never retired the old one (`ingestion/project_bootstrap.py`). Fixed (`17ccd8f`): on change the previous snapshot is marked MERGED at once; the mop's `bootstrap_supersede` keeps the newest per (project, path) and purges the rest with all vector rows. Run live after `engram backup create` (…234539Z): **purged 3,386 episodes, 6,949 vectors, 0 errors**. The other half of `recall-outage` is a true paraphrase — the question's most discriminative term (`broke`, df 54) occurs in no answer row; the answer rows say `empty`/`outage` — and the vector lane's nearest were vision docs. Not a ranking bug; a vocabulary gap the keyword lane cannot bridge. |
| — | **the keyword lane's episode page was a fraction of k all day: `SearchBM25<Label>(q, k)` cuts to k across EVERY label, then filters** | `helix-db/.../ops/bm25/search_bm25.rs` (2026-09-03): `s.search(query, k)` over the whole index, label filter afterwards. Measured on a copy: `("Engram", 50)` → 7 episodes, `("semanticize", 50)` → 0 with 30 episodes containing it; at k=500 → 59 and 30. The lanes fetch ~45, so the page was mostly cue and chunk docs. First suspected as purge damage; the pre-purge backup shows the same numbers, and rare-token coverage is 88% (misses are machinery notifications). Fixed in the engine: fetch 16×k, filter by label, then `.take(k)`; the index already scores every match before truncating. Verified after the rebuild (`a4a9438`): `("Engram", 50)` → 50, `("semanticize", 50)` → 30, warm 4–68 ms per lane. **Meter: 9, 9, 9 of 14** (sd 0) with `flip-condition`, `recall-outage` and `vector-write-path` all at 1.0 — the two targeted singles and the one they dragged along. Day's line on one instrument: 1/12 → 5.7 → 6 → 7 → 8 → 9. Machinery demotion at rank time (`8c8ac1a`) held 9, 9, 9 — no gain, no regression. Remaining: `deleted-phases` (its answer lived in the purged edit history; 26 conversation episodes still hold it and rank low behind this session's own captures) and the four pairs, whose halves now all hit but whose top-3 is crowded by the same self-captures. Next honest step is a clean read: the pre-work snapshot (`…20260903T135931Z`) or a rig this session never discussed. |
| — | **T0 caveat on the meter: the brain now contains this session's own analyses of the rig questions, and some hits are self-referential** | Offline traces 2026-09-03 (evening): for `pair-ts-flip` the row tagged as the Thompson answer is `[assistant|Engram] Ranking. The instrument is clean now…` — my own report from this session, which quotes the rig's tokens. The dogfood loop feeds the instrument. Not a ranking defect, but any meter reading taken while working the meter on this brain is partly measuring the work log; a clean read needs either a rig with questions this session never discussed, or a corpus snapshot from before the work (`…20260903T135931Z` exists). Recorded so the 9/14 is not quoted as pure retrieval gain. |
| 24 | ~~effective config is launcher-dependent~~ **FIXED** (`499301b`) | install config is now the highest dotenv (cwd → repo → `~/.engram/.env`), matching what the LaunchAgent exports; the ambiguity banner no longer fires on this machine. |
| — | the cold brain's `--pause-shell` crash-safe resume DOES work (marker + `_resume_shell`), so the 18-day shell outage (2026-08-15 → 09-02) was not the brain stranding it — cause still unknown; likely the same Jetsam | `brain_cli.py` `run_brain_command` stranded-shell recovery; verified live: shell back at 200 after each of four SIGBUS runs. |
| 19 | ~~`recall_graph_gate` returns `None` in 0.11 ms after probe timeout~~ **FIXED** | `7dda96d`. Refusals raise `GraphGateTimeoutError` (same contract as `native_transport.py:54` `NativeQueryError`, not a second vocabulary); `recall_graph_gate_refusals` counts them. Reproduced first, through the real pipeline with a real `asyncio.wait_for` expansion timeout: the gate answered `('value', None)` for an episode that **exists**, byte-identical to its answer for one that does not, and **3/3** documents reached the cross-encoder as empty text while the store held content for all three — coverage **0.0 through the gate vs 1.0 direct** — after which `scored.sort()` reordered real results by the rank of those empty strings. Caller audit, one decision each: rerank **degrades** (keeps pre-rerank order, `recall_reranker_skipped_probe_timeout`); temporal ×4, durable-slot reserve **stop + record**; entity-match and entity-query pools **pre-check + record**; working-memory pool stops expanding but **keeps its entities** (the blanket `except` would have returned `[]` and deleted the pool the falsy sentinel used to preserve); durable feeder serves partial ids and does not cache a refused listing. Four neuters proved RED, including "remove the Step 1.8 pre-check → `GraphGateTimeoutError` escapes `retrieve()`", which is what makes the audit test able to fail. Output unchanged at shipped defaults except the reranker. **Not measured live** — the shell was owned by another lane |
| 26 | ~~`_resolve_episode_helix_id` full group scan on cache miss~~ | **MOSTLY DONE 2026-07-24** (`562653c`). The cliff was never the 533 ms — it was that the 533 ms was **paid again on every recall and thrown away every time**. `wait_for` cancels the coroutine before its caching loop runs while the executor job completes anyway, so the pre-fix stage cached *nothing*: three consecutive runs, 2000-episode fixture, 40 ms stage budget → timed out 3/3, **0 episodes ever given graph signal**, 4/8/12 scans, 6337 ms of worker time, 100% discarded. Now: one scan per group per 5 s, shared by concurrent callers and `shield`ed, so recall 1 pays the cliff and recalls 2–3 complete with 60/60 covered off 1 scan / 747 ms. Also fixed: an id the scan does not find is remembered for the TTL — previously a deleted / cross-group / ticket-21-gap episode re-paid a full scan on **every** call, so "0.263 ms warm" never applied to it. `tests/test_episode_helix_id_cold_scan.py` (9 tests), red on 5 separate neuters. **STILL OPEN:** a single cold call still costs a full scan. The real fix is the missing `find_episode_by_episode_id` route — the Entity twin is `schema.hx:261` — which another lane owns; with it, this scan becomes a fallback. Keep as T2 until that lands |
| — | the fan-out bound in `episode_graph_signal` does **not** reduce worker burn — corrected claim | Four arms, isolated: adding `asyncio.Semaphore(8)` cut executor submissions 60 → 8 but left jobs actually run at **4 vs 4** and worker time at **3042 ms vs 3043 ms**, because `run_in_executor` cancels its own still-pending future when the gather wrapper is cancelled. The queued surplus was never going to run. What the bound *does* buy is head-of-line blocking on the shared executor — **ticket 31's mechanism**: one independent read submitted 5 ms into the stage waited **40.5 ms unbounded vs 5.2 ms bounded** (median of 5). Recorded because the wrong rationale was written into a code comment first and only measurement caught it. Peak in-flight is now counted, not inferred: `recall_episode_graph_signal_inflight_max` |
| — | **captured assistant responses are silently truncated at 2 000 characters, mid-word, with no marker anywhere** | `setup.py:861` writes the installed hook line `RESPONSE="${RESPONSE:0:2000}"`, then prefixes `[assistant\|<project>] ` — which is exactly why **129 live episodes are 2 019 chars** (2 000 + `[assistant\|Engram] `) and **419 are 2 033** (2 000 + a 33-char header). Measured by reading every Episode node in the live store: **454 episodes** land in the 1 985–2 048 **byte** window and **1 522** in 4 033–4 096 (the second cluster is `project_bootstrap.py:265` `raw_content[:max_chars]` at 4 000, source `auto:bootstrap`) — ~21 % of a 9 403-episode corpus sitting against a cap. `ep_293a18033a09`'s stored content ends `"…on lite/SQLite test brains those rea"`. Nothing records that a truncation happened: no ellipsis, no `content_truncated` flag, no counter — so the tail is not merely unretrievable, it is **not known to be missing**, and every "the episode contains X" scan is scanning a prefix. This is write-side data loss, so **T1**, not T2. Falsified by capturing a >2 000-char response and finding the tail in the store. **Not fixed here:** the effective file is `~/.engram/hooks/capture-response.sh`, a user config file, and raising the cap changes capture cost/latency — that is a product decision, not a chore |
| 34 | ~~`slowest_read` is a running max that never decays~~ **FIXED** | Estimator is now an EWMA seeded at 0 in `activation/read_budget.py` (one `ReadBudget`, three strategies). Reproduced first: 0.5 ms/read store, 50 ms budget, `max_reads=64` → clean 64 reads / 498 reached; **one 25 ms read → 6 reads / 34 reached** (93% of reach, 19.5 ms of 50 unspent); cold 30 ms FIRST read → 1 read / **0 reached**. After: **36 / 274** and **28 / 210**, both with <0.5 ms unspent. Clean store unchanged. Visibility: `recall_spread_budget_unspent_ms` + `recall_spread_reads` + `recall_spread_stop_<reason>`. Three neuters proved RED |
| — | ~~**BM25 breaker blackout: the two cold calls after every restart or idle gap turned the keyword lane off for 60–80 s**~~ **FIXED `6ff5cfe`** | 2026-09-04. Open now means *serialized probes* (one BM25 call in flight, judged at 3× budget, closes on the first warm answer, no wait after a failed probe); write-side candidate searches join skip-only. Verified live: OPEN 08:19:30 → CLOSED 08:20:28 with no recall having to wait for a window. `91ded55` also releases a probe slot that never reports (graph path swallowed `CancelledError`) and exposes `halfOpenProbe/openForS/probeGrantedForS/staleProbes` on `/api/storage`. |
| — | **the session's most recent observe answers any later query that shares ≥3 common tokens with it, as `cache_satisfied`, without running the pipeline** | 2026-09-04, `recall_surface.py` `_packet_query_match` (`best_score >= 3`). Live: 'second capture on the restarted shell with the project field' was answered 7 min later by the *fifth* capture (overlap: capture, project, field) and the real episode never ran through recall. Open: the rule needs a high-signal-token requirement or a recency-scoped intent, and a measurement of how often a real session trips it. |
| — | ~~**every non-launchd process embedded nothing: `FASTEMBED_CACHE_PATH` lives in `~/.engram/.env` as a raw variable, exported only by the LaunchAgents**~~ **FIXED `c833c83`** | 2026-09-04. `engram doctor`, `engram brain run` from a terminal and the meters resolved `~/.engram/models/fastembed` (unquantized model only) instead of `…/hf` (the configured `-Q` model); the mop's vector backfill said `provider_unavailable`, doctor said `fail`, the shell embedded fine. The provider now reads the variable from `DEFAULT_ENV_FILES` (last wins); doctor passes from a plain shell. Same family as the July FastEmbed outage: the repair needs a positive probe from *every* launch path, not one. |
| — | **the cold-brain LaunchAgent has written nothing to its log since 2026-08-15 (`rc=-10` SIGBUS child), `runs = 21`, `last exit code = 1`, machine on AC** | 2026-09-04, open. Whether launchd stopped firing or the command exits before logging is not yet known; the mop windows run today were all manual. Until this is resolved the 2 h hygiene cadence the product depends on is not happening. |
| — | ~~**an agent-proposed edge whose endpoints it did not also propose was dropped on `quiet` and forked duplicates on `standard`**~~ **FIXED `2b9ddbf`** | 2026-09-04 (resident-agent step 1). Endpoints now resolve against the existing graph before the auto-create rescue or the drop. Verified live on the quiet profile: `remember()` with a relationship only, both endpoints already in the graph, neither proposed → edge `rel_b8a801b8375a` created, no new entities. The MCP `observe` tool now exposes `proposed_entities`/`proposed_relationships`; the prompt states Engram has no external extractor. |
| — | ~~**every projection on the live brain ran the narrow regex extractor because the ladder tried a dead Ollama host first and fell through silently**~~ **REMOVED (resident-agent step 2)** | 2026-09-04. `extraction_provider` accepts only `narrow`; the anthropic/ollama rungs, the `auto` ladder, `ollama_extractor.py` and the doctor's Ollama probe are gone. A `.env` that still names them is mapped to narrow with a WARNING naming the line to delete (never a crash: `ActivationConfig` forbids unknown keys, and `server/.env` still carries three such lines). Kill-rig producers are `proposals`/`narrow` only; the fresh-agent judge has no external option. Tests no longer read any `.env` (two shutdown tests had flipped on the operator's `ENGRAM_RUNTIME_ROLE=shell`). |
| — | **dashboard-chat interaction emitters (`selected`/`dismissed`/`used`/`surfaced` from `dashboard_chat`) exist in the tree but sit behind the chat route's 501; the emitter declaration counts call sites, not reachability, so `selectedCountEmitters` names a surface that cannot move** | 2026-09-04, `interaction_surfaces.py` INTERACTION_EMITTERS (comment records it). Two honest exits: delete the chat tool loop and feedback emitters with their seam tests, or give the declaration a reachability state. Neither done today; the tree-scan test pins the current shape. |

### T2 — silent-inert (unfinished features)
| # | item | note |
|---|---|---|
| 20 | ~~`if not candidates:` early-return skips Step 5.5~~ **FIXED, and the ticket's mechanism was wrong** | `7229f89`. `passage_first_entity_budget=0` truncates the FINAL entity slots (Step 6, `pipeline.py:2432`), **not** the candidate pool, so "the entity lane is ALWAYS empty on the default tier" is false — 13 live recalls carried **49–56** entity candidates on 9/9 that reached the pipeline, and the early return fired **0/9**. The defect is real anyway and wider than the ticket: **two** early returns surface episodes above Step 5.5 (`if not candidates:` and the no-semantic-anchor guard one block below), and on the default `passage_first` tier the episode/cue channel is the only channel that reaches the caller, so on those paths the rerank was unreachable rather than ineffective. Both now call `_rerank_special_results` first, same substage timeout, same `GraphGateTimeoutError` degrade as Step 5.5, a metric on every give-up. Byte-identical at shipped defaults (`reranker_provider=noop`, `reranker_rerank_episodes=False`, and live `quiet` force-disables the reranker). Probe `tests/retrieval/test_rerank_reachability.py` (8 tests), RED on 3 neuters. **Fourth cause recorded:** even when Step 5.5 IS reached on the default tier, its entity permutation is discarded by `entity_results = scored[:0]`, so only the `reranker_rerank_episodes` half can ever reach output |
| — | **the reranker records NOTHING when `reranker_enabled=False`** | `pipeline.py` Step 5.5 is wrapped in `if cfg.reranker_enabled and reranker is not None:`, so on the live `quiet` profile (which force-sets it False, `config.py:3183`) there is no `recall_reranker_*` key of any kind — measured: **0 reranker keys across 13 live recalls**. "The rerank did nothing" and "the rerank was never armed" are indistinguishable in the payload, which is how this stage came to be measured dead three separate times. Falsified by finding a skip metric on a reranker-disabled recall. **T2** |
| 28 | ~~`pool_total_limit` has zero read sites~~ **FIXED** | `candidate_pool.py:76` now derives the cap from `cfg.pool_total_limit * scale`; the shadow expression is deleted, so there is no loser to fall back to. Two defects, not one: the knob was inert (20/80/400/1000 all produced 85), **and** the shadow value was the *sum of the four pools it caps*, i.e. ≥ their union by construction — a depth cap that could only ever trim entity-query-unique candidates. Probe: `tests/retrieval/test_pool_total_limit_contract.py` (14 tests, ratchets over every key the function returns). ~~**CONFIG EDIT OWED (T0 owns `config.py`):** set `pool_total_limit` default `80 → 85` to keep the DEFAULT lane byte-identical at scale 1.0.~~ **PAID `7b1588b` (2026-07-24)** — `config.py:526` `pool_total_limit: int = Field(default=85, ge=20, le=1000)`, the 80→85 reasoning in the comment above it (`config.py:520-525`); struck 2026-09-04. Routed lanes intentionally tighten (TEMPORAL 125→85, DIRECT_LOOKUP 115→85, ASSOCIATIVE 105→85 at the live 847-entity corpus): query-type multipliers redistribute depth, they no longer enlarge the cap. Unmeasured live — the ticket's own note says raising the pool measured NEGATIVE, so the direction is supported but n=1 |
| — | `pool_entity_query_limit` is the only pool limit that does not scale with corpus size | `candidate_pool.py:913` reads it via `limits.get("pool_entity_query_limit", cfg.…)`, but `compute_dynamic_limits` never returns that key — the dict branch is unreachable and the fallback fires 100% of the time. Not a lie (the knob works), but at 50k entities its six siblings grow 7× and this one stays at 20. Falsified by adding the key to the returned dict and seeing the entity-query pool widen. **T2** |
| — | a spreading traversal cancelled by the stage wall clock DISCARDS every completed read | `pipeline.py` Step 4 wraps the traversal in `asyncio.wait_for`, which cancels; a cancelled coroutine never reaches its return, so k−1 finished adjacency reads are thrown away when read k overruns. Found working ticket 34, which asked about the *first* read — that part is off by one (on read 1 there is nothing to discard) but the general case is real and **no estimator can fix it**: a surprise 150 ms read is unpredictable from a 0.5 ms history under any statistic. Now at least *visible* — `recall_spread_reads` survives the cancel while `recall_spread_reached` is 0, proved by `test_a_discarded_traversal_still_reports_what_was_discarded`. The fix is a caller-owned frontier (pass `bonuses`/`hop_distances` in, as `traversal_stats` already is) so a cancelled traversal degrades to shallow instead of to nothing. **Deliberately not smuggled into ticket 34's commit** — it changes the timeout path, which is the exact place `390866b` lost cold-recall rows. **T2** |
| — | the durable-feeder TTL cache can never hit across recalls | `candidate_pool.py:316` keys on `(id(graph_store), group_id)`, but `pipeline.py:587` wraps the store in a **fresh** `GatedGraphStore` every recall, so the key is a new object identity each time. Measured: two recalls against one underlying store → **2** `get_identity_core_entities` listings; the same wrapper reused twice → **1**, so the cache works and is merely keyed on something per-request. Two consequences: the 5 s TTL buys nothing (every recall re-lists), and a recycled CPython `id()` could serve one store's listing to another. Falsified by keying on the underlying store (`getattr(store, "underlying", store)`) and seeing the second listing disappear. Found while auditing ticket 19. **T2** |
| 7 | ~~cue loop: registration fixed, but never recorded a use~~ **THE PROCESS-BOUNDARY HALF IS FIXED; THE GATE IS NOT YET KNOWN TO MOVE** | The surfaced half lived in a **non-persistent in-process ring** (`retrieval/feedback.py` `SurfacedUsageBuffer`), so `record_observed_usage_events` could fire only when surfacing and echo landed in ONE shell lifetime — architecturally impossible for an agent session that spans restarts. Now backed by a SQLite sidecar (`retrieval/usage_surface_store.py`, `surfaced-usage.sqlite3`, next to the packet-cache sidecar), self-arming from whatever config reaches the hot path rather than from a startup call nobody would remember. **The echo mask and the dedup marks are persisted with the payload, and that is the load-bearing half**: a durable cue without its mask makes a post-restart verbatim parrot read as novel reuse, i.e. the durability fix would *manufacture* the number the gate reads. Definition chosen: *restated, not parroted* — the agent's own next capture contains a ≥2-token/≥10-char phrase from the cue at a position no 5-gram of the surfaced payload covers. Rejected: "explicitly acknowledged" (needs agent opt-in; the one emitter that ever did is dashboard-only, which is why the counter read zero) and bare token overlap (counts parroting). New bound, stated because durability removed the implicit one: a surfacing is eligible for **30 min** (= the dedup window, so a fired cue cannot refire without being re-surfaced). Second call site added: `remember` ran no citation scan at all, so the highest-signal turn was the one that could not record a use. Probe: `tests/test_usage_surface_durability.py` (20 tests) drives surface → **discard the buffer** → new buffer → counter, RED on 8 neuters — **two neuters (mask-not-persisted, capture-never-binds) passed the first probe and the tests were tightened until they failed.** **DO NOT read this as "the flip is unblocked."** Still unverified: that live recall surfaces `cue_episode` results at all, that an agent's next capture reuses a cue phrase inside the window, and that the helix `update_episode_cue` usage trailer round-trips live (no native fixture — ticket #35 ENOSPC). Also unchanged: ~8.4k episodes predate cues |
| 21 | ~~some episodes unretrievable by their own verbatim text~~ **DIAGNOSED — the premise was wrong** | **CORRECTED 2026-07-24.** It is **not** an index/embedding gap. Read-only LMDB forensics on the live brain (heed3 reader, `READ_ONLY\|NO_LOCK`, never wrote a byte; tool in the session scratchpad at `bm25read/`): all three "unretrievable" episodes are **present and correctly indexed**. `ep_293a18033a09` — Episode node present, `group_id="default"`, BM25 doc_length 317 / 223 unique terms, live `EpisodeVec` + `CueVec` + 2 `EpisodeChunk` vectors all `deleted=0`; replaying the real BM25 scorer over the real inverted index puts it at **global rank 2 / rank 1 among Episode-label docs** for its own opening sentence and **global rank 4 / episode-rank 1** for `"what is the flip condition for usage ranking"`. `ep_103d89337b0c` (dl 323) and `ep_bb61718b8e60` (dl 262) are the same shape. So hypotheses (a) missing/corrupt embeddings, (b) BM25 doc-id collision, (d) group/scope mismatch are **all falsified**. The mechanism is the RRF lane veto — next row. Ticket 21's own claim "no ranker can fix it" is right for the wrong reason: no ranker can, but the **fusion** can |
| — | ~~**the BM25 lane cannot contribute a single document to any hybrid search — it is fetched 3× deep and discarded by construction**~~ **SHIPPED `38129d1` (2026-09-02) + `11dab64` (2026-09-03)** | `search.py` `search` (entities, :1600), `search_episodes` (:1684), `search_episode_cues` (:1788) are byte-identical in shape: `bm25_fetch_limit = limit * 3`, `vec_fetch_limit = limit` (whenever `group_id` is set, i.e. always in production), `_rrf_fusion(fts, vec, 0.3, 0.7)` then `[:limit]`. RRF weights rank *r* at `w/(60+r+1)`, so the **worst** vector-lane item scores `0.7/(60+limit)` and the **best** BM25-only item scores `0.3/61 = 0.004918`; `0.7/(60+limit) > 0.3/61` for every `limit ≤ 82`. The shipped episode-lane limit is `episode_retrieval_max(5) × 3 × 2` (`retrieval_strategy` defaults to `passage_first`, `pipeline.py:905`) = **30**, and `episode_retrieval_max` is `Field(le=20)` so the knob cannot reach the crossover. Lane *overlap* — the normal case — widens the band further. Net: the slowest native call in the stage (the one the BM25 circuit breaker exists for) runs at 3× depth and can only ever **reorder** what the ANN lane already found. Probe: `tests/retrieval/test_episode_search_lane_fusion.py` (18 tests), RED on two independent neuters (equal lane weights → 9 failures; a 3-slot BM25 reserve → 10). **Fix deliberately NOT shipped** — it changes recall composition on the hot path and this lane could not measure live. **T2** **Struck 2026-09-04:** `search.py:102` `fts_lane_reserve(limit, fts_weight, vec_weight)` reads the lane weight as a PAGE SHARE (0.3/0.7 → 30 % of the page; `0.0` restores the pre-fix veto exactly) and `_rrf_fusion` (`:195-236`) seats the keyword lane's top rows INSIDE the `[:limit]` cut via `_apply_fts_lane_reserve` (`:126-192`), so the fusion no longer discards the lane by construction. `11dab64` then gave seated rows spread positions and the fused score of the position they occupy, because the first version left them at 14-15 of a 15-row page the pipeline never reads past `episode_retrieval_max`. Probe `tests/retrieval/test_episode_search_lane_fusion.py` was rewritten in both commits, and the packet-cache fingerprint now covers `storage/*/search.py` (`38129d1`) so an A/B cannot be served this fix's own control arm. Evidence is the 2026-09-03 lane trace in `11dab64` (keyword lane held the answer at ranks 3-4 and now survives the cut), not a meter delta. |
| — | ~~**`SearchBM25<Label>(q, k)` takes the global top-k across EVERY node label and only THEN filters by label**~~ **FIXED `a4a9438` (2026-09-03), first-order half** | `native/helix-repo/.../ops/bm25/search_bm25.rs:56-95` — `s.search(txn, query, k)` is label-agnostic (one inverted index, one `total_docs`, one `avgdl` for the whole graph) and the label test is a `filter_map` **after** `results.truncate(limit)`. Generated caller: `helix-python/src/queries.rs:5787` `search_bm25("Episode", &data.query, data.k)`. Measured on the live index: **1 343 243 BM25 docs**, of which **1 182 202 (88 %) are `ConsolDecisionTrace`** and only **9 403 (0.7 %) are Episodes**; replaying `k=150` for `"flip condition usage ranking"` returned **60 Episodes out of 150 slots** (the rest EpisodeCue/Evidence/Entity/Consol*), and for the opening-sentence query 100/150. So 33–60 % of every BM25 fetch is spent on labels the caller discards, the loss is query-dependent and unbounded, and a query whose terms are common in consolidation audit rows could return **zero** episodes at any k. Second-order: `avgdl = 70.7` is set by those 1.18 M audit rows while Episodes average **339.8**, so every episode carries a ~4.8× BM25 length penalty *because internal bookkeeping nodes are in the user-facing text index*. Falsified by pre-filtering by label inside `search`, or by keeping Consol\* nodes out of BM25. **Blocked here — `native/**` is owned by another lane. T2** **Struck 2026-09-04:** `search_bm25.rs:56-67` now fetches `16 × k` from the label-agnostic index, filters by label, and cuts to `k` AFTER the filter (`.take(k_usize)`, `:107`); measured in the commit on a store copy: `("Engram", 50)` 7 → 50 episodes, `("semanticize", 50)` 0 → 30, warm 4-68 ms per lane. **NOT fixed by it:** the second-order `avgdl` penalty (one index, one `avgdl` set by ~1.18 M `ConsolDecisionTrace` rows) and the presence of Consol\* nodes in the user-facing text index both stand, and a 16× overfetch is a bound, not a guarantee, for a query whose terms are common in audit rows. |
| — | `HNSWConfig::new` clamps `ef` to `10..=512`, so the declared `ef_search: 768` can never take effect | `native/helix-repo/.../vector_core/vector_core.rs:64-70` — `let ef = ef.unwrap_or(768).clamp(10, 512)`. `queries.rs:105` sets `ef_search: Some(768)`. Ticket 27 pinned the three `config.hx.json` copies to the `queries.rs` literal `768`, which is the *declared* value — the **effective** value is 512. Ticket 4's "ef_search rebuild" item should target 512, and the authority test should assert the clamped value, not the literal. **T0-adjacent (a declared number that is not the effective one); recorded here, blocked on `native/**`** |
| 23 | graph consumer built, **flags OFF — and the kill rig says KEEP THEM OFF** | 2026-09-02: ticket 25 landed, so the rig ran (see row 3). Arm B (`entity_episode_traversal_source=candidates`, the consumer live) reached the gold episode **0/55** at k=5 and k=10, identical to arm A (vector only, 0/55), while adding 6.6 rows and 1.4k chars per recall; the consumer byte probe confirmed 75 248 edge-derived chars reached the answerer on 55/55, so this is a live mechanism that buys nothing on the planted corpus, not a dead one. Verdict KILL on K1, K2, K4 under the pre-registration (`GRAPH_THESIS.md §5 @ d7c764e`). Caveat carried verbatim: "Scores a retrieval list, not an agent's task outcome — do not let it flip a default without an agent-task arm." Not blocked any more; not armed. |
| — | ~~`evidence_client_proposals_enabled` is a no-op kill switch~~ **FIXED** | `03cba85`. One read site (`GraphManager._client_proposals_accepted`); both places that asked "were proposals supplied?" now ask it. Reproduced first, end to end on a lite store: with the flag **False** an agent-proposed Decision still committed — `flag OFF must not commit an agent-proposed fact; found ['keep Engram fully local']`. **Two defects, not one:** the harness scoreboard computed `is_proposal_path` as *"the proposal path ran OR proposals were supplied"*, which stop being the same question once the flag can decline them — with the gate in but the scoreboard untouched, a **narrow** extraction reported `client_proposal_commits=1, client_proposal_rejects=2, client_proposal_share=0.5`. `tests/test_client_proposal_kill_switch.py` (7 tests, 3 controls), RED on two neuters — **the first probe passed on neuter 2 and had to be tightened until it failed.** Unchanged at shipped defaults (flag defaults True; every profile that sets it sets it True). Not measured live |
| 37 | ~~`apply_chat_recall_feedback` reachable only from the dashboard~~ **FIXED — half (a) for `used`, half (b) for the rest** | `ba362f8`. The ticket offered (a) *give the other surfaces an emitter* or (b) *declare the deadness*; the vocabulary is not in one epistemic state, so one answer would have been wrong for half of it. **(a) for `used`:** the detector was never the missing part — the observe/remember echo scan already writes a `used`-TIER ACCESS EVENT with ticket 7's M5.1 predicate (UNCHANGED: a contiguous 2–5 token cue phrase in the agent's next capture at a position no 5-gram of the surfaced payload covers). Only the INTERACTION was missing, so `used_count` and the adaptive-threshold learner at `control.py` that reads `interaction_counts["used"]` sat at 0 for every non-dashboard surface. Lane 3B's owed halves landed whole (`GraphManager.record_echoed_memory_usage` + the probe; half 2 was already restored into `capture_surface.py` by ticket 7). Deliberately NOT routed through `apply_memory_interaction`, which would write the used-tier access event a **second** time — neuter C catches exactly that. **(b) for the rest:** `dismissed` has no honest capture-side derivation (a capture has no bounded response to partition, so "surfaced and not echoed in this observe" fires on memories the agent used and did not capture — and it feeds `false_recall_rate` *and* the learner, so an over-firing dismissal pushes recall the wrong way, not merely noisily); `selected` is an artifact of the dashboard's LLM tool loop; `confirmed`/`corrected` are in `_VALID_TYPES` and emitted by **nothing, anywhere**. So `retrieval/interaction_surfaces.py` declares verb → surfaces-with-a-real-emitter and the controller publishes `unmeasurable_interactions` + `interaction_surfaces_observed` beside the counts. Counts stay ints (honest counts of emissions); the derived RATE goes **absent** — `false_recall_rate` is `None` when no observed surface can emit `dismissed`, because `0.0` there is the plausible-but-wrong metric §2.1 forbids. `brain_loop_report` had to change with it: `_float()` turns `None` into `0.0`, which would have re-fabricated the number one level up (neuter G). `lifecycle_summary` cue block now carries `selectedCountEmitters`, so the live `selectedCount: 0` against `surfacedCount: 1208` finally reads as *"one emitter, and it is the dashboard"*. Two properties keep the declaration honest and both are pinned: observed counts override the table (a stale table under-reports deadness, it can never bury a live signal — neuter E), and a source scan fails if a verb declared emitter-less has an emitter or a verb declared emittable has none (neuter H, red in both directions). Probes: `tests/test_observed_usage_interaction.py` (6) RED on 3 neuters, `tests/test_interaction_surface_declaration.py` (13) RED on 6, `tests/test_config.py` RED on 1; all GREEN restored. **NOT CLAIMED:** that `used_count` will move on the live brain — that still needs a real recall to surface a payload and the agent's next capture to reuse a phrase inside the 30 min window; the narrower claim is that the interaction can now be emitted outside the dashboard, and where it still cannot, the metric says so. Isolated lite regime, no live measurement (another lane holds the shell) |
| — | ~~the surfaced-usage sidecar borrows an unrelated sidecar's directory~~ **FIXED** | `ba362f8`, ticket 7's owed config knob. `recall_usage_surface_path` + `get_usage_surface_path` / `configure_runtime_usage_surface`, wired into both runtime entrypoints (`main.py`, `mcp/server.py`). It borrowed the DIRECTORY of `recall_packet_cache_path`, falling back to `cue_index_outbox_path` — which worked only because both default enabled: running with the packet cache off, i.e. **the documented way to isolate an A/B (§2.4)**, moved the sidecar, and both off unbound it entirely — silently, because unbound degrades to the process-local ring, the exact pre-ticket-7 behaviour. Legacy borrows kept as a fallback so a running install keeps its existing file rather than starting a fresh one. **FOLLOW-UP:** the new field is *not* in `FINGERPRINT_EXCLUDED_FIELDS`, so it joins the packet-cache fingerprint. That is the safe direction per ticket 29's own asymmetry (over-inclusion costs ≤1 TTL of warmth), but it is sidecar plumbing that cannot change packet content, and an operator CLI that does not call `configure_runtime_usage_surface` now fingerprints differently from the server. The exclusion list belongs to the packet-cache lane; left for them |
| — | ~~**a client-proposed EDGE is dropped at materialization on the live profile, while the scoreboard counts it as a commit**~~ **FIXED `2b9ddbf` (2026-09-04)** | Four arms, isolated lite stores, `consolidation_profile=quiet` (the live value) vs `standard`. An edge whose endpoints the agent did **not** also propose as entities: `quiet` → **0 edges, even when both endpoint entities already exist in the graph**; `standard` → 2 edges. Mechanism: `apply.py:285` resolves endpoints only through this episode's `entity_map` (built from *this episode's committed entity candidates*) and never against the existing graph; the only rescue is `_auto_create_endpoint`, gated on `graph_auto_create_endpoints`, which is `False` by default (`config.py:2982`) and set True **only** by the `standard` profile (`config.py:3250`). The evidence layer commits the relationship (`client_proposal_span_verified`) and `client_proposal_commits` counts that DECISION, so the counter reads healthy while zero edges land. **This is the answer to "128 proposals, no new predicates" — see the row below.** Falsified by proposing an edge-only annotation on `quiet` and seeing an edge appear. **T1** (the metric is wrong, not merely missing) **Struck 2026-09-04:** `extraction/apply.py:204-245` `_resolve_existing_endpoint` resolves an endpoint this episode did not propose against the EXISTING graph (`resolve_entity_fast` over `find_entity_candidates`), and `apply_relationship_fact` calls it (`:345`, `:349`) before the `graph_auto_create_endpoints` rescue (`:352`) and before the `missing_entities` drop, so the edge lands on `quiet` when both endpoints exist; the `client_proposal_commits` counter itself is unchanged and still counts evidence commits, not materialized edges; resolved endpoints are returned in the result metadata as `resolved_endpoints` (`:343-351`, `:613-614`). Probe `tests/test_apply_graph_write_fixes.py::test_edge_endpoints_resolve_against_the_existing_graph_on_every_profile`. Isolated-store evidence only; not re-measured on the live brain. |
| — | ~~on `standard`, the same rescue creates DUPLICATE endpoints instead of resolving them~~ **FIXED `2b9ddbf` (2026-09-04), same change** | Same four arms: `endpoints_in_graph=True, entities_proposed=False` on `standard` returned **4** entities for 2 names. `_auto_create_endpoint` (`apply.py:235`) mints a fresh `ent_<uuid>` unconditionally — it never calls `resolve_entity_fast`, which the entity path does. So the profile that makes edges land also silently forks every endpoint it rescues. **T2** **Struck 2026-09-04:** the existing-graph resolution at `apply.py:345-349` runs BEFORE `_auto_create_endpoint` (`:352-376`), so the rescue now only mints `ent_<uuid>` (`:279`) for a name the graph does not already hold; an endpoint already in the graph is matched, not forked. `_auto_create_endpoint` itself still never calls `resolve_entity_fast` — it no longer needs to once it is reached only on a miss. |
| — | "no new semantic predicates from client proposals" is a DESIGNED INVARIANT, not a defect | `ALLOWED_CLIENT_PREDICATES` (`extraction/promotion.py:37`) is a closed 34-item frozenset; anything outside it is capped to 0.40 (`client_proposals.py:274`) and hard-rejected `predicate_not_allowed` (`commit_policy.py:203`). The agent **cannot** introduce a predicate the vocabulary does not already contain. The graph-thesis investigation was measuring the wrong thing: new predicates were never possible, so their absence is not evidence of an inert path. The real leak is the row above (edges dropped at materialization). Live scoreboard corroborates the shape: `client_proposal_rejects: 41` and `predicate_not_allowed_rejects: 41` — **every single reject was a predicate reject** |
| — | ~~**the test suite writes to the operator's LIVE harness scoreboard**~~ **CLOSED `c833c83` (2026-09-04)** | `harness_metrics.py:91 harness_metrics_path()` falls back to `~/.engram/harness-metrics.json` and **nothing in `tests/conftest.py` overrides it** [RV]. Measured: running five extraction test files moved live `client_proposal_commits` **1769 → 1818**, `narrow_extractions` 1227 → 1241, `external_extractor_skipped` 1340 → 1374. So `engram harness` and the "128 client proposals in a 14-day window" figure are pytest-contaminated and cannot separate live agent traffic from fixtures. One-line fix: an autouse `conftest.py` fixture setting `ENGRAM_HARNESS_METRICS_PATH` (done locally in `test_client_proposal_kill_switch.py`; **repo-wide fix not applied — `conftest.py` is shared**). **T0** **Struck 2026-09-04:** not by the one-liner proposed here but by the wider fixture — `tests/conftest.py:35-46` `_isolated_engram_home` (autouse) gives every test a throwaway `ENGRAM_HOME`, and `harness_metrics_path()` resolves `harness-metrics.json` under `ENGRAM_HOME` whenever no `ENGRAM_HARNESS_METRICS_PATH` override is set (`extraction/harness_metrics.py:93-97`), so no test can reach `~/.engram/harness-metrics.json` any more. **What this does not undo:** every scoreboard figure read before 2026-09-04 (the 1769→1818 here, the 14-day "128 proposals") stays pytest-contaminated — the leak is closed, the history is not cleaned. |
| — | ~~`graph_kill_rig/runner.py:340` reaches `manager._search`, and the facade boundary test fails on it~~ **FIXED `4907143` (2026-09-04)** | `tests/test_graph_manager_facade_boundaries.py::test_runtime_code_does_not_reach_through_graph_manager_private_fields` is RED at HEAD, extra item `('engram/evaluation/graph_kill_rig/runner.py', 'run_rig', 'manager', '_search')`. Introduced by committed `93bbbf7` (ticket 33); `runner.py` is unmodified in the worktree, so this is **not** a working-tree artifact. A red boundary test trains everyone to ignore boundary tests. **T2**, owned by the kill-rig lane **Struck 2026-09-04:** both `manager._search` reaches were removed from `runner.py` in `4907143` (grep for `_search` in `graph_kill_rig/runner.py` returns nothing at HEAD) and `tests/test_graph_manager_facade_boundaries.py` is GREEN — 92 passed, run 2026-09-04. |
| — | ~~`resolve_usage_surface_path` stringifies whatever it is handed and creates the directory~~ **FIXED before it was committed, `09cdc02` (2026-07-24)** | The ticket-#7 sidecar wrote `server/MagicMock/mock._cfg.recall_packet_cache_path/surfaced-usage.sqlite3` into the repo root when a test passed a `MagicMock` manager. It is not Mock-specific: any cfg whose path field is not a real path yields a garbage **relative** directory under CWD instead of a refusal or a degrade. Found incidentally; belongs to the ticket-#7 lane's uncommitted `retrieval/usage_surface_store.py`. **T2** **Struck 2026-09-04:** `retrieval/usage_surface_store.py:79-90` refuses anything that is not a non-empty `str` resolving to an ABSOLUTE path (`isinstance(configured, str)`, `Path(...).is_absolute()`) and falls through to the next lender or to unbound — the "Refuses rather than guesses" contract in its docstring (`:72-77`); a `MagicMock` cfg now yields an unbound registry, not a directory under CWD. |
| — | `spread_candidate_injection_max=32` saturates on 100% of recalls | frozen guess sitting at its ceiling, discarding ~450 entities per request |
| — | ~~**episodes carried no project field; the hook resolved the project and dropped it, scoping re-parsed the `[role\|project]` header from content**~~ **LANDED `91ded55` + `b6941e5`** | 2026-09-04 (Step 3). `Episode.project` rides in `encoding_context_json` (no Helix schema change), flows auto-observe → store → scope multiplier → presenter. Verified live that it *persisted but never presented* through three separate drop points (result builder, packet mirror, capture-time `recent_observation` packet) — the same silent-inert shape as ever: the write landed, the read had no consumer. |
| — | ~~**machinery episodes vectorised before the capture-time gate kept their vectors: the only de-indexer lived inside the OFF reindex sweep**~~ **LANDED `e680abd`** | 2026-09-04 (Step 4). `reindex_sweep_episodes(deindex_only=True)` as its own mop drain, newest-first, 500/window, own cursor, one-time completion. Four manual windows (2 000 newest episodes): 105+/500 classify machinery, **none held vectors** — the gate works for everything captured since it shipped; the debt is older and the drain keeps walking. |
| — | **the two pre-pipeline lanes (`fast_preflight_hit`, durable-first `hit`) present episode rows with no content and no project** | 2026-09-04. They build items from cue rows without fetching the episode; the deep pipeline and both packet paths present `project`. Small, open: the consumer sees a bare id on the fastest lane. |
| — | **7 017 episode vector rows carry no readable float payload by id: MMR diversity, inhibitory spreading and state-dependent retrieval are no-ops on 81 % of the corpus** | 2026-09-04, from the same census. `search.py` already warns about it (`rows matched but returned NO vector data`). Fix is native (the by-id route should read the vector from HNSW storage) or a one-time re-embed of the old rows; neither shipped today. |
| — | ~~**the graph kill rig VOIDs itself on a race it owns: capture-time indexing is deferred to a background lane and the rig runs preflight and closes the stores before it drains (24/60 gold episodes vectored, 5× `SQLiteVectorStore not initialized`)**~~ **FIXED `bc85c64` (2026-09-04), race half** | 2026-09-04, Step 5 first run, `--producer narrow`. The producer probe also failed honestly: narrow proposed edges whose endpoints never committed (`missing_entities`), zero semantic relationships — the extraction lever again, not the rig. **Struck 2026-09-04:** `graph_kill_rig/runner.py:248-255` `_drain_capture_indexing` awaits the capture service's serialized background index lane, and the ingest path calls it after `_ingest` and before `_close` (`:334-339`); per `bc85c64`, with the drain `vector_index_probe` passes 60/60 and the rig reaches a verdict (the Step 5 row in T3). The producer half is not a rig defect and is unchanged by this: `4907143` removed the external producers so narrow is the only one, and the 2026-09-04 rerun recorded VOID on the organic producer (zero bridges from real text) — see the T3 Step 5 row. |
| — | **re-proposal drain (resident-agent step 5): every episode now records who projected it (`projected_agent` / `projected_narrow`), recall items carry `extractedBy`, `remember(episode_id=…)` re-projects an existing episode with the agent's proposals, and the operator tool `list_unstructured_episodes` pages the narrow-projected backlog** | 2026-09-04. Agent-driven and bounded by design: only what the agent reads or deliberately works gets restructured; no batch LLM. Legacy `projected` rows read as narrow because the external rungs never ran on this brain. Verified live: a narrow-projected item shows `extractedBy: narrow`. |

### T3 — capability and decision
| # | item | note |
|---|---|---|
| 3 | ~~the graph kill experiment~~ **RUN, status RESULT, verdict KILL** | 2026-09-02, `uv run python -m engram.evaluation.graph_kill_rig --scratch … --n 60` (producer `proposals`, seed 17, 210 episodes, 55/60 bridges verified, all 60 gold episodes vectorised, floor 36 met; every preflight check passed, so reachability was NOT suppressed). Evidence: `docs/product/experiments/graph_kill_rig_2026-09-02_proposals_n60_seed17.json`. Arm A reach@5 0/55 · arm B 0/55 · arm C (one extra recall round, merged) **1/55** at +80 ms p50. K1 (C ≥ B), K2 (B−A inside HNSW jitter), K4 (residual 0%: the LINKING episode ranks in A's top-10 on every question, so the graph can only be a latency optimisation). Reach this low on ALL arms says the planted corpus's gold episodes are unreachable by anything on this stack — a fact about the corpus/scorer as much as the graph; a real-corpus rerun (§2.7) is the next honest step before this verdict is quoted outside the ledger. |
| 2 | M3 concurrency: parallelise episode/cue/chunk lanes | **PARTIALLY DONE, ONE STAGE UPSTREAM** (`7229f89`). The fan-out that was actually costing recalls was `graph_expansion.expand_query_from_graph`, not the episode/cue/chunk lanes: it now carries its own deadline (returns the partial expansion instead of being cancelled and discarding every completed read), runs independent per-term lookups under `asyncio.Semaphore(8)`, and no longer issues the **identical** `get_relationships` call twice per entity (measured 20 reads where 10 were needed). Probe `tests/retrieval/test_graph_expansion_fanout.py` (10 tests), RED on 4 neuters. **Still open:** the episode/cue/chunk lanes themselves are still serial — but see the ledger row above, they were never the binding constraint measured live (episode 6.6 ms, cue 2.6 ms, chunk 12.6 ms against a 976–3023 ms pre-pipeline cascade) |
| — | ~~**the pre-pipeline rescue cascade costs 2–8× what `retrieve()` costs, and can starve it to zero**~~ **FIXED `d71dba7` + `65f4230` + `6b8bc3c` (2026-09-02), then `2ae5c65` (2026-09-03)** | Measured live, 13 REST recalls, warm helix shell: `durable_entity_first` + `recall_fast_preflight` + `durable_entity_rescue` ran **976–3023 ms** before `retrieve()` started, against **371–477 ms** for the pipeline itself. On the worst run 3907 ms of a 4000 ms wall was gone before the pipeline began and it was **cancelled at 92 ms** (`recallRetrieveCancelled`), returning a single `durable_entity_rescue_after_timeout` row for 5024 ms of work. `recall_surface.py:816` recomputes the deep-recall timeout from the wall clock *remaining*, so every millisecond the cascade spends is taken directly out of the pipeline. Falsified by timing a recall with the cascade disabled. **T1** — this is not "the pipeline is slow", it is "the pipeline is not given time to run" **Struck 2026-09-04:** the T1 row "explicit recall was degraded on 10/10 live calls" carries the before/after on the same 10 queries — 0/10 ok, median 6698 ms, every item from the packet cache → 10/10 ok, median 1665 ms, 50 items from the pipeline. Levers in `retrieval/recall_surface.py`: `durable_entity_first` capped at a tenth of the wall, max 0.4 s (`:709-724`); the rescue wrapper no longer doubles the cap it is handed; the duplicate second pre-pipeline durable rescue deleted (§2.14, Konner to confirm); timed-out probes back off 30 s (`:706-707`, `:802-804`); and `2ae5c65` starts the pipeline FIRST with the preflight as a latency hedge only (`recall_pipeline_first_hit`/`_miss`, `:793-797`). This row's falsifier (a recall with the cascade disabled) was answered by the before/after instead; the standing falsifier is the T1 row's: `recall_stats_timeout` or `durable_entity_first` ≥ 400 ms on consecutive live recalls. |
| 13 | P8 compaction-swap — memory as the agent's swap space | the only item that creates capability rather than repairing leakage |
| 12 | P7 presence curve — should session-start exist at all? | reframe the briefing as a design experiment, not a filter |
| 9 | P4 small-brain / fresh-install behaviour | everything was tuned on a pathological 9,343-episode corpus |
| 4 | RF flip, `ef_search` rebuild, `reindex_sweep` re-test, feeder flip, website sync | note the ef_search item targets a dead file — see 27 |
| — | **Step 2 read, untouched questions: fresh agent with Engram 8/14 vs project files 4/14 (+4), at 69 k chars vs 13 k; on the session-topic battery 8/10 vs 6/10 (+2), 62 k vs 13 k** | 2026-09-04, `engram battery --fresh-agent --against-live --battery-path tests/rigs/agent_experience_battery_untouched_2026-09.json`, containment judge, warm shell. Two bars, keep them apart: the fresh-agent suite scores a hit when *any* expected group is present; the meter needs *all* groups — the same 14 questions are 8/14 on the first bar and 3–4/14 on the second. Read together: Engram usually surfaces half of an answer for an unseen question, at ~5× the characters of the repo files. The lift is real on the untouched set (+4) and smaller on the session-topic set (+2), the opposite of what self-reference would predict — the session-topic questions are the ones CLAUDE.md also answers. |
| — | **Step 5: the graph kill rig ran as the experiment — verdict KILL on the planted control (reach@5 A=0, B=0, C=1 of 51; K1, K2, K4 all trip), VOID on the organic producer (zero bridges from real text)** | 2026-09-04, `docs/product/GRAPH_THESIS.md` §5 'Run 2026-09-04', envelopes under `docs/product/artifacts/`. Arm B ran WITH the M3.1 surfacing port (`entity_episode_traversal_source=candidates`): the July 22/36 did not reproduce. The edge is surfaced, the episode behind it never is, at +7.7 rows / +1.5 k chars per recall. Design space (§4 D/E reranker) not killed; the surfacing consumer is. |

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

- **2026-07-24** — `7229f89`, lane 2 (tickets 20 + 2): **two ticket premises corrected by
  measurement, and the fix landed one stage upstream of where the ticket pointed.** The
  investigation that outranked both tickets — "12 of 13 live REST recalls never enter
  `retrieve()`" — **does not reproduce at HEAD**: 9 of 13 entered with 47–52 pipeline stage keys.
  The real split is cost, not reachability: the pre-pipeline rescue cascade burns **976–3023 ms**
  against **371–477 ms** of `retrieve()`, and on the worst run the pipeline was **cancelled at
  92 ms** because 3907 ms of a 4000 ms wall was already spent (new T1 row). Ticket 20's mechanism
  is also wrong — `passage_first_entity_budget=0` truncates Step 6's entity slots, not the
  candidate pool, and live recalls carried **49–56** entity candidates on 9/9 — but the defect is
  real and **wider**: *two* early returns surface episodes above Step 5.5, and on the default
  tier the episode/cue channel is the only one that reaches the caller. Both now rerank first.
  Ticket 2's fan-out turned out to be in `graph_expansion`, which is not the "zero cost, ~3ms"
  its docstring claimed: a **case-only live A/B** (same six words, one letter's case) gave
  `engram` → 3.1 ms, gate open, **spread reach 197** versus `Engram` → **77.3 ms timeout**, gate
  ARMED, **reach 0** and 4 stages skipped, while `recall_stats` finished in 220–373 ms and the
  primary search in 7–14 ms — so the probe that arms the graph gate was blaming a store that was
  demonstrably fine. Worse, `tell me about helixdb storage limits` extracts **zero terms**, issues
  **zero reads**, and still recorded a 76.5 ms `graph_expand_timeout`: a starved event loop
  reported as an over-budget graph. Fixed with a self-deadline (partial expansion instead of
  cancel-and-discard), `Semaphore(8)` on the independent lookups, deletion of a **duplicated**
  `get_relationships` call (20 reads where 10 were needed), and a probe that discriminates on
  reads **issued** rather than reads completed. **This makes ticket 31's "process-level bimodal"
  suspect** — capitalisation alone reproduces the 0%/100% split with no process change.
  **And it caught the instrument:** the `(ref …)` nonce used to defeat the packet cache is itself
  the thing that armed the gate on 9/9 in the first census (0/10 without it), so a per-query
  nonce is not a neutral way to satisfy §2.4. Ten neuters, ten distinct REDs, GREEN restored at
  byte-identical shas; **one new test was found vacuous and rewritten** (a `<= 8` concurrency
  bound that the structure could never exceed at `max_entities=5`). Four committed stubs that
  hung without touching the store were repaired rather than re-blessed. **No live measurement
  after the fix** — the shell is shared with three other lanes and was not restarted.

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
- **2026-09-04** — `91ded55`, `b6941e5`, `6ff5cfe`: Step 1 read (T0 row: the honest number for
  unseen questions is **3–4/14 warm, 0 cold**, and the 9/14 was self-reference), Step 3 landed
  (`Episode.project` at capture, presented on every recall path — three drop points found live),
  and the BM25 breaker's blackout replaced by serialized probes after the cold runs measured 0.
  Process slip to own: two of today's commits were staged with a path-scoped `git add -A
  server/engram server/tests` against §2.9; the diff was checked by hand before each, but the rule
  exists so that check is not needed. Open from today: the `recent_observation` ≥3-token hijack
  (T1 row), and the machine itself — swap 3.3/4 GB with the shell at 3.3 GB resident — which makes
  every live number on this box a *swapping-machine* number until the working set shrinks.
- **2026-09-04** — Step 2 read (T3 row): untouched questions, fresh agent **8/14 with Engram vs 4/14
  from project files**, +4, at 69 k vs 13 k chars; the bar is any-group, the meter's is all-groups
  (3–4/14 on the same questions). Battery file committed so the number can be re-run.
- **2026-09-04** — `e680abd`, `c833c83`: Step 4 landed (machinery de-index drain; 4 manual windows,
  no vectored machinery in the newest 2 000 rows), the FastEmbed cache-path split fixed (every
  non-launchd process had embedded nothing), test isolation fixed after a test wrote into the live
  hygiene state. Opened: the cold-brain LaunchAgent has been silent since 2026-08-15.
- **2026-09-04** — `773755b`: the backfill presence probe was lying (row vs float payload); machinery
  de-index window 7 removed 44 older machinery vectors; Step 5's first kill-rig run is VOID on a
  rig-owned race (background index lane not drained) plus an honest producer failure (narrow
  commits no bridges from real text). Brain-agent row corrected: launchd fires, battery skips.
- **2026-09-04** — `bc85c64`: Step 5 done. Kill rig, planted control, all pre-flight checks green:
  **KILL** (reach@5 0/51 with and without the graph; the agent's own second query 1/51). Organic
  narrow producer: VOID, zero bridges. Result recorded in GRAPH_THESIS.md §5 with envelopes.
- **2026-09-04** — `2b9ddbf`: Konner's rule recorded — **never Ollama, never Tailscale; the resident
  agent is the only extractor.** Step 1 of that plan: agent edges land on every profile (endpoint
  resolution against the existing graph), observe accepts proposals, prompt says so. Live-verified.
  Next: collapse the extraction ladder (12 source files, 7 test files reference Ollama).
- **2026-09-04** — resident-agent step 2: the extraction ladder is collapsed to narrow; Ollama and
  Anthropic extraction removed from code, config, doctor, rig and fresh-agent; retired `.env`
  lines warn instead of crash. Full non-Helix suite green after fixing 12 regressions, most of
  them yesterday's rank-time multipliers and today's project key leaking into pinned fixtures,
  plus a dotenv-isolation flaw in the test harness. `CLAUDE.md` is git-ignored: its ladder
  paragraph was updated on disk only.
- **2026-09-04** — resident-agent step 5: projector stamped on every episode, `extractedBy` on recall
  items, `remember(episode_id=…)` re-projection, operator `list_unstructured_episodes`. Steps 1, 2
  and 5 of the resident-agent plan are landed; step 3 is a handoff (env lines), step 4 is answered
  by the planted kill-rig run (perfect proposals still reached nothing: the consumer, not the
  producer, is the open problem).
- **2026-09-04** — resident-agent step 3 (with Konner's go-ahead): the Ollama block and every
  external API key line removed from `server/.env`, `~/.engram/.env` and the repo `.env`; shell
  restarts with no retirement warning; `engram doctor` from a plain terminal: extraction narrow,
  embeddings local, no external model. Keys that were on disk are to be rotated.
- **2026-09-04** — state assessment (16-agent workflow) written to
  `docs/product/investigations/ASSESSMENT_2026-09-04.md`: 45 open rows tiered, brain verdict
  **fresh-after-export** (the 4,581 May–June turns exist nowhere but the LMDB env; no export route
  exists), a 12-step plan with probes, and 11 recorded disagreements including three judges picking
  three different winners. The ENOSPC blocker row is falsified today (fresh native store created
  2/2); eleven closed-in-code rows are still unstruck — reconciliation owed.
- **2026-09-04** — `c413c55`: resident-agent step 10 merged (three reviewed worktree lanes): `engram
  setup` asks for no key; the chat route answers 501 and `engram serve` no longer imports the SDK;
  hyde, the triage/infer/merge LLM passes, the server-side edge adjudicator and the EntityExtractor
  body are deleted with 23 knobs retired through the warning mapper; docs rewritten (SKILL.md
  changed → clawhub republish owed). Suite 5,521 green. Also landed since the export: `engram
  backup export`/`import` (e42a3ff, ca81741), the capture-time machinery gate (0657615), the
  session-start bootstrap gated off, the ledger reconciliation (c044ae5). Open: the embedding
  default flip (in flight), the fresh-brain switch (blocked on Konner's `!` block), the export's
  re-seed and re-measure.
- **2026-09-04 (evening)** — **the fresh-brain switch is done.** Export verified (8,695 rows: 6,131
  conversation / 1,021 machinery / 620 session / 228 probe / 695 bootstrap; 953 entities, 1,751
  edges, 7,618 cues, 12 identity-core) at `~/.engram/exports/dogfood-axi-20260904` and a copy under
  `~/.helix/backups/`; Konner ran the switch block (17 GB manifest-less backup + May-20 dir
  deleted, sidecars parked in `~/.engram/retired-20260904`, `ENGRAM_HELIX__DATA_DIR` →
  `~/.helix/engram-native-2026-09`, installed hook truncation marker). The old 7.7 GB dir and both
  Sep-3 backups are untouched. Re-seed of the 6,131 conversation rows through the capture path
  (`engram backup import`, 5/s) in progress; 48 h machine probe sampling to
  `~/.engram/logs/machine-probe.jsonl`. Two traps found on the way: `uv sync` pruned the
  undeclared native extension (fixed: `be3ae9e`, helix-native is a uv path source), and
  `str.splitlines` broke the export on U+2028 (fixed: `6c1f78f`). `test_loop_ritual` ran LIVE
  against the fresh shell during a suite run and promoted a test entity (deleted); §2.12 stands.
  Step-0 baseline of the OLD brain was aborted as unmeasurable (swap 4.8/5 GB) — owed later from
  the retained dir on a quiet machine.
