# Recall Performance Engineering Plan

**Status:** DESIGN — investigation complete, ready for a focused build effort.
**Goal it unblocks:** `docs/product/AGENT_EXPERIENCE_GOAL.md` battery (B1) — episode-ANSWER recall on the 17 GB dogfood brain (helix native PyO3, ~9.2k episodes + chunk vectors).
**Author:** synthesis of five investigation lanes (perf-autopsy, pipeline-arch, vector research, sub-doc research, restructure options).
**Brain under test:** `~/.helix/engram-native-dogfood-axi`, 18.2 GB `data.mdb`, 9209 episodes / 833 entities, 768-d nomic vectors.

---

## 1. ROOT CAUSE — why native recall is slow

### Verdict: the "~985 ms HNSW vector search" is a MEASUREMENT MISLABEL. It is a fixable perf/architecture bug, NOT a fundamental HNSW limit.

Native HNSW `SearchV` measured on the clone (one process): **1.3–3 ms warm, 80–157 ms cold** — ~300–700× faster than the alleged 985 ms. The anomaly premise ("HNSW is 100× too slow") is **refuted**. What the pipeline labels `recall_primary_search` is a *hybrid* — vector ∥ BM25 → RRF — and the seconds live in **BM25 + page-faulting + redundant work**, not ANN traversal.

### Measured per-stage breakdown (clone, 9209 ep / 833 ent / 768-d)

| Stage | Warm | Cold (fresh process) | Source |
|---|---|---|---|
| Engine construct (18 GB mmap open) | — | 2.2 ms | — |
| FastEmbed model load | — | 17,227 ms (once/process) | not per-query |
| Query embed (nomic 768-d) | 13 ms | 14 ms | `search.py` `_embed_text` |
| **Native SearchV — episode HNSW** | **1.6 ms** | **157 ms** | `search.py:1123` |
| Native SearchV — entity/cue/chunk | 1.3 ms | 1.3 / 1.5 / 30 ms | — |
| SearchV k=1/10/50/100 | 1.1 / 1.3 / 2.6 / 4.0 ms | — | ef scales fine |
| SearchV episode filtered (prod gid path) | 2.9 ms | 80 ms | `search.py:1110` |
| **Native BM25 — episode** | **58–138 ms** | **2,621–2,757 ms** | `search.py:1238` |
| get_stats fast-count (gid) | 241 ms | 240–292 ms | `graph.py:2522` |
| get_stats full-scan / gid=None | 678 ms | 1,203–1,583 ms | `graph.py:2354` |
| **FULL hybrid `search_episodes()`** | **134 ms** | **2,600 ms+** | `search.py:1671` |
| **FULL hybrid `search()` (entity)** | **136 ms** | **2,600 ms+** | `search.py:1603` |

### The four compounding mechanisms (ranked by leverage)

1. **BM25 is the dominant cost, and it page-faults perpetually.** BM25 episode search is 58–138 ms warm but **2.6 s cold**, because the brain (18.2 GB `data.mdb`) is **larger than physical RAM (17.2 GB)** — the LMDB mmap cannot stay resident, so BM25's scattered posting-list reads perpetually major-fault (~1 ms/page over storage). Every hybrid lane co-schedules BM25 via `asyncio.gather(self._embed_text, self._bm25_search_*)` (`search.py:1603,1671,1775`) and awaits BOTH before the vector search, so a lane's wall = `max(embed, BM25) + vector` — **BM25-bound whenever the breaker is closed or half-open probing** (`search.py:280`, re-probes every 300 s).
   - **Live proof:** BM25 circuit breaker is **OPEN right now** on the dogfood brain — `~/.engram/bm25-breaker-state.json` shows `engram-native-dogfood-axi` `last_elapsed_ms 349.0`, opened today. BM25 blew its budget and tripped, exactly as measured.

2. **`ef_search = 512` is 4–13× too high** for this N. `server/engram/storage/helix/config.hx.json:5` → `"ef_search": 512` (`m=16`, `ef_construction=200`). HelixDB's stock default is worse (768). For **~10–20 k vectors, ef_search should be 48–128.** ef_search is the query-time recall/latency knob and scales cost ~linearly in candidates explored ([Milvus HNSW params](https://milvus.io/ai-quick-reference/what-are-the-key-configuration-parameters-for-an-hnsw-index-such-as-m-and-efconstructionefsearch-and-how-does-each-influence-the-tradeoff-between-index-size-build-time-query-speed-and-recall)). At 18 k nodes, ef=512 explores a large fraction of the whole graph — thousands of node visits × M=16 links — and **each hop is a random LMDB page fault** ([Meilisearch/Hannoy: ~25% of query time is `mdb_get`, "search speeds up as more of the DB is in the page cache"](https://blog.kerollmops.com/from-trees-to-graphs-speeding-up-vector-search-10x-with-hannoy)). ef and page-faults MULTIPLY. Warm this is invisible (1.6 ms); cold + memory-pressured it is the difference between 30 ms and 157 ms — and it is free to fix.

3. **The query is embedded 4× per recall.** `search()` (`search.py:1603`), `search_episodes()` (`1671`), `search_episode_cues()` (`1775`), `search_episode_chunks()` (`1904`) each call `self._embed_text(query)` independently. `_last_query_vec` is overwritten, never reused. That is 4 × 13 ms = ~52 ms of pure redundant embed, serial across lanes.

4. **Budget arithmetic guarantees starvation even with zero timeouts.** Serial substage caps sum to primary(1500) + episode(250) + cue(150) + chunk(150) = **2050 ms under a 1500 ms wall** (`config.py:191,286,295,322`; wall `budgets.py` `recall_budget_explicit_search_ms=1500`). Chunk is last in line → **structurally unreachable** before any timeout even fires.

### The causal chain to the symptom (chunk never fires → episode-answers miss)

1. Cold/probing BM25 (2.6 s) blows the 1500 ms primary budget → `primary_search_timed_out = True` (`pipeline.py:613`).
2. That flag **hard-gates OFF chunk search**: `pipeline.py:933-934` — `cfg.chunk_search_enabled and not primary_search_timed_out and hasattr(...)`.
3. It also **switches episode/cue to the `_fast` BM25-only variants** (`pipeline.py:798,868`) — the *slowest* native path — so they also time out at 250/150 ms.
4. Recall degrades to project-file fallback @ 0.35; entity-backed questions still HIT via durable-first/rescue, episode-ANSWER questions MISS. Battery stuck at 3–5/10.

**Bottom line: fix BM25 residency + ef_search + redundant embed + the serial ordering, and episode+cue+chunk all fit inside ~50 ms warm, leaving the whole budget for materialize + an optional reranker.** A correct HNSW at this N is 1–3 ms ([HNSW is near-constant in N; 50 k=0.16 ms@ef40, 200 k=0.15 ms; flat brute of 200 k=18 ms](https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/)).

---

## 2. THE STRATEGY

**Highest-leverage fix = neutralize BM25 as a synchronous blocker AND stop gating chunk on the primary-search clock.** Not one lever — a two-punch, because the symptom is a *product* of "BM25 blows the clock" × "the clock gates chunk." Kill either factor and chunk survives; kill both and warm recall drops under 200 ms end-to-end.

Justification from the evidence:
- The vector index is already fast (1.3–3 ms). No ANN rearchitecture is warranted — Lanes C and D both independently conclude "no architecture change needed; a correct index does 1–3 ms."
- BM25 is the measured seconds-scale cost and is *already tripping its own breaker in production*. It contributes little to episode-answer recall (semantic lane does the work) but dominates latency and, via the timeout flag, *causes the chunk gate to close*.
- The chunk gate (`pipeline.py:934`) couples the answer-locality lane to the entity-search clock for no semantic reason — Steps 1.1/1.2/1.3 only need `query`, not entity results (Lane B, Lane E).

So the plan sequences: (M1) make ef_search + embed-cache free wins, (M2) decouple BM25 from the critical path, (M3) break the chunk gate + parallelize 1.1/1.2/1.3, (M4) fix the budget arithmetic, then (M5) the memory-residency structural fix. Cheapest highest-impact first.

---

## 3. SEQUENCED MILESTONES

Every milestone is flag-guarded and A/B'd against the live continuity gate (`uv run engram continuity --against-live --organic`) and the agent-experience battery. Land in order; each is independently revertible.

### M1 — Free config wins: ef_search + query-vector cache *(land first)*
**Change:**
- `config.hx.json:5` `ef_search: 512 → 96` (start conservative; sweep 48/64/96/128 for recall@k parity). Requires index reload, no rebuild.
- Add a per-recall query-vector memo: embed once, reuse across `search`/`search_episodes`/`search_episode_cues`/`search_episode_chunks` (thread `_last_query_vec` or a passed vector through the 4 call sites `search.py:1603,1671,1775,1904`).

**Expected impact:** cold entity/episode vector 157 ms → ~30–60 ms; redundant embed −40 ms/recall. Warm recall largely unchanged (already fast) but cold-path tail shrinks ~2×. Latency: −40 to −100 ms typical, more on cold.
**Battery impact:** low direct, but reduces the frequency of `primary_search_timed_out` firing → chunk survives more often. Est. +0–1 battery indirectly.
**Continuity-gate risk:** LOW. ef_search=96 must hold recall@10 parity — verify with the deep-recall harness (42/42 gate) before/after. Query-cache is pure memoization, semantically identical.
**Flag guard:** `ENGRAM_ACTIVATION__RETRIEVAL_HNSW_EF_SEARCH` (new, defaults to current 512 until A/B passes); query-cache behind `retrieval_query_vec_cache_enabled=False`.
**A/B:** recall@10 parity + latency delta on the 42-question gate + battery spot-check.
**Effort:** S (0.5–1 day). ef is a one-line JSON + reload; cache is ~30 lines.

### M2 — Take BM25 off the synchronous critical path
**Change:** stop `await`-ing BM25 inside the hybrid lanes when the breaker is open OR when the vector lane already returned enough candidates. Options (pick after spike): (a) fire BM25 with its own short timeout and RRF-merge only if it returns before the vector lane finishes; (b) demote native BM25 to a best-effort background enrichment that never blocks the primary result; (c) keep the breaker but shorten its budget so it trips fast and stays open under memory pressure. `search.py:1603,1671,1775` gather sites; breaker `search.py:280`.
**Expected impact:** removes the 2.6 s cold / 130 ms warm BM25 tax from the primary clock → `primary_search_timed_out` stops firing on cold pages. This is the single biggest latency win: hybrid `search_episodes()` 2,600 ms → ~60 ms cold.
**Battery impact:** HIGH — this is the mechanism that closes the chunk gate. Est. +2–3 battery.
**Continuity-gate risk:** MEDIUM. BM25 contributes lexical recall for exact-term queries; dropping it could regress name/keyword hits. Mitigate: durable-first exact-name rescue (`recall_surface.py:1471`) already covers the lexical-lookup shape; verify continuity gate holds with BM25 demoted. Keep BM25 for the entity lane if needed, drop it from episode/cue/chunk.
**Flag guard:** `retrieval_bm25_blocking_mode` = `blocking` (current) | `best_effort` | `off`.
**A/B:** continuity gate 42/42 must not regress; measure lexical-query subset separately.
**Effort:** M (1–2 days).

### M3 — Break the chunk gate + run 1.1/1.2/1.3 concurrently
**Change:**
- Remove `and not primary_search_timed_out` from `pipeline.py:934`. Chunk search only needs `query` — gate it on `chunk_search_enabled` alone.
- Dispatch episode (1.1), cue (1.2), chunk (1.3) as one `asyncio.gather` instead of the current serial awaits (`pipeline.py:807-833 / 877-899 / 939-955`). The concurrency substrate exists and is safe: native transport runs on `ThreadPoolExecutor(max_workers=4)` (`config.py:78`, `native_transport.py:125,155,208`), and `generate_candidates` already gathers 3 pools (`candidate_pool.py:858-893`). Prefer the engine-side `batch()` (`native_transport.py:324`) — one Python↔Rust round-trip for all three vector lanes.
**Expected impact:** with M1/M2, all three lanes are 1.6–30 ms warm; run in parallel the sub-document lane costs ~max(three) ≈ 30 ms instead of serial 550 ms of budget. Chunk FIRES on the episode-answer queries that currently miss.
**Battery impact:** HIGH — this is the direct fix for episode-ANSWER MISS. Est. +2–3 battery (the B1 unblock).
**Continuity-gate risk:** LOW — adds a lane, removes a false gate; does not touch the durable-first/rescue path that carries the gate (`recall_surface.py:644-651,830-934`, upstream of the deep pipeline). Verify GIL is released in the Rust `engine.query` so gather is true parallelism (see Open Questions).
**Flag guard:** `retrieval_chunk_gate_on_primary_timeout` (default True → set False); `retrieval_subdoc_parallel` (default False → True).
**A/B:** battery episode-answer subset (the stuck questions) + continuity 42/42 + latency.
**Effort:** M (1–2 days).

### M4 — Fix the budget arithmetic so substages fit the wall
**Change:** the serial caps sum (2050 ms) exceeds the 1500 ms wall. With M1–M3 the real warm cost is ~200 ms, so lower the substage caps to reality: primary 1500→600, episode 250→120, cue 150→80, chunk 150→80 (`config.py:191,286,295,322`) and raise the explicit search wall headroom only if needed. Ensures no lane is starved by paper budget.
**Expected impact:** deterministic budget fit; removes the "structurally unreachable chunk" condition even on the slow tail.
**Battery impact:** MEDIUM — locks in M2/M3 gains against tail latency. Est. +0–1.
**Continuity-gate risk:** LOW-MEDIUM — tighter caps could clip a genuinely slow cold lane; keep a one-shot warmup (already present: `f51d10d background continuity warmup`) so first-query cold cost is amortized.
**Flag guard:** all four are existing config fields; A/B via env override.
**A/B:** p95 latency + battery + continuity, cold-start included.
**Effort:** S (0.5 day, mostly tuning).

### M5 — Structural: keep the working set resident (mmap > RAM problem)
**Change:** the brain (18.2 GB) > RAM (17.2 GB) is the deep cause of BM25/HNSW cold faults. Options, in preference order: (a) split BM25 posting lists and HNSW graph pages into a smaller companion store so the hot index fits RAM independent of the 18 GB episode payload; (b) `madvise(WILLNEED)`/prefetch the HNSW graph + BM25 postings at engine open (warmup already exists for continuity — extend it to index pages); (c) shard the brain by group_id so any single recall touches a resident subset; (d) compress/quantize vectors (PQ/SQ) to shrink the resident footprint. This is the durable fix; M1–M4 make the system fast *warm*, M5 makes cold acceptable.
**Expected impact:** cold recall 2.6 s → sub-200 ms; removes the memory-pressure eviction that reopens the whole problem after idle.
**Battery impact:** MEDIUM (tail/cold reliability) — protects the M1–M4 wins over a long session.
**Continuity-gate risk:** MEDIUM-HIGH if sharding/companion-store touches the write path — sequence LAST, behind its own flag, with full continuity + battery + backup validation.
**Flag guard:** dedicated per-option flag; ships dark until proven.
**Effort:** L (3–5+ days; (b) prefetch is the cheapest first cut, ~1 day).

**Ordering rationale:** M1 is free and lowers timeout frequency. M2 removes the seconds-scale blocker. M3 is the direct B1 unblock and depends on M2 (so the gate it removes doesn't immediately re-trip). M4 locks the budget. M5 is the durable structural fix, highest risk, last.

---

## 4. WHAT WE ALREADY RULED OUT

- **Blaming HNSW / rearchitecting the vector index.** REFUTED by measurement: native SearchV is 1.3–3 ms warm, 80–157 ms cold — 300–700× faster than the "985 ms" label. The 985 ms was the *hybrid* wall (BM25 + faults + redundant embed), never the ANN. Do not swap the index, do not tune M/ef_construction beyond M1's ef_search sweep, do not add a second vector store for speed. ([HNSW near-constant in N](https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/))
- **The reverted durable-first cap.** A prior attempt capped/short-circuited the `durable_entity_first` rescue (`recall_surface.py:644-651,1385`, timeout `min(0.75, max(0.2, 1500*0.4))=600 ms`, wall 2× → 1.2 s) to reclaim wall time before the deep pipeline. It was reverted because: (a) durable-first is the path that carries the continuity gate (B3, 819 ms live) and the entity-backed HITs — capping it regressed the questions that currently PASS to buy time for the ones that miss; (b) it attacked *serial tax before* the pipeline, not the *actual blocker* (BM25 inside the pipeline). Net: it traded a passing lane for a failing one. **Do not re-cap durable-first.** The correct target is BM25 + the chunk gate (M2/M3), which are orthogonal to durable-first and leave the continuity path untouched (it runs upstream of the deep pipeline in `recall_surface.py`).
- **Raising the wall/timeouts to "let chunk finish."** Rejected — it makes recall slower without fixing the cause; the budget already exceeds the wall (M4) and the cost is BM25, not chunk. Recall is pure dead time before the model speaks and must stay under the 1 s flow threshold ([Nielsen response-time limits](https://www.nngroup.com/articles/response-times-3-important-limits/)).

---

## 5. OPEN QUESTIONS / UPSTREAM (helix-py) ITEMS

1. **Does the Rust `engine.query` release the GIL (PyO3 `allow_threads`)?** M3's parallel gather is only true parallelism if it does. If it holds the GIL, the 4-worker `ThreadPoolExecutor` (`native_transport.py:125`) serializes and M3 must use the engine-side `batch()` (`native_transport.py:324`) instead. **Verify in the helix-py PyO3 binding before building M3.** (Lane A/E flagged; not yet confirmed.)
2. **Is `ef_search` runtime-tunable or index-baked?** `config.hx.json` is read at index open — confirm changing ef_search takes effect on reload without a full HNSW rebuild (measurement suggests reload-only, verify on the clone).
3. **Native BM25 pathological cold cost (2.6 s / 20 s):** is this a HelixDB posting-list layout issue fixable upstream (contiguous postings, mmap `madvise`) rather than worked-around in Engram? File an upstream perf issue with the profile. Reference the Meilisearch LMDB lazy-read patch as prior art ([patching LMDB for 3× faster vector store](https://blog.kerollmops.com/patching-lmdb-how-we-made-meilisearch-s-vector-store-3x-faster)).
4. **`get_stats` full-scan 1.2–1.6 s cold (`graph.py:2354`):** Step 0 stats runs before entity gen and eats budget. Can it always use the fast-count gid path (`graph.py:2522`, 240 ms) or be cached per group_id? Confirm no correctness dependence on the full scan.
5. **HNSW graph + BM25 posting prefetch at open (M5b):** does helix-py expose a warmup/`WILLNEED` hook, or must Engram touch pages via a dummy query? Cheapest cold-start fix — scope with upstream.
6. **Recall self-reports `budget.durationMs=0.17ms, skipReason=cache_satisfied` while the HTTP call takes 3.6 s** (Lane A live obs): seconds are inside `build_api_recall_surface` but OUTSIDE the metered deep-recall window. Locate the unmetered pre-deep cost (likely durable-first + preflight + stats, `recall_surface.py:644-702`) and bring it under the meter so the budget number is honest.

---

### Success criteria for the next build effort (B1 achievable)
- Warm end-to-end recall p95 ≤ 300–500 ms, hard ceiling 1 s (matches Mem0 ~80 ms p50 / Zep p95 300–632 ms shipping baselines — [2026 agent-memory benchmark](https://dev.to/varun_pratapbhardwaj_b13/5-ai-agent-memory-systems-compared-mem0-zep-letta-supermemory-superlocalmemory-2026-benchmark-59p3)).
- Chunk search FIRES on episode-answer queries (verify `stage_timings_ms` shows `search_episode_chunks` non-null).
- Continuity gate 42/42 holds; battery episode-answer subset flips MISS→HIT; overall battery 3–5/10 → 7+/10.
