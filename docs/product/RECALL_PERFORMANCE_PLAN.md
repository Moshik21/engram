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

### Measured and ruled out — 2026-07-24: the cross-encoder re-ranker (BOTH paths)

**Do not re-try re-ranking as an answer-locality fix.** Reordering cannot add an answer that retrieval never returned, and measurement shows that is exactly the situation.

**Correction to the record first.** The note at `config.py:457-467` claimed flipping `reranker_provider` `noop`→`local` was "measured as a regression (battery 3/10, lost flip-condition and durable-lane)". That reading was invalid: the installed shell runs `consolidation_profile=quiet`, and the quiet preset force-disables the whole stage at `config.py:3020` (`_set("reranker_enabled", False)`). Effective live config was `enabled=False provider=noop episodes=False` — the provider value was inert, and 3/10 is inside the known 3–6 battery jitter. Arming the experiment at all required flipping `reranker_enabled` back on for `quiet`.

**Arms measured** (live dogfood brain, 9248 episodes / 834 entities; battery `--against-live`, medians of 3+ runs):

| Arm | Battery runs | Median | Battery wall | Continuity |
|---|---|---|---|---|
| Baseline (`noop`, stage off) | 1, 3, 3, 4, 3 | **3/10** | 8.8–17.0 s | PASS, recall_ms 328.7 |
| A — `provider=local` + `rerank_episodes=True` (entity+episode, fetch-all) | 0, 2, 4 | **2/10** | 47.7–92.3 s | PASS, recall_ms 495.9 |
| B — type-separated, fetch-bounded episode rerank | 3, 2, 1 | **2/10** | 22.8–68.0 s | not run (arm already failing) |
| Baseline re-verified after revert | 2, 3, 3 | **3/10** | 12.9–22.7 s | PASS, recall_ms 314.7 |

**Arm A never ran the cross-encoder.** The substage budget is `retrieval_reranker_timeout_ms=75` (`config.py:483`), but the ON path (`pipeline.py:1920-1978`) fetched every candidate before scoring — ~30 `get_entity` + ~50 `get_episode_by_id` round-trips on the native transport. Diagnostics: at 600 ms it reported `recallRerankerTimeout: 601 ms`; with the substage timeout disabled entirely it was killed by the parent budget at `recallRerankerCancelled: 824 / 1663 / 2035 ms`. The model itself is fine and fast — a standalone `TextCrossEncoder` load is 4.3 s (once, lazy) and scoring 35 mixed-length docs is 67–73 ms. **The cost was the document fetch, not the model.** Arm A's battery loss is therefore not a ranking effect at all; it is the ~600 ms/query of wasted substage plus abandoned background fetches contending with the native engine (battery wall 8.8–17 s → 47.7–92.3 s).

**Arm B** made the path executable — type-separated (entity order untouched, so the durable reserved lane is never exposed to the short-snippet effect), capped at the 12 strongest episode/cue candidates, `chunk_context` reused when already in hand, and the candidates' own scores *permuted* into cross-encoder order rather than overwritten with raw logits (which would mix scales with the un-reranked tail). It completed — `recallRerankerEpisodeDocs: 12`, `recallReranker: 507–735 ms` — but still cost ~500 ms/query (the ≤12 episode fetches, not the ~40–70 ms of scoring) and did not help.

**The kill shot — the answers are not in the candidate set.** Probing the live recall surface at `limit=25` on all ten battery questions returns only **5–7 rows**, and expected-answer containment lands as:

```
flip-condition       rows=7  answer_at_ranks=[]
recall-outage        rows=6  answer_at_ranks=[]
ts-kill              rows=7  answer_at_ranks=[]
north-star           rows=7  answer_at_ranks=[6]
deleted-phases       rows=7  answer_at_ranks=[0]
durable-lane         rows=7  answer_at_ranks=[3]
fastembed-outage     rows=7  answer_at_ranks=[]
vector-write-path    rows=7  answer_at_ranks=[5]
bm25-breaker         rows=6  answer_at_ranks=[]
founder-identity     rows=5  answer_at_ranks=[0]
```

Five of ten questions — including **every** episode-answer question the hypothesis predicted would improve (`recall-outage`, `ts-kill`, `fastembed-outage`, `vector-write-path`, `bm25-breaker`) — have the answer nowhere in the returned set. A re-ranker is a permutation; it cannot create those rows. Of the five where the answer *is* present, two are already at rank 0. **A perfect oracle re-ranker over this pool scores 5/10** — a +2 ceiling that requires flawless ordering, against a measured −1. That is not a lever worth 500 ms/query.

**What this redirects to.** The bottleneck is candidate generation and result depth, not ordering: `limit=25` yielding 5–7 rows is itself the finding. Chase M2/M3 (BM25 off the critical path, break the chunk gate) and the recall-depth question — why the pool collapses to <10 rows — before touching ranking again.

**Measurement caveat, recorded honestly:** a concurrent session modified `storage/helix/native_transport.py`, `storage/helix/graph.py` and `storage/helix/client.py` at 11:13–11:15, inside the arm-A/arm-B window. Those edits may account for part of the latency degradation seen in the arms. They do **not** touch the containment result above, which was re-measured after reverting both arms and reproduces the 3/10 baseline median.

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

---

## 6. M5 MEASURED (2026-07-24) — the brain is 56% dead pages; compaction shipped

M5 above listed four structural options (companion store / prefetch / shard /
quantize) and did not consider the simplest one: **the file is mostly free
pages**. It is. Measured on a full clone of the dogfood brain
(`~/.helix/engram-native-dogfood-axi`, clone-only — the live dir was never
opened), one process at a time.

### 6.1 Page census (raw `mdb_env_stat`/`mdb_stat` over the clone)

```
page_size     16,384            map_size      21,474,836,480 (20.00 GiB)
last_pgno     1,111,897         allocated     18,217,336,832 (16.97 GiB)
freelist entries 8,607          FREE pages    626,793 -> 10,269,376,512 (9.56 GiB) = 56.37%
live pages (27 sub-dbs)         292,304 -> 4.46 GiB
unaccounted                     192,801 -> 2.94 GiB (DUPSORT sub-btree pages mdb_stat omits)
```

Biggest real consumers: `bm25_reverse_index` 164,919 pg / 2.70 GiB (71.9 M
postings), `nodes` 82,938 pg / 1.36 GiB, `vectors` 18,764 pg / 307 MB,
`bm25_inverted_index` 15,711 pg / 257 MB.

### 6.2 Compacting copy (`mdb_env_copy2` + `MDB_CP_COMPACT`)

| | bytes | GiB |
|---|---|---|
| original | 18,217,336,832 | 16.97 |
| compacted | 7,942,291,456 | 7.40 |
| **saved** | **10,275,045,376** | **9.57 (56.4%)** |

**Bloat ratio 2.29x**, 71 s for the raw copy (152 s end-to-end through the CLI,
which also runs two exact full-graph stat scans). Compacted freelist: 0 entries.

**The hypothesis is CONFIRMED in mechanism but the magnitude is 2.3x, not the
~17x the plan guessed.** Real data is 7.4 GiB, not "under 1 GB" — the estimate
omitted the BM25 reverse index (71.9 M postings) and the dupsort secondary
indices. The 7.40 GiB result equals `4.46 + 2.94`, i.e. every "unaccounted"
page was a live dupsort index page, not a leak.

### 6.3 Why it still matters at 2.3x

`hw.memsize` on this machine is **16.00 GiB**. 16.97 GiB cannot stay resident —
eviction is guaranteed, which is the measured root cause of the 2.6 s cold BM25
tax (§1). **7.40 GiB fits, with room for the model and the shell.** The gain is
a threshold crossing, not a linear speedup.

Paired cold measurements (page cache flushed by streaming the *other* file
before each run):

| workload | original | compacted | delta |
|---|---|---|---|
| full-graph scan (`get_stats exact`) | 12,043 ms | 3,405 ms | **3.5x** |
| BM25 cold q1 / q2 | 659.5 / 743.7 ms | 483.8 / 525.3 ms | 1.36x / 1.42x |
| BM25 warm | 26–33 ms | 29–39 ms | none |
| vector cold / warm | 98.3 / 7.9 ms | 108.6 / 7.2 ms | none |

Per-query cold BM25 only improves ~1.4x on this SSD (1.5 GB/s). The durable win
is residency across a session, plus the 3.5x on whole-graph scans.

### 6.4 Integrity evidence

- All 27 LMDB sub-db entry counts byte-identical (71,904,360 / 1,331,961 /
  23,222 / 12,197 / 3,579 / …).
- Real stack (`ENGRAM_MODE=helix`, native, one process): entities 692, episodes
  8930, relationships 276, cue_count 2823, projected_cue_count 717,
  linked_entity_count 7421, `state_counts` (projected 4945, cue_only 3220,
  scheduled 659, failed 19, merged 58, projecting 12, cued 16, queued 1) —
  **identical on both copies**.
- Vector search 10/10 entities+episodes, BM25 hit counts identical across three
  probe queries.
- Write path on the compacted copy: `create_entity` → readback → hard
  `delete_entity` → readback `None`; entry counts returned identical.
- Through the shipped CLI on the same clone: **41 integer counts compared, zero
  mismatches.**

### 6.5 What shipped

`engram backup compact` — a compacting copy that is **verified before it is
swapped in**, never in place, and never destructive (the original is renamed
aside, not deleted).

- `native/helix-repo/helix-python/src/lib.rs` — `HelixEngine.compact(dest_dir)`:
  keeps `Arc<HelixGraphEngine>` on the pyclass and calls
  `graph_env.copy_to_path(dest/data.mdb, CompactionOption::Enabled)` under
  `py.allow_threads`. Graph, HNSW vectors and BM25 share one LMDB env, so a
  single copy reclaims all three. Requires `make build-native`; the CLI says so
  explicitly when the installed extension is older.
- `server/engram/storage/helix/native_transport.py` `compact()` →
  `client.py` `compact()` → `graph.py` `compact()` — thin pass-throughs;
  non-native transports raise `ImportError`.
- `server/engram/backup_cli.py` — `_compact()` plus the pure
  `compare_brain_counts()` verification gate.
- `server/tests/test_backup_compact.py` — 17 tests: the gate (a single lost
  episode or nested projection state is a hard failure; empty stats cannot
  silently pass), the guards (no `data.mdb`, non-empty staging, no disk
  headroom, cross-volume `--apply`, shell up, stale extension), and the
  copy/swap behaviour.

Guards, in order: data dir must hold `data.mdb`; staging dir must be empty;
free space ≥ source size + 2 GiB reserve; `--apply` requires staging on the
same volume (the swap is a rename); exclusive access (shell down + brain flock)
via `require_exclusive_local_access`; counts must match exactly or the swap is
refused and the command exits 1.

### 6.6 Live procedure (NOT yet run against the live brain)

Needs a supervised window and founder greenlight. Roughly 3 minutes of work
plus the copy; 17 GB of free disk required.

```bash
make build-native                       # picks up HelixEngine.compact()
engramctl stop                          # shell must be down
engram backup create --to /Volumes/ext  # external snapshot first
cd server
uv run engram backup compact            # stage + verify, no swap
# read the report: bloat_ratio, saved_pct, verify OK
uv run engram backup compact --apply    # swap in the verified copy
engramctl start
uv run engram continuity --against-live --organic   # gate must hold
# only after the gate passes:  rm -rf ~/.helix/engram-native-dogfood-axi.pre-compact.*
```

Rollback: the pre-compact directory is a complete brain — stop the shell and
rename it back.

### 6.7 Follow-up: `db_max_size_gb` is a latent availability risk

`native/helix-repo/helix-python/src/queries.rs:105` sets
`db_max_size_gb: Some(20)`. The live file is at 16.97 / 20.00 GiB = **84.8% of
map_size** — roughly 3 GiB of churn from a hard `MDB_MAP_FULL` write failure.
Compaction drops that to 37%, which buys time but does not remove the ceiling.
Raising `db_max_size_gb` costs nothing at rest (LMDB does not preallocate) and
should be done in the same rebuild that ships `compact()`. Not changed here:
untested, and out of scope for a measurement task.

---

## 7. M6 MEASURED AND NOT SHIPPED (2026-07-24) — two-stage recall

Built the two-stage retrieval path (retrieve deep, rerank precisely with the
local cross-encoder, return small) end to end, measured it against the
agent-experience battery, and **reverted it**. The mechanism works and is
verified; the battery could not resolve whether it helps, because the
instrument's own variance turned out to be larger than the effect. Recording
the numbers so this is not re-litigated from assumptions a fourth time.

### 7.1 What the investigations established first (these still hold)

- **The answer is usually IN the pool.** For 3 of 5 battery misses the answer
  episode was retrieved, scored, then discarded by the 5-row cut — sighted at
  ranks 10, 19, 42 of ~40-60 candidates. For 2 (`ts-kill`, `flip-condition`) it
  never came back at all, not even when the query *was its own opening
  sentence*, while control episodes returned at rank 1 under the identical
  probe. That is an index/embedding gap, a separate black hole. Ceiling for any
  ranking work: 7/10, not 10/10.
- **The surfaced score could not support a shape-aware cut.** It was purely
  positional (RRF, `0.4 x 61/(60+rank)`), and the two fused lanes' raw scores
  run on *opposite* scales — episode search descends with rank (real cosine),
  chunk search ascends (`_extract_chunk_score`, `pipeline.py`). A
  fraction-of-top cutoff read off that field reads rank, not relevance.
- **All three surfaces are one code path.** REST, MCP and axi converge on
  `_run_explicit_recall_with_budget` with identical budgets, and this install
  has no MCP server configured — the live integration is `engram axi hook-run`
  over REST. The battery's surface IS the agent's surface.
- **The battery reads the TOP 3** (`engram/evaluation/battery.py`, `limit=3`).
  Raising caps alone cannot move it. Depth is necessary but not sufficient.

### 7.2 What was built (reconstructable from here)

- `_rerank_episode_pool` / `_maybe_rerank_episode_pool` in
  `engram/retrieval/pipeline.py`: rank the whole episode/cue candidate pool with
  the FastEmbed cross-encoder and write a squashed 0..1 relevance back onto the
  candidates, so the passage-first top-k cut selects by relevance instead of by
  two incomparable lane scales.
- Affordability, which is the entire difference from the attempts that measured
  negative before (they lost to *latency thrash*, not bad ordering):
  1. **Reuse the chunk passage already retrieved** rather than re-fetching. It
     is by construction the query-local span. Measured live:
     `recallRerankerFetch = 0.9-1.7ms`, `recallRerankerDocCoverage = 1.0`.
     Document assembly is essentially free.
  2. **Cap documents by count AND characters** (40 x 600 chars).
  3. **A substage timeout that can actually finish** (75ms -> 1000ms). At 75ms
     the rerank always timed out and its result was discarded — it cost latency
     and bought nothing, which is what the old "measured and ruled out" verdict
     was really measuring.
- `engram/retrieval/recall_shape.py`: honest N-of-M disclosure
  (`"returned 7 of 44 candidates; ask again with a higher limit if the answer is
  not here"`), an output-**token** budget wired into the RecallBudget's own
  `max_output_tokens` (previously carried and never enforced), and a **relative**
  score cutoff, all floored by `recall_shape_min_results` so they can never
  remove the top-k a caller reads.
- `quiet` profile flip: `reranker_enabled=True`, `reranker_provider="local"`,
  `reranker_rerank_episodes=True`, kill switch
  `ENGRAM_ACTIVATION__RERANKER_PROVIDER=noop`.

Full suite green (5,099 passed), ruff clean, plus 8 new rerank tests and 16 new
shaping tests. Working patch saved off-tree for this session only.

### 7.3 Cross-encoder cost table (measured, reusable)

| documents | chars/doc | warm latency |
|---|---|---|
| 40 | 400 | 418ms |
| 40 | 600 | 360ms |
| 48 | 600 | 442ms |
| 48 | 800 | 632ms |
| 60 | 800 | 857ms |
| 50 | ~2400 (400 words) | 2,650ms |
| 80 | ~2400 (400 words) | 5,033ms |

In-process, contending with the embedder, the 40x600 case measures 520-960ms.
Against a 4000ms recall wall that already spends ~2,900ms, this is the entire
budget envelope. **Depth is bounded by rerank cost, not by search cost.**

### 7.4 The result: the instrument cannot resolve the effect

Interleaved A/B, identical code state, restart + warm before each arm:

| arm | battery runs | median |
|---|---|---|
| OFF (session start) | 4, 3, 3 | 3 |
| OFF (at HEAD 3caba69) | 4, 4, 3 | 4 |
| **OFF (A/B arm 1)** | 4, 4, 4, 4, 4 | **4** |
| **ON (A/B arm 1)** | 3, 4, 4, 4, 4 | **4** |
| **OFF (A/B arm 2)** | 3, 2, 3 | **3** |
| **ON (A/B arm 2)** | 1, 4, 4 | **4** |
| ON (warmed, earlier state) | 5, 5, 5, 4, 4 | 5 |
| ON (post-restart, same code) | 3, 3, 3, 3, 3 | 3 |

The same build measured 5,5,5,4,4 and 3,3,3,3,3 twenty minutes apart, and 1 to 4
inside a single arm. The hit SET changes run to run (`durable-lane`,
`deleted-phases`, `north-star` appearing and vanishing) because rescue lanes
race the wall: `cache_satisfied`, `fast_preflight_hit`, `durable_entity_first`
and `partial_on_timeout` each serve a different response, and only one of them
reaches the deep pipeline at all.

`north-star` looked like a regression in the ON arm. It is not: it is served by
`fast_preflight_hit` with `n=1` and no hit **in both arms** — a lane the rerank
never touches.

**Decision rule applied:** the battery median did not improve (4 vs 4 in the
tightest paired comparison), so the change was reverted despite the mechanism
being verified.

### 7.5 What DID hold up, and is worth rebuilding on

- **`bm25-breaker` moved from never-hit to always-hit.** It hit 5/5 in the ON
  arm and 0/13 across every OFF arm. It is exactly the question the
  investigation flagged as found-but-ranked-low (ranks 19, 36, 44). That is a
  mechanism-attributable win, not jitter.
- **Fresh-agent suite** (a less jittery instrument): engram score 4 -> 5, lift
  vs the project-file control -2 -> -1.
- **Token cost is real and unpaid.** Engram surfaced 79,066 chars against the
  project-file control's 13,275 (before: 56,086) — roughly 6x the context for
  one fewer correct answer. The relative score floor was inert on live
  distributions (top 0.400 / min 0.256, ratio 0.64, far above a 0.05 floor), so
  the shaping disclosed honestly but trimmed nothing.

### 7.6 Two real bugs found on the way — independently actionable

1. **The recall graph gate starves the rerank (and anything else that needs
   episode content).** `GatedGraphStore` blocks `get_episode_by_id` whenever the
   stats/graph probe timed out, which is *most* queries on this brain
   (`recall_stats_timeout` fires routinely). The content fetch was answered with
   `None` in 0.11ms, collapsing document coverage to 0.2-0.55 on exactly the
   questions whose candidates carried no chunk passage. Reading through the gate
   under an explicit bound took coverage to 1.0 and is what moved `bm25-breaker`
   from rank 19/36/44 to rank 1-2. The gate is protecting against unbounded
   latency by silently returning wrong answers to bounded callers.
2. **The pipeline skips Step 5.5 entirely when the entity lane is empty**
   (`pipeline.py`, the `if not candidates:` early return). That is the ordinary
   path for the *default* episode-vector tier
   (`passage_first_entity_budget=0`), so the core tier's reranking — and any
   future stage-5.5 work — is unreachable there. Any rerank must be shared by
   both call sites.

### 7.7 Before re-attempting this: fix the instrument

The battery cannot currently distinguish a +1 effect from noise, and this knob
has now been mis-measured three times because of it. The blocker is not the
ranker; it is that four rescue lanes race the wall and a different one wins each
run. Either pin the lane under test (a `deep_pipeline_only` measurement mode) or
report per-question hit rate over N runs instead of a single median, before
spending more effort on ranking.
