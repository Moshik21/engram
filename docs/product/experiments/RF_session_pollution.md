# RF — Session-Pollution Investigation (state-clean 31-32 vs session-shaped 15-17)

**Date:** 2026-07-17
**Question:** M4.3 measured reach@10 31-32 state-clean vs 15-17 session-shaped on the
same brain+queries, and a 56→61 candidate-pool drift across identical consecutive
recalls with `record_access=False`. Which wave2 session mechanism, and is it design
or defect?

## Verdict (one paragraph)

**DEFECT, single mechanism, fully isolated: working memory leaks surfaced *episode*
IDs into the *entity* candidate pool, and the MMR embedding-coverage window
amplifies +5 phantom candidates into a 2x reachability loss.** Disabling only the
episode-typed WM entries (arm S_NOEPIS) recovers session reach@10 from **15/36 to
32/36 — exactly state-clean parity** — while keeping entity-priming live. It is NOT
conversation-context topic shaping (all wave2 conv features contribute zero: W2
arms byte-match their non-W2 counterparts), NOT WM spreading seeds (seed-energy-0
changes nothing), and NOT a mis-measurement of held-out paraphrases (the displaced
candidates — previous queries' episode IDs and diversity-unpenalized wrong-person
entities — have no topical relationship to the query). Working memory is enabled by
default even with `recall_profile=off` (config.py:523), so this is live on every
install whose recall surfaces results in one process session.

## Setup

- Brain: byte-copy of the M4.3/M3.1 oracle brain (lite/SQLite + local FastEmbed,
  102 episodes / 72 entities / 36 edges, 36 planted bridge questions). Fully
  local, no keys, no judge.
- All arms: `ts_enabled=False` (removes TS churn), `passage_first_entity_budget=3`,
  `limit=10`, `record_access=False`, metric = reachability@5/@10 of the gold
  bridge episode + candidate-pool size per recall (spy on
  `candidate_pool.generate_candidates`).
- Session-shaped = ONE manager runs all 36 queries in order; state-clean = fresh
  manager per query. Monkeypatches are script-level module-attr patches; no server
  code edited.
- Rig: `<scratchpad>/experiments/rf_session_pollution/run_bisect.py` (+
  `diag_q1.py`, `diag_wm.py`, `diag_mmr.py`); artifacts `arm_*.json`,
  `summaries.json` in the same dir.

## Bisection arms (n=36)

| arm | change | reach@5 | reach@10 | mean rank | mean pool (first→last) |
|-----|--------|---------|----------|-----------|------------------------|
| FRESH (state-clean ref) | fresh manager per query | 21 | 32 | 5.38 | 37.3 (36→36) |
| S_BASE (session) | none — shipped defaults | **8** | **15** | 6.13 | 47.7 (36→48) |
| S_BASE_X2 | each query recalled 2x (M4.3 shape) | 8 | 15 | 6.13 | 47.4 |
| S_WM_OFF | `working_memory_enabled=False` | **21** | **32** | 5.38 | 37.3 |
| S_SEED0 | WM on, `working_memory_seed_energy=0` | 8 | 15 | 6.13 | 47.7 |
| S_NOPOOL | WM on, `_working_memory_pool`→[] (membership off, seeds on) | **21** | **32** | 5.38 | 37.3 |
| S_NOEPIS | WM on, drop only `item_type=="episode"` WM entries | **21** | **32** | 5.38 | 42.5 |
| S_NOWRITE | `RecallWorkingMemoryUpdater.add_result` no-op | 21 | 32 | 5.38 | 37.3 |
| W2_FRESH | `recall_profile="wave2"`, fresh per query | 21 | 32 | 5.38 | 37.3 |
| W2_S | `recall_profile="wave2"`, session-shaped | 8 | 15 | 6.13 | 47.7 |

Reading:

- **Reproduction exact:** FRESH 21/32 matches M4.3's state-clean B3 (21/32);
  S_BASE 8/15 matches the session-shaped 15-17 band. Gap = 17 of 36 gold episodes
  lost (32→15 @10).
- **Working memory is 100% of the gap** (S_WM_OFF fully recovers), via **pool
  membership only** (S_NOPOOL recovers; S_SEED0 does not move a single query —
  WM spreading seeds are causally inert here).
- **The episode-typed entries are the whole defect** (S_NOEPIS recovers all 17
  queries while still injecting entity-typed WM candidates — pool still grows to
  42.5 mean).
- **Wave2 conversation machinery contributes zero** in recall-only sessions:
  W2_FRESH ≡ FRESH and W2_S ≡ S_BASE to the decimal. Structural reason: recall
  queries are ingested with `update_fingerprint=False` and a non-live source
  (context.py:469-475, 95-96), and session entities are only written on the
  capture path (extraction/apply.py:866) — so fingerprint/session-entity/near-miss
  state stays empty when the session consists of recalls.

## Pool-drift probe (same query 3x, one manager)

| probe | pool sizes | added 1→2 (kind) | added 2→3 | surfaced ids stable 1v2 | scores stable |
|-------|-----------|-------------------|-----------|--------------------------|---------------|
| defaults | 36→42→42 | +6 (5 episode, 1 entity) | 0 | yes | **no** |
| noepis | 36→37→37 | +1 (entity) | 0 | no (entity churn) | no |
| wm_off | 36→36→36 | 0 | 0 | yes | **yes** |

**Answer to "additive growth or score mutation": BOTH, with additive growth
dominant and saturating.** Recall N's surfaced ids (5 episodes + up to 3 entities
+ entity 1-hop neighbors at 0.5x) join recall N+1's entity pool; growth stops when
WM's LRU (capacity 20, ttl 300s) already contains them (added 2→3 = 0; session-long
pool plateaus at 48-50 from 36). Secondary score mutation: even with identical
pool ids, scores drift between consecutive identical recalls (WM recency decay
shifts RRF/seed contributions) — this is what M4.3's positional TS draws amplified
into 0/36 repeat stability. M4.3's "56→61" was the same +5-episode leak measured
one stage later (post-scoring candidate list; my session probe shows 63 there).

## Mechanism (traced end-to-end, cited)

1. **Write:** recall materialization adds every surfaced episode to WM with
   `item_type="episode"` (primary_results.py:187-194) and every surfaced entity
   (primary_results.py:301-308). Neither write is gated on `record_access` —
   only ACT-R access recording is (primary_results.py:285). "Read-only" recall
   is not read-only for session state.
2. **Leak:** `_working_memory_pool` iterates `working_memory.get_candidates()`
   and inserts **every item id into the entity candidate pool, ignoring
   `_item_type`** (candidate_pool.py:203-205; same type-blindness in the
   single-pool injection at pipeline.py:543-548 and the WM seed step at
   pipeline.py:1190-1197). Five episode IDs from the previous query enter the
   entity pool at recency score 1.0.
3. **Amplifier:** MMR fetches embeddings only for `scored[:retrieval_top_n*2]`
   (top 20; pipeline.py:1917-1922) and gives candidates *without* an embedding
   `max_sim = 0.0` — i.e. **zero diversity penalty** (mmr.py:73-80). The phantom
   episode ids occupy top-20 window slots, pushing real mid-score entities out of
   embedding coverage; those now-unpenalized entities leapfrog the
   diversity-penalized topic entity.
4. **Kill:** the topic entity falls below the 3-slot entity budget, so the
   entity→episode traversal channel (which surfaces gold at parent×0.6) expands
   wrong-person episodes instead. Concrete q1 trace: FRESH entity slots =
   [Thaddeus 0.52, Marisol 0.31, **sourdough starter 0.47**] → gold at rank 5
   (0.4674×0.6=0.2804). SESSION (after one unrelated query) MMR order flips to
   [Thaddeus, **Tilda 0.3139, Hideo 0.3111**, Idris, Abelard, sourdough…] —
   sourdough (score 0.4674!) demoted below the budget cut by two lower-scored,
   diversity-unpenalized entities; its traversal never fires; the two wrong
   entities append their own episodes at 0.188/0.187; gold gone from top-10.
   (diag_mmr.py output; MMR *input* top-14 ids AND scores are identical between
   fresh and session — the flip is purely selection-stage.)

## Design vs defect — explicit call

- **DEFECT (gear misalignment):** the episode-ID type leak (steps 2-3). Episode
  ids in an entity pool have no topical-shaping semantics — they are phantom
  candidates that can never materialize as entities, and their damage lands on
  whatever query comes next, related or not. 100% of the measured reachability
  loss (17/36 queries) is this leak. The held-out-paraphrase framing is NOT the
  explanation: nothing about the displaced candidates is session-topical.
- **DESIGN, behaving acceptably:** entity-typed WM priming (membership +
  1-hop neighbors + seeds). With the leak filtered (S_NOEPIS) it costs zero
  reachability on 36 topically-unrelated back-to-back queries (21/32 = clean)
  while producing mild rank churn on repeats (noepis probe: surfaced ids not
  byte-stable). That churn is the designed session-shaping signal, small and
  currently harmless. Keep, with the determinism caveat below.
- **DEFECT (secondary, contract):** WM writes ignore `record_access`
  (primary_results.py:187,301), so even `record_access=False` recalls mutate
  ranking state — this is the root of "recall mutates pool with
  record_access=False" and the reason M1.2's determinism contract holds only for
  state-identical recalls. `should_record_ranking_feedback` (request_policy.py:28,
  service.py:93-96) gates TS feedback but nothing gates WM.
- **DEFECT (latent amplifier, independent):** MMR's fixed 20-id embedding window
  + zero-penalty-for-missing (pipeline.py:1917, mmr.py:73-80) converts ANY pool
  inflation (this leak, activation candidates, future features) into selection
  corruption. This is the same failure shape as M4.1's activation collapse:
  pool contamination → entity-slot eviction → traversal channel death.

## Fix shape (per mechanism)

1. **Type-filter WM consumption (the fix; ~3 lines, proven by S_NOEPIS):** skip
   `item_type=="episode"` entries in `_working_memory_pool`
   (candidate_pool.py:203), the single-pool injection (pipeline.py:546), and the
   WM seed step (pipeline.py:1192). Keep the episode entries in the buffer
   (get_recent_queries/context uses are unaffected). Expected: session reach@10
   15→32 on this rig; re-run `run_bisect.py` arms FRESH/S_BASE/S_NOEPIS as the
   regression gate.
2. **Gate WM writes on `record_access`** (primary_results.py:187,301) or add an
   explicit `session_state=False` recall kwarg — restores the read-only-recall
   determinism contract (probe: scores byte-stable with WM off, unstable with).
   Benchmarks/evals pin `record_access=False`, so today they measure a
   *state-clean* system that no live session experiences; either gate the writes
   or make evals session-shaped, but stop measuring one system and shipping the
   other.
3. **Harden MMR embedding coverage** (independent, smaller): fetch embeddings
   for all MMR candidates, or give missing-embedding candidates a conservative
   penalty (not a free pass), or at minimum exclude non-entity ids before the
   window is cut (pipeline.py:1917). Without this, the next pool-inflating
   feature reproduces the same collapse.
4. **Keep** entity-typed WM priming as designed (no cap needed at current
   capacity 20/ttl 300s — pool growth saturates at +6-14 and costs no
   reachability once the leak is filtered).

Not exercised here (no cues in this brain, recall-only sessions): cue-feedback
promotion writes on the read path (service.py:179-185) and conv-fingerprint
pollution from live capture turns — both worth separate small probes; neither can
explain the M4.3 gap (this rig reproduces it exactly without them).

## Reproduction

```
SCRATCH=<scratchpad>
HOME=$SCRATCH/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=$SCRATCH/experiments/fastembed-cache \
uv run python $SCRATCH/experiments/rf_session_pollution/run_bisect.py   # 10 arms + 3 drift probes
uv run python $SCRATCH/experiments/rf_session_pollution/diag_mmr.py     # MMR flip trace (q1)
uv run python $SCRATCH/experiments/rf_session_pollution/diag_wm.py      # WM contents + pool diff
```
