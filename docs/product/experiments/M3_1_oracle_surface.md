# M3.1(a) — Oracle-Surface Experiment (entity->episode traversal from the candidate pool)

**Date:** 2026-07-17
**Question:** Does feeding `append_entity_linked_episodes` from the ranked entity
CANDIDATE pool (instead of final results, which the default entity budget of 0
leaves empty — review §1.3) let graph structure surface answer episodes that
vector search misses? This is the "Melanie -> pottery connected-but-not-surfaced"
fix.

## Verdict

**POSITIVE.** Bridge-question reachability@5 went **2/36 -> 22/36** (5.6% -> 61%)
and reachability@10 **2/36 -> 23/36** (64%) for a retrieval-only config change on
the same brain, at +2.9 ms mean recall latency. 21 of the 23 hits arrived via the
entity-traversal channel (`score_breakdown.entity_traversal=True`).

**Production knob:** `entity_episode_traversal_source="candidates"`
(new `ActivationConfig` field, default `"results"` = today's behavior, i.e.
starved; implemented in this session as the M3.1(b) flag). Pairs with
`entity_episode_max_entities` (K; 10 and 20 scored identically here).

## Setup

- Scratch **lite** brain (SQLite + FTS5 + local FastEmbed `nomic-embed-text-v1.5`,
  768d quantized ONNX). Fully local, no external keys, no LLM judge.
- Ingestion via the real path: `GraphManager.ingest_episode` with
  **client proposals** (harness-as-extractor hard path, `model_tier="opus"`,
  verbatim `source_span`s). Extraction quality is deliberately NOT under test —
  the graph is planted so retrieval surfacing is the only variable.
- Corpus: **102 episodes, 72 entities, 36 edges, 36 bridge questions**
  (36 bridge pairs x 2 episodes + 30 filler; 3 interleaved personas).
- Bridge pattern (planted "Melanie -> pottery"):
  - `ep1`: person **A** + topic **B** co-mentioned; edge `A -PREFERS-> B`.
  - `ep2` (**gold**): a concrete fact about **B**, never mentions A.
  - query: mentions only **A** ("What is the latest news about Marisol's hobby?").
  - Reaching gold requires: query -> A (search) -> spread A->B -> B in candidate
    pool -> entity->episode traversal B -> ep2. Bridge structure verified 36/36
    before scoring (entity links + edges present).
- Metric: **REACHABILITY@K** (gold episode id anywhere in the top-K of the
  returned recall list, K=5,10) + mean rank of gold + wall clock. `limit=10`,
  `record_access=False`, per-arm fresh manager over the same DB.

## Results (n=36 bridge questions)

| Arm | reach@5 | reach@10 | found | mean rank (found) | via traversal | mean ms | p95 ms |
|-----|---------|----------|-------|-------------------|---------------|---------|--------|
| A baseline (default config; source=results) | 2 | 2 | 2 | 4.0 | 0 | 44.5 | 48.4 |
| B source=candidates, K=10 | **22** | **23** | 23 | **2.74** | 21 | 47.4 | 54.4 |
| B' source=candidates, K=20 | 22 | 23 | 23 | 2.74 | 21 | 49.9 | 51.0 |

- Arm A confirms the review §1.3 starvation: traversal is default-enabled but
  never fires (via_traversal=0) because entity budget 0 leaves zero entity
  entries in final results.
- Arm B appends ~10 traversal episodes per recall (mean results 5 -> 14.8);
  gold lands at mean rank 2.74 because the parent topic entity's pool score
  (boosted by spreading from the seeded person) times `entity_episode_weight`
  outranks lexical near-miss episodes.
- Arm C (RRF-vs-cosine scale bucketing) was **skipped** — not trivially
  reachable; K=20 vs K=10 parity shows the residual misses are not a K-cap
  problem anyway (see below).

## Residual misses (13/36 in arm B)

Diagnosis per miss: in 12/13, **neither the person nor the topic entity acted as
a traversal parent** — the bridge endpoints never ranked inside the traversed
candidate pool for that query (spread-backfilled candidates carry raw cosine
scores and lose to lexically-similar distractor entities; the known
RRF-vs-cosine scale mismatch). K=20 changes nothing, so the fix is pool
*scoring* (M3.1 follow-up / review §1.2 scale bucketing), not pool *size*.

## Incidental findings (worth their own tickets)

1. **Free-form predicates are silently dropped on the proposal path.** First run
   used `INTERESTED_IN`; `proposals_to_evidence` hard-caps any predicate not in
   `ALLOWED_CLIENT_PREDICATES` to confidence 0.4 (`predicate_not_allowed`) and
   the edge never commits — the brain ended with **0 relationships** and arm B
   showed zero lift (traversal fired but expanded distractor entities). An
   agent following the MCP prompt with a natural predicate gets a silently
   edge-free graph — same family as the silent-inert bug class. Consider
   mapping non-allowlisted predicates through the canonicalizer instead of
   dropping, or surfacing the rejection.
2. The default FastEmbed cache (`~/.engram/models/fastembed`) was missing
   `onnx/model.onnx` (only `model_quantized.onnx` present) — local embeddings
   were dead until pointed at a repaired cache. Matches review §4's
   "FastEmbed silently broken" note.

## Production wiring shipped with this experiment (M3.1(b))

Default-preserving flag, off (`"results"`) unless opted in:

- `server/engram/config.py` — `entity_episode_traversal_source: "results" | "candidates"`.
- `server/engram/retrieval/pipeline.py` — `retrieve(..., entity_candidates_out=...)`
  exposes the top-K entity candidates (post rerank/MMR, pre budget cut).
- `server/engram/retrieval/service.py` — passes the out-param only when the flag
  is `"candidates"` (stub `retrieve_fn`s and default path byte-identical).
- `server/engram/retrieval/post_process.py` / `episode_traversal.py` — traversal
  consumes the candidate pool when the flag is on, final results otherwise.
- Tests: `server/tests/test_recall_episode_traversal.py` — flag-off ignores a
  provided pool (identical behavior); flag-on traverses the pool with the K cap.

## Reproduction

```
SCRATCH=<scratchpad>/experiments/m31_oracle_surface
HOME=<scratchpad>/fakehome ENGRAM_MODE=lite \
FASTEMBED_CACHE_PATH=<scratchpad>/experiments/fastembed-cache \
uv run python $SCRATCH/run_experiment.py   # ingest + arms A/B (--reuse to skip ingest)
uv run python $SCRATCH/extra_arms.py       # K=20 arm + miss diagnosis (reuses brain.db)
```

Artifacts: `arm_A_baseline.json`, `arm_B_candidates.json`,
`arm_B20_candidates.json`, `tag_to_id.json`, `brain.db` in the scratch dir.

## Recommendation

Flip `entity_episode_traversal_source="candidates"` (keep K=10) as the
graph->episode surfacing channel — additive-only (appended synthetic results
never evict primary episodes), +3 ms mean, and it converts the connected-but-
not-surfaced failure from 94% to 36% miss rate on planted bridges. Land the
candidate-pool score normalization (review §1.2) next to attack the residual
misses.
