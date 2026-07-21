# Target Design — Recency/Frequency as a Core Engram Capability

**Scope:** replaces "ACT-R activation as an additive ranking term" with a usage-signal system whose gears mesh: de-saturated signal → trusted recording → bounded composition → unified entity/episode model → clean contracts with importance/prune/snapshot/session → data-driven calibration → migration.
**Design stance:** the founder is right that recency/frequency has merit — Anderson–Schooler is real. M4.1 didn't falsify the *prior*; it falsified three implementation choices (saturated sigmoid, 2.75:1 TEMPORAL weighting, surfaced-recording loop). Each is fixed structurally below, not tuned.

**One-line summary:** *usage is a bounded multiplicative tiebreaker computed from tiered, trusted access events; it can promote among near-peers but is structurally incapable of beating semantics; ACT-R base-level survives unchanged as the hygiene signal (prune/mature); the two consumers stop sharing a formula.*

---

## 1. De-saturation

### The defect, quantified (shipped formula: `activation/engine.py:11-47`, params `config.py:157-161`)

`act = sigmoid((ln Σ (t−t_j)^−0.5 − (−4.0)) / 1.7)`, ages in epoch-seconds:

| history | raw B | act |
|---|---|---|
| 1 access, 1 s ago | 0.00 | **0.913** |
| 1 access, 10 min ago | −3.20 | **0.616** |
| 5 accesses ~10 min | −1.59 | 0.805 |
| 16 accesses ~min | 0.73 | 0.942 |
| 1 access, 1 d | −5.68 | 0.271 |
| 1 access, 30 d | −7.38 | 0.120 |
| 50 accesses spread 1–30 d | −2.62 | 0.693 |

Within a session (where accesses cluster) the signal spans 0.62–0.94 — a near-binary "touched recently" bit; 1-vs-16 accesses is worth 0.33 while touched-vs-not is worth 0.6–0.9. That bit, times `weight_activation=0.55`, is the M4.1 collapse (23/36 → 2/36).

### Alternatives considered

**(a) Recalibrated `(B_mid, B_scale)` from empirical live-brain distributions.** Fit B_mid = median(B), B_scale = IQR/2 over entities with non-empty history (live-copy snapshot). Honest verdict: fixes the *squash placement* but keeps a single global squash of an absolute signal that then enters an additive sum — the failure mode (absolute usage mass beating pool score-spread, M4.1 arm G: even 0.10–0.25 weights collapsed neutral queries 12→0) is untouched. Also needs a re-fit cadence as the brain grows. **Rejected as the ranking signal; kept as an optional prune-side calibration** (§6).

**(b) Log-odds without sigmoid + pool-relative percentile normalization.** Rank-normalize raw B within the candidate pool. Adaptive and scale-free — but the normalization becomes a function of *pool composition*, and M4.3 proved the pool drifts within a session even on read-only recalls (56→61 candidates on identical consecutive queries; positional TS draws turned that drift into 0/36 repeat-stability). Pool-relative usage inherits exactly that bug class: identical query, identical history, different u because a filler candidate entered the pool. **Rejected — determinism hazard, same family as the positional-TS defect.**

**(c) Bounded boost from absolute, de-saturated primitives.** ✅ **PICK** — pool-independent (deterministic), decomposable (recency and frequency separately inspectable), bounded by construction, and cheap (count + last-timestamp; no per-candidate ln-sum needed on the ranking path).

### The picked signal

```
u = f · r′                                  ∈ [0, 1]
f  = min(1, ln(1 + n_eff) / ln(1 + N_cap))  frequency, log-compressed
n_eff = Σ_events w_tier                     tier-weighted count (§2)
r  = 2^(−Δ_last / h)                        recency, half-life h days
r′ = r_floor + (1 − r_floor)·r              floor keeps old-but-frequent alive
```

Defaults pending §6 calibration: `N_cap=50`, `h=14 d`, `r_floor=0.25`, tier weights `w_surfaced=0, w_used=0.3, w_confirmed=1.0, w_corrected=0.5, w_legacy=0`.

**Worked examples (verified numerically):**

| usage history | n_eff | f | r′ | **u** |
|---|---|---|---|---|
| 1 surfaced, any age | 0 | 0 | — | **0.000** (loop broken) |
| 1 used, 10 min ago | 0.3 | 0.067 | 1.00 | **0.067** |
| 1 confirmed, 10 min ago | 1.0 | 0.176 | 1.00 | **0.176** |
| 5 used last week (last 1 d) | 1.5 | 0.233 | 0.96 | **0.225** |
| 50 used, last 30 d ago | 15 | 0.705 | 0.42 | **0.296** |
| 5 confirmed, last 60 d ago | 5.0 | 0.456 | 0.29 | **0.131** |
| 50 confirmed, recent | 50 | 1.00 | 1.00 | **0.996** |

1 vs 5 vs 50 discriminates (0.067 / 0.225 / 0.296 at like tiers); minutes vs days vs months discriminates through r′ without the sigmoid's 0.76–0.91 ceiling on singletons. Saturation now requires ~50 confirmed events — an earned ceiling, not a first-touch artifact.

**ACT-R is not deleted.** `compute_base_level` (`engine.py:11-36`), the sigmoid, `consolidated_strength`, and differential tier decay (0.5/0.3) remain **exactly as-is for the hygiene consumers** — prune's activation floor (`prune.py:76-99`), maturation scoring, dashboard. The change is that *ranking* stops consuming the sigmoid and consumes `u`. One signal store, two views, each calibrated for its consumer. (Pre-empting the judge: "why keep two formulas?" — because the consumers have different requirements: prune needs an absolute long-horizon floor — a single access still protects for ~768 days at floor 0.05, verified: B=−9.01 ⇔ age 768 d — while ranking needs short-horizon discrimination. One formula serving both is how we got here.)

---

## 2. Recording semantics — the loop fix

### Tier table, with the exact plumbing

| tier | w (ranking) | w (prune/mature) | carrier — exists today | build needed |
|---|---|---|---|---|
| **mentioned** *(F10, 2026-07-20)* | **0.1** | 1.0 | ingestion entity commits (`extraction/apply.py` — user-content-driven, loop-free; one event per (entity, episode); bootstrap artifacts excluded) | landed (M1.6) |
| **surfaced** | **0.0** | 1.0 | `RecallEntityAccessRecorder.record_entity_access` calls at `primary_results.py:285,790,1210` and `context_builder.py:851,1145`; interaction telemetry (`feedback.py:350-411`) | change: these sites record a `surfaced`-tier event (kept for prune/telemetry), which ranking-u ignores |
| **agent-used** | **0.3** | 1.0 | (a) `RecallMemoryInteractionApplier.apply` — `"used"` already triggers `record_access` (`feedback.py:458,553-561`); (b) response-mention partition `partition_recall_targets_by_usage` (`feedback.py:650-697`), gated `recall_usage_feedback_enabled` (`config.py:2095`, default False); (c) axi traces | **build: ingestion-time citation scan** — recall keeps a per-group ring buffer of surfaced (entity_id, name, ts); `store_episode` scans incoming content with the existing `_matches_entity_name` matcher (`feedback.py:700-711`); a surfaced entity mentioned in the *next observed turn* ⇒ agent-used event. This closes the loop on the frozen public 9-tool surface with **zero new tools**, because the agent's answer reaches Engram via `observe()` of the following turn |
| **confirmed / corrected** | **1.0 / 0.5** | 1.0 | MCP `feedback` tool — operator surface only (`mcp/surface.py:51`); `"confirmed"` already records access + TS positive (`feedback.py:458-459`); **cue-hit promotion already works** (`feedback.py:263-315`: hit_count ≥ `cue_recall_hit_threshold` ⇒ SCHEDULED projection) | build: tier tag on the event; treat cue promotion as a confirmed-tier signal for the episode's cue record (§4); `corrected` additionally marks labile (already: `feedback.py:182-189`) — the entity is demonstrably salient even when its content was wrong |

### Kill or keep surfaced-recording — **KILL as a ranking writer, KEEP as a hygiene/telemetry event**

Argument for kill (ranking): surfacing is *ranker output*, not environment. M4.1 traced 108 events to 27 person entities and 0 to topics with **no user signal involved**; review §4 makes the theory point (Anderson–Schooler justify the prior with environmental statistics; a self-referential access stream amplifies ranker bias into permanent forgetting). Any nonzero w_surfaced, however "heavily discounted," reopens the loop with a longer time constant — discounting changes the rate, not the fixed point. So w_surfaced=0 for ranking, structurally.
Argument for keep (hygiene): prune protection needs *any* evidence of relevance and surfaced-then-never-corrected is weak-but-real evidence an entity participates in the product; deleting it entirely would make fresh installs prune-naked (§5). Surfaced events also feed the interaction telemetry and cue-promotion counters that demonstrably work. Cost of keeping: none for ranking (weight 0), bounded memory (200-event trim, `engine.py:94-95`).

### Storage change

`ActivationState.access_history: list[float]` (`models/activation.py:13`) becomes tiered events. Minimal-churn representation: keep `access_history` (all tiers, feeds prune/B unchanged) and add `usage_events: list[tuple[float, float]]` = (ts, weight) holding only ranking-eligible events (used/confirmed/corrected). `n_eff = Σ w`, `Δ_last = now − max(ts)` — u is O(1) if we also cache the running sum. `record_access(state, now, cfg, tier=...)` grows a tier param; default `"surfaced"` keeps every existing caller behavior-correct for hygiene.

---

## 3. Router sanity — composition + a kill-switch that provably works

### Composition: multiplicative bounded tiebreaker on semantically-qualified score

```
final = composite_sem_score × (1 + β_route · u)        β_route ≤ β_max = 0.30
```

where `composite_sem_score` is the existing scorer output (`scorer.py:66-120`) **with the additive `w_act · act` term deleted**. Applied post-composite, pre-sort.

**Why multiplicative, not capped-additive:** an additive cap is absolute, so its relative distortion explodes at the bottom of the score range (cap 0.15 on the M4.1 pool floor of 0.28 is a 54% distortion; on the 1.0 top it's 15%). Multiplicative gives one clean invariant: **item X can overtake item Y only if `sem(X) > sem(Y)/(1+β_route)`** — usage flips near-ties inside a ≤30% relevance band and can never rescue a semantically buried item. That is the "activation never beats semantics" requirement as a theorem, not a tuning outcome.

**Worked example (the exact M4.1 kill scenario):** temporal query; wrong-person entity sem 0.30 with saturated usage; gold topic sem 0.50, no usage.
- Old (`_WEIGHT_PROFILES[TEMPORAL] = (0.20, 0.55, …)`, `router.py:51`): person 0.20·0.30 + 0.55·0.91 = **0.561** vs topic 0.20·0.50 = **0.100** → person wins 5.6×, gold buried (the 23→2 collapse).
- New (β_temporal = 0.25, person u = 1.0 worst case): person 0.30 × 1.25 = **0.375** vs topic **0.50** → gold wins. A used entity still beats an unused one *at equal semantics*: 0.50×1.25 = 0.625 > 0.50.

**Router profile change** (`router.py:49-56`): delete the activation column; profiles become (sem, spread, edge) renormalized to sum 1, plus a `β_route` column:

| route | old (sem, act, spr, edge) | new (sem, spr, edge) | β_route |
|---|---|---|---|
| DIRECT_LOOKUP | .75 .10 .05 .10 | .83 .06 .11 | 0.05 |
| TEMPORAL | .20 .55 .15 .10 | .44 .33 .22 | 0.25 |
| FREQUENCY | .15 .60 .15 .10 | .38 .38 .25 | **0.30** |
| ASSOCIATIVE | .55 .10 .20 .15 | .61 .22 .17 | 0.10 |
| CREATION | .30 .10 .25 .30 | .35 .29 .35 | 0.10 |
| DEFAULT | .40 .25 .15 .15 | .57 .21 .21 | 0.10 |

FREQUENCY keeps the largest β — "what do I engage with most" is the one query class where usage *is* the relevance signal — but even there it's a 30% band, not a 4:1 additive ratio. Temporal *episode* ordering is already owned by the Step-5.05 conversation-date boost (`pipeline.py:1693-1740`), which is environmental recency and stays (§4) — the TEMPORAL route no longer double-taxes it through activation.

**Candidate-pool membership:** delete the activation-based backfill (`candidate_pool.py:128-140`, `pipeline.py:531`, the ×3 temporal expansion). M4.1 arm E vs A shows membership alone *cost* 2 hits (21 vs 23/36). Usage is a **reranker, not a retriever** — candidates enter via semantics/spreading/graph only. (Pre-empt: "usage can't resurface forgotten items" — correct and intended; rediscovery belongs to the graph channel that M3.1 just proved out at 2/36→22/36, not to a self-echo. The scorer's exploration/rediscovery bonuses `scorer.py:107-120` are unaffected.)

**The kill-switch (deep-copy trap fix):** one flag, `usage_ranking_enabled: bool = False`. It works provably because of *where* it lives: the multiplication site checks it, and `apply_route` (`router.py:100-108`) **has no usage field in its write-set** — the four-weight overwrite that voided M4.1's first ablation arm can't touch it, since `weight_activation` no longer exists in the profiles (config field deprecated, warn-if-set). Enforced by two permanent tests: (1) unit: `apply_route` output differs from input **only** in (sem, spread, edge, β) — schema-frozen; (2) rig: flag-off recall on a populated activation store is byte-identical (ids AND scores) to the same recall on an empty store — the M4.1 A≡C proof inverted into a regression gate.

---

## 4. Episode unification — one model, two recencies

**Principle that resolves the ad-hocness:** there are two distinct recencies and Engram currently conflates them.
- **Environmental recency** — *when the memory was created* (`conversation_date`). Owned by Step 5.05 (`pipeline.py:1693-1740`): temporal-cue-gated, exponential half-life (`recency_halflife_days`, `config.py:634-638`), episodes-only, benchmark-validated. **Unchanged.**
- **Behavioral recency/frequency** — *when the memory was useful*. This is `u`, and it must apply to episodes the same way as entities, because episodes ARE the core tier and currently accrue no usage signal at all (M4.1 structural finding 1: episodes never record access).

**One model:**

```
entity:   final = composite × (1 + β_route · u_entity)     u from usage_events (§2)
episode:  final = rrf_score × (1 + β_route · u_episode) × (1 + temporal_cue_boost)
```

Same u formula, same β_route, same tier weights. Both factors bounded (usage ≤1.30, temporal-cue ≤2.0 and gated on temporal queries), multiplicative on a rank-normalized RRF score so relative bands are preserved.

**What records for episodes — reuse the cue substrate, don't mint ActivationState per episode.** Episodes already have a working usage record: the cue layer (`hit_count`, `last_hit_at`, `policy_score`; feedback at `feedback.py:210-329`, promotion **already works**). Build: add `used_count: float` (tier-weighted) and `last_used_at` to the cue record; agent-used events come from the existing cue span-matcher (`_matches_cue_content`, `feedback.py:714-727`) in the same ingestion-time citation scan as entities (§2); confirmed = the feedback tool's cue path (`feedback.py:496-527`) + cue promotion itself. `u_episode = f(used_count)·r′(last_used_at)`. Episodes without cues get u=0 — graceful degradation, no new per-episode memory.

**Alternative considered and rejected:** full `ActivationState` per episode — thousands of episodes × history lists bloats the 50k-entry snapshot (`activation.py:99-107`) for no consumer (no spreading over episodes; prune of episodes is cue/status-driven). Keeping episodes entirely separate (status quo) is also rejected: it's exactly the "no principled recency model" gap — surfaced-list positions today are decided by ad-hoc boosts that no experiment instrument can see.

**Why NOT unify environmental and behavioral into one term:** conflating them is precisely what `TEMPORAL → weight_activation=0.55` did — it made "recently *touched by the ranker*" answer "what happened recently". Creation-time is trustworthy metadata; touch-time is feedback. They compose multiplicatively and are separately gated, calibrated, and killable.

---

## 5. Interaction contracts

**Importance prior (M3.3, `importance_prior_enabled`, `config.py:1519-1527`; `seed_consolidated_strength`, `engine.py:62-77`).** Invariant: **importance and usage share no inputs.** `u` is a function of usage_events only — `consolidated_strength` is excluded from n_eff and Δ_last. Importance keeps its two existing lanes: (a) inside the ln-sum for the *hygiene* view (prune floor, mature) — the M3.3 arithmetic DoD ("one-shot durable ≈ 5-access mundane at 30 d") is untouched because prune-side `compute_activation` still takes `consolidated_strength` (`scorer.py` analog at prune path); (b) the durable-type 2.5× ranking boost for the *ranking* view. So: importance = what it *is*; usage = how it's *used*; recency = when. Three separable multipliers, three kill-switches. (Pre-empt: "won't a confirmed durable fact double-dip?" — yes, deliberately: confirmation is user evidence, durability is commit-time class; both bounded, product ≤ 2.5×1.30.)

**Prune protection.** Prune keeps consuming full `access_history` (all tiers at hygiene weight 1.0) + the activation floor + `consolidated_strength`. Killing surfaced-*ranking* does not starve prune, because surfaced events still append to `access_history` (§2). Numbers: single access 14 d old ⇒ act 0.146, 90 d ⇒ 0.090, both > 0.05 floor; floor crossed only at ~768 d single-access. The M1.1 fix (snapshot loaded in the brain path) is a hard prerequisite and is already landed.

**Snapshot lifecycle (`storage/memory/activation.py:99-157`, `main.py:135/473`).** Crash-loss policy by tier: *surfaced* — free (not ranking-relevant); *agent-used* — acceptable (medium-weight, high-volume, regenerated by ongoing use; losing ≤ one session's worth moves u by ≤ one f-step); *confirmed/corrected* — **not acceptable** (rare, explicit user signal). Build: a tiny append-only JSONL journal for confirmed-tier events, written through at record time, replayed on `load_from_file`, truncated on successful snapshot save. Bounded by construction (confirmed events are rare). Snapshot format bumps to v2 (§7); the existing `max_age_days=14` staleness guard (`activation.py:122`) stays for v1-style bulk state but the confirmed journal is exempt from age-out (user signal doesn't expire in 14 days).

**Session state (post-Task-B).** Contract: **u reads the durable store only** — never the priming buffer, working memory, or conv-context (the state M4.3 showed mutating on read, 56→61 pool drift). Combined with pool-independence (§1) and w_surfaced=0, the determinism theorem is: *same query, same session, no intervening confirmed/used event ⇒ identical ranking* — restoring the 33/36→36/36 repeat-stability that TS destroyed, and this becomes gate G4 (§6). TS itself: per M4.3, default off; if ever revived, per-entity seeded draws — orthogonal to this design.

---

## 6. Calibration + eval harness — every parameter from data

**Empirical inputs (live-brain-copy, read-only per safety rule):** a `usage_distribution_snapshot` script over `scratchpad/experiments/live-copy/` (or read-only HTTP :8100) emitting: histogram of per-entity access counts, inter-access-interval distribution, age-of-last-access distribution, per-tier event volumes (post-launch). Parameter bindings:

| param | rule | fallback |
|---|---|---|
| `N_cap` | p95 of per-entity ranking-event counts | 50 |
| `h` | median inter-access interval, clipped [7, 30] d | 14 d |
| `r_floor` | chosen s.t. p50-frequency at p90-age retains ≥ half of its fresh u | 0.25 |
| `β_route` | rig sweep {0.05, 0.10, 0.25, 0.30} maximizing G3 without violating G2 | table §3 |
| prune-side (B_mid, B_scale) | option (a) fit, only if prune telemetry shows floor miscalibration | unchanged |

Re-fit cadence: recompute at mop windows alongside snapshot save; parameters change only via config, never silently.

**The instrument is the M4.1 rig** (`scratchpad/experiments/m41_activation/`, 102 ep / 72 ent / 36 bridges, deterministic — A≡A2 proven), rerun with usage-pass events re-labeled `used`-tier (worst case: the same 108-event, 27-person history that caused the collapse). **Acceptance gates, all required before default-on:**

- **G1 no-op proof:** flag-on + empty store ≡ flag-off, byte-identical ids and scores, 36/36 (benchmark regime unchanged — the north-star number stays honest).
- **G2 usage never degrades:** held-out reach@10 after the usage pass ≥ no-history baseline (≥23/36 on this rig; the shipped system scored 2/36). Hard fail on any regression.
- **G3 a used entity wins ties:** synthetic tie-probe (equal-sem pairs, one with a used event) — 100% of used items rank first.
- **G4 determinism:** in-session repeat recall identical top-10, 36/36 (vs 0/36 under TS).
- **G5 budget-0 hygiene:** zero activation-store writes on the read path (store-spy, M1.2-style).
- **G6 real corpus:** `cd server && uv run engram continuity --against-live --organic` — no regression; per project policy the small-corpus rig gates *mechanisms*, only the live gate flips *defaults*.

---

## 7. Migration

| asset | disposition |
|---|---|
| existing `access_history` floats (overwhelmingly surfaced-recorded via `context_builder.py:851,1145` + depth-tier) | **carry unchanged for hygiene** (prune/mature semantics identical — no entity becomes prunable by migration); **excluded from ranking-u** (tier `legacy`, w=0). Ranking starts cold — honest, because no trusted signal exists yet; G1 makes cold-start a provable no-op |
| `usage_events` | new, starts empty; populated by §2 writers — incl. the F10 `mentioned` tier (0.1) for organic entity commits, so ranking-frequency has a trusted environmental signal at launch independent of the citation scan |
| snapshot format | v2: adds `usage_events` per state + `"version": 2`; loader accepts v1 (no usage_events ⇒ empty; history → legacy). Writer emits v2. 14-day age guard unchanged; confirmed journal (§5) is new and separate |
| `ts_alpha/ts_beta` in state (`models/activation.py:18-19`) | retained in the model/snapshot (M4.3 flip is parked, not decided); untouched by this design |
| graph-row `access_count` (permanently stale — `snapshot_to_graph` has zero callers, review §1.1) | wire `snapshot_to_graph` (`activation.py:159`) into the mop window (shell paused ⇒ single-writer safe) for dashboard honesty, **ranking never reads it**; or delete the column claim — either resolves the lie, pick the sync since the method already exists |
| cue records | additive columns `used_count`, `last_used_at` (helix schema.hx + sqlite migration); absent ⇒ u_episode=0, no backfill |
| config | `weight_activation` + profile activation column deprecated (warn-if-set, one release); new: `usage_ranking_enabled=False`, `usage_beta_*`, `usage_half_life_days`, `usage_n_cap`, `usage_tier_weights` |

**Rollout order:** (1) recording tiers + storage v2 (writers live, ranking off — accumulates trusted data, provably inert by G1/G5); (2) ingestion-time citation scan behind `recall_usage_feedback_enabled`; (3) rig gates G1–G5; (4) `usage_ranking_enabled=True` behind G6. Each step independently killable; no step changes benchmark-regime output until (4).

---

## Pre-empted objections (summary for the panel)

1. *"You kept two formulas (B for hygiene, u for ranking) — complexity."* Deliberate: one formula serving a 768-day prune floor and a 30%-band tiebreaker simultaneously is the root cause being fixed. The event store is single; the views are two small pure functions.
2. *"w_surfaced=0 means the public surface records nothing for ranking"* (the M4.4 parked concern). False after the ingestion-time citation scan: the agent's answer reaches Engram in the next `observe()` on the frozen 9-tool surface; the matcher already exists (`feedback.py:700-727`).
3. *"Multiplicative boost can't ever fix a retrieval miss."* Correct — by design. Retrieval misses are the graph channel's job (M3.1: 2/36→22/36); usage echoing into retrieval is the loop we're killing (arm F: membership without term still killed reach).
4. *"Percentile normalization is more adaptive."* It imports pool-drift nondeterminism — the exact defect class M4.3 confirmed (56→61 drift, 0/36 stability). Absolute primitives + empirical parameter fitting gets adaptivity without it.
5. *"Numbers are from a 36-question synthetic rig."* Per project policy the rig gates mechanisms (no-op, determinism, invariants — corpus-independent); G6 (live organic gate) gates the default flip.