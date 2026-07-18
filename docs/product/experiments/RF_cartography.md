# Recency/Frequency Cartography — the complete producer→store→consumer map

Snapshot date: 2026-07-17/18. All paths relative to `server/engram/`.
CAVEAT: `retrieval/pipeline.py`, `retrieval/candidate_pool.py`,
`retrieval/context_builder.py` are under concurrent edit by a sibling workflow —
their line numbers are as-of this snapshot; step labels (`Step 5.05` etc.) are
the stable anchors. All other files verified stable this session.

---

## 1. The stores — where R/F state actually lives (two disagreeing systems)

| Store | Fields | Backing | Lifetime | file:line |
|---|---|---|---|---|
| `ActivationState` (activation store) | `access_history[≤200]`, `access_count`, `last_accessed`, `consolidated_strength`, `last_compacted`, `ts_alpha`, `ts_beta` | in-RAM dict (`MemoryActivationStore`) in lite AND native-helix modes; Redis only in Docker full mode | process lifetime + JSON snapshot (§9) | models/activation.py:8-19; storage/memory/activation.py:31-97; storage/factory.py:113,168,216 (memory), 234-288 (redis) |
| Graph entity rows | `access_count`, `last_accessed` columns | SQLite / Helix native | durable | models/entity.py:26-27; sqlite/graph.py:305-320 (written at create, then never); helix/graph.py:583,867,977 |
| Epoch convention | `access_history` stores epoch-seconds; ACT-R ages are `now - t_j` floored at `min_age_seconds=1.0` | — | — | activation/engine.py:30-34; config.py:158 |

**The two stores disagree by construction.** The ONLY sync path is
`MemoryActivationStore.snapshot_to_graph` (storage/memory/activation.py:159-175)
— **zero production callers** (only tests/storage/test_activation_store.py).
Graph-row `access_count` is 0 at create and changes only when merge sums the
two rows (sqlite/graph.py:1645-1658; helix/graph.py:2995-3010).

Dependency notes:
- Change `max_history_size` (engine.py:94-95, config.py:159) → compact's
  absorbed-strength math and snapshot size shift; 200-cap already truncates
  heavy entities' frequency signal.
- Change graph-row semantics → prune's candidate SELECT changes (§3 prune row);
  scoring does NOT change (scoring never reads graph rows for access).
- `ActivationState.ts_*` ride the same object → any "reset activation" op also
  resets the bandit posterior, and vice versa.

## 2. Access-event producers (every writer of `record_access`)

`record_access` appends timestamp + increments count + trims to 200
(activation/engine.py:80-95; store entry points memory/activation.py:52-67,
redis/activation.py:156-161).

| # | Producer | Trigger | Gate(s) | file:line | If changed → |
|---|---|---|---|---|---|
| P1 | Recall entity materializer | entity SURFACED in final recall results | `record_access=True` AND entity wins a final slot — **`passage_first_entity_budget=0` (default) ⇒ never fires** (M4.1 arm B0: 0 events/36 recalls) | retrieval/primary_results.py:285-299 → feedback.py:163-180 (`RecallEntityAccessRecorder`) | zeroing this breaks nothing shipped; it is the rich-get-richer loop's engine when budget≥1 |
| P2 | Explicit recall surface (MCP/API `recall`) | every explicit recall | manager default `record_access=True` (graph_manager.py:2028), `interaction_type="surfaced"` (recall_surface.py:800-801) — feeds P1's gate | retrieval/recall_surface.py:796-802 | this is where "explicit recall strengthens memory" lives; still inert at budget 0 |
| P3 | Auto-recall | agent-turn auto recall | `record_access=False` whenever `recall_usage_feedback_enabled` (ON in every recall_profile ≠ off, incl. quiet/wave2) — so default installs never record on auto-surface | retrieval/auto_recall.py:947-952,1001; config.py:2820 | flipping usage_feedback off silently re-enables surface-time recording |
| P4 | Confirmed-use feedback | `apply_memory_interaction(type∈{used,confirmed})` | only caller in prod = dashboard knowledge-chat response scan (`partition_recall_targets_by_usage` regex-match of entity name in the reply) | retrieval/feedback.py:458,553-561; chat_feedback.py:34-56; graph_manager.py:1270-1291 | **this IS the M4.4 "confirmed-use" path, already built** — MCP/agents never call it (no public feedback tool) |
| P5 | Ingest/extraction | every entity commit (new or matched) during episode projection | always (no flag) | extraction/apply.py:847 | ingestion = 1 access per mention; disabling starves prune/mature counts for organically-captured entities |
| P6 | Reconsolidation | labile-window summary update on re-mention | `reconsolidation_enabled` (standard only) + 5-min in-RAM window | extraction/apply.py:793 | cross-process dead (window in RAM, brain runs 2h later) |
| P7 | Replay phase (cold brain) | deferred extraction links/creates entities | replay enabled + selected | consolidation/phases/replay.py:359,389 | brain-side writes land in the brain's store copy and are **discarded at brain exit** (brain never saves the snapshot, §9) |
| P8 | Project bootstrap | bootstrapped file/project entities | bootstrap enabled | ingestion/project_bootstrap.py:169,205,423 | seeds bootstrap artifacts with day-one recency advantage |
| P9 | get_context delivery | every entity actually DELIVERED in the context payload (skips recall-recorded + truncation-cut) | always when get_context runs | retrieval/context_builder.py:1203-1216 (dedup guard 1203, truncation guard 1207) | **the highest-traffic live producer on default installs** — same self-reinforcement family as P1 (review §"cross-checker uncovered") |
| P10 | Project-entity create in get_context | new Project entity minted | first get_context per project | context_builder.py:790 | — |
| P11 | Intention create | `intend()` | prospective enabled | retrieval/prospective.py:448 | — |
| P12 | Benchmark/corpus harnesses | synthetic | benchmark only | benchmark/corpus.py:1443, echo_chamber.py:106 | — |

**Episodes NEVER record access — no writer exists for episode-side ACT-R state
anywhere** (models/episode.py has no access fields). The episode channel's only
"recency" is §6.

Explicit non-writers (by design): spreading activation (no record-on-spread —
holds everywhere, review §2), benchmarks (`record_access=False`,
benchmark/longmemeval/adapter.py:435, locomo/runner.py:181), near-miss
materialization (near_miss.py — feedback only, no access), `epistemic_evidence.py:80`.

## 3. Activation readers — every `compute_activation` call site

Formula: `B = ln(consolidated_strength + Σ age^-d)`, sigmoid((B−B_mid)/B_scale),
B_mid=-4.0, B_scale=1.7, d=0.5 (engine.py:11-47; config.py:157-161). Saturation
fact: 1 access ⇒ 0.76-0.91, 16 ⇒ 0.98 (M4.1).

### Ranking-path readers (change these ⇒ ranks change)

| Reader | Uses | decay override? | file:line | If changed → |
|---|---|---|---|---|
| Entity scorer term | `w_act × act` in composite; skips if no history (base_act=0) | YES: mat_tier→`_tier_to_decay` (0.5/0.4/0.3) gated `memory_maturation_enabled` | retrieval/scorer.py:84-93,39-47 (and TS twin 241-247) | THE term M4.1 proved collapses reach 23→2; zeroing it = arm E (recovers 21) |
| Seed energy | `energy = sem × max(act,0.15)`; temporal_mode: `max(act,0.15)` alone | no | activation/spreading.py:42-71 | activation shapes WHERE spreading starts; floor 0.15 keeps cold entities seedable |
| Activation candidate pool (multi-pool P2) | `get_top_activated` membership + act as pool score | no — **constructs fresh `ActivationConfig()` at candidate_pool.py:140, ignoring live cfg/env overrides** (silent-inert hazard for any B_mid/B_scale retune) | retrieval/candidate_pool.py:120-148; ×3 size multiplier for TEMPORAL/FREQUENCY at 29-36 | M4.1 arm F: membership alone is harmless; but the fresh-cfg bug means "tune the sigmoid" won't reach this pool |
| Temporal bypass (single-pool Step 1.6) | injects `get_top_activated` ids at sem=0.0 | no | pipeline.py:536-550 | only when `multi_pool_enabled=False` (non-default) |
| Goal priming | goals need `act ≥ goal_priming_activation_floor` to fire; seed energy = `goal_priming_boost × act` | no | retrieval/goals.py:98-107,151 | goals only ever activate via P1-P11 writers; on default installs (no recording) goal priming ~never fires |
| Exploration/rediscovery (frequency side) | novelty `1/(1+ln(1+access_count))`; rediscovery ramps with days-since-last-access | n/a (reads count/history directly, not the sigmoid) | scorer.py:107-124 | counter-weights are 0.05/0.02 — ~10× weaker than the loop (review §4) |
| Prospective firing | intention warmth = act / threshold | no | retrieval/prospective.py:549-563,930-985 | — |
| Surprise detection | low-act + high-sem = surprising | no | retrieval/surprise.py:81 | wave3 only |

### Consolidation/hygiene readers

| Reader | Uses | file:line | If changed → |
|---|---|---|---|
| Prune (dead-entity pass) | graph-row `access_count ≤ 2` SELECT (sqlite:1529-1558, helix:2755-2813) **on permanently-stale rows**, then live-store double-check: count>2 skip, `act > 0.05` floor skip, emotional adjust — all inside `if state:` (skipped when store cold) | consolidation/phases/prune.py:40-45,76-99 | M1.1 made brain load the snapshot (brain_cli.py:412-419) so `state` is now warm-ish; protections still depend on the snapshot being fresh (§9 crash-loss) |
| Prune (low-value pass) | same live-store safety net | prune.py:204-215 | — |
| Compact | `get_top_activated(limit=10000)` scan; drops old timestamps → absorbs into `consolidated_strength` (§7) | compact.py:44-117 | warm-tier only; not in the mop |
| Dream seed select | bell-curve band [floor,ceiling] on act | dream.py:210-232 | standard-only |
| Dream LTD sweep | `act < floor` ⇒ decay unboosted edges | dream.py:376-390 | standard-only; edge-weight recency proxy |
| Microglia edge protect | endpoint `act > 0.3` blocks edge demotion | microglia.py:381-390 | standard-only |

### Display-only readers (no ranking effect)

atlas/builder.py:381; retrieval/graph_state.py:326-336,367-385 (activation
timeline), 638-659, 764-811, 830-841, 985-991; retrieval/lookup.py:394;
context_builder.py:575-580 (payload field), 1052-1076 (get_context "Recent
Activity" layer-3 = `get_top_activated` ordering — display AND then re-recorded
via P9 → get_context has its own closed rich-get-richer loop);
feedback.py:107-124 (event payloads); lifecycle_summary, dashboard API.

## 4. The router — every profile row and what fires it

`retrieval/router.py`. Classification order (64-92): TEMPORAL regex → FREQUENCY
regex → CREATION regex → ASSOCIATIVE regex → top-hit>0.8 DIRECT_LOOKUP → DEFAULT.
`apply_route` (100-108) **deep-copies cfg and force-overwrites the 4 core
weights** → any config-level `weight_*` ablation is illusory for routed queries
(M4.1 structural finding 3). Applied at pipeline Step 1.5 (multi-pool: ~483-485;
single-pool: ~521-524); routing runs even when `enable_routing=False` if the
query classifies TEMPORAL. Second consumer: benchmark/methods.py:424-456.

| Profile (router.py:49-56) | sem | act | spread | edge | act:sem | Trigger |
|---|---|---|---|---|---|---|
| DIRECT_LOOKUP | 0.75 | 0.10 | 0.05 | 0.10 | 0.13 | top search score > 0.8 |
| TEMPORAL | 0.20 | **0.55** | 0.15 | 0.10 | **2.75** | `recent(ly)|lately|last|today|yesterday|this week/month|just now|earlier|few days|what's new` (router.py:15-19) |
| FREQUENCY | 0.15 | **0.60** | 0.15 | 0.10 | **4.0** | `most|frequently|focus|top|primary|important|engaged|referenced|...` (27-31) |
| ASSOCIATIVE | 0.55 | 0.10 | 0.20 | 0.15 | 0.18 | connect/link/between/related (21-25) |
| CREATION | 0.30 | 0.10 | 0.25 | 0.30 | 0.33 | wrote/created/built/... (33-37) |
| DEFAULT | 0.40 | 0.25 | 0.15 | 0.15 | 0.63 | fallback (= config defaults 167-170) |

Also query-type-coupled: pool multipliers (candidate_pool.py:29-36 — TEMPORAL/
FREQUENCY get 3× activation-pool, DIRECT 2× search, ASSOCIATIVE/CREATION 2×
graph) and temporal_mode seed rule (spreading.py:42,51-52). If profiles are
zeroed for activation, the ×3 TEMPORAL activation-pool membership stays (harmless
per arm E/F, but wasted budget) and temporal_mode seeding still keys off
activation — three separate switches, not one.

## 5. Session-state family — every per-session mutable that touches scoring

All owned per-`GraphManager` instance (graph_manager.py:214-276,462-476). NONE
persist. All mutate ON READ (recall) even with `record_access=False` — the
proven pool-drift source (M4.3: 56→61 candidates, repeat-stability 0/36 with TS).

| State | Structure | Written by | Read by (scoring effect) | Lifetime / reset | file:line |
|---|---|---|---|---|---|
| WorkingMemoryBuffer | LRU 20, TTL 300s linear-decay + deque(5) recent queries | EVERY materialized recall result (entities AND episodes, regardless of record_access) primary_results.py:187-194,301-308; every query post_process.py:111-115 | candidate injection Step 1.7 (`0.1×recency`); extra seeds Step 3.5 (`working_memory_seed_energy=0.3 × recency`); WM pool 4 + 1-hop 0.5× damping (candidate_pool.py:185-226); ACT-R-strategy sole seed source (spreading.py:74-96) | GC with manager; `clear()` never called in prod | working_memory.py:20-98; cfg 523-526 |
| Priming buffer | `dict[eid → (boost, expiry)]` TTL 30s | post-recall: top-3 entity results' 1-hop neighbors get `0.15×edge_weight` (priming.py:30-62, via post_process.py:117-127) | additive `prime_boost` in scorer (scorer.py:132-134/288-290, pipeline Step 4.7 expiry filter) | entries expire 30s; dict itself never cleared (expired entries leak until overwritten) | priming.py; cfg 2336-2359 (wave3+) |
| ConversationContext.fingerprint | EMA(α=0.85) of live-turn embeddings, L2-normed | `add_turn` via MCP turn ingestion (mcp/server.py:584 → context.py:350-365); recall queries appended as turns with `update_fingerprint=False` (post_process.py:162-168 → context.py:460-475) | `conv_context_rerank_weight (0.05) × cos(fp, entity)` boost Step 4.6 (pipeline ~1378-1396; scorer ctx_boost 127-130) | manager lifetime; `clear()` (context.py:267-278) never called in prod | context.py:38-129; cfg 2223-2285 (wave2+ = quiet default) |
| ConversationContext.session_entities | `mention_weight` cumulative per entity | extraction commit path only (apply.py:866) — recalled-but-not-reingested entities never enter | Step 3.6 seeds: `0.2 × min(1, weight/5)` energy (pipeline ~1199-1209) | manager lifetime, unbounded growth | context.py:131-166 |
| ConversationContext cognitive state | arousal EMA + mode | pipeline Step 4.95 `update_cognitive_state` (~1567) | `state_biases` → `s_boost` in scorer (state-dependent retrieval, conservative/standard only) | manager lifetime | context.py:240-265; retrieval/state.py |
| GoalPrimingCache | `{group → (ts, goals)}` TTL 60s | `identify_active_goals` fills (goals.py:136-138) | Step 2.5 → Step 3.9 goal seeds | TTL 60s; `invalidate()` callers only on goal mutation | goals.py:24-47 |
| LabileWindowTracker | recalled entities, TTL 300s, max 50 | entity access recorder marks on every recorded access (feedback.py:182-189) | reconsolidation on re-ingest (apply.py:781-800) — summary rewrite + P6 access | in-RAM only; dead cross-process | retrieval/reconsolidation.py:22-75; cfg 2481-2485 |
| Near-miss window | positions [limit, limit+5) of the scored list | split at service level (request_policy.py:41-50, scorer extract_near_misses 363-371) | near-miss cue FEEDBACK writes cue hit/policy rows (near_miss.py:114-127) — graph writes on the read path | per-call | cfg conv_near_miss_* 2270-2278 |
| Packet/surprise caches | TTL caches | recall/get_context | packet short-circuit (skips recall entirely → skips all of the above) | TTL | packet_cache.py:263, surprise.py:146 |

Reset points: process restart only (plus per-entry TTLs). No API/tool clears
session state. Fresh-manager = the M4.3 "state-clean" condition (reach@10 31-32
vs 15-17 session-shaped).

## 6. Episode-channel recency — the de-facto episode recency model

Episodes have NO activation state; their recency machinery is entirely below.

| Mechanism | What it does | Gate | file:line | If changed → |
|---|---|---|---|---|
| Temporal cue detector | keyword dict: earliest/latest/count/state words | `temporal_retrieval_enabled=True` (default ON) | pipeline.py:52-93 | independent of router regex — a query can be router-TEMPORAL but cue-negative and vice versa (two vocabularies, silent divergence) |
| Step 5.05 recency multiplier | wants_latest/state: `score ×= 1+e^(-age/30d)` (up to 2×); wants_earliest: `×= 1+(1-e^(-age/30d))`; applied to scored episodes AND episode_candidates; O(N) sequential `get_episode_by_id` | same | pipeline.py:1681-1752 (`recency_halflife_days=30` cfg 634) | this is the recommendation's "episode recency already handled" mechanism; basis = `conversation_date` (falls back to nothing — undated episodes get NO boost) |
| Current-value entity boost | `×1.2` for role-bearing entities on latest/state queries | `current_value_entity_boost` cfg 640 | pipeline ~1738-1761 | pairs with supersession/current-value surfacing |
| Step 5.06 date guarantee | top-3 date-sorted episodes force-appended for earliest/latest queries | wants_earliest/latest | pipeline ~1754-1801 | the only HARD recency guarantee in recall; bypasses scores entirely |
| Temporal contiguity | adjacent same-session episodes boosted `0.5×parent` | `temporal_contiguity_enabled=False` default | post_process.py (cfg 601-627) | OFF — dead by default |
| created_at DESC ordering | list_episodes / find_episodes / recent-episode surfaces | always | sqlite/graph.py:998-1066,1262; falkordb:883-931 | API/dashboard/conversation surfaces' implicit recency; not part of recall scoring |
| Timeline packets | recall+facts rows sorted by conversation_date→valid_from→created_at; first/last/span computed deterministically | packets path | retrieval/timeline.py:96-183 | the temporal-answer surface; depends on date fields, not scores |
| Cue layer "recency" | cue hit_count/last_hit_at/policy_score updated on every surfaced/near-miss cue (graph writes on read); ≥2 hits ⇒ promotion to projection | `cue_recall_enabled` (wave2+) | feedback.py:210-329 (promotion 263-315); cfg cue_recall_* 713-726 | frequency-of-recall signal for EPISODES — the only episode-side "access count" analog; auto-surface gated at `auto_recall_min_cue_score=0.15` because cue ceiling 0.40×0.65=0.26 < 0.3 (cfg 1793-1808, M2.1 fix) |
| Episode traversal | appended episodes score `parent_entity_score × entity_episode_weight (0.6)` — inherits the ENTITY channel's activation contamination when budget/flag route through it | `entity_episode_traversal_enabled`, source flag cfg 589 | retrieval/episode_traversal.py:73 | M4.1 mechanism: post-usage, all traversal parents are recently-accessed Persons |
| Decomposer | temporal sub-query templating (first/last/original...) | `query_decomposition_enabled` default ON | retrieval/decomposer.py:312-313 | third temporal vocabulary |

## 7. `consolidated_strength` — the ln-sum prior

Semantics: sits INSIDE `ln(cs + Σ age^-d)` (engine.py:29-34) ⇒ a permanent
activation floor that never decays on its own.

| Edge | Direction | file:line | Note |
|---|---|---|---|
| Compact absorption | dropped timestamps' `Σ age^-d` added; prior mass `×= compact_strength_decay` first (default **1.0** = no decay until eval flips it; suggested 0.9) | compact.py:103-117,128-141; cfg 1048 | monotone-accumulator defect half-fixed (knob shipped, default preserves old behavior) |
| M3.3 importance seeds | commit-time: identity 0.02, durable-type 0.01, client-proposal 0.01, cap 0.05 | extraction/apply.py:38-48,598-627,848-855; engine.py:62-77; `importance_prior_enabled=False` cfg 1519 (EVAL-GATED) | 0.05 cap ⇒ B=ln(0.05)=-3.0 ⇒ sigmoid ≈ 0.64 standalone — a seeded one-shot fact floors at act≈0.64 vs 0.0 unseeded: large, deliberate |
| Read: every activation compute | passed by scorer (85-90), TS scorer (241-246), prune double-check (prune.py:81-86,208-213), get_top_activated (memory store 89-94), batch_compute | — | NOTE: spreading seeds (spreading.py:48,64), goals (goals.py:102), microglia (386), dream (222,384), candidate_pool (143), and most display readers call `compute_activation(history, now, cfg)` WITHOUT cs → **an importance-seeded entity with zero accesses is invisible to seeding/goals/dream/prune-band logic even with the flag on** (silent-inert seam if M3.3 ever flips on) |
| Snapshot | persisted + restored | memory/activation.py:107-149 | survives restarts like history |

## 8. Thompson-sampling bandit family (adjacent frequency state)

| Edge | file:line | Note |
|---|---|---|
| Draw | Beta(ts_alpha, ts_beta) per candidate, `ts_weight=0.08 × sem × sample`; seeded blake2b(group,query) but consumed in LIST ORDER (positional — M4.3 defect) | scorer.py:261-268; pipeline Step 5 ~1626-1645 |
| Learn (recall) | Step 7: positive for returned entity ids, negative for the rest — ONLY if entity results actually surfaced (M1.2 guard) ⇒ dead at budget 0 | pipeline.py:2127-2157; activation/feedback.py:9-32 |
| Learn (explicit) | confirmed → positive, corrected → negative | retrieval/feedback.py:459-460,563-571 |
| Gate | `should_record_ranking_feedback`: passive interactions never, true-usage always, else `record_access` | request_policy.py:29-38 |
| Status | M4.3 verdict: `ts_enabled=False` recommended, flip parked; still default True (cfg 366) | — |

## 9. Snapshot persistence lifecycle (`~/.engram/activation-snapshot.json`)

Path helper memory/activation.py:20-28. Format: `{saved_at, states{...+group_id}}`,
≤50k entities, stale >14d ignored, live-state-wins on merge (122-157).

| Runtime | Load | Save | file:line |
|---|---|---|---|
| Shell (`engram serve`) | startup | clean shutdown only (`_shutdown`) | main.py:130-137, 463-474 |
| MCP stdio | startup (records file mtime) | ONLY IF: loaded ∧ no shell alive ∧ snapshot mtime unchanged since init — else defers/skips | mcp/server.py:320-336, 425-451 |
| Cold brain (`engram brain run`) | yes, read-only (`load_activation_snapshot=True`) | **never** | brain_cli.py:318-325, 412-419; storage/bootstrap.py:272-306 |
| One-shot CLI / consolidation module | via `open_local_stores(load_activation_snapshot=...)` — default False | never | bootstrap.py:272 |
| REST-less crash (any runtime) | — | **all accesses since load are lost** (save is shutdown-hook only) | — |

Consequences: brain-window P7 (replay) and any brain-side record_access are
discarded by design; two concurrent MCP sessions → last-clean-exit wins under
the mtime rule (second exits skip-save); the ownership protocol prevents
clobber but guarantees under-counting. `snapshot_to_graph` remains uncalled →
graph rows stay stale forever (§1).

## 10. Config-knob census (family only) — live/dead on the DEFAULT install

Default install = quiet profile + recall wave2 + `passage_first_entity_budget=0`
+ runtime_role=shell. "Live" = can change bytes in a default recall/brain cycle.

| Knob (config.py:line) | Default | Status on default install |
|---|---|---|
| `decay_exponent` 157 | 0.5 | LIVE (every compute_activation) — but see saturation |
| `min_age_seconds` 158 | 1.0 | LIVE (age floor — with epoch-seconds this makes 1 fresh access contribute up to 1.0 to the ln-sum) |
| `max_history_size` 159 | 200 | LIVE |
| `B_mid` 160 / `B_scale` 161 | −4.0 / 1.7 | LIVE, saturating; NOT respected by `_activation_pool` (fresh cfg, candidate_pool.py:140) |
| `weight_activation` 168 | 0.25 | **ILLUSORY for routed queries** — router overwrites (router.py:100-108); effective value = profile row |
| `weight_semantic/spreading/edge_proximity` 167-170 | .40/.15/.15 | same illusion |
| `weight_name_match` 171 | 0.15 | LIVE (not router-overwritten) |
| `seed_threshold` 182 | 0.3 | LIVE but rank-cutoff-not-relevance vs RRF scores (review §3) |
| `activation_ttl_days` 183 | 90 | DEAD except Redis mode (redis/activation.py:29) — memory store never expires |
| `exploration_weight` 184 | 0.05 | DEAD when `ts_enabled=True` (TS scorer replaces the exploration term; scorer.py:264-268) — live only in the non-TS scorer |
| `rediscovery_weight/halflife` 185-186 | 0.02/30d | LIVE in both scorers (271-280) but ≈0 forever on installs where nothing records access |
| `ts_enabled/ts_weight/ts_±increment` 366-369 | True/0.08/1/1 | draw LIVE (jitter), learning DEAD at budget 0 (M1.2 guard) — "doubly inert" M4.3 |
| `pool_activation_limit` 393 | 20 | LIVE (pool membership) — harmless without the score term (arm E/F) |
| `working_memory_*` 523-526 | on/20/300s/0.3 | LIVE (WM writes happen on every recall) |
| `episode_retrieval_weight` 530 | 0.8 | LIVE (episode channel scale) |
| `entity_episode_traversal_source` 589 | "results" | LIVE flag, default = starved path (M3.1) |
| `temporal_retrieval_enabled` 630 / `recency_halflife_days` 634 | True/30 | LIVE — the real episode recency model |
| `current_value_entity_boost` 640 | 1.2 | LIVE on latest/state queries |
| `passage_first_entity_budget` 676 | 0 | LIVE — the master valve: 0 kills P1 recording, TS learning, priming source (entity results), near-miss entity entries |
| `passage_first_durable_entity_slots` ~686 | 0 | dead (Gate-G knob, off) |
| `cue_recall_weight` 719 / `auto_recall_min_cue_score` 1801 | 0.65/0.15 | LIVE (wave2 sets cue_recall_enabled) |
| `consolidation_prune_{activation_floor,min_age_days,min_access_count}` 964-966 | .05/14(30 quiet)/2 | LIVE in cold-brain mop prune |
| `compact_strength_decay` 1048 | 1.0 | knob live, default = no-op by design (eval-gated) |
| `importance_prior_enabled` 1519 | False | DEAD (eval-gated; writer wired at apply.py:848-855) |
| `goal_priming_*` 1628-1663 | off | DEAD on quiet (conservative/standard only) |
| `conv_context/fingerprint/session_seeds/near_miss` 2223-2278 | off, but **quiet→wave2 forces ON** (config.py:2818-2837) | LIVE on default installs |
| `conv_context_rerank_weight` 2280 | 0.05 | LIVE (fingerprint boost) |
| `retrieval_priming_*` 2336-2359 | off (wave3+) | DEAD on default (wave2) — priming buffer stays empty; NOTE M4.3's session drift persists anyway via WM/conv-context |
| `memory_maturation_enabled` 2472 + `decay_exponent_semantic` 2475 | quiet sets True | LIVE for scoring decay; only identity_core ever reaches semantic tier |
| `reconsolidation_*` 2481-2485 | off (standard only) | DEAD on default |
| `working_memory_seed_energy` 526 | 0.3 | LIVE |
| `actr_*` 377-379, `spreading_strategy` | bfs prod | ACT-R strategy benchmark-only (review §4) |

## 11. Black holes / unknowns the rebuild must not repeat

1. **Master coupling: one flag, five behaviors.** `passage_first_entity_budget`
   simultaneously controls surfacing, access recording (P1), TS learning,
   priming-buffer input, and near-miss entity payloads. Any redesign that
   re-enables entity slots re-enables the M4.1 collapse machinery wholesale.
2. **Route-overwrite trap** (router.py:100-108): every future ablation/tuning of
   the four core weights MUST patch `_WEIGHT_PROFILES`, not cfg. Two experiment
   arms have already been voided by this.
3. **Fresh-cfg clone** in `_activation_pool` (candidate_pool.py:140): sigmoid
   retunes will silently not reach pool ranking.
4. **`compute_activation` signature drift**: ~10 call sites omit
   `consolidated_strength` (§7) — flipping M3.3 on makes scorer/prune honor the
   prior while seeds/goals/dream/microglia don't. Grep target:
   `compute_activation(state.access_history, now, cfg)` (no 4th arg).
5. **Three temporal vocabularies** (router regex 15-19, cue detector
   pipeline.py:52-93, decomposer 312-313) classify independently; a redesign of
   "temporal queries" must reconcile or explicitly scope all three.
6. **get_context is a second closed loop** (P9 + layer-3 get_top_activated
   ordering, context_builder.py:1052/1203-1216): fixing recall recording alone
   leaves the highest-traffic surface self-reinforcing.
7. **Snapshot under-count semantics** (§9): brain-side and crash-window accesses
   are lost by design; prune protections are only as fresh as the last clean
   shell/MCP exit. Any "activation matters" redesign needs a durability story.
8. **Graph-row access fields are cosmetic** (§1): prune's SQL pre-filter runs on
   stale zeros — it works only because zero ≤ threshold, i.e. the filter is
   accidentally vacuous rather than correct. Either wire `snapshot_to_graph` or
   delete the columns.
9. **Confirmed-use plumbing already exists and is orphaned on the agent path**
   (P4): `apply_memory_interaction("used")` + response-mention partitioning are
   production code consumed only by the dashboard chat. M4.4 is a wiring
   problem (public feedback tool / harness citation), not a build problem.
10. **Episode channel has no frequency state at all** (§2/§6): cue hit_count is
    the only per-episode usage counter, and it exists only when the cue layer is
    on and only for cue-surfaced episodes. A frequency prior for episodes is
    greenfield.
11. **Session-state mutation-on-read** (§5): WM + conv-context writes happen on
    every recall regardless of `record_access` — the proven determinism/
    cross-persona pollution source (M4.3 finding 2, ticket-worthy). Any eval
    harness must choose fresh-manager or session-shaped explicitly.
12. **`exploration_weight` vs TS aliasing** (§10): two exploration terms share
    the slot; only one runs depending on `ts_enabled`. Flipping TS off
    (M4.3 recommendation) silently re-enables the deterministic 0.05 exploration
    term — the A/B is not "TS vs nothing".
