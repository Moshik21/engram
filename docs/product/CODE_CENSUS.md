# CODE CENSUS — 2026-07-24

**Scope:** `server/engram/` (412 .py, 157,576 lines), `server/tests/` (426 .py, 133,253 lines),
`server/engram/config.py` (3,435 lines / 660 fields), `storage/helix/schema.hx` (1,601 lines / 198 QUERY),
`native/helix-repo/**`, `docs/**`, `dashboard/src/**`, `website/**`, `installer/`.

**HEAD moved during the census:** started `d7c764e`, finished `d8bf60f`. `config.py` shrank 3,469 → 3,435
lines mid-run. Two concurrent workflows were editing `retrieval/pipeline.py`, `config.py`, `activation/**`,
`__main__.py`, and the untracked `evaluation/meter.py` + `evaluation/graph_kill_rig/**`. **Re-locate every
citation by symbol name, not line number.**

**Method:** fully static. Five lanes ran AST reference graphs, token indexes, dominating-condition analysis,
and read-only git. No test was run, no app code executed, no HTTP call made, no service touched. Findings
marked **[RV]** were independently re-verified by the synthesis lane at `d8bf60f` with a fresh grep.

**Standing rule honored:** nothing was deleted, renamed, moved, or committed. This document is the *input*
to a deletion decision, not the decision.

---

## 1. THE HEADLINE NUMBERS

| Label | Items | Lines | What it means |
|---|---|---|---|
| **ABANDONED** (safe to propose for deletion) | **13** | **~153** | Genuinely finished with. |
| **INERT-BUT-INTENDED** (bugs, not clutter) | **~45** | n/a — 45 *features* | Complete code, discarded output or unreachable default. |
| **INVISIBLY-LOAD-BEARING** (trap map) | **32** | **~14,000** | Looks dead to static analysis; is live. |
| **UNKNOWN** (honest, do not act) | **~19** | ~3,200 | Could not determine statically. |
| **Misleading docs** (correct, never delete) | **26** | 21 files | Assert things measurement has refuted. |

### The ratio, which is the finding

> **~153 lines are safely deletable out of 290,829 lines of Python. That is 0.05%.**
>
> **For every 1 line this census clears for deletion, ~91 lines were flagged by naive static analysis and
> turned out to be load-bearing.** (~14,000 : ~153.)
>
> **Unfinished features outnumber dead ones roughly 3.5 : 1** (45 INERT-BUT-INTENDED vs 13 ABANDONED),
> and the inert ones are worth vastly more.

Three supporting counts:

- **Config:** of 660 declared fields, exactly **15 have no reader**. Only **1 of those 15 is a real bug** —
  and it is a bug precisely *because* it is not abandoned (a profile writes it deliberately; nothing reads
  it). A grep for "unused config" finds the harmless 14 and misses the harmful 1.
- **Deleted-feature residue:** the two 2026-07 deletion commits (`f984b91` tier collapse, `5ca08d2` Thompson
  Sampling kill) left **zero dangling Python imports** and **one** surviving dead knob out of 660. The
  Python-level hygiene was near-perfect. All residue is in the layers the deleting commits did not reach:
  HelixQL, TypeScript, and docs.
- **Tests:** the "tests for deleted behaviour" category came back **empty**. Zero tests reference
  mature/semanticize/schema_formation. **Zero tests and zero docs are proposed for deletion anywhere in this
  census.**

### Method self-check (why the delete list is short on purpose)

Lane 1 produced **~180 false ABANDONED positives across two of its three passes** before finding a
same-file-reference bug in its own AST walker. It caught the error only because `create_app` and the entire
`engram dogfood` CLI appeared as orphans — obviously absurd. Had the false positives been obscure symbols
instead of famous ones, they would have shipped. **Treat every single-item ABANDONED call below as one bug
away from wrong.** That is the argument for the founder's no-deletion rule, not against it.

---

## 2. THE BUG LIST — INERT-BUT-INTENDED, ranked by what wiring would buy

These are unfinished features masquerading as clutter. **The fix is to wire them, not remove them.**
Ticket column cross-references the live task list so nothing is double-counted.

### Tier 1 — recall quality (the product's core claim)

**B1. The entity channel is starved by two defaults, and the ACT-R weights have nowhere to land.**
`config.py:802 passage_first_entity_budget` (default 0) + `config.py:630 entity_episode_traversal_source`
(default `"results"`) **[RV]**. `retrieval/pipeline.py:2220-2221` forces the exact budget;
`retrieval/service.py:127,131` and `retrieval/episode_traversal.py:41` both gate on `== "candidates"`.
`retrieval/scorer.py score_candidates()` — the sole applier of the ACT-R weights — runs only on the entity
channel, which the budget then truncates to `[]`. `config.py:640-652` states this in-source, verbatim: *"So
55% of the ACT-R weight budget cannot reach an answer."* Measured lift when flipped on a planted corpus:
**2/36 → 22/36** (M3_1_oracle_surface.md:116).
*Ticket #23, #3. Caveat before flipping:* GRAPH_THESIS §4-D prices an unpaid bill — at live defaults the
traversal appends up to 50 rows off three Project hubs holding 50.9% of edges, i.e. a recency dump.
**Disposition: this is the census's #1 item and it is already armed in `graph_kill_rig/arms.py`.**

**B2. `pool_total_limit` is shadowed by a computed value under its own name.** `config.py:416` **[RV]**.
`retrieval/candidate_pool.py:80` writes the key from a computation over four *other* pool knobs;
`candidate_pool.py:1023` reads `limits["pool_total_limit"]` with **no config fallback** — while
`candidate_pool.py:889` does the same pattern *correctly* with
`limits.get("pool_entity_query_limit", cfg.pool_entity_query_limit)`. So
`ENGRAM_ACTIVATION__POOL_TOTAL_LIMIT=200` is silently ignored. This is the exact `ef_search` shadowing shape,
in Python, sitting on the first knob an operator reaches for against **ticket #5 ("recall returns only 5-7
rows — candidate depth, not ranking")**. *New finding.* **One-line fix: mirror line 889's pattern.**

**B3. `evidence_client_proposals_enabled` is read by nothing.** `config.py:2965` **[RV: 11 hits repo-wide,
every one a write or an assertion]**. Declared with a 6-line docstring describing suppression semantics;
`config.py:3197` `_set(True)` by the standard profile *with a written safety rationale*
("safe now that proposals are span-verified and confidence-capped"); set by `evaluation/continuity.py:58`
and 7 test sites. Meanwhile `extraction/apply.py:211,227,239,295-306,320,604,619,666-693,757,871` and
`extraction/commit_policy.py:184` act on `client_proposal` **unconditionally**. Setting it `False` does not
disable client proposals. It is the only one of the 88 profile-set fields with no reader. *New finding —
this is a safety claim that does not hold.* **Wire the guard in `extraction/apply.py`, or retract the
docstring.**

**B4. Both output channels of usage decay are unconsumed.** `consolidation/usage_decay.py:166
demote_surfaced_never_used_results` has **zero production callers** **[RV: only 3 test lines, its own
docstring, and a config comment]** — so the shipped flag `config.py:1364
usage_decay_presenter_demotion_enabled` **cannot do anything at any value**. Separately, `usage_decay.py:127,
284,366` writes a `usage_decay_demoted_at` marker onto every demoted episode/entity and counts
`prune_feed_ready` — and **`rg "usage_decay" consolidation/phases/prune.py` returns zero lines [RV]**. The
feed has no consumer. `run_usage_decay` itself IS live (`hygiene_ops.py:686-714`), so this burns work every
mop window and discards it twice.

**B5. The only emitter of the `"used"` interaction is reachable only from the dashboard.**
`retrieval/chat_feedback.py:14 apply_chat_recall_feedback` **[RV]** — *note: the brief's
`apply_chat_response_feedback` does not exist; this is the real symbol, and it is NOT callerless.* Callers:
`chat_runtime.py:16,264` ← `stream_api_chat_sse_events` (`:310` ← `:145`) ← `api/knowledge.py:722` POST
`/api/knowledge/chat`. Outside `evaluation/smoke.py:491` it is the only path that marks a recall result used.
**No MCP or agent surface reaches it.** *This is the mechanism behind **ticket #7** ("the cue loop is 92%
absent and has NEVER recorded a use").* Deleting it would permanently cement a dead metrics gate; leaving it
dashboard-only means the gate stays dead for every MCP user, which is all of them.

**B6. The `activation_anomaly` notification channel was built and never given an emitter.**
`notifications/models.py:14` whitelists it; `dashboard/src/store/types.ts:1594` already renders it; **no
emitter has ever existed** (`git log -S` finds only the commit that introduced the notification system)
**[RV]**. The other four types all have emitters (`collector.py:120,157,195`, `temporal.py:84`).
*An activation-anomaly emitter would have surfaced **ticket #25** (spreading times out on ~100% of recalls,
burning ~74ms to return `{}`) months ago instead of it being found by hand.*

**B7. The graph traversal primitives the depth work needs are already written and were never called.**
`schema.hx:1506 get_two_hop_neighbors`, `:1514 get_entity_neighborhood`, `:1510 get_entity_cooccurrences` —
zero callers anywhere; `git log -S` dates all three to `b9a9258`, the original Helix backend commit, i.e.
**born dead, not orphaned**. The live path is 1-hop only (`helix/graph.py:1799,1806,1923,1931`). Project
memory records the binding lever as *"RETRIEVAL SURFACING not extraction — 1-hop neighbors never surfaced."*
*Ticket #3.* **Evaluate wiring `get_entity_neighborhood` before the depth A/B rerun.**

### Tier 2 — the instruments (currently blocking every decision)

**B8. The agent-experience battery's own self-test passes at 0/0.**
`server/tests/rigs/test_agent_experience_battery.py:72` **[RV verbatim]**: asserts
`total + len(skipped_live_only) == 10` and `score == total` and `f"{total}/{total}" in report`. With
`total == 0` all four hold, `machinery_clean` is vacuously true over an empty list, the report renders
`"0/0"`, and `format_battery_report(floor=0)` prints **PASS**. `total = len(runnable)` where `runnable`
excludes `live_only` questions — **one JSON edit converts a 10/10 gate into a green 0/0.** Today all 10 run,
so this is latent. But the same assertion pins the rig at **ceiling** (`score == total`, a perfect 10/10 on
a corpus where `_seed_episode_for` plants the expected tokens verbatim), which means **the rig has zero
headroom to show an improvement** — consistent with **ticket #18** ("cannot resolve a ±1 effect").
**Fix: `assert total == 10` plus un-pin the ceiling.**

**B9. Eight schema-drift guards have been dark since they were written.**
`test_helix_schema_contract.py:104,125,153,181,224,254,275,309` all skip via
`_native_generated_query_texts()` (`:347`) because `helixdb-cfg/.helix/dev/helix-repo-copy/helix-container/
src/queries.rs` **does not exist [RV]** and `.helix/` is gitignored. No codegen step exists in
`.github/workflows/ci.yml`, so they skip in CI too. These are the guards against schema.hx-to-Rust drift —
**the guard against the repo's own named silent-inert class.**
**Cheap partial fix: the helper skips if *any* path is missing, but `native/helix-repo/helix-python/src/
queries.rs` exists and is git-tracked (620 KB) [RV] — skip only on the ignored path and 3 of 8 run today.**

**B10. The durable-names blind spot was fixed on 2026-07-24 and immediately repeated one predicate
downstream.** `test_recall_surface.py:2781 test_relationship_triple_entity_filter` **[RV verbatim]** pins a
keep-list of exactly three entries — `GOLDEN_DECISION_1783643390`, `Konner Moshier`, `recall profile` — none
of which contains an arrow, a colon-joined token, a URL, or a `file:line`. It therefore cannot exercise
either regex in `extraction/promotion.py:290-297`, which is live in the durable rescue
(`context_builder.py:2458,2519`, `result_selection.py:11`). Lane 5 re-implemented the two regexes standalone
and ran them over 9 real facts from this repo's own `MEMORY.md`/`CLAUDE.md`: **6 of 9 are killed** —
`"triage -> merge -> calibrate -> infer -> replay"`, `"agent-proposes -> Ollama -> narrow"`,
`"oracle-surface POSITIVE (2 -> 22 of 36)"`, `"battery 3 -> 6 machinery-clean, then 6 -> 3"`, name
`pipeline.py:1284:5`, name `run:2026-07-24T10:30`. **0 of 3 keep-list entries expose it.**
*Directly relevant to **ticket #17** (prose-fragment Decisions as the next squatter class).*
**Fix the test first, watch it go red, then narrow the predicate to anchored triples.**

**B11. Four spreading/activation tests assert inertness and therefore cannot fail while the mechanism is
dead.** `test_community_spreading.py:557` (`abs(bonuses.get('n1',0) - bonuses.get('n2',0)) < 1e-6` — passes
identically when `spread()` returns `{}`, which **ticket #25** says is ~100% of live recalls),
`test_typed_edge_weights.py:154`, `test_fan_strength.py:77` (`assert len(bonuses) == 0`),
`test_episode_graph_signal.py:310` (`sr.score >= base_scores.get(sr.node_id, 0.0) - 1e-9` — the `.get`
default sits on the wrong side of the inequality, neutering exactly the divergence case the docstring
claims to guard). **One-line fix each: pin non-emptiness before asserting the zero.**

**B12. The episode-lane determinism guard cannot observe nondeterminism.**
`test_episode_graph_signal.py:361 test_repeat_recall_is_score_stable` builds `kwargs` once at `:363` and
reuses it for both `_run` calls at `:368,:369`, microseconds apart, so the ACT-R wall-clock decay term
cannot move. Reported to pass at +0/+1/+5s and fail at +30s. **Fix: inject a frozen `now`, as
`test_retrieval_scorer.py:723` already does.**

**B13. Three tests are permanently skipped by a double bind — including a prune-safety invariant.**
`test_negation_polarity.py:113,183` and `test_valid_to_semantics.py:115`: module `pytestmark` requires
HelixDB reachable, but the fixture always constructs `HelixGraphStore`, and the body then skips with
"SQLite-only test" because `HelixGraphStore` has **zero `self.db` occurrences**. Unreachable in both worlds.
The `valid_to` one guards *"an entity whose only edge is a TTL edge must NOT be returned by
`get_dead_entities`"* — **an invariant that governs deletion and is currently unverified anywhere.**

**B14. `requires_docker` is declared, filtered against, and carried by zero tests.**
`pyproject.toml:109` + `Makefile:109,113` `pytest -m "not requires_docker"` **[RV: zero test files use it]**.
`make test` advertises that it excludes Docker-dependent tests, and there are none to exclude — so
**FalkorDB (2,476 lines) + Postgres (1,326 lines) have no automated verification at all.**
**The marker is the ready-made hook for one boot smoke test.**

### Tier 3 — silent-inert vectors in the config read/write path

**B15. `_set()` performs no field-name validation.** `config.py:3101`, 157 `_set(` call sites **[RV]**.
`if not force and field in self.model_fields_set: return` then `object.__setattr__(...)`. A typo'd name is
never in `model_fields_set` (guard never fires) and `object.__setattr__` will not raise — the profile
silently writes a phantom attribute. Worse: the ~90 `getattr(cfg, "name", default)` readers **would** see it,
so a typo can produce a *half-applied* config. No test asserts that every `_set()` name is a declared field.
**One-line guard: `assert field in type(self).model_fields`.**

**B16. Every config read has a silent default, so a rename turns features off with all tests green.**
`_stage_timeout_seconds(cfg, field_name)` at `candidate_pool.py:737` and `pipeline.py:347` does
`int(getattr(cfg, field_name, 0) or 0)` with the name passed as a *string literal* from ~25 call sites; a
third copy at `post_process.py:210`. ~40 `*_timeout_ms` fields and ~50 more (`budgets.py` has 15 getattr
sites, `context_builder.py` 11, `capture_service.py` 11 **[RV]**) resolve this way. **A rename in
`config.py` can turn every retrieval timeout off across the whole pipeline, and nothing fails.** This is the
project's own silent-inert bug class living in the config *read* path.

**B17. HNSW and map-size config is enforced by Rust literals, and the shipped JSON advertises different
numbers.** `storage/helix/config.hx.json:5 ef_search: 512` / `:11 db_max_size_gb: 50` vs
`native/helix-repo/helix-python/src/queries.rs:100 ef_search: Some(768)` / `:105 db_max_size_gb: Some(20)`
**[RV both]**. `NativeTransport.initialize()` (`native_transport.py:107-160`) reads no config path at all.
Consequences: (a) **ticket #4's ef_search rebuild** and `RECALL_PERFORMANCE_PLAN.md` M1 both aim an edit at
a file that cannot take effect — the change must land in `queries.rs` and be rebuilt; (b) **ticket #1's**
"84.8% of map_size" is 84.8% of **20 GB**, not the 50 GB the shipped JSON describes, so any headroom an
operator computes from the JSON is **2.5× too optimistic on the exact axis that causes MAP_FULL**.
Also: `config.hx.json` declares `"embedding_model": "gemini:..."`, against the fully-local north star.
**Related:** `config.py:119 hnsw_ef_runtime` has zero readers — it is the natural home for the ef_search
knob and is connected to nothing.

**B18. Engram drains an offline queue that nothing ever fills.** `utils/offline_queue.py:21 append_to_queue`
has zero production callers **[RV: tests + `docs/REFERENCE.md:837-840`, which presents it as a public API]**,
while the reader is fully wired at every startup — `main.py:107-111 → OfflineReplayService`
(`ingestion/offline_replay.py:39-48`). **Half a loop. Decide whether this is an external-writer contract
(document it) or an unfinished internal path (wire a writer).**

**B19. `ServerConfig.host/.port/.log_level` are read nowhere; `fly.toml` sets them anyway.** `config.py:43`.
Only `auto_observe_enabled` of ServerConfig's four fields is live (`api/knowledge.py:291`). Every launch path
hardcodes: `__main__.py:503-507,855-859` from argparse defaults, and `server/Dockerfile`'s CMD literal.
`fly.toml:9-10` declares `ENGRAM_SERVER__HOST="0.0.0.0"` / `PORT="8100"` and **neither has any effect.**
The values coincide today, so nothing is broken — an operator changing the port would see nothing happen.
Note the security comment above `host` ("must not bind the LAN unless explicitly asked") documents a
guarantee this field does not enforce.

**B20. Rate limiting is inert on every local install.** `config.py:133`, 5 fields, all read at
`main.py:373-377` and then discarded: `redis_for_metering` is assigned only when `mode == EngineMode.FULL`,
and `security/rate_limit.py:36-41` opens `check()` with `if self._redis is None: return True, -1` — it
returns *allowed* before consulting `self._limits`. So on native/lite/Helix-HTTP,
`ENGRAM_RATE_LIMIT__ENABLED=true` changes nothing. Compounding: `AuthConfig.enabled=False` by default and
`~/.engram/.env` sets `ENGRAM_AUTH__ENABLED=false`. **Either add an in-process window for non-FULL modes, or
make the config refuse `enabled=true` outside FULL loudly instead of accepting it silently.**

### Tier 4 — smaller, cheap, real

| # | Item | Evidence | Fix |
|---|---|---|---|
| B21 | `extraction/harness_metrics.py:157 record_external_extractor_invoked` | Zero call sites **[RV]**. Its siblings are wired. Already documented as dead: `GRAPH_THESIS.md:82` (M24) — *"it appears to corroborate M12 and must not be cited for it."* | Wire at the invocation site. Deleting converts a *known*-dead instrument into an *absent* one. |
| B22 | `schema.hx:1569 link_episode_chunk` never called **[RV: only its own definition]** | `E::HasEpisodeChunk` (`:1547`) has zero instances; `get_episode_chunks` (`:1594`) traverses it and would return `[]` for every episode. Chunking itself IS live via `episode_id` string on vector hits (`helix/search.py:1481,1499,1919-1941`). | Decide: link chunks (gives enumeration + cascade cleanup) or record that chunks are index-only. |
| B23 | `models/episode.py:71-73 memory_tier / consolidation_cycles / entity_coverage` frozen at defaults | Sole writer was the deleted `semantic_transition` phase (`git show f984b91^`). Now `worker_batching.py:276-282` only copies them forward. Still persisted in all three backends (`sqlite/schema.sql:67-68`, `schema.hx:355,385`, `falkordb/graph.py:805-806`). | Re-introduce a writer (an episode consolidation counter is a real triage signal) or mark frozen. |
| B24 | `recall_tier_aware_truncation` is **doubly** dead | Flag `config.py:579` default False, set True by no profile (only `test_recall_result_builder.py:40`); and `result_builder.py:65-80` branches on `memory_tier in ("transitional","semantic")` which B23 makes unreachable. `graph_manager.py:2019-2021` still documents it as working. | Re-ask the design question deliberately; it was answered by accident when the producer was deleted. |
| B25 | `embeddings/graph/gnn_inference.py` never receives the trainer's weights | `gnn.py:216-225` builds `layer_{i}_self_weight`/`_neigh_weight` and `gnn.py:107-108` stores `self._last_weights`; nothing loads them into `GraphSAGEInference`. The two halves were built to fit and never joined. | Low urgency (GNN is on standby below 200 entities), high clarity. |
| B26 | `loop_adjustment.py:375 load_active_adjustment_async` never called | All 3 prod sites use the sync twin (`hygiene_cli.py:96-99`, `scheduler.py:103-106`, `loop_adjustment.py:699`), whose own comment at `:348` says async stores need the async one and whose graph branch only fires on a `get_loop_adjustment_sync` attribute. **Guessing:** on Helix-native the sync path silently falls through to `_load_from_file`, so the graph-backed loop overlay is never read. | Verify against the native store's method surface, then wire. |
| B27 | `models/consolidation.py:11 ConsolidationStatus` unused while 3 layers depend on its literals | Zero importers. `consolidation/engine.py:179,268,271` writes raw strings; `dashboard/src/store/consolidationSlice.ts:28` + `api/client.ts:1348` consume them by value. The enum is the only written-down statement of the wire contract. | Use it at the write sites, or add a test pinning literals against it. |
| B28 | `embeddings/provider.py:475 prefix_cosine_similarity` — zero references anywhere | Its own docstring names its intended consumers: *"Use for bulk comparisons in consolidation (triage, merge, dream)."* None of the three call it. **Guessing:** I did not measure whether those loops are similarity-bound. | Wire, or record as a rejected optimization. |
| B29 | `NerveCenterConfig.enabled` (`config.py:3308`) defaults True and **is never read** | Only 2 of 10 fields have readers — `consolidation_trigger.py:412-413` **[RV]**. The subsystem presents as on and cannot be turned off. The 7 tuning multipliers below it are the delete candidate *contingent on this decision*. | Fix `enabled` first; then decide the subsystem as one unit. |
| B30 | `runtime_state.py build_fast_runtime_packet()` writes `"status": "not_inspected"` no consumer reads | Carried over from the brief; corroborated by `INSTRUMENT_AUDIT.md` AUDIT-11. | Add the consumer, not drop the field. |

---

## 3. THE SAFE-DELETE LIST — ABANDONED only, ordered by size

**Total: 13 items, ~153 lines.** Every one has been checked for decorator registration, string-factory
dispatch, entry points, `python -m`, argparse, conftest, the bash installer, and dashboard TS.

| # | Path | Lines | Evidence nothing reaches it |
|---|---|---|---|
| D1 | `config.example.yaml` | 36 | `EngramConfig.model_config` (`config.py:3329-3335`) registers no `yaml_file` and no `settings_customise_sources`; there is **no `import yaml` anywhere in `server/engram`** and no `config.yaml` reference in `server/`, `Makefile`, or `installer/`. Worse than dead: its values contradict the code (`B_mid: -0.5` vs -4.0, `spread_energy_budget: 5.0` vs 50.0, `weight_semantic: 0.50` vs 0.40) and it sets `server.host: 0.0.0.0` against the security-motivated default. **Prefer regenerating from live defaults over deleting.** |
| D2 | `extraction/narrow/base.py` (`NarrowExtractor` Protocol) | 26 | **[RV: the only occurrence of the bare name is its own `class` statement at `:14`.]** No class annotates against it, no `isinstance` uses it. The live type is `NarrowExtractorAdapter` in a different module (`extraction/narrow_adapter.py:16` ← `graph_manager.py:629`, `extraction/factory.py:96-99`). **⚠ Lanes disagreed** (Lane 1 said UNKNOWN — "26 lines of interface documentation is nearly free"). Recommended disposition is **adopt it** (annotate the four narrow extractors) rather than delete. Founder's call. |
| D3 | `server/engram/mcp/tools.py` | 24 | Pure re-export shim of six names from `mcp.surface`; **zero importers anywhere** — prod, tests, scripts, docs, installer **[RV]**. Its own docstring concedes the tools live elsewhere. Checked for the obvious trap (a plugin loader importing by string): none. |
| D4 | `server/engram/scratch/verify_synapse.py` | 19 | Ad-hoc scratch script, zero importers, `if __name__ == '__main__'` print. **`pyproject.toml [tool.hatch.build.targets.wheel] packages = ["engram"]` means it ships inside the installed wheel to every user** — the only concrete user-facing harm in the census. Second hazard: its function is named `test_synapse`, so pytest would collect it if `engram/scratch/` ever entered a test path. **Prefer moving to `server/scripts/` over deleting.** |
| D5 | `retrieval/feedback.py:1214 partition_recall_entities_by_usage` | 10 | Tests only (`test_recall_feedback.py:23,53`) **[RV]**. Thin entity-only wrapper delegating to the live generalized `partition_recall_targets_by_usage` (`:1219` ← `chat_feedback.py:10,34,100`). Superseded when cue_episode results entered the target shape. |
| D6 | `schema.hx:1342,1352,1362 create_consol_maturation / _semantic_transition / _schema` | 9 | Producers deleted in `f984b91`. **⚠ Keep the matching `find_consol_*_by_cycle` readers** — `helix/consolidation.py:1458-1462` calls them deliberately, with the comment *"entries stay so retention drains their legacy audit rows from existing brains."* **Caveat: deleting HelixQL requires a native rebuild and the benefit is unmeasured (see Black Hole H4). Bundle with other schema work or skip.** |
| D7 | `retrieval/scorer.py:260 extract_near_misses` | 9 | Tests only (`test_conversation_retrieval.py:18,280,284,291`) **[RV]**. The live split is `request_policy.py:41 split_primary_and_near_miss_results` ← `retrieval/service.py:166`. Semantics differ (unbounded tail vs bounded window), so they are not interchangeable. **⚠ MUST fix `docs/product/experiments/RF_cartography.md:144` in the same change** — that doc cites this function as part of the *live* near-miss mechanism, so the stale doc is the actual hazard. |
| D8 | `retrieval/feedback.py:1167 extract_recall_entities` | 7 | **[RV: literally one hit repo-wide — its own `def`.]** Zero references in prod, tests, docs, or scripts. Entity-only filter over the live `extract_recall_targets` (`:1160`). The safest single item in this census. |
| D9 | `dashboard/src/store/types.ts:1591-1592` + `components/nerve/ImmunitySweep.tsx:27,29,38,40` | 6 | `"schema_discovery"` / `"entity_maturation"` were removed from the backend whitelist in `f984b91`; `notifications/models.py:38` raises `ValueError` on any type outside it, so these can never reach the client. Cross-boundary residue: the deleting commit cleaned Python and missed TypeScript. |
| D10 | `evaluation/brain_loop_report.py:2016 report_to_dict` | 3 | **[RV: one hit — its own `def`.]** Docstring says *"Compatibility helper for dataclass-based callers"*; there are none. |
| D11 | `storage/helix/graph.py:413-414 _schema_member_id_cache` | 2 | Exactly one occurrence repo-wide: the `__init__` assignment. Written, never read, never populated. |
| D12 | `config.py:517 feedback_enabled` (+ 3 assertions in `test_config_roadmap.py:45,161,171`) | 1 (+3) | **[RV: 4 hits total.]** Last survivor of the feedback_events store deleted in `f984b91`, which also deleted the sibling `feedback_ttl_days` on the very next line. Zero production readers; the three tests only assert its own default — the classic signature of a flag whose consumer is gone. **⚠ `test_config_roadmap.py` is modified in the working tree by a concurrent workflow — coordinate.** |
| D13 | `config.py:51 wal_mode` | 1 | Zero readers **[RV]**. WAL is applied **unconditionally** at `storage/sqlite/graph.py:148 PRAGMA journal_mode=WAL` **[RV]** with no reference to the config value. `wal_mode=False` silently does nothing. A knob that promises a choice the code does not offer. **⚠ Lanes disagreed** — Lane 2 held UNKNOWN because it had not checked the PRAGMA; Lane 3's evidence resolves it. |

### Downgraded to UNKNOWN — items a lane proposed for deletion and I am refusing

- **`retrieval/state.py:135,163,174,189`** (`infer_time_bucket`, `entity_type_to_domain`,
  `compute_state_boost`, `infer_cognitive_state`, ~44 lines). Lane 1 proposed deletion but flagged doubt.
  **Cross-lane evidence resolves against deletion:** Lane 2 independently found `state_dependent_retrieval_
  enabled` is one of the 30 flags **no profile ever sets**, gating `state_arousal_match_weight` /
  `state_domain_weight` at `state.py:106-124`. These four helpers implement a *different* formula from the
  live `compute_state_bias` (`pipeline.py:1714-1720`) — competing designs, not duplicates. This is more
  likely the unbuilt half of a parked feature than residue. **Also correct the false section header at
  `state.py:129` ("Additional helpers used by scorer and tests") — `scorer.py` references none of them.**
- **`storage/helix/config.hx.json`** (14 lines). Lane 2 proposed deletion but conceded it did not verify
  whether an out-of-tree `helix deploy` toolchain consumes it. **Do not delete. Add a one-line header
  naming `queries.rs` as the authority** (see B17).
- **`schema.hx` SchemaMember node/edge/queries** (~22 lines). Lane 3 guessed the `N::SchemaMember` /
  `E::HasSchemaMember` declarations must stay because existing brains may hold instances, and unlike the
  Consol\* audit nodes **there is no drain path that reads them**. **Better move: ADD a drain** (reuse
  `find_schema_members` + `hard_delete_schema_member`) so the legacy nodes are actually reclaimed.

---

## 4. THE DO-NOT-TOUCH LIST — a trap map for the next sweep

Everything here is reached by a path static analysis cannot see. **A naive sweep would have deleted roughly
14,000 working lines.** Be generous reading this list; it is cheap insurance.

### Resolved by string, not by import

| Trap | Where the string lives |
|---|---|
| **9 benchmark adapter classes, ~1,200 lines** — `BaselineAdapter`, `VectorRagAdapter`, `ContextSummaryAdapter`, `MarkdownCanonicalAdapter`, `HybridRagTemporalAdapter`, `LangGraphStoreMemoryAdapter`, `Mem0StyleMemoryAdapter`, `GraphitiTemporalGraphAdapter`, `EngramAdapter` (`benchmark/showcase/adapters.py:196,915,1000,1042,1068,1147,1282,1429,1745`) | `create_primary_adapter` (`adapters.py:2054-2096`) and `create_ablation_adapter` (`:2099`) dispatch by literal name. Entry: `scripts/benchmark_showcase.py:13`. **The single largest and most dangerous false positive in the repo.** |
| **`activation/bfs.py`, `activation/ppr.py`, `activation/actr.py`** | `activation/strategy.py:36 create_strategy` resolves `"bfs"`/`"ppr"`/`"actr"` from a config string, with the imports *inside* the branches (`:38-49`). Invisible to any import graph. |
| **~40 `*_timeout_ms` fields + ~50 more** | `_stage_timeout_seconds(cfg, "field_name")` (`candidate_pool.py:737`, `pipeline.py:347`, third copy `post_process.py:210`); `getattr(cfg, "name", default)` in `budgets.py`, `hygiene_ops.py`, `loop_adjustment.py`. An IDE "find usages" returns **nothing** for these fields. |
| **`ingestion/decision_materializer.py:244` reads `DEFAULT_DECISION_VOCABULARY`** (`config.py:21-37`) | Live via string lookup. Note the fresh-install risk: the shipped default vocabulary hardcodes *this* project's predicates (`public_launch_path`, `integration_profile`, `subject_terms:Engram`). On any other install it matches almost nothing — relevant to **ticket #9**. |

### Registered by decorator or entry point

- **All ~20 `@mcp.tool()` functions** (`mcp/server.py:937,1017,1050,1209,1238,1378,1488,1540,1586,1638,1677,
  1708,1721,1767,1779,1826,1884`) and **all ~40 `@router.*` FastAPI routes** across `engram/api/**`.
- **MCP resources and prompts** (`server.py:2045,2053,2065,2080,2089`). Confirmed **[RV]**: `apply_mcp_surface`
  (`mcp/surface.py:106`) calls **only** `mcp.remove_tool` (`:135`) — there is no `remove_resource` and no
  prompt filter, so `engram://graph/stats`, `engram://entity/{id}`, `engram://entity/{id}/neighbors`,
  `engram_system` and `engram_context_loader` are **live at every surface level including the default
  `public`**, even though nothing references them.
- **`server/engram/__main__.py`** (~1,010 lines) — zero importers; declared in `pyproject.toml
  [project.scripts]` as `engram = "engram.__main__:main"`.
- **`evaluation/dogfood.py`** (~4,518 lines, ~25 public functions) — no external importer; reached entirely
  through `__main__.py:202-208` argparse + `run_dogfood_command` (`:644-650`) dispatching in-file.
- **`consolidation/drain_evidence.py` and `consolidation/__main__.py`** — 5-line `python -m` shims,
  documented in `CLAUDE.md` and `docs/ARCHIVE_HANDOFF_2026-07-10.md:215,268,301-303`.

### Test-only importers that are the *correct* topology

- **`quality/native_surface_manifest.py`** (428 lines) — a static contract manifest whose only legitimate
  consumer *is* its test. Its docstring says so: *"new REST/MCP surfaces must be classified instead of
  hiding in the route table."*
- **`storage/helix/availability.py`** (119 lines) — zero prod importers, 5 test importers rooted at
  `tests/conftest.py:25-29`, driving every `@pytest.mark.skipif(not _helix_available())` in the suite.
  *(Separately: its docstrings claim a `doctor` consumer that does not exist — see MD18.)*
- **Four production symbols that exist to make modules testable:** `candidate_pool.py:282
  clear_durable_feeder_cache`, `packet_cache.py:688 packet_cache_json_size`, `context.py:382
  manager_conversation_embed_fn`, `native_surface_manifest.py:427 identifiers_by_kind`. Deleting these takes
  the tests with them.
- **`test_ts_kill.py`** — looks like "a test of deleted behaviour" and is the opposite: an anti-resurrection
  contract asserting `score_candidates_thompson` is gone, `engram.activation.feedback` raises ImportError,
  the `ts_*` knobs are absent from `model_fields`, and **v2 snapshot/journal loaders tolerate-and-drop
  legacy `ts_alpha`/`ts_beta` rather than crashing** — that last part guards live on-disk data.
- **`test_salience_classifier.py`** — the model to copy, not to touch: real captured corpus from the live
  brain, a **distinctness gate** (added after a prior instance of this exact bug class padded the corpus with
  duplicates), a zero-false-positive gate, a recall>0.8 gate, and five targeted guards.

### Deliberate residue that looks like leftovers

- **`storage/sqlite/graph.py:416-421 DELETE FROM schema_members`** — a legacy-DB migration for pre-`f984b91`
  lite DBs, wrapped in try/except with the comment *"silent-ok: fresh DBs have no schema_members table."*
  **This is the correct pattern; the Helix side lacks it.**
- **`entity_dedup_policy.py:64 "Schema"`, `:70 "ClarificationIntent"`** — minters deleted, but they remain in
  `_CANONICAL_ENTITY_TYPES` because *"entity_type is an open case-sensitive string ... Case is normalized once
  at commit time; stored rows are never rewritten"* (`:45-52`). `"Schema"` additionally gates merge exemption
  at `phases/merge.py:49`, protecting legacy rows from fuzzy merges.
- **All Thompson-Sampling residue** (`storage/memory/activation.py:453`, `storage/redis/activation.py:72-73`,
  tombstone comments at `scorer.py:124`, `pipeline.py:2272`, `feedback.py:1095`) — tolerate-and-drop shims
  plus recorded verdicts. **Cite `5ca08d2` as the reference pattern for how a deletion should handle its own
  residue.**
- **`config.py:2879 memory_maturation_enabled`** — sits directly under a comment saying the maturation phase
  was deleted, so a sweep reading the comment deletes the flag. It is **live**, read by `retrieval/scorer.py`
  and `phases/prune.py`, `_set(True)` by four profiles. It was correctly *repurposed* from "run the phase" to
  "gate tier-aware decay and prune resistance." **The in-repo example of retiring a phase without orphaning
  its knobs.**
- **`storage/helix/graph.py:3550-3552 "# Maturation queries"`** — a section banner naming a deleted phase
  over **two live methods** (`get_entity_episode_count` ← `graph.py:2980`; `find_entities_by_type` ←
  `context_builder.py:2388`). **Retitle the header; do not delete the methods.** This is exactly how a sweep
  deletes something working.
- **`storage/helix/proto/helix_pb2_grpc.py:61,85,109`** (~137 lines) — protoc-generated servicer/server
  halves, lint-exempted at `pyproject.toml:96`. Engram is a gRPC *client*. Hand-edits are undone by the next
  `protoc` run.

### Env-var-only and script-only lanes

- **`storage/falkordb/graph.py`** (2,476 lines) — reachable: `storage/factory.py:234,288`,
  `storage/resolver.py:92-96,155-169`, the `full` extra, both compose files, `installer/engramctl:273`.
  Implements **65/65** protocol methods. **But zero test files reference it** (see B14). *Declared* and
  *verified* are different claims.
- **`storage/postgres/consolidation.py`** (1,326 lines) — constructed iff `ENGRAM_POSTGRES__DSN` is non-empty
  (`storage/bootstrap.py:144-150`), which no profile, compose file, or doc sets. Zero tests.
- **`benchmark/longmemeval/**`** (2,258 lines) — dataset and outputs were *deliberately untracked*, not
  deleted (`3e775d6`; `.gitignore:103-104`). The gitignore lines are the proof of intent.
- **`benchmark/locomo/**` + `scripts/benchmark_locomo.py`** (563 lines) — `GRAPH_THESIS.md:547-548` explicitly
  holds a rerun open. Blocker: the dataset it needs is not in the repo and not documented as downloadable.
- **`config.py:3168-3196` — the `integration_profile == "rework"` branch.** 29 `_set` calls are the **sole**
  enabler of ~13 named subsystems (cue policy learning, targeted projection, projector v2, epistemic
  routing/executor/reconcile, artifact bootstrap/recall, decision graph, answer contract, claim state
  modeling, cue vector index) **plus** the `passage_first_entity_budget=3` override that B1 identifies as the
  choke point. Live is `off`. Documented at `docs/REFERENCE.md:602-676`. **"Is feature X on?" has exactly one
  answer for all thirteen.**
- **42 parameters live only inside branches closed under the live profile** (`quiet`/`wave2`/`off`) and
  **30 boolean flags that no profile ever sets**. Every one has a complete implementation behind it. Full
  enumeration in Lane 2's findings; the gates are `observer_reflect_enabled`, `goal_priming_enabled`,
  `gc_mmr_enabled`, `reranker_enabled`, `cross_domain_penalty_enabled`, `state_dependent_retrieval_enabled`,
  `cue_policy_learning_enabled`, `immunity_enabled`, `reconsolidation_enabled`,
  `recall_tier_aware_truncation_enabled`, and the `episode_graph_signal_*` family. **Decide each GATE's fate;
  never the parameters under it.**

### In-flight — exclude from all disposition decisions

- **`server/engram/evaluation/graph_kill_rig/**`** and **`server/engram/evaluation/meter.py`** (~1,100-1,500
  lines) — **UNTRACKED [RV: still `??` at `d8bf60f`]**, along with `tests/test_graph_kill_rig.py`,
  `tests/test_recall_meter.py`, `tests/rigs/recall_meter_rig.json`. `graph_kill_rig` gained four files
  *between two tool calls* during Lane 1's pass. `__main__.py` is modified in the working tree and imports
  `meter.py`. **This is the concurrent measurement workflow. It has every signature of an abandoned lane —
  no `__init__.py` originally, zero importers — and is the newest code in the repo.** The strongest live
  illustration of why an unused-code sweep is dangerous here.

### Three config tests that look tautological and are not

Of 28 fully-tautological "construct then assert the same literal" tests, **three are real assertions** —
they set a value and assert a *separate layer* did not stomp it: `test_consolidation_profiles.py:90`
(`test_explicit_override_after_profile`), `test_runtime_role_and_quiet.py:28`, `test_entity_mat_tier.py:73`.
**For those, the echo IS the assertion. Keep them.**

---

## 5. THE MISLEADING-DOCS LIST — ranked by likelihood of causing a wrong decision

**Zero docs are proposed for deletion.** A refuted doc is evidence; the fix is a banner or a correction.

| # | Location | The false sentence | What refutes it |
|---|---|---|---|
| MD1 | `CLAUDE.md:5` | *"uses ACT-R cognitive architecture for activation-aware retrieval"* | `GRAPH_THESIS` M3: `activation` = 0.0 on **55/55** live recall items. M4: episode/cue/chunk `ScoredResult`s are built with `spreading=0.0, edge_proximity=0.0` **literals** (`pipeline.py:842-843,908-909,992-993`) — no graph or activation signal *can* reorder an episode. M4.1: activation-as-ranker measured NEGATIVE (reach 23→2/36). **Highest blast radius in the repo: auto-loaded into every session. `README.md:212` already states the correct position — the two most-read files contradict each other.** Ticket #23. |
| MD2 | `docs/REFERENCE.md:1534-1538` | The section titled *"honest ceiling contract"*, with four hyperlinked artifacts | **[RV] All four are MISSING from disk:** `longmemeval_ceiling_llm_graphoff.json`, `lme_lite_llmreader.json`, `cs_off.json`, `cs_blend.json`. The **only** cited artifact that still exists is `longmemeval_final.json` — the one the same doc explicitly retracts as a broken-harness artifact. `server/results/` is git-ignored, so git cannot say whether they were deleted or never written, and all four links render dead on GitHub. **An unverifiable number inside an "honest contract" section is the exact failure `INSTRUMENT_AUDIT.md` forbids.** |
| MD3 | `server/results/longmemeval_final.md:19` | *"**Overall accuracy: 20.2%** (101/500)"* under a clean "LongMemEval Benchmark Report (ORACLE)" heading | The retraction exists only in `docs/REFERENCE.md:1540` (*"a broken-harness artifact (`Embedding calls: 0`); do not cite it"*) — a different file. `rg -i "retract|broken|artifact|do not cite"` over the artifact itself returns **0**. It is git-ignored, so review will never catch it, and it is the only LongMemEval artifact left on disk for a grep to find. **Prepend the banner to the artifact; do not delete it — a retraction is only credible if the thing it retracts survives to carry it.** |
| MD4 | `docs/install/lite.md:4-5,127-135` | *"the full **17-phase** consolidation pipeline, graph embeddings, and **schema formation**"*; *"Memory maturation (episodic → transitional → semantic)"*; *"Activation-aware retrieval with spreading activation"* | **[RV] `ls server/engram/consolidation/phases/` = 15 phases, no mature/semanticize/schema.** Spreading completes on **0/15** live recalls (`GRAPH_THESIS` M7, timing out at ~74.5ms against `retrieval_spread_timeout_ms=75`). **This is the doc a new user reads first — the highest-consequence stale doc for anyone who is not the founder.** |
| MD5 | `docs/product/RECALL_PERFORMANCE_PLAN.md:49,187` | *"Battery stuck at 3-5/10"*; success criterion *"3-5/10 → 7+/10"* | **The same file refutes itself 138 lines later.** §7.4 (`:407-433`) tabulates the *same build* measuring `5,5,5,4,4` and `3,3,3,3,3` twenty minutes apart, and `1`→`4` inside one arm, concluding at `:468`: *"The battery cannot currently distinguish a +1 effect from noise, and this knob has now been mis-measured three times because of it."* The doc's own recorded OFF range is 1-5, wider than its header anchor. Ticket #18. Also `:3` still says **"Status: DESIGN — ready for a focused build effort"** while M1/M2/M4 shipped (`43b3df4`), M3 shipped (`fdc05d3`), M5 shipped, M6 was built and reverted — **a reader trusting the header rebuilds four shipped milestones.** |
| MD6 | `docs/product/AGENT_EXPERIENCE_GOAL.md:206-209` | *"Post recency chunk re-index — 3/10 — **regressed by** recall latency"*; *"Post salvage-on-timeout fix — 5/10 — **recovered** part of the regression"* | Every delta in that table (+3, −3, +2) is inside the ±2 noise band established by MD5's §7.4, so the causal attributions are unsupported. **Live consequence: M1.3 was PARKED (`reindex_sweep_enabled=False`) on the strength of the 6→3 reading.** Ticket #4 lists "reindex_sweep re-test". **Re-open that park decision as noise-bound rather than measured.** |
| MD7 | `docs/CHANNEL_SEPARATION_DESIGN.md:32` | *"**Episodes strictly first**: multi-hop **17/18** (graph-ON now beats OFF)"* — asserted flat, once, with no caveat | `GRAPH_THESIS:549-554`: the figure is ambiguous between two readings, `CHANNEL_SEPARATION_DESIGN.md:32` uses reading (b), and **"no raw result file exists for (b)."** Project memory independently records that the 17/18 breakthrough did not reproduce. The thesis' own verdict: *"The single largest strategic pivot in the project — tiering — was taken on the strength of a noise-bound non-reproduction of a number whose meaning is ambiguous."* **Single highest-value one-line doc edit available.** |
| MD8 | `skills/engram-memory/SKILL.md:274` — **published externally** (clawhub `engram-brain`) | *"**Knowledge graph**: Entities and relationships are extracted and connected (depth tier; the **proven** benefit is surfacing related episodes)."* | `GRAPH_THESIS` M2: **0 bytes** of relationship JSON across 55 recall items on 10 queries (third independent reproduction). M11: `relationship_ids: []` on 3/3 `get_context` packets. M9: the surfacing mechanism the sentence names defaults to `"results"` and is set by no profile. **The 2→22/36 result the word "proven" refers to was measured on a planted synthetic corpus behind a knob that is off.** This misleads *strangers' agents*. Also: the tool table lists 6 of the 9 frozen public tools (missing `forget`, `bootstrap_project`, `get_runtime_state` **[RV: surface.py PUBLIC_TOOLS = exactly those 9]**), `:155` warns against two tools not on the public surface at all, and `:329` says notifications are enabled by profiles consumers do not run. **Republish debt: 0.3.5 and 0.3.6 are both unpublished.** |
| MD9 | Eight files assert the Ollama extraction rung is live | `CLAUDE.md:128`, `AGENTS.md:27`, `README.md:264`, `docs/REFERENCE.md:130,197,206,1577,1684`; `GRAPH_THESIS` §4-C still *prices rebuilding it* as a live design option | The founder has ruled Ollama out. **No file in this repo records that ruling.** Compounding: `GRAPH_THESIS` M12 shows an LLM extractor has never run on this machine (**1894/1894** provider-resolution lines say `Narrow`) and M14 shows the configured endpoint is a dead Tailscale IP with an uninstalled model. **An undocumented decision that contradicts every written source is strictly worse than a stale doc — the next agent reads the docs and re-proposes Ollama, exactly as §4-C already does.** |
| MD10 | `CONTRIBUTING.md:104-112` — tracked, public, last touched 2026-03-11 | Four false statements in nine lines: *"12-phase consolidation: ... mature, semanticize, schema"* (**[RV] 15, none of those**); *"FalkorDB + Redis for production"* (Helix native is the recommended path); *"MCP server: stdio transport with **15 tools**"* (**[RV] frozen at 9**); *"see `CLAUDE.md` in the project root"* — **`CLAUDE.md` is git-ignored (`.gitignore:87`), so no contributor who clones the repo has that file.** Bonus: the Issues link points at `github.com/anthropics/Engram`. |
| MD11 | `docs/product/experiments/GRAPH_DENSITY_CENSUS.md:373,412` | *"**Gate 1 — restore a real extractor.** This is the single highest-leverage..."*; *"≥50% of entities have a non-structural incident edge (today: **22.3%**)"* | `GRAPH_THESIS` §3: *"Repairing the producer first and re-running the A/B would produce a third uninterpretable null, this time sourced in the consumer"* — at least five consumer defects are producer-independent. §2.4-4: the 22.3% counts `MENTIONED_WITH`, written by consolidation PMI (`phases/infer.py:236`) not extraction, so **"on the defensible reading the figure is 1.7%"** — a 30× gap, not a 2× gap, which moves the re-open criterion from near to far. **The census's data stands; its recommendations are superseded.** Ticket #8. |
| MD12 | `docs/REFERENCE.md:1269` | *"Full pipeline with spreading activation shows **+28% P@5**... Frequency queries are the standout at 94% precision — **this is where ACT-R activation shines**."* | Same refutations as MD1, plus M7 (spreading completes 0/15). The +28% is a synthetic 1K-entity `benchmark_ab.py` number run with `record_access=False`. **The numbers can stay if their regime is labelled; "where ACT-R shines" cannot.** |
| MD13 | `docs/REFERENCE.md:66,112` vs `:1256,1259` | *"~97ms search latency"* (twice) vs the benchmark table's *"Native search avg **238 ms**"* and *"nearly 2× faster than HTTP (425ms)"* | Same document, **2.5× apart, no reconciling note**; `README.md:249` propagates the 97ms. `RECALL_PERFORMANCE_PLAN` §1 measures the live hybrid at **134 ms warm / 2,600 ms+ cold** on the 18GB brain, so neither published figure describes the shipped path. I could not determine which is intended. |
| MD14 | The literal string **"17-phase"** in 5 files | `docs/install/lite.md:4,127`, `docs/install/helix.md:6`, `docs/install/openclaw.md:169`, `docs/REFERENCE.md:1782` | **[RV] 15.** This string is the fingerprint of the pre-deletion era; one global correction closes all five. |
| MD15 | `server/docs/improvement_roadmap.md:11,31` | *"**Thompson Sampling Exploration** — ... Effort: Research"* listed as **deferred priority #3** | Thompson Sampling was **built, measured, and KILLED** in `5ca08d2`, with anti-resurrection guards in `test_ts_kill.py`. **A doc that actively instructs a future agent to rebuild a rejected feature.** Same file `:14` lists Personalized PageRank as a to-do; PPR already exists (`activation/ppr.py`, reached via `create_strategy`). |
| MD16 | `docs/product/experiments/RF_cartography.md:144` | Cites *"scorer `extract_near_misses` 363-371"* as part of the live near-miss mechanism | Wrong line numbers and wrong file role — the live splitter is `request_policy.py:41`. **This stale citation is the only thing blocking safe-delete item D7; fix them in the same change.** |
| MD17 | `README.md:24` | *"Engram is fully-local, long-term episode memory for AI agents — **and it provably works**... deep recall reaches the gold memory in **42/42** gate queries."* | The 42/42 is real and correctly cited. But `AGENT_EXPERIENCE_GOAL.md:12-14` already wrote the missing clause: the agent-facing battery scored **3/10 on the same brain**; *"Both numbers are true: the mechanisms work, the experience doesn't."* `RECALL_PERFORMANCE_PLAN:150` adds that the live surface at `limit=25` returns only 5-7 rows with the answer absent on 5 of 10 questions. **Keep 42/42; refuse to let a passing proxy carry "provably works."** *(In fairness: `README.md:212` and `:221` are the most honest writing in the repo.)* |
| MD18 | `storage/helix/availability.py` docstrings | *"for tests and operator probes"*; *"Returns a JSON-serializable dict for tests/doctor"* | `doctor.py` never imports it. **Either wire `doctor.py` to `helix_available()` or fix the two docstrings — they document a consumer that does not exist.** |
| MD19 | `CLAUDE.md:8` / `AGENTS.md:35` | *"`server/engram/` (**~90 Python files**)"* vs AGENTS.md's *"~400 py files"* | **[RV] 412.** Trivial, but it is in the file every session loads, and the two disagree by 4.5×. |
| MD20 | `config.example.yaml` + `config.py:3327` | The `EngramConfig` docstring: *"Supports env vars, .env files, **and YAML**."* | No YAML source is registered anywhere (D1). The docstring is what keeps `config.example.yaml` looking live. **Strike ", and YAML" or register a YAML source — do not leave both.** |
| MD21 | `config.py:663` and `:649` docstrings | *"~4× the measured worst case (**9.5 ms**)"*; *"Per-call cost is a BACKEND property (**0.105 ms warm** on Helix native)"* | 60 × 0.105 = 6.3, so the second number **derives from** the first, and the whole cost model is a warm-cache extrapolation. The orchestrator reports the measured worst case is **399 ms** (~42× error — the budget is 10× too *small*, not 4× generous), and **ticket #26** records a cold-cache **2000×** cliff in `_resolve_episode_helix_id` (533 ms vs 0.263 ms). **There are ~40 timeout fields in this file and the cost model has no cold term anywhere.** Cold is the state a user experiences at session start. Also in this family: `episode_graph_signal_source`'s description says spreading *"timed out on 9/10 live recalls"*, while ticket #25 records ~100%. |
| MD22 | `BRAIN_ARCHITECTURE.md:393,396,414,525,528,536,580,589,619-620,635` and `SYSTEM_MAP.md:390,514` | Full phase orders containing `mature → semanticize → schema`; *"New files: consolidation/phases/mature.py, semanticize.py"*; *"`SchemaFormationPhase` — clusters mature semantic entities"* | Both dated 2026-03-05, both git-ignored (local-only, invisible to review, zero inbound links). **Do NOT delete — they are the only surviving record of why those phases were designed.** Prepend a `SUPERSEDED 2026-07` banner naming `CLAUDE.md` as authoritative, and propose archiving under `docs/archive/`. |
| MD23 | `docs/AGENT_HANDOFF_2026-07-16.md:57-77` | *"Live metrics (2026-07-16 morning): ~749 entities, ~8632 episodes... Continuity **PASS** (recall ~1.8s)"* | Live on 2026-07-24: 9,323 episodes / 840 entities / 1,637 relationships / 9,080 cues (`GRAPH_THESIS` M1). The doc is honestly dated — **but `docs/CURRENT_HANDOFF.md:3` and `AGENTS.md:6` designate it the canonical agent entrypoint, which converts staleness into a hazard.** |
| MD24 | `Makefile:6` | *"docker-compose defaults to consolidation_profile=standard (worker + consolidation + **maturation**)"* | Deleted phase. **Also noticed while reading:** `CLAUDE.md` says `make up` starts HelixDB, but `Makefile:15` `up` runs plain `docker compose up` against the **FalkorDB + Redis** stack; HelixDB is `make up-helix`. |
| MD25 | `docs/assets/readme/memory-tiers.svg`, `harness-flow.svg:60` | Both render a three-tier **maturation** diagram | Rendered in the README on GitHub. |
| MD26 | `website/public/llms.txt:37`, `llms-full.txt:116,123,137,138`, `DocsPage.tsx:108,122-123`, `SciencePage.tsx:147,151`, `RoadmapPage.tsx:22-23`, `VisionPage.tsx:444,491`, `HomePage.tsx:106` | *"17-phase consolidation ... mature, semanticize, schema"*; *"**Mature**: graduates entities through episodic, transitional, and semantic tiers"*; *"**Semanticize**: promotes episodes based on entity coverage and cycle count"* | Deleted 2026-07. **`llms.txt`/`llms-full.txt` are specifically the AI-agent-facing description of the product — a machine-readable false claim, not marketing drift. Prioritise those two over the `.tsx` pages.** Ticket #4 ("website sync"). |

**Docs verified current and worth protecting:** `docs/product/GRAPH_THESIS.md`,
`docs/product/INSTRUMENT_AUDIT.md`, `docs/product/experiments/GRAPH_DENSITY_CENSUS.md` (modulo MD11),
`docs/vision/science-of-engram.md` (carries its own dated amendment), `docs/CURRENT_HANDOFF.md`,
`docs/GOLDEN_LOOP.md`, `docs/product/WEEKLY_NORTH_STAR.md`, and `README.md:212,221`.

**Structural hazard: the best doc in the corpus is unreachable by navigation.** `GRAPH_THESIS.md` has **zero
inbound markdown links repo-wide [RV]** — not from README, CLAUDE.md, AGENTS.md, NEXT_LEVEL_OBJECTIVE, nor
from the census it corrects. It survives only because **code** reaches it:
`evaluation/graph_kill_rig/thresholds.py:3` transcribes its §5 pre-registered thresholds *verbatim*,
`arms.py:8,105` cite §4 and M16, `tests/test_episode_graph_signal.py:233` names itself *"the direct falsifier
of GRAPH_THESIS M3/M4"*. Same shape for `INSTRUMENT_AUDIT.md` (0 inbound links; enforced by
`test_metric_honesty_contract.py`). **The refuted plans are discoverable and the refutations are not.**
**And a verbatim transcription desynchronises the first time either side is edited — the repo's own dominant
bug class, at the doc layer.** Make `thresholds.py` assert against the doc rather than copy it.

---

## 6. WHAT THE SHAPE MEANS

**This repo does not have a dead-code problem. It has a finishing problem.** The numbers are not close.

- **~153 lines** are safely deletable, out of **290,829** lines of Python. **0.05%.**
- **~14,000 lines** looked dead to a competent static analysis and were load-bearing — a **91:1** ratio of
  false positives to true ones.
- **45 complete features** are wired to nothing, or wired to something that discards their output, or one
  default away from firing.
- **Of 660 config fields, 15 have no reader — and only 1 of those 15 is a real bug.** The dangerous shape is
  never *"nobody touches this."* It is **"the write side and the read side disagree."**
  `evidence_client_proposals_enabled` has a declaration, a docstring promising suppression semantics, a
  profile that sets it True with a written safety rationale, a continuity harness that sets it, seven tests
  that assert it round-trips — **and zero readers.** A grep for unused fields finds the harmless 14 and
  misses the harmful 1.

**The honest answer to the question you asked: there is far less dead code here than you expected.** The
deletion commits of the last two months were unusually clean — `f984b91` removed three phases and left zero
dangling imports and exactly one orphaned knob out of 660; `5ca08d2` removed Thompson Sampling and
deliberately left migration shims, tombstone comments, and anti-resurrection guard tests. That is better
hygiene than most codebases achieve. **The residue that survived is all in the layers the Python deletion
did not reach — HelixQL, TypeScript, and docs — and the docs are by far the worst of the three.**

**So the remedy is not a prune. It is a consumer test at merge time.** Four contract tests, each roughly
20 lines, would have caught the majority of this census before it became findable:

1. **Write-side/read-side lint.** Assert that every field named in a `_set()` call resolves to a declared
   field *and* has at least one reader. **Catches B3, B29, D12 — and would have caught them at authoring
   time.** (`config.py` has 157 `_set(` sites and no validation of any of them.)
2. **String-dispatch resolution test.** Assert every string literal passed to `getattr`-on-config and to
   `_stage_timeout_seconds` resolves to a declared field. **Closes B15 and B16 — the vector by which a
   rename silently turns off every retrieval timeout with all tests green.**
3. **Doc-truth test.** Assert (a) `CLAUDE.md`'s phase list == `ls consolidation/phases/`, (b) every
   `server/results/*.json` path cited under `docs/` exists on disk, (c) the public tool list in docs ==
   `surface.py PUBLIC_TOOLS`. **Catches MD2, MD4, MD8, MD10, MD14, MD26 — six of the top-ten misleading
   docs.** `INSTRUMENT_AUDIT.md` is currently the *only* doc in the repo with a code-side guard, and it is
   the only one that has not drifted. That is not a coincidence; it is the experiment already run.
4. **Factory-resolution test.** Assert every `BASELINE_CATALOG` id resolves through `create_primary_adapter`,
   and every `create_strategy` name resolves to a class. **Turns the two largest invisible-linkage traps
   into visible, tested edges** — which also means a future sweep sees them.

Add a fifth for the instruments, since they are what is actually blocking work right now: **every rig assertion
of the form `score == total` must be preceded by a floor on `total`** (B8), and **every test asserting a
mechanism produced zero must first assert the mechanism ran** (B11, B12).

**One more shape worth naming.** Roughly 25 of the 30 never-set boolean flags carry no explanation. But five
do — `observer_reflect_enabled`, `reindex_sweep_enabled` (with its measured 6→3/10 battery regression),
`importance_prior_enabled` and `supersession_enabled` (marked `EVAL-GATED` at `extraction/apply.py:38,53`).
**A flag with a written "why it is off" is not debt — it is a parked experiment with its verdict attached,
and it is good engineering.** The actionable split is not on/off; it is **annotated/unannotated**. Writing
one line of "why" for the other 25 is a bigger win than deleting all 153 abandoned lines, and it is the same
afternoon of work.

---

## 7. BLACK HOLES — what this census could not determine

**H1 — Nothing here was confirmed against a running system.** Every reachability claim is static text
analysis. A symbol this census calls load-bearing may sit behind a flag that is off, a budget of 0, or a
timeout that always fires. **`episode_traversal.py` and `scorer.py score_candidates` pass the reachability
check cleanly and are both dead in production** — that is B1. *To determine:* a runtime call-count probe, or
simply reading the config-default audit in §2 Tier 1 alongside the call graph. **The call-graph lane cannot
find the inert-by-default class by construction.**

**H2 — The effective config of the live process is unknown.** Every "live profile" claim in this census is
conditional on `~/.engram/.env` being what the LaunchAgent actually sources — **ticket #24** records that CLI
gives `standard`/node2vec-on while launchd gives `quiet`/node2vec-off. This census has the same defect it is
documenting: it reasons about a machine it did not read. *To determine:* **not more auditing — make the
runtime report its own effective config with per-field provenance** (value + which source won: literal
default / profile / dotenv path / process env). Until that exists, all 23 config findings are provisional.

**H3 — Does the live brain contain `mat_tier='transitional'` entities?** This single number decides
ABANDONED vs LOAD-BEARING for four production readers (`retrieval/scorer.py:44`, `entity_probe.py:216`,
`phases/prune.py:113`, `result_builder.py:71`). `CLAUDE.md` asserts maturation never graduated anything — but
the deleted phase was gated only on `memory_maturation_enabled`, which `standard`/`quiet`/`conservative` all
set True, and it ran in the Warm tier for months before 2026-07-17. *To determine:* **one read-only count
when the shell is free.** Same question for legacy `Schema` entities (are they surfacing as recall squatters
like the relationship-triples did?) and for stranded `SchemaMember` nodes, which have **no drain path at
all** unlike the `Consol*` audit nodes.

**H4 — Do unused HelixQL queries cost anything?** 23 of 198 queries have no Python caller **[RV: 198
QUERY declarations]**. If dead queries are free (no compile time, no binary size, no schema-migration
surface) this is documentation debt only; if not, it is the cheapest reclaim in the schema. *To determine:*
a native build (cargo/maturin) — forbidden this session. **This is why D6 carries a caveat.**

**H5 — Dead methods inside live classes are completely invisible to this census.** Only top-level `def`s were
classified. `GraphManager` alone is ~1,300+ lines of methods. **This is probably the largest remaining mass
in the repo and nobody looked.** *To determine:* an AST pass over class bodies with same-class call
resolution — the identical technique Lane 1 used at module level, one scope deeper.

**H6 — Dynamic dispatch that was not anticipated looks exactly like an orphan.** Decorators, string
factories, entry points, `python -m`, argparse, conftest, the bash installer and dashboard TS were all
hand-checked. **Not audited:** `getattr` with *computed* names, `globals()[...]`, `importlib.import_module`,
`pkgutil` walks, pydantic validator registration. If any exist, some ABANDONED label above is wrong.

**H7 — Are the 9 non-public MCP tools dead for every default install?** `apply_mcp_surface` removes tools by
name at runtime under `ENGRAM_MCP_SURFACE=public` (the default) **[RV: it calls only `remove_tool`]**. So
`loop_apply`, `loop_clear`, `loop_status`, `loop_propose_from_report`, `loop_steward_once`,
`record_recall_evaluation`, `record_session_continuity_evaluation`, `intend`, and `trigger_consolidation` may
be statically registered and runtime-removed everywhere. *This census flagged the question and did not answer
it.*

**H8 — Do the FULL (FalkorDB + Redis) and Postgres lanes still boot?** 3,802 combined lines implement 65/65
protocol methods, which is all static analysis can say. **"Conforms to the protocol" and "works" are
different claims.** Zero test files, and the `requires_docker` marker that would gate a smoke test has zero
users (B14). *To determine:* one Docker boot test — forbidden this session.

**H9 — Symbol matching was by bare name, so the orphan list is conservative and incomplete.** Names like
`forget`, `chat`, `main`, `apply`, `observe_image`, `trigger_consolidation` collide across modules; any
collision makes a genuine orphan look referenced. **Safe for the no-deletion rule, incomplete as an
inventory.** The masking rate was not quantified.

**H10 — The dashboard TypeScript was not audited** beyond its 16 test files (which are in materially better
shape than the Python suite: zero `.skip`/`.todo`/`.only`, zero blocks without an `expect`). Dead React
components, unused API-client methods, and orphaned Zustand slices are unmeasured.

**H11 — ~80 of 101 flagged test assertions are unreviewed.** Lane 5's "all assertions guarded by a loop or
`if`" pass produced 101 hits; ~20 were hand-verified and 4 reported. The rest are mostly false positives
(where the loop indexes a dict and would `KeyError`), **but that list is where the next real finding most
likely hides.** Saved at
`/private/tmp/claude-501/-Users-konnermoshier-Engram/2317d745-9e2f-4edd-b244-df7d3875b056/scratchpad/lane5census/s3.txt`.
Separately, 24 test functions with no in-body assertion were detected and **none were individually cleared** —
at least one (`test_mcp_authority_client_adoption.py:18`) delegates to a properly-asserting helper, so the
detector over-reports, but some of the other 23 may be genuine.

**H12 — The "OFF = 0" reproduction exists in a running workflow's head, not in the repo.** The lowest OFF
single-run recorded in any committed doc is **1** (`RECALL_PERFORMANCE_PLAN.md:141`). **If the concurrent
measurement workflow does not commit its per-run OFF distribution, this census will have documented the
"3-5/10" anchor without documenting its killer, and the next reader re-derives 3-4.**

**H13 — `CLAUDE.md` and `AGENTS.md` — the two files every agent session loads — are git-ignored/untracked.**
Consequences: `CONTRIBUTING.md:112` sends contributors to a file their clone does not contain; corrections to
the highest-blast-radius instructions in the project cannot be reviewed, cannot be attributed, and are lost on
a fresh clone. **I could not determine whether this is deliberate (private instructions) or accidental.**
Same class: `server/results/` is git-ignored, so **evidence loss is undetectable** — four artifacts cited by
the public `REFERENCE.md` honest-contract section are gone and git cannot say whether they were deleted,
moved, or never written (MD2).

**H14 — Doc coverage was partly mechanical.** Read in full: `RECALL_PERFORMANCE_PLAN`, `GRAPH_THESIS`,
`README`, `SKILL.md`, `CONTRIBUTING`, `CHANNEL_SEPARATION_DESIGN` head, `install/lite`. **Grep-sampled only:**
17 files under `docs/design/**`, 13 under `docs/product/experiments/**`, `docs/product/investigations/**`,
`ARCHIVE_HANDOFF_2026-07-10.md` (~6.5k lines), `Tech_Spec_v2.md`, `ROADMAP.md`, `CROSS_TOPIC_ANALYSIS.md`,
`CONSOLIDATION_HEALTH_ANALYSIS.md`, `refined/*.md` (11 files, all git-ignored), the other two skills, and
everything under `dashboard/`. **There are almost certainly more stale-phase and stale-benchmark hits there,
particularly in the git-ignored `refined/` set and the 2026-03-era root docs.**

**H15 — The repo changed under the census.** HEAD moved `d7c764e → d8bf60f`; `config.py` shrank by 34 lines;
the concurrent spreading lane added and then removed `spread_max_reads` / `spread_candidate_injection_max`
(excluded as in-flight); `graph_kill_rig/` gained four files between two tool calls. **Re-verify every line
anchor by symbol name before acting.**

---

### Suggested afternoon, in order

1. **B15** — one-line `_set()` field-name guard. Ten minutes, closes a silent-inert vector on the most
   consequential function in `config.py`.
2. **B2** — mirror `candidate_pool.py:889`'s pattern at `:1023`. One line, unblocks the depth knob on
   ticket #5.
3. **MD7 + MD1** — two doc lines: the 17/18 errata and the ACT-R sentence in `CLAUDE.md`. Highest
   wrong-decision-prevented per keystroke in the entire census.
4. **B8** — `assert total == 10` in the battery self-test. One line, protects ticket #18's instrument.
5. **B9** — make `_native_generated_query_texts()` skip only on the ignored path; 3 of 8 drift guards start
   running today against the tracked 620 KB `queries.rs`.
6. **B10** — extend the triple-entity keep-list with arrow/`file:line`/URL shapes, watch it go red, then
   narrow the predicate.
7. **D8, D10, D11, D3** — the four zero-doubt deletions (~36 lines), if deleting anything at all is worth
   the diff.
8. **Doc-truth test (remedy #3)** — ~20 lines, closes six of the top-ten misleading docs permanently.

*B1 is the biggest item in this census and is deliberately **not** on this list: it is already armed in
`graph_kill_rig/arms.py` and belongs to the concurrent measurement workflow, not to an afternoon.*
