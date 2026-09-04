# What Is Engram's Graph For?

**Decision document. 2026-07-24.** Synthesis of five independent evidence lenses, three
competing theses, and the graph density census
(`docs/product/experiments/GRAPH_DENSITY_CENSUS.md`), with every load-bearing claim
re-verified against the running shell and the source tree.

**Verification window:** 2026-07-24 20:55–21:25 UTC, read-only `GET` against
`127.0.0.1:8100` (shell PID 43152, `startedAt` 2026-07-24T21:08:40Z) plus direct reads of
`server/engram/`. **Line numbers in `server/engram/retrieval/` drift** — a concurrent
workflow was editing `pipeline.py`, `reranker.py`, and `config.py` during this window
(`pipeline.py` was 2234 lines when I read it; lens digests cite a longer version). Every
citation below was re-located and re-read at the stated line, not copied forward.

---

## 1. The verdict

**We still do not know what Engram's graph is worth, and the reason is not the extraction
outage.** The founding thesis — "one brain per person, projects are entity clusters
connected by topology" (`git show 6b4eeaf:README.md`, `docs/vision/founder-letter.md:16-40`)
— has never been evaluated by any rig in this repo, not once, in five months. The narrower
retrieval thesis has been evaluated five times and every one of those runs measured a system
with a broken half. But the strong form of the hypothesis this workflow was handed — *we
have never run a working producer and a working consumer at the same time* — is **false**,
and its falsification is the most decision-relevant fact in this document: on 2026-07-17 both
halves ran together, on a planted corpus, and **the graph won** — bridge reachability@5 went
2/36 → 22/36 for +2.9 ms (`docs/product/experiments/M3_1_oracle_surface.md:12-14,45-49`), with
the same experiment independently proving the edges were load-bearing (when predicates were
silently dropped and the brain ended with 0 relationships, "arm B showed zero lift",
`M3_1_oracle_surface.md:73-78`). That result was recommended for production
(`M3_1_oracle_surface.md:116`), was gated on "a real-corpus eval"
(`DECISIONS_2026-07-17.md:171`), and the real-corpus eval has never been run. Seven days
later its knob is still `default="results"` (`config.py:630-631`) with **zero profile
setters anywhere in `server/engram/`**. So the honest verdict is not "unproven for lack of an
extractor." It is: *the one graph mechanism that has ever measured positive in this codebase
was parked behind an experiment nobody scheduled, while the graph's other nine mechanisms
were left switched on and silently discarded.* Underneath that sits a fact bigger than any
flag: the architecture the entire 2026 literature converged on — graph as a **reranker over
text units** — is not disabled in Engram, it is **unbuilt**. Episode, cue, and chunk results
are constructed with `spreading=0.0, edge_proximity=0.0` as literal constants
(`pipeline.py:842-843, 908-909, 992-993`) and scored as pure semantic similarity; the only
function that applies `weight_edge_proximity`/`weight_spreading` (`scorer.py:49`
`score_candidates`) has exactly one production call site (`pipeline.py:1750`), on the entity
channel, whose output `passage_first_entity_budget=0` then truncates with `scored[:0]`
(`pipeline.py:2177-2181`). **No graph signal can reorder an answer-bearing episode in this
system, and none ever has.** The next move is therefore not "restore Ollama" and not "retire
the tier." It is to run the one pre-registered experiment that was already owed, with a kill
arm attached that nobody has ever included.

---

## 2. What we measured vs what we inferred

### 2.1 Measured — verified in this session

| # | Claim | Value | How verified |
|---|---|---|---|
| M1 | Live brain counts | 9323 episodes / 840 entities / **1637** relationships / 9080 cues | `GET /api/storage` at 21:09 UTC. Agrees with census (1637) and two independent BFS crawls (Lens 3, Lens 5) that closed to the exact authoritative edge count. |
| M2 | Zero graph structure reaches a recall answer | **0 bytes** of relationship JSON across 55 items (34 episode / 9 cue_episode / 12 entity) on 10 queries | My own `GET /api/knowledge/recall` probes, `scratchpad/probe.py`. Third independent reproduction (Lens 1: 80/80; retirement thesis: 55/55). |
| M3 | Graph scoring channels contribute nothing at runtime | `edgeProximity`, `spreading`, `activation` all **0.0 on 55/55** items | Same probe, reading `scoreBreakdown` per item. |
| M4 | No graph signal *can* reorder an episode | Architectural, not a flag | `pipeline.py:842-843, 908-909, 992-993` build episode/cue/chunk `ScoredResult`s with `edge_proximity=0.0, spreading=0.0` literals; `ep_score = original_weight_semantic * sem_sim * _ep_score_weight`. `grep -rn score_candidates` → 2 production call sites total (`benchmark/methods.py:572`, `pipeline.py:1750`); `scorer.py:28` defaults `result_type="entity"`. |
| M5 | The choke point is one integer | `passage_first_entity_budget=0` → `entity_results = scored[:0]` | `config.py:717-728` (default 0, self-documented "0 = no entities in top-k"); `pipeline.py:2177-2181`. Only `integration_profile='rework'` sets 3 (`config.py:3208-3209`); live `integration_profile='off'`. |
| M6 | The graduated design exists and is inert *because* of M5 | `passage_first_channel_separated` appends `entity_results[:remaining]` — of an empty list | `config.py:742-753`, `pipeline.py:2186-2196`. Two knobs encode one question; neither can answer it alone. |
| M7 | Spreading activation completes on **0 of 15** live recalls | timed out at ~74.5 ms on 13/15, not reached on 2/15 | My own n=15 probe, `scratchpad/lat2.py`, reading `stageTimingsMs.recallSpread` vs `recallSpreadTimeout`. `retrieval_spread_timeout_ms=75`. Lens 1 measured 1/20 completions; mine is 0/15. Same conclusion, tighter. |
| M8 | Structural-embedding weight has no data source | `recallGraphStructuralEmptySource` on **8/8** probes reaching step 4.8 | Same probe set. `weight_graph_structural=0.1`, `graph_embedding_node2vec_enabled=False` in the live-faithful config (see M13). |
| M9 | `entity_episode_traversal_source` is `"results"`, set by no profile | default `"results"`; `grep -rn` over `server/engram/` returns only *readers* | `config.py:630-641` (which documents the trap verbatim: "which the default entity budget of 0 leaves empty"); `episode_traversal.py:39-46`; `service.py:121`. `recallEntityEpisodeTraversal` fired on 7/10 probes as a no-op loop. |
| M10 | `recall(lookup_kind='facts')` returns nothing at defaults | `GET /api/knowledge/facts?q=Engram` → `{"items":[]}` | Live. With `include_epistemic=true` the *only* fact returned for that query is `Engram DECIDED "Cold Decision hit requires healthy search index"` — the continuity gate's own fixture. Filter: `lookup.py:14-20` (5 predicates) + `lookup.py:536-543` (either endpoint typed `Decision`/`Artifact`). |
| M11 | `get_context` delivers zero edges | `entityCount=3, factCount=3`, `relationship_ids: []` on **3/3** packets | Live `GET /api/knowledge/context?project_path=…`. The three "fact packets" today are 1 continuity fixture + 2 prose fragments from a commit-message episode. |
| M12 | An LLM extractor has **never** run on this machine | **1894 of 1894** provider-resolution log lines say `Narrow` | `grep -h "Extraction provider" ~/.engram/logs/*.log \| uniq -c`. Decisive because `factory.py:52` (Anthropic) and `factory.py:71-75` (Ollama) log INFO on *success* — a model resolution would be visible. 255 `falling back to narrow` warnings in `engram.log`; the currently-running PID emitted one at 14:08:40 local. |
| M13 | The effective config is **cwd- and launcher-dependent** | Two careful auditors get different answers | Instantiating `EngramConfig()` from `server/` gives `consolidation_profile='standard'`, `node2vec=True`, `cross_domain_penalty=True`. Reproducing the launchd invocation (`set -a; source ~/.engram/.env; set +a; exec …`, `WorkingDirectory=server/`) gives `quiet` / `False` / `False` — which matches `GET /api/knowledge/runtime` exactly. `DEFAULT_ENV_FILES` at `config.py:15-19` reads `~/.engram/.env`, repo `.env`, then **cwd `.env`**; the plist sources the first as real env vars, so it wins, while `server/.env` supplies anything the first omits. |
| M14 | Producer misconfig: right idea, dead address, absent model | `extraction_provider='ollama'`, `ollama_base_url='<removed: external ollama host>'` (curl exit code 000, unreachable), `ollama_model='gemma4-e4b-nothink:32k'` | Live-faithful config (M13) + `server/.env:34-36`. A working Ollama is on `127.0.0.1:11434` with 4 models — `qwen3.5-ornith-m3`, `qwen3.5-ornith`, `ornith`, `hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:Q4_K_M`. **`gemma4-e4b-nothink:32k` is not among them.** `factory.py:66-79` probes only `/api/tags` and never validates the model, so fixing the URL alone converts a loud failure into a silent one. |
| M15 | The supersession producer asserts anti-knowledge | slot key = `(subject, canonical_predicate)`, no temporal ordering | `decision_materializer.py:155-164` selects `existing` by subject+predicate; `:204-211` creates `SUPERSEDED_BY` for **every** existing decision whose `decision_object` differs. Lens 5's crawl: 260 `SUPERSEDED_BY` over 58 Decision nodes, 118 (45.4%) mutual pairs, 465 directed 3-cycles, `validTo` null on 260/260. Not a DAG; it asserts both directions. |
| M16 | The battery cannot detect graph value, by construction | every token of one group must land inside **one** top-3 result | `battery.py:98-100` (`group_contained` = all-tokens), `:111-114` (`any(... for text in result_texts)`), `:85-95` (`list(rows)[:3]`). All 10 questions in `tests/rigs/agent_experience_battery.json` are single-episode-servable; largest group is 3 tokens. |
| M17 | The continuity gate writes its own answer into the live brain | `decision_name="Cold Decision hit requires healthy search index"`, `promote_if_missing: bool = True` | `continuity.py:296,298`, write at `:477`. **Fairness correction:** `--organic` forces it off (`continuity.py:363`) and the docstring says so plainly (`:305-311`). The default invocation does not. The 14 live `DECIDED` edges are its residue. |
| M18 | Graph data *is* present and reachable — the emptiness is plumbing | `GET /api/entities/ent_8eae92a955cf/neighbors?depth=1` → 173 nodes / 279 edges with full predicate/weight/validFrom detail | Live. Also confirms the "directory listing" character: the Engram Project hub has degree 279. |
| M19 | The MCP resource surface is **not** filtered | `apply_mcp_surface` calls only `mcp.remove_tool(...)` | `surface.py:106-148`. `@mcp.resource("engram://entity/{entity_id}/neighbors")` at `server.py:2064-2072` therefore stays reachable at `ENGRAM_MCP_SURFACE=public`, returning real 1-hop structure. It is the best-formed graph consumer in the codebase and nothing instruments it. |
| M20 | Atlas has no agent-facing consumer | `grep -c atlas server/engram/mcp/server.py` = **0** | Live grep. Atlas is REST/dashboard only. |
| M21 | The harness-as-extractor producer *is* working | 128 client-proposal commits, 27 `remember_with_proposals`, 3 `predicate_not_allowed_rejects` | `~/.engram/harness-metrics.json`; counters verified live-wired at `projection_execution.py:263-274`, `apply.py:768-771`. Window opened 2026-07-10 (`ed03b99`), so ~14 days. `ALLOWED_CLIENT_PREDICATES` (`promotion.py:37-71`) is 31 predicates including `SUPERSEDES`, `CONTRADICTS`, `PREFERS`, `DECIDED`. |
| M22 | The durable-entity lane costs ~25% of recall wall time to return relationship-free stubs | 6725.6 ms of 26534.2 ms across n=15 = **25.3%** | Client-side wall clock (`scratchpad/lat2.py`) vs `durableEntityFirst + durableEntityRescue` stage timings. Those rows hardcode `"relationships": []` at `recall_surface.py:1592`. |
| M23 | The graph gate is self-defeating | one slow graph read disables every downstream graph read | `recall_graph_gate.py:20-26` (`graph_probe_timed_out`), `:38-60` (`GatedGraphStore` over 7 methods), `retrieval_skip_secondary_graph_after_probe_timeout=True` live. **This defect gets worse as the graph gets denser** — it is anti-correlated with repairing the producer. |
| M24 | A seventh inert instrument | `external_extractor_invoked: 0` is a dead counter | `record_external_extractor_invoked()` defined at `harness_metrics.py:157-160`, **zero call sites**. It appears to corroborate M12 and must not be cited for it. Its siblings *are* wired (`record_narrow_extraction`, `record_client_proposal_outcomes`, `record_remember_call`). |

### 2.2 Instruments that lie — do not cite these numbers

| Surface | Reports | Truth | Line |
|---|---|---|---|
| `/api/stats` relationships | 750 | 1637 | `storage/helix/graph.py:2387-2405` — extrapolates from `entities[:10]` scaled by `entity_count/10` |
| `/api/stats` projection yield | `relationship_count: 0`, `avg_relationships_per_projected_episode: 0.0` | unknown, never computed | `storage/helix/graph.py:2511, 2517, 2578` — literal constants inside the returned dict |
| `/api/episodes` per-episode linkage | `entities: []`, `factsCount: 0` | unknown | `retrieval/graph_state.py:1050-1051` — literal constants |
| `/api/graph/atlas` edge count | 756 | 1637 | Same 10-entity extrapolation, laundered through `atlas/builder.py:51-53`. 756 = 9 × 840 / 10 exactly. It scales with entity count so it will always look plausible. |
| `harness-metrics.json` | `external_extractor_invoked: 0` | uninformative | `harness_metrics.py:157` — dead counter (M24) |
| `/api/entities/search?type=Decision&limit=100` | 24 items, `status: ok` | 83 Decisions exist | Silent 71% truncation, mechanism unresolved (Lens 5) |
| `/api/knowledge/runtime` trigger rates | `graph_lift_rate: 0.0` | unknown lifetime | Genuinely computed (`retrieval/control.py:207-247`) but from an **in-process** deque. Zero means "not this process," not "never." |

**Seven inert or fabricating instruments, in one day.** Every prior graph verdict in this
project was read off at least one of them.

### 2.3 Inferred, not measured — flagged as such

- **MCP `recall` `related_facts` is empty.** `presenter.py:452-465` builds it from
  `item["relationships"]`, the same field the REST twin serializes; that field measured 0
  bytes on 12/12 entity items (M2) and is hardcoded `[]` in the rescue lane
  (`recall_surface.py:1592`). Code-verified shared source, but **no MCP stdio process was
  launched** (CLAUDE.md forbids multi-opening the native graph; guardrails forbid touching
  the shell). Confidence: high, but it is an inference.
- **Producer attribution of the 1637 edges** (Lens 3: 1303 materializer / 188 consolidation /
  123 deleted-phase residue / 14 test harness / 9 extractor / **0 LLM**). Per-edge provenance
  is structurally impossible — `models/relationship.py` has no producer field — so this is
  predicate→unique-writer attribution. Sound where a predicate has one writer; ambiguous for
  ~8 of 1637 edges. The conclusion is insensitive to how those 8 resolve.
- **"No graph refutation ever ran with producer, consumer, and instrument simultaneously
  sound."** Tabulated from five primary sources (Lens 2). It holds — see §3 — but it is a
  tabulation, not a measurement.

### 2.4 Disagreements between lenses — named, not smoothed

1. **Live config values.** Lens 1 read `cross_domain_penalty_enabled=False`,
   `node2vec=False`; my first instantiation read `True`, `True`. **Lens 1 is right.** The
   resolution is M13: the effective config is a function of *how the process was launched*.
   That is the same defect that caused the 65-day extraction outage, and it means "the live
   config" is not a well-defined object without specifying the launcher.
2. **Durable-lane latency share.** Retirement thesis: 38.9%. Mine: **25.3%**. Both are
   defensible measurements of different denominators (summed stage keys vs client wall
   clock). I report mine because the wall clock is the number the agent actually pays. Note
   my first attempt produced 53% from a bad denominator and I discarded it — the stage-timing
   surface has no total-wall-clock key, so any share computed from it is a guess.
3. **The census's atlas corroboration does not reproduce.** Census cited "1334 intra-region +
   303 bridge = 1637"; Lens 5 measured 1276 + 452 = 1728. The 1637 headline still stands on
   `/api/storage` and two independent crawls — but one of its three cited instruments was
   the fabricating one (§2.2).
4. **The census's re-open criterion is 30× away, not 2×.** Its "≥50% of entities have a
   non-structural incident edge (today: 22.3%)" counts `MENTIONED_WITH` as non-structural.
   `MENTIONED_WITH` is written by consolidation PMI (`consolidation/phases/infer.py:236`),
   not by extraction. On the defensible reading the figure is **1.7%** (Lens 5).
5. **The regex confound does not apply to the June experiments.** The brief frames the
   2026-06-04 refutations as possibly measuring regex output. They did not: the answerability
   A/B ran on an ollama-extracted graph (69 entities / 85 edges), and the LongMemEval
   re-baseline ran a clean-LLM-extract arm in which graph-ON *also* lost (55.6% → 51.4%). The
   regex confound is specific to **the live dogfood brain the census measured**.
6. **Semantic edge count: 10 or 9?** The census says 10; Lens 3 says 9, because
   `IMPLEMENTED_BY` is written by `decision_materializer.py:53`, not by any extractor. Lens 3
   is right. Immaterial to any conclusion.
7. **"Is 5% real a verdict or an outage report?"** Lens 3's unprompted correction stands and
   is important: **both.** The semantic lane is a pure outage (9 edges, zero model-produced).
   But 91.1% of the graph came from producers that were never broken for a moment — the repo
   scanner, the decision materializer, consolidation infer, dream. They produced exactly what
   they were built to produce: a filesystem index and a document-supersession chain. *Sizing
   the outage does not license the inference that fixing it fixes the graph.*

---

## 3. The confound, adjudicated

**Question: has a working producer and a working consumer ever run simultaneously?**

**On organic data: NO. Never. Not once in five months.** Five refutations are on record and
in zero of them were producer, consumer, and instrument all sound:

| Run | Producer | Consumer | Instrument | Result |
|---|---|---|---|---|
| R1 LongMemEval re-baseline (2026-05-29) | OK (narrow **and** clean-LLM arms) | displacing blend | oracle variant, **zero distractors** — structurally blind to retrieval benefit, sees only displacement cost | ON 58.5% < OFF 62.7% |
| R2 graph_thesis (`3326bce`) | **VERIFIED** (aborts on narrow fallback; gate confirms 17/18 bridge→answer links present) | single ranked top-k, entity budget 3 | — | 14/18 → 10/18. Author's own attribution: "entity results DISPLACE the answer-bearing episodes" |
| R3 answerability A/B (2026-06-04) | OK (ollama) | ranker never surfaced 1-hop neighbours; **merge never ran** so the graph was fragmented | 26B judge below the accuracy floor | NULL; ≥half the misses were *connected-but-not-surfaced* |
| R4 edge-triples-in-evidence | OK | — | — | net-NEGATIVE. Producer output was semantically empty (ingest timestamps + generic predicates) |
| R5 channel-separation 150Q | OK | — | oracle, zero distractors | OFF 53.3% vs blend 52.0% |
| Census (2026-07-24) | **absent** (regex; 1894/1894 Narrow) | default-off (budget 0, traversal `results`) | fabricating (§2.2) | "the graph is 5% real" |

The broken half was usually the **consumer**, not the producer. That inverts the brief's
implicit framing.

**On synthetic data: YES — ONCE, and the graph won.** M3.1 (2026-07-17) ran a working
producer (harness-as-extractor via client proposals, `model_tier="opus"`, verbatim
`source_span`s — `M3_1_oracle_surface.md:26-29`) against a working consumer
(`entity_episode_traversal_source="candidates"`), on the same brain, with only the read path
varying. Bridge reachability@5: **2/36 → 22/36**, +2.9 ms mean latency, 21 of 23 hits via the
traversal channel. Its incidental finding #1 is the strongest single piece of pro-graph
evidence in the repository and nearly got argued away: when free-form predicates were
silently dropped and the brain ended with **0 relationships**, "arm B showed zero lift." *The
edge does the hop; membership does the fetch; both are load-bearing.*

### What the census does and does not license

**Licensed:**
- The live dogfood brain contains a repository index and a document-supersession chain, not a
  knowledge graph. 1637 edges, ~9 of them extracted semantic relations, 6 of those with
  sentence-fragment or `:decision_statement:` squatter endpoints.
- Running a depth A/B **on this corpus** would produce a sixth uninterpretable result.
- The graph is not growing organically: 814 (May) → 495 (Jun) → 328 (Jul), of which 4
  semantic; both bulk producers are off at runtime.

**Not licensed:**
- *Any* statement about the value of graph memory. The consumer returns 0 bytes on a graph of
  **any** quality (M2, M4, M5). A perfect graph would deliver identically nothing today.
- The census's four-part re-open criterion as written **can be fully satisfied while the
  experiment still measures nothing**, because all four clauses are about density and
  topology and none is about the read path. It needs a fifth clause: *a recall on the A/B
  corpus returns > 0 bytes of edge-derived content to the answerer.*
- The census's Gate ordering. It puts "restore a real extractor" first. But at least five
  consumer defects are **producer-independent** — `passage_first_entity_budget=0`,
  `entity_episode_traversal_source='results'`, the durable-context early return that shadows
  `ContextBuilder`, atlas having no MCP surface, and the graph gate (M23) which *worsens* as
  the graph densifies. Repairing the producer first and re-running the A/B would produce a
  third uninterpretable null, this time sourced in the consumer.

### The correction the workflow's own hypothesis needs

The hypothesis "we never ran both halves at once" is **true as archaeology on organic data
and false in general** (M3.1). Two consequences:

1. **It implies a symmetric repair, and the symmetry is false.** The consumer is the
   harder-broken half and the cheaper one to fix. One config value with a measured +11×
   result is the entire distance between "0 bytes reach the agent" and "graph structure
   reaches the agent."
2. **The producer that works is not the one the census wants to repair.** M3.1's producer was
   the *harness agent* (client proposals), which is shipped, is the north star's preferred
   rung, and committed 128 facts in 14 days (M21) — versus 9 regex edges in 65 days.
   "Restore Ollama" is an untested assumption about where the leverage is.

There is one place the never-both-halves defence is **weakest**, and it should stay visible:
on knowledge-update, where `extraction/apply.py:389/417/456` genuinely does invalidate on the
live write path, graph-ON measured **−25pt**; and Zep's own vendor-authored paper (arXiv
2501.13956) reports knowledge-update as its weakest category (76.9% → 74.4% on gpt-4o-mini, a
loss). That is the first place I would try to falsify this entire framing.

---

## 4. The design space

Not a binary. Seven positions, ordered from least to most committed. Each is stated as *what
it costs / what it buys / what would have to be true*.

### A. Retire the tier
**Cost:** 2–3 weeks of careful deletion; the risk is in the seams, not the deleted code —
`pipeline.py` carries 8 `_skip_secondary_graph_after_probe_timeout` references and wraps the
store in `GatedGraphStore`, and unpicking that touches the hottest path in the product. A
migration for 1637 edges and 840 entities. Irreversibly lost: the bitemporal edge model and
any near-term path to unnamed-bridge multi-hop.
**Buys:** **10,488 LOC** measured by `wc -l` (4974 in graph subsystems — `embeddings/graph/`,
`atlas/`, `atlas_surface.py`, `helix/atlas.py`, `activation/{spreading,bfs,ppr,community}.py`,
`graph_probe.py`, `graph_state.py`, `recall_graph_gate.py`; plus 5514 in the 8 graph-only
consolidation phases). Also 8 of 15 phases, ~13–24% of the 375 test files (**pattern-dependent
— my conservative grep gives 49, the retirement thesis's broader one gives 88; neither is
wrong, and the spread is itself worth noting**), 69 config fields, 25.3% of recall wall time
(M22), and the ongoing burden of every future engineer learning 15 phases and a 1601-line
HelixQL schema (`storage/helix/schema.hx`, verified) before touching recall.
**Would have to be true:** that M3.1's win does not survive organic extraction. **Nobody has
measured that.** Retiring now discards a measured positive on the strength of an unrun
experiment — which is the mirror image of the mistake the project keeps making.
**Note the withdrawn plank:** the disk/RAM argument is fabricated. 1637 edges against 9317
episode vectors in one 7.6 GB LMDB is rounding error, and no per-subsystem attribution exists
anywhere. The retirement thesis withdrew it itself; I concur.

### B. Freeze in place
**Cost:** near zero. Both producers are already off (`artifactBootstrapEnabled: false`,
`decisionGraphEnabled: false`, live). Set the remaining graph read flags to their off
position and ship one release.
**Buys:** a falsifiable prediction — *nothing regresses*. If something does, retirement is
wrong and we learn it for the price of one release.
**Would have to be true:** nothing. This is the honest default while the deciding experiment
runs, and it is compatible with every other option.

### C. Rebuild the producer (the census's Gate 1)
**Cost:** two config values, plus a positive verification probe. Ollama is live on
`127.0.0.1:11434`; the config points at a dead Tailscale IP; **and the configured model is
not installed** (M14). Fix URL *and* model together or you ship a new silent-inert failure —
`factory.py:66-79` validates neither. Then: unknown wall-clock cost to re-project 4248
never-extracted episodes on a 16 GB machine with a documented Jetsam kill. Run on a clone.
**Buys:** a semantic lane that has never existed.
**Would have to be true:** (i) that a zero-shot 9B Ornith build produces usable extraction
JSON from Claude Code session transcripts — untested, and Ornith is reasoning-tuned, the model
class most likely to emit chain-of-thought before the JSON; (ii) that the published ~7B
extraction floor holds for this input distribution; (iii) that growing the symbolic space
helps rather than hurts — LatentGraphMem (arXiv 2601.03417) measures the opposite, that
retrieval becomes *more* query-sensitive as the symbolic space grows.
**And:** it fixes nothing the consumer needs. Producer repair with the consumer as-is
delivers 0 bytes to the agent, identically.

### D. Rebuild the consumer — the surfacing port only (the M3.1 recommendation)
**Cost:** one config field flip, +2.9 ms measured. Plus the prerequisite: the 75 ms spread
budget and the graph gate (M23) must be fixed first, or the mechanism cannot fire —
spreading completes on 0/15 recalls today.
**Buys:** the only mechanism in this repo with a positive measured result. Additive by
construction: `append_entity_linked_episodes` appends synthetic rows post-assembly and can
never evict an answer episode.
**Would have to be true:** that the +11× survives organic extraction and a non-planted graph.
**The real bill, which nobody has priced:** at defaults the traversal fires from the top
`entity_episode_max_entities=5` candidates pulling `entity_episode_max_per_entity=10` episodes
each — up to 50 appended rows with no final truncation. M3.1 measured mean returned rows
5 → 14.8 on a clean 102-episode corpus. On the live brain, where the top entity candidates
*are* the three Project hubs holding 50.9% of all edges, traversal returns "the 10 newest
episodes mentioning Engram" — a recency dump. **The knob `results|candidates` freezes the
wrong question. The question is not whether to traverse but *from what*.** The graduated
answer is a selectivity gate — an IDF-analogue on per-entity episode fan-out — which requires
a statistic no surface exposes today.

### E. Rebuild the consumer — the reranker port (what the outside view says)
**Cost:** real engineering. This is **unbuilt**, not disabled (M4). Episodes would have to
carry graph-derived signal into their score, which means either routing episode candidates
through `score_candidates` or adding a graph term to the episode lane.
**Buys:** the architecture three independent 2026 systems converged on — HippoRAG 2 (PPR over
passages, the one graph method that does not regress on simple QA), AtomMem (random-walk over
atomic facts; its ablation moves answer quality 4–7 J points while moving Recall@10 only 1–2 —
*the graph re-orders evidence the vector lane already had*), and Mem0 v3, which deleted ~4000
lines of external graph traversal in April 2026 and replaced it with entity linking that only
boosts ranking.
**Would have to be true:** that D wins first. E is strictly more expensive than D and shares
its precondition. Do not build E before D reports.

### F. Change what the graph is FOR — currency, not connectivity
Keep one predicate, one hop, rendered as a **stamp on a row already being returned**
(`status: current | superseded_by <id> @ <date>`, ~8–15 tokens, displacing nothing) rather
than as evidence or as ranking.
**Cost:** the slot key is genuine research, not engineering. `Relationship` has `valid_to` but
no `superseded_by`, and `apply.py:431-434` states the superseding link is deliberately
discarded at write time — so the system can say "this died" but never "this was replaced by
that." Persisting it means a model change + three stores + a `schema.hx` migration + a
decision about 1637 existing edges, 118 of which are demonstrably false (M15).
**Buys:** the one job where the discriminating information is provably absent from the text of
either endpoint — vector-proof by construction. And a MISS is self-correcting while a STALE
HIT is silent.
**Would have to be true:** (i) that stale-only retrievals actually happen at a meaningful
rate — unmeasured, and it is the entire thesis; (ii) that the slot key can be written at all —
today's only implementation collapses to one slot per project (M15); (iii) that the
knowledge-update −25pt result was a displacement artifact rather than a verdict on
supersession — which is a *reinterpretation of someone else's null*, precisely the move this
project has made five times and been wrong about.
**Ongoing cost, and it is the worst in the whole design space:** a FALSE supersession is
strictly worse than none. A miss sends the agent to read the code; a wrong "SUPERSEDED" stamp
makes it silently discard a correct memory and never notice. The live system already
demonstrates this at scale — 260 edges asserting supersessions that did not happen. This
needs a precision gate forever, plus an instrument that can detect false stamps, which does
not exist.
**Also note:** `SUPERSEDES` is in `ALLOWED_CLIENT_PREDICATES` while `SUPERSEDED_BY` is in the
epistemic filter set. The harness can propose supersession edges that the fact surface then
hides. Neither half was designed against the other.

### G. The narrow surviving use — router, not answer source
The graph's only job is to turn a query naming **A** into episodes about **B**, where A→B is a
*contingent, private* fact the user asserted — not a distributional similarity. This is the
one genuinely structural argument: BM25 needs a shared token (by construction there is none);
dense vectors need A and B near in a space learned from a general pretraining corpus, and
"the auth flake is the CI clock-skew thing" appears in no pretraining distribution. An edge is
the only object in the system that stores a private asserted association as a retrievable
thing.
**Where it collapses, stated plainly:** the linking sentence is *itself an episode*. A vector
search retrieves it, the agent reads "clock skew," and issues a second recall. So the
structural claim reduces to: hybrid cannot do this in **one pass**, and cannot do it **at all**
only in the residual where the linking episode itself fails to rank. **Nobody has measured
that residual.** Every prior Engram A/B compared graph vs *single-pass* vector and never vs
*iterated* vector — the graph has never been tested against its actual competitor.

**Positions to reject outright.** Putting edge triples or entity summaries into the evidence
block. Four independent results agree and it should stop being re-litigated: Engram's own
adversarially-verified 2026-06-04 finding; `3326bce`'s 14/18 → 10/18 with 17/18 bridges
verified present; AtomMem's measured degradation past k=20 while recall keeps rising; and
HippoRAG 2's summary-in-evidence collapse on simple QA (NQ 46.9–50.7 vs 61.9 dense baseline).
**The graph spends ranking or annotation, never context.**

---

## 5. The deciding experiment

**One experiment. It is the one the repo already owes itself
(`DECISIONS_2026-07-17.md:171`: "real-corpus eval"), with a kill arm attached that no Engram
A/B has ever included.**

### Why this one
It is the only experiment that (i) reuses an existing rig, (ii) crosses both axes — producer
quality × consumer surfacing — which **no published study has ever done** (Lens 4 searched
eight papers; every one varies a single axis), (iii) contains an arm that can kill the graph
outright, and (iv) has a known reference point (2/36 baseline, 22/36 planted ceiling) so the
organic result is interpretable on a scale that already exists.

### Setup
Reuse the M3.1 rig verbatim: lite brain (SQLite + FTS5 + local FastEmbed, fully local, zero
external keys), `reachability@K` metric, no LLM judge, `record_access=False`, per-arm fresh
manager over the same DB. `run_experiment.py --reuse`. **On a clone. Never the live brain.**

Corpus: extend M3.1's 36 bridge questions to **N ≈ 60**, and re-ingest through a *repaired
extractor* so the graph is organic rather than planted. Keep the original 36 planted-graph
questions as a fixed control so the organic result is anchored to a known 2/36 → 22/36.

Bridge construction (unchanged): `ep1` co-mentions person **A** and topic **B** with an edge
`A→B`; `ep2` (gold) states a concrete fact about **B** and never mentions A; the query names
only A. Verify bridge structure present **before** scoring, per-question.

### Pre-flight — the run is VOID if any of these fails
1. **Producer positive probe** (the silent-inert lesson): project 20 known-content episodes
   and assert **≥ 1 committed semantic relationship**. Never "no error in the log."
   *If Ollama is the producer: fix `ollama_base_url` AND `ollama_model` together (M14).*
2. **Consumer byte probe** (the census's missing fifth clause): a recall on the A/B corpus
   returns **> 0 bytes** of edge-derived content to the answerer. Today the answer is 0 bytes
   on 55/55 items (M2). Until this passes, the experiment measures the consumer, not the graph.
3. **Gate probe:** spreading activation must complete on ≥ 80% of A/B-corpus recalls. Today it
   completes on **0/15** on the live brain (M7). If the 75 ms budget binds on the A/B corpus
   too, raise it *for the experiment only* and record the value — do not ship it.
4. **Residual measurement (cheap, run first, reported separately):** on the A/B corpus,
   measure the fraction of questions where the *linking episode itself* fails to rank in arm
   A's top-10. This is the graph's actual addressable market.
5. **Packet-cache isolation** (added 2026-07-24, task #18; `INSTRUMENT_AUDIT.md` AUDIT-14).
   The recall packet cache is keyed `group_id:scope:digest(query):digest(project_path)`
   (`retrieval/packet_cache.py` `build_key`) — **no build, config or arm component** — with a
   300 s TTL and SQLite persistence that survives a restart. Running arm B after arm A on the
   same queries can therefore serve **arm A's cached packets to arm B**, reporting "no
   difference" for a change of any size, with a clean low-variance number. Isolate by one of:
   `recall_packet_cache_enabled=False`; a distinct `group_id` per arm; clearing the cache
   between arms; or spacing every repeat beyond the TTL. **State which was used** — a run
   that does not is VOID.

### Arms
| Arm | Description |
|---|---|
| **A** | Today's shipped default: single-pass hybrid, `passage_first_entity_budget=0`, `entity_episode_traversal_source="results"` |
| **B** | A + `entity_episode_traversal_source="candidates"` — additive entity→episode surfacing, **zero triples in evidence** |
| **C** | **The kill arm.** Same single-pass retriever as A, no graph, but one *second* recall round seeded from arm A's results — i.e. what the harness agent can already do for itself with a loop |

### Metrics
Primary: **reachability@5** of the gold episode. Secondary, all mandatory: mean rows returned
to the answerer, total tokens handed to the answerer, p50 added latency.

### Pre-registered thresholds — written before any result is seen

**SUCCESS** (the graph earns a shipped default; flip `entity_episode_traversal_source`):
- `B − A ≥ +6/N` on reachability@5 (≥ 2× the ±3 noise floor implied by `969b00d`'s
  `depth_flips=2` per slice from HNSW build nondeterminism), **AND**
- `B − C ≥ +3/N` — the graph must beat the agent's own second query, **AND**
- p50 added latency ≤ +10 ms, **AND**
- mean rows returned ≤ 20 (AtomMem's measured degradation threshold; past it, added recall
  costs answer quality).

**KILL** (any one is sufficient; the graph's surfacing thesis is dead and §4-A becomes the
recommendation):
- **K1 — the round-trip test.** `C ≥ B` on reachability@5. The graph is buying a round-trip,
  not a capability. The correct build is a cheap iterable recall plus a prompt that tells the
  agent to search twice. *No prior Engram A/B has ever run this arm, which is why this has
  never been visible.*
- **K2 — the noise test.** `B − A ≤ +3/N`. Within HNSW jitter.
- **K3 — the context test.** Mean rows returned > 20 with no reachability gain that survives
  the token cost. The additive port is net-negative too, and the graph dies at the port level.
- **K4 — the market test.** The pre-flight residual (#4 above) is < 10% of the question set.
  The thesis is then true but commercially irrelevant: the linking episode almost always ranks
  on its own, and the graph is a latency optimization dressed as a capability.

**AMBIGUOUS** (`+3 < B−A < +6`, or B wins but C is within 3): re-run once with a different
HNSW seed. If still ambiguous, that **is** the answer — the effect is smaller than this
project's measurement noise, and per §4-B the correct move is freeze, not build.

### What this experiment explicitly does not settle
It scores a retrieval list. The north star says the real consumer is the harness agent, and
**no rig anywhere in this repo scores an agent's task outcome** (P5, open, no rig). Even a
clean win here is a win on a proxy that has never been validated against the actual consumer.
**Do not let it flip a default without an agent-task arm.** Say so in the results doc.

### The rig (built 2026-07-24; first run as the experiment 2026-09-04 — see the result below)

`server/engram/evaluation/graph_kill_rig/` — arms A/B/C, the VOID pre-flight, and the
thresholds above **encoded as a pure function** (`thresholds.evaluate`) whose truth table is
pinned by `server/tests/test_graph_kill_rig.py`. They cannot be renegotiated after a result is
on the table, which is the point.

```bash
cd server && uv run python -m engram.evaluation.graph_kill_rig --scratch <dir> --n 60
uv run python -m engram.evaluation.graph_kill_rig --scratch <dir> --fault drop-relationships  # prove it refuses
```

Six checks gate the run, not four: the thesis's producer / consumer-byte / spread / residual,
plus **vector-index coverage** (the rig's own first run measured a keyword-only system —
`FastEmbedProvider.dimension()` returns 768 on a provider whose ONNX never loaded, and
`FASTEMBED_CACHE_PATH` is a plain env var the launchd unit exports and a CLI run does not:
M13 again) and a **scored-set floor** at N=36 (absolute thresholds are gameable by shrinking N).
On any failure it emits `status: VOID` with reachability and the verdict **withheld**, exit 2.

Three findings from building it, all pre-experiment:

1. **`--fault starve-spread` measured 72,240 chars of traversal rows reaching the answerer
   while ZERO of them were edge-derived.** That is M3.1 incidental finding #1 reproduced on
   demand: a loose byte probe passes on a graph that contributed nothing. The consumer byte
   probe therefore counts only rows carrying a non-zero `spreading` bonus or literal
   relationship JSON — `activation/bfs.py:162` writes `bonuses` for neighbours only (seeds get
   `hop_distances[seed]=0` at `:53-55` and no bonus), so `spreading > 0` is exactly "an edge
   was walked". Note this signal is also launcher-proof: `recall_spread_reached` and
   `recall_spread_injected` both existed in `pipeline.py` while the rig was written and neither
   is emitted by the tree it now runs against.
2. **K is still not the binding constraint on traversal.** On a 210-episode corpus with 55
   committed `WORKS_ON` edges and 51/60 bridges verified in the store, arm B appended ~7 rows
   per query, carried real spreading signal, and reached the gold episode 0/51 times at both
   `entity_episode_max_entities=5` and `=20`. That is M3.1's own residual diagnosis —
   spread-backfilled candidates carry raw cosine and lose to lexically-similar distractors, so
   the topic entity never becomes a traversal parent. **The pool-scoring fix M3.1 filed as a
   follow-up was never landed, and the deciding experiment will measure its absence.**
3. **A bridge corpus can auto-fire K4.** The residual measured **0.0%** — arm A ranked the
   linking episode in its top-10 on 51/51 questions. K4 kills on residual < 10%, so corpus
   construction alone can decide the verdict. Whoever runs this must either build genuine
   distractor pressure on A or report that the market test fired for corpus reasons. This is a
   property of the question set, not of the graph.

The rig scores with lane 1's recall meter (`evaluation/meter.py`) as a second reading —
`--scorer multi_source_cover`, `max_sources=2`, every question carrying a token group that
spans the link and gold episodes — beside the id-based default. Never `engram battery`.

### Run 2026-09-04 — RESULT: **KILL** (planted control); organic run VOID on the producer

First run as the experiment, after one rig defect was fixed (`bc85c64`: the rig closed the
ingest brain before the capture service's background index lane had run, so 36/60 gold episodes
had no vector and the run VOIDed on its own race). Envelopes:
`docs/product/artifacts/graph_kill_rig_2026-09-04_{proposals,narrow}.json`.

**`--producer proposals` (M3.1's planted control) — every pre-flight check passed** (51/60 bridges
verified, 60/60 gold vectored, 78 360 edge-derived chars reached the answerer on 51/51 questions,
spread completion 102/102, residual 3.9 %), so the numbers are licensed:

| arm | reach@5 | reach@10 | mean rows | mean chars | p50 ms |
|---|---|---|---|---|---|
| A — no graph | 0/51 | 0/51 | 6.0 | 1 813 | 87 |
| B — graph consumer | 0/51 | 0/51 | 13.7 | 3 349 | 90 |
| C — agent's own second query | 1/51 | 1/51 | 8.2 | 2 377 | 215 |

Verdict against the pre-registered thresholds: **K1** (C = 1 ≥ B = 0), **K2** (B − A = 0, inside
the ±3 jitter), **K4** (residual 3.9 % < 10 %). S1 `B − A ≥ +6` false; S2 `B − C ≥ +3` false.
The graph arm walked an edge on every recall and paid +7.7 rows / +1.5 k chars per query for it,
and reached the gold episode exactly as often as no graph at all: **zero**. The edge is surfaced;
the episode on the far side of it is not — the "1-hop neighbours never surfaced" finding of the
2026-06-04 answerability A/B, reproduced under the pre-registered instrument.

**`--producer narrow` (organic) — VOID**: zero committed semantic relationships; narrow proposed
edges whose endpoints never committed (`missing_entities`), 60/60 gold vectored. The organic
producer cannot build a single bridge from real repository text, so the organic arm measures the
extractor, exactly as §5 says it would. That is the extraction lever (2026-06-04), unchanged.

Arm B is the pre-registered one: `entity_episode_traversal_source="candidates"`, the M3.1
surfacing port that scored 22/36 on 2026-07-17 (`arms.py` `ARM_B_OVERRIDES`). Under this
instrument — harvested text, verified bridges, the kill arm attached — **that result did not
reproduce: 0/51.** What this does not settle (unchanged from the section above): a differently
built consumer (§4 D/E reranker) could still turn the surfaced edge into the surfaced episode;
this run kills the graph with the surfacing port on, not the design space. Caveat on C: one
question in 51 is not a round-trip capability either.

### Not a valid instrument
`engram battery`. Its scoring rule requires all tokens of one group inside **one** top-3
result (M16); all 10 questions are single-episode-servable. A two-source answer scores MISS by
construction. Re-running it after any graph repair produces a number that means nothing.

**Use `engram meter` instead** (built 2026-07-24 for this experiment, task #18;
`server/engram/evaluation/meter.py`, `docs/product/experiments/RECALL_METER.md`). It scores
answers assembled from up to two rows, reports per-question hit rate and variance over N runs,
attributes each answer to the rescue lane that produced it, derives the minimum N needed to
resolve a 1-answer difference at the observed variance, and **refuses to emit a headline score
when the run set cannot support one**. It carries the battery's ten questions verbatim, so one
capture can be scored under both rules and the difference is visible rather than argued.

---

## 6. Black holes

Listed as found. Not tidied.

**About the hypothesis this workflow was built on**
- The strong form — "we never ran both halves at once" — is **false** (M3.1, §3). The surviving
  form is "never on organic data with an answer-quality metric." That is a much narrower claim
  and it should be restated wherever the original is recorded.
- The hypothesis implies a symmetric repair. The symmetry is false: at least five consumer
  defects are producer-independent. Acting on the hypothesis as written would produce a sixth
  uninterpretable null.
- **Where the hypothesis is weakest, and I would attack it here first:** knowledge-update.
  `apply.py:389/417/456` genuinely invalidates on the live write path — both halves ran — and
  the sign was −25pt, corroborated by Zep's own weakest category. Every defence of that result
  is a *reinterpretation of a null*, which is the exact move this project has made five times.

**About the founder's framing**
- "The real consumer is the harness agent" is well-argued and externally **under-evidenced**.
  The only controlled coding-agent memory study with a memory-vs-no-memory arm that Lens 4
  found is a vendor-affiliated n=9 pilot whose headline was "persistent memory does not improve
  code quality — all three conditions scored 84–96%," with gains only in tokens and turns on
  complex tasks, and no-memory winning outright on simple tasks. If that replicates, Engram's
  value proposition is **cost and turns, not correctness** — which changes what the deciding
  experiment should even measure.
- Meanwhile the live runtime reports `agentRecallCount: 0`, `turnsWithoutRecall: 6`,
  `contextLoadedThisSession: false`. **Before litigating whether the depth tier earns a slot in
  the context packet, notice that the agent is not reaching for the packet at all.** Both the
  rebuild case and the retirement case may be arguing about the wrong tier.
- "Every knob is a question we haven't answered yet" cuts against this project's own
  instruments too. The battery is not a knob but it is the same failure mode: ten questions,
  all single-hop, scored by substring containment, is a design that silently answers "what
  should memory be good at?" with "reciting a phrase from one recent episode."

**About the producer**
- **Where did the 128 client-proposal commits land?** They do not appear as new semantic
  predicates in the July mix (PART_OF 243, SUPERSEDED_BY 55, DECIDED 14, DOCUMENTED_IN 11,
  USES 2, AIMS_FOR 2, ANNOUNCED_AS 1). Either they committed mostly *entities*, or they are
  being absorbed somewhere untraced — a possible eighth inert path. **This is the highest-value
  unresolved producer question**, because it determines whether the producer needs rebuilding
  at all. `client_proposal_commits` counts committed evidence candidates of mixed fact_class
  (`projection_execution.py:249-274`), so the entity/relationship split is not readable from
  the counter.
- Whether the installed Ornith 9B produces usable extraction JSON on Claude Code transcripts:
  **untested.** `ollama list` reports `capabilities: ["completion"]` with no `tools` entry.
  Reasoning-tuned qwen3.5 derivative. Assume nothing.
- Per-episode extraction cost and throughput for a local model: **not measured, deliberately.**
  16 GB machine, ~4.5M pageouts logged, 5.6 GB models, documented Jetsam kill history, and a
  concurrent workflow owning the live shell. Hard bound only: `ollama_extractor.py:14` caps
  input at 8000 chars and the system prompt is 6091 chars → ~3.5k tokens per episode × 4248
  never-projected episodes. Wall clock UNKNOWN. Measure on a clone.
- What fraction of the 9323 episodes are `auto:bootstrap` repo-scan noise. The census claims
  51% of the 400 most recent. This determines how much of the 4248-episode backlog is worth
  re-extracting at all — re-extracting a directory listing with a 9B model buys nothing.
- Whether the Tailscale host was *ever* reachable from an ad-hoc process. The logs cover only
  the three logged processes. The 2026-06-04 "extraction is the graph lever" result (ollama: 24
  entities/23 edges vs narrow: 12 fragments/0 edges) could have run from a `python -c` that
  never touched these logs. **I can prove the server never used a model; I cannot prove that
  experiment was fake.** Both prior findings may be honest — one measured a manual run, the
  other measured production.

**About the consumer**
- **Has any agent ever read `engram://entity/{id}/neighbors`?** It is reachable on the public
  surface (M19), returns real 1-hop structure, and is the best-formed graph consumer in the
  codebase. `grep` over the 29 MB `engram.log` finds 0 hits for the URI and **no counter
  instruments resource reads at all**. I cannot distinguish "never used" from "used but
  uninstrumented." Cheapest close in the document: add a counter.
- Whether the 75 ms spread budget is the binding constraint or a symptom. 13/15 timeouts at
  exactly ~74.5 ms proves the budget binds; I could not measure how long an *unbounded* spread
  takes on this brain without editing config (forbidden). So "raise the timeout" is a guess —
  80 ms or 8000 ms, unknown.
- Whether the durable-first short-circuit helps or hurts. It costs 25.3% of recall wall time
  (M22) and returns relationship-free stubs. Rate measured; **no A/B on answer quality with vs
  without.**
- Whether `recall_rescue_drop_triple_entities=True` removes legitimate content. Given that 6 of
  9 semantic edges have `:decision_statement:` squatters as endpoints, it may be correct today
  and wrong after a producer repair. Yield never measured.
- Why `/api/entities/search?type=Decision&limit=100` returns 24 of 83 with `status: ok`.
  Shortfall confirmed, missing rows identified (the `dec_*` squatters), **mechanism unresolved**.
- Unnamed-bridge multi-hop — the class the record repeatedly names as the graph's unique win —
  **has never been tested, because the mechanism it requires was never built.**
  `pipeline.py:534-540` decomposes into parallel `sub_queries` and merges them at `:661-662` —
  all in *one* pass; `decomposer.py:1` is explicitly "LLM-free query decomposition"; there is no
  second retrieval round conditioned on round-1 results anywhere. Every "multi-hop" number in
  Engram's record is a *nameable-bridge* number — i.e. the case where episode vectors have a
  lexical shortcut.

**About the evidence base itself**
- **The strongest positive graph result has no committed artifact.** multi_hop +0.174, CI95
  [0.036, 0.321] exists only inside commit message `969b00d`. `server/results/` has no
  `depth_tier*.json`. The negatives all have committed artifacts. **The evidentiary record is
  asymmetric in favour of the refutation.**
- The same commit reports `depth_flips=2` per slice from HNSW nondeterminism — the same
  magnitude as some effects it certifies. Whether +0.174 survives that jitter is unknown; the
  commit itself flags seeded HNSW as "next must-do" and no commit closed it.
- The LoCoMo ~9× graph-ON win is recorded in memory as CONTAMINATED with a clean rerun pending.
  `server/results/` contains **no LoCoMo file at all**. Cannot confirm the run happened.
- "17/18" is genuinely ambiguous and both readings appear in the same commit message
  (`3326bce`): (a) the gate found 17/18 multi-hops had the bridge→answer link; (b)
  episodes-first assembly scored 17/18. `CHANNEL_SEPARATION_DESIGN.md:32` uses (b). No raw
  result file exists for (b). **The single largest strategic pivot in the project — tiering —
  was taken on the strength of a noise-bound non-reproduction of a number whose meaning is
  ambiguous.**
- No external 2×2 exists. Every published study varies extractor quality with a fixed consumer,
  or the consumer with a fixed extractor. The hypothesis is externally **unfalsified and
  untested** — which cuts both ways. The closest single-axis evidence (the measured
  decoupling of extraction yield from answer quality; AtomMem's small graph delta; Mem0's
  production deletion) points toward a **modest** rather than transformative result.
- Every agent-memory benchmark found is conversational personalization over prose. Engram's
  corpus is 88% filesystem/document scaffolding and coding-session transcripts. **No external
  result covers that regime.**
- I did not audit consolidation-side graph consumers (dream spreading, merge neighbour
  Jaccard, infer PMI, microglia). Under `runtime_role=shell` they do not run in the shell; the
  2h cold brain runs some of them. Unmeasured here.

**About measurement itself**
- **"The live config" is not a well-defined object.** M13: the same code gives
  `consolidation_profile='standard'` or `'quiet'`, `node2vec` True or False, depending on how
  the process was launched. Two careful auditors disagreed in this very workflow, and the
  disagreement was not a mistake by either.
- `engram doctor` has a **correct** check for the extraction outage
  (`doctor.py:837-838`) that would have caught it on day one — but the probe is gated on `raw`,
  which comes from the same cwd-sensitive resolution chain. Run from the repo root it reports
  all-clear. **The detector and the defect share a root cause.**

---

## 7. What this says about Engram beyond the graph

The graph is not a special case. It is the clearest instance of a general pattern:

> **Mechanism shipped → measured once → result ambiguous → default-off flag → moved on.**
> The flag is the tombstone. The purpose was never validated against a consumer.

The pattern's signature is a subsystem that is *computed and discarded*, where the discard is
locally known — `pipeline.py` literally contains comments naming its own dead drops — but was
never traced to its consequence for the tier as a whole.

**Where else it is hiding, with evidence:**

| Subsystem | The mechanism | The missing consumer |
|---|---|---|
| **ACT-R activation** — the architecture's headline | Lazy activation from access_history, differential decay by memory tier | `activation: 0.0` on **55/55** live items I measured (M3). The system's namesake signal contributes nothing to any live ranking. |
| **Cues** — 9080 of them, ~1 per episode | Deterministic cue layer, cue-backed latent episodes | Open task #7: "the cue loop is 92% absent and has **NEVER** recorded a use." A capture mechanism at full scale with no recorded consumption. |
| **Graph embeddings** | Node2Vec / TransE / GNN, 1228 LOC + a 414-LOC phase | `weight_graph_structural=0.1` in live scoring; `recallGraphStructuralEmptySource` on 8/8 probes. **A scoring channel that has never had a producer, for any method, for any seed.** |
| **Atlas** | 1422 LOC, 21 regions, temporal history | `grep -c atlas mcp/server.py` = 0 (M20). Dashboard-only; it can never enter an agent's context. Its own edge count is fabricated (§2.2). |
| **Observer/reflect** | Synthesis phase | `observer_reflect_enabled: bool = False` (`config.py:2817`) and **force-set False post-init** (`config.py:3097`) even under `standard`. Ships dark. The one time it ran, its output was a diffuse mega-observation that never reached top-k. |
| **Claim-state ladder** | An 8-state graduated model — mentioned → … → superseded (`epistemic.py:583-592`) — exactly the contextual design the north star asks for | Default off; set only by `rework`; routed to `chat_runtime`, not MCP recall; and its **terminal state has no writer**. Reachable only via a `SUPERSEDED_BY` edge whose only producer is the materializer that is off and was wrong anyway (M15). |
| **Consolidation phase tree** | 15 phases | The 2026-07-17 review found 14 of 18 had never run. Three (mature/semanticize/schema) were later deleted as structurally unreachable. |
| **Instruments** | Dashboards, gates, stats surfaces | **Seven** inert or fabricating instruments found in a single day (§2.2). Two of them — the projection yield and the per-episode entity list — are precisely the surfaces that would have made a 65-day extraction outage visible. |

**Three structural lessons, stated as design rules:**

1. **A boolean is a frozen question, and two booleans encoding one question is worse.**
   `passage_first_channel_separated` implements exactly the additive, non-displacing assembly
   the north star asks for — and it is inert because it multiplies against a budget of zero.
   The graduated design was built, then frozen behind a binary that makes it unreachable.
   `entity_episode_traversal_source: results|candidates` has the same shape: it freezes
   *whether* to traverse when the real question is *from what*. **Before adding a flag, ask
   what statistic would make it a gradient. If none exists, that is the thing to build.**

2. **A mechanism with no instrumented consumer is indistinguishable from a deleted one.**
   The `engram://entity/{id}/neighbors` resource is the sharpest case: a working, reachable,
   well-formed graph consumer that no counter observes, so nobody can say whether it has ever
   been used. **Every read surface needs a use counter before it needs a feature.** This is
   cheaper than any experiment in this document and would close the highest-value unknown in
   Lens 1.

3. **The instrument is part of the system under test.** Four fabricating metrics, a gate that
   writes its own answer (M17), a battery that cannot score the thing it is cited for (M16),
   a doctor check that shares a root cause with the defect it detects, and a config whose
   value depends on the launcher (M13). **Repairs need a positive verification probe, not an
   absence of errors** — the lesson already recorded from the FastEmbed outage, and the reason
   the deciding experiment in §5 has a VOID pre-flight rather than a checklist.

**The uncomfortable synthesis.** Engram's core tier is validated and works. Its depth tier has
consumed 10,488 LOC (measured), 8 of 15 consolidation phases, 13–24% of the test suite, 69
config fields, and 25.3% of recall latency — to deliver, today, a continuity-gate fixture and
two sentence fragments. That is not a verdict on graph memory. It is a verdict on a development
loop that ships mechanisms faster than it builds consumers, and then measures the result with
instruments built by the same loop. **The deciding experiment in §5 is worth running. But the
cheapest thing in this entire document is a use counter on every read surface, and it is the
thing that would have prevented all of this.**

---

## Appendix — reproduction

```bash
# Live counts (do NOT trust /api/stats relationships — it extrapolates from 10 entities)
curl -s -m 30 127.0.0.1:8100/api/storage | jq .counts

# Zero-graph-payload probe (n=10, read-only)
#   GET /api/knowledge/recall?q=<q>&limit=10 → count relationship JSON bytes on entity items
#   and nonzero edgeProximity / spreading / activation in scoreBreakdown

# Spread completion + durable-lane wall-clock share (n=15, read-only)
#   client-side perf_counter around the request; read stageTimingsMs.recallSpread vs
#   .recallSpreadTimeout, and durableEntityFirst + durableEntityRescue

# The ONLY faithful way to read the live effective config (M13):
cd /Users/konnermoshier/Engram/server && zsh -c \
 'set -a; source ~/.engram/.env; set +a; exec ./.venv/bin/python -c "
from engram.config import EngramConfig; print(EngramConfig().activation.passage_first_entity_budget)"'
# Verify against GET /api/knowledge/runtime before trusting the output.

# Provider outage (decisive — factory logs INFO on success too, factory.py:52,71)
grep -h "Extraction provider" ~/.engram/logs/*.log | uniq -c    # → 1894 Narrow, 0 others
```

Raw probe artifacts:
`/private/tmp/claude-501/-Users-konnermoshier-Engram/2317d745-9e2f-4edd-b244-df7d3875b056/scratchpad/`
(`probe.py`, `stages.py`, `lat2.py`).
