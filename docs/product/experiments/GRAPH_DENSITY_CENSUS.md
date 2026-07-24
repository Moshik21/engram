# Graph Density Census — is there even a graph?

Snapshot: 2026-07-24, live dogfood brain (`~/.helix/engram-native-dogfood-axi`,
group `default`), read-only HTTP against the running shell on `127.0.0.1:8100`.
No production code was touched. Method: complete BFS crawl of
`/api/entities/{id}/neighbors?depth=1` seeded from 320 entities (100-per-type
`/api/entities/search` + 21 atlas region hubs/centers), iterated to closure.

**The crawl reached closure at 834 nodes / 1637 edges — exactly matching the
authoritative `count_relationships_by_group` total. This is a complete census,
not a sample.**

Verdict up front: **NO-GO on the depth/answerability A/B.** Not because the
graph is empty — it is not — but because 88% of it is filesystem/document
scaffolding produced by two write paths that are now disabled, 90% of its 2-hop
paths route through five nodes, and the live extractor contributes ~0.6%
semantic edges. An A/B run today would measure a directory listing.

---

## 0. Correcting the premise

The task brief cited **750 relationships**. That number is wrong and it comes
from an instrument bug.

| Source | Relationships | How |
|---|---|---|
| `/api/stats` | **750** | extrapolation from a 10-entity sample |
| `/api/storage` | **1637** | exact `COUNT` over `E<RelatesTo>` |
| Independent full crawl (this census) | **1637** | union of every edge returned by 834 depth-1 neighbor calls |
| Atlas region sums | **1637** | 1334 intra-region + 303 bridge |

`storage/helix/graph.py:2387-2405`:

```python
# Relationship count is expensive (per-entity edge scan); approximate from first few
sample_ents = entities[:10]
...
if entity_count > 10 and sample_hids:
    relationship_count = int(relationship_count * entity_count / 10)
```

The first 10 entities happened to carry 9 outgoing edges → `9 * 834 / 10 = 750`.
The true figure is 1637. **Relationships do not outnumber entities the wrong
way**: the ratio is 1.96 edges per entity, not 0.90.

Two more silently-inert instruments found while measuring (same bug class as
`project_silent_inert_bugs.md`):

- `storage/helix/graph.py:2511-2513` — `projection_metrics.yield.relationship_count`
  and `avg_relationships_per_projected_episode` are **hardcoded literal `0`** on
  the exact stats path. The "0.0 relationships per projected episode" visible in
  `/api/stats` is a constant, not a measurement.
- `retrieval/graph_state.py:1050-1051` — `_build_episode_summary_item` hardcodes
  `"entities": []` and `"factsCount": 0`. Every episode in `/api/episodes` reports
  zero linked entities regardless of truth, so per-episode yield cannot be read
  from that surface at all.

Corrected headline numbers:

| Metric | Value |
|---|---|
| Entities | 834 |
| Relationships | **1637** |
| Episodes | 9286 |
| Cues | 9044 |
| Episode→entity links (`HasEntity`) | 7673 |
| Entities per episode | 0.090 |
| Edges per entity | 1.96 |
| Mean degree | 3.93 (4.28 over non-orphans) |

---

## 1. What graph exists

### 1.1 Entity type distribution (n=834)

| Type | Count | Orphans (degree 0) |
|---|---|---|
| Artifact | 530 | 0 |
| Technology | 171 | 50 |
| Decision | 82 | 5 |
| Concept | 23 | 5 |
| Identifier | 12 | 6 |
| Project | 10 | 0 |
| Correction | 3 | 3 |
| Schema | 2 | 0 |
| Person | 1 | 0 |

### 1.2 Predicate distribution (n=1637)

| Predicate | Count | Share | Class |
|---|---|---|---|
| PART_OF | 816 | 49.8% | structural (file→project containment) |
| SUPERSEDED_BY | 260 | 15.9% | structural (decision chain) |
| DOCUMENTED_IN | 202 | 12.3% | structural (decision→doc) |
| MENTIONED_WITH | 163 | 10.0% | co-occurrence (`consolidation/phases/infer.py:236`) |
| INSTANCE_OF | 123 | 7.5% | structural (→Schema signature node) |
| DREAM_ASSOCIATED | 25 | 1.5% | synthetic, TTL |
| DECIDED | 14 | 0.9% | structural |
| ANNOUNCED_AS | 13 | 0.8% | structural |
| DECIDED_IN | 11 | 0.7% | structural |
| USES | 3 | 0.18% | **semantic** |
| AIMS_FOR | 3 | 0.18% | **semantic** |
| WORKS_AT / PARTNER_OF / LIKES / IMPLEMENTED_BY | 1 each | 0.24% | **semantic** |

**Structural/taxonomic: 1439 (87.9%). Co-occurrence: 163 (10.0%).
Dream: 25 (1.5%). Genuine extracted semantic relations: 10 (0.61%).**

All ten semantic edges, in full:

```
Users/konnermoshier                       -USES->          Claude..
Users/konnermoshier                       -AIMS_FOR->      take Engram to the next level
Users/konnermoshier                       -AIMS_FOR->      get a solid handoff for future agents
writeide-work:decision_statement:**Why i  -WORKS_AT->      TechCorp
writeide-work:decision_statement:**Why i  -PARTNER_OF->    Jane
MCP Contract\n\nOpenClaw should           -USES->          Engram
The SaaS version should feel              -LIKES->         Engram
writeide-work:decision_statement:**Why i  -USES->          it.
writeide-work:decision_statement:**Why i  -AIMS_FOR->      think about our PR flow
Engram:decision_statement:bundle: ## Bui  -IMPLEMENTED_BY-> Makefile
```

Six of the ten have a `:decision_statement:` squatter or a sentence fragment as
an endpoint. **Zero of the ten are usable knowledge.**

Top src→predicate→tgt shapes:

```
636  Artifact  -PART_OF->        Project
260  Decision  -SUPERSEDED_BY->  Decision
202  Decision  -DOCUMENTED_IN->  Artifact
163  Technology-PART_OF->        Project
 89  Artifact  -INSTANCE_OF->    Schema
 87  Technology-MENTIONED_WITH-> Technology
```

### 1.3 Topology — a five-spoke star, not a graph

| Metric | Value |
|---|---|
| Entities with ≥1 edge | 765 (91.7%) |
| Orphans (degree 0) | 69 (8.3%) |
| Connected components | 76 |
| Giant component | 752 (90.2%) |
| Other components | 1×size-3, 5×size-2, 69 singletons |
| Degree buckets | 0:69, 1:416, 2:111, 3-5:142, 6-10:53, 11-25:35, 26-50:2, 51+:6 |
| Edge endpoints held by top-10 entities | **39.0%** |
| Edge endpoints held by top-50 entities | 54.7% |
| Mean shortest path (40-seed BFS) | 3.29 |

Top-degree nodes:

```
371  Project    MachineShopScheduler
279  Project    Engram
183  Project    writeide-work
110  Decision   MachineShopScheduler:decision_statement:4. **Making a Decision**:
103  Decision   writeide-work:decision_statement:**Why is the code public?**…
 98  Schema     "Artifact: PART_OF-Project, DOCUMENTED_IN-Artifact"
 49  Decision   Engram:decision_statement:bundle: ## Build the public install…
 34  Technology SQLite
 25  Schema     "Decision: DOCUMENTED_IN-Artifact, SUPERSEDED_BY-Decision"
 24  Technology API
```

**The load-bearing measurements:**

- **50.9%** of all edges are incident to just the 3 `Project` nodes.
- **50.8%** of entities (424) have a neighbourhood consisting *entirely* of
  hub/schema nodes — their only graph fact is "lives in this repo".
- **362** degree-1 entities have their single edge pointing at a `Project` hub.
- **90.2%** of all 2-hop paths route through one of 5 nodes (3 Projects +
  2 Schemas).
- Deleting those 5 nodes shatters the 752-node giant component into
  **502 components, 493 of them singletons** (largest surviving: 308).
- Median 2-hop neighbourhood size is **183 entities** (p90 337, max 677) —
  "expand one hop" returns a directory listing, which is precisely the
  retrieval-noise failure the depth tier is supposed to fix.
- Only **186 entities (22.3%)** have even one non-structural incident edge.

### 1.4 Provenance: how much is materializer artifact vs extracted knowledge

| Class | Count | Share | Producer |
|---|---|---|---|
| File-path `Artifact` (`docs/…md`, `server/…py`) | 523 | 62.7% | `ingestion/project_bootstrap.py:291` repo scan |
| `X:decision_statement:…` squatters | 59 | 7.1% | `ingestion/decision_materializer.py` |
| Schema-signature pseudo-entities | 2 | 0.2% | residue of the deleted schema-formation phase; still merge-exempt (`consolidation/phases/merge.py:49`) |
| **Materializer-class subtotal** | **584** | **70.0%** | — |
| Other `Technology` | 171 | 20.5% | narrow extractor (see below) |
| Other `Decision` | 23 | 2.8% | mixed |
| Other `Concept` / `Identifier` / `Project` / `Artifact` / `Correction` / `Person` | 56 | 6.7% | mixed |

**1283 of 1637 edges (78.4%) touch a materializer-class entity; 586 have
materializer-class nodes at both ends.**

The remaining "extracted" entities are largely fragments, exactly as
`project_extraction_is_the_graph_lever.md` predicted for the narrow provider:

- **162 of 171 (94.7%)** `Technology` entities are slash/path/`N/A` fragments:
  `no-go/depth`, `MACHINES/STATUS`, `insert/update`, `merge/infer`, `lite/full`,
  `7a04e113-4e1d-4b53-8a7e-9584b69f11ef/tasks`, `onnx/model_quantized.onnx`,
  `N/A`. The 9 survivors are generic tokens: API, SQLite, MCP, CLI, SDK,
  TypeScript, Next.js, page.tsx, React — none carries a fact.
- All 12 `Identifier` entities are fragments: `arm-B/G2`, `6-18/s`, `07/02`,
  `1/4HH`, `2/hour`, `6379/0`.
- `Concept` (23) is mostly sentence scrap: `it.`, `If you`,
  `Waiting for the full`, `Right now the benchmark`,
  `MCP Contract\n\nOpenClaw should`, `The SaaS version should feel`.
- Duplicates survive un-merged: `Engram` exists 6× as `Concept` plus 1× as
  `Project`; `Helix native store` 3×; `it.` 2×; `Cold Decision hit requires
  healthy search index` twice as `Decision`.
- 5 of 10 `Project` entities are dead test fixtures
  (`engram-followup-fresh-1782833127`, `engram-fresh-verify-59519`, …).

**Generously counted, entities that encode a real, usable fact about the user's
world number roughly 40-45 of 834 (~5%).**

### 1.5 The graph is frozen, not growing

Edge creation by month: 2026-05 → **814**, 2026-06 → **495**, 2026-07 → **328**.
Of the 328 July edges: PART_OF 243, SUPERSEDED_BY 55, DECIDED 14,
DOCUMENTED_IN 11, ANNOUNCED_AS 1 — and **4 semantic** (2 USES, 2 AIMS_FOR).

Both producers of that July volume are now **off** at runtime
(`GET /api/knowledge/runtime`):

```json
"features": { "artifactBootstrapEnabled": false, "decisionGraphEnabled": false }
```

(`bootstrap_project` still runs on explicit MCP call — `artifactBootstrap.lastObservedAt`
is `2026-07-24T16:01:49`, artifactCount 200 — which is where today's 113 new edges
and 125 new `Artifact` entities came from. It is not an organic write path.)

So the graph's density is a **legacy deposit from two disabled scaffolding
paths**, and the only lane still adding edges organically is contributing single
digits per month.

---

## 2. Why it is this thin — the write-path trace

### 2.1 CQRS: most episodes never reach an extractor

`projection_state` over all 9286 episodes (`/api/stats`, exact path):

| State | Count | Share |
|---|---|---|
| projected | 4968 | 53.5% |
| **cue_only** | **3377** | **36.4%** |
| scheduled | 818 | 8.8% |
| merged | 58 | 0.6% |
| cued | 33 | 0.4% |
| failed | 19 | 0.2% |
| projecting | 12 | 0.1% |
| queued | 1 | 0.0% |

- `observe` → `store_episode()` → QUEUED, **no LLM, no extraction**. This is the
  default the MCP system prompt biases toward ("if uncertain, observe it").
- Triage promotes only the top ~35%; the rest are marked CUE_ONLY. 3377 episodes
  (36%) are *by design* never projected — a cue string is all that survives.
- `worker_enabled=False` under the live `quiet` profile
  (`config.py:3020`), and `runtime_role=shell` means the shell runs **no**
  consolidation phases. Deferred projection is the cold brain's job, every 2h.
- 864 episodes are stuck mid-flight (scheduled/cued/projecting/queued).

**Even at 100% success, 4318 of 9286 episodes (46.5%) have never been through an
extractor.** But note the more damning ratio: the 4968 that *were* projected
produced only 834 entities total — and 584 of those came from the repo scanner,
not from episodes. **Episode-derived entities ≈ 250 from 4968 projections =
0.05 entities per projected episode.**

### 2.2 The live extractor is the narrow deterministic one, and it is failing over

`config.py:1916` — `extraction_provider` default is `"narrow"`. The live
deployment sets `ollama`, but:

```
236× [WARNING] extraction_provider='ollama' but Ollama not reachable at
     http://100.106.100.46:11434 — falling back to narrow extraction
```

(`~/.engram/logs/engram.log`; `extraction/factory.py:62-89` — the `strict` path
returns `_make_narrow(cfg)`.) The configured Ollama host is a Tailscale address
that is not answering. **Every projection since has run on regex.**

`extraction/narrow/relationship_extractor.py` is 32 regexes over 464 lines
covering WORKS_AT, LIVES_IN, MARRIED_TO, USES, LIKES, DISLIKES, KNOWS,
STUDIES_AT, MANAGES, PREFERS, AIMS_FOR, STARTED, ACQUIRED, HAS_CONDITION,
ATTENDED, PARTNER_OF, PARENT_OF, CHILD_OF, SIBLING_OF. Over 9286 episodes it has
produced **10 committed edges**.

### 2.3 The commit policy self-limits as the graph grows

This is the mechanism, and it is a real defect:

- `extraction/commit_policy.py:78` — base relationship commit threshold `0.75`.
- `extraction/commit_policy.py:169-170` — `if entity_count > 500: return base + 0.05`
  → at 834 entities the **effective relationship threshold is 0.80**.
- `extraction/narrow/relationship_extractor.py:368` — the third-person pattern
  block (the 9 largest predicate classes: WORKS_AT, LIVES_IN, MARRIED_TO, USES,
  LIKES, DISLIKES, KNOWS, STUDIES_AT, MANAGES) emits **`confidence=0.75`**.
- First-person patterns emit 0.80 (line 410, exactly at threshold); family
  patterns 0.85 (line 445).

**Every third-person relationship the narrow extractor can find is born 0.05
below the bar it must clear, and only because the graph crossed 500 entities.**
It does not even accumulate as deferred debt: the hot-path junk gate
(`commit_policy.py:183-199`, `consolidation/evidence_drain.classify_extraction_candidate`)
rejects pattern scrap outright. Live deferred evidence is **10 items** total —
there is no backlog to adjudicate, because the candidates are being destroyed.

The "adaptive density" heuristic was written on the assumption that a dense
graph is a good graph and should get pickier. Here the density is 88% filesystem
scaffolding, so the heuristic reads repo-scan volume as extraction quality and
throttles the only semantic lane that exists.

### 2.4 Summary of the causal chain

```
observe-by-default (CQRS)      → 46.5% of episodes never extracted
triage top-35%                 → 36.4% permanently CUE_ONLY
Ollama unreachable             → 100% of extraction runs on regex
narrow regex on prose          → fragment entities ("no-go/depth", "insert/update")
                                 94.7% of Technology entities are scrap
entity_count > 500             → relationship threshold 0.75 → 0.80
narrow third-person = 0.75     → structurally below threshold, junk-gated, destroyed
                               ⇒ 10 semantic edges in 9286 episodes
meanwhile: bootstrap + decision materializer (now OFF) deposited
                               ⇒ 1439 structural edges = 87.9% of the graph
```

---

## 3. GO / NO-GO for the depth/answerability A/B (task #3)

### NO-GO. Running it today would be the third mis-measurement.

The 2026-06-04 attempt returned NULL because the graph was inert. This graph is
not inert — it is *actively misleading*, which is worse:

1. **The treatment has nothing to retrieve.** "Expand 1 hop from a hit entity"
   returns a median 183-entity 2-hop neighbourhood, 90.2% of it routed through
   `Project` hubs. The neighbours are co-repo files. Adding them to evidence
   can only dilute — which reproduces the already-known
   *edge-triple-in-evidence is net-NEGATIVE* result rather than testing depth.
2. **The graph answers the wrong questions.** 87.9% of edges answer "what file
   is in what repo" and "which decision superseded which". Answerability
   questions are about the user's world — people, preferences, commitments,
   causes. There are **10** such edges, 6 of them malformed.
3. **A null result would be uninterpretable again.** With 0.61% semantic edges
   you cannot distinguish "graph depth does not help" from "there is no depth
   to traverse". That is the exact ambiguity that burned the 2026-06-04 run.
4. **The instruments are broken.** `/api/stats` extrapolates relationships from
   10 entities, hardcodes per-episode relationship yield to 0, and
   `/api/episodes` hardcodes `entities: []`. An A/B cannot report yield deltas
   through surfaces that report constants.
5. **The corpus is not organic.** 51% of the 400 most recent episodes are
   `auto:bootstrap` repo-scan episodes. Half of what looks like "memory" is a
   `find` command.

### What must change first, cheapest path

**Gate 0 — repair the instruments (hours, no risk).** Nothing downstream is
measurable until these three report truth:
`storage/helix/graph.py:2387-2405` (real count, use `count_relationships_by_group`
on the exact path too), `storage/helix/graph.py:2511-2513` (compute the
relationship yield instead of returning 0), `retrieval/graph_state.py:1050-1051`
(hydrate or delete the field — a hardcoded `[]` is worse than absent).

**Gate 1 — restore a real extractor (hours).** This is the single highest-leverage
change and `project_extraction_is_the_graph_lever.md` already proved it
(narrow: 12 fragments / 0 edges vs Ollama: 24 entities / 23 edges on the same
input). Concretely: point `ollama_base_url` at a *reachable* local Ollama
(the configured `100.106.100.46:11434` has been dead for 236 logged attempts),
or run the agent-as-extractor proposal path. Per
`project_fully_local_north_star.md`, an external key is acceptable **only** to
benchmark the ceiling, not to operate.
Verification probe (the lesson from `project_silent_inert_bugs.md`): after the
change, project 20 known-content episodes and assert ≥1 committed semantic
relationship. Do not trust "no error in the log".

**Gate 2 — unblock the commit path (one-line, needs a decision).**
`commit_policy.py:169-170` raises the relationship threshold to 0.80 above 500
entities while the narrow third-person class emits 0.75. Either the density
adjustment should not apply to `relationship` fact-class, or it should key off
*semantic* entity count rather than total (which is 70% scaffolding). Do not
just lower the threshold blindly — measure the precision cost on a labelled
slice first.

**Gate 3 — backfill projection, in this order (hours of compute).** 4318
episodes have never been projected. Backfilling them *before* Gate 1 multiplies
the fragment population; backfilling *after* is the test of whether the corpus
contains extractable knowledge at all. Run it on a **clone**, not the live
dogfood brain.

**Gate 4 — build the A/B corpus deliberately.** The dogfood brain should not be
the substrate for the depth A/B even after repair: it is 62.7% file paths and its
history is contaminated by two now-disabled materializers. Use a purpose-built
conversational corpus with known ground-truth multi-hop questions, and per
`project_graph_answerability_ab.md` report **reachability**, not accuracy — the
local judges sit below the accuracy instrument's floor.

### Re-open criterion (the precondition, stated as a number)

Re-run the depth A/B only when, on the A/B corpus:

- semantic (non-PART_OF/DOCUMENTED_IN/INSTANCE_OF/SUPERSEDED_BY/MENTIONED_WITH)
  edges are **≥ 20%** of all edges (today: 0.61%), and
- **≥ 50%** of entities have a non-structural incident edge (today: 22.3%), and
- **< 30%** of 2-hop paths route through the top-5 hubs (today: 90.2%), and
- median 2-hop neighbourhood size is **< 30** entities (today: 183).

Until then the honest answer to "is there even a graph?" is: there is a
**repository index and a document-supersession chain**, correctly built and
1637 edges strong, plus **ten broken sentences** that constitute the entire
temporal knowledge graph the product markets.

---

## Appendix — reproduction

```bash
# exact counts (do NOT trust /api/stats relationships)
curl -s 127.0.0.1:8100/api/storage | jq .counts

# full crawl to closure (read-only; ~30s, 834 depth-1 neighbor calls)
#   seed:  /api/entities/search?type=<T>&limit=100  for the 9 types
#        + hubEntityIds/centerEntityId from /api/graph/atlas
#   union every edge from /api/entities/{id}/neighbors?depth=1&max_nodes=2000
#   iterate frontier until empty → 834 nodes / 1637 edges
```

Caveats: `/api/entities/search` caps `limit` at 100 with no cursor, so seeding
alone cannot enumerate 530 Artifacts — closure of the crawl (verified against
the exact edge count) is what makes this a census. Episode-level entity linkage
could not be measured from `/api/episodes` because the field is hardcoded empty;
the 7673 `HasEntity` total from `/api/stats` was used instead. The shell was
restarted twice mid-census by a concurrent workflow; all figures above were
re-confirmed after the last restart.
