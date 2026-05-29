# Recall Channel Separation — Design

Status: proposed (2026-05-29). Author: investigation across 4 graph-on/off evals + the
`scripts/benchmark_graph_thesis.py` harness. Builds on the now-implemented recall
pipeline (`docs/RECALL_REDESIGN.md`).

## Problem (empirically established)

Recall returns **one ranked top-k list** that mixes two retrieval primitives —
**episodes** (raw conversation evidence) and **entities** (resolved graph
knowledge: summaries, attributes, relationships) — plus cues/intentions. These
primitives compete for the same `limit` slots, but different question intents need
different primitives:

| Intent | Best primitive | Example |
|--------|----------------|---------|
| needle / multi-hop | **episode** | "the chat where I discussed the migration" |
| current value / temporal / knowledge-update | **entity / claim** | "what is Priya's title *now*" |
| prospective | **intention / cue** | "remind me to…" |
| synthesis / "what do I know about X" | **entities + relationships** | overview of a topic |

Because they share one ranked list, **any fixed entity/episode budget is wrong for
some intent.** Measured on the trusted multi-persona eval (3 personas, 18 multi-hop +
12 control, native + clean LLM extraction, `benchmark_graph_thesis.py`):

- **Entity budget = 3** (the shipped default, sized against the inflated overfetch
  limit): graph-ON multi-hop **10/18** (graph-OFF 14/18) — entities evict answer
  episodes. Net **−4**.
- **Entity budget = 1** (display-sized cap): multi-hop **11/18**, but it **broke the
  `prospective_trigger`/`cue_prospective` showcase scenarios** (the reminder entity
  got capped out).
- **Episodes strictly first** (entities additive at the tail): multi-hop **17/18**
  (graph-ON now *beats* OFF) **but broke `temporal_override`/`correction_chain`/
  `summary_drift_resistance`** (those answer *from an entity*, which got pushed past
  the limit).

No single blend satisfies all scenarios simultaneously. This is the core finding:
**the limitation is the single-ranked-list architecture, not a tuning value.**

(Separately: the graph machinery itself is sound — the apply/merge/resolution path is
correct, and the self-loop bug was fixed in `bbe74cd`. The remaining issue is purely
how results are *assembled and presented*.)

## Design: independent result channels

Stop forcing primitives into one ranked list. Recall returns **distinct,
independently-budgeted channels**; the consuming LLM agent receives all of them and
uses what the question needs (aligned with Engram's thesis: the agent is the
intelligence — don't make Engram guess the intent and guess wrong):

- **`evidence`** — episodes / cue-episodes (raw passages). Budget: most slots.
- **`facts`** — resolved entity claims (subject–predicate–object, current value,
  temporal validity). Budget: independent.
- **`reminders`** — fired intentions / prospective cues. Budget: independent.
- **`related`** — graph context entities (additive breadth for synthesis). Budget:
  independent.

Each channel has its **own budget**, so a fact entity can never evict an answer
episode and vice-versa. The eviction tension disappears by construction.

### Result-shape change

`manager.recall()` currently returns `list[dict]`, each item carrying
`result_type` ∈ {`episode`, `cue_episode`, `entity`}, plus `episode`/`entity`/`cue`,
and `score` (built by `RecallPrimaryResultMaterializer.materialize`,
`engram/retrieval/primary_results.py`; consumed via `result.get("episode"/"entity"/
"cue")` in `engram/retrieval/recall_surface.py`).

Backward-compatible plan: **keep the flat `list[dict]`** (existing consumers keep
working) but (a) guarantee per-channel representation in it via per-channel budgets,
and (b) add a parallel `channels: {evidence: [...], facts: [...], reminders: [...],
related: [...]}` grouping for consumers that want it. The MCP/API surfaces and
`get_context` then present labeled sections instead of one list.

## Entry points (where to change)

1. **`engram/retrieval/pipeline.py` — Step 6 assembly (~line 1916).** Replace the
   single `entity_budget` + mixed sort with **per-channel budgets** computed against
   the *consumer* limit (not the overfetch `top_n`). Produce per-channel ranked
   sub-lists.
2. **`engram/retrieval/request_policy.py` — `split_primary_and_near_miss_results`.**
   This is where the consumer-facing truncation drops slots; make it channel-aware
   (truncate *within* each channel's budget) instead of `scored_results[:limit]`.
   The reverted band-aid (`git show d0743a4`) shows the wrong fix (single cap);
   per-channel budgeting is the right one.
3. **`engram/retrieval/service.py` — `RecallService.recall` (~line 132).** Pass the
   real consumer `limit` + per-channel budget config through; assemble `channels`.
4. **`engram/retrieval/recall_surface.py` — `build_mcp_recall_surface`,
   `build_api_recall_surface`, `build_mcp_explicit_recall_tool_surface`.** Present
   channels as labeled sections (reuse the existing `packets` abstraction /
   `assemble_memory_packets` in `engram/retrieval/packets.py`, which already supports
   typed packets).
5. **`engram/retrieval/context_builder.py` — `MemoryContextBuilder.get_context`.**
   Render `evidence` / `facts` / `reminders` sections in both the structured and
   briefing formats.
6. **Consumers to update**: `engram/mcp/server.py` (recall + get_context tools),
   `engram/api/knowledge.py` (REST recall), `engram/retrieval/auto_recall.py`.

## Config

Add per-channel budget fields to `ActivationConfig` (`engram/config.py`), e.g.
`recall_evidence_budget`, `recall_facts_budget`, `recall_reminders_budget`,
`recall_related_budget` (defaults sized so evidence leads but facts/reminders are
guaranteed ≥1 when present). Deprecate the overloaded `passage_first_entity_budget`
(keep as an override).

## Phased plan

- **P0 — per-channel budgeting (dissolves the displacement).** Compute budgets
  against the consumer limit; guarantee each present channel its slots; truncate
  within channels. Flat `list[dict]` preserved. *Gate:* graph-ON ≥ graph-OFF on
  multi-hop **and** all showcase scenarios pass (the combination no single blend
  achieved).
- **P1 — labeled channels in surfaces.** `channels` field + sectioned `get_context`/
  MCP/API presentation. *Gate:* surface-contract + presenter-boundary tests; showcase.
- **P2 — temporal claim/fact layer.** First-class `(subject, predicate, object,
  valid_from, valid_to, superseded_by)` answers surfaced as the `facts` channel,
  built on existing temporal relationships (`valid_to`). Directly serves
  current-value/temporal/knowledge-update without raw-episode noise or lossy
  summaries. *Gate:* temporal/knowledge-update categories improve.
- **P3 — fact-rich, current entity summaries.** Entity summaries carry resolved
  current facts ("Priya — Director at Nimbus (Apr 2024), ex-Acme") so surfacing an
  entity *answers* rather than points. *Gate:* `facts` channel hit-rate.
- **P4 (optional) — query-intent weighting/routing + iterative retrieval** for
  non-nameable-bridge multi-hops (where the bridge entity isn't named in the query;
  needs decompose→re-retrieve, which single-pass can't do).

## Eval gates (the verifier already exists)

Success is defined by passing **both** simultaneously — which no single blend did:

1. `scripts/benchmark_graph_thesis.py data/graphthesis/*.json --top-k 5`
   (native + clean LLM extraction via `EVIDENCE_EXTRACTION_ENABLED=false` + a real,
   un-shadowed `ANTHROPIC_API_KEY`): **graph-ON multi-hop ≥ graph-OFF (14/18)**,
   controls neutral.
2. `tests/benchmark/test_showcase_runner.py` — ALL golden scenarios pass
   (`temporal_override`, `correction_chain`, `summary_drift_resistance`,
   `prospective_trigger`, `negation_correction`, `latent_open_loop_cue`).
3. Full non-helix suite green; LongMemEval-oracle graph-OFF baseline unchanged
   (~62–63% category — the episode-vector floor must not regress).

## Risks / notes

- **Backward compat:** keep the flat `list[dict]`; `channels` is additive. Surface
  boundary tests (`tests/test_public_surface_presenter_boundaries.py`) constrain how
  surfaces present — update boundary maps as needed.
- **Latency:** channels are a re-org of already-retrieved candidates, not new
  fetches — negligible cost. `observe` hot path unaffected.
- **The graph-thesis personas only cover *nameable-bridge* multi-hops** (the bridge
  entity is named in the query). Non-nameable bridges need P4 (iterative retrieval);
  do not claim full multi-hop coverage from P0–P3.
- **Don't re-introduce the silent failure mode:** if a channel is empty because
  extraction produced nothing, surface that (a metered/observable signal), not a
  silently shorter result.
