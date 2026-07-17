# I1 — get_context / context_builder adversarial audit (2026-07-17)

I1 READ-ONLY AUDIT — get_context / context_builder (no edits made; no files touched)

## Headline: the tiered context engine is mostly dead on the product surface

1. **VERIFIED — Durable pack short-circuits everything on any non-empty brain.** `build_mcp_context_surface` tries `_durable_context_payload_from_manager` first (context_builder.py:225-255) and returns if ANY of identity_core / Decision / Preference / Goal / Commitment / Correction / **Person** entities exist (`_list_durable_entities_by_type`, :2121, Person is durable per extraction/promotion.py:59-71). `max_packets = min(5, recall_packet_explicit_limit=3) = 3` (:1825). Consequence on every live brain, for both MCP (mcp/server.py:1200) and REST (:122): `MemoryContextBuilder.get_context` — layers 2-5 (topic recall, project neighbors, artifacts, recency, intentions, pinned contexts) and `template_briefing` — **never runs**. Only chat_runtime.py:407 and showcase adapters still reach the tiered builder. evaluation/continuity.py also measures the durable path, so the tiered engine is eval-dark too.

2. **VERIFIED — Durable path is topic-blind.** Identity-core listing fills to 3 and returns before name rescue (:1852-1863 early return :1862-1863; rescue :1866 only "if type/identity path was thin"). The type probes ignore the query entirely (:2123-2139). Bonus: for this repo the derived hint "Engram" casefolds into the discard set `{"engram","default","project"}` (:1816) and becomes the generic default query. `get_context(topic_hint=X)` on the dogfood brain returns the same first-3 identity packets for every topic and project.

3. **VERIFIED — No access recording on the durable path.** :851/:1145 recording never executes when the durable payload returns (nothing in `_durable_context_payload_from_manager` or `_finalize_durable_context_payload` records access). The review doc's premise "get_context records access" holds only for chat/direct-manager callers. Feedback-loop-wise this is accidentally *good*, but it means surfaced identity facts earn zero recency/frequency signal while chat-surfaced entities do — the ACT-R stream is now sampled by surface, not by use.

## Feedback loop (fallthrough/chat path)

4. **VERIFIED — Rich-get-richer loop, plus recording of never-surfaced entities.** Layer 3 is literally `get_top_activated` (:995), and every collected entity gets `record_access` (:1144-1145) — surfacing-because-activated → activated-because-surfaced. Worse: char-budget truncation happens at :1139-1141 BEFORE the recording loop, so entities cut out of the delivered context still get strengthened; in briefing format only ~3 lines/tier are shown (:706-713) yet up to ~44 entities (12+12+5+20 limits) are recorded.

5. **VERIFIED — Double recording for layer-2 entities.** `_context_recall` → `manager.recall` with `record_access=True` default (graph_manager.py:2941-2944, 2031); materializer records + opens labile window (retrieval/primary_results.py:285-287, feedback.py:163-189); then :1145 records the same entities again. Two access events per get_context per topic entity.

6. **VERIFIED — get_context is a write surface.** A read call creates a Project entity + records access when lookup misses (:841-855). No dedup guard between find and create → concurrent first calls can create duplicate Project entities (SUSPECTED race, no test).

## Staleness / cache correctness

7. **VERIFIED — forget() leaves every context cache stale.** `_invalidate_durable_context_after_remember` fires only from remember/observe commits (ingestion/capture_surface.py:995,1274). forgetting.py never invalidates the durable process cache (45s TTL, :39), the manager packet cache, or the briefing cache (300s TTL). A forgotten/corrected identity fact keeps being served by the highest-traffic surface for up to 45-300s.

8. **VERIFIED — silent-inert cache write.** The durable pack is mirrored into the manager packet cache under scope `"durable_context"` (:2053-2070) but `_context_cache_lookup_keys` only ever reads `identity_core`/`session_recent`/`project_home` (:1688-1714); grep shows no other reader of that scope. Write computed, stored, never consumed (the module-level process cache serves the fast path instead).

9. **SUSPECTED — briefing cache staleness/type drift.** Cache key is a 3-tuple `(group_id, topic_hint, growth_key)` (:665) vs the declared 2-tuple type (:528, :539, graph_manager.py:229) — annotation drift only; but the key omits project_path and graph content, so within TTL a summary/relationship change that doesn't bump episode/cue counts serves a stale briefing.

## Silent degradation / budget math

10. **VERIFIED — inert budget knob.** `entity_context_timeout = min(0.05, _budgeted_timeout(...))` (:808-811) where `_budgeted_timeout ≥ 0.25` (:1494-1495) — always exactly 50ms regardless of max_tokens; the budgeting term can never bind. Same shape: `_CONTEXT_RECALL_TIMEOUT_SECONDS=8.0` is unreachable for max_tokens<8000 (:778-781 yields 1.0s at the default 2000).

11. **VERIFIED — 50ms enrichment timeout degrades silently on large brains.** `entity_to_context_data` does get_activation + get_relationships + up to 5 sequential name resolves + get_entity (:572-598); on native large brains this plausibly exceeds 50ms → fallback returns `activation=0.0, facts=[]` with only a debug log (:1319-1334). Then layer sorting by activation (:880, :983) becomes arbitrary, briefing "Known context" picks the first 3 arbitrarily, and fact_count silently collapses — no degradation flag in the payload.

12. **VERIFIED — unguarded awaits in the tiered path.** Only TimeoutError is caught around project lookup (:829) and topic recall (:892); `create_entity`/`record_access` (:850-851), `get_top_activated` (:995), `get_entity` (:1000), and the final record/publish loop (:1145-1152) are bare. Any storage raise after the context is fully built still destroys the whole response (manager re-raises, graph_manager.py:3043-3055). Post-storage-hardening impact SUSPECTED (native raise frequency unknown), code path VERIFIED.

13. **SUSPECTED — briefing silently swaps to structured when `briefing_enabled=False`** (:1169 gate skips the whole block; result at :1219 carries no `briefing_degraded` flag). Default is True (config.py:1737) so dormant. Also `_render_durable_briefing` can return `""` → briefing-format payload with empty context and no degraded flag (:2032-2035, :2090-2091) — needs packets with no title/summary, low likelihood.

14. **SUSPECTED — pinned-content leak into template briefing.** Header "## Pinned Contexts" matches no tier keyword (:690-703); if `pinned_result` contains markdown list lines they append into whichever tier was last active (usually tier3 "Recent activity").

15. **VERIFIED (minor) — defensive `_close_awaitable` drops async cache impls.** If a caller wires async `get_cached_packets`/`cache_packets`, lookups/writes are silently discarded (:1386-1388, :1479-1480). Current wiring is sync (graph_manager.py:1023,1043) so not live.

## Answers to the specific hunt questions
- Does briefing surfacing strengthen what it surfaces? On MCP/API: no — it records nothing (finding 3). On chat/fallthrough: yes, and it strengthens ~10x more than it surfaces (findings 4-5).
- Does the template briefing reflect graph state or fall back silently? On the product surface it's `_render_durable_briefing` over 3 topic-blind identity packets (findings 1-2); `template_briefing` runs only on near-empty brains or chat, with degradation flags present (:1192-1196, :1209-1210) — that part is honest.
- Largest-brain stale/wrong output: topic-blind identity-only context (2), forgotten-fact serving (7), zero-fact/zero-activation silent degradation (11).

Suggested fix order: 2 (blend topic rescue before early-return) → 7 (invalidate on forget) → 4/5 (record only delivered entities, once) → 11 (degradation flag + raise timeout) → 8/10 (delete inert code).

No changes per file (investigation was read-only); ruff/pytest not applicable.