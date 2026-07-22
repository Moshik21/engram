# The Science Behind Engram

Engram is inspired by memory science, not a claim of biological equivalence.

It does not simulate the brain neuron by neuron. It borrows a set of principles
from neuroscience and cognitive science that seem useful for building better AI
memory systems.

The word "engram" comes from neuroscience: a memory trace distributed across
biological tissue. The exact details are still an active area of research, but
the central idea is powerful: memory is not just stored data. It is a trace that
can be strengthened, reactivated, updated, and sometimes lost.

That is the spirit Engram tries to bring into software.

## 1. Episodic Capture Before Durable Knowledge

Humans appear to rely on fast episodic capture and slower long-term integration.
In cognitive science, this is often discussed through complementary learning
systems: one system rapidly stores specific experiences, while another gradually
integrates stable structure across many experiences.

Engram mirrors that pattern:

- episodes capture what happened
- cues make recent episodes retrievable quickly
- projection extracts structure when warranted
- consolidation gradually turns repeated evidence into cleaner semantic memory

This is one of the main ways Engram differs from "extract everything
immediately" systems. Not every turn deserves expensive durable structure on
day one.

## 2. Cue-Dependent Retrieval

Human recall is cue-dependent. We often do not search memory by exact string
match. We retrieve because something in the present partially overlaps with a
stored trace.

That is why Engram uses cues and layered retrieval rather than treating memory
as a plain document store.

The practical implication is important:

- memory can remain latent
- retrieval can happen before full extraction
- the system can recover relevant episodes from partial overlap

This is closer to reminding than to search.

## 3. Activation, Recency, Frequency, and Relevance

Memory retrieval is not static. What comes to mind depends on how recently
something was used, how often it was reinforced, and how well it matches the
current situation.

Engram uses ACT-R-inspired activation ideas for that reason — but we measured
where they help and where they do not, and this section is amended accordingly.

**AMENDED 2026-07 (measured, not retracted quietly).** The original design used
ACT-R activation computed from access history as a *ranking* signal. Real-corpus
experiments refuted that specific use:

- The sigmoid-normalized activation term saturates at a single access, so it
  cannot discriminate 1 use from 50 (a 1-second-old single access scored ~0.91;
  the redesigned signal de-saturates it to 0.067).
- Access history conflated *surfaced* with *used* — the ranker was reinforcing
  its own output (an echo loop), not learning from behavior.
- With a populated activation store, reachability collapsed from 23 to 2/36 on
  the eval rig.

Evidence: `docs/product/experiments/M4_1_activation_arm.md`,
`docs/product/experiments/RF_target_design.md`,
`docs/product/RECENCY_FREQUENCY_GOAL.md`.

What **survived** is the underlying Anderson–Schooler insight: frequency ×
recency base-level learning over *real environmental statistics*. It was rebuilt
as `u = f · r'` over tiered, echo-guarded behavioral **usage events** (used /
confirmed / corrected — never mere surfacing), composed as a bounded
multiplicative tiebreaker (`final = semantic × (1 + β·u)`, β ≤ 0.30), so usage
can break ties but never overturn semantic relevance. It ships default-off until
organic usage data exists (`docs/product/experiments/M2_6_rerun_2026-07-21.md`).

ACT-R activation itself is **retained where it works**: forgetting, prune
floors, and memory hygiene — the decay side of the theory, not the ranking side.

One more honest note: *importance* (durable facts like identity, decisions,
preferences) is implemented as a **reserved lane**, not a multiplicative boost —
durable-typed results are surfaced in their own lane regardless of raw match
strength, and the bounded usage tiebreaker cannot cross that lane (see the F6
amendment in `docs/product/RECENCY_FREQUENCY_GOAL.md`).

## 4. Consolidation

Biological memory appears to consolidate over time. Recent experiences are not
the same thing as stable knowledge. Some memories become more abstract and more
general; others fade.

Engram reflects that with offline consolidation:

- triage decides what deserves further work
- merge resolves duplicates
- infer adds likely relationships
- replay revisits missed structure
- prune removes low-value clutter

*(Amended 2026-07: the standalone mature/semanticize phases were removed —
graduation was structurally unreachable in practice. Memory tiers remain:
identity-core entities promote to the semantic tier when flagged, with slower
decay and longer prune horizons.)*

This is not a literal replay of the hippocampus during sleep, but it follows a
similar design idea: memory quality should improve offline, not only in the hot
path of a conversation.

## 5. Reconsolidation

One of the most interesting findings in memory research is that recall is not
just read access. Retrieved memories can become labile and subject to update
before they restabilize. This is often discussed as reconsolidation.

That principle matters enormously for AI memory.

A good memory system should not treat retrieval as frozen lookup. It should
allow certain memories to be revised when new evidence arrives, especially when
the memory was recently brought back into use.

Engram borrows from this idea in its reconsolidation window and correction flow:
recently retrieved memories can be updated instead of forcing endless duplicate
facts to accumulate.

## 6. Forgetting Is A Feature

Humans do not simply preserve everything forever. Forgetting can reduce noise,
decrease interference, and help the system stay adaptive.

Software memory systems often ignore this and assume perfect retention is a
virtue. In practice, indiscriminate retention creates its own failure modes:

- stale facts keep surfacing
- weak signals crowd out important ones
- contradictions accumulate
- retrieval gets noisier over time

Engram treats forgetting and pruning as part of memory health, not a bug.

## 7. Memory Should Improve Through Use

Another lesson from cognitive science is that use matters. A surfaced memory is
not the same thing as a memory that was actually used to guide behavior.

That distinction is operationally important.

Engram now separates:

- surfaced
- selected
- used
- dismissed
- corrected

Only used / confirmed / corrected events carry ranking weight; surfaced events
count only for hygiene. An echo guard ensures a "used" event fires only when
the entity reappears in *novel* tokens — feeding the ranker's own output back
produces ~zero events (gate G7, `docs/product/RECENCY_FREQUENCY_GOAL.md`).
That lets the system learn from actual value instead of reinforcing itself for
merely showing the user something.

## What Engram Is Not Claiming

Engram is not claiming:

- that software memory is the same thing as human memory
- that its phases map one-to-one onto brain anatomy
- that neuroscience provides a finished blueprint for agent memory

The claim is narrower and stronger:

Memory science offers a better set of design principles than flat storage plus
prompt stuffing.

Those principles lead to systems that are:

- more selective
- more interpretable
- more adaptive
- more realistic about forgetting
- better suited to long-term continuity

That is why Engram takes memory seriously as a system, not as a feature.
