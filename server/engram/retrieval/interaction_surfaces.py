"""Which write surface can emit which recall-interaction verb — declared, not inferred.

Ticket #37. `apply_chat_recall_feedback` (retrieval/chat_feedback.py) was the
only emitter of `used`/`dismissed`, and it hangs off POST /api/knowledge/chat —
the dashboard. `selected` comes from the same endpoint's tool loop. So on an
MCP/axi install, i.e. the actual product, `dismissed_count: 0` and
`selected_count: 0` were not measurements: nothing on those surfaces could ever
say otherwise. Live cue metrics read `selectedCount: 0` against
`surfacedCount: 1208` and the honest reading of that pair was never available.

A zero that means "no emitter" is a plausible-but-wrong metric — exactly what
STANDING_GOAL 2.1 forbids, and worse than an absent one, because it *ends*
investigation. This module is the declaration that lets a reader tell the two
apart, and `RecallNeedController.snapshot()` publishes it next to the counts.

Half of ticket #37 is (a) — `used` now has a real non-dashboard emitter
(`GraphManager.record_echoed_memory_usage`, fed by the observe/remember echo
scan). This module is half (b), for the verbs where (a) is not honestly
available:

* `dismissed` — the capture surface has no bounded response to partition, so
  "surfaced and not echoed in this one observe" would fire on memories the
  agent used in its answer and simply did not capture. It feeds
  `false_recall_rate` and the adaptive-threshold learner, so an over-firing
  dismissal does not merely add noise, it pushes recall in the wrong direction.
* `selected` — "the model picked this result to cite" is an artifact of the
  dashboard's LLM tool loop. MCP agents never report a selection back, and
  inventing one from "it was in the payload" is just `surfaced` renamed.
* `confirmed` / `corrected` — in the accepted vocabulary
  (`RecallMemoryInteractionApplier._VALID_TYPES`) with no emitter on any
  surface, dashboard included.

The declaration is only worth what keeps it true: every verb listed as
emittable by a surface must have a test that drives the real path and observes
it (see tests/test_interaction_surface_declaration.py and
tests/test_observed_usage_interaction.py).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# --- surfaces ---------------------------------------------------------------
# The dashboard knowledge-chat endpoint: POST /api/knowledge/chat, its tool loop
# and its post-response feedback pass.
DASHBOARD_CHAT = "dashboard_chat"
# observe / remember writes — MCP, axi, the auto-observe hook, the REST write.
AGENT_CAPTURE = "agent_capture"
# recall itself, from any surface that is not the dashboard chat loop.
AGENT_RECALL = "agent_recall"
# smoke / showcase / benchmark rigs.
EVALUATION = "evaluation"

# The `source` string the capture-side echo scan emits under.
OBSERVED_ECHO_SOURCE = "observed_echo"

# verb -> the surfaces with a REAL emitter today. Enumerated from the call
# sites, not from the accepted vocabulary: an empty set means "in _VALID_TYPES
# and emitted by nothing", which is the state `confirmed`/`corrected` are in.
# 2026-09-04: the dashboard chat route answers HTTP 501 (Engram runs no
# external model), so the DASHBOARD_CHAT emitters below still exist in the tree
# (the tree-scan test counts call sites) but are unreachable from any client;
# a lifecycle number that lists dashboard_chat as an emitter is naming code,
# not a surface that can move today.
INTERACTION_EMITTERS: dict[str, frozenset[str]] = {
    # retrieval/service.py, retrieval/recall_surface.py, evaluation/smoke.py
    "surfaced": frozenset({AGENT_RECALL, DASHBOARD_CHAT, EVALUATION}),
    # public_surface_policy.chat_tool_recall_policy -> retrieval/chat_tools.py (behind the 501)
    "selected": frozenset({DASHBOARD_CHAT}),
    # retrieval/chat_feedback.py (behind the 501) + GraphManager.record_echoed_memory_usage
    "used": frozenset({DASHBOARD_CHAT, AGENT_CAPTURE}),
    # retrieval/chat_feedback.py (behind the 501)
    "dismissed": frozenset({DASHBOARD_CHAT}),
    "confirmed": frozenset(),
    "corrected": frozenset(),
}


def surface_for_source(source: str | None) -> str:
    """Classify an interaction `source` string into the surface that produced it."""
    if not source:
        return AGENT_RECALL
    if source == OBSERVED_ECHO_SOURCE:
        return AGENT_CAPTURE
    if source.startswith("chat_"):
        return DASHBOARD_CHAT
    if source.startswith(("evaluation_", "showcase_", "benchmark")):
        return EVALUATION
    return AGENT_RECALL


def emittable_interactions(surfaces: Iterable[str]) -> frozenset[str]:
    """Verbs at least one of ``surfaces`` can emit."""
    observed = set(surfaces)
    return frozenset(
        verb for verb, emitters in INTERACTION_EMITTERS.items() if emitters & observed
    )


def unmeasurable_interactions(
    surfaces: Iterable[str],
    *,
    observed_counts: Mapping[str, int] | None = None,
) -> tuple[str, ...]:
    """Verbs whose 0 means "nothing here could say", not "it did not happen".

    ``observed_counts`` is the self-correcting half and is not optional in
    spirit: a verb that actually fired is measurable whatever this table
    believes, so a stale declaration can under-report deadness but can never
    label a live signal dead.
    """
    emittable = emittable_interactions(surfaces)
    counts = observed_counts or {}
    return tuple(
        sorted(
            verb
            for verb in INTERACTION_EMITTERS
            if verb not in emittable and not counts.get(verb)
        )
    )
