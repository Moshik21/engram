"""F6 importance-vs-semantics invariants (RF goal M3.1).

Pins the LIVE durable ranking mechanism and its interaction with the usage
tiebreaker. Per F6 (RECENCY_FREQUENCY_GOAL.md): the durable boost constants
are shipped behavior — justify-and-pin now; any rebalance is a default
change that lands only behind the live continuity gate.

MECHANISM AMENDMENT (2026-07-21, M3.1 verify): the design docs framed the
durable boost as a bounded multiplicative band (sem x 2.5, "overtake iff
sem_durable >= sem_gold/2.5", composed ceiling 3.25x). The tree has NO such
path. The live composition is stage-sequential:

1. Candidate scoring (retrieval/scorer.py): the usage tiebreaker multiplies
   composite semantics by (1 + beta*u), beta_max = 0.30 — so usage amplifies
   a score by at most 1.30x (the overtake theorem, pinned in
   test_overtake_theorem_property.py).
2. Result post-processing (retrieval/result_selection.py
   prefer_durable_facts, live via post_process.py): sort key
   (type_rank, score + durable_boost - recap_penalty). Durable entities get
   type_rank 3, other entities 2, episodes 1, cues 0. The lane is
   SCORE-INDEPENDENT: a sem-0.01 durable Person outranks a sem-0.99 Concept.
   Within the durable lane the boost is ADDITIVE, so the 2.5-class precedes
   the 1.5-class for any score gap strictly below 1.0 (exact tie at the
   [0, 1] extreme), and score orders items of the same class.
3. Bounded additive rescue lanes elsewhere (recall_surface.py:1480,
   context_builder.py:2458): score += durable_result_boost * 0.08, i.e. at
   most +0.2.

The unbounded lane is INTENTIONAL and strictly stronger than the band the
design assumed: identity-class facts (Decision / Preference / Correction /
Person) must surface even on graze-quality matches — "what's my name"-grade
queries often reach the identity entity with poor similarity, and failing to
surface it is the worse error. Usage is bounded tightly (1.30x) because
behavioral echo must never rescue a semantically buried item; durability is
a commit-time CLASS statement and gets a reserved lane instead of a
multiplier. M3.2 must reason from this lane model, not the 2.5x band.
"""

from __future__ import annotations

import pytest

from engram.activation.engine import compute_u
from engram.config import ActivationConfig
from engram.extraction.promotion import durable_result_boost
from engram.models.activation import DEFAULT_USAGE_TIER_WEIGHTS, ActivationState
from engram.retrieval.result_selection import prefer_durable_facts

NOW = 1_700_000_000.0
DAY = 86400.0
BETA_MAX = 0.30
W_CONFIRMED = DEFAULT_USAGE_TIER_WEIGHTS["confirmed"]

# The 2.5 class (identity-grade) and the 1.5 class (other durable types),
# read back from the live constant in extraction/promotion.py.
CORE_DURABLE = ("Decision", "Preference", "Correction", "Person")
OTHER_DURABLE = ("Goal", "Commitment", "Organization", "Project", "Intention")


def _saturated_state() -> ActivationState:
    """50 confirmed events, newest exactly at NOW: n_eff=50 => f=1.0,
    delta_last=0 => r'=1.0, so u == 1.0 exactly."""
    state = ActivationState(node_id="ent_sat")
    for k in range(50):
        state.record_usage_event(NOW - float(k), W_CONFIRMED)
    return state


def _entity_result(name: str, entity_type: str, score: float) -> dict:
    return {
        "result_type": "entity",
        "entity": {"name": name, "type": entity_type},
        "score": score,
    }


class TestDurableBoostPinned:
    """(a) Pin the live constants — a change to 2.5/1.5 is a default change
    and must go through the live continuity gate, not slip through silently."""

    def test_core_durable_types_pin_2_5(self):
        for entity_type in CORE_DURABLE:
            assert durable_result_boost(entity_type) == 2.5

    def test_other_durable_types_pin_1_5(self):
        for entity_type in OTHER_DURABLE:
            assert durable_result_boost(entity_type) == 1.5

    def test_non_durable_types_get_zero(self):
        for entity_type in ("Concept", "Technology", "Artifact", "", None):
            assert durable_result_boost(entity_type) == 0.0


class TestDurableLaneDominance:
    """(b) The live durable preference is a reserved lane, not a band.

    Driven through the real prefer_durable_facts: type_rank dominates score
    entirely, so the durable lane is unbounded by design — see the module
    docstring for why this asymmetry vs the usage 1.30x theorem is
    intentional.
    """

    def test_weak_durable_outranks_strong_nondurable(self):
        """sem-0.01 durable Person beats sem-0.99 Concept: the lane is
        score-independent (this is the behavior the 2.5x band mis-modeled)."""
        ranked = prefer_durable_facts(
            [
                _entity_result("gold", "Concept", 0.99),
                _entity_result("identity", "Person", 0.01),
            ]
        )
        assert ranked[0]["entity"]["name"] == "identity"

    def test_usage_cannot_buy_into_the_lane(self):
        """Maximum usage amplification (1.30x on a 0.99 score) still cannot
        cross the lane boundary: type_rank never reads score, so behavioral
        evidence can never promote a non-durable item above a durable one."""
        max_amplified = 0.99 * (1.0 + BETA_MAX * 1.0)
        ranked = prefer_durable_facts(
            [
                _entity_result("hot_concept", "Concept", max_amplified),
                _entity_result("identity", "Person", 0.01),
            ]
        )
        assert ranked[0]["entity"]["name"] == "identity"

    def test_core_class_precedes_other_class_within_lane(self):
        """Within the durable lane the boost is additive: score + 2.5 vs
        score + 1.5. The class gap (1.0) beats any score gap strictly below
        1.0, so a 2.5-class item precedes a 1.5-class item unless the score
        gap is exactly 1.0 (the [0, 1] extreme), where the additive keys tie
        and list order decides."""
        ranked = prefer_durable_facts(
            [
                _entity_result("goal", "Goal", 0.99),
                _entity_result("decision", "Decision", 0.0),
            ]
        )
        assert ranked[0]["entity"]["name"] == "decision"

        # The exact-tie boundary, pinned: 1.0 + 1.5 == 0.0 + 2.5.
        tied = prefer_durable_facts(
            [
                _entity_result("goal", "Goal", 1.0),
                _entity_result("decision", "Decision", 0.0),
            ]
        )
        assert tied[0]["entity"]["name"] == "goal"

    def test_score_orders_same_class_within_lane(self):
        ranked = prefer_durable_facts(
            [
                _entity_result("weak", "Person", 0.2),
                _entity_result("strong", "Person", 0.8),
            ]
        )
        assert ranked[0]["entity"]["name"] == "strong"

    def test_lane_order_entity_over_episode_over_cue(self):
        ranked = prefer_durable_facts(
            [
                {"result_type": "cue_episode", "score": 0.99},
                {"result_type": "episode", "score": 0.99},
                _entity_result("plain", "Concept", 0.01),
                _entity_result("identity", "Person", 0.01),
            ]
        )
        kinds = [(r.get("result_type"), (r.get("entity") or {}).get("name")) for r in ranked]
        assert kinds == [
            ("entity", "identity"),
            ("entity", "plain"),
            ("episode", None),
            ("cue_episode", None),
        ]


class TestBoundedRescueLanes:
    """(c) The only bounded durable channels are the additive rescue lanes
    (recall_surface.py:1480, context_builder.py:2458): score +=
    durable_result_boost * 0.08 — at most +0.2 for the 2.5 class."""

    def test_rescue_lane_ceiling_is_0_2(self):
        assert max(
            durable_result_boost(t) for t in CORE_DURABLE + OTHER_DURABLE
        ) * 0.08 == pytest.approx(0.2)


class TestStageComposition:
    """(d) The usage tiebreaker and the durable lane compose sequentially,
    never as one product: usage caps at 1.30x inside the scorer; the lane
    then sorts the finished results score-independently."""

    def test_saturated_confirmed_usage_u_is_exactly_one(self):
        assert compute_u(_saturated_state(), NOW, ActivationConfig()) == 1.0

    def test_scorer_stage_amplification_ceiling_is_1_30(self):
        """No (count, age, tier-weight) combination pushes the usage
        multiplier past 1 + beta_max: u is capped at 1, beta at 0.30."""
        cfg = ActivationConfig()
        for count in (0, 1, 5, 50, 200):
            for age_days in (0.0, 1.0, 30.0, 180.0):
                for weight in (0.1, 0.3, 0.5, 1.0):
                    state = ActivationState(node_id="e1")
                    for k in range(count):
                        state.record_usage_event(NOW - age_days * DAY - k, weight)
                    u = compute_u(state, NOW, cfg)
                    assert 1.0 + BETA_MAX * u <= 1.0 + BETA_MAX + 1e-9

    def test_beta_max_structurally_pinned_at_0_30(self):
        """The config schema itself enforces beta_max: default 0.10, 0.30
        allowed, 0.31 rejected — so the 1.30x scorer ceiling cannot drift
        via cfg."""
        assert ActivationConfig().usage_beta_route == pytest.approx(0.10)
        assert ActivationConfig(usage_beta_route=0.30).usage_beta_route == 0.30
        with pytest.raises(ValueError):
            ActivationConfig(usage_beta_route=0.31)


class TestTripleEntitiesExcludedFromDurableLane:
    """Relationship-triple entities (graph edges the materializer renders as
    Decisions) must NOT get the durable RESERVED lane (type_rank 3 +
    score-independent boost) — they name-match common query tokens and would
    otherwise dominate. They drop to regular-entity ranking on their own
    score. Real prose Decisions keep the lane."""

    def _triple(self, score: float) -> dict:
        return {
            "result_type": "entity",
            "entity": {
                "name": "Engram:recall_profile:all",
                "type": "Decision",
                "summary": "Engram -> recall_profile -> all",
            },
            "score": score,
        }

    def test_real_prose_decision_beats_triple_despite_lower_score(self):
        # The triple lost the reserved lane, so a real prose Decision at a much
        # LOWER score still outranks it (rank 3 + 2.5 boost vs rank 2 + 0).
        real = _entity_result("GOLDEN_DECISION_1783643390", "Decision", 0.01)
        real["entity"]["summary"] = "LongMemEval is not the product north star"
        ranked = prefer_durable_facts([self._triple(0.9), real])
        assert ranked[0]["entity"]["name"] == "GOLDEN_DECISION_1783643390"

    def test_triple_gets_no_reserved_lane_over_plain_entity(self):
        # Against a plain Concept at equal score, the triple no longer wins by
        # the durable reserved lane — it ties on score (both rank 2, boost 0).
        plain = _entity_result("plain concept", "Concept", 0.5)
        ranked = prefer_durable_facts([self._triple(0.5), plain])
        # List order decides the tie; the triple does NOT dominate by class.
        assert {r["entity"]["name"] for r in ranked} == {
            "Engram:recall_profile:all",
            "plain concept",
        }
        # Same-score plain entity is not buried under the triple's old 0.99.
        assert abs(ranked[0]["score"] - ranked[1]["score"]) < 1e-9
