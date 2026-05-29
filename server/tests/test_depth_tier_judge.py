"""Unit tests for the strict depth-tier fact-presence judge (no network).

These confirm the judge is ANSWER-QUALITY, not session-recall in disguise:
  * a vague-but-topical answer (right entity, missing the specific value) FAILS,
  * a stale-value answer FAILS the current_value forbidden/newest-wins check,
  * synthesis scores coverage and guards against false-recall.
Also covers the paired-stats helpers (deterministic, seeded).
"""

from __future__ import annotations

from engram.benchmark.depth.judge import (
    delta_bootstrap_ci,
    judge_current_value,
    judge_multi_hop,
    judge_synthesis,
    mcnemar_p,
    pass_rate_bootstrap_ci,
)

# --------------------------------------------------------------------------- #
# multi_hop: gold present AND forbidden absent.                                #
# --------------------------------------------------------------------------- #


def test_multi_hop_passes_when_specific_gold_present():
    ev = ["Great news on the team — Dana moved up to Staff Engineer this week."]
    v = judge_multi_hop("q1", ev, answer="Staff Engineer", forbidden=["teammate"])
    assert v.passed
    assert v.gold_present
    assert not v.forbidden_hit


def test_multi_hop_fails_on_vague_but_topical_answer():
    # Right entity (Dana) and right topic (job/team) but the SPECIFIC value
    # ("Staff Engineer") is absent. A lenient/topical judge would pass this; the
    # strict judge must FAIL it — this is the core trustworthiness property.
    ev = ["Dana is on my team and recently got some good career news at work."]
    v = judge_multi_hop("q1", ev, answer="Staff Engineer", forbidden=["teammate"])
    assert not v.passed
    assert not v.gold_present
    assert v.missing_facts == ["Staff Engineer"]


def test_multi_hop_accepted_surface_forms():
    ev = ["We are standardizing all new services on Go from now on."]
    v = judge_multi_hop(
        "q5", ev, answer="Go", accepted_forms=["standardizing all new services on Go"]
    )
    assert v.passed


def test_multi_hop_fails_when_forbidden_present():
    ev = ["Dana moved up to Staff Engineer.", "Dana is my teammate on Atlas."]
    v = judge_multi_hop("q1", ev, answer="Staff Engineer", forbidden=["teammate"])
    assert v.gold_present
    assert v.forbidden_hit
    assert not v.passed


# --------------------------------------------------------------------------- #
# current_value: newest-wins.                                                  #
# --------------------------------------------------------------------------- #


def test_current_value_passes_when_gold_outranks_stale():
    # Newest-first ordering: the gold (Staff Engineer) precedes the stale value.
    ev = [
        "[2024/03/20] Dana moved up to Staff Engineer this week.",
        "[2024/01/08] working alongside a teammate named Dana.",
    ]
    v = judge_current_value(
        "q11", ev, answer="Staff Engineer",
        forbidden=["teammate named Dana"], evidence_is_ordered=True,
    )
    assert v.passed
    assert v.gold_present


def test_current_value_fails_when_stale_value_outranks_gold():
    # Stale value ("teammate") ranks ABOVE the gold -> newest-wins violated.
    ev = [
        "[2024/01/08] working alongside a teammate named Dana.",
        "[2024/03/20] Dana moved up to Staff Engineer this week.",
    ]
    v = judge_current_value(
        "q11", ev, answer="Staff Engineer",
        forbidden=["teammate named Dana"], evidence_is_ordered=True,
    )
    assert v.gold_present
    assert v.forbidden_hit
    assert not v.passed
    assert "newest-wins" in v.notes


def test_current_value_fails_on_stale_only_answer():
    # Only the stale value is retrieved; the latest value is absent.
    ev = ["[2024/01/08] working alongside a teammate named Dana."]
    v = judge_current_value(
        "q11", ev, answer="Staff Engineer",
        forbidden=["teammate named Dana"], evidence_is_ordered=True,
    )
    assert not v.gold_present
    assert not v.passed


# --------------------------------------------------------------------------- #
# synthesis: coverage + false-recall guard.                                    #
# --------------------------------------------------------------------------- #


def test_synthesis_passes_on_full_coverage():
    ev = [
        "First day. Got staffed on Project Atlas alongside Dana.",
        "Dana moved up to Staff Engineer.",
        "Dana used to be at Cobalt Robotics before all this.",
    ]
    v = judge_synthesis(
        "q12", ev,
        required_facts=["Project Atlas", "Staff Engineer", "Cobalt Robotics"],
        forbidden_facts=["Brightwave Capital"],
    )
    assert v.passed
    assert v.coverage == 1.0
    assert not v.false_recall


def test_synthesis_fails_below_threshold():
    ev = ["Got staffed on Project Atlas alongside Dana."]
    v = judge_synthesis(
        "q12", ev,
        required_facts=["Project Atlas", "Staff Engineer", "Cobalt Robotics"],
        coverage_threshold=0.66,
    )
    assert abs(v.coverage - (1 / 3)) < 1e-9
    assert not v.passed
    assert "Staff Engineer" in v.missing_facts


def test_synthesis_false_recall_guard_fails_pass():
    # Full coverage but a forbidden (different-entity) fact leaked in -> fail.
    ev = [
        "Project Atlas with Dana.",
        "Dana moved up to Staff Engineer.",
        "Dana used to be at Cobalt Robotics.",
        "Sofia works at Brightwave Capital.",
    ]
    v = judge_synthesis(
        "q12", ev,
        required_facts=["Project Atlas", "Staff Engineer", "Cobalt Robotics"],
        forbidden_facts=["Brightwave Capital"],
    )
    assert v.coverage == 1.0
    assert v.false_recall
    assert not v.passed


# --------------------------------------------------------------------------- #
# paired statistics (deterministic).                                           #
# --------------------------------------------------------------------------- #


def test_mcnemar_all_discordant_one_direction():
    # core fails all, depth passes all -> strongest evidence for an effect.
    core = [False] * 8
    depth = [True] * 8
    p = mcnemar_p(core, depth)
    assert 0.0 <= p <= 0.05


def test_mcnemar_no_discordance_is_one():
    core = [True, False, True]
    depth = [True, False, True]
    assert mcnemar_p(core, depth) == 1.0


def test_bootstrap_ci_is_deterministic_and_ordered():
    passes = [True, True, False, True, False, True, True, False]
    ci_a = pass_rate_bootstrap_ci(passes, seed=7)
    ci_b = pass_rate_bootstrap_ci(passes, seed=7)
    assert ci_a == ci_b  # seeded => reproducible
    assert ci_a[0] <= ci_a[1]


def test_delta_ci_excludes_zero_for_strong_effect():
    core = [False] * 10
    depth = [True] * 10
    lo, hi = delta_bootstrap_ci(core, depth, seed=3)
    assert lo > 0  # CI strictly above 0 => a win
