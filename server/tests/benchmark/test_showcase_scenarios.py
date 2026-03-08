"""Scenario catalog tests for the showcase benchmark."""

from engram.benchmark.showcase.scenarios import build_showcase_scenarios


def test_full_showcase_catalog_has_thirteen_scenarios():
    scenarios = build_showcase_scenarios(mode="full", seed=7)
    assert len(scenarios) == 13
    assert {scenario.id for scenario in scenarios} == {
        "cue_delayed_relevance",
        "temporal_override",
        "negation_correction",
        "open_loop_recovery",
        "prospective_trigger",
        "cross_cluster_association",
        "latent_open_loop_cue",
        "multi_session_continuity",
        "context_budget_compression",
        "meta_contamination_resistance",
        "selective_extraction_efficiency",
        "correction_chain",
        "summary_drift_resistance",
    }


def test_quick_mode_uses_flagship_subset():
    scenarios = build_showcase_scenarios(mode="quick", seed=7)
    assert [scenario.id for scenario in scenarios] == [
        "cue_delayed_relevance",
        "temporal_override",
        "prospective_trigger",
        "cross_cluster_association",
    ]


def test_scale_mode_focuses_long_horizon_scenarios():
    scenarios = build_showcase_scenarios(mode="scale", seed=7)
    assert {scenario.id for scenario in scenarios} == {
        "cue_delayed_relevance",
        "open_loop_recovery",
        "multi_session_continuity",
        "context_budget_compression",
        "selective_extraction_efficiency",
        "summary_drift_resistance",
    }


def test_scenarios_expose_answer_and_budget_metadata():
    scenarios = {
        scenario.id: scenario
        for scenario in build_showcase_scenarios(mode="quick", seed=7)
    }

    cue = scenarios["cue_delayed_relevance"]
    assert cue.answer_task is not None
    assert cue.gold_answer == {"subject": "cedar cache migration", "needed": "one extra smoke test"}
    assert cue.budget_profile.retrieval_limit > 0

    temporal = scenarios["temporal_override"]
    probe = temporal.probes[0]
    assert probe.disallowed_result_types == ["episode", "cue_episode"]
    assert probe.historical_evidence_allowed is False
