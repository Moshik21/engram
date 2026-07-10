"""Unit tests for the standing measurement-rig aggregation math (no network).

These confirm the repeated-run aggregation is DETERMINISTIC: for fixed per-run
inputs the per-class mean / std, the pooled paired-delta bootstrap CI / McNemar,
and the per-query verdict-flip counts are exactly reproducible. The flip count is
a first-class output — it is the rig's residual-nondeterminism meter, so it must
be an exact disagreement count (not a sampled statistic).
"""

from __future__ import annotations

from engram.benchmark.depth.judge import aggregate_repeated_runs, verdict_flip_count

# --------------------------------------------------------------------------- #
# verdict_flip_count: exact disagreement across runs.                          #
# --------------------------------------------------------------------------- #


def test_flip_count_zero_when_all_runs_agree():
    runs = [
        {"q1": True, "q2": False, "q3": True},
        {"q1": True, "q2": False, "q3": True},
        {"q1": True, "q2": False, "q3": True},
    ]
    out = verdict_flip_count(runs)
    assert out["flips"] == 0
    assert out["unstable_qids"] == []
    assert out["n_queries"] == 3


def test_flip_count_counts_only_disagreeing_qids():
    # q2 flips (False/True/False), q3 flips (True/True/False); q1 stable.
    runs = [
        {"q1": True, "q2": False, "q3": True},
        {"q1": True, "q2": True, "q3": True},
        {"q1": True, "q2": False, "q3": False},
    ]
    out = verdict_flip_count(runs)
    assert out["flips"] == 2
    assert out["unstable_qids"] == ["q2", "q3"]  # sorted, deterministic
    assert out["n_queries"] == 3


def test_flip_count_handles_qid_present_in_only_some_runs():
    # q9 appears once -> a single observed verdict -> NOT a flip.
    runs = [
        {"q1": True, "q9": True},
        {"q1": True},
    ]
    out = verdict_flip_count(runs)
    assert out["flips"] == 0
    assert out["n_queries"] == 2


def test_flip_count_empty_input():
    out = verdict_flip_count([])
    assert out == {"flips": 0, "unstable_qids": [], "n_queries": 0}


# --------------------------------------------------------------------------- #
# aggregate_repeated_runs: mean / std / CI / flips, all deterministic.         #
# --------------------------------------------------------------------------- #


def _run(core_rate, depth_rate, core_v, depth_v):
    return {
        "core_pass_rate": core_rate,
        "depth_pass_rate": depth_rate,
        "core_verdicts": core_v,
        "depth_verdicts": depth_v,
    }


def test_aggregate_mean_std_exact_for_known_inputs():
    # Two runs, identical verdicts (no flips). core rates 0.0/0.0, depth 1.0/1.0.
    runs = [
        _run(0.0, 1.0, {"q1": False, "q2": False}, {"q1": True, "q2": True}),
        _run(0.0, 1.0, {"q1": False, "q2": False}, {"q1": True, "q2": True}),
    ]
    agg = aggregate_repeated_runs(runs)
    assert agg["n_runs"] == 2
    assert agg["n_pooled"] == 4  # 2 qids x 2 runs
    assert agg["core_pass_rate_mean"] == 0.0
    assert agg["core_pass_rate_std"] == 0.0
    assert agg["depth_pass_rate_mean"] == 1.0
    assert agg["depth_pass_rate_std"] == 0.0
    assert agg["delta_mean"] == 1.0
    assert agg["delta_std"] == 0.0
    assert agg["core_flips"] == 0
    assert agg["depth_flips"] == 0


def test_aggregate_population_std_matches_hand_calc():
    # depth rates 1.0 and 0.0 -> mean 0.5, population std (ddof=0) = 0.5.
    runs = [
        _run(0.0, 1.0, {"q1": False}, {"q1": True}),
        _run(0.0, 0.0, {"q1": False}, {"q1": False}),
    ]
    agg = aggregate_repeated_runs(runs)
    assert agg["depth_pass_rate_mean"] == 0.5
    assert agg["depth_pass_rate_std"] == 0.5  # sqrt(((1-.5)^2+(0-.5)^2)/2) = 0.5
    assert agg["delta_mean"] == 0.5
    assert agg["delta_std"] == 0.5
    # q1's depth verdict flips across the two runs -> first-class flip output.
    assert agg["depth_flips"] == 1
    assert agg["depth_unstable_qids"] == ["q1"]
    assert agg["core_flips"] == 0


def test_aggregate_pooled_ci_excludes_zero_for_strong_effect():
    # Every pooled paired observation is core-FAIL / depth-PASS -> CI strictly > 0.
    runs = [
        _run(0.0, 1.0, {f"q{i}": False for i in range(5)}, {f"q{i}": True for i in range(5)}),
        _run(0.0, 1.0, {f"q{i}": False for i in range(5)}, {f"q{i}": True for i in range(5)}),
    ]
    agg = aggregate_repeated_runs(runs)
    assert agg["n_pooled"] == 10
    assert agg["delta_ci_95"][0] > 0
    assert agg["ci_excludes_zero"] is True
    assert agg["mcnemar_p"] <= 0.05  # 10 discordant pairs one direction


def test_aggregate_is_fully_deterministic_across_calls():
    runs = [
        _run(
            0.5,
            0.75,
            {"q1": True, "q2": False, "q3": True, "q4": False},
            {"q1": True, "q2": True, "q3": True, "q4": False},
        ),
        _run(
            0.25,
            0.75,
            {"q1": False, "q2": False, "q3": True, "q4": False},
            {"q1": True, "q2": True, "q3": True, "q4": False},
        ),
    ]
    a = aggregate_repeated_runs(runs, seed=99)
    b = aggregate_repeated_runs(runs, seed=99)
    assert a == b  # byte-identical aggregation for fixed inputs + seed


def test_aggregate_empty_runs_safe():
    agg = aggregate_repeated_runs([])
    assert agg["n_runs"] == 0
    assert agg["n_pooled"] == 0
    assert agg["core_pass_rate_mean"] == 0.0
    assert agg["ci_excludes_zero"] is False
    assert agg["core_flips"] == 0


# --------------------------------------------------------------------------- #
# _headline_filter: verdict keys MUST be persona-qualified.                    #
# Regression for the qid-namespace collision: every persona reuses q1..q6 /    #
# q11..q15, so a bare-qid key collapsed cross-persona queries (last-writer-     #
# wins), silently discarding real per-persona depth passes from the headline   #
# pool AND from the repeat aggregator / flip count.                            #
# --------------------------------------------------------------------------- #


def _mh_query(qid, *, core_pass, depth_pass):
    return {
        "qid": qid,
        "type": "multi_hop",
        "core_pass": core_pass,
        "depth_pass": depth_pass,
        "core_false_recall": False,
        "depth_false_recall": False,
        "gate": {"bridge_in_graph": True, "bridge_linked_to_answer": True},
    }


def test_headline_filter_keys_are_persona_qualified():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from benchmark_depth_tier import _headline_filter

    # Two personas BOTH expose multi_hop q1; one is a depth WIN, the other a
    # depth FAIL. A bare-qid key would keep only the last (collapsing to 1
    # entry); persona-qualified keys must keep BOTH.
    all_personas = [
        {"persona_id": "alpha", "queries": [_mh_query("q1", core_pass=False, depth_pass=True)]},
        {"persona_id": "beta", "queries": [_mh_query("q1", core_pass=False, depth_pass=False)]},
    ]
    f = _headline_filter(all_personas, "multi_hop")
    assert len(f["depth_verdicts"]) == 2, "cross-persona q1 must NOT collapse"
    assert set(f["depth_verdicts"]) == {"alpha:q1", "beta:q1"}
    # The depth WIN from alpha survives the headline pool.
    assert f["depth_verdicts"]["alpha:q1"] is True
    assert f["depth_verdicts"]["beta:q1"] is False
    assert sum(f["depth_verdicts"].values()) == 1  # 1 of 2 depth passes (was 0 under the bug)
