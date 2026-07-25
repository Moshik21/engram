"""Contract tests for the recall meter (task #18).

The meter exists because ``engram battery`` cannot resolve a +/-1-answer
retrieval change: its scoring rule requires every token of an answer group to
land inside ONE top-3 row, so a two-source answer is a MISS by construction.

``test_multi_source_answer_scores_hit`` is the CANARY of this file. If the
union rule ever collapses back to the battery's one-row rule, the meter becomes
a second copy of the instrument it replaces and every graph result measured with
it is void. That test must fail loudly if the mechanism dies, and it was proved
capable of failing (neuter -> red -> restore -> green) before it was committed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from engram.evaluation import battery
from engram.evaluation.meter import (
    CAPTURE_SCHEMA,
    RUNTIME_FAST_PATH,
    _is_cache_served,
    chi2_quantile,
    fetch_cache_provenance,
    format_meter_report,
    load_rig,
    min_runs_per_arm,
    minimal_cover,
    read_cache_provenance,
    result_texts,
    score_capture,
    score_rows,
    sigma_upper_confidence,
)

RIG_PATH = Path(__file__).resolve().parent / "rigs" / "recall_meter_rig.json"


def _payload(*row_texts: str) -> dict:
    """A recall payload shaped like /api/knowledge/recall."""
    return {
        "items": [
            {"resultType": "episode", "episode": {"id": f"ep_{i}", "content": text}}
            for i, text in enumerate(row_texts)
        ],
        "lifecycle": {"fallbackStatus": "not_run"},
        "budget": {"degraded": False, "timeout": False},
        "status": "ok",
    }


GAP_S = 400.0  # above the 300s packet-cache TTL, so repeats are independent


def _capture(runs: list[list[dict]], *, questions: list[dict], limit: int = 3) -> dict:
    """Runs spaced GAP_S apart unless a row carries its own ``atS``."""
    captures = []
    for i, rows in enumerate(runs):
        spaced = [{**row, "atS": row.get("atS", i * GAP_S)} for row in rows]
        captures.append({"run": i, "questions": spaced})
    return {
        "schema": CAPTURE_SCHEMA,
        "serverUrl": "http://test",
        "limit": limit,
        "runs": len(runs),
        "rig": {"questions": questions},
        "captures": captures,
    }


def _row(qid: str, rows: list[str], *, lane: str = "not_run", **kw) -> dict:
    base = {
        "id": qid,
        "rows": rows,
        "lane": lane,
        "cacheServed": False,
        "status": "ok",
        "degraded": False,
        "timeout": False,
        "latencyMs": 100.0,
        "error": None,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# THE CANARY: multi-source answers must be scoreable
# ---------------------------------------------------------------------------


class TestMultiSourceScoring:
    def test_multi_source_answer_scores_hit(self):
        """An answer assembled from TWO rows is a HIT for the meter.

        This is the single most important property of the instrument. The
        battery scores this exact case MISS, which is why it is blind to the
        multi-hop phenomenon the graph experiment exists to detect.
        """
        rows = [
            "the vector index was empty brain-wide",
            "unrelated chatter about the dashboard",
            "the BM25 breaker fired on every timeout",
        ]
        scored = score_rows([["vector index", "BM25 breaker"]], rows, k=3, max_sources=2)
        assert scored["hitUnion"] is True, "multi-source rule is DEAD"
        assert scored["hitSingle"] is False, "fixture is not actually multi-source"
        assert scored["sources"] == 2
        assert scored["coverRows"] == [0, 2]

    def test_battery_scores_the_same_case_miss(self):
        """The failure this instrument exists to fix, asserted on the battery."""
        payload = _payload(
            "the vector index was empty brain-wide",
            "unrelated chatter about the dashboard",
            "the BM25 breaker fired on every timeout",
        )
        question = {"id": "x", "q": "?", "expected_tokens": [["vector index", "BM25 breaker"]]}
        result = battery.score_question(question, battery.top3_result_texts(payload))
        assert result["hit"] is False

    def test_cover_is_bounded(self):
        """Three-row scatter is coincidence, not an assembled answer."""
        rows = ["alpha here", "beta here", "gamma here"]
        assert minimal_cover(["alpha", "beta", "gamma"], rows, 2) is None
        assert minimal_cover(["alpha", "beta", "gamma"], rows, 3) == (0, 1, 2)

    def test_cover_is_minimal_and_prefers_top_rows(self):
        rows = ["alpha beta", "alpha", "beta"]
        assert minimal_cover(["alpha", "beta"], rows, 2) == (0,)
        rows2 = ["alpha", "beta", "alpha beta"]
        assert minimal_cover(["alpha", "beta"], rows2, 2) == (2,)

    def test_missing_token_is_never_covered(self):
        assert minimal_cover(["alpha", "absent"], ["alpha"], 2) is None

    def test_empty_group_is_not_a_vacuous_hit(self):
        """Deliberate divergence from the battery.

        ``battery.group_contained(text, [])`` is ``all([])`` = True, so an empty
        group scores a free HIT there. The meter refuses it.
        """
        assert minimal_cover([], ["anything"], 2) is None
        assert score_rows([[]], ["anything"], k=3, max_sources=2)["hitUnion"] is False

    def test_k_truncates_before_scoring(self):
        rows = ["alpha", "beta", "filler", "gamma"]
        assert score_rows([["alpha", "gamma"]], rows, k=3, max_sources=2)["hitUnion"] is False
        assert score_rows([["alpha", "gamma"]], rows, k=4, max_sources=2)["hitUnion"] is True


# ---------------------------------------------------------------------------
# The battery rule must be reproduced EXACTLY so the head-to-head is honest
# ---------------------------------------------------------------------------


class TestBatteryEquivalence:
    @pytest.mark.parametrize(
        "rows,groups",
        [
            (["alpha beta"], [["alpha", "beta"]]),
            (["alpha", "beta"], [["alpha", "beta"]]),
            (["alpha", "beta"], [["alpha"], ["beta"]]),
            (["nothing"], [["alpha", "beta"]]),
            (["ALPHA BeTa"], [["alpha", "beta"]]),
            (["a b", "c", "d"], [["a", "b"], ["c", "d"]]),
        ],
    )
    def test_hit_single_equals_battery_hit(self, rows, groups):
        payload = _payload(*rows)
        question = {"id": "x", "q": "?", "expected_tokens": groups}
        expected = battery.score_question(question, battery.top3_result_texts(payload))["hit"]
        actual = score_rows(groups, result_texts(payload, 3), k=3, max_sources=2)["hitSingle"]
        assert actual is expected

    def test_result_texts_matches_battery_extraction(self):
        payload = _payload("one", "two", "three", "four")
        assert result_texts(payload, 3) == battery.top3_result_texts(payload)


# ---------------------------------------------------------------------------
# The minimum-N derivation
# ---------------------------------------------------------------------------


class TestResolution:
    @pytest.mark.parametrize(
        "df,expected",
        [
            (1, 0.003932),
            (2, 0.102587),
            (5, 1.145476),
            (9, 3.325113),
            (19, 10.117013),
            (29, 17.708366),
        ],
    )
    def test_chi2_quantile_matches_published_table(self, df, expected):
        """Computed, not table-looked-up — the table is the CHECK."""
        assert chi2_quantile(0.05, df) == pytest.approx(expected, abs=1e-4)

    def test_min_runs_matches_the_closed_form(self):
        # N >= 2 * sd^2 * (1.96 + 0.8416)^2 / delta^2
        for sd in (0.5, 1.0, 1.5, 2.0):
            expected = math.ceil(2.0 * sd**2 * (1.959963985 + 0.841621234) ** 2)
            assert min_runs_per_arm(sd, 1.0) == max(2, expected)

    def test_min_runs_scales_with_effect_size(self):
        assert min_runs_per_arm(1.0, 2.0) < min_runs_per_arm(1.0, 1.0)

    def test_zero_variance_still_needs_two_runs(self):
        assert min_runs_per_arm(0.0, 1.0) == 2

    def test_sigma_upper_bound_exceeds_point_estimate(self):
        upper = sigma_upper_confidence(1.0, 10)
        assert upper is not None and upper > 1.0
        assert sigma_upper_confidence(1.0, 40) < upper  # tightens with n
        assert sigma_upper_confidence(1.0, 1) is None

    def test_zero_observed_variance_does_not_certify_zero_variance(self):
        """The over-claim this guard exists to stop.

        Normal theory multiplies sd=0 by a constant and returns 0, which would
        let 8 identical runs certify "N >= 2 runs/arm". Rule of three: an event
        never seen in n trials still has a 95% upper rate of 3/n.
        """
        assert sigma_upper_confidence(0.0, 8) == pytest.approx(math.sqrt(3 / 8))
        assert min_runs_per_arm(sigma_upper_confidence(0.0, 8), 1.0) >= 6
        # and it tightens as runs accumulate
        assert sigma_upper_confidence(0.0, 40) < sigma_upper_confidence(0.0, 8)


# ---------------------------------------------------------------------------
# Honest refusal
# ---------------------------------------------------------------------------


QUESTIONS = [
    {"id": "q1", "q": "one", "expected_tokens": [["alpha"]]},
    {"id": "q2", "q": "two", "expected_tokens": [["beta"]]},
]


class TestRefusals:
    def test_refuses_below_min_runs(self):
        capture = _capture(
            [[_row("q1", ["alpha"]), _row("q2", ["beta"])]],
            questions=QUESTIONS,
        )
        report = score_capture(capture)
        assert report["status"] == "unresolved"
        assert report["comparison"]["usable"] is False
        assert any("run(s) captured" in r for r in report["refusals"])

    def test_refuses_on_error_storm(self):
        runs = []
        for _ in range(6):
            runs.append(
                [
                    _row("q1", [], error="TimeoutError: boom"),
                    _row("q2", ["beta"]),
                ]
            )
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["status"] == "degraded"
        assert report["comparison"]["usable"] is False
        assert any("errored" in r for r in report["refusals"])

    def test_refuses_when_variance_exceeds_the_run_budget(self):
        """High per-run variance => N is not enough, and it says the number."""
        runs = []
        for i in range(4):
            hit = i % 2 == 0
            runs.append(
                [
                    _row("q1", ["alpha"] if hit else ["nope"]),
                    _row("q2", ["beta"] if hit else ["nope"]),
                ]
            )
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["status"] == "unresolved"
        assert report["comparison"]["minRunsPerArm"] is not None
        assert report["comparison"]["minRunsPerArm"] > 4
        assert any("cannot resolve" in r for r in report["refusals"])

    def test_resolves_when_stable(self):
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(12)]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["status"] == "resolved"
        assert report["comparison"]["usable"] is True
        assert report["score"]["sdUnion"] == 0.0
        assert report["score"]["meanUnion"] == 2.0

    def test_formatter_withholds_headline_when_unresolved(self):
        capture = _capture(
            [[_row("q1", ["alpha"]), _row("q2", ["beta"])]],
            questions=QUESTIONS,
        )
        text = format_meter_report(score_capture(capture))
        assert "NO HEADLINE SCORE" in text
        assert "# Recall meter: RESOLVED" not in text
        assert "REFUSAL:" in text

    def test_formatter_prints_headline_when_resolved(self):
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(12)]
        text = format_meter_report(score_capture(_capture(runs, questions=QUESTIONS)))
        assert "RESOLVED" in text
        assert "NO HEADLINE SCORE" not in text

    def test_no_rows_is_excluded_not_scored_zero(self):
        """AUDIT-11 shape: 'we did not look' must not report as 'nothing there'."""
        runs = [[_row("q1", []), _row("q2", ["beta"])] for _ in range(4)]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        q1 = next(r for r in report["questions"] if r["id"] == "q1")
        assert q1["excluded"] == "no_usable_runs"
        assert q1["pHitUnion"] is None
        assert report["score"]["questionsScored"] == 1
        assert any("excluded" in r for r in report["refusals"])


# ---------------------------------------------------------------------------
# Lane attribution and the shuffled control
# ---------------------------------------------------------------------------


class TestCacheGuard:
    """The packet cache makes repeated identical queries LOOK deterministic.

    Live evidence 2026-07-24: 12 back-to-back passes returned sd=0.0, with
    ``cache_satisfied`` serving 24/168 probes; the same rig 10 minutes earlier
    scored one answer higher. Within-block variance is a FLOOR on the real
    measurement noise, not an estimate of it, and an instrument that reports it
    as an estimate certifies a resolving power it does not have.
    """

    def test_cache_lane_is_detected_at_capture_time(self):
        """The detector itself, not just its effect.

        Found by neutering: with ``_is_cache_served`` forced to False the whole
        suite stayed green, because every other test sets ``cacheServed`` in the
        fixture. The detector runs only at capture time, so it needs its own
        test or it can die silently.
        """
        assert _is_cache_served("cache_satisfied", {}) is True
        assert _is_cache_served("not_run", {"skipReason": "cache_satisfied"}) is True
        assert _is_cache_served("not_run", {"skip_reason": "cache_satisfied"}) is True
        assert _is_cache_served("not_run", {}) is False
        assert _is_cache_served("durable_entity_first", {"skipReason": None}) is False

    def test_cache_served_probe_is_not_counted_as_a_sample(self):
        runs = []
        for i in range(4):
            runs.append(
                [
                    _row("q1", ["alpha"], lane="cache_satisfied", cacheServed=True),
                    _row("q2", ["beta"]),
                ]
            )
        report = score_capture(_capture(runs, questions=QUESTIONS))
        q1 = next(r for r in report["questions"] if r["id"] == "q1")
        assert q1["cacheServedRuns"] == 4
        assert q1["usableRuns"] == 0
        assert q1["excluded"] == "cache_served"
        assert report["probes"]["cacheServed"] == 4
        assert report["status"] == "unresolved"
        assert any("packet cache" in r for r in report["refusals"])

    def test_probes_inside_the_ttl_are_refused_as_non_independent(self):
        runs = [
            [
                _row("q1", ["alpha"], atS=i * 17.0),
                _row("q2", ["beta"], atS=i * 17.0),
            ]
            for i in range(6)
        ]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["status"] == "unresolved"
        assert set(report["cache"]["unspacedQuestions"]) == {"q1", "q2"}
        assert any("packet-cache TTL" in r for r in report["refusals"])

    def test_pre_guard_capture_is_refused_not_silently_scored(self):
        """A capture without timing/cache provenance cannot be certified.

        Old captures would otherwise sail through the guard reporting sd=0 —
        the guard would be present and inert, which is this project's dominant
        bug class.
        """
        legacy_rows = [
            [
                {"id": "q1", "rows": ["alpha"], "lane": "not_run", "error": None},
                {"id": "q2", "rows": ["beta"], "lane": "not_run", "error": None},
            ]
            for _ in range(4)
        ]
        capture = {
            "schema": CAPTURE_SCHEMA,
            "limit": 3,
            "runs": 4,
            "rig": {"questions": QUESTIONS},
            "captures": [{"run": i, "questions": rows} for i, rows in enumerate(legacy_rows)],
        }
        report = score_capture(capture)
        assert report["status"] == "unresolved"
        assert report["cache"]["untimedProbes"] == 8
        assert any("pre-guard capture" in r for r in report["refusals"])

    def test_properly_spaced_probes_are_accepted(self):
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(12)]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["cache"]["unspacedQuestions"] == []
        assert report["status"] == "resolved"

    def test_four_identical_runs_are_not_enough(self):
        """Rule of three: zero observed flips in 4 runs still bounds sd at 0.87."""
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(4)]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["status"] == "unresolved"
        assert report["comparison"]["minRunsPerArmConservative"] > 4


class TestLaneAndControl:
    def test_lane_is_recorded_per_question_per_run(self):
        runs = [
            [_row("q1", ["alpha"], lane="not_run"), _row("q2", ["beta"], lane="not_run")],
            [
                _row("q1", ["alpha"], lane="durable_entity_first"),
                _row("q2", ["beta"], lane="not_run"),
            ],
            [_row("q1", ["alpha"], lane="not_run"), _row("q2", ["beta"], lane="not_run")],
        ]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        q1 = next(r for r in report["questions"] if r["id"] == "q1")
        assert q1["lanes"] == {"not_run": 2, "durable_entity_first": 1}
        assert q1["laneStable"] is False
        q2 = next(r for r in report["questions"] if r["id"] == "q2")
        assert q2["laneStable"] is True

    def test_control_detects_a_promiscuous_rig(self):
        """Tokens that appear everywhere must show up as union false positives."""
        promiscuous = [
            {"id": "q1", "q": "one", "expected_tokens": [["the"]]},
            {"id": "q2", "q": "two", "expected_tokens": [["the"]]},
        ]
        runs = [[_row("q1", ["the alpha"]), _row("q2", ["the beta"])] for _ in range(3)]
        report = score_capture(_capture(runs, questions=promiscuous))
        assert report["control"]["trials"] == 6
        assert report["control"]["unionRate"] == 1.0

    def test_control_is_clean_on_a_discriminating_rig(self):
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(3)]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["control"]["unionRate"] == 0.0


# ---------------------------------------------------------------------------
# The rig itself
# ---------------------------------------------------------------------------


class TestRig:
    def test_rig_loads_and_is_well_formed(self):
        rig = load_rig(RIG_PATH)
        ids = [q["id"] for q in rig["questions"]]
        assert len(ids) == len(set(ids)), "duplicate question ids"
        for question in rig["questions"]:
            assert question["q"]
            assert question["expected_tokens"], f"{question['id']} has no answer groups"
            for group in question["expected_tokens"]:
                assert group, f"{question['id']} has an empty group (vacuous hit)"
            assert question["kind"] in {"single_source", "multi_source"}

    def test_single_source_questions_match_the_battery_verbatim(self):
        """The head-to-head is only honest if the questions are identical."""
        rig = load_rig(RIG_PATH)
        with open(battery.BATTERY_PATH, encoding="utf-8") as f:
            original = json.load(f)
        originals = {q["id"]: q for q in original["questions"]}
        carried = [q for q in rig["questions"] if q["kind"] == "single_source"]
        assert carried, "the meter rig carries no battery questions"
        for question in carried:
            source = originals[question["id"]]
            assert question["q"] == source["q"]
            assert question["expected_tokens"] == source["expected_tokens"]

    def test_multi_source_groups_span_two_battery_questions(self):
        """Provenance is machine-verified, not asserted in prose.

        Every token of a multi_source group must come from one of the two named
        battery questions, and a group must draw from BOTH — otherwise the
        question is single-source wearing a multi-source label, and the rig
        would silently stop testing the property the instrument exists for.
        """
        rig = load_rig(RIG_PATH)
        by_id = {q["id"]: q for q in rig["questions"]}
        multi = [q for q in rig["questions"] if q["kind"] == "multi_source"]
        assert multi, "the rig has no multi-source questions"
        for question in multi:
            parents = question["from"]
            assert len(parents) == 2
            token_sources = {
                p: {t for group in by_id[p]["expected_tokens"] for t in group} for p in parents
            }
            for group in question["expected_tokens"]:
                owners = set()
                for token in group:
                    matched = [p for p in parents if token in token_sources[p]]
                    assert matched, f"{question['id']}: token {token!r} is not battery ground truth"
                    owners.update(matched)
                assert owners == set(parents), (
                    f"{question['id']}: group {group} does not span both parents"
                )


# ---------------------------------------------------------------------------
# Cache provenance read from the SERVER, not assumed (ticket #29 / AUDIT-14)
# ---------------------------------------------------------------------------


class TestServerCacheProvenance:
    """AUDIT-14 shipped half a mitigation: the meter excluded cache-served
    probes, but the TTL it enforced was a compile-time constant and the key's
    identity was unknowable, so cache independence was *hoped for*. The server
    now reports both, and these tests are what make the meter use them.
    """

    def test_provenance_states_are_distinct(self):
        ok = read_cache_provenance(
            {"fingerprint": "abc123", "ttl_seconds": 45.0, "enabled": True, "key_schema": "pc2"}
        )
        assert ok["status"] == "ok"
        assert (ok["fingerprint"], ok["ttlSeconds"], ok["enabled"]) == ("abc123", 45.0, True)

        # A server that predates the fix must not be reported as verified.
        legacy = read_cache_provenance({"entry_count": 3, "fresh_count": 1})
        assert legacy["status"] == "unreported"
        assert legacy["fingerprint"] is None
        assert legacy["ttlSeconds"] is None

        assert read_cache_provenance(None)["status"] == "unreported"

    def test_camel_case_payload_is_accepted(self):
        """/api/knowledge/packet-cache camelises; /runtime does not."""
        camel = read_cache_provenance(
            {"fingerprint": "abc123", "ttlSeconds": 45.0, "enabled": False, "keySchema": "pc2"}
        )
        assert camel["status"] == "ok"
        assert camel["ttlSeconds"] == 45.0
        assert camel["enabled"] is False

    def test_unreachable_server_is_recorded_not_swallowed(self):
        def boom(url, timeout):
            raise OSError("connection refused")

        provenance = fetch_cache_provenance("http://test", fetch=boom)
        assert provenance["status"] == "unreachable"
        assert "connection refused" in provenance["detail"]
        assert provenance["ttlSeconds"] is None

    def test_provenance_is_read_off_the_runtime_fast_packet(self):
        seen = {}

        def fake(url, timeout):
            seen["url"] = url
            return {"stats": {"packetCache": {"fingerprint": "fp1", "ttl_seconds": 30.0}}}

        provenance = fetch_cache_provenance("http://test/", fetch=fake)
        assert seen["url"] == "http://test" + RUNTIME_FAST_PATH
        assert provenance["fingerprint"] == "fp1"

    def test_server_reported_ttl_beats_the_callers_assumption(self):
        """A 17s gap is under the 300s default but over the server's real 5s.

        Enforcing the assumption would refuse a capture that is in fact
        independent — and, in the other direction, would certify one that is
        not when a server runs a longer TTL than the default.
        """
        runs = [
            [_row("q1", ["alpha"], atS=i * 17.0), _row("q2", ["beta"], atS=i * 17.0)]
            for i in range(12)
        ]
        capture = _capture(runs, questions=QUESTIONS)
        capture["serverCache"] = {
            "status": "ok",
            "enabled": True,
            "ttlSeconds": 5.0,
            "fingerprint": "fp1",
            "keySchema": "pc2",
        }
        report = score_capture(capture)
        assert report["cache"]["ttlSeconds"] == 5.0
        assert report["cache"]["ttlSource"] == "server"
        assert report["cache"]["unspacedQuestions"] == []
        assert not any("TTL" in r for r in report["refusals"])
        assert report["status"] == "resolved"

    def test_a_verified_bypass_makes_fast_repeats_legitimate(self):
        """`recall_packet_cache_enabled=False` is the measurement bypass.

        Verified against the measured process, not the config file the rig
        believes it edited (§2.11). Without the check the meter would demand
        300s between passes — an hour for 12 runs — from a server that
        provably cannot replay a packet.
        """
        runs = [
            [_row("q1", ["alpha"], atS=i * 2.0), _row("q2", ["beta"], atS=i * 2.0)]
            for i in range(12)
        ]
        bypassed = _capture(runs, questions=QUESTIONS)
        bypassed["serverCache"] = {
            "status": "ok",
            "enabled": False,
            "ttlSeconds": 300.0,
            "fingerprint": "fp1",
            "keySchema": "pc2",
        }
        report = score_capture(bypassed)
        assert report["cache"]["bypassed"] is True
        assert report["cache"]["unspacedQuestions"] == []
        assert not any("TTL" in r for r in report["refusals"])
        assert report["status"] == "resolved"

        # Same probes, cache ON: the guard must still fire. Without this pair
        # the bypass check could be unconditional and nothing would notice.
        enabled = _capture(runs, questions=QUESTIONS)
        enabled["serverCache"] = {**bypassed["serverCache"], "enabled": True}
        strict = score_capture(enabled)
        assert strict["cache"]["bypassed"] is False
        assert set(strict["cache"]["unspacedQuestions"]) == {"q1", "q2"}
        assert strict["status"] == "unresolved"

    def test_missing_provenance_falls_back_conservatively(self):
        """An old capture, or an un-restarted server, must not get a free pass."""
        runs = [
            [_row("q1", ["alpha"], atS=i * 17.0), _row("q2", ["beta"], atS=i * 17.0)]
            for i in range(6)
        ]
        report = score_capture(_capture(runs, questions=QUESTIONS))
        assert report["cache"]["provenanceStatus"] == "unreported"
        assert report["cache"]["ttlSource"] == "assumed"
        assert report["cache"]["bypassed"] is False
        assert report["status"] == "unresolved"

    def test_the_fingerprint_is_printed_on_every_report(self):
        """Two arms printing the same fingerprint were not isolated by the key.

        The formatter is where an A/B reader actually sees it, so the line is
        emitted whether or not the run resolved.
        """
        runs = [[_row("q1", ["alpha"]), _row("q2", ["beta"])] for _ in range(12)]
        capture = _capture(runs, questions=QUESTIONS)
        capture["serverCache"] = {
            "status": "ok",
            "enabled": True,
            "ttlSeconds": 300.0,
            "fingerprint": "deadbeefcafe0001",
            "keySchema": "pc2",
        }
        text = format_meter_report(score_capture(capture))
        assert "deadbeefcafe0001" in text
        assert "packet cache:" in text
