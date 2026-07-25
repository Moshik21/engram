"""Contract tests for the graph deciding-experiment rig.

Two jobs, and the second is the important one:

1. Pin the pre-registered threshold truth table so it cannot be renegotiated
   after a number is on the table.
2. **Neuter every pre-flight check and prove it goes red.** A pre-flight that
   cannot fail is decoration, and this repo's dominant bug class is exactly a
   mechanism that runs and whose result is silently discarded. Each probe below
   is fed a healthy input (green) and a dead one (red).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.evaluation.graph_kill_rig import preflight, thresholds
from engram.evaluation.graph_kill_rig.arms import (
    QuestionRun,
    QuestionScore,
    Row,
    merge_variants,
    second_round_query,
    to_row,
)
from engram.evaluation.graph_kill_rig.corpus import build_corpus
from engram.evaluation.graph_kill_rig.runner import _void

REPO_ROOT = Path(__file__).resolve().parents[2]


def _arm(name: str, reach5: int, *, n: int = 60, rows: float = 12.0, p50: float = 40.0):
    return thresholds.ArmResult(
        arm=name,
        n=n,
        reach_at_5=reach5,
        reach_at_10=reach5,
        mean_rows=rows,
        mean_chars=1000.0,
        p50_ms=p50,
    )


def _row(**kwargs) -> Row:
    base = {
        "result_type": "episode",
        "episode_id": "ep1",
        "entity_id": None,
        "score": 0.5,
        "chars": 120,
        "traversal": False,
        "spreading": 0.0,
        "edge_proximity": 0.0,
        "activation": 0.0,
        "relationship_chars": 0,
    }
    base.update(kwargs)
    return Row(**base)


def _run(qid: str, rows: list[Row], **timings) -> QuestionRun:
    return QuestionRun(qid=qid, rows=rows, ms=40.0, stage_timings=dict(timings))


# ── the pre-registered truth table ──────────────────────────────────


class TestThresholds:
    def test_success_requires_all_four(self):
        verdict = thresholds.evaluate(
            _arm("A", 10), _arm("B", 20), _arm("C", 14), residual_rate=0.5
        )
        assert verdict.verdict == "SUCCESS"
        assert all(verdict.success_criteria.values())

    def test_k1_round_trip_kills_even_when_b_beats_a(self):
        """The kill arm matching B is sufficient — this arm has never been run."""
        verdict = thresholds.evaluate(_arm("A", 2), _arm("B", 22), _arm("C", 22), residual_rate=0.5)
        assert verdict.verdict == "KILL"
        assert any(r.startswith("K1") for r in verdict.kill_reasons)

    def test_k2_noise_floor(self):
        verdict = thresholds.evaluate(_arm("A", 10), _arm("B", 13), _arm("C", 5), residual_rate=0.5)
        assert verdict.verdict == "KILL"
        assert any(r.startswith("K2") for r in verdict.kill_reasons)

    def test_k3_context_cost(self):
        verdict = thresholds.evaluate(
            _arm("A", 10), _arm("B", 15, rows=48.0), _arm("C", 14), residual_rate=0.5
        )
        assert verdict.verdict == "KILL"
        assert any(r.startswith("K3") for r in verdict.kill_reasons)

    def test_k3_does_not_fire_when_the_gain_clears_the_bar(self):
        verdict = thresholds.evaluate(
            _arm("A", 10), _arm("B", 20, rows=48.0), _arm("C", 14), residual_rate=0.5
        )
        assert not any(r.startswith("K3") for r in verdict.kill_reasons)

    def test_k4_market_kills_a_run_that_met_every_success_criterion(self):
        """The one kill that can coexist with SUCCESS. KILL still wins."""
        verdict = thresholds.evaluate(
            _arm("A", 10), _arm("B", 20), _arm("C", 14), residual_rate=0.02
        )
        assert verdict.verdict == "KILL"
        assert any(r.startswith("K4") for r in verdict.kill_reasons)
        assert all(verdict.success_criteria.values())
        assert any("KILL takes precedence" in note for note in verdict.notes)

    def test_ambiguous_band_recommends_freeze(self):
        verdict = thresholds.evaluate(
            _arm("A", 10), _arm("B", 15), _arm("C", 11), residual_rate=0.5
        )
        assert verdict.verdict == "AMBIGUOUS"
        assert any("FREEZE" in note for note in verdict.notes)

    def test_latency_criterion_can_fail_alone(self):
        verdict = thresholds.evaluate(
            _arm("A", 10, p50=40.0),
            _arm("B", 20, p50=90.0),
            _arm("C", 14),
            residual_rate=0.5,
        )
        assert verdict.verdict == "AMBIGUOUS"
        assert verdict.success_criteria["p50_added_le_10ms"] is False

    def test_shrinking_n_below_the_floor_raises(self):
        with pytest.raises(ValueError, match="below the pre-registered floor"):
            thresholds.evaluate(
                _arm("A", 5, n=20), _arm("B", 15, n=20), _arm("C", 5, n=20), residual_rate=0.5
            )

    def test_arms_must_score_the_same_question_set(self):
        with pytest.raises(ValueError, match="different question sets"):
            thresholds.evaluate(
                _arm("A", 10, n=60), _arm("B", 20, n=59), _arm("C", 14, n=60), residual_rate=0.5
            )

    def test_kill_arm_is_scored_at_its_strongest(self):
        weak, strong = _arm("C_concat", 9), _arm("C_merged", 17)
        assert thresholds.select_kill_arm([weak, strong]).arm == "C_merged"


# ── every probe must be able to go red ──────────────────────────────


class TestProducerProbe:
    def _report(self, *, semantic: int, present: int) -> preflight.BridgeReport:
        return preflight.BridgeReport(
            present=[f"q{i}" for i in range(present)],
            missing={},
            semantic_relationship_count=semantic,
            structural_relationship_count=4,
            predicate_counts={"WORKS_ON": semantic, "PART_OF": 4},
        )

    def test_green_when_a_semantic_edge_committed(self):
        assert preflight.producer_probe(
            self._report(semantic=60, present=60), questions_requested=60
        ).passed

    def test_red_on_an_edgeless_brain(self):
        check = preflight.producer_probe(
            self._report(semantic=0, present=0), questions_requested=60
        )
        assert not check.passed
        assert "zero committed semantic relationships" in check.detail

    def test_structural_predicates_alone_do_not_count_as_a_producer(self):
        """1637 live edges are a repo index; none of them is evidence of extraction."""
        report = preflight.BridgeReport(
            present=[],
            missing={},
            semantic_relationship_count=0,
            structural_relationship_count=1637,
            predicate_counts={"PART_OF": 1300, "SUPERSEDED_BY": 337},
        )
        assert not preflight.producer_probe(report, questions_requested=60).passed


class TestConsumerByteProbe:
    def test_green_when_a_spread_bonus_reached_a_row(self):
        runs = [_run("q0", [_row(traversal=True, spreading=0.21)])]
        check = preflight.consumer_byte_probe(runs)
        assert check.passed
        assert check.measured["edge_derived_chars"] == 120

    def test_red_when_traversal_fires_on_a_relationshipless_brain(self):
        """M3.1's incidental finding: traversal appended rows, 0 relationships, zero lift.

        A loose byte count passes here. The strict one must not.
        """
        runs = [_run("q0", [_row(traversal=True, spreading=0.0, edge_proximity=0.0)])]
        check = preflight.consumer_byte_probe(runs)
        assert not check.passed
        assert check.measured["traversal_chars_loose"] == 120
        assert check.measured["edge_derived_chars"] == 0

    def test_red_when_edge_proximity_is_only_the_membership_hop(self):
        """edge_proximity>0 at min_hop=0 means 'a seed the query already matched'."""
        runs = [_run("q0", [_row(traversal=True, edge_proximity=0.6, spreading=0.0)])]
        assert not preflight.consumer_byte_probe(runs).passed

    def test_green_on_literal_relationship_json(self):
        runs = [_run("q0", [_row(result_type="entity", relationship_chars=250)])]
        assert preflight.consumer_byte_probe(runs).passed


class TestSpreadGateProbe:
    def test_green_at_full_completion_with_reach(self):
        runs = [
            _run(f"q{i}", [_row()], recall_spread=12.0, recall_spread_reached=4) for i in range(10)
        ]
        assert preflight.spread_gate_probe(runs).passed

    def test_red_below_the_eighty_percent_bar(self):
        runs = [
            _run(f"q{i}", [_row()], recall_spread=12.0, recall_spread_reached=4) for i in range(7)
        ] + [_run(f"t{i}", [_row()], recall_spread_timeout=74.5) for i in range(3)]
        check = preflight.spread_gate_probe(runs)
        assert not check.passed
        assert check.measured["completion_rate"] == 0.7

    def test_red_when_the_stage_completes_having_walked_zero_edges(self):
        """The silent-inert case: 'completed' is not the same as 'worked'."""
        runs = [
            _run(f"q{i}", [_row()], recall_spread=0.4, recall_spread_reached=0) for i in range(10)
        ]
        check = preflight.spread_gate_probe(runs)
        assert not check.passed
        assert "walked ZERO edges" in check.detail

    def test_injected_entities_count_as_reach_on_trees_without_the_reach_metric(self):
        runs = [
            _run(f"q{i}", [_row()], recall_spread=12.0, recall_spread_injected=3) for i in range(10)
        ]
        assert preflight.spread_gate_probe(runs).passed

    def test_red_when_the_graph_gate_skipped_the_stage(self):
        runs = [_run(f"q{i}", [_row()], recall_spread_skipped_probe_timeout=0.0) for i in range(10)]
        assert not preflight.spread_gate_probe(runs).passed


class TestSpreadReachEvidence:
    """The reach clause must not depend on a stage counter that can vanish.

    ``recall_spread_reached`` and ``recall_spread_injected`` were both emitted
    by pipeline.py while this rig was written; neither is emitted by the tree it
    now runs against. Row-level ``spreading`` is the durable evidence.
    """

    def test_row_spreading_alone_proves_an_edge_was_walked(self):
        runs = [_run(f"q{i}", [_row(spreading=0.04)], recall_spread=2.7) for i in range(10)]
        check = preflight.spread_gate_probe(runs)
        assert check.passed
        assert check.measured["recalls_that_walked_an_edge"] == 10

    def test_stage_counters_still_corroborate_when_present(self):
        runs = [
            _run(f"q{i}", [_row(spreading=0.0)], recall_spread=2.7, recall_spread_reached=3)
            for i in range(10)
        ]
        assert preflight.spread_gate_probe(runs).passed

    def test_no_evidence_at_all_is_a_refusal(self):
        runs = [_run(f"q{i}", [_row(spreading=0.0)], recall_spread=2.7) for i in range(10)]
        assert not preflight.spread_gate_probe(runs).passed


class TestVectorIndexProbe:
    class _FakeVectors:
        def __init__(self, present):
            self._present = set(present)

        class _Cursor:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

        @property
        def db(self):
            return self

        async def execute(self, _sql, params):
            ids = params[1:]
            return self._Cursor([{"id": i} for i in ids if i in self._present])

    class _FakeSearch:
        def __init__(self, present):
            self._vectors = TestVectorIndexProbe._FakeVectors(present)

    @pytest.mark.asyncio
    async def test_green_when_every_gold_holds_a_vector(self):
        search = self._FakeSearch({"g1", "g2"})
        check = await preflight.vector_index_probe(
            search, gold_episode_ids=["g1", "g2"], group_id="g"
        )
        assert check.passed

    @pytest.mark.asyncio
    async def test_red_on_a_dead_embedder(self):
        """dimension() returns 768 on a provider whose ONNX never loaded."""
        search = self._FakeSearch(set())
        check = await preflight.vector_index_probe(
            search, gold_episode_ids=["g1", "g2"], group_id="g"
        )
        assert not check.passed
        assert check.measured["gold_with_vectors"] == 0

    @pytest.mark.asyncio
    async def test_red_when_coverage_cannot_be_verified_at_all(self):
        class _Opaque:
            pass

        check = await preflight.vector_index_probe(_Opaque(), gold_episode_ids=["g1"], group_id="g")
        assert not check.passed
        assert "cannot be verified" in check.detail


class TestResidualProbe:
    def test_measures_unranked_linking_episodes(self):
        scores = [QuestionScore(qid=f"q{i}", gold_rank=None, link_rank=2) for i in range(9)]
        scores.append(QuestionScore(qid="q9", gold_rank=None, link_rank=None))
        check, rate = preflight.residual_probe(scores)
        assert check.passed
        assert rate == pytest.approx(0.1)

    def test_link_outside_top_k_counts_as_residual(self):
        scores = [QuestionScore(qid="q0", gold_rank=None, link_rank=11)]
        _check, rate = preflight.residual_probe(scores, k=10)
        assert rate == 1.0

    def test_red_when_nothing_to_measure(self):
        check, rate = preflight.residual_probe([])
        assert not check.passed
        assert rate is None


class TestScoredSetFloor:
    def test_red_when_bridges_fall_under_the_anchor_n(self):
        assert not preflight.scored_set_floor_probe(12, floor=36).passed

    def test_green_at_the_floor(self):
        assert preflight.scored_set_floor_probe(36, floor=36).passed


# ── the refusal envelope must not leak a number ─────────────────────


class TestVoidEnvelope:
    def test_void_suppresses_reachability_and_the_verdict(self):
        failed = preflight.Check(name="consumer_byte_probe", passed=False, detail="0 bytes")
        envelope = _void({}, [failed], residual=0.4, arms_run={"A": {"mean_rows": 5.0}})
        assert envelope["status"] == "VOID"
        assert envelope["verdict"] is None
        assert envelope["arms"] is None
        assert envelope["refusal_reasons"]
        blob = repr(envelope)
        assert "reach_at_5" not in blob


# ── arm C is text-only, deterministic, and never crippled ───────────


class TestArmC:
    def _rows(self):
        return [
            {
                "result_type": "episode",
                "episode_id": "e1",
                "episode": {"id": "e1", "content": "Handoff recorded: recall_graph_gate now"},
            },
            {
                "result_type": "episode",
                "episode": {"id": "e2", "content": "recall_graph_gate and pipeline.py moved"},
            },
        ]

    def test_second_query_extends_the_original_with_terms_it_read(self):
        follow_up = second_round_query("What is Petra Osei working on?", self._rows())
        assert follow_up.startswith("What is Petra Osei working on?")
        assert "recall_graph_gate" in follow_up

    def test_second_query_reads_only_row_text(self):
        """No graph handle is available to it — the affordance is exactly an agent's."""
        rows = [{"episode": {"id": "e1", "content": "nothing salient at all here"}}]
        assert second_round_query("Q?", rows) == "Q?"

    def test_second_query_is_deterministic(self):
        rows = self._rows()
        assert second_round_query("Q?", rows) == second_round_query("Q?", rows)

    def test_merge_variants_dedupe_and_never_lose_round_one(self):
        round_one = self._rows()
        round_two = [
            {"result_type": "episode", "episode": {"id": "e2", "content": "dupe"}},
            {"result_type": "episode", "episode": {"id": "e9", "content": "new"}},
        ]
        variants = merge_variants(round_one, round_two)
        concat_ids = [r["episode"]["id"] for r in variants["concat"]]
        assert concat_ids == ["e1", "e2", "e9"]
        assert len(variants["merged"]) == 3


class TestRowAttribution:
    def test_traversal_row_without_a_spread_bonus_is_not_edge_derived(self):
        raw = {
            "result_type": "episode",
            "episode": {"id": "e1", "content": "x"},
            "score": 0.3,
            "score_breakdown": {"entity_traversal": True, "spreading": 0.0},
        }
        row = to_row(raw)
        assert row.traversal is True
        assert row.edge_derived is False

    def test_spread_bonus_makes_a_row_edge_derived(self):
        raw = {
            "result_type": "episode",
            "episode": {"id": "e1", "content": "x"},
            "score": 0.3,
            "score_breakdown": {"entity_traversal": True, "spreading": 0.07},
        }
        assert to_row(raw).edge_derived is True


# ── the corpus is a bridge corpus, and knows when it is not ─────────


class TestCorpus:
    @pytest.fixture(scope="class")
    def corpus(self):
        return build_corpus(repo_root=REPO_ROOT, n=40, seed=3, filler=5)

    def test_builds_the_requested_bridges_from_real_repo_material(self, corpus):
        assert len(corpus.questions) == 40
        assert corpus.provenance["repo_head"]
        assert corpus.provenance["topics_harvested"] >= 40

    def test_gold_never_names_the_person(self, corpus):
        by_tag = {e.tag: e for e in corpus.episodes}
        for question in corpus.questions:
            assert question.person not in by_tag[question.gold_tag].content

    def test_query_never_names_the_topic(self, corpus):
        for question in corpus.questions:
            assert question.topic not in question.query
            assert question.person in question.query

    def test_link_episode_carries_the_edge_proposal(self, corpus):
        by_tag = {e.tag: e for e in corpus.episodes}
        for question in corpus.questions:
            rels = by_tag[question.link_tag].proposed_relationships
            assert rels and rels[0]["object"] == question.topic

    def test_invariant_check_catches_a_leaked_person(self):
        from engram.evaluation.graph_kill_rig.corpus import (
            BridgeQuestion,
            CorpusEpisode,
            _assert_bridge_invariants,
        )

        episodes = [
            CorpusEpisode(tag="link000", content="Ada owns widget", role="link"),
            CorpusEpisode(tag="gold000", content="widget changed, thanks Ada", role="gold"),
        ]
        questions = [
            BridgeQuestion(
                qid="q000",
                person="Ada",
                topic="widget",
                query="What is Ada working on?",
                link_tag="link000",
                gold_tag="gold000",
            )
        ]
        with pytest.raises(ValueError, match="LEAKS the person"):
            _assert_bridge_invariants(episodes, questions)
