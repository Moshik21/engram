"""Tests for AdaptiveCommitPolicy."""

from engram.extraction.commit_policy import AdaptiveCommitPolicy, CommitThresholds
from engram.extraction.evidence import EvidenceBundle, EvidenceCandidate


def _make_candidate(fact_class: str, confidence: float) -> EvidenceCandidate:
    return EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class=fact_class,
        confidence=confidence,
        source_type="narrow_extractor",
        extractor_name="test",
    )


def _make_bundle(candidates: list[EvidenceCandidate]) -> EvidenceBundle:
    return EvidenceBundle(episode_id="ep1", candidates=candidates)


class TestCommitThresholds:
    def test_default_thresholds(self):
        t = CommitThresholds()
        assert t.entity == 0.70
        assert t.relationship == 0.75
        assert t.attribute == 0.65
        assert t.temporal == 0.60

    def test_for_class(self):
        t = CommitThresholds()
        assert t.for_class("entity") == 0.70
        assert t.for_class("relationship") == 0.75
        assert t.for_class("unknown") == 0.70  # fallback


class TestAdaptiveCommitPolicy:
    def test_commit_above_threshold(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([_make_candidate("entity", 0.85)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert len(decisions) == 1
        assert decisions[0].action == "commit"

    def test_defer_borderline(self):
        policy = AdaptiveCommitPolicy()
        # 0.60 is within defer band (0.70 - 0.15 = 0.55)
        bundle = _make_bundle([_make_candidate("entity", 0.60)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "defer"

    def test_reject_below_band(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([_make_candidate("entity", 0.40)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "reject"

    def test_cold_start_lowers_threshold(self):
        policy = AdaptiveCommitPolicy()
        # entity threshold = 0.70, cold start lowers by 0.15 -> 0.55
        bundle = _make_bundle([_make_candidate("entity", 0.58)])
        decisions = policy.evaluate(bundle, entity_count=10)
        assert decisions[0].action == "commit"

    def test_dense_graph_raises_threshold(self):
        policy = AdaptiveCommitPolicy()
        # entity threshold = 0.70, dense raises by 0.05 -> 0.75
        bundle = _make_bundle([_make_candidate("entity", 0.72)])
        decisions = policy.evaluate(bundle, entity_count=600)
        assert decisions[0].action == "defer"

    def test_non_adaptive_uses_base_threshold(self):
        policy = AdaptiveCommitPolicy(adaptive=False)
        bundle = _make_bundle([_make_candidate("entity", 0.58)])
        decisions = policy.evaluate(bundle, entity_count=10)
        assert decisions[0].action == "defer"  # 0.58 < 0.70

    def test_relationship_threshold_higher(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([_make_candidate("relationship", 0.72)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "defer"  # 0.72 < 0.75

    def test_temporal_threshold_lower(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([_make_candidate("temporal", 0.62)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "commit"  # 0.62 >= 0.60

    def test_multiple_candidates(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle(
            [
                _make_candidate("entity", 0.90),  # commit
                _make_candidate("entity", 0.60),  # defer
                _make_candidate("entity", 0.30),  # reject
            ]
        )
        decisions = policy.evaluate(bundle, entity_count=100)
        assert [d.action for d in decisions] == ["commit", "defer", "reject"]

    def test_empty_bundle(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions == []

    def test_custom_thresholds(self):
        policy = AdaptiveCommitPolicy(thresholds=CommitThresholds(entity=0.50))
        bundle = _make_bundle([_make_candidate("entity", 0.55)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "commit"

    def test_decision_has_effective_confidence(self):
        policy = AdaptiveCommitPolicy()
        bundle = _make_bundle([_make_candidate("entity", 0.85)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].effective_confidence == 0.85

    def test_custom_defer_band(self):
        policy = AdaptiveCommitPolicy(defer_band=0.05)
        # entity threshold 0.70, defer band only 0.05 -> defer range is 0.65-0.70
        bundle = _make_bundle([_make_candidate("entity", 0.60)])
        decisions = policy.evaluate(bundle, entity_count=100)
        assert decisions[0].action == "reject"  # 0.60 < 0.65 (0.70 - 0.05)
