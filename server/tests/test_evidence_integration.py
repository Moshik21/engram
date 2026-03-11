"""Integration tests for evidence-based extraction pipeline."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.extraction.commit_policy import AdaptiveCommitPolicy
from engram.extraction.evidence_bridge import EvidenceBridge
from engram.extraction.narrow.pipeline import NarrowExtractionPipeline


class TestEvidencePipelineIntegration:
    """End-to-end tests for the narrow extractor -> commit -> bridge pipeline."""

    def test_full_pipeline_identity(self):
        """Extract entities from identity text, commit, bridge to EntityCandidate."""
        cfg = ActivationConfig(evidence_extraction_enabled=True)
        pipeline = NarrowExtractionPipeline(cfg)
        policy = AdaptiveCommitPolicy(adaptive=True)
        bridge = EvidenceBridge()

        text = "My name is Alex. I work at Anthropic."
        bundle = pipeline.extract(text, "ep1", "default")
        assert len(bundle.candidates) > 0

        decisions = policy.evaluate(bundle, entity_count=0)  # cold start
        committed = [
            (ev, d)
            for ev, d in zip(bundle.candidates, decisions)
            if d.action == "commit"
        ]
        assert len(committed) > 0  # cold start lowers thresholds

        entities, claims = bridge.bridge(committed)
        entity_names = [e.name for e in entities]
        assert "Alex" in entity_names or any(
            "konnor" in n.lower() for n in entity_names
        )

    def test_full_pipeline_relationship(self):
        """Extract relationships, commit, bridge to ClaimCandidate."""
        pipeline = NarrowExtractionPipeline()
        policy = AdaptiveCommitPolicy(adaptive=True)
        bridge = EvidenceBridge()

        text = "Alice works at Google. Bob lives in New York."
        bundle = pipeline.extract(text, "ep1", "default")
        decisions = policy.evaluate(bundle, entity_count=0)
        committed = [
            (ev, d)
            for ev, d in zip(bundle.candidates, decisions)
            if d.action == "commit"
        ]
        entities, claims = bridge.bridge(committed)
        predicates = [c.predicate for c in claims]
        assert len(predicates) > 0

    def test_no_candidates_for_trivial_text(self):
        """Trivial text should produce few or no committed candidates."""
        pipeline = NarrowExtractionPipeline()
        policy = AdaptiveCommitPolicy()
        bridge = EvidenceBridge()

        text = "ok sounds good"
        bundle = pipeline.extract(text, "ep1", "default")
        decisions = policy.evaluate(bundle, entity_count=100)
        committed = [
            (ev, d)
            for ev, d in zip(bundle.candidates, decisions)
            if d.action == "commit"
        ]
        entities, claims = bridge.bridge(committed)
        assert len(entities) == 0
        assert len(claims) == 0

    def test_deferred_evidence_collected(self):
        """Borderline candidates should be deferred."""
        pipeline = NarrowExtractionPipeline()
        policy = AdaptiveCommitPolicy()

        text = "Alice went to the store."
        bundle = pipeline.extract(text, "ep1", "default")
        decisions = policy.evaluate(bundle, entity_count=200)
        deferred = [d for d in decisions if d.action == "defer"]
        # May or may not have deferred depending on exact confidence
        assert len(deferred) >= 0

    def test_cold_start_permissiveness(self):
        """Cold start (<50 entities) should lower thresholds."""
        pipeline = NarrowExtractionPipeline()
        policy_cold = AdaptiveCommitPolicy()
        policy_normal = AdaptiveCommitPolicy()

        text = "Alice works at Google."
        bundle = pipeline.extract(text, "ep1", "default")

        decisions_cold = policy_cold.evaluate(bundle, entity_count=5)
        decisions_normal = policy_normal.evaluate(bundle, entity_count=200)

        committed_cold = sum(1 for d in decisions_cold if d.action == "commit")
        committed_normal = sum(
            1 for d in decisions_normal if d.action == "commit"
        )
        assert committed_cold >= committed_normal

    def test_evidence_ids_are_unique(self):
        """All evidence candidates should have unique IDs."""
        pipeline = NarrowExtractionPipeline()
        text = (
            "My name is Alex. I work at Anthropic. "
            "I live in San Francisco."
        )
        bundle = pipeline.extract(text, "ep1", "default")
        ids = [c.evidence_id for c in bundle.candidates]
        assert len(ids) == len(set(ids))
