"""Tests for client_proposals module."""

from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.client_proposals import (
    MODEL_TIER_CONFIDENCE,
    proposals_to_evidence,
)
from engram.extraction.promotion import ALLOWED_CLIENT_PREDICATES


class TestProposalsToEvidence:
    def test_entity_proposals(self):
        entities = [
            {"name": "Alice", "entity_type": "Person"},
            {"name": "Google", "entity_type": "Organization"},
        ]
        results = proposals_to_evidence(entities, None, "ep1", "default", "sonnet")
        assert len(results) == 2
        assert all(r.source_type == "client_proposal" for r in results)
        assert all(r.confidence == MODEL_TIER_CONFIDENCE["sonnet"] for r in results)

    def test_relationship_proposals(self):
        rels = [{"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"}]
        results = proposals_to_evidence(None, rels, "ep1", "default", "opus")
        assert len(results) == 1
        assert results[0].fact_class == "relationship"
        assert results[0].confidence == MODEL_TIER_CONFIDENCE["opus"]

    def test_empty_proposals(self):
        results = proposals_to_evidence(None, None, "ep1", "default")
        assert results == []

    def test_skip_empty_names(self):
        entities = [{"name": "", "entity_type": "Person"}, {"name": "Alice"}]
        results = proposals_to_evidence(entities, None, "ep1", "default")
        assert len(results) == 1
        assert results[0].payload["name"] == "Alice"

    def test_skip_empty_predicates(self):
        rels = [{"subject": "Alice", "predicate": "", "object": "Google"}]
        results = proposals_to_evidence(None, rels, "ep1", "default")
        assert results == []

    def test_default_model_tier(self):
        entities = [{"name": "Test"}]
        results = proposals_to_evidence(entities, None, "ep1", "default")
        assert results[0].confidence == 0.70

    def test_unknown_model_tier(self):
        entities = [{"name": "Test"}]
        results = proposals_to_evidence(
            entities,
            None,
            "ep1",
            "default",
            "unknown_model",
        )
        assert results[0].confidence == 0.70  # fallback to default

    def test_extractor_name_includes_tier(self):
        entities = [{"name": "Test"}]
        results = proposals_to_evidence(entities, None, "ep1", "default", "haiku")
        assert results[0].extractor_name == "client_haiku"

    def test_signals_include_model_tier(self):
        entities = [{"name": "Test"}]
        results = proposals_to_evidence(entities, None, "ep1", "default", "opus")
        assert "client_proposal" in results[0].corroborating_signals
        assert "model_opus" in results[0].corroborating_signals

    def test_entity_summary_included(self):
        entities = [
            {"name": "Alice", "entity_type": "Person", "summary": "A developer"},
        ]
        results = proposals_to_evidence(entities, None, "ep1", "default")
        assert results[0].payload["summary"] == "A developer"

    def test_relationship_polarity(self):
        rels = [
            {
                "subject": "User",
                "predicate": "LIKES",
                "object": "Python",
                "polarity": "negative",
            },
        ]
        results = proposals_to_evidence(None, rels, "ep1", "default")
        assert results[0].payload["polarity"] == "negative"


class TestPredicateCanonicalization:
    """M0.8: allowlist gate canonicalizes first and is closed under the mapping."""

    def test_allowlist_closed_under_canonicalization(self):
        canonicalizer = PredicateCanonicalizer()
        for predicate in ALLOWED_CLIENT_PREDICATES:
            assert canonicalizer.canonicalize(predicate) in ALLOWED_CLIENT_PREDICATES, (
                f"{predicate} canonicalizes out of the allowlist"
            )

    def test_lives_in_canonicalizes_and_passes(self):
        rels = [{"subject": "Alice", "predicate": "LIVES_IN", "object": "Berlin"}]
        results = proposals_to_evidence(None, rels, "ep1", "default", "opus")
        assert results[0].payload["predicate"] == "LOCATED_IN"
        assert results[0].payload["proposed_predicate"] == "LIVES_IN"
        assert "predicate_not_allowed" not in results[0].corroborating_signals

    def test_located_in_direct_proposal_allowed(self):
        # The canonical target itself (the vocabulary actually in the graph)
        # must not be rejected (pre-M0.8 trap).
        rels = [{"subject": "Alice", "predicate": "LOCATED_IN", "object": "Berlin"}]
        results = proposals_to_evidence(None, rels, "ep1", "default", "opus")
        assert results[0].payload["predicate"] == "LOCATED_IN"
        assert "proposed_predicate" not in results[0].payload
        assert "predicate_not_allowed" not in results[0].corroborating_signals

    def test_depends_on_and_requires_both_allowed(self):
        for predicate in ("DEPENDS_ON", "BLOCKED_BY", "REQUIRES"):
            rels = [{"subject": "A", "predicate": predicate, "object": "B"}]
            results = proposals_to_evidence(None, rels, "ep1", "default")
            assert results[0].payload["predicate"] == "REQUIRES"
            assert "predicate_not_allowed" not in results[0].corroborating_signals

    def test_interested_in_rejected_visibly(self):
        # No invented synonyms: INTERESTED_IN is not canonical-mapped and stays
        # rejected — with the signal visible so the surface can explain why.
        rels = [{"subject": "User", "predicate": "INTERESTED_IN", "object": "Jazz"}]
        results = proposals_to_evidence(None, rels, "ep1", "default", "opus")
        assert results[0].payload["predicate"] == "INTERESTED_IN"
        assert "predicate_not_allowed" in results[0].corroborating_signals
        assert results[0].confidence <= 0.40
