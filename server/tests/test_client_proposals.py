"""Tests for client_proposals module."""

from engram.extraction.client_proposals import (
    MODEL_TIER_CONFIDENCE,
    proposals_to_evidence,
)


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
