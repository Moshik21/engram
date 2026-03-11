"""Tests for RelationshipPatternExtractor."""

import pytest

from engram.extraction.narrow.relationship_extractor import (
    RelationshipPatternExtractor,
)


@pytest.fixture
def extractor():
    return RelationshipPatternExtractor()


class TestRelationshipPatternExtractor:
    def test_name_is_set(self, extractor):
        assert extractor.name == "relationship_pattern"

    def test_third_person_works_at(self, extractor):
        results = extractor.extract("Alice works at Google.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "WORKS_AT"]
        assert len(rels) == 1
        assert rels[0].payload["subject"] == "Alice"
        assert rels[0].payload["object"] == "Google"

    def test_first_person_works_at(self, extractor):
        results = extractor.extract("I work at Anthropic.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "WORKS_AT"]
        assert len(rels) == 1
        assert rels[0].payload["subject"] == "User"
        assert rels[0].payload["object"] == "Anthropic"
        assert "first_person" in rels[0].corroborating_signals

    def test_family_pattern(self, extractor):
        results = extractor.extract("My wife is Sarah.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "PARTNER_OF"]
        assert len(rels) == 1
        assert rels[0].payload["object"] == "Sarah"
        assert rels[0].confidence >= 0.80

    def test_contradiction_nearby_lowers_confidence(self, extractor):
        results = extractor.extract("I no longer work at Google.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "WORKS_AT"]
        assert len(rels) == 1
        assert rels[0].payload["polarity"] == "negative"
        assert rels[0].confidence < 0.75

    def test_dedup_same_relationship(self, extractor):
        results = extractor.extract(
            "Alice works at Google. Alice works at Google.",
            "ep1",
            "default",
        )
        rels = [c for c in results if c.payload.get("predicate") == "WORKS_AT"]
        assert len(rels) == 1

    def test_lives_in(self, extractor):
        # LIVES_IN canonicalizes to LOCATED_IN
        results = extractor.extract("I live in New York.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "LOCATED_IN"]
        assert len(rels) == 1

    def test_empty_text(self, extractor):
        results = extractor.extract("", "ep1", "default")
        assert results == []

    def test_child_of_pattern(self, extractor):
        results = extractor.extract("My dad is Robert.", "ep1", "default")
        rels = [c for c in results if c.payload.get("predicate") == "CHILD_OF"]
        assert len(rels) == 1
        assert rels[0].payload["object"] == "Robert"
