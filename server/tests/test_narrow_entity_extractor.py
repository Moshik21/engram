"""Tests for IdentityEntityExtractor."""

import pytest

from engram.extraction.narrow.entity_extractor import IdentityEntityExtractor


@pytest.fixture
def extractor():
    return IdentityEntityExtractor()


class TestIdentityEntityExtractor:
    def test_name_is_set(self, extractor):
        assert extractor.name == "identity_entity"

    def test_identity_my_name_is(self, extractor):
        results = extractor.extract("My name is Alex Chen.", "ep1", "default")
        names = [c.payload["name"] for c in results]
        assert "Alex Chen" in names
        entity = [c for c in results if c.payload["name"] == "Alex Chen"][0]
        assert entity.payload["entity_type"] == "Person"
        assert entity.confidence >= 0.85
        assert "identity_pattern" in entity.corroborating_signals

    def test_works_at_identity(self, extractor):
        results = extractor.extract("I work at Anthropic.", "ep1", "default")
        names = [c.payload["name"] for c in results]
        assert "Anthropic" in names
        entity = [c for c in results if c.payload["name"] == "Anthropic"][0]
        assert entity.payload["entity_type"] == "Organization"

    def test_lives_in(self, extractor):
        results = extractor.extract("I live in San Francisco.", "ep1", "default")
        names = [c.payload["name"] for c in results]
        assert "San Francisco" in names
        entity = [c for c in results if c.payload["name"] == "San Francisco"][0]
        assert entity.payload["entity_type"] == "Location"

    def test_proper_name_detection(self, extractor):
        results = extractor.extract("Alice went to the store with Bob.", "ep1", "default")
        names = [c.payload["name"] for c in results]
        assert "Alice" in names
        assert "Bob" in names

    def test_technical_token(self, extractor):
        results = extractor.extract(
            "We use React and TypeScript for the frontend.", "ep1", "default",
        )
        names = [c.payload["name"] for c in results]
        assert "React" in names or "TypeScript" in names
        tech = [c for c in results if c.payload.get("entity_type") == "Technology"]
        assert len(tech) >= 1

    def test_stopwords_excluded(self, extractor):
        results = extractor.extract(
            "The Monday meeting was about This project.", "ep1", "default",
        )
        names = [c.payload["name"] for c in results]
        assert "The" not in names
        assert "Monday" not in names
        assert "This" not in names

    def test_dedup_by_name(self, extractor):
        results = extractor.extract("My name is Alice. Alice is great.", "ep1", "default")
        alice_candidates = [c for c in results if c.payload["name"] == "Alice"]
        assert len(alice_candidates) == 1

    def test_source_span_set(self, extractor):
        results = extractor.extract("My name is Alex.", "ep1", "default")
        assert any(c.source_span for c in results)

    def test_empty_text(self, extractor):
        results = extractor.extract("", "ep1", "default")
        assert results == []

    def test_family_member(self, extractor):
        results = extractor.extract("My wife Sarah loves cooking.", "ep1", "default")
        names = [c.payload["name"] for c in results]
        assert "Sarah" in names
