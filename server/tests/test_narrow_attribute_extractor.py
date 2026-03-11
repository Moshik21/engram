"""Tests for AttributeEvidenceExtractor."""

import pytest

from engram.extraction.narrow.attribute_extractor import AttributeEvidenceExtractor


@pytest.fixture
def extractor():
    return AttributeEvidenceExtractor()


class TestAttributeEvidenceExtractor:
    def test_name_is_set(self, extractor):
        assert extractor.name == "attribute"

    def test_preference_pattern(self, extractor):
        results = extractor.extract("I prefer Python over JavaScript.", "ep1", "default")
        prefs = [c for c in results if c.payload.get("attribute_type") == "preference"]
        assert len(prefs) >= 1
        assert prefs[0].payload["value"] == "Python"

    def test_favorite_pattern(self, extractor):
        results = extractor.extract("My favorite language is Rust.", "ep1", "default")
        prefs = [c for c in results if c.payload.get("attribute_type") == "preference"]
        assert len(prefs) >= 1

    def test_number_with_context(self, extractor):
        results = extractor.extract("The project has 500 GB of data.", "ep1", "default")
        quantities = [c for c in results if c.payload.get("attribute_type") == "quantity"]
        assert len(quantities) >= 1
        assert quantities[0].payload["value"] == 500.0
        assert quantities[0].payload["unit"] == "GB"

    def test_state_learning(self, extractor):
        results = extractor.extract("I'm currently learning Rust.", "ep1", "default")
        states = [c for c in results if c.payload.get("attribute_type") == "state"]
        assert len(states) >= 1
        assert "Rust" in states[0].payload["value"]

    def test_empty_text(self, extractor):
        results = extractor.extract("Nothing noteworthy here.", "ep1", "default")
        assert results == []

    def test_role_state(self, extractor):
        results = extractor.extract(
            "I'm a software engineer at Anthropic.", "ep1", "default",
        )
        states = [c for c in results if c.payload.get("state_signal") == "role_state"]
        assert len(states) >= 1
