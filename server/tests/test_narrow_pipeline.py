"""Tests for NarrowExtractionPipeline."""

import pytest

from engram.extraction.narrow.pipeline import NarrowExtractionPipeline


@pytest.fixture
def pipeline():
    return NarrowExtractionPipeline()


class TestNarrowExtractionPipeline:
    def test_empty_text(self, pipeline):
        bundle = pipeline.extract("", "ep1", "default")
        assert bundle.candidates == []
        assert bundle.total_ms >= 0

    def test_rich_text(self, pipeline):
        text = "My name is Alex. I work at Anthropic. I live in San Francisco."
        bundle = pipeline.extract(text, "ep1", "default")
        assert len(bundle.candidates) > 0
        assert bundle.episode_id == "ep1"
        assert bundle.group_id == "default"

    def test_extractor_stats(self, pipeline):
        text = "Alice works at Google."
        bundle = pipeline.extract(text, "ep1", "default")
        assert "identity_entity" in bundle.extractor_stats
        assert "relationship_pattern" in bundle.extractor_stats
        assert "temporal" in bundle.extractor_stats
        assert "attribute" in bundle.extractor_stats

    def test_cross_corroboration_boosts(self, pipeline):
        # "Anthropic" appears as both entity (from identity extractor)
        # and in relationship (from relationship extractor)
        text = "I work at Anthropic. Anthropic is great."
        bundle = pipeline.extract(text, "ep1", "default")
        anthropic_entities = [
            c
            for c in bundle.candidates
            if c.fact_class == "entity" and c.payload.get("name") == "Anthropic"
        ]
        if anthropic_entities:
            assert any(
                "cross_extractor_corroboration" in c.corroborating_signals
                for c in anthropic_entities
            )

    def test_deduplication(self, pipeline):
        text = "My name is Alex. Alex is a developer."
        bundle = pipeline.extract(text, "ep1", "default")
        alex_entities = [
            c
            for c in bundle.candidates
            if c.fact_class == "entity" and c.payload.get("name", "").lower() == "alex"
        ]
        assert len(alex_entities) == 1

    def test_multiple_fact_classes(self, pipeline):
        text = "My name is Alex. I work at Anthropic. Since 2024-01-15 I prefer Python."
        bundle = pipeline.extract(text, "ep1", "default")
        fact_classes = {c.fact_class for c in bundle.candidates}
        assert "entity" in fact_classes
        assert "relationship" in fact_classes

    def test_timing_stats(self, pipeline):
        bundle = pipeline.extract("Alice works at Google.", "ep1", "default")
        assert bundle.total_ms > 0
        for stats in bundle.extractor_stats.values():
            assert "count" in stats
            assert "duration_ms" in stats

    def test_all_candidates_have_required_fields(self, pipeline):
        text = "My name is Alex. I work at Anthropic."
        bundle = pipeline.extract(text, "ep1", "default")
        for c in bundle.candidates:
            assert c.evidence_id.startswith("evi_")
            assert c.episode_id == "ep1"
            assert c.group_id == "default"
            assert c.fact_class in (
                "entity",
                "relationship",
                "attribute",
                "temporal",
            )
            assert 0.0 <= c.confidence <= 1.0
            assert c.source_type == "narrow_extractor"
            assert c.extractor_name != ""
