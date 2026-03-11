"""Tests for TemporalEvidenceExtractor."""

import pytest

from engram.extraction.narrow.temporal_extractor import TemporalEvidenceExtractor


@pytest.fixture
def extractor():
    return TemporalEvidenceExtractor()


class TestTemporalEvidenceExtractor:
    def test_name_is_set(self, extractor):
        assert extractor.name == "temporal"

    def test_iso_date(self, extractor):
        results = extractor.extract(
            "Meeting on 2026-03-15 with Alice.",
            "ep1",
            "default",
        )
        assert len(results) >= 1
        assert any(c.payload.get("temporal_type") == "date" for c in results)

    def test_relative_date(self, extractor):
        results = extractor.extract("I met Bob yesterday.", "ep1", "default")
        dates = [c for c in results if c.payload.get("temporal_type") == "date"]
        assert len(dates) >= 1
        assert dates[0].payload["temporal_marker"].lower() == "yesterday"

    def test_duration(self, extractor):
        results = extractor.extract("I worked there for 5 years.", "ep1", "default")
        durations = [c for c in results if c.payload.get("temporal_type") == "duration"]
        assert len(durations) >= 1
        assert durations[0].payload["amount"] == 5
        assert durations[0].payload["unit"] == "year"

    def test_nearby_entity_attached(self, extractor):
        results = extractor.extract("Alice joined on 2026-01-15.", "ep1", "default")
        dates = [c for c in results if c.payload.get("temporal_type") == "date"]
        assert len(dates) >= 1
        assert dates[0].payload.get("nearby_entity") == "Alice"

    def test_empty_text(self, extractor):
        results = extractor.extract("No dates here.", "ep1", "default")
        assert results == []

    def test_dedup_same_date(self, extractor):
        results = extractor.extract("Meeting yesterday. Also yesterday.", "ep1", "default")
        dates = [c for c in results if c.payload.get("temporal_marker", "").lower() == "yesterday"]
        assert len(dates) == 1
