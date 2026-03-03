"""Tests for temporal hint resolution."""

from datetime import datetime

from engram.extraction.temporal import resolve_temporal_hint


class TestResolveTemporalHint:
    def test_last_month(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("last month", ref)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 1

    def test_last_month_january_wraps(self):
        ref = datetime(2026, 1, 15)
        result = resolve_temporal_hint("last month", ref)
        assert result is not None
        assert result.year == 2025
        assert result.month == 12

    def test_last_week(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("last week", ref)
        assert result is not None
        assert result.day == 8  # 15 - 7

    def test_last_year(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("last year", ref)
        assert result is not None
        assert result.year == 2025
        assert result.month == 1

    def test_days_ago(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("5 days ago", ref)
        assert result is not None
        assert result.day == 10

    def test_weeks_ago(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("2 weeks ago", ref)
        assert result is not None
        assert result.day == 1

    def test_months_ago(self):
        ref = datetime(2026, 6, 15)
        result = resolve_temporal_hint("3 months ago", ref)
        assert result is not None
        assert result.month == 3
        assert result.year == 2026

    def test_since_month(self):
        ref = datetime(2026, 6, 15)
        result = resolve_temporal_hint("since January", ref)
        assert result is not None
        assert result.month == 1
        assert result.year == 2026

    def test_since_month_year(self):
        result = resolve_temporal_hint("since March 2024")
        assert result is not None
        assert result.month == 3
        assert result.year == 2024

    def test_recently(self):
        ref = datetime(2026, 3, 15)
        result = resolve_temporal_hint("recently", ref)
        assert result is not None
        assert result.day == 8  # 15 - 7

    def test_iso_passthrough(self):
        result = resolve_temporal_hint("2024-06-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_garbage_returns_none(self):
        result = resolve_temporal_hint("xyzabc123")
        assert result is None

    def test_empty_returns_none(self):
        result = resolve_temporal_hint("")
        assert result is None

    def test_none_input(self):
        # Whitespace only
        result = resolve_temporal_hint("   ")
        assert result is None

    def test_reference_date_override(self):
        ref = datetime(2025, 8, 20)
        result = resolve_temporal_hint("last month", ref)
        assert result is not None
        assert result.month == 7
        assert result.year == 2025
