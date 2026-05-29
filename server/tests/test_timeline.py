"""Tests for the dated timeline read surface (#7)."""

from __future__ import annotations

import pytest

from engram.retrieval.timeline import (
    build_timeline_rows,
    build_timeline_surface,
    plan_temporal,
    render_timeline_markdown,
)


def _episode(eid, content, conversation_date=None, created_at=None, rtype="episode"):
    return {
        "result_type": rtype,
        "episode": {
            "id": eid,
            "content": content,
            "conversation_date": conversation_date,
            "created_at": created_at,
        },
    }


def _fact(subject, predicate, obj, valid_from=None, created_at=None, source_episode=None):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "valid_from": valid_from,
        "created_at": created_at,
        "source_episode": source_episode,
    }


class TestBuildTimelineRows:
    def test_orders_ascending_and_resolves_dates(self):
        results = [
            _episode("e2", "Bought the Dell XPS 13", conversation_date="2023-04-12"),
            _episode("e1", "Got the Samsung Galaxy S22", conversation_date="2023-02-03"),
        ]
        rows = build_timeline_rows(results, [])
        assert [r.source_id for r in rows] == ["e1", "e2"]  # ascending by date
        assert rows[0].date_iso == "2023-02-03"
        assert rows[0].date_basis == "conversation_date"

    def test_created_at_fallback_flagged(self):
        results = [_episode("e1", "no conv date", created_at="2023-05-01T10:00:00")]
        rows = build_timeline_rows(results, [])
        assert len(rows) == 1
        assert rows[0].date_basis == "created_at"

    def test_undated_rows_dropped(self):
        results = [_episode("e1", "no dates at all")]
        assert build_timeline_rows(results, []) == []

    def test_facts_use_valid_from(self):
        facts = [_fact("User", "LIVES_IN", "Austin", valid_from="2024-01-01")]
        rows = build_timeline_rows([], facts)
        assert len(rows) == 1
        assert rows[0].kind == "fact"
        assert rows[0].date_basis == "valid_from"
        assert "user" in rows[0].label.lower() and "austin" in rows[0].label.lower()

    def test_mixed_tz_aware_and_naive_sorts_without_error(self):
        results = [
            _episode("e1", "aware", conversation_date="2023-01-01T00:00:00+05:00"),
            _episode("e2", "naive", conversation_date="2023-01-02T00:00:00"),
        ]
        rows = build_timeline_rows(results, [])  # must not raise on comparison
        assert len(rows) == 2

    def test_cue_episode_uses_cue_text(self):
        result = {
            "result_type": "cue_episode",
            "episode": {"id": "e1", "content": "full", "conversation_date": "2023-03-01"},
            "cue": {"cue_text": "cue snippet"},
        }
        rows = build_timeline_rows([result], [])
        assert rows[0].kind == "cue"
        assert rows[0].label == "cue snippet"


class TestPlanTemporal:
    def test_first_last_and_span(self):
        rows = build_timeline_rows(
            [
                _episode("e1", "webinar", conversation_date="2023-03-01"),
                _episode("e2", "workshop", conversation_date="2023-03-15"),
            ],
            [],
        )
        plan = plan_temporal(rows)
        assert plan["first"]["source_id"] == "e1"
        assert plan["last"]["source_id"] == "e2"
        assert plan["span_days"] == 14

    def test_current_value_is_latest_per_subject_predicate(self):
        facts = [
            _fact("User", "LIVES_IN", "NYC", valid_from="2020-01-01"),
            _fact("User", "LIVES_IN", "Austin", valid_from="2024-01-01"),
        ]
        rows = build_timeline_rows([], facts)
        plan = plan_temporal(rows)
        current = plan["current_values"]
        assert len(current) == 1
        assert "austin" in current[0]["label"].lower()  # latest wins

    def test_empty(self):
        plan = plan_temporal([])
        assert plan["first"] is None and plan["span_days"] is None


def test_render_markdown_flags_ingestion_dates():
    rows = build_timeline_rows(
        [
            _episode("e1", "real event", conversation_date="2023-02-03"),
            _episode("e2", "ingested only", created_at="2023-05-01T00:00:00"),
        ],
        [],
    )
    md = render_timeline_markdown(rows)
    assert "2023-02-03 — real event" in md
    assert "~ingested" in md  # created_at row flagged


@pytest.mark.asyncio
async def test_build_timeline_surface_end_to_end():
    class FakeManager:
        async def recall(self, query, group_id="default", limit=10):
            return [_episode("e1", "started running", conversation_date="2023-06-01")]

        async def search_facts(self, group_id="default", query="", include_expired=False, limit=10):
            return [_fact("User", "USES", "Helix", valid_from="2023-07-01")]

    out = await build_timeline_surface(FakeManager(), "what happened", limit=10)
    assert out["row_count"] == 2
    assert out["event_time_row_count"] == 2  # both have real event dates
    assert out["rows"][0]["date"] == "2023-06-01"  # ascending
    assert out["planner"]["span_days"] == 30
    assert "2023-06-01 — started running" in out["markdown"]
