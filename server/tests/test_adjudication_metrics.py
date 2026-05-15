from __future__ import annotations

import pytest

from engram.models.episode import Episode
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.dates import utc_now


@pytest.mark.asyncio
async def test_sqlite_stats_include_group_scoped_open_adjudication_metrics(tmp_path) -> None:
    store = SQLiteGraphStore(str(tmp_path / "adjudication-metrics.db"))
    await store.initialize()
    try:
        await store.create_episode(
            Episode(
                id="ep_native_metrics",
                content="Native metrics evidence",
                source="test",
                group_id="native_brain",
                created_at=utc_now(),
            )
        )
        await store.create_episode(
            Episode(
                id="ep_other_metrics",
                content="Other metrics evidence",
                source="test",
                group_id="other_brain",
                created_at=utc_now(),
            )
        )
        await store.store_evidence(
            [
                _evidence("ev_pending", "pending"),
                _evidence("ev_deferred", "deferred"),
                _evidence("ev_approved", "approved"),
                _evidence("ev_committed", "committed"),
            ],
            group_id="native_brain",
        )
        await store.store_evidence(
            [_evidence("ev_other", "deferred", episode_id="ep_other_metrics")],
            group_id="other_brain",
        )
        await store.store_adjudication_requests(
            [
                _adjudication("adj_pending", "pending"),
                _adjudication("adj_deferred", "deferred"),
                _adjudication("adj_error", "error"),
                _adjudication("adj_rejected", "rejected"),
            ],
            group_id="native_brain",
        )
        await store.store_adjudication_requests(
            [_adjudication("adj_other", "error", episode_id="ep_other_metrics")],
            group_id="other_brain",
        )

        metrics = (await store.get_stats("native_brain"))["adjudication_metrics"]

        assert metrics["evidence_status_counts"] == {
            "pending": 1,
            "deferred": 1,
            "approved": 1,
        }
        assert metrics["request_status_counts"] == {
            "pending": 1,
            "deferred": 1,
            "error": 1,
        }
        assert metrics["open_evidence_count"] == 3
        assert metrics["open_request_count"] == 3
        assert metrics["open_work_count"] == 6
    finally:
        await store.close()


def _evidence(
    evidence_id: str,
    status: str,
    *,
    episode_id: str = "ep_native_metrics",
) -> dict:
    return {
        "evidence_id": evidence_id,
        "episode_id": episode_id,
        "fact_class": "relationship",
        "confidence": 0.7,
        "source_type": "test",
        "extractor_name": "metrics-test",
        "payload": {"subject": evidence_id},
        "source_span": f"Evidence {evidence_id}",
        "corroborating_signals": [],
        "ambiguity_tags": [],
        "ambiguity_score": 0.0,
        "status": status,
    }


def _adjudication(
    request_id: str,
    status: str,
    *,
    episode_id: str = "ep_native_metrics",
) -> dict:
    return {
        "request_id": request_id,
        "episode_id": episode_id,
        "status": status,
        "ambiguity_tags": ["relationship_direction"],
        "evidence_ids": [],
        "selected_text": f"Adjudication {request_id}",
        "request_reason": f"metrics:{status}",
    }
