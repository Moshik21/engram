from __future__ import annotations

import importlib.util

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.graph import HelixGraphStore


@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_open_evidence_and_adjudication_statuses_round_trip(
    tmp_path,
) -> None:
    """Native PyO3 Helix should expose all lite-mode open adjudication statuses."""
    store = HelixGraphStore(
        HelixDBConfig(
            transport="native",
            data_dir=str(tmp_path / "native-open-adjudication-status"),
        )
    )
    await store.initialize()
    try:
        await store.store_evidence(
            [
                _evidence("ev_pending", "pending", 0.2),
                _evidence("ev_deferred", "deferred", 0.9),
                _evidence("ev_approved", "approved", 0.5),
                _evidence("ev_committed", "committed", 1.0),
            ],
            group_id="native_brain",
        )
        await store.store_evidence(
            [_evidence("ev_other", "deferred", 0.99)],
            group_id="other_brain",
        )

        evidence = await store.get_pending_evidence("native_brain")

        assert [item["evidence_id"] for item in evidence] == [
            "ev_deferred",
            "ev_approved",
            "ev_pending",
        ]
        assert {item["status"] for item in evidence} == {
            "pending",
            "deferred",
            "approved",
        }

        await store.update_evidence_status(
            "ev_deferred",
            "committed",
            {"committed_id": "rel_native_status"},
            group_id="native_brain",
        )
        evidence_after_commit = await store.get_pending_evidence("native_brain")
        assert [item["evidence_id"] for item in evidence_after_commit] == [
            "ev_approved",
            "ev_pending",
        ]

        await store.store_adjudication_requests(
            [
                _adjudication("adj_pending", "pending", "2026-05-14T12:02:00Z"),
                _adjudication("adj_deferred", "deferred", "2026-05-14T12:01:00Z"),
                _adjudication("adj_error", "error", "2026-05-14T12:03:00Z"),
                _adjudication("adj_rejected", "rejected", "2026-05-14T12:00:00Z"),
            ],
            group_id="native_brain",
        )

        requests = await store.get_pending_adjudication_requests("native_brain")
        deferred = await store.get_adjudication_request("adj_deferred", "native_brain")
        metrics = (await store.get_stats("native_brain"))["adjudication_metrics"]

        assert [item["request_id"] for item in requests] == [
            "adj_deferred",
            "adj_pending",
            "adj_error",
        ]
        assert {item["status"] for item in requests} == {"pending", "deferred", "error"}
        assert deferred is not None
        assert deferred["status"] == "deferred"
        assert metrics["evidence_status_counts"] == {
            "pending": 1,
            "deferred": 0,
            "approved": 1,
        }
        assert metrics["request_status_counts"] == {
            "pending": 1,
            "deferred": 1,
            "error": 1,
        }
        assert metrics["open_evidence_count"] == 2
        assert metrics["open_request_count"] == 3
        assert metrics["open_work_count"] == 5
    finally:
        await store.close()


def _evidence(evidence_id: str, status: str, confidence: float) -> dict:
    return {
        "evidence_id": evidence_id,
        "episode_id": "ep_native_status",
        "fact_class": "relationship",
        "confidence": confidence,
        "source_type": "test",
        "extractor_name": "native-status-test",
        "payload": {"subject": evidence_id},
        "source_span": f"Evidence {evidence_id}",
        "corroborating_signals": [],
        "ambiguity_tags": [],
        "ambiguity_score": 0.0,
        "status": status,
    }


def _adjudication(request_id: str, status: str, created_at: str) -> dict:
    return {
        "request_id": request_id,
        "episode_id": "ep_native_status",
        "status": status,
        "ambiguity_tags": ["relationship_direction"],
        "evidence_ids": [],
        "selected_text": f"Adjudication {request_id}",
        "request_reason": f"native_status:{status}",
        "created_at": created_at,
    }
