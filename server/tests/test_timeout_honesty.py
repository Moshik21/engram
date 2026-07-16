"""Timeout / project-file fallback must not look like clean success."""

from __future__ import annotations

from engram.retrieval.budgets import RecallBudget
from engram.retrieval.recall_surface import (
    _attach_recall_budget_metadata,
    _mark_project_file_recall_fallback,
    _recall_budget_metadata,
)


def _budget() -> RecallBudget:
    return RecallBudget(
        profile="explicit",
        surface="api",
        mode="axi_recall",
        max_wall_ms=2000,
        max_search_ms=1500,
        max_graph_ms=500,
        max_packet_ms=500,
        max_results=5,
        max_packets=3,
        max_output_tokens=800,
        allow_deep_recall=True,
        allow_embeddings=True,
        allow_graph_probe=True,
        timeout_degrades=True,
    )


def test_project_file_fallback_marks_degraded_not_ok() -> None:
    meta = _recall_budget_metadata(
        _budget(),
        status="ok",
        duration_ms=3500.0,
        budget_miss=True,
        timeout=False,
    )
    _mark_project_file_recall_fallback(meta, packet_count=2)
    assert meta["status"] == "degraded"
    assert meta["degraded"] is True
    assert meta["fallback_status"] == "project_file_recall_fallback"
    assert meta["skip_reason"] == "project_file_fallback"

    response: dict = {
        "operation": "recall",
        "lifecycle": {},
        "items": [],
        "packets": [{"packet_type": "project_home"}],
    }
    _attach_recall_budget_metadata(response, meta, camel_case=True)
    assert response["status"] == "degraded"
    assert response["lifecycle"]["degraded"] is True
    assert response["lifecycle"]["fallbackStatus"] == "project_file_recall_fallback"
    assert response["budget"]["degraded"] is True


def test_timeout_metadata_already_degraded() -> None:
    meta = _recall_budget_metadata(
        _budget(),
        status="degraded",
        duration_ms=4000.0,
        skip_reason="recall_timeout",
        timeout=True,
        budget_miss=True,
        fallback_status="miss",
    )
    assert meta["status"] == "degraded"
    assert meta["timeout"] is True
    assert meta["degraded"] is True
