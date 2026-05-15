"""Shared observe/remember presenter contract tests."""

from __future__ import annotations

from engram.ingestion.presenter import (
    memory_write_contract,
    present_api_memory_write,
    present_api_observe_skip,
    present_mcp_memory_write,
)


def test_observe_presenters_share_capture_contract():
    contract = memory_write_contract("observe", "ep_observe", attachment_kind="image")

    api = present_api_memory_write(
        contract,
        status="stored",
        include_legacy_episode_id=True,
    )
    mcp = present_mcp_memory_write(
        contract,
        message="Image stored for background processing.",
    )

    assert api["episodeId"] == mcp["episode_id"] == "ep_observe"
    assert api["episode_id"] == "ep_observe"
    assert api["operation"] == mcp["operation"] == "observe"
    assert api["lifecycle"]["stage"] == mcp["lifecycle"]["stage"] == "cue"
    assert api["lifecycle"]["captureStatus"] == mcp["lifecycle"]["capture_status"]
    assert api["lifecycle"]["projectionMode"] == mcp["lifecycle"]["projection_mode"]
    assert api["lifecycle"]["projectionStatus"] == mcp["lifecycle"]["projection_status"]
    assert api["lifecycle"]["attachmentKind"] == mcp["lifecycle"]["attachment_kind"]


def test_remember_presenters_share_projection_contract_with_adjudications():
    adjudication = {
        "request_id": "adj_123",
        "ambiguity_tags": ["negation_scope"],
        "selected_text": "Alice works at Google, but maybe not anymore.",
        "candidate_evidence": [
            {
                "evidence_id": "evi_1",
                "fact_class": "relationship",
                "payload": {"subject": "Alice"},
            },
        ],
        "instructions": "Resolve only if highly confident.",
    }
    contract = memory_write_contract(
        "remember",
        "ep_project",
        adjudication_requests=[adjudication],
    )

    api = present_api_memory_write(contract, status="remembered")
    mcp = present_mcp_memory_write(
        contract,
        message="Memory received. Evidence extracted and evaluated.",
    )

    assert api["episodeId"] == mcp["episode_id"] == "ep_project"
    assert api["operation"] == mcp["operation"] == "remember"
    assert api["lifecycle"]["stage"] == mcp["lifecycle"]["stage"] == "project"
    assert api["lifecycle"]["projectionMode"] == "synchronous"
    assert mcp["lifecycle"]["projection_mode"] == "synchronous"
    assert api["lifecycle"]["projectionStatus"] == "attempted"
    assert mcp["lifecycle"]["projection_status"] == "attempted"
    assert api["adjudicationRequests"][0]["requestId"] == "adj_123"
    assert api["adjudicationRequests"][0]["ambiguityTags"] == ["negation_scope"]
    assert mcp["adjudication_requests"][0]["request_id"] == "adj_123"


def test_auto_observe_skip_response_keeps_capture_semantics():
    response = present_api_observe_skip("skipped", reason="too_short")

    assert response["status"] == "skipped"
    assert response["reason"] == "too_short"
    assert response["operation"] == "observe"
    assert response["lifecycle"] == {
        "stage": "capture",
        "captureStatus": "skipped",
        "projectionMode": None,
        "projectionStatus": None,
    }
