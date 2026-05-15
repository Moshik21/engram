"""Shared memory write presenters for observe/remember surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

MemoryWriteOperation = Literal["observe", "remember"]


def _operation_lifecycle(operation: MemoryWriteOperation) -> dict[str, str]:
    if operation == "remember":
        return {
            "stage": "project",
            "projection_mode": "synchronous",
            "projection_status": "attempted",
        }
    return {
        "stage": "cue",
        "projection_mode": "background",
        "projection_status": "queued",
    }


def _copy_adjudication_requests(
    requests: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [dict(request) for request in requests or []]


def memory_write_contract(
    operation: MemoryWriteOperation,
    episode_id: str,
    *,
    adjudication_requests: Sequence[Mapping[str, Any]] | None = None,
    attachment_kind: str | None = None,
) -> dict[str, Any]:
    """Normalize a write-path result into a surface-neutral lifecycle contract."""
    lifecycle = _operation_lifecycle(operation)
    return {
        "operation": operation,
        "episode_id": episode_id,
        "capture_status": "stored",
        "lifecycle_stage": lifecycle["stage"],
        "projection_mode": lifecycle["projection_mode"],
        "projection_status": lifecycle["projection_status"],
        "attachment_kind": attachment_kind,
        "adjudication_requests": _copy_adjudication_requests(adjudication_requests),
    }


def _api_lifecycle(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = {
        "stage": contract.get("lifecycle_stage"),
        "captureStatus": contract.get("capture_status"),
        "projectionMode": contract.get("projection_mode"),
        "projectionStatus": contract.get("projection_status"),
    }
    attachment_kind = contract.get("attachment_kind")
    if attachment_kind:
        lifecycle["attachmentKind"] = attachment_kind
    return lifecycle


def _mcp_lifecycle(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = {
        "stage": contract.get("lifecycle_stage"),
        "capture_status": contract.get("capture_status"),
        "projection_mode": contract.get("projection_mode"),
        "projection_status": contract.get("projection_status"),
    }
    attachment_kind = contract.get("attachment_kind")
    if attachment_kind:
        lifecycle["attachment_kind"] = attachment_kind
    return lifecycle


def present_api_adjudication_requests(
    requests: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert adjudication request shape to REST camelCase."""
    return [
        {
            "requestId": request.get("request_id"),
            "ambiguityTags": request.get("ambiguity_tags", []),
            "selectedText": request.get("selected_text", ""),
            "candidateEvidence": [
                {
                    "evidenceId": item.get("evidence_id"),
                    "factClass": item.get("fact_class"),
                    "payload": item.get("payload", {}),
                }
                for item in request.get("candidate_evidence", [])
                if isinstance(item, Mapping)
            ],
            "instructions": request.get("instructions", ""),
        }
        for request in requests or []
    ]


def present_api_memory_write(
    contract: Mapping[str, Any],
    *,
    status: str | None = None,
    message: str | None = None,
    include_legacy_episode_id: bool = False,
) -> dict[str, Any]:
    """Format observe/remember results for REST while preserving legacy keys."""
    operation = contract.get("operation")
    response: dict[str, Any] = {
        "status": status or ("remembered" if operation == "remember" else "observed"),
        "episodeId": contract.get("episode_id"),
        "operation": operation,
        "lifecycle": _api_lifecycle(contract),
    }
    if include_legacy_episode_id:
        response["episode_id"] = contract.get("episode_id")
    if message:
        response["message"] = message
    adjudication_requests = present_api_adjudication_requests(
        contract.get("adjudication_requests"),
    )
    if adjudication_requests:
        response["adjudicationRequests"] = adjudication_requests
    return response


def present_api_observe_skip(
    status: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Format skipped auto-capture responses with lifecycle semantics."""
    response: dict[str, Any] = {
        "status": status,
        "operation": "observe",
        "lifecycle": {
            "stage": "capture",
            "captureStatus": "skipped",
            "projectionMode": None,
            "projectionStatus": None,
        },
    }
    if reason:
        response["reason"] = reason
    return response


def present_mcp_memory_write(
    contract: Mapping[str, Any],
    *,
    status: str = "stored",
    message: str,
) -> dict[str, Any]:
    """Format observe/remember results for MCP snake_case responses."""
    response: dict[str, Any] = {
        "status": status,
        "episode_id": contract.get("episode_id"),
        "message": message,
        "operation": contract.get("operation"),
        "lifecycle": _mcp_lifecycle(contract),
    }
    adjudication_requests = _copy_adjudication_requests(
        contract.get("adjudication_requests"),
    )
    if adjudication_requests:
        response["adjudication_requests"] = adjudication_requests
    return response
