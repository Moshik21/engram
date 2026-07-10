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


def _copy_committed_entities(
    entities: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [dict(entity) for entity in entities or [] if isinstance(entity, Mapping)]


def _copy_committed_relationships(
    relationships: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [
        dict(relationship)
        for relationship in relationships or []
        if isinstance(relationship, Mapping)
    ]


def memory_write_contract(
    operation: MemoryWriteOperation,
    episode_id: str,
    *,
    adjudication_requests: Sequence[Mapping[str, Any]] | None = None,
    attachment_kind: str | None = None,
    committed_entities: Sequence[Mapping[str, Any]] | None = None,
    committed_relationships: Sequence[Mapping[str, Any]] | None = None,
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
        "committed_entities": _copy_committed_entities(committed_entities),
        "committed_relationships": _copy_committed_relationships(committed_relationships),
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


def present_api_committed_entities(
    entities: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert committed entity rows to REST camelCase."""
    rows: list[dict[str, Any]] = []
    for entity in entities or []:
        if not isinstance(entity, Mapping):
            continue
        row: dict[str, Any] = {
            "id": entity.get("id"),
            "name": entity.get("name"),
            "entityType": entity.get("entity_type") or entity.get("entityType"),
            "identityCore": bool(
                entity.get("identity_core")
                if "identity_core" in entity
                else entity.get("identityCore")
            ),
        }
        summary = entity.get("summary")
        if summary:
            row["summary"] = summary
        rows.append(row)
    return rows


def present_api_committed_relationships(
    relationships: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert committed relationship rows to REST camelCase."""
    rows: list[dict[str, Any]] = []
    for rel in relationships or []:
        if not isinstance(rel, Mapping):
            continue
        rows.append(
            {
                "id": rel.get("id"),
                "subject": rel.get("subject"),
                "predicate": rel.get("predicate"),
                "object": rel.get("object"),
            }
        )
    return rows


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
    committed_entities = present_api_committed_entities(
        contract.get("committed_entities"),
    )
    if committed_entities:
        response["committedEntities"] = committed_entities
    committed_relationships = present_api_committed_relationships(
        contract.get("committed_relationships"),
    )
    if committed_relationships:
        response["committedRelationships"] = committed_relationships
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


def present_mcp_committed_entities(
    entities: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Copy committed entity rows for MCP snake_case responses."""
    rows: list[dict[str, Any]] = []
    for entity in entities or []:
        if not isinstance(entity, Mapping):
            continue
        row: dict[str, Any] = {
            "id": entity.get("id"),
            "name": entity.get("name"),
            "entity_type": entity.get("entity_type") or entity.get("entityType"),
            "identity_core": bool(
                entity.get("identity_core")
                if "identity_core" in entity
                else entity.get("identityCore")
            ),
        }
        summary = entity.get("summary")
        if summary:
            row["summary"] = summary
        rows.append(row)
    return rows


def present_mcp_committed_relationships(
    relationships: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Copy committed relationship rows for MCP snake_case responses."""
    return _copy_committed_relationships(relationships)


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
    committed_entities = present_mcp_committed_entities(
        contract.get("committed_entities"),
    )
    if committed_entities:
        response["committed_entities"] = committed_entities
    committed_relationships = present_mcp_committed_relationships(
        contract.get("committed_relationships"),
    )
    if committed_relationships:
        response["committed_relationships"] = committed_relationships
    return response
