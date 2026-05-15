"""Shared public-surface helpers for episode adjudication work items."""

from __future__ import annotations

import inspect
from typing import Any


async def load_episode_adjudication_requests(
    manager: Any,
    *,
    episode_id: str,
    group_id: str,
) -> list[dict]:
    """Return episode adjudication work items when supported by the manager."""
    getter = getattr(manager, "get_episode_adjudications", None)
    if getter is None:
        return []
    result = getter(episode_id, group_id)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, list) else []


async def build_api_adjudication_resolution_surface(
    manager: Any,
    *,
    group_id: str,
    request_id: str,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
    reject_evidence_ids: list[str] | None = None,
    model_tier: str = "default",
    rationale: str | None = None,
) -> dict:
    """Resolve an adjudication request and present the REST response shape."""
    outcome = await _submit_adjudication_resolution(
        manager,
        group_id=group_id,
        request_id=request_id,
        entities=entities,
        relationships=relationships,
        reject_evidence_ids=reject_evidence_ids,
        model_tier=model_tier,
        rationale=rationale,
    )
    return {
        "status": outcome.status,
        "requestId": outcome.request_id,
        "committedIds": outcome.committed_ids,
        "supersededEvidenceIds": outcome.superseded_evidence_ids,
        "replacementEvidenceIds": outcome.replacement_evidence_ids,
    }


async def build_mcp_adjudication_resolution_surface(
    manager: Any,
    *,
    group_id: str,
    request_id: str,
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
    reject_evidence_ids: list[str] | None = None,
    model_tier: str = "default",
    rationale: str | None = None,
) -> dict:
    """Resolve an adjudication request and present the MCP response shape."""
    outcome = await _submit_adjudication_resolution(
        manager,
        group_id=group_id,
        request_id=request_id,
        entities=entities,
        relationships=relationships,
        reject_evidence_ids=reject_evidence_ids,
        model_tier=model_tier,
        rationale=rationale,
    )
    return {
        "status": outcome.status,
        "request_id": outcome.request_id,
        "committed_ids": outcome.committed_ids,
        "superseded_evidence_ids": outcome.superseded_evidence_ids,
        "replacement_evidence_ids": outcome.replacement_evidence_ids,
    }


async def _submit_adjudication_resolution(
    manager: Any,
    *,
    group_id: str,
    request_id: str,
    entities: list[dict] | None,
    relationships: list[dict] | None,
    reject_evidence_ids: list[str] | None,
    model_tier: str,
    rationale: str | None,
) -> Any:
    return await manager.submit_adjudication_resolution(
        request_id,
        entities=entities,
        relationships=relationships,
        reject_evidence_ids=reject_evidence_ids,
        source="client_adjudication",
        model_tier=model_tier,
        rationale=rationale,
        group_id=group_id,
    )
