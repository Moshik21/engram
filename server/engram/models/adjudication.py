"""Structured work items for edge adjudication."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from engram.utils.dates import utc_now


@dataclass
class AdjudicationRequest:
    """A grouped ambiguous-memory work item awaiting resolution."""

    episode_id: str
    group_id: str = "default"
    ambiguity_tags: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    selected_text: str = ""
    request_reason: str = ""
    status: str = "pending"
    resolution_source: str | None = None
    resolution_payload: dict | None = None
    attempt_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    resolved_at: datetime | None = None
    request_id: str = field(default_factory=lambda: f"adj_{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict:
        """Serialize to a storage-friendly dict."""
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "status": self.status,
            "ambiguity_tags": list(self.ambiguity_tags),
            "evidence_ids": list(self.evidence_ids),
            "selected_text": self.selected_text,
            "request_reason": self.request_reason,
            "resolution_source": self.resolution_source,
            "resolution_payload": self.resolution_payload,
            "attempt_count": self.attempt_count,
            "created_at": (
                self.created_at.isoformat()
                if hasattr(self.created_at, "isoformat")
                else str(self.created_at)
            ),
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at is not None and hasattr(self.resolved_at, "isoformat")
                else self.resolved_at
            ),
        }


def adjudication_request_from_dict(data: dict) -> AdjudicationRequest:
    """Hydrate a storage row back into an AdjudicationRequest."""
    created_at = data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = utc_now()
    elif not isinstance(created_at, datetime):
        created_at = utc_now()

    resolved_at = data.get("resolved_at")
    if isinstance(resolved_at, str):
        try:
            resolved_at = datetime.fromisoformat(resolved_at)
        except ValueError:
            resolved_at = None
    elif not isinstance(resolved_at, datetime):
        resolved_at = None

    return AdjudicationRequest(
        request_id=data.get("request_id", ""),
        episode_id=data.get("episode_id", ""),
        group_id=data.get("group_id", "default"),
        status=data.get("status", "pending"),
        ambiguity_tags=list(data.get("ambiguity_tags", [])),
        evidence_ids=list(data.get("evidence_ids", [])),
        selected_text=data.get("selected_text", ""),
        request_reason=data.get("request_reason", ""),
        resolution_source=data.get("resolution_source"),
        resolution_payload=dict(data.get("resolution_payload", {}) or {})
        if data.get("resolution_payload") is not None
        else None,
        attempt_count=int(data.get("attempt_count", 0) or 0),
        created_at=created_at,
        resolved_at=resolved_at,
    )
