"""Classify and drain deferred evidence backlog (audit + reject-junk)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

Disposition = Literal["keep", "reject_junk"]

_PATH_LIKE = re.compile(
    r"(^|[\s\"'`(])(?:~/?|/)?(?:Users|private|tmp|var|docs|claude-\d+)[/\\]",
    re.IGNORECASE,
)
_FILE_SUFFIX = re.compile(r"\.(?:md|tsx?|jsx?|py|json|sql|hx|yml|yaml|toml)$", re.IGNORECASE)
_UUID_PATH = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/",
    re.IGNORECASE,
)
_BOOTSTRAP_MARKERS = (
    "[project-bootstrap|",
    "[session-start|",
    "README.md]",
    "CLAUDE.md]",
    "Agents.md]",
)


@dataclass(frozen=True)
class EvidenceDisposition:
    disposition: Disposition
    reason: str | None = None


@dataclass(frozen=True)
class DrainAuditSummary:
    total: int
    keep: int
    reject_junk: int
    by_reason: dict[str, int]
    samples: dict[str, list[dict[str, Any]]]


def _parse_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") or row.get("payload_json") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {}
    return payload if isinstance(payload, dict) else {}


def _entity_name(row: dict[str, Any]) -> str:
    if row.get("fact_class") != "entity":
        return ""
    payload = _parse_payload(row)
    name = payload.get("name")
    return str(name).strip() if name is not None else ""


def classify_deferred_evidence(row: dict[str, Any]) -> EvidenceDisposition:
    """Return keep vs reject_junk for a deferred evidence row."""
    source_span = str(row.get("source_span") or "")
    extractor = str(row.get("extractor_name") or "")
    fact_class = str(row.get("fact_class") or "")
    try:
        confidence = float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    name = _entity_name(row)

    if fact_class == "entity":
        if not name or name == "?":
            return EvidenceDisposition("reject_junk", "unknown_name")
        if "\n" in name or "\r" in name:
            return EvidenceDisposition("reject_junk", "markdown_fragment_name")
        if _PATH_LIKE.search(name) or name.startswith(("docs/", "Users/", "private/", "tmp/")):
            return EvidenceDisposition("reject_junk", "path_like_name")
        if _FILE_SUFFIX.search(name):
            return EvidenceDisposition("reject_junk", "file_reference_name")
        if "claude-" in name and "/" in name:
            return EvidenceDisposition("reject_junk", "harness_path_name")
        if _UUID_PATH.search(name):
            return EvidenceDisposition("reject_junk", "harness_uuid_path")
        if extractor == "identity_entity" and confidence <= 0.55:
            return EvidenceDisposition("reject_junk", "low_confidence_identity")

    if any(marker in source_span for marker in _BOOTSTRAP_MARKERS):
        return EvidenceDisposition("reject_junk", "bootstrap_span")

    if "&middot;" in source_span or "  ·  " in source_span:
        return EvidenceDisposition("reject_junk", "html_nav_fragment")

    if _PATH_LIKE.search(source_span) and fact_class in {"entity", "relationship"}:
        return EvidenceDisposition("reject_junk", "path_like_span")

    return EvidenceDisposition("keep")


def audit_deferred_evidence(rows: list[dict[str, Any]]) -> DrainAuditSummary:
    """Summarize how deferred rows would be classified."""
    keep = 0
    reject = 0
    by_reason: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in (
        "unknown_name",
        "path_like_name",
        "file_reference_name",
        "harness_path_name",
        "harness_uuid_path",
        "low_confidence_identity",
        "bootstrap_span",
        "html_nav_fragment",
        "path_like_span",
        "markdown_fragment_name",
    )}

    for row in rows:
        result = classify_deferred_evidence(row)
        if result.disposition == "reject_junk":
            reject += 1
            reason = result.reason or "unspecified"
            by_reason[reason] += 1
            bucket = samples.get(reason)
            if bucket is not None and len(bucket) < 3:
                payload = _parse_payload(row)
                bucket.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "fact_class": row.get("fact_class"),
                        "confidence": row.get("confidence"),
                        "name": payload.get("name"),
                        "source_span": str(row.get("source_span") or "")[:160],
                    },
                )
        else:
            keep += 1

    return DrainAuditSummary(
        total=len(rows),
        keep=keep,
        reject_junk=reject,
        by_reason=dict(by_reason),
        samples=samples,
    )


async def load_deferred_evidence(graph_store: Any, group_id: str) -> list[dict[str, Any]]:
    """Load all deferred evidence rows for a group."""
    loader = getattr(graph_store, "find_evidence_by_status", None)
    if callable(loader):
        return await loader(group_id, "deferred")

    query_rows = getattr(graph_store, "_query_open_status_rows", None)
    if callable(query_rows):

        rows = await query_rows(
            "find_evidence_by_status",
            "find_pending_evidence",
            group_id,
            ("deferred",),
        )
        converter = getattr(graph_store, "_evidence_dict_to_storage", None)
        if callable(converter):
            return [converter(d) for d in rows]
        return list(rows)

    pending = await graph_store.get_pending_evidence(group_id=group_id, limit=100_000)
    return [row for row in pending if row.get("status") == "deferred"]


async def reject_junk_evidence(
    graph_store: Any,
    *,
    group_id: str,
    rows: list[dict[str, Any]],
    dry_run: bool = True,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Reject classified junk deferred evidence rows."""
    rejected = 0
    kept = 0
    by_reason: Counter[str] = Counter()
    errors = 0

    for row in rows:
        result = classify_deferred_evidence(row)
        if result.disposition != "reject_junk":
            kept += 1
            continue

        reason = result.reason or "unspecified"
        by_reason[reason] += 1
        evidence_id = row.get("evidence_id")
        if not evidence_id:
            errors += 1
            continue

        if dry_run:
            rejected += 1
            continue

        try:
            await graph_store.update_evidence_status(
                evidence_id,
                "rejected",
                updates={"commit_reason": f"drain_evidence:{reason}"},
                group_id=group_id,
            )
            rejected += 1
            if rejected % batch_size == 0:
                logger.info("Rejected %d junk evidence rows so far", rejected)
        except Exception:
            errors += 1
            logger.debug("Failed to reject evidence %s", evidence_id, exc_info=True)

    return {
        "dry_run": dry_run,
        "total": len(rows),
        "rejected": rejected,
        "kept": kept,
        "errors": errors,
        "by_reason": dict(by_reason),
    }
