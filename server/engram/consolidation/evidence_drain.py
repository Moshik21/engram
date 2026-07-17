"""Classify and drain deferred evidence backlog (audit + reject-junk)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

Disposition = Literal["keep", "reject_junk"]

_PATH_LIKE = re.compile(
    r"(^|[\s\"'`(])(?:~/?|/)?(?:Users|private|tmp|var|docs|claude-\d+|"
    r"Downloads|Desktop|Documents|Library)[/\\]",
    re.IGNORECASE,
)
_FILE_SUFFIX = re.compile(
    r"\.(?:md|tsx?|jsx?|py|json|sql|hx|yml|yaml|toml|xlsx?|csv|pdf)$",
    re.IGNORECASE,
)
_UUID_PATH = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/",
    re.IGNORECASE,
)
_SLASH_PAIR = re.compile(r"^[^/\s]{1,40}/[^/\s]{1,40}$")
_MARKUP_NOISE = re.compile(
    r"https?://|www\.|shields\.io|alt=\"|badge/|cpus:\s*|memory:\s*\d|"
    r"networks:\s*|docker-compose|image:\s*\w",
    re.IGNORECASE,
)
_BOOTSTRAP_MARKERS = (
    "[project-bootstrap|",
    "[session-start|",
    "README.md]",
    "CLAUDE.md]",
    "Agents.md]",
)
# Valid package-style left/right tokens (shadcn/ui, react-dom style keeps elsewhere).
_TECH_SLASH_TOKENS = frozenset(
    {
        "ui",
        "ux",
        "js",
        "ts",
        "tsx",
        "jsx",
        "go",
        "rs",
        "py",
        "io",
        "db",
        "sql",
        "pl",
        "api",
        "sdk",
        "cli",
        "vue",
        "ng",
        "css",
        "dom",
        "html",
        "wasm",
        "npm",
        "pnpm",
        "yarn",
        "shadcn",
        "radix",
        "next",
        "react",
        "node",
        "deno",
        "bun",
        "openai",
        "anthropic",
        "claude",
        "helix",
        "engram",
        "sqlite",
        "lite",
        "postgres",
        "redis",
        "falkor",
        "pdf",
        "renderer",
        "vercel",
        "vite",
        "vitest",
        "pytest",
        "docker",
        "k8s",
        "kubernetes",
    }
)
_RELATIONSHIP_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "to",
        "the",
        "a",
        "an",
        "of",
        "for",
        "with",
        "that",
        "this",
        "it",
        "its",
        "is",
        "are",
        "be",
        "as",
        "by",
        "from",
        "on",
        "in",
        "at",
        "more",
        "than",
        "into",
        "over",
        "under",
        "about",
        "via",
        "per",
        "vs",
        "vs.",
        "such",
        "so",
        "if",
        "then",
        "when",
        "while",
        "not",
        "no",
        "yes",
    }
)
_INCOMPLETE_ENDPOINT_SUFFIXES = (
    " to",
    " and",
    " the",
    " a",
    " an",
    " more",
    " for",
    " of",
    " with",
    " that",
    " this",
    " from",
    " into",
    " by",
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


def _normalize_endpoint(value: str) -> str:
    return value.strip().rstrip(".,;:!?\"'`)]}>").strip().lower()


def _is_broken_relationship_endpoint(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    if not re.search(r"[A-Za-z]", text):
        return True
    norm = _normalize_endpoint(text)
    if norm in _RELATIONSHIP_STOPWORDS:
        return True
    if len(text.split()) > 8:
        return True
    lowered = text.lower()
    if any(lowered.endswith(suffix) for suffix in _INCOMPLETE_ENDPOINT_SUFFIXES):
        return True
    # Trailing glue like "reason to" / "agents trust and"
    tokens = lowered.split()
    if tokens and tokens[-1] in _RELATIONSHIP_STOPWORDS:
        return True
    return False


def _is_token_slash_pair_junk(name: str) -> bool:
    """Reject extraction scrap like before/after, 5/6 — keep shadcn/ui style."""
    if not _SLASH_PAIR.match(name):
        return False
    left, right = name.split("/", 1)
    left_l, right_l = left.lower(), right.lower()
    if left_l in _TECH_SLASH_TOKENS or right_l in _TECH_SLASH_TOKENS:
        return False
    # Scoped npm-style packages only (ai-sdk/openai, react-pdf/renderer).
    # Do not exempt arbitrary hyphens (awarded/no-award, center/lower-middle).
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", left_l) and re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", right_l
    ):
        return False
    # Tiny / numeric scrap: A/B, N/A, 5/6, 1x/2x, 2-3/4
    if len(left) <= 2 or len(right) <= 2:
        return True
    if left.isdigit() or right.isdigit():
        return True
    if re.search(r"\d", left) or re.search(r"\d", right):
        return True
    # Word or word-word scrap pairs without tech tokens
    token = r"[A-Za-z][A-Za-z0-9_]{0,24}(?:-[A-Za-z0-9_]{1,24})?"
    if re.fullmatch(token, left) and re.fullmatch(token, right):
        return True
    return False


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
    payload = _parse_payload(row)

    if fact_class == "entity":
        if not name or name == "?":
            return EvidenceDisposition("reject_junk", "unknown_name")
        if "\n" in name or "\r" in name:
            return EvidenceDisposition("reject_junk", "markdown_fragment_name")
        if _PATH_LIKE.search(name) or name.startswith(
            (
                "docs/",
                "Users/",
                "private/",
                "tmp/",
                "Downloads/",
                "Desktop/",
                "Documents/",
                "server/",
            )
        ):
            return EvidenceDisposition("reject_junk", "path_like_name")
        if _FILE_SUFFIX.search(name):
            return EvidenceDisposition("reject_junk", "file_reference_name")
        if "claude-" in name and "/" in name:
            return EvidenceDisposition("reject_junk", "harness_path_name")
        if _UUID_PATH.search(name):
            return EvidenceDisposition("reject_junk", "harness_uuid_path")
        if _is_token_slash_pair_junk(name):
            return EvidenceDisposition("reject_junk", "token_slash_pair")
        if extractor == "identity_entity" and confidence <= 0.55:
            return EvidenceDisposition("reject_junk", "low_confidence_identity")

    if fact_class == "relationship":
        subject = str(payload.get("subject") or payload.get("source") or "").strip()
        obj = str(payload.get("object") or payload.get("target") or "").strip()
        if _is_broken_relationship_endpoint(subject) or _is_broken_relationship_endpoint(obj):
            return EvidenceDisposition("reject_junk", "broken_relationship_endpoint")

    if fact_class in {"attribute", "temporal"} and _MARKUP_NOISE.search(source_span):
        return EvidenceDisposition("reject_junk", "markup_noise_span")

    if any(marker in source_span for marker in _BOOTSTRAP_MARKERS):
        return EvidenceDisposition("reject_junk", "bootstrap_span")

    if "&middot;" in source_span or "  ·  " in source_span:
        return EvidenceDisposition("reject_junk", "html_nav_fragment")

    if _PATH_LIKE.search(source_span) and fact_class in {
        "entity",
        "relationship",
        "attribute",
        "temporal",
    }:
        return EvidenceDisposition("reject_junk", "path_like_span")

    return EvidenceDisposition("keep")


def select_junk_evidence_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deferred rows classified as junk, optionally capped.

    Prefer this over ``rows[:budget]`` so drain budgets fill with rejectable
    sludge instead of the first N keep-rows in store order.
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        if classify_deferred_evidence(row).disposition != "reject_junk":
            continue
        selected.append(row)
        if limit is not None and len(selected) >= max(0, int(limit)):
            break
    return selected


def audit_deferred_evidence(rows: list[dict[str, Any]]) -> DrainAuditSummary:
    """Summarize how deferred rows would be classified."""
    keep = 0
    reject = 0
    by_reason: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
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
            "token_slash_pair",
            "broken_relationship_endpoint",
            "markup_noise_span",
        )
    }

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


def candidate_as_evidence_row(candidate: Any) -> dict[str, Any]:
    """Project an EvidenceCandidate (or similar) into a drain classification row."""
    payload = getattr(candidate, "payload", None)
    if not isinstance(payload, dict):
        payload = {}
    return {
        "evidence_id": getattr(candidate, "evidence_id", "") or "",
        "fact_class": getattr(candidate, "fact_class", "") or "",
        "confidence": float(getattr(candidate, "confidence", 0.0) or 0.0),
        "extractor_name": getattr(candidate, "extractor_name", "") or "",
        "source_type": getattr(candidate, "source_type", "") or "",
        "source_span": getattr(candidate, "source_span", None) or "",
        "payload": payload,
        "status": "deferred",
    }


def classify_extraction_candidate(candidate: Any) -> EvidenceDisposition:
    """Hot-path classification so junk never enters the deferred swamp."""
    return classify_deferred_evidence(candidate_as_evidence_row(candidate))


def scaled_drain_budget(
    deferred_count: int,
    *,
    base_budget: int = 500,
    max_budget: int = 5000,
) -> int:
    """Grow the drain budget when debt is large so backlog can actually shrink.

    ~500/cycle cannot clear 18k rows while intake continues. Scale toward a
    soft cap while always keeping at least ``base_budget``.
    """
    base = max(1, int(base_budget))
    cap = max(base, int(max_budget))
    count = max(0, int(deferred_count))
    if count <= base:
        return base
    # Aim to clear ~25% of backlog per cycle under load, within [base, cap].
    target = max(base, count // 4)
    return min(cap, target)


def select_redundant_entity_evidence(
    rows: list[dict[str, Any]],
    existing_names: set[str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Entity deferred rows whose name already exists in the graph (casefold)."""
    if not existing_names:
        return []
    normalized = {n.casefold() for n in existing_names if n}
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("fact_class") or "") != "entity":
            continue
        # Never auto-drop client promotions — product write path.
        if str(row.get("source_type") or "") == "client_proposal":
            continue
        name = _entity_name(row)
        if not name or name.casefold() not in normalized:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= max(0, int(limit)):
            break
    return selected


def select_stale_low_value_evidence(
    rows: list[dict[str, Any]],
    *,
    max_age_days: float = 21.0,
    min_deferred_cycles: int = 5,
    limit: int | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Low-value deferred rows that aged out without corroboration.

    High-signal client proposals are preserved for real adjudication.
    Pattern-extractor sludge that never rose is rejected so the queue shrinks.
    """
    from engram.extraction.promotion import is_high_signal_entity_type

    clock = now or datetime.now(timezone.utc)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("source_type") or "") == "client_proposal":
            continue
        payload = _parse_payload(row)
        entity_type = str(payload.get("entity_type") or "")
        if is_high_signal_entity_type(entity_type):
            continue
        signals = row.get("corroborating_signals") or []
        corroboration_gated = (
            isinstance(signals, list)
            and "proper_name" in signals
            and "identity_pattern" not in signals
        )
        try:
            cycles = int(row.get("deferred_cycles") or 0)
        except (TypeError, ValueError):
            cycles = 0
        age_ok = False
        age_days = None
        created = row.get("created_at")
        if created is not None:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                elif isinstance(created, datetime):
                    created_dt = created
                else:
                    created_dt = None
                if created_dt is not None:
                    if created_dt.tzinfo is None:
                        created_dt = created_dt.replace(tzinfo=timezone.utc)
                    age_days = (clock - created_dt).total_seconds() / 86400.0
                    age_ok = age_days >= float(max_age_days)
            except Exception:
                age_ok = False
        if corroboration_gated:
            # Corroboration-gated rows (bare proper names) are WAITING for a
            # second cross-episode mention; the adjudication hold advances
            # deferred_cycles, so the cycles shortcut and the recovery floor
            # would drain them in hours instead of the intended 21-day window
            # (I2, docs/product/investigations/). Age is the only honest
            # staleness signal for them.
            if age_days is None or age_days < 21.0:
                continue
        elif cycles < int(min_deferred_cycles) and not age_ok:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= max(0, int(limit)):
            break
    return selected


def should_force_commit_evidence(row: dict[str, Any]) -> bool:
    """Force-commit is for high-signal facts, not pattern-extractor sludge."""
    from engram.extraction.promotion import is_high_signal_entity_type

    if str(row.get("source_type") or "") == "client_proposal":
        return True
    signals = row.get("corroborating_signals") or []
    if isinstance(signals, list) and "high_signal_type" in signals:
        return True
    if str(row.get("fact_class") or "") == "entity":
        payload = _parse_payload(row)
        return is_high_signal_entity_type(str(payload.get("entity_type") or ""))
    # Relationships/attrs from narrow extractors should not force-materialize
    # after aging — they are the growth engine of graph sludge.
    return False


async def reject_evidence_rows(
    graph_store: Any,
    *,
    group_id: str,
    rows: list[dict[str, Any]],
    reason_prefix: str,
    dry_run: bool = True,
    batch_size: int = 200,
    reason_for_row: Any | None = None,
) -> dict[str, Any]:
    """Reject an explicit list of evidence rows with a drain reason prefix."""
    rejected = 0
    errors = 0
    by_reason: Counter[str] = Counter()
    for row in rows:
        evidence_id = row.get("evidence_id")
        if not evidence_id:
            errors += 1
            continue
        if reason_for_row is not None:
            reason = str(reason_for_row(row) or "unspecified")
        else:
            reason = "unspecified"
        by_reason[reason] += 1
        if dry_run:
            rejected += 1
            continue
        try:
            await graph_store.update_evidence_status(
                evidence_id,
                "rejected",
                updates={"commit_reason": f"{reason_prefix}:{reason}"},
                group_id=group_id,
            )
            rejected += 1
            if rejected % batch_size == 0:
                logger.info("Rejected %d evidence rows (%s) so far", rejected, reason_prefix)
        except Exception:
            errors += 1
            logger.debug("Failed to reject evidence %s", evidence_id, exc_info=True)
    return {
        "dry_run": dry_run,
        "total": len(rows),
        "rejected": rejected,
        "errors": errors,
        "by_reason": dict(by_reason),
    }


async def reject_junk_evidence(
    graph_store: Any,
    *,
    group_id: str,
    rows: list[dict[str, Any]],
    dry_run: bool = True,
    batch_size: int = 200,
    prioritize_junk: bool = False,
    max_reject: int | None = None,
) -> dict[str, Any]:
    """Reject classified junk deferred evidence rows.

    When ``prioritize_junk`` is True, only junk rows are considered (optionally
    capped by ``max_reject``). Prefer this over slicing ``rows[:budget]`` so a
    drain budget is not wasted on keep-rows that appear first in store order.
    """
    work = select_junk_evidence_rows(rows, limit=max_reject) if prioritize_junk else list(rows)
    rejected = 0
    kept = 0
    by_reason: Counter[str] = Counter()
    errors = 0

    for row in work:
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
        "total": len(work),
        "pool_size": len(rows),
        "rejected": rejected,
        "kept": kept,
        "errors": errors,
        "by_reason": dict(by_reason),
        "prioritize_junk": prioritize_junk,
    }
