"""Shared UTC datetime helpers.

The storage layer historically persists naive UTC timestamps. These helpers
keep that behavior stable while avoiding deprecated ``datetime.utcnow()`` APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone

# Optional process-global clock override. ``None`` in production (the
# real wall clock is used). A test/eval harness may set this to a fixed
# naive-UTC ``datetime`` so every ``utc_now()`` caller — including the ~36
# modules that import the name directly — observes one frozen instant
# without per-module patching. This is the single seam the depth-tier eval
# uses to pin entity ``updated_at``/``created_at`` for determinism.
_now_override: datetime | None = None


def set_now_override(value: datetime | None) -> None:
    """Pin (or clear) the process-global ``utc_now`` value. Eval/test only."""

    global _now_override
    _now_override = value


def utc_now() -> datetime:
    """Return the current UTC time as a naive datetime.

    Engram's SQLite/Falkor stores and API serializers currently assume naive
    UTC values and append ``Z`` at the edge. Preserve that convention here so
    warning cleanup does not silently change on-disk timestamp formats.
    """

    if _now_override is not None:
        return _now_override
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""

    return utc_now().isoformat()
