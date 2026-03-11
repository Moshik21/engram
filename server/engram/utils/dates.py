"""Shared UTC datetime helpers.

The storage layer historically persists naive UTC timestamps. These helpers
keep that behavior stable while avoiding deprecated ``datetime.utcnow()`` APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a naive datetime.

    Engram's SQLite/Falkor stores and API serializers currently assume naive
    UTC values and append ``Z`` at the edge. Preserve that convention here so
    warning cleanup does not silently change on-disk timestamp formats.
    """

    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""

    return utc_now().isoformat()
