"""Contract for the eval-scoped ``utc_now`` override seam.

The depth-tier eval pins a single deterministic ``now`` across ingest + recall
via ``set_now_override``. These tests lock in that the override is
behavior-preserving by default (``None`` => real clock) and that it reaches both
``utc_now`` and ``utc_now_iso`` while set.
"""

from __future__ import annotations

from datetime import datetime

from engram.utils import dates


def test_default_uses_real_clock() -> None:
    """With no override, two calls reflect the moving wall clock."""
    assert dates._now_override is None
    a = dates.utc_now()
    b = dates.utc_now()
    assert b >= a  # monotonic-ish; real clock, not pinned


def test_override_pins_utc_now_and_iso() -> None:
    pinned = datetime(2026, 5, 28, 12, 0, 0)
    try:
        dates.set_now_override(pinned)
        assert dates.utc_now() == pinned
        assert dates.utc_now() == pinned  # stable across calls
        assert dates.utc_now_iso() == pinned.isoformat()
    finally:
        dates.set_now_override(None)
    assert dates._now_override is None
    # cleared => real clock again
    assert dates.utc_now() != pinned
