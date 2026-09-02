"""Recall-in-flight gate: background indexing yields to the agent's recall.

The native worker pool is shared, so a recall's episode reads queue behind
the outbox drain's HNSW inserts. Measured live 2026-09-02 under a 962-item
drain: episode materialisation 1.2 s per recall, targeted lookups that cost
milliseconds standalone. The drain checks this gate between items; it never
blocks a recall, and it never waits longer than ``max_wait`` so a stuck
counter cannot stall indexing.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

_inflight = 0
_idle_since = time.monotonic()
_SETTLE_SECONDS = 0.25


@contextlib.asynccontextmanager
async def active() -> AsyncIterator[None]:
    """Mark an explicit recall as in flight for the duration of the block."""
    global _inflight, _idle_since
    _inflight += 1
    try:
        yield
    finally:
        _inflight -= 1
        if _inflight == 0:
            _idle_since = time.monotonic()


def inflight() -> int:
    return _inflight


async def wait_idle(max_wait: float = 2.0, poll: float = 0.05) -> float:
    """Wait until no recall is in flight (plus a short settle), bounded by ``max_wait``.

    Returns the seconds actually waited.
    """
    started = time.monotonic()
    while (
        _inflight > 0 or (time.monotonic() - _idle_since) < _SETTLE_SECONDS
    ) and (time.monotonic() - started) < max_wait:
        await asyncio.sleep(poll)
    return time.monotonic() - started
