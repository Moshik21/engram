"""Offline queue for auto-observe resilience.

When the REST server is unreachable, clients can append entries to a local
JSONL file (~/.engram/capture-queue.jsonl). On the next session start (or
via the /api/knowledge/replay-queue endpoint), queued entries are replayed
and ingested.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path.home() / ".engram" / "capture-queue.jsonl"


def append_to_queue(
    entry: dict,
    queue_path: Path | None = None,
) -> None:
    """Append a single entry to the offline queue (JSONL)."""
    path = queue_path or DEFAULT_QUEUE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def drain_queue(
    queue_path: Path | None = None,
) -> list[dict]:
    """Read and remove all entries from the offline queue.

    Returns the list of entries. The queue file is truncated atomically
    by renaming away, so concurrent writers don't lose data.
    """
    path = queue_path or DEFAULT_QUEUE_PATH
    if not path.exists():
        return []

    tmp = path.with_suffix(".jsonl.draining")
    try:
        os.rename(str(path), str(tmp))
    except OSError:
        return []

    entries: list[dict] = []
    try:
        with open(tmp) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed queue entry at line %d", line_no)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass

    logger.info("Drained %d entries from offline queue", len(entries))
    return entries
