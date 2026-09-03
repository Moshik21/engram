"""Supersede stale project-bootstrap snapshots.

Bootstrap stores a new `[project-bootstrap|<project>|<path>]` episode every
time a file's content hash changes and never retires the previous one.
Measured 2026-09-03 on the dogfood brain: 695 distinct files held 4,081
snapshot episodes -- 3,386 stale versions, 34% of the whole corpus -- and they
crowded both recall lanes (22 of the keyword lane's top 40 for one question)
while inflating the document frequency of the project's own name.

Only the newest snapshot per (project, path) is knowledge; the rest are edit
history. This pass keeps the newest and purges the others: episode node,
cues, and every vector row (episode, cue, chunk), bounded per window.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^\[project-bootstrap\|(?P<project>[^|\]\n]+)\|(?P<path>[^\]\n]+)\]")


@dataclass
class BootstrapSupersedeResult:
    scanned: int = 0
    files: int = 0
    stale: int = 0
    purged: int = 0
    vectors_deleted: int = 0
    errors: int = 0
    dry_run: bool = False
    purged_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "files": self.files,
            "stale": self.stale,
            "purged": self.purged,
            "vectors_deleted": self.vectors_deleted,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


def snapshot_key(content: str | None) -> tuple[str, str] | None:
    m = _HEADER_RE.match((content or "").lstrip())
    if not m:
        return None
    return m.group("project").strip(), m.group("path").strip()


def stale_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every bootstrap row that is not the newest for its (project, path)."""
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    stale: list[dict[str, Any]] = []
    for row in rows:
        key = snapshot_key(row.get("content"))
        if key is None:
            continue
        current = newest.get(key)
        if current is None:
            newest[key] = row
            continue
        # created_at is ISO-8601; string order is time order.
        if str(row.get("created_at") or "") > str(current.get("created_at") or ""):
            stale.append(current)
            newest[key] = row
        else:
            stale.append(row)
    return stale


async def supersede_bootstrap_snapshots(
    graph_store: Any,
    search_index: Any,
    group_id: str,
    *,
    budget: int = 200,
    dry_run: bool = False,
) -> BootstrapSupersedeResult:
    result = BootstrapSupersedeResult(dry_run=dry_run)
    rows = await graph_store._query("find_episodes_by_group", {"gid": group_id})
    result.scanned = len(rows)
    keys = {snapshot_key(r.get("content")) for r in rows}
    keys.discard(None)
    result.files = len(keys)
    stale = stale_snapshots(rows)
    result.stale = len(stale)
    purge_vectors = getattr(search_index, "purge_episode_vectors", None)
    purge_episode = getattr(graph_store, "purge_episode", None)
    for row in stale[: max(0, int(budget))]:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            continue
        if dry_run:
            result.purged += 1
            result.purged_ids.append(episode_id)
            continue
        try:
            if callable(purge_vectors):
                result.vectors_deleted += int(await purge_vectors(episode_id, group_id) or 0)
            if callable(purge_episode):
                await purge_episode(episode_id, group_id)
            result.purged += 1
            result.purged_ids.append(episode_id)
        except Exception:
            result.errors += 1
            logger.warning("bootstrap supersede: purge failed for %s", episode_id, exc_info=True)
    if result.purged:
        logger.info(
            "Bootstrap supersede: purged=%d stale=%d files=%d vectors=%d dry_run=%s",
            result.purged, result.stale, result.files, result.vectors_deleted, dry_run,
        )
    return result
