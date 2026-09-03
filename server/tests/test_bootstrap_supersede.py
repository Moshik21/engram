"""Stale project-bootstrap snapshots are superseded: newest per file survives."""

from __future__ import annotations

import pytest

from engram.consolidation.bootstrap_supersede import (
    snapshot_key,
    stale_snapshots,
    supersede_bootstrap_snapshots,
)

pytestmark = pytest.mark.asyncio


def _row(eid: str, path: str, created: str, project: str = "Engram") -> dict:
    return {
        "episode_id": eid,
        "content": f"[project-bootstrap|{project}|{path}]\n# doc {created}",
        "created_at": created,
    }


def test_snapshot_key_reads_the_header_only() -> None:
    assert snapshot_key("[project-bootstrap|Engram|docs/a.md]\nbody") == ("Engram", "docs/a.md")
    assert snapshot_key("[assistant|Engram] not a snapshot") is None
    assert snapshot_key("") is None


def test_stale_is_everything_but_the_newest_per_file() -> None:
    rows = [
        _row("e1", "docs/a.md", "2026-07-01T00:00:00"),
        _row("e2", "docs/a.md", "2026-09-01T00:00:00"),
        _row("e3", "docs/a.md", "2026-08-01T00:00:00"),
        _row("e4", "docs/b.md", "2026-08-01T00:00:00"),
        _row("e5", "docs/a.md", "2026-08-01T00:00:00", project="other"),
        {"episode_id": "e6", "content": "[user|Engram] chat", "created_at": "2026-09-02T00:00:00"},
    ]
    assert sorted(r["episode_id"] for r in stale_snapshots(rows)) == ["e1", "e3"]


class _Graph:
    def __init__(self, rows):
        self.rows = rows
        self.purged: list[str] = []

    async def _query(self, endpoint, payload):
        assert endpoint == "find_episodes_by_group"
        return list(self.rows)

    async def purge_episode(self, episode_id, group_id):
        self.purged.append(episode_id)
        return True


class _Index:
    def __init__(self) -> None:
        self.purged: list[str] = []

    async def purge_episode_vectors(self, episode_id, group_id):
        self.purged.append(episode_id)
        return 3


async def test_supersede_purges_stale_rows_within_budget_and_reports() -> None:
    rows = [
        _row("e1", "docs/a.md", "2026-07-01T00:00:00"),
        _row("e2", "docs/a.md", "2026-09-01T00:00:00"),
        _row("e3", "docs/a.md", "2026-08-01T00:00:00"),
    ]
    graph, index = _Graph(rows), _Index()
    result = await supersede_bootstrap_snapshots(graph, index, "default", budget=1)
    assert result.scanned == 3 and result.files == 1 and result.stale == 2
    assert result.purged == 1 and result.vectors_deleted == 3
    assert graph.purged == index.purged and graph.purged[0] in {"e1", "e3"}


async def test_dry_run_touches_nothing() -> None:
    rows = [_row("e1", "docs/a.md", "2026-07-01"), _row("e2", "docs/a.md", "2026-09-01")]
    graph, index = _Graph(rows), _Index()
    result = await supersede_bootstrap_snapshots(graph, index, "default", budget=10, dry_run=True)
    assert result.purged == 1 and graph.purged == [] and index.purged == []
