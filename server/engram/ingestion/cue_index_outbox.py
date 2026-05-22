"""Durable outbox for episode-cue vector indexing."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram.models.episode_cue import EpisodeCue


@dataclass(frozen=True)
class CueIndexOutboxItem:
    """A cue waiting for vector indexing."""

    cue: EpisodeCue
    attempts: int = 0
    last_error: str | None = None


class CueIndexOutbox:
    """SQLite-backed retry queue for cue vector indexing."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    @property
    def path(self) -> Path:
        return self._path

    def enqueue(self, cue: EpisodeCue) -> None:
        """Persist a cue before in-process indexing is scheduled."""
        self._ensure_schema()
        payload = json.dumps(cue.model_dump(mode="json"), sort_keys=True)
        now = time.time()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO cue_index_outbox (
                    episode_id, group_id, cue_json, status, attempts,
                    last_error, created_at, updated_at
                )
                VALUES (?, ?, ?, 'pending', 0, NULL, ?, ?)
                ON CONFLICT(episode_id, group_id) DO UPDATE SET
                    cue_json = excluded.cue_json,
                    status = 'pending',
                    last_error = NULL,
                    updated_at = excluded.updated_at
                """,
                (cue.episode_id, cue.group_id, payload, now, now),
            )
            conn.commit()

    def pending(self, *, limit: int = 100) -> list[CueIndexOutboxItem]:
        """Return pending or failed cues for retry."""
        self._ensure_schema()
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT cue_json, attempts, last_error
                FROM cue_index_outbox
                WHERE status IN ('pending', 'failed')
                ORDER BY updated_at ASC
                LIMIT ?
                """,
                (max(1, int(limit or 1)),),
            ).fetchall()
        items: list[CueIndexOutboxItem] = []
        for cue_json, attempts, last_error in rows:
            cue = self._cue_from_json(cue_json)
            if cue is None:
                continue
            items.append(
                CueIndexOutboxItem(
                    cue=cue,
                    attempts=int(attempts or 0),
                    last_error=str(last_error) if last_error else None,
                ),
            )
        return items

    def mark_done(self, *, episode_id: str, group_id: str) -> None:
        """Remove a successfully indexed cue from the outbox."""
        self._ensure_schema()
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM cue_index_outbox WHERE episode_id = ? AND group_id = ?",
                (episode_id, group_id),
            )
            conn.commit()

    def mark_failed(self, *, episode_id: str, group_id: str, error: str) -> None:
        """Keep a cue retryable after an indexing failure."""
        self._ensure_schema()
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE cue_index_outbox
                SET status = 'failed',
                    attempts = attempts + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE episode_id = ? AND group_id = ?
                """,
                (error[:500], time.time(), episode_id, group_id),
            )
            conn.commit()

    def pending_count(self) -> int:
        """Return the number of cues still awaiting successful indexing."""
        self._ensure_schema()
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM cue_index_outbox WHERE status IN ('pending', 'failed')"
            ).fetchone()
        return int(row[0] if row else 0)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=1.0)

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cue_index_outbox (
                    episode_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    cue_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (episode_id, group_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cue_index_outbox_status_updated
                ON cue_index_outbox(status, updated_at)
                """
            )
            conn.commit()
        self._initialized = True

    @staticmethod
    def _cue_from_json(value: str) -> EpisodeCue | None:
        try:
            raw: dict[str, Any] = json.loads(value)
            return EpisodeCue.model_validate(raw)
        except Exception:
            return None
