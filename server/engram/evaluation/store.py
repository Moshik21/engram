"""SQLite-backed local evaluation sample store."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiosqlite

from engram.benchmark.metrics import RecallEvalSample, SessionContinuitySample

RECALL_RUNTIME_SNAPSHOT_RETENTION = 25


@dataclass
class StoredRecallEvalSample:
    """Persisted label for one recall decision."""

    group_id: str
    recall_triggered: bool
    recall_helped: bool
    packets_surfaced: int = 0
    packets_used: int = 0
    false_recalls: int = 0
    recall_needed: bool | None = None
    source: str = "manual"
    query: str | None = None
    notes: str | None = None
    id: str = field(default_factory=lambda: f"ers_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    def to_sample(self) -> RecallEvalSample:
        return RecallEvalSample(
            recall_triggered=self.recall_triggered,
            recall_helped=self.recall_helped,
            packets_surfaced=self.packets_surfaced,
            packets_used=self.packets_used,
            false_recalls=self.false_recalls,
            recall_needed=self.recall_needed,
        )


@dataclass
class StoredSessionContinuitySample:
    """Persisted label for one multi-turn continuity task."""

    group_id: str
    baseline_score: float
    memory_score: float
    open_loop_expected: bool = False
    open_loop_recovered: bool = False
    temporal_expected: bool = False
    temporal_correct: bool = False
    source: str = "manual"
    scenario: str | None = None
    notes: str | None = None
    id: str = field(default_factory=lambda: f"esc_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    def to_sample(self) -> SessionContinuitySample:
        return SessionContinuitySample(
            baseline_score=self.baseline_score,
            memory_score=self.memory_score,
            open_loop_expected=self.open_loop_expected,
            open_loop_recovered=self.open_loop_recovered,
            temporal_expected=self.temporal_expected,
            temporal_correct=self.temporal_correct,
        )


@dataclass
class StoredRecallRuntimeMetricsSnapshot:
    """Persisted runtime Recall Gate metrics for reopened reports."""

    group_id: str
    metrics: dict[str, Any]
    source: str = "runtime"
    id: str = field(default_factory=lambda: f"erm_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


class SQLiteEvaluationStore:
    """Stores local labels used by the brain-loop evaluation report."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._owns_db = False

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        if db:
            self._db = db
            self._owns_db = False
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            self._owns_db = True

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_recall_samples (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                recall_triggered INTEGER NOT NULL,
                recall_helped INTEGER NOT NULL,
                packets_surfaced INTEGER NOT NULL DEFAULT 0,
                packets_used INTEGER NOT NULL DEFAULT 0,
                false_recalls INTEGER NOT NULL DEFAULT 0,
                recall_needed INTEGER,
                source TEXT NOT NULL DEFAULT 'manual',
                query TEXT,
                notes TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self._ensure_column(
            "evaluation_recall_samples",
            "recall_needed",
            "INTEGER",
        )
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_recall_group_time
                ON evaluation_recall_samples(group_id, timestamp)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_session_samples (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                baseline_score REAL NOT NULL,
                memory_score REAL NOT NULL,
                open_loop_expected INTEGER NOT NULL DEFAULT 0,
                open_loop_recovered INTEGER NOT NULL DEFAULT 0,
                temporal_expected INTEGER NOT NULL DEFAULT 0,
                temporal_correct INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual',
                scenario TEXT,
                notes TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_session_group_time
                ON evaluation_session_samples(group_id, timestamp)
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_recall_runtime_metrics (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'runtime',
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_recall_runtime_group_time
                ON evaluation_recall_runtime_metrics(group_id, timestamp)
        """)
        await self.db.commit()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteEvaluationStore not initialized.")
        return self._db

    async def save_recall_sample(self, sample: StoredRecallEvalSample) -> None:
        await self.db.execute(
            "INSERT INTO evaluation_recall_samples "
            "(id, group_id, recall_triggered, recall_helped, packets_surfaced, "
            "packets_used, false_recalls, recall_needed, source, query, notes, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample.id,
                sample.group_id,
                int(sample.recall_triggered),
                int(sample.recall_helped),
                max(0, sample.packets_surfaced),
                max(0, sample.packets_used),
                max(0, sample.false_recalls),
                None if sample.recall_needed is None else int(sample.recall_needed),
                sample.source,
                sample.query,
                sample.notes,
                sample.timestamp,
            ),
        )
        await self.db.commit()

    async def save_session_sample(self, sample: StoredSessionContinuitySample) -> None:
        await self.db.execute(
            "INSERT INTO evaluation_session_samples "
            "(id, group_id, baseline_score, memory_score, open_loop_expected, "
            "open_loop_recovered, temporal_expected, temporal_correct, source, "
            "scenario, notes, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sample.id,
                sample.group_id,
                sample.baseline_score,
                sample.memory_score,
                int(sample.open_loop_expected),
                int(sample.open_loop_recovered),
                int(sample.temporal_expected),
                int(sample.temporal_correct),
                sample.source,
                sample.scenario,
                sample.notes,
                sample.timestamp,
            ),
        )
        await self.db.commit()

    async def save_recall_metrics_snapshot(
        self,
        snapshot: StoredRecallRuntimeMetricsSnapshot,
    ) -> None:
        await self.db.execute(
            "INSERT INTO evaluation_recall_runtime_metrics "
            "(id, group_id, metrics_json, source, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                snapshot.id,
                snapshot.group_id,
                json.dumps(snapshot.metrics, sort_keys=True),
                snapshot.source,
                snapshot.timestamp,
            ),
        )
        await self.db.execute(
            "DELETE FROM evaluation_recall_runtime_metrics "
            "WHERE group_id = ? AND id NOT IN ("
            "SELECT id FROM evaluation_recall_runtime_metrics "
            "WHERE group_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?"
            ")",
            (
                snapshot.group_id,
                snapshot.group_id,
                RECALL_RUNTIME_SNAPSHOT_RETENTION,
            ),
        )
        await self.db.commit()

    async def get_latest_recall_metrics_snapshot(self, group_id: str) -> dict[str, Any]:
        cursor = await self.db.execute(
            "SELECT metrics_json FROM evaluation_recall_runtime_metrics "
            "WHERE group_id = ? ORDER BY timestamp DESC LIMIT 1",
            (group_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return {}
        payload = json.loads(row["metrics_json"])
        if isinstance(payload, dict):
            return payload
        return {}

    async def get_recall_samples(
        self,
        group_id: str,
        limit: int = 500,
    ) -> list[RecallEvalSample]:
        cursor = await self.db.execute(
            "SELECT * FROM evaluation_recall_samples WHERE group_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (group_id, max(1, limit)),
        )
        rows = await cursor.fetchall()
        return [self._row_to_recall_sample(row).to_sample() for row in rows]

    async def get_session_samples(
        self,
        group_id: str,
        limit: int = 500,
    ) -> list[SessionContinuitySample]:
        cursor = await self.db.execute(
            "SELECT * FROM evaluation_session_samples WHERE group_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (group_id, max(1, limit)),
        )
        rows = await cursor.fetchall()
        return [self._row_to_session_sample(row).to_sample() for row in rows]

    async def close(self) -> None:
        if self._db and self._owns_db:
            await self._db.close()
        self._db = None
        self._owns_db = False

    @staticmethod
    def _row_to_recall_sample(row) -> StoredRecallEvalSample:
        return StoredRecallEvalSample(
            id=row["id"],
            group_id=row["group_id"],
            recall_triggered=bool(row["recall_triggered"]),
            recall_helped=bool(row["recall_helped"]),
            packets_surfaced=row["packets_surfaced"],
            packets_used=row["packets_used"],
            false_recalls=row["false_recalls"],
            recall_needed=(
                None if row["recall_needed"] is None else bool(row["recall_needed"])
            ),
            source=row["source"],
            query=row["query"],
            notes=row["notes"],
            timestamp=row["timestamp"],
        )

    async def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cursor = await self.db.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        if any(row[1] == column for row in rows):
            return
        await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @staticmethod
    def _row_to_session_sample(row) -> StoredSessionContinuitySample:
        return StoredSessionContinuitySample(
            id=row["id"],
            group_id=row["group_id"],
            baseline_score=row["baseline_score"],
            memory_score=row["memory_score"],
            open_loop_expected=bool(row["open_loop_expected"]),
            open_loop_recovered=bool(row["open_loop_recovered"]),
            temporal_expected=bool(row["temporal_expected"]),
            temporal_correct=bool(row["temporal_correct"]),
            source=row["source"],
            scenario=row["scenario"],
            notes=row["notes"],
            timestamp=row["timestamp"],
        )
