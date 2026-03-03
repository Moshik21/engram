"""SQLite-backed implicit feedback store."""

from __future__ import annotations

import time

import aiosqlite

from engram.models.feedback import FeedbackEvent, FeedbackStats


class SQLiteFeedbackStore:
    """Stores implicit feedback events in SQLite for learn-to-rank."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        """Create feedback_events table if it doesn't exist."""
        if db:
            self._db = db
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback_events (
                id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                query TEXT NOT NULL,
                group_id TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_entity ON feedback_events(entity_id, group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_group ON feedback_events(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_events(timestamp)"
        )
        await self.db.commit()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteFeedbackStore not initialized.")
        return self._db

    async def record_event(self, event: FeedbackEvent) -> None:
        """Record a single feedback event."""
        await self.db.execute(
            "INSERT INTO feedback_events "
            "(id, entity_id, event_type, query, group_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.entity_id,
                event.event_type,
                event.query,
                event.group_id,
                event.timestamp,
            ),
        )
        await self.db.commit()

    async def get_entity_feedback(
        self,
        entity_id: str,
        group_id: str,
    ) -> list[FeedbackEvent]:
        """Get all feedback events for an entity in a group."""
        cursor = await self.db.execute(
            "SELECT id, entity_id, event_type, query, group_id, timestamp "
            "FROM feedback_events WHERE entity_id = ? AND group_id = ? "
            "ORDER BY timestamp DESC",
            (entity_id, group_id),
        )
        rows = await cursor.fetchall()
        return [
            FeedbackEvent(
                id=row["id"],
                entity_id=row["entity_id"],
                event_type=row["event_type"],
                query=row["query"],
                group_id=row["group_id"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    async def get_feedback_stats(self, group_id: str) -> dict[str, FeedbackStats]:
        """Get aggregated feedback stats per entity for a group."""
        cursor = await self.db.execute(
            "SELECT entity_id, event_type, COUNT(*) as cnt "
            "FROM feedback_events WHERE group_id = ? "
            "GROUP BY entity_id, event_type",
            (group_id,),
        )
        rows = await cursor.fetchall()

        stats: dict[str, FeedbackStats] = {}
        for row in rows:
            eid = row["entity_id"]
            if eid not in stats:
                stats[eid] = FeedbackStats(entity_id=eid)
            s = stats[eid]
            cnt = row["cnt"]
            s.total_events += cnt
            event_type = row["event_type"]
            if event_type == "returned":
                s.returned_count = cnt
            elif event_type == "re_accessed":
                s.re_accessed_count = cnt
            elif event_type == "mentioned_in_remember":
                s.mentioned_count = cnt
            elif event_type == "ignored":
                s.ignored_count = cnt

        return stats

    async def cleanup(self, ttl_days: int = 90) -> int:
        """Delete feedback events older than ttl_days. Returns count deleted."""
        cutoff = time.time() - (ttl_days * 86400)
        cursor = await self.db.execute(
            "DELETE FROM feedback_events WHERE timestamp < ?",
            (cutoff,),
        )
        await self.db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
