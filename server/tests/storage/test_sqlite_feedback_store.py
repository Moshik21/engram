from __future__ import annotations

import aiosqlite
import pytest

from engram.models.feedback import FeedbackEvent
from engram.storage.sqlite.feedback import SQLiteFeedbackStore


@pytest.mark.asyncio
async def test_sqlite_feedback_store_does_not_close_borrowed_db_connection(
    tmp_path,
) -> None:
    db = await aiosqlite.connect(tmp_path / "feedback.db")
    db.row_factory = aiosqlite.Row
    store = SQLiteFeedbackStore(str(tmp_path / "feedback.db"))
    await store.initialize(db=db)
    try:
        await store.record_event(
            FeedbackEvent(
                id="fb_one",
                entity_id="ent_one",
                event_type="returned",
                query="what did we decide",
                group_id="default",
                timestamp=1.0,
            )
        )
        await store.close()

        row = await (await db.execute("SELECT COUNT(*) FROM feedback_events")).fetchone()
        assert row[0] == 1
    finally:
        await db.close()
