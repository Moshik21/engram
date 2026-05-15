from __future__ import annotations

from collections.abc import Callable

import aiosqlite
import pytest

from engram.consolidation.store import SQLiteConsolidationStore
from engram.evaluation.store import SQLiteEvaluationStore
from engram.storage.sqlite.atlas import SQLiteAtlasStore
from engram.storage.sqlite.conversations import SQLiteConversationStore
from engram.storage.sqlite.feedback import SQLiteFeedbackStore
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore

BorrowedDbStoreFactory = Callable[[str], object]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_name", "store_factory"),
    [
        ("atlas", SQLiteAtlasStore),
        ("consolidation", SQLiteConsolidationStore),
        ("conversations", SQLiteConversationStore),
        ("evaluation", SQLiteEvaluationStore),
        ("feedback", SQLiteFeedbackStore),
        ("fts5_search", FTS5SearchIndex),
        ("vectors", SQLiteVectorStore),
    ],
)
async def test_sqlite_store_close_preserves_borrowed_connection(
    tmp_path,
    store_name: str,
    store_factory: BorrowedDbStoreFactory,
) -> None:
    db_path = tmp_path / f"{store_name}.db"
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    store = store_factory(str(db_path))
    await store.initialize(db=db)  # type: ignore[attr-defined]
    try:
        await store.close()  # type: ignore[attr-defined]

        row = await (await db.execute("SELECT 1 AS still_open")).fetchone()
        assert row["still_open"] == 1
    finally:
        await db.close()
