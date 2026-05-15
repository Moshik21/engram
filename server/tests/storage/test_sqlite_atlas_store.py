from __future__ import annotations

import aiosqlite
import pytest

from engram.storage.sqlite.atlas import SQLiteAtlasStore


@pytest.mark.asyncio
async def test_sqlite_atlas_store_does_not_close_borrowed_db_connection(
    tmp_path,
) -> None:
    db = await aiosqlite.connect(tmp_path / "atlas.db")
    db.row_factory = aiosqlite.Row
    store = SQLiteAtlasStore(str(tmp_path / "atlas.db"))
    await store.initialize(db=db)
    try:
        await store.close()

        row = await (await db.execute("SELECT COUNT(*) FROM atlas_snapshots")).fetchone()
        assert row[0] == 0
    finally:
        await db.close()
