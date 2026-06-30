"""Tests for the public engram showcase demo path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.showcase.beats import SHOWCASE_BEATS
from engram.showcase.export import export_showcase_payload
from engram.showcase.resources import bundled_demo_db_path
from engram.showcase.runner import format_showcase_run, run_showcase_beats
from engram.showcase.seed import seed_demo_db


@pytest.mark.asyncio
async def test_seed_demo_db_creates_liam_entities(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.db"
    await seed_demo_db(db_path, group_id="showcase")

    from engram.storage.sqlite.graph import SQLiteGraphStore

    store = SQLiteGraphStore(str(db_path))
    await store.initialize()
    try:
        entities = await store.find_entities(name="Liam", group_id="showcase")
        assert len(entities) == 1
        identity = await store.get_identity_core_entities("showcase")
        assert [entity.name for entity in identity] == ["Liam"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_showcase_run_hits_liam_continuity(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.db"
    await seed_demo_db(db_path, group_id="showcase")
    results = await run_showcase_beats(db_path=db_path)

    assert len(results) == len(SHOWCASE_BEATS)
    assert all(result.passed for result in results)
    continuity = results[0]
    assert "liam" in continuity.matched_tokens
    output = format_showcase_run(results)
    assert "Liam" in output or "liam" in output.lower()
    assert "3/3 beats passed" in output


@pytest.mark.asyncio
async def test_showcase_export_writes_three_beats(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.db"
    await seed_demo_db(db_path, group_id="showcase")
    out_path = tmp_path / "export.json"
    payload = await export_showcase_payload(db_path=db_path, out_path=out_path)

    assert out_path.is_file()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(loaded["beats"]) >= 3
    assert loaded["summary"]["passed"] == 3
    assert payload["kind"] == "engram_showcase_export"


def test_bundled_demo_db_is_packaged() -> None:
    path = bundled_demo_db_path()
    assert path.is_file()
    assert path.stat().st_size > 0