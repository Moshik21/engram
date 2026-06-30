"""Tests for the public engram showcase demo path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.showcase.beats import SHOWCASE_BEATS
from engram.showcase.export import export_showcase_payload
from engram.showcase.resources import (
    _atomic_copy_db,
    bundled_demo_db_path,
    bundled_demo_db_source,
    prepare_showcase_db,
    showcase_runtime_db_path,
)
from engram.showcase.runner import (
    format_showcase_run,
    run_showcase_beats,
    showcase_open_instructions,
)
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
    results, _runtime = await run_showcase_beats(db_path=db_path)

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


@pytest.mark.asyncio
async def test_prepare_showcase_db_copies_bundled_source_to_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "demo-run.db"
    prepared = prepare_showcase_db(copy_to=runtime)

    assert prepared == runtime
    assert runtime.is_file()
    source = bundled_demo_db_source()
    assert runtime.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_bundled_demo_db_unchanged_after_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = bundled_demo_db_source()
    before = source.read_bytes()

    runtime = tmp_path / "demo-run.db"
    monkeypatch.setattr(
        "engram.showcase.resources.showcase_runtime_db_path",
        lambda: runtime,
    )

    results, resolved = await run_showcase_beats()
    assert resolved == runtime
    assert all(result.passed for result in results)
    assert source.read_bytes() == before


@pytest.mark.asyncio
async def test_default_showcase_run_uses_stable_runtime_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engram.showcase.resources.SHOWCASE_CACHE_DIR",
        tmp_path / "showcase",
    )

    _results, resolved = await run_showcase_beats()
    assert resolved == showcase_runtime_db_path()
    assert resolved.is_file()


@pytest.mark.asyncio
async def test_showcase_run_is_idempotent_across_consecutive_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engram.showcase.resources.SHOWCASE_CACHE_DIR",
        tmp_path / "showcase",
    )

    first_results, _ = await run_showcase_beats()
    second_results, _ = await run_showcase_beats()

    assert all(result.passed for result in first_results)
    assert all(result.passed for result in second_results)


@pytest.mark.asyncio
async def test_packaged_demo_db_path_without_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged_bytes = (
        Path(__file__).resolve().parent.parent / "engram" / "data" / "demo.db"
    ).read_bytes()
    fake_cache = tmp_path / "cache" / "demo.db"
    monkeypatch.setattr(
        "engram.showcase.resources._source_tree_demo_db",
        lambda: tmp_path / "missing" / "demo.db",
    )
    monkeypatch.setattr(
        "engram.showcase.resources.SHOWCASE_CACHE_DIR",
        tmp_path / "cache",
    )
    monkeypatch.setattr(
        "engram.showcase.resources._materialize_packaged_demo_db",
        lambda cache_path: _materialize_packaged_fixture(cache_path, packaged_bytes),
    )

    source = bundled_demo_db_source()
    assert source == fake_cache
    assert source.read_bytes() == packaged_bytes

    runtime = tmp_path / "cache" / "demo-run.db"
    monkeypatch.setattr(
        "engram.showcase.resources.showcase_runtime_db_path",
        lambda: runtime,
    )
    results, resolved = await run_showcase_beats()
    assert resolved == runtime
    assert all(result.passed for result in results)


def _materialize_packaged_fixture(cache_path: Path, payload: bytes) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return cache_path


def test_showcase_open_instructions_use_runtime_path(tmp_path: Path) -> None:
    runtime = tmp_path / "demo-run.db"
    instructions = showcase_open_instructions(runtime)
    assert str(runtime) in instructions
    assert "ENGRAM_SQLITE__PATH=" in instructions


def test_atomic_copy_db_replaces_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    dest = tmp_path / "dest.db"
    source.write_bytes(b"fresh-db")
    dest.write_bytes(b"stale-db")

    _atomic_copy_db(source, dest)

    assert dest.read_bytes() == b"fresh-db"
