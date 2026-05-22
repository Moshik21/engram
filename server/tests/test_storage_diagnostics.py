import asyncio
import time
from pathlib import Path

import pytest

from engram.config import EngramConfig
from engram.storage import diagnostics as diagnostics_module
from engram.storage.diagnostics import (
    StorageDiagnostics,
    collect_storage_paths,
    resolve_helix_native_data_dir,
)


class FakeGraphStore:
    def __init__(self, stats: dict) -> None:
        self.stats = stats
        self.group_ids: list[str] = []

    async def get_stats(self, group_id: str) -> dict:
        self.group_ids.append(group_id)
        return self.stats


class SlowGraphStore:
    async def get_stats(self, group_id: str) -> dict:
        await asyncio.sleep(0.2)
        return {
            "episodes": 9,
            "entities": 9,
            "relationships": 9,
            "cue_metrics": {"cue_count": 9},
        }


@pytest.mark.asyncio
async def test_storage_diagnostics_reports_paths_counts_and_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engram_home = tmp_path / "home" / ".engram"
    helix_dir = tmp_path / "helix-native"
    sqlite_path = tmp_path / "engram.db"
    log_path = engram_home / "logs" / "engram.log"
    queue_path = engram_home / "capture-queue.jsonl"

    helix_dir.mkdir()
    engram_home.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)
    (helix_dir / "data.mdb").write_bytes(b"x" * 1024)
    sqlite_path.write_bytes(b"s" * 128)
    queue_path.write_text("{}", encoding="utf-8")
    log_path.write_text("started\n", encoding="utf-8")
    monkeypatch.setenv("ENGRAM_HOME", str(engram_home))

    config = EngramConfig(mode="helix")
    config.helix.transport = "native"
    config.helix.data_dir = str(helix_dir)
    config.sqlite.path = str(sqlite_path)
    graph_store = FakeGraphStore(
        {
            "episodes": 3,
            "entities": 5,
            "relationships": 7,
            "cue_metrics": {"cue_count": 2},
        }
    )

    diagnostics = await StorageDiagnostics.create(
        config=config,
        mode="helix",
        graph_store=graph_store,
        group_id="default",
    )

    (helix_dir / "growth.bin").write_bytes(b"g" * 512)
    graph_store.stats = {
        "episodes": 4,
        "entities": 6,
        "relationships": 9,
        "cue_metrics": {"cue_count": 3},
    }

    snapshot = await diagnostics.snapshot(group_id="default")

    assert snapshot["counts"] == {
        "episodes": 3,
        "entities": 5,
        "relationships": 7,
        "cues": 2,
    }
    assert snapshot["growthSinceStartup"]["episodes"] == 0
    assert snapshot["growthSinceStartup"]["bytes"] == 0
    assert snapshot["diagnostics"]["countsStatus"] == "cached"
    assert graph_store.group_ids == ["default"]

    snapshot = await diagnostics.snapshot(group_id="default", live=True)

    assert snapshot["backend"] == "helix_native"
    assert snapshot["groupId"] == "default"
    assert snapshot["counts"] == {
        "episodes": 4,
        "entities": 6,
        "relationships": 9,
        "cues": 3,
    }
    assert snapshot["growthSinceStartup"]["episodes"] == 1
    assert snapshot["growthSinceStartup"]["entities"] == 1
    assert snapshot["growthSinceStartup"]["relationships"] == 2
    assert snapshot["growthSinceStartup"]["cues"] == 1
    assert snapshot["growthSinceStartup"]["bytes"] >= 512
    assert snapshot["disk"]["totalBytes"] >= snapshot["disk"]["startupBytes"]
    assert snapshot["diagnostics"]["live"] is True
    assert snapshot["diagnostics"]["countsStatus"] == "live"
    assert snapshot["diagnostics"]["pathsStatus"] == "live"
    assert {item["label"] for item in snapshot["paths"]} >= {
        "Helix native data",
        "Packet cache",
        "Cue index outbox",
        "SQLite companion",
        "Capture queue",
        "Server log",
    }
    assert graph_store.group_ids == ["default", "default"]


@pytest.mark.asyncio
async def test_storage_diagnostics_default_counts_use_write_through_deltas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engram_home = tmp_path / "home" / ".engram"
    engram_home.mkdir(parents=True)
    monkeypatch.setenv("ENGRAM_HOME", str(engram_home))
    config = EngramConfig(mode="helix")
    config.helix.transport = "native"
    config.helix.data_dir = str(tmp_path / "helix-native")
    config.sqlite.path = str(tmp_path / "engram.db")
    graph_store = FakeGraphStore(
        {
            "episodes": 3,
            "entities": 5,
            "relationships": 7,
            "cue_metrics": {"cue_count": 2},
        }
    )
    diagnostics = await StorageDiagnostics.create(
        config=config,
        mode="helix",
        graph_store=graph_store,
        group_id="default",
    )

    diagnostics.record_counts_delta(
        "default",
        episodes=1,
        entities=2,
        relationships=3,
        cues=1,
    )
    graph_store.stats = {
        "episodes": 99,
        "entities": 99,
        "relationships": 99,
        "cue_metrics": {"cue_count": 99},
    }

    snapshot = await diagnostics.snapshot(group_id="default")

    assert snapshot["counts"] == {
        "episodes": 4,
        "entities": 7,
        "relationships": 10,
        "cues": 3,
    }
    assert snapshot["growthSinceStartup"] == {
        "bytes": 0,
        "episodes": 1,
        "entities": 2,
        "relationships": 3,
        "cues": 1,
    }
    assert snapshot["diagnostics"]["countsStatus"] == "write_through"
    assert graph_store.group_ids == ["default"]


@pytest.mark.asyncio
async def test_storage_diagnostics_startup_timeout_uses_empty_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EngramConfig(mode="helix")
    config.helix.transport = "native"
    config.helix.data_dir = str(tmp_path / "helix-native")
    config.sqlite.path = str(tmp_path / "engram.db")

    def slow_collect_paths(_config: EngramConfig, _mode: str) -> list[dict]:
        time.sleep(0.2)
        return [{"bytes": 1024}]

    monkeypatch.setattr(diagnostics_module, "collect_storage_paths", slow_collect_paths)

    diagnostics = await StorageDiagnostics.create(
        config=config,
        mode="helix",
        graph_store=SlowGraphStore(),
        group_id="default",
        startup_timeout_seconds=0.01,
    )

    assert diagnostics.startup_counts == {
        "episodes": 0,
        "entities": 0,
        "relationships": 0,
        "cues": 0,
    }
    assert diagnostics.startup_bytes == 0

    snapshot = await diagnostics.snapshot(live=False)
    assert snapshot["counts"] == {
        "episodes": 0,
        "entities": 0,
        "relationships": 0,
        "cues": 0,
    }
    assert snapshot["diagnostics"]["countsStatus"] == "cached"
    assert snapshot["diagnostics"]["pathsStatus"] == "cached"

    live_snapshot = await diagnostics.snapshot(live=True, timeout_seconds=0.01)
    assert live_snapshot["counts"] == {
        "episodes": 0,
        "entities": 0,
        "relationships": 0,
        "cues": 0,
    }
    assert live_snapshot["diagnostics"]["countsStatus"] in {"cached_timeout", "timeout"}


def test_storage_paths_use_native_default_when_unconfigured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = EngramConfig(mode="helix")
    config.helix.transport = "native"
    config.helix.data_dir = ""
    config.sqlite.path = str(tmp_path / "engram.db")

    assert resolve_helix_native_data_dir(config) == tmp_path / ".helix" / "engram-native"

    paths = collect_storage_paths(config, "helix")

    assert paths[0]["label"] == "Helix native data"
    assert paths[0]["path"] == str(tmp_path / ".helix" / "engram-native")
