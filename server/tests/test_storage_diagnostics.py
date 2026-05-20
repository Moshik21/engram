from pathlib import Path

import pytest

from engram.config import EngramConfig
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
    assert {item["label"] for item in snapshot["paths"]} >= {
        "Helix native data",
        "SQLite companion",
        "Capture queue",
        "Server log",
    }
    assert graph_store.group_ids == ["default", "default"]


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
