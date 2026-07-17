"""M1.1: activation snapshot warms brain and MCP store-open paths.

The shell saves the snapshot at shutdown and owns writes; the brain and
one-shot CLI paths load it read-only so prune protections see real usage.
MCP stdio loads at store build and saves back only when it loaded the file
and no shell is running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from engram.config import EngramConfig
from engram.models.activation import ActivationState
from engram.storage.bootstrap import open_local_stores
from engram.storage.memory.activation import (
    MemoryActivationStore,
    activation_snapshot_path,
)
from engram.storage.resolver import EngineMode


def _write_snapshot(home: Path, saved_at: float, entity_id: str = "ent-1") -> Path:
    payload = {
        "saved_at": saved_at,
        "states": {
            entity_id: {
                "node_id": entity_id,
                "access_history": [saved_at - 60.0, saved_at - 30.0],
                "spreading_bonus": 0.0,
                "last_accessed": saved_at - 30.0,
                "access_count": 5,
                "consolidated_strength": 0.2,
                "last_compacted": 0.0,
                "ts_alpha": 1.0,
                "ts_beta": 1.0,
                "group_id": "default",
            }
        },
    }
    path = home / "activation-snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeGraphStore:
    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass


class FakeSearchIndex:
    async def initialize(self, db: object | None = None) -> None:
        pass

    async def close(self) -> None:
        pass


def _patch_local_runtime_stores(monkeypatch) -> MemoryActivationStore:
    activation = MemoryActivationStore()

    def fake_create(mode: EngineMode, config: EngramConfig):
        return FakeGraphStore(), activation, FakeSearchIndex()

    monkeypatch.setattr(
        "engram.storage.bootstrap.create_local_runtime_stores",
        fake_create,
    )
    return activation


def test_snapshot_path_honors_engram_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    assert activation_snapshot_path() == tmp_path / "activation-snapshot.json"


async def test_brain_path_open_local_stores_loads_snapshot(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    _write_snapshot(tmp_path, time.time())
    activation = _patch_local_runtime_stores(monkeypatch)
    config = EngramConfig(_env_file=None)

    async with open_local_stores(
        config,
        mode=EngineMode.LITE,
        local_runtime=True,
        load_activation_snapshot=True,
    ) as stores:
        assert stores.activation_store is activation
        state = await stores.activation_store.get_activation("ent-1")
        assert state is not None
        assert state.access_count == 5
        assert len(state.access_history) == 2


async def test_open_local_stores_ignores_stale_snapshot(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    _write_snapshot(tmp_path, time.time() - 30 * 86400.0)  # beyond 14d max age
    _patch_local_runtime_stores(monkeypatch)
    config = EngramConfig(_env_file=None)

    async with open_local_stores(
        config,
        mode=EngineMode.LITE,
        local_runtime=True,
        load_activation_snapshot=True,
    ) as stores:
        assert await stores.activation_store.get_activation("ent-1") is None


async def test_open_local_stores_default_does_not_load(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    _write_snapshot(tmp_path, time.time())
    _patch_local_runtime_stores(monkeypatch)
    config = EngramConfig(_env_file=None)

    async with open_local_stores(
        config,
        mode=EngineMode.LITE,
        local_runtime=True,
    ) as stores:
        assert await stores.activation_store.get_activation("ent-1") is None


async def test_mcp_init_loads_snapshot(monkeypatch, tmp_path) -> None:
    import engram.mcp.server as mcp_server

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    _write_snapshot(tmp_path, time.time())

    activation = MemoryActivationStore()

    async def fake_resolve_mode(mode: str) -> EngineMode:
        return EngineMode.LITE

    def fake_create_stores(mode: EngineMode, config: EngramConfig):
        return FakeGraphStore(), activation, FakeSearchIndex()

    class FakeManager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def close_runtime_resources(self) -> None:
            pass

    monkeypatch.setattr(mcp_server, "resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(mcp_server, "create_stores", fake_create_stores)
    monkeypatch.setattr(mcp_server, "create_extractor", lambda config: object())
    monkeypatch.setattr(mcp_server, "GraphManager", FakeManager)
    monkeypatch.setattr(mcp_server, "_should_start_mcp_background_runtime", lambda config: False)
    monkeypatch.setattr(mcp_server, "_should_start_mcp_cue_index_outbox", lambda config: False)
    monkeypatch.setattr(mcp_server, "_should_start_mcp_redis_publisher", lambda mode: False)

    await mcp_server._init()
    try:
        state = await activation.get_activation("ent-1")
        assert state is not None
        assert state.access_count == 5
        assert mcp_server._activation_snapshot_loaded is True
        assert mcp_server._runtime_activation_store is activation
    finally:
        # Disarm the exit save so teardown never probes the host shell.
        monkeypatch.setattr(mcp_server, "_activation_snapshot_loaded", False)
        await mcp_server._shutdown()


def _arm_mcp_save(monkeypatch, tmp_path) -> MemoryActivationStore:
    import engram.mcp.server as mcp_server

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    store = MemoryActivationStore()
    now = time.time()
    store._states["ent-1"] = ActivationState(
        node_id="ent-1",
        access_history=[now],
        last_accessed=now,
        access_count=1,
    )
    monkeypatch.setattr(mcp_server, "_runtime_activation_store", store)
    monkeypatch.setattr(mcp_server, "_activation_snapshot_loaded", True)
    # Arm the full ownership state: file absent at init (mtime None).
    monkeypatch.setattr(mcp_server, "_activation_snapshot_mtime_at_init", None)
    return store


def test_mcp_save_skipped_when_shell_running(monkeypatch, tmp_path) -> None:
    import engram.brain_runtime as brain_runtime
    import engram.mcp.server as mcp_server

    _arm_mcp_save(monkeypatch, tmp_path)
    monkeypatch.setattr(brain_runtime, "serve_process_alive", lambda: True)
    monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)

    mcp_server._save_activation_snapshot_if_owner()

    assert not (tmp_path / "activation-snapshot.json").exists()
    assert mcp_server._activation_snapshot_loaded is True  # ownership untouched


def test_mcp_save_when_no_shell(monkeypatch, tmp_path) -> None:
    import engram.brain_runtime as brain_runtime
    import engram.mcp.server as mcp_server

    _arm_mcp_save(monkeypatch, tmp_path)
    monkeypatch.setattr(brain_runtime, "serve_process_alive", lambda: False)
    monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)

    mcp_server._save_activation_snapshot_if_owner()

    path = tmp_path / "activation-snapshot.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "ent-1" in payload["states"]
    assert mcp_server._activation_snapshot_loaded is False  # single-shot save


def test_mcp_save_skipped_when_snapshot_changed_since_load(monkeypatch, tmp_path) -> None:
    """A snapshot written by another process after our load must never be clobbered."""
    import engram.brain_runtime as brain_runtime
    import engram.mcp.server as mcp_server

    _arm_mcp_save(monkeypatch, tmp_path)
    monkeypatch.setattr(brain_runtime, "serve_process_alive", lambda: False)
    monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)

    # Another process (a restarted shell) wrote a newer snapshot after our init.
    path = tmp_path / "activation-snapshot.json"
    payload = {"saved_at": time.time(), "states": {"shell-ent": {}}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    mcp_server._save_activation_snapshot_if_owner()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "shell-ent" in payload["states"]  # newer file untouched
    assert "ent-1" not in payload.get("states", {})
    assert mcp_server._activation_snapshot_loaded is True  # ownership not consumed
