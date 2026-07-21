"""RF M4.1: periodic activation-snapshot saves (interval + dirty-count gated).

The shell lifespan task and the MCP stdio tool-call piggyback both route
through MemoryActivationStore.maybe_save_periodic, so a kill -9 mid-session
loses at most ~interval of accesses instead of everything since the last
clean shutdown. All existing ownership rules (owner-only save, mtime guard,
journal fold) still apply to periodic saves.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engram.config import ActivationConfig
from engram.storage.memory.activation import MemoryActivationStore

INTERVAL = 600.0
DIRTY_MIN = 50


async def _record_n(store: MemoryActivationStore, n: int, base: float) -> None:
    for i in range(n):
        await store.record_access(f"ent-{i}", base + i, group_id="g")


class TestStorePeriodicSave:
    @pytest.mark.asyncio
    async def test_due_save_is_durable_without_clean_shutdown(self, tmp_path: Path):
        """Fake clock: record events, advance past interval, trigger the
        periodic path — the snapshot is durable with NO shutdown call."""
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)
        path = tmp_path / "snap.json"

        saved = store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + INTERVAL + 1
        )

        assert saved == DIRTY_MIN
        # A fresh store (simulated restart after kill -9) restores the events.
        fresh = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        assert fresh.load_from_file(path) == DIRTY_MIN
        state = await fresh.get_activation("ent-0")
        assert state is not None
        assert state.access_history == [base]

    @pytest.mark.asyncio
    async def test_not_due_before_interval(self, tmp_path: Path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)
        path = tmp_path / "snap.json"

        saved = store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + 10
        )

        assert saved == 0
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_not_due_below_dirty_min(self, tmp_path: Path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN - 1, base)
        path = tmp_path / "snap.json"

        saved = store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + INTERVAL + 1
        )

        assert saved == 0
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_zero_interval_disables(self, tmp_path: Path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)

        assert not store.periodic_save_due(interval_seconds=0, dirty_min=0, now=base + 10_000)

    @pytest.mark.asyncio
    async def test_save_resets_dirty_and_rearms_timer(self, tmp_path: Path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)
        path = tmp_path / "snap.json"
        assert store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + INTERVAL + 1
        )

        # Dirty again, but inside the re-armed interval: not due.
        await _record_n(store, DIRTY_MIN, base + INTERVAL + 2)
        saved = store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + INTERVAL + 100
        )
        assert saved == 0

        # A full interval later it fires again (same 50 entities, re-dirtied).
        saved = store.maybe_save_periodic(
            path, interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=base + 2 * INTERVAL + 2
        )
        assert saved == DIRTY_MIN

    @pytest.mark.asyncio
    async def test_dirty_min_zero_still_requires_one_access(self, tmp_path: Path):
        """An idle store never rewrites an unchanged snapshot."""
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()

        assert not store.periodic_save_due(
            interval_seconds=INTERVAL, dirty_min=0, now=base + INTERVAL + 1
        )
        await store.record_access("ent-0", base, group_id="g")
        assert store.periodic_save_due(
            interval_seconds=INTERVAL, dirty_min=0, now=base + INTERVAL + 1
        )

    @pytest.mark.asyncio
    async def test_defer_rearms_without_saving(self, tmp_path: Path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)
        now = base + INTERVAL + 1

        assert store.periodic_save_due(interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=now)
        store.defer_periodic_save(now)
        assert not store.periodic_save_due(interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=now)
        # Dirty events stay counted: due again a full interval later.
        assert store.periodic_save_due(
            interval_seconds=INTERVAL, dirty_min=DIRTY_MIN, now=now + INTERVAL
        )


def _arm_mcp(monkeypatch, tmp_path: Path) -> MemoryActivationStore:
    import engram.mcp.server as mcp_server

    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    store = MemoryActivationStore()
    monkeypatch.setattr(mcp_server, "_runtime_activation_store", store)
    monkeypatch.setattr(mcp_server, "_activation_snapshot_loaded", True)
    monkeypatch.setattr(mcp_server, "_activation_snapshot_mtime_at_init", None)
    monkeypatch.setattr(mcp_server, "_activation_cfg", ActivationConfig())
    return store


class TestMcpToolCallPiggyback:
    @pytest.mark.asyncio
    async def test_periodic_save_without_clean_shutdown(self, monkeypatch, tmp_path: Path):
        """The MCP opportunistic path persists the snapshot mid-session."""
        import engram.brain_runtime as brain_runtime
        import engram.mcp.server as mcp_server

        store = _arm_mcp(monkeypatch, tmp_path)
        monkeypatch.setattr(brain_runtime, "serve_process_alive", lambda: False)
        monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)

        saved = mcp_server._maybe_save_activation_snapshot_periodic(now=base + INTERVAL + 1)

        assert saved == DIRTY_MIN
        path = tmp_path / "activation-snapshot.json"
        assert path.exists()
        # Ownership is NOT consumed (unlike the single-shot shutdown save)...
        assert mcp_server._activation_snapshot_loaded is True
        # ...and our own write refreshed the observed mtime, so the guard
        # still passes for the next owned save.
        assert mcp_server._activation_snapshot_mtime_at_init == path.stat().st_mtime

    @pytest.mark.asyncio
    async def test_skipped_while_shell_running_and_probe_deferred(self, monkeypatch, tmp_path):
        import engram.brain_runtime as brain_runtime
        import engram.mcp.server as mcp_server

        store = _arm_mcp(monkeypatch, tmp_path)
        probes = {"n": 0}

        def _alive() -> bool:
            probes["n"] += 1
            return True

        monkeypatch.setattr(brain_runtime, "serve_process_alive", _alive)
        monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)

        now = base + INTERVAL + 1
        assert mcp_server._maybe_save_activation_snapshot_periodic(now=now) == 0
        assert not (tmp_path / "activation-snapshot.json").exists()
        assert probes["n"] == 1
        # The refusal re-armed the timer: the expensive probe does not run
        # again on the next tool call inside the interval.
        assert mcp_server._maybe_save_activation_snapshot_periodic(now=now + 1) == 0
        assert probes["n"] == 1

    @pytest.mark.asyncio
    async def test_skipped_when_snapshot_changed_since_load(self, monkeypatch, tmp_path: Path):
        """A newer snapshot from another process must never be clobbered."""
        import json

        import engram.brain_runtime as brain_runtime
        import engram.mcp.server as mcp_server

        store = _arm_mcp(monkeypatch, tmp_path)
        monkeypatch.setattr(brain_runtime, "serve_process_alive", lambda: False)
        monkeypatch.setattr(brain_runtime, "shell_is_healthy", lambda *a, **kw: False)
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)

        path = tmp_path / "activation-snapshot.json"
        payload = {"saved_at": time.time(), "states": {"shell-ent": {}}}
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert mcp_server._maybe_save_activation_snapshot_periodic(now=base + INTERVAL + 1) == 0
        assert "shell-ent" in json.loads(path.read_text(encoding="utf-8"))["states"]

    @pytest.mark.asyncio
    async def test_noop_before_snapshot_load(self, monkeypatch, tmp_path: Path):
        import engram.mcp.server as mcp_server

        store = _arm_mcp(monkeypatch, tmp_path)
        monkeypatch.setattr(mcp_server, "_activation_snapshot_loaded", False)
        base = time.time()
        await _record_n(store, DIRTY_MIN, base)

        assert mcp_server._maybe_save_activation_snapshot_periodic(now=base + INTERVAL + 1) == 0
        assert not (tmp_path / "activation-snapshot.json").exists()


class TestShellPeriodicTask:
    @pytest.mark.asyncio
    async def test_task_started_and_disabled_by_zero_interval(self):
        from engram.config import EngramConfig
        from engram.main import _start_activation_snapshot_task

        store = MemoryActivationStore()
        config = EngramConfig(_env_file=None)
        task = _start_activation_snapshot_task(store, config)
        assert task is not None
        task.cancel()

        config.activation.activation_snapshot_interval_seconds = 0
        assert _start_activation_snapshot_task(store, config) is None
        # Stores without the periodic API (e.g. Redis) are skipped, not broken.
        assert _start_activation_snapshot_task(object(), config) is None
