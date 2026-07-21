"""Advisory single-owner lock for the helix-native LMDB data dir (I3).

Both entry points that open the native env in-process — `engram serve`
(main.py) and `engram mcp` stdio (mcp/server.py) — take this flock so a
second local process is refused fast instead of silently sharing the env.
Evidence: docs/product/investigations/I3_mcp_concurrent_open.md.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.storage.resolver import EngineMode

NATIVE_SHELL_LOCK_FILENAME = "engram-shell.lock"
_native_shell_locks: dict[str, Any] = {}


def _native_backend_selected(config: EngramConfig, mode: EngineMode) -> bool:
    """Whether this init will open the helix-native (in-process LMDB) backend."""
    if mode != EngineMode.HELIX:
        return False
    transport = config.helix.transport
    if transport == "native":
        return True
    if transport == "auto":
        import importlib.util

        return importlib.util.find_spec("helix_native") is not None
    return False


def _native_data_dir(config: EngramConfig) -> Path:
    """Resolved native LMDB data dir (mirrors config.get_packet_cache_path)."""
    return (
        Path(config.helix.data_dir).expanduser()
        if config.helix.data_dir
        else Path.home() / ".helix" / "engram-native"
    )


def _acquire_native_shell_lock(data_dir: Path) -> None:
    """I3: exclusive advisory flock so two processes never share one native dir.

    LMDB's own cross-process writer lock is a non-robust POSIX semaphore on
    macOS (a SIGKILLed session wedges every later writer), and each session
    keeps derived state (activation snapshot ownership, cue outbox, engine
    cache) in process memory — so a silent second open loses updates even when
    LMDB behaves. A sentinel file INSIDE the data dir (separate from LMDB's
    data.mdb/lock.mdb) is flocked LOCK_EX|LOCK_NB, mirroring
    engram.brain_runtime.exclusive_brain_lock, and held for process lifetime;
    flock auto-releases if the process dies, so a killed session never wedges
    the next one. On conflict, startup fails fast naming the holder PID.
    """
    import fcntl

    lock_path = data_dir / NATIVE_SHELL_LOCK_FILENAME
    key = str(lock_path)
    if key in _native_shell_locks:
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fh.seek(0)
        holder = fh.read().strip()
        fh.close()
        raise RuntimeError(
            f"Another Engram process already has {data_dir} open "
            f"({holder or 'holder unknown'}; lock file {lock_path}). "
            "Refusing to open the native graph twice — stop the other "
            "session (or point ENGRAM_HELIX__DATA_DIR elsewhere) and retry."
        ) from exc
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} acquired={time.time()}\n")
    fh.flush()
    _native_shell_locks[key] = fh


def _release_native_shell_lock(data_dir: Path) -> None:
    """Release a held shell flock (tests / explicit teardown only)."""
    import fcntl

    fh = _native_shell_locks.pop(str(data_dir / NATIVE_SHELL_LOCK_FILENAME), None)
    if fh is None:
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        # silent-ok: close() below drops the flock regardless.
        pass
    fh.close()
