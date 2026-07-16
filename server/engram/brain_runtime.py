"""Cold-brain process helpers: exclusive lock + status file.

See docs/design/hot-cold-process-split.md.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def engram_home() -> Path:
    return Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()


def brain_lock_path() -> Path:
    return engram_home() / "brain.lock"


def brain_status_path() -> Path:
    return engram_home() / "brain-status.json"


@dataclass
class BrainStatus:
    """Last cold-brain run outcome (written after each cycle)."""

    ok: bool
    started_at: str
    finished_at: str
    duration_s: float
    tier: str
    profile: str
    paused_shell: bool
    pid: int
    error: str | None = None
    cycle_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path | None = None) -> None:
        target = path or brain_status_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def read_brain_status(path: Path | None = None) -> dict[str, Any] | None:
    target = path or brain_status_path()
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@contextmanager
def exclusive_brain_lock(lock_path: Path | None = None) -> Iterator[Path]:
    """Acquire an exclusive flock so only one brain process runs at a time."""
    import fcntl

    path = lock_path or brain_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another brain process holds {path}; skip or wait") from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} acquired={time.time()}\n")
        fh.flush()
        yield path
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
