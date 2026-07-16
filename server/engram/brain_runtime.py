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
    # Monotonic duration excludes system sleep; a large wall/monotonic delta
    # means the machine slept mid-run (the shell was paused the whole time).
    duration_monotonic_s: float | None = None
    system_slept: bool | None = None

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


# ─── Shell pause marker (crash-safe resume) ──────────────────────
#
# The brain writes this marker BEFORE stopping the shell and clears it only
# after the shell is confirmed healthy again. Any brain run (or engramctl)
# that finds the marker with the shell down knows the shell was stranded by
# a crashed/killed brain window and must resume it.


def shell_pause_marker_path() -> Path:
    return engram_home() / "shell-paused-by-brain.json"


def write_pause_marker(path: Path | None = None) -> None:
    target = path or shell_pause_marker_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"pid": os.getpid(), "paused_at": utc_now_iso()}) + "\n",
        encoding="utf-8",
    )


def read_pause_marker(path: Path | None = None) -> dict[str, Any] | None:
    target = path or shell_pause_marker_path()
    if not target.is_file():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def clear_pause_marker(path: Path | None = None) -> None:
    target = path or shell_pause_marker_path()
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ─── Exclusive-access probes ─────────────────────────────────────


def brain_lock_is_held(lock_path: Path | None = None) -> bool:
    """Non-destructively probe whether a brain process holds the flock."""
    import fcntl

    path = lock_path or brain_lock_path()
    if not path.exists():
        return False
    try:
        with open(path, "a+", encoding="utf-8") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def serve_process_alive() -> bool:
    """Whether an 'engram serve' process exists (health probe can lag startup)."""
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-f", "engram serve"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    pids = [p for p in result.stdout.split() if p.strip().isdigit()]
    return any(int(p) != os.getpid() for p in pids)


def shell_is_healthy(port: int | None = None, timeout: float = 1.5) -> bool:
    import urllib.request

    if port is None:
        raw = os.environ.get("ENGRAM_API_PORT", "8100")
        try:
            port = int(raw)
        except ValueError:
            port = 8100
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


class ExclusiveAccessError(RuntimeError):
    """Local graph access refused: the shell (or another brain) may hold it."""


@contextmanager
def require_exclusive_local_access(
    *,
    force: bool = False,
    lock_path: Path | None = None,
) -> Iterator[Path]:
    """Guard a local native-graph open against the running shell and other brains.

    Refuses when the shell answers /health or an 'engram serve' process exists
    (unless force=True), then takes the exclusive brain flock so concurrent
    CLI/brain openers serialize. Raises ExclusiveAccessError on refusal.
    """
    if not force:
        if shell_is_healthy():
            raise ExclusiveAccessError(
                "Engram shell is running on this host; this command opens the "
                "graph directly and must not run concurrently. Use the shell "
                "HTTP API, 'engram brain run', or stop the shell first "
                "(--force-local to override at your own risk)."
            )
        if serve_process_alive():
            raise ExclusiveAccessError(
                "An 'engram serve' process exists but /health is not "
                "responding (starting or stopping); refusing to open the "
                "graph concurrently. Retry when the shell is fully up or down."
            )
    try:
        with exclusive_brain_lock(lock_path) as path:
            yield path
    except RuntimeError as exc:
        if isinstance(exc, ExclusiveAccessError):
            raise
        raise ExclusiveAccessError(str(exc)) from exc


# ─── Power gate ──────────────────────────────────────────────────


def on_battery_power(pmset_output: str | None = None) -> bool:
    """True when the machine runs on battery (skip scheduled brain windows).

    A 2h window fired during a DarkWake on battery freezes mid-run when the
    machine sleeps, leaving the shell paused all night (observed 2026-07-16:
    10h44m outage). Skipping on battery is always safe — the next AC window
    catches up.
    """
    if pmset_output is None:
        import subprocess

        try:
            result = subprocess.run(
                ["pmset", "-g", "ps"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            pmset_output = result.stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
    return "Battery Power" in (pmset_output or "")
