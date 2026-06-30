"""Resolve bundled showcase demo database paths."""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib import resources
from pathlib import Path

SHOWCASE_CACHE_DIR = Path.home() / ".engram" / "showcase"
SHOWCASE_RUNTIME_DB_NAME = "demo-run.db"


def _source_tree_demo_db() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "demo.db"


def _materialize_packaged_demo_db(cache_path: Path) -> Path:
    traversable = resources.files("engram").joinpath("data/demo.db")
    with resources.as_file(traversable) as resolved:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, cache_path)
    return cache_path


def bundled_demo_db_source() -> Path:
    """Return a stable path to the read-only bundled demo.db bytes."""
    source_tree = _source_tree_demo_db()
    if source_tree.is_file():
        return source_tree

    cache_path = SHOWCASE_CACHE_DIR / "demo.db"
    if cache_path.is_file():
        return cache_path

    try:
        return _materialize_packaged_demo_db(cache_path)
    except (FileNotFoundError, ModuleNotFoundError, TypeError) as exc:
        raise FileNotFoundError(
            "Bundled showcase demo.db is missing. Run: uv run engram showcase seed"
        ) from exc


def bundled_demo_db_path() -> Path:
    """Back-compat alias for callers that only need the bundled source path."""
    return bundled_demo_db_source()


def showcase_runtime_db_path() -> Path:
    return SHOWCASE_CACHE_DIR / SHOWCASE_RUNTIME_DB_NAME


def prepare_showcase_db(
    *,
    db_path: Path | None = None,
    copy_to: Path | None = None,
) -> Path:
    """Resolve a writable showcase database for run/export.

    Explicit ``db_path`` values are used in place (tests and custom seeds).
    The bundled demo database is always copied to a stable runtime path so
    packaged installs and the source-tree demo.db stay pristine.
    """
    if db_path is not None:
        source = db_path.expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Showcase database not found: {source}")
        if copy_to is None:
            return source
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy_to)
        return copy_to

    source = bundled_demo_db_source()
    runtime = copy_to or showcase_runtime_db_path()
    runtime.parent.mkdir(parents=True, exist_ok=True)
    _atomic_copy_db(source, runtime)
    return runtime


def _atomic_copy_db(source: Path, dest: Path) -> None:
    """Copy a sqlite database atomically so repeat runs never corrupt the runtime file."""
    fd, tmp_name = tempfile.mkstemp(suffix=".db", dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def resolve_demo_db_path(*, db_path: Path | None = None, copy_to: Path | None = None) -> Path:
    """Resolve an explicit path or a fresh runtime copy of the bundled database."""
    return prepare_showcase_db(db_path=db_path, copy_to=copy_to)
