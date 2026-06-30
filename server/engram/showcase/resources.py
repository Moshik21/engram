"""Resolve bundled showcase demo database paths."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path


def bundled_demo_db_path() -> Path:
    """Return the packaged demo.db path from the installed engram package."""
    try:
        traversable = resources.files("engram").joinpath("data/demo.db")
        with resources.as_file(traversable) as resolved:
            return Path(resolved)
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        fallback = Path(__file__).resolve().parent.parent / "data" / "demo.db"
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(
            "Bundled showcase demo.db is missing. Run: uv run engram showcase seed"
        )


def resolve_demo_db_path(*, db_path: Path | None = None, copy_to: Path | None = None) -> Path:
    """Resolve an explicit path or the bundled demo database."""
    source = db_path.expanduser() if db_path is not None else bundled_demo_db_path()
    if not source.is_file():
        raise FileNotFoundError(f"Showcase database not found: {source}")
    if copy_to is None:
        return source
    copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, copy_to)
    return copy_to