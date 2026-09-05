"""Project identity from a filesystem path (2026-09-04).

The hook and the recall surface used the working directory's basename as the
project name, so one repository became several projects ('Engram', 'server',
'app', 'reports') and a home-directory cwd became a project called after the
user. Measured on the fresh store: the four stable misses of the untouched rig
were all rows tagged 'server' demoted by the other-project multiplier when the
question came from the repo root.

Identity is the repository root's name (the nearest ancestor holding a
``.git`` entry); a path with no repository keeps its basename; a home or
root directory is no project at all. At recall the ACCEPTED set also carries
the root's immediate subdirectory names, so rows captured before this change
(tagged with a subdirectory) still count as the same project.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


def _is_home_or_root(path: Path) -> bool:
    try:
        return path == Path.home() or path == path.anchor or str(path) in ("/", "")
    except Exception:
        return False


def repo_root(path: str | os.PathLike[str] | None) -> Path | None:
    """Nearest ancestor (inclusive) that holds a ``.git`` dir or file."""
    if not path:
        return None
    try:
        current = Path(path).expanduser()
        if not current.is_absolute():
            current = current.resolve()
    except Exception:
        return None
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def project_name(path: str | os.PathLike[str] | None) -> str | None:
    """The project a path belongs to, or None for no path / home / root."""
    if not path:
        return None
    p = Path(str(path).rstrip("/") or "/").expanduser()
    if _is_home_or_root(p):
        return None
    root = repo_root(p)
    name = (root or p).name.strip()
    return name or None


@functools.lru_cache(maxsize=64)
def _subdir_names(root: str) -> frozenset[str]:
    try:
        return frozenset(
            e.name.lower()
            for e in os.scandir(root)
            if e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
        )
    except OSError:
        return frozenset()


def accepted_project_names(path: str | os.PathLike[str] | None) -> frozenset[str]:
    """Lower-cased names that count as *this* project at recall time.

    The root name, the path's own basename, and the root's immediate
    subdirectory names (legacy rows were tagged with those).
    """
    name = project_name(path)
    if not name:
        return frozenset()
    names = {name.lower()}
    p = Path(str(path).rstrip("/") or "/").expanduser()
    if p.name:
        names.add(p.name.lower())
    root = repo_root(p)
    if root is not None:
        names |= _subdir_names(str(root))
    return frozenset(names)
