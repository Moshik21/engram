"""Backup/restore for the native brain: `engram backup create|verify|restore`.

The 17GB dogfood graph and the operator state under ~/.engram previously had
zero backup tooling — one disk failure destroyed the data moat. Backups are
taken under the exclusive-access guard (shell down + brain flock) so the LMDB
copy is crash-consistent, and use APFS clonefile when available (instant,
near-zero extra space on the same volume).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "backup-manifest.json"
# ~/.engram state worth carrying (logs and model caches are reproducible).
_HOME_STATE_FILES = (
    ".env",
    "brain-status.json",
    "loop-adjustment.json",
    "loop-adjustments.jsonl",
    "hygiene-state.json",
    "activation-snapshot.json",
    "promotion-window.json",
)


def configure_backup_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="backup_command", required=True)

    create_p = sub.add_parser("create", help="Snapshot the native data dir + operator state")
    create_p.add_argument(
        "--to",
        type=Path,
        default=None,
        help="Backup root (default: ~/.helix/backups). NOTE: same-volume backups "
        "do not survive disk failure — point this at external storage for real "
        "protection.",
    )
    create_p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Native data dir override (default: resolved from config)",
    )
    create_p.add_argument(
        "--force-local",
        action="store_true",
        help="Skip the shell-down safety check (unsafe: copy may be torn)",
    )

    verify_p = sub.add_parser("verify", help="Verify a backup against its manifest")
    verify_p.add_argument("path", type=Path, help="Backup directory to verify")

    restore_p = sub.add_parser(
        "restore",
        help="Restore a backup over the configured data dir (shell must be down)",
    )
    restore_p.add_argument("path", type=Path, help="Backup directory to restore from")
    restore_p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Restore target override (default: resolved from config)",
    )
    restore_p.add_argument(
        "--yes",
        action="store_true",
        help="Confirm replacing the current data dir (it is renamed aside first)",
    )


def _resolve_data_dir(override: Path | None) -> Path:
    if override is not None:
        return override.expanduser()
    from engram.config import EngramConfig
    from engram.storage.diagnostics import resolve_helix_native_data_dir

    return Path(resolve_helix_native_data_dir(EngramConfig())).expanduser()


def _engram_home() -> Path:
    return Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()


def _clone_or_copy(src: Path, dst: Path) -> str:
    """APFS clonefile when possible (instant, CoW); portable copy otherwise."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["cp", "-c", "-R", str(src), str(dst)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return "clonefile"
    shutil.copytree(src, dst, dirs_exist_ok=False)
    return "copy"


def _dir_manifest(root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            sizes[str(path.relative_to(root))] = path.stat().st_size
    return sizes


def _create(args: argparse.Namespace) -> int:
    data_dir = _resolve_data_dir(args.data_dir)
    if not data_dir.is_dir():
        print(f"backup: data dir not found: {data_dir}", file=sys.stderr)
        return 2

    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    backup_root = (args.to or Path.home() / ".helix" / "backups").expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = backup_root / f"{data_dir.name}.{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with require_exclusive_local_access(force=bool(args.force_local)):
            started = time.monotonic()
            method = _clone_or_copy(data_dir, target / "data")
            state_dir = target / "engram-home"
            state_dir.mkdir(parents=True, exist_ok=True)
            home = _engram_home()
            copied_state: list[str] = []
            for name in _HOME_STATE_FILES:
                src = home / name
                if src.is_file():
                    shutil.copy2(src, state_dir / name)
                    copied_state.append(name)
            manifest: dict[str, Any] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_data_dir": str(data_dir),
                "method": method,
                "duration_s": round(time.monotonic() - started, 2),
                "data_files": _dir_manifest(target / "data"),
                "engram_home_files": copied_state,
            }
            (target / _MANIFEST_NAME).write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
    except ExclusiveAccessError as exc:
        print(f"backup: {exc}", file=sys.stderr)
        print(
            "backup: stop the shell first (engramctl stop) so the copy is "
            "crash-consistent, then re-run.",
            file=sys.stderr,
        )
        return 2

    total = sum(manifest["data_files"].values())
    print(f"Backup created: {target}")
    print(f"  method={method} files={len(manifest['data_files'])} bytes={total}")
    if str(target).startswith(str(Path.home())):
        print(
            "  note: same-volume backup protects against corruption, not disk "
            "failure — copy to external storage for real protection."
        )
    print(f"  verify:  engram backup verify {target}")
    print(f"  restore: engram backup restore {target} --yes  (shell must be down)")
    return 0


def _verify(args: argparse.Namespace) -> int:
    target = args.path.expanduser()
    manifest_path = target / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"verify: cannot read manifest: {exc}", file=sys.stderr)
        return 2
    data_root = target / "data"
    expected: dict[str, int] = manifest.get("data_files") or {}
    problems: list[str] = []
    for rel, size in expected.items():
        path = data_root / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif path.stat().st_size != int(size):
            problems.append(f"size mismatch: {rel}")
    # LMDB readability spot-check: data.mdb must start with a valid page.
    mdb = data_root / "data.mdb"
    if mdb.is_file():
        with open(mdb, "rb") as fh:
            head = fh.read(4096)
        if len(head) < 4096:
            problems.append("data.mdb truncated below one page")
    if problems:
        for problem in problems[:20]:
            print(f"verify: {problem}", file=sys.stderr)
        print(f"verify: FAILED ({len(problems)} problems)", file=sys.stderr)
        return 1
    print(f"verify: OK ({len(expected)} files match manifest)")
    return 0


def _restore(args: argparse.Namespace) -> int:
    target = args.path.expanduser()
    if _verify(argparse.Namespace(path=target)) != 0:
        print("restore: refusing to restore a backup that fails verify", file=sys.stderr)
        return 2
    data_dir = _resolve_data_dir(args.data_dir)
    if not args.yes:
        print(
            f"restore: would replace {data_dir} with {target}/data — re-run with --yes",
            file=sys.stderr,
        )
        return 2

    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    try:
        with require_exclusive_local_access():
            aside = None
            if data_dir.exists():
                aside = data_dir.with_name(
                    f"{data_dir.name}.pre-restore.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                )
                data_dir.rename(aside)
            _clone_or_copy(target / "data", data_dir)
    except ExclusiveAccessError as exc:
        print(f"restore: {exc}", file=sys.stderr)
        return 2
    print(f"Restored {target} -> {data_dir}")
    if aside is not None:
        print(f"  previous data kept at: {aside}")
    print("  start the shell: engramctl start")
    return 0


async def run_backup_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "backup_command", None)
    if cmd == "create":
        return _create(args)
    if cmd == "verify":
        return _verify(args)
    if cmd == "restore":
        return _restore(args)
    print(f"Unknown backup command: {cmd}", file=sys.stderr)
    return 2
