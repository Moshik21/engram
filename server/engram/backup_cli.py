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

    compact_p = sub.add_parser(
        "compact",
        help="Compacting copy of the native brain (reclaims LMDB free pages; shell must be down)",
    )
    compact_p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Native data dir override (default: resolved from config)",
    )
    compact_p.add_argument(
        "--to",
        type=Path,
        default=None,
        help="Where to stage the compacted copy (default: <data-dir>.compact.<stamp>). "
        "--apply requires the same volume as the data dir.",
    )
    compact_p.add_argument(
        "--apply",
        action="store_true",
        help="Swap the verified copy into place (the original is renamed aside, not deleted)",
    )
    compact_p.add_argument(
        "--force-local",
        action="store_true",
        help="Skip the shell-down safety check (unsafe: copy may be torn)",
    )
    compact_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    _configure_ovfix_parser(sub)


def _configure_ovfix_parser(sub) -> None:
    p = sub.add_parser(
        "ovfix",
        help="Repair stale LMDB overflow-page headers left by a compaction (shell must be down)",
    )
    p.add_argument("--data-dir", type=Path, default=None, help="Native data dir override")
    p.add_argument("--apply", action="store_true", help="Write the repair (default: report only)")
    p.add_argument("--force-local", action="store_true", help="Skip the shell-down safety check")


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


# ─── compact (reclaim LMDB free pages) ────────────────────────────

# Compaction can never exceed the source size, so source + reserve is a safe
# floor; the reserve keeps the machine off a full disk if it is close already.
_COMPACT_RESERVE_BYTES = 2 * 1024**3
# LMDB rebuilds these; everything else in the data dir is carried across.
_LMDB_FILES = frozenset({"data.mdb", "lock.mdb"})


def _flatten_counts(stats: dict[str, Any], prefix: str = "") -> dict[str, int]:
    """Integer leaves of a get_stats() snapshot, keyed by dotted path.

    Floats (rates, averages) are skipped: they are derived, and comparing them
    would trade real signal for rounding noise.
    """
    flat: dict[str, int] = {}
    for key, value in stats.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_counts(value, f"{path}."))
        elif isinstance(value, int) and not isinstance(value, bool):
            flat[path] = value
    return flat


def compare_brain_counts(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Count mismatches between two get_stats() snapshots (empty == identical).

    This is the gate that makes compaction safe to apply: a copy that lost or
    gained a single entity, episode, cue or projection state never gets swapped
    in.
    """
    before_flat = _flatten_counts(before)
    after_flat = _flatten_counts(after)
    problems = [
        f"{key}: {expected} -> {after_flat.get(key)}"
        for key, expected in sorted(before_flat.items())
        if after_flat.get(key) != expected
    ]
    problems.extend(
        f"{key}: absent from source stats (nothing to verify against)"
        for key in ("entities", "episodes")
        if key not in before_flat
    )
    return problems


def _nearest_existing(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(_nearest_existing(path)).free


def _same_volume(left: Path, right: Path) -> bool:
    return os.stat(_nearest_existing(left)).st_dev == os.stat(_nearest_existing(right)).st_dev


def _open_native_graph(data_dir: Path) -> Any:
    from engram.config import EngramConfig
    from engram.storage.factory import create_stores
    from engram.storage.resolver import EngineMode

    config = EngramConfig(mode="helix")
    config.helix.data_dir = str(data_dir)
    config.helix.transport = "native"
    graph_store, _activation, _search = create_stores(EngineMode.HELIX, config)
    return graph_store


# ─── overflow-header repair ───────────────────────────────────────
# LMDB master3 (heed3) decides whether a page is the current txn's own dirty
# page by comparing the txn stamp in the page header to the txn counter
# (IS_MUTABLE). Its compacting copy rewrites leaf/branch pages with fresh
# headers but copies overflow pages VERBATIM and restarts the counter, so
# every pre-existing overflow record keeps a page number and a stamp from its
# previous life. The first in-place update of such a record writes into the
# read-only map: SIGBUS. Measured 2026-09-02 on the dogfood brain: 16 cue
# records crashed the process on update (the cold brain died on the first of
# them 203 times), a fresh engine.compact() copy crashed on EVERY record, and
# a DROP was refused with MDB_PROBLEM. Rewriting pgno := position and
# stamp := 1 (what compaction writes for the pages it does rewrite) on every
# overflow-head page whose number disagrees with its position fixed all of it:
# 16/16 updated, 10,747 no-op updates clean, delete + recreate clean.
_LMDB_PAGE_SIZE = 16384
_LMDB_P_OVERFLOW = 0x04


def repair_overflow_headers(data_mdb: Path, *, apply: bool) -> list[tuple[int, int, int]]:
    """Return (position, stale_pgno, stale_stamp) for each stale overflow head; fix if apply."""
    import struct

    size = data_mdb.stat().st_size
    pages = size // _LMDB_PAGE_SIZE
    stale: list[tuple[int, int, int]] = []
    with open(data_mdb, "r+b" if apply else "rb") as fh:
        for index in range(pages):
            fh.seek(index * _LMDB_PAGE_SIZE)
            header = fh.read(24)
            if len(header) < 24:
                break
            pgno, stamp, _pad, flags, lower, upper = struct.unpack("<QQHHHH", header)
            # overflow-head signature: flags == P_OVERFLOW, page count in
            # `lower`, `upper` unused. A free page matching it is harmless to
            # touch: LMDB never reads a free page's header before overwriting.
            if flags != _LMDB_P_OVERFLOW or upper != 0 or not (1 <= lower <= 64):
                continue
            if pgno == index:
                continue
            stale.append((index, pgno, stamp))
            if apply:
                fh.seek(index * _LMDB_PAGE_SIZE)
                fh.write(struct.pack("<QQ", index, 1))
    return stale


async def _write_probe(data_dir: Path) -> str | None:
    """Prove the store accepts an in-place write; None when it does, else the error.

    Count parity is a READ check; it passed on copies that crashed on their
    first write. One identical-payload cue update exercises the overflow
    overwrite path that failed.
    """
    import json

    try:
        import helix_native
    except ImportError as exc:
        return f"write probe unavailable: {exc}"
    engine = helix_native.HelixEngine(data_dir=str(data_dir))
    try:
        raw = engine.query("find_cues_by_group", json.dumps({"gid": "default"}))
        rows = json.loads(raw)
        rows = rows if isinstance(rows, list) else next(iter(rows.values()), [])
        if not rows:
            return None
        cue = rows[0]
        skip = ("id", "label", "episode_id", "group_id", "created_at")
        payload = {k: v for k, v in cue.items() if k not in skip}
        payload["id"] = cue["id"]
        engine.query("update_cue", json.dumps(payload))
        return None
    except Exception as exc:  # the probe's whole job is to surface this
        return f"write probe failed: {exc}"
    finally:
        engine.close()


async def _ovfix(args: argparse.Namespace) -> int:
    data_dir = _resolve_data_dir(args.data_dir)
    source = data_dir / "data.mdb"
    if not source.is_file():
        print(f"ovfix: no data.mdb under {data_dir} (native brain only)", file=sys.stderr)
        return 2
    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    try:
        with require_exclusive_local_access(force=bool(args.force_local)):
            stale = repair_overflow_headers(source, apply=bool(args.apply))
            probe = await _write_probe(data_dir) if args.apply else None
    except ExclusiveAccessError as exc:
        print(f"ovfix: {exc}", file=sys.stderr)
        print("ovfix: stop the shell first (engramctl stop), then re-run.", file=sys.stderr)
        return 2
    verb = "repaired" if args.apply else "would repair"
    print(f"ovfix: {verb} {len(stale)} stale overflow-head headers under {data_dir}")
    for index, pgno, stamp in stale[:10]:
        print(f"  page {index}: header pgno={pgno} stamp={stamp}")
    if probe:
        print(f"ovfix: {probe}", file=sys.stderr)
        return 1
    if args.apply:
        print("ovfix: write probe OK")
    return 0


async def _snapshot_and_compact(data_dir: Path, staging: Path) -> tuple[dict[str, Any], int]:
    """Read exact counts and write the compacting copy from ONE engine open."""
    graph_store = _open_native_graph(data_dir)
    await graph_store.initialize()
    try:
        # group_id=None on purpose: an integrity check must count the WHOLE brain,
        # not one group — data lost outside the default group must still fail.
        stats = await graph_store.get_stats(group_id=None, exact=True)
        size = await graph_store.compact(str(staging))
        return stats, size
    finally:
        await graph_store.close()


async def _brain_stats(data_dir: Path) -> dict[str, Any]:
    graph_store = _open_native_graph(data_dir)
    await graph_store.initialize()
    try:
        return await graph_store.get_stats(group_id=None, exact=True)
    finally:
        await graph_store.close()


def _gib(value: int) -> float:
    return round(value / 1024**3, 2)


async def _compact(args: argparse.Namespace) -> int:
    data_dir = _resolve_data_dir(args.data_dir)
    source = data_dir / "data.mdb"
    if not source.is_file():
        print(f"compact: no data.mdb under {data_dir} (native brain only)", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = (
        args.to.expanduser()
        if args.to is not None
        else data_dir.with_name(f"{data_dir.name}.compact.{stamp}")
    )
    if staging.exists() and any(staging.iterdir()):
        print(f"compact: staging dir is not empty: {staging}", file=sys.stderr)
        return 2

    source_bytes = source.stat().st_size
    needed = source_bytes + _COMPACT_RESERVE_BYTES
    free = _free_bytes(staging.parent)
    if free < needed:
        print(
            f"compact: not enough free space at {staging.parent}: {free} bytes free, "
            f"need {needed} (source {source_bytes} + {_COMPACT_RESERVE_BYTES} reserve)",
            file=sys.stderr,
        )
        return 2

    apply = bool(args.apply)
    if apply and not _same_volume(staging.parent, data_dir.parent):
        print(
            "compact: --apply swaps by rename and needs the staging dir on the "
            "same volume as the data dir",
            file=sys.stderr,
        )
        return 2

    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    try:
        with require_exclusive_local_access(force=bool(args.force_local)):
            staging.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            before, compacted_bytes = await _snapshot_and_compact(data_dir, staging)
            for sidecar in sorted(data_dir.iterdir()):
                if sidecar.name in _LMDB_FILES:
                    continue
                if sidecar.is_file():
                    shutil.copy2(sidecar, staging / sidecar.name)
                elif sidecar.is_dir():
                    shutil.copytree(sidecar, staging / sidecar.name, dirs_exist_ok=True)
            # The copy's overflow pages carry their old headers (see
            # repair_overflow_headers); repair before anything reads them
            # for writing, then prove a write works. Count parity alone
            # passed on copies that crashed on their first write.
            repaired = repair_overflow_headers(staging / "data.mdb", apply=True)
            after = await _brain_stats(staging)
            problems = compare_brain_counts(before, after)
            probe = await _write_probe(staging)
            if probe:
                problems.append(probe)
            duration_s = round(time.monotonic() - started, 2)

            aside: Path | None = None
            if apply and not problems:
                aside = data_dir.with_name(f"{data_dir.name}.pre-compact.{stamp}")
                data_dir.rename(aside)
                staging.rename(data_dir)
    except ExclusiveAccessError as exc:
        print(f"compact: {exc}", file=sys.stderr)
        print(
            "compact: stop the shell first (engramctl stop) so the copy is "
            "crash-consistent, then re-run.",
            file=sys.stderr,
        )
        return 2
    except ImportError as exc:
        # Missing capability, not a locking problem — do not send the operator
        # off to stop a shell that is already down.
        print(f"compact: {exc}", file=sys.stderr)
        return 2

    saved = source_bytes - compacted_bytes
    report: dict[str, Any] = {
        "data_dir": str(data_dir),
        "staging": str(data_dir if aside is not None else staging),
        "source_bytes": source_bytes,
        "compacted_bytes": compacted_bytes,
        "saved_bytes": saved,
        "saved_pct": round(100.0 * saved / source_bytes, 2) if source_bytes else 0.0,
        "bloat_ratio": round(source_bytes / compacted_bytes, 3) if compacted_bytes else 0.0,
        "duration_s": duration_s,
        "verified_counts": len(_flatten_counts(before)),
        "repaired_overflow_headers": len(repaired),
        "write_probe": "ok" if not probe else probe,
        "verify_problems": problems,
        "applied": aside is not None,
        "previous_data_dir": str(aside) if aside is not None else None,
    }
    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"Compacted {data_dir}")
        print(f"  before: {source_bytes} bytes ({_gib(source_bytes)} GiB)")
        print(f"  after:  {compacted_bytes} bytes ({_gib(compacted_bytes)} GiB)")
        print(
            f"  saved:  {saved} bytes ({_gib(saved)} GiB, {report['saved_pct']}%) "
            f"ratio {report['bloat_ratio']}x in {duration_s}s"
        )
        if problems:
            print(f"  verify: FAILED ({len(problems)} count mismatches)")
            for problem in problems[:20]:
                print(f"    {problem}")
        else:
            print(f"  verify: OK ({report['verified_counts']} counts match)")
        if aside is not None:
            print(f"  applied — previous data kept at: {aside}")
            print("  start the shell: engramctl start")
        else:
            print(f"  staged at: {staging}")
            if not problems:
                print("  apply:  engram backup compact --apply   (shell must be down)")
    if problems:
        print("compact: refusing to apply a copy whose counts differ", file=sys.stderr)
        return 1
    return 0


async def run_backup_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "backup_command", None)
    if cmd == "create":
        return _create(args)
    if cmd == "verify":
        return _verify(args)
    if cmd == "restore":
        return _restore(args)
    if cmd == "compact":
        return await _compact(args)
    if cmd == "ovfix":
        return await _ovfix(args)
    print(f"Unknown backup command: {cmd}", file=sys.stderr)
    return 2
