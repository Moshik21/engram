"""M5: `engram backup compact` — reclaim LMDB free pages, verified before swap.

The dogfood brain measured 16.97 GiB on disk holding 7.40 GiB of live data:
56.4% of the file was LMDB pages freed by write churn and never returned to the
OS. That is the difference between a brain that fits in 16 GB of RAM and one
that is guaranteed to be evicted, so compaction is a real recall lever — but
only if a copy that lost data can never be swapped in.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from engram.backup_cli import _flatten_counts, compare_brain_counts, run_backup_command

STATS: dict[str, Any] = {
    "entities": 692,
    "relationships": 276,
    "episodes": 8930,
    "cue_metrics": {"cue_count": 2823, "cue_coverage": 0.3161},
    "projection_metrics": {
        "state_counts": {"projected": 4945, "cue_only": 3220},
        "yield": {"linked_entity_count": 7421},
        "failure_rate": 0.0038,
    },
}


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "backup_command": "compact",
        "data_dir": None,
        "to": None,
        "apply": False,
        "force_local": False,
        "format": "json",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _run(args: argparse.Namespace) -> int:
    return asyncio.run(run_backup_command(args))


@pytest.fixture()
def brain(tmp_path: Path) -> Path:
    data = tmp_path / "engram-native"
    data.mkdir()
    (data / "data.mdb").write_bytes(b"\x00" * 65536)
    (data / "lock.mdb").write_bytes(b"\x00" * 128)
    (data / "packet-cache.sqlite3").write_bytes(b"sqlite")
    return data


def _fake_compact(after: dict[str, Any], *, compacted_bytes: int = 16384):
    """Patch the two engine-opening helpers with an in-memory compaction."""

    async def snapshot_and_compact(data_dir: Path, staging: Path) -> tuple[dict[str, Any], int]:
        (staging / "data.mdb").write_bytes(b"\x00" * compacted_bytes)
        return STATS, compacted_bytes

    async def brain_stats(data_dir: Path) -> dict[str, Any]:
        return after

    return patch("engram.backup_cli._snapshot_and_compact", snapshot_and_compact), patch(
        "engram.backup_cli._brain_stats", brain_stats
    )


def _no_shell():
    return patch("engram.brain_runtime.shell_is_healthy", return_value=False), patch(
        "engram.brain_runtime.serve_process_alive", return_value=False
    )


# ─── the verification gate ────────────────────────────────────────


def test_flatten_counts_keeps_ints_and_drops_derived_floats() -> None:
    flat = _flatten_counts(STATS)
    assert flat["entities"] == 692
    assert flat["cue_metrics.cue_count"] == 2823
    assert flat["projection_metrics.state_counts.projected"] == 4945
    assert flat["projection_metrics.yield.linked_entity_count"] == 7421
    assert "cue_metrics.cue_coverage" not in flat
    assert "projection_metrics.failure_rate" not in flat


def test_identical_snapshots_have_no_problems() -> None:
    assert compare_brain_counts(STATS, json.loads(json.dumps(STATS))) == []


def test_a_single_lost_episode_is_a_problem() -> None:
    after = json.loads(json.dumps(STATS))
    after["episodes"] = 8929
    assert compare_brain_counts(STATS, after) == ["episodes: 8930 -> 8929"]


def test_lost_nested_projection_state_is_a_problem() -> None:
    after = json.loads(json.dumps(STATS))
    del after["projection_metrics"]["state_counts"]["cue_only"]
    assert compare_brain_counts(STATS, after) == [
        "projection_metrics.state_counts.cue_only: 3220 -> None"
    ]


def test_empty_source_stats_cannot_silently_verify() -> None:
    problems = compare_brain_counts({}, {})
    assert problems == [
        "entities: absent from source stats (nothing to verify against)",
        "episodes: absent from source stats (nothing to verify against)",
    ]


# ─── guards ───────────────────────────────────────────────────────


def test_refuses_a_dir_without_data_mdb(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-brain"
    empty.mkdir()
    assert _run(_args(data_dir=empty)) == 2


def test_refuses_a_non_empty_staging_dir(brain: Path, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "leftover").write_text("x")
    assert _run(_args(data_dir=brain, to=staging)) == 2


def test_refuses_when_the_volume_lacks_headroom(brain: Path) -> None:
    class _Usage:
        free = 1024

    with patch("engram.backup_cli.shutil.disk_usage", return_value=_Usage()):
        assert _run(_args(data_dir=brain)) == 2


def test_refuses_apply_across_volumes(brain: Path, tmp_path: Path) -> None:
    staging = tmp_path / "elsewhere"
    with patch("engram.backup_cli._same_volume", return_value=False):
        assert _run(_args(data_dir=brain, to=staging, apply=True)) == 2


def test_refuses_while_the_shell_is_up(brain: Path) -> None:
    with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
        assert _run(_args(data_dir=brain)) == 2
    # A refused run leaves no staging litter next to the brain.
    assert not list(brain.parent.glob("*.compact.*"))


# ─── copy + swap ──────────────────────────────────────────────────


def test_stages_a_verified_copy_without_touching_the_original(brain: Path, capsys) -> None:
    compact_patch, stats_patch = _fake_compact(STATS)
    shell_patch, serve_patch = _no_shell()
    with compact_patch, stats_patch, shell_patch, serve_patch:
        assert _run(_args(data_dir=brain)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["verify_problems"] == []
    assert report["applied"] is False
    assert report["source_bytes"] == 65536
    assert report["compacted_bytes"] == 16384
    assert report["bloat_ratio"] == 4.0
    assert report["saved_pct"] == 75.0
    assert report["verified_counts"] == len(_flatten_counts(STATS))
    # Original untouched; the copy carries the non-LMDB sidecars.
    assert (brain / "data.mdb").stat().st_size == 65536
    staging = Path(report["staging"])
    assert (staging / "packet-cache.sqlite3").read_bytes() == b"sqlite"
    assert not (staging / "lock.mdb").exists()


def test_apply_swaps_in_the_copy_and_keeps_the_original_aside(brain: Path, capsys) -> None:
    compact_patch, stats_patch = _fake_compact(STATS)
    shell_patch, serve_patch = _no_shell()
    with compact_patch, stats_patch, shell_patch, serve_patch:
        assert _run(_args(data_dir=brain, apply=True)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is True
    assert (brain / "data.mdb").stat().st_size == 16384
    aside = Path(report["previous_data_dir"])
    assert (aside / "data.mdb").stat().st_size == 65536


def test_apply_refuses_to_swap_when_counts_differ(brain: Path, capsys) -> None:
    lossy = json.loads(json.dumps(STATS))
    lossy["projection_metrics"]["yield"]["linked_entity_count"] = 7000
    compact_patch, stats_patch = _fake_compact(lossy)
    shell_patch, serve_patch = _no_shell()
    with compact_patch, stats_patch, shell_patch, serve_patch:
        assert _run(_args(data_dir=brain, apply=True)) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is False
    assert report["verify_problems"] == [
        "projection_metrics.yield.linked_entity_count: 7421 -> 7000"
    ]
    # The live brain is exactly where it was.
    assert (brain / "data.mdb").stat().st_size == 65536
    assert not list(brain.parent.glob("*.pre-compact.*"))


def test_a_stale_extension_is_not_reported_as_a_lock_problem(brain: Path, capsys) -> None:
    async def boom(data_dir: Path, staging: Path):
        raise ImportError("the installed helix_native has no compact(); make build-native")

    shell_patch, serve_patch = _no_shell()
    with patch("engram.backup_cli._snapshot_and_compact", boom), shell_patch, serve_patch:
        assert _run(_args(data_dir=brain)) == 2

    err = capsys.readouterr().err
    assert "make build-native" in err
    assert "stop the shell" not in err


# ─── transport plumbing ───────────────────────────────────────────


def test_transport_compact_reports_a_stale_extension() -> None:
    from engram.storage.helix.native_transport import NativeTransport

    transport = NativeTransport.__new__(NativeTransport)
    transport._engine = object()
    transport._executor = object()
    with pytest.raises(ImportError, match="make build-native"):
        asyncio.run(transport.compact("/tmp/nowhere"))


def test_transport_compact_requires_initialization() -> None:
    from engram.storage.helix.native_transport import NativeTransport

    transport = NativeTransport.__new__(NativeTransport)
    transport._engine = None
    transport._executor = None
    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(transport.compact("/tmp/nowhere"))


def test_client_compact_requires_the_native_transport() -> None:
    from engram.storage.helix.client import HelixClient

    client = HelixClient.__new__(HelixClient)
    client._native_transport = None
    with pytest.raises(ImportError, match="native transport"):
        asyncio.run(client.compact("/tmp/nowhere"))
