"""M6.4: engram backup create/verify/restore round-trip on a copy."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def fake_brain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data = tmp_path / "engram-native"
    data.mkdir()
    (data / "data.mdb").write_bytes(b"\x00" * 8192)
    (data / "lock.mdb").write_bytes(b"\x00" * 128)
    (data / "packet-cache.sqlite3").write_bytes(b"sqlite")
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("ENGRAM_MODE=helix\n")
    (home / "brain-status.json").write_text("{}")
    monkeypatch.setenv("ENGRAM_HOME", str(home))
    return {"data": data, "home": home, "root": tmp_path}


def _run(args: argparse.Namespace) -> int:
    from engram.backup_cli import run_backup_command

    return asyncio.run(run_backup_command(args))


def _no_shell():
    return patch("engram.brain_runtime.shell_is_healthy", return_value=False), patch(
        "engram.brain_runtime.serve_process_alive", return_value=False
    )


class TestBackupRoundTrip:
    def test_create_verify_restore(self, fake_brain: dict[str, Path], capsys):
        backups = fake_brain["root"] / "backups"
        p1, p2 = _no_shell()
        with p1, p2:
            rc = _run(
                argparse.Namespace(
                    backup_command="create",
                    to=backups,
                    data_dir=fake_brain["data"],
                    force_local=False,
                )
            )
        assert rc == 0
        created = list(backups.iterdir())
        assert len(created) == 1
        target = created[0]
        manifest = json.loads((target / "backup-manifest.json").read_text())
        assert "data.mdb" in manifest["data_files"]
        assert ".env" in manifest["engram_home_files"]

        # Verify passes.
        assert _run(argparse.Namespace(backup_command="verify", path=target)) == 0

        # Corrupt the live dir, then restore from backup.
        (fake_brain["data"] / "data.mdb").write_bytes(b"corrupted")
        with p1, p2:
            rc = _run(
                argparse.Namespace(
                    backup_command="restore",
                    path=target,
                    data_dir=fake_brain["data"],
                    yes=True,
                )
            )
        assert rc == 0
        assert (fake_brain["data"] / "data.mdb").stat().st_size == 8192
        # Previous data kept aside.
        aside = [p for p in fake_brain["root"].iterdir() if "pre-restore" in p.name]
        assert len(aside) == 1

    def test_verify_fails_on_tamper(self, fake_brain: dict[str, Path]):
        backups = fake_brain["root"] / "backups"
        p1, p2 = _no_shell()
        with p1, p2:
            _run(
                argparse.Namespace(
                    backup_command="create",
                    to=backups,
                    data_dir=fake_brain["data"],
                    force_local=False,
                )
            )
        target = next(backups.iterdir())
        (target / "data" / "data.mdb").write_bytes(b"short")
        assert _run(argparse.Namespace(backup_command="verify", path=target)) == 1

    def test_create_refused_when_shell_up(self, fake_brain: dict[str, Path], capsys):
        with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
            rc = _run(
                argparse.Namespace(
                    backup_command="create",
                    to=fake_brain["root"] / "b",
                    data_dir=fake_brain["data"],
                    force_local=False,
                )
            )
        assert rc == 2
        assert "stop the shell" in capsys.readouterr().err

    def test_restore_requires_yes(self, fake_brain: dict[str, Path], capsys):
        backups = fake_brain["root"] / "backups"
        p1, p2 = _no_shell()
        with p1, p2:
            _run(
                argparse.Namespace(
                    backup_command="create",
                    to=backups,
                    data_dir=fake_brain["data"],
                    force_local=False,
                )
            )
            target = next(backups.iterdir())
            rc = _run(
                argparse.Namespace(
                    backup_command="restore",
                    path=target,
                    data_dir=fake_brain["data"],
                    yes=False,
                )
            )
        assert rc == 2
        assert "--yes" in capsys.readouterr().err
