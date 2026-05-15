"""Regression coverage for public installer local mode selection."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_engramctl_setup(tmp_path: Path, mode: str) -> tuple[str, str]:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    engram_home = home / ".engram"
    home.mkdir()
    bin_dir.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "ENGRAM_HOME": str(engram_home),
            "ENGRAM_INSTALL_BIN_DIR": str(bin_dir),
            "ENGRAM_INSTALL_NONINTERACTIVE": "1",
            "ENGRAM_INSTALL_SKIP_NATIVE_VERIFY": "1",
            "ENGRAM_ANTHROPIC_API_KEY": "sk-test",
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "installer/engramctl"), "setup", "--mode", mode],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return (engram_home / ".env").read_text(), result.stdout + result.stderr


def test_engramctl_setup_helix_configures_local_native_mode(tmp_path: Path) -> None:
    content, output = _run_engramctl_setup(tmp_path, "helix")

    assert "Selected mode: helix" in output
    assert "ENGRAM_INSTALL_VARIANT=lite" in content
    assert "ENGRAM_MODE=helix" in content
    assert "ENGRAM_HELIX__TRANSPORT=native" in content


def test_engramctl_setup_lite_keeps_sqlite_local_mode(tmp_path: Path) -> None:
    content, output = _run_engramctl_setup(tmp_path, "lite")

    assert "Selected mode: lite" in output
    assert "ENGRAM_INSTALL_VARIANT=lite" in content
    assert "ENGRAM_MODE=lite" in content
    assert "ENGRAM_HELIX__TRANSPORT" not in content


def test_install_script_forwards_explicit_helix_mode() -> None:
    install_script = (ROOT / "scripts/install.sh").read_text()

    assert '""|helix|lite|auto|full|openclaw)' in install_script
    assert '"$BIN_DIR/engramctl" setup --mode "$MODE"' in install_script
    assert 'package_spec="engram[local,native]"' in install_script
    assert "#subdirectory=server[local,native]" in install_script


def test_engramctl_setup_helix_verifies_native_runtime() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert '_verify_helix_native_runtime "$engram_cmd"' in engramctl
    assert "doctor --mode helix --skip-server --no-smoke --no-lifecycle" in engramctl
    assert "Helix native setup did not produce a runnable native runtime." in engramctl


def test_engramctl_update_preserves_helix_native_package_extras() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert 'engine_mode="${ENGRAM_MODE:-lite}"' in engramctl
    assert 'package_spec="engram[local,native]"' in engramctl
    assert 'uv tool upgrade "$package_spec"' in engramctl
