"""Regression coverage for public installer local mode selection."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_engramctl_setup(tmp_path: Path, mode: str) -> tuple[str, str]:
    return _run_engramctl(tmp_path, "setup", "--mode", mode)


def _run_engramctl(tmp_path: Path, *args: str) -> tuple[str, str]:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    engram_home = home / ".engram"
    home.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    for key in ("ENGRAM_MODE", "ENGRAM_HELIX__TRANSPORT", "ENGRAM_HELIX__DATA_DIR"):
        env.pop(key, None)
    env.update(
        {
            "HOME": str(home),
            "ENGRAM_HOME": str(engram_home),
            "ENGRAM_INSTALL_BIN_DIR": str(bin_dir),
            "ENGRAM_INSTALL_NONINTERACTIVE": "1",
            "ENGRAM_INSTALL_SKIP_NATIVE_VERIFY": "1",
            "ENGRAM_API_PORT": "18100",
            "ENGRAM_ANTHROPIC_API_KEY": "sk-test",
            "ENGRAM_SKILL_SOURCE": str(ROOT / "skills/engram-memory"),
        }
    )

    result = subprocess.run(
        ["bash", str(ROOT / "installer/engramctl"), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return (engram_home / ".env").read_text(), result.stdout + result.stderr


def _engramctl_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    engram_home = home / ".engram"
    home.mkdir(exist_ok=True)
    bin_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    for key in ("ENGRAM_MODE", "ENGRAM_HELIX__TRANSPORT", "ENGRAM_HELIX__DATA_DIR"):
        env.pop(key, None)
    env.update(
        {
            "HOME": str(home),
            "ENGRAM_HOME": str(engram_home),
            "ENGRAM_INSTALL_BIN_DIR": str(bin_dir),
            "ENGRAM_INSTALL_NONINTERACTIVE": "1",
            "ENGRAM_INSTALL_SKIP_NATIVE_VERIFY": "1",
            "ENGRAM_API_PORT": "18100",
            "ENGRAM_ANTHROPIC_API_KEY": "sk-test",
            "ENGRAM_SKILL_SOURCE": str(ROOT / "skills/engram-memory"),
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
        }
    )
    return env


def _write_fake_engram(bin_dir: Path, calls_file: Path) -> None:
    fake_engram = bin_dir / "engram"
    fake_engram.write_text(
        """#!/usr/bin/env bash
python3 - "$ENGRAM_FAKE_CALLS" "$@" <<'PY'
import json
import sys

with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[2:]) + "\\n")
PY
echo "fake engram $*"
""",
    )
    fake_engram.chmod(0o755)
    calls_file.write_text("")


def _run_engramctl_with_env(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "installer/engramctl"), *args],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


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
    assert "is_native_install_mode()" in install_script
    assert '[ "${MODE:-helix}" = "helix" ] || [ "${MODE:-}" = "auto" ]' in install_script
    assert '[ "${MODE:-}" = "openclaw" ]' in install_script
    assert '"$BIN_DIR/engramctl" quickstart --mode "$MODE"' in install_script
    assert (
        '"$BIN_DIR/engramctl" quickstart --mode helix --install-openclaw'
        " --connect openclaw"
    ) in install_script
    assert 'if [ "$MODE" = "full" ]; then' in install_script
    assert 'if [ "$MODE" = "full" ] || [ "$MODE" = "openclaw" ]; then' not in install_script
    assert 'package_spec="engram[local,native]"' in install_script
    assert "#subdirectory=server[local,native]" in install_script
    assert 'uv_tool_install_engram "$github_spec" "GitHub"' in install_script
    assert (
        install_script.index('uv_tool_install_engram "$github_spec" "GitHub"')
        < install_script.index('uv_tool_install_engram "$package_spec" "PyPI"')
    )
    assert "resolve_helix_native_requirement" in install_script
    assert (
        'uv tool install --reinstall-package engram --with "$helix_native_req" "$package_spec"'
        in install_script
    )
    assert "HELIX_NATIVE_SUBDIR" in install_script
    assert "discover_helix_native_release_wheel" in install_script
    assert "local_helix_native_requirement" in install_script
    assert 'HELIX_NATIVE_SUBDIR="native/helix-repo/helix-python"' in install_script
    assert "helixdb-cfg/.helix" not in install_script
    assert "cargo --version" in install_script
    assert "rustup default stable" in install_script


def test_engramctl_quickstart_configures_native_without_repo_commands(tmp_path: Path) -> None:
    content, output = _run_engramctl(
        tmp_path,
        "quickstart",
        "--mode",
        "helix",
        "--no-start",
        "--no-doctor",
    )

    assert "Engram Quickstart" in output
    assert "Quickstart complete" in output
    assert "ENGRAM_MODE=helix" in content
    assert "ENGRAM_HELIX__TRANSPORT=native" in content
    assert "cd server" not in output
    assert "uv run" not in output


def test_engramctl_quickstart_can_install_openclaw_skill(tmp_path: Path) -> None:
    _content, output = _run_engramctl(
        tmp_path,
        "quickstart",
        "--mode",
        "helix",
        "--install-openclaw",
        "--no-start",
        "--no-doctor",
    )

    assert "Installed OpenClaw skill at" in output
    assert ".openclaw/skills/engram-brain" in output
    assert (tmp_path / "home/.openclaw/skills/engram-brain/SKILL.md").exists()


def test_engramctl_start_honors_configured_api_port() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert 'local port="${ENGRAM_API_PORT:-8100}"' in engramctl
    assert '"$engram_cmd" serve --host 127.0.0.1 --port "$port"' in engramctl
    assert "curl --connect-timeout 1 --max-time 5 -fsS" in engramctl
    assert 'doctor --mode "${ENGRAM_MODE:-lite}" --server-url "$(api_base_url)"' in engramctl
    assert "return 1" in engramctl
    assert "Quickstart could not confirm Engram is ready." in engramctl


def test_engramctl_exposes_release_startup_commands() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert "quickstart [--project PATH]" in engramctl
    assert "[--install-openclaw] [--connect CLIENT]" in engramctl
    assert "[--axi-hooks] [--capture]" in engramctl
    assert "doctor                         Run the installed-user readiness gate" in engramctl
    assert "connect <client>" in engramctl
    assert "Add --axi for read-only AXI startup context" in engramctl
    assert "codex|claude-code|cursor|windsurf|claude-desktop|openclaw" in engramctl
    assert "bootstrap [--include GLOB] <project-dir> [...]" in engramctl
    assert "storage                        Show resolved storage paths and disk growth" in engramctl
    assert "quickstart) command_quickstart" in engramctl
    assert "doctor) command_doctor" in engramctl
    assert "connect) command_connect" in engramctl
    assert "bootstrap) command_bootstrap" in engramctl
    assert "storage) command_storage" in engramctl


def test_engramctl_connect_can_install_axi_hooks() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert "install_axi_hook_for_client" in engramctl
    assert "--axi)" in engramctl
    assert "--capture)" in engramctl
    assert "local args=(" in engramctl
    assert 'axi hooks install "$client"' in engramctl
    assert '--engram-command "$engram_cmd"' in engramctl
    assert "engramctl connect codex --axi" in engramctl
    assert "engramctl connect claude-code --axi" in engramctl
    assert '"$engram_cmd" "${args[@]}"' in engramctl
    assert '"$engram_cmd" hooks' not in engramctl


def test_engramctl_connect_codex_axi_executes_installed_hook_command(tmp_path: Path) -> None:
    env = _engramctl_env(tmp_path)
    calls_file = tmp_path / "fake-engram-calls.jsonl"
    env["ENGRAM_FAKE_CALLS"] = str(calls_file)
    _write_fake_engram(tmp_path / "bin", calls_file)

    _run_engramctl_with_env(env, "setup", "--mode", "helix")
    result = _run_engramctl_with_env(env, "connect", "codex", "--axi")

    calls = [line for line in calls_file.read_text().splitlines() if line.strip()]
    assert "Configured Codex MCP" in result.stdout + result.stderr
    assert len(calls) == 1
    args = json.loads(calls[0])
    assert args == [
        "axi",
        "hooks",
        "install",
        "codex",
        "--server-url",
        "http://127.0.0.1:18100",
        "--timeout",
        "3",
        "--budget",
        "800",
        "--engram-command",
        str(tmp_path / "bin/engram"),
    ]
    config = (tmp_path / "home/.codex/config.toml").read_text()
    assert "[mcp_servers.engram]" in config
    assert 'url = "http://127.0.0.1:18100/mcp"' in config


def test_engramctl_connect_claude_axi_keeps_capture_opt_in(tmp_path: Path) -> None:
    env = _engramctl_env(tmp_path)
    calls_file = tmp_path / "fake-engram-calls.jsonl"
    env["ENGRAM_FAKE_CALLS"] = str(calls_file)
    _write_fake_engram(tmp_path / "bin", calls_file)
    project = tmp_path / "project"
    project.mkdir()

    _run_engramctl_with_env(env, "setup", "--mode", "helix")
    _run_engramctl_with_env(
        env,
        "connect",
        "claude-code",
        "--project",
        str(project),
        "--axi",
    )

    args = json.loads(calls_file.read_text().splitlines()[0])
    assert args[:4] == ["axi", "hooks", "install", "claude-code"]
    assert "--capture" not in args
    config = (project / ".mcp.json").read_text()
    assert '"url": "http://127.0.0.1:18100/mcp"' in config


def test_engramctl_storage_reports_native_and_sqlite_paths() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert (
        'local helix_data_dir="${ENGRAM_HELIX__DATA_DIR:-$HOME/.helix/engram-native}"'
        in engramctl
    )
    assert 'local sqlite_path="${ENGRAM_SQLITE__PATH:-$LITE_DB_FILE}"' in engramctl
    assert '"http://127.0.0.1:${port}/api/storage"' in engramctl
    assert "format_storage_json" in engramctl
    assert "offline_storage_status" in engramctl
    assert "Engram Storage" in engramctl


def test_engramctl_storage_offline_smoke_shows_native_data_path(tmp_path: Path) -> None:
    _content, _output = _run_engramctl_setup(tmp_path, "helix")

    _content, output = _run_engramctl(tmp_path, "storage")

    assert "Engram Storage (offline)" in output
    assert "Server API is not responding" in output
    assert ".helix/engram-native" in output


def test_engramctl_connect_uses_release_clean_mcp_paths() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert '$HOME/.codex/config.toml' in engramctl
    assert "write_codex_mcp_config" in engramctl
    assert "[mcp_servers.engram]" in engramctl
    assert "remote_mcp_client_enabled = true" in engramctl
    assert 'project_path/.mcp.json' in engramctl
    assert 'project_path/.cursor/mcp.json' in engramctl
    assert '$HOME/.codeium/windsurf/mcp_config.json' in engramctl
    assert '"type": "http"' in engramctl
    assert 'url": "$url"' in engramctl
    assert "openclaw mcp set engram" in engramctl
    assert '"transport": "streamable-http"' in engramctl
    assert "OpenClaw CLI not found; MCP config was not written." in engramctl
    assert 'OPENCLAW_SKILL_SLUG="${ENGRAM_OPENCLAW_SKILL_SLUG:-engram-brain}"' in engramctl
    assert '$HOME/.openclaw/skills/$OPENCLAW_SKILL_SLUG' in engramctl


def test_engramctl_connect_codex_writes_global_toml(tmp_path: Path) -> None:
    _run_engramctl_setup(tmp_path, "helix")

    _content, output = _run_engramctl(tmp_path, "connect", "codex")

    config = (tmp_path / "home/.codex/config.toml").read_text()
    assert "Configured Codex MCP" in output
    assert "[mcp]" in config
    assert "remote_mcp_client_enabled = true" in config
    assert "[mcp_servers.engram]" in config
    assert 'url = "http://127.0.0.1:18100/mcp"' in config


def test_engramctl_install_mode_keeps_openclaw_native_first() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert "install [--mode full] [--no-open]" in engramctl
    assert "OpenClaw mode is native-first." in engramctl
    assert "engramctl quickstart --mode helix --install-openclaw --connect openclaw" in engramctl


def test_release_bundle_emits_openclaw_slugged_skill_asset() -> None:
    build_script = (ROOT / "scripts/build_install_bundle.py").read_text()
    release_workflow = (ROOT / ".github/workflows/release.yml").read_text()

    assert 'OPENCLAW_SKILL_SLUG = "engram-brain"' in build_script
    assert 'f"{OPENCLAW_SKILL_SLUG}-skill.tar.gz"' in build_script
    assert "engram-brain-skill.tar.gz" in release_workflow
    assert "engram-brain-skill.sha256" in release_workflow
    assert "engram-memory-skill.tar.gz" in release_workflow
    assert "Build Helix Native Wheels" in release_workflow
    assert "PyO3/maturin-action@v1" in release_workflow
    assert "working-directory: native/helix-repo/helix-python" in release_workflow
    assert "dist/native-wheels/*.whl" in release_workflow


def test_engramctl_bootstrap_supports_user_approved_include_patterns() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert "--include)" in engramctl
    assert "include_patterns" in engramctl
    assert 'body["include_patterns"] = sys.argv[2:]' in engramctl
    assert "--max-time 180" in engramctl


def test_engramctl_bootstrap_works_without_include_patterns(tmp_path: Path) -> None:
    env = _engramctl_env(tmp_path)
    engram_home = Path(env["ENGRAM_HOME"])
    bin_dir = Path(env["ENGRAM_INSTALL_BIN_DIR"])
    project = tmp_path / "project"
    payloads = tmp_path / "payloads.jsonl"
    project.mkdir()
    engram_home.mkdir(parents=True, exist_ok=True)
    (engram_home / ".env").write_text("ENGRAM_API_PORT=18100\n")
    fake_curl = bin_dir / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--data" ]; then
    printf '%s\n' "$2" >> "$ENGRAM_CAPTURED_PAYLOADS"
    shift 2
    continue
  fi
  shift
done
exit 0
"""
    )
    fake_curl.chmod(0o755)
    env["ENGRAM_CAPTURED_PAYLOADS"] = str(payloads)

    subprocess.run(
        ["bash", str(ROOT / "installer/engramctl"), "bootstrap", str(project)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(payloads.read_text().strip())
    assert payload == {"project_path": str(project)}


def test_engramctl_setup_helix_verifies_native_runtime() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert '_verify_helix_native_runtime "$engram_cmd"' in engramctl
    assert "doctor --mode helix --skip-server --no-smoke --no-lifecycle" in engramctl
    assert "Helix native setup did not produce a runnable native runtime." in engramctl


def test_engramctl_update_preserves_helix_native_package_extras() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert 'engine_mode="${ENGRAM_MODE:-lite}"' in engramctl
    assert 'package_spec="engram[local,native]"' in engramctl
    assert 'install_engram_tool_with_native "$package_spec"' in engramctl
    assert (
        'uv tool install --force --reinstall-package engram --with "$helix_native_req"'
        in engramctl
    )
    assert "resolve_helix_native_requirement" in engramctl
    assert "discover_helix_native_release_wheel" in engramctl
    assert "local_helix_native_requirement" in engramctl
    assert (
        engramctl.index('install_engram_tool_with_native "$github_spec"')
        < engramctl.index('install_engram_tool_with_native "$package_spec"')
    )
    assert "cargo --version" in engramctl
    assert "rustup default stable" in engramctl


def test_engramctl_does_not_require_api_key_for_native_quickstart() -> None:
    engramctl = (ROOT / "installer/engramctl").read_text()

    assert "ANTHROPIC_API_KEY is required" not in engramctl
    assert "Anthropic API key (optional, press enter to use deterministic extraction)" in engramctl


def test_openclaw_skill_uses_public_installer_not_empty_native_extra() -> None:
    skill = (ROOT / "skills/engram-memory/SKILL.md").read_text()

    assert '"kind":"shell"' in skill
    assert "install.sh | bash -s -- openclaw" in skill
    assert '"package":"engram[local,native]"' not in skill
    assert "Release wheels are preferred" in skill


def test_custom_helix_native_source_is_packaged_for_release() -> None:
    native_root = ROOT / "native/helix-repo/helix-python"

    assert (ROOT / "native/helix-repo/helix-db").exists()
    assert (ROOT / "native/helix-repo/helix-macros").exists()
    assert (ROOT / "native/helix-repo/metrics").exists()
    assert not (ROOT / "native/helix-repo/helix-container").exists()
    assert (native_root / "pyproject.toml").exists()
    assert (native_root / "Cargo.toml").exists()
    assert (native_root / "build.rs").exists()
    assert (native_root / "src/lib.rs").exists()
    assert (native_root / "src/queries.rs").exists()
    assert "abi3-py310" in (native_root / "Cargo.toml").read_text()
    assert 'join("src/queries.rs")' in (native_root / "build.rs").read_text()


def test_engram_wheel_build_includes_package_directory_without_vcs_filter() -> None:
    pyproject = (ROOT / "server/pyproject.toml").read_text()

    assert "[tool.hatch.build.targets.wheel]" in pyproject
    assert 'packages = ["engram"]' in pyproject
