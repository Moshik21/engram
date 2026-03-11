"""Tests for the interactive setup wizard and config editor."""

from engram.setup import (
    _ask,
    _collect_config,
    _generate_env,
    _load_env,
    _mask_value,
    _print_mcp_config,
    _render_menu,
    _welcome,
    config_editor,
)


def test_welcome_prints(capsys):
    """Welcome screen prints without error."""
    _welcome()
    out = capsys.readouterr().out
    assert "Engram" in out
    assert "memory" in out


def test_ask_default(monkeypatch):
    """_ask with default returns default on empty input."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    result = _ask("Test", default="hello")
    assert result == "hello"


def test_ask_choices_valid(monkeypatch):
    """_ask with choices accepts valid input."""
    monkeypatch.setattr("builtins.input", lambda _: "lite")
    result = _ask("Mode", choices=["lite", "full", "auto"])
    assert result == "lite"


def test_ask_choices_rejects_invalid(monkeypatch):
    """_ask with choices rejects invalid then accepts valid."""
    responses = iter(["bad", "lite"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    result = _ask("Mode", choices=["lite", "full", "auto"])
    assert result == "lite"


def test_ask_required_rejects_empty(monkeypatch):
    """_ask without default rejects empty input then accepts value."""
    responses = iter(["", "value"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    result = _ask("Required field")
    assert result == "value"


def test_ask_secret(monkeypatch):
    """_ask with secret=True uses getpass."""
    monkeypatch.setattr("engram.setup.getpass.getpass", lambda _: "secret123")
    result = _ask("Key", secret=True)
    assert result == "secret123"


def test_generate_env_writes_keys(tmp_path):
    """_generate_env writes expected keys to file."""
    env_path = tmp_path / ".env"
    config = {
        "ANTHROPIC_API_KEY": "sk-test-123",
        "VOYAGE_API_KEY": None,
        "ENGRAM_MODE": "lite",
        "ENGRAM_FALKORDB__PASSWORD": None,
        "ENGRAM_REDIS__URL": None,
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "off",
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": "off",
        "ENGRAM_AUTH__ENABLED": None,
        "ENGRAM_AUTH__BEARER_TOKEN": None,
        "ENGRAM_ENCRYPTION__ENABLED": None,
        "ENGRAM_ENCRYPTION__MASTER_KEY": None,
    }
    _generate_env(config, env_path)
    content = env_path.read_text()
    assert "ANTHROPIC_API_KEY=sk-test-123" in content
    assert "ENGRAM_MODE=lite" in content
    assert "ENGRAM_ACTIVATION__INTEGRATION_PROFILE=off" in content
    # Unconfigured values should be commented
    assert "# VOYAGE_API_KEY=" in content


def test_generate_env_backs_up_existing(tmp_path):
    """_generate_env creates backup of existing .env."""
    env_path = tmp_path / ".env"
    env_path.write_text("OLD=content\n")

    config = {
        "ANTHROPIC_API_KEY": "sk-new",
        "VOYAGE_API_KEY": None,
        "ENGRAM_MODE": "auto",
        "ENGRAM_FALKORDB__PASSWORD": None,
        "ENGRAM_REDIS__URL": None,
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "off",
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": "off",
        "ENGRAM_AUTH__ENABLED": None,
        "ENGRAM_AUTH__BEARER_TOKEN": None,
        "ENGRAM_ENCRYPTION__ENABLED": None,
        "ENGRAM_ENCRYPTION__MASTER_KEY": None,
    }
    _generate_env(config, env_path)

    # New file should have new content
    assert "sk-new" in env_path.read_text()
    # Backup should exist
    backups = list(tmp_path.glob(".env.backup.*"))
    assert len(backups) == 1
    assert "OLD=content" in backups[0].read_text()


def test_mcp_config_output(capsys):
    """MCP config output contains correct structure."""
    config = {
        "ANTHROPIC_API_KEY": "sk-test-xyz",
        "VOYAGE_API_KEY": None,
        "ENGRAM_MODE": "auto",
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "standard",
        "ENGRAM_ACTIVATION__RECALL_PROFILE": "all",
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": "rework",
    }
    _print_mcp_config(config)
    out = capsys.readouterr().out
    assert "mcpServers" in out
    assert "engram.mcp.server" in out
    assert "sk-test-xyz" in out
    assert "ENGRAM_MODE" in out
    assert "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE" in out
    assert "ENGRAM_ACTIVATION__RECALL_PROFILE" in out
    assert "ENGRAM_ACTIVATION__INTEGRATION_PROFILE" in out
    assert "Claude Desktop" in out
    assert "Claude Code" in out


def test_collect_config_defaults_are_recall_ready(monkeypatch):
    """Wizard defaults should produce a practical end-to-end MCP setup."""
    responses = iter(
        [
            "auto",  # mode
            "engram_dev",  # Falkor password
            "engram_dev",  # Redis password
            "",  # consolidation profile -> default standard
            "",  # recall profile -> default all
            "",  # integration profile -> default rework
            "n",  # auth
            "n",  # encryption
        ]
    )
    monkeypatch.setattr("engram.setup.getpass.getpass", lambda _: "secret123")
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    config = _collect_config()

    assert config["ENGRAM_MODE"] == "auto"
    assert config["ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE"] == "standard"
    assert config["ENGRAM_ACTIVATION__RECALL_PROFILE"] == "all"
    assert config["ENGRAM_ACTIVATION__INTEGRATION_PROFILE"] == "rework"


# --- Config editor tests ---


def test_load_env_parses_file(tmp_path):
    """_load_env parses key=value lines, skips comments."""
    env = tmp_path / ".env"
    env.write_text("# comment\nANTHROPIC_API_KEY=sk-123\n# VOYAGE_API_KEY=\nENGRAM_MODE=lite\n\n")
    result = _load_env(env)
    assert result == {"ANTHROPIC_API_KEY": "sk-123", "ENGRAM_MODE": "lite"}


def test_load_env_missing_file(tmp_path):
    """_load_env returns empty dict for missing file."""
    result = _load_env(tmp_path / "nonexistent")
    assert result == {}


def test_mask_value_hides_secrets():
    """_mask_value masks long secrets, shows short ones as dots."""
    # Long secret: show first 4 + dots + last 4
    masked = _mask_value("sk-ant-very-long-key-1234", True)
    assert masked.startswith("sk-a")
    assert masked.endswith("1234")
    assert "•" in masked

    # Short secret: all dots
    masked = _mask_value("short", True)
    assert "•" in masked or "\u2022" in masked

    # Not secret: shown as-is
    assert _mask_value("lite", False) == "lite"

    # Empty: shows "(not set)"
    assert "not set" in _mask_value("", False)


def test_render_menu_shows_all_settings(capsys, tmp_path):
    """_render_menu displays all settings with numbers."""
    env_path = tmp_path / ".env"
    config = {"ANTHROPIC_API_KEY": "sk-test", "ENGRAM_MODE": "auto"}
    keys = _render_menu(config, env_path, dirty=False)
    out = capsys.readouterr().out
    # Should show section headers
    assert "API Keys" in out
    assert "Engine" in out
    assert "Security" in out
    # Should show numbered settings
    assert "1." in out
    assert "Anthropic API key" in out
    # Should return all keys
    assert len(keys) == 12
    assert "ANTHROPIC_API_KEY" in keys
    assert "ENGRAM_ACTIVATION__RECALL_PROFILE" in keys
    assert "ENGRAM_ACTIVATION__INTEGRATION_PROFILE" in keys


def test_render_menu_shows_unsaved_indicator(capsys, tmp_path):
    """_render_menu shows unsaved changes indicator."""
    _render_menu({}, tmp_path / ".env", dirty=True)
    out = capsys.readouterr().out
    assert "unsaved" in out.lower()


def test_config_editor_no_file(capsys, tmp_path):
    """config_editor prints message when no .env exists."""
    config_editor(env_path=tmp_path / "missing" / ".env")
    out = capsys.readouterr().out
    assert "No config found" in out
    assert "setup" in out


def test_config_editor_quit(monkeypatch, tmp_path):
    """config_editor exits on 'q'."""
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=sk-test\n")
    monkeypatch.setattr("builtins.input", lambda _: "q")
    config_editor(env_path=env_path)  # should not hang
