"""Tests for the interactive setup wizard and config editor."""

from engram.setup import (
    _HOOK_SCRIPT_TEMPLATES,
    _HOOK_SCRIPTS,
    _ask,
    _collect_config,
    _generate_env,
    _load_env,
    _mask_value,
    _print_mcp_config,
    _render_menu,
    _repo_hook_script,
    _smoke_test,
    _welcome,
    config_editor,
    install_hooks_interactive,
)

_EXTERNAL_KEY_TOKENS = ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY")


def _refuse_secret_prompt(_prompt: str) -> str:
    raise AssertionError("setup must not prompt for a secret / API key")


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
    """_generate_env writes expected keys to file and no external API key line."""
    env_path = tmp_path / ".env"
    config = {
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
    assert "ENGRAM_MODE=lite" in content
    assert "ENGRAM_ACTIVATION__INTEGRATION_PROFILE=off" in content
    # Unconfigured values should be commented
    assert "# ENGRAM_AUTH__ENABLED=" in content
    # No external-model key line, not even a commented placeholder.
    assert "API Keys" not in content
    for token in _EXTERNAL_KEY_TOKENS:
        assert token not in content


def test_generate_env_backs_up_existing(tmp_path):
    """_generate_env creates backup of existing .env."""
    env_path = tmp_path / ".env"
    env_path.write_text("OLD=content\n")

    config = {
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
    assert "ENGRAM_MODE=auto" in env_path.read_text()
    # Backup should exist
    backups = list(tmp_path.glob(".env.backup.*"))
    assert len(backups) == 1
    assert "OLD=content" in backups[0].read_text()


def test_mcp_config_output(capsys):
    """MCP config output contains correct structure."""
    config = {
        "ENGRAM_MODE": "auto",
        "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "standard",
        "ENGRAM_ACTIVATION__RECALL_PROFILE": "all",
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": "rework",
    }
    _print_mcp_config(config)
    out = capsys.readouterr().out
    assert "mcpServers" in out
    assert "engram.mcp.server" in out
    # The pasted MCP env block carries no external-model key, not even empty.
    for token in _EXTERNAL_KEY_TOKENS:
        assert token not in out
    assert "ENGRAM_MODE" in out
    assert "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE" in out
    assert "ENGRAM_ACTIVATION__RECALL_PROFILE" in out
    assert "ENGRAM_ACTIVATION__INTEGRATION_PROFILE" in out
    assert "Claude Desktop" in out
    assert "Claude Code" in out
    assert "Agent adoption checklist" in out
    assert "claim_authority(project_path, user_message, file_memory_present=True)" in out
    assert "agent_protocol.required_tools_before_answer" in out
    assert "bootstrap_project(project_path)" in out
    assert "`remember` high-signal cross-context facts" in out
    assert "engram adoption --authority claim-authority.json" in out


def test_collect_config_defaults_are_recall_ready(monkeypatch, capsys):
    """Wizard defaults should produce a practical end-to-end MCP setup."""
    responses = iter(
        [
            "auto",  # mode
            "engram_dev",  # Falkor password
            "engram_dev",  # Redis password
            "",  # consolidation profile -> default quiet (consumer)
            "",  # recall profile -> default wave2
            "",  # integration profile -> default off
            "n",  # auth
            "n",  # encryption
        ]
    )
    # Any secret prompt (the old Anthropic / Voyage key questions) fails the test.
    monkeypatch.setattr("engram.setup.getpass.getpass", _refuse_secret_prompt)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    config = _collect_config()

    assert config["ENGRAM_MODE"] == "auto"
    # Consumer defaults: quiet shell footprint, wave2 recall, integration off.
    assert config["ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE"] == "quiet"
    assert config["ENGRAM_ACTIVATION__RECALL_PROFILE"] == "wave2"
    assert config["ENGRAM_ACTIVATION__INTEGRATION_PROFILE"] == "off"
    # No external-model key is asked for, required, or carried in the config.
    for token in _EXTERNAL_KEY_TOKENS:
        assert token not in config
    out = capsys.readouterr().out
    assert "API Keys" not in out
    assert "the resident agent proposes; narrow writes cues" in out


def test_collect_config_helix_uses_native_transport(monkeypatch):
    """Wizard helix mode should select the no-Docker native transport."""
    input_responses = iter(
        [
            "helix",  # mode
            "",  # consolidation profile -> default quiet (consumer)
            "",  # recall profile -> default wave2
            "",  # integration profile -> default off
            "n",  # auth
            "n",  # encryption
        ]
    )
    monkeypatch.setattr("engram.setup.getpass.getpass", _refuse_secret_prompt)
    monkeypatch.setattr("builtins.input", lambda _: next(input_responses))

    config = _collect_config()

    assert config["ENGRAM_MODE"] == "helix"
    assert config["ENGRAM_HELIX__TRANSPORT"] == "native"
    assert config["ENGRAM_FALKORDB__PASSWORD"] is None
    assert config["ENGRAM_REDIS__URL"] is None


def test_hooks_interactive_prints_live_adoption_verifier_command(capsys, tmp_path):
    """Hook installer output should tell operators how to verify live traces."""
    install_hooks_interactive(
        hooks_dir=tmp_path / "hooks",
        settings_path=tmp_path / "settings.json",
    )

    out = capsys.readouterr().out
    assert "Validate a live client run" in out
    assert "engram adoption --authority claim-authority.json" in out
    assert "--calls claude-stream.jsonl ~/.engram/adoption-trace.jsonl" in out
    assert "--session-id <client-session-id>" in out
    assert "--require-live-evidence" in out


# --- Config editor tests ---


def test_load_env_parses_file(tmp_path):
    """_load_env parses key=value lines, skips comments."""
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nENGRAM_AUTH__BEARER_TOKEN=tok-123\n# ENGRAM_REDIS__URL=\nENGRAM_MODE=lite\n\n"
    )
    result = _load_env(env)
    assert result == {"ENGRAM_AUTH__BEARER_TOKEN": "tok-123", "ENGRAM_MODE": "lite"}


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
    config = {"ENGRAM_MODE": "auto"}
    keys = _render_menu(config, env_path, dirty=False)
    out = capsys.readouterr().out
    # Should show section headers -- and no external-model key section.
    assert "API Keys" not in out
    assert "Engine" in out
    assert "Security" in out
    # Should show numbered settings
    assert "1." in out
    assert "Anthropic API key" not in out
    # Should return all keys
    assert len(keys) == 11
    for token in _EXTERNAL_KEY_TOKENS:
        assert token not in keys
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
    env_path.write_text("ENGRAM_MODE=lite\n")
    monkeypatch.setattr("builtins.input", lambda _: "q")
    config_editor(env_path=env_path)  # should not hang


# --- No external model, no external keys (2026-09-04 resident-agent rule) ---


def test_setup_never_references_external_model_keys():
    """Anti-resurrection: the wizard and every hook it installs carry no external key."""
    import inspect

    import engram.setup as mod

    source = inspect.getsource(mod)
    assert "import anthropic" not in source
    for token in _EXTERNAL_KEY_TOKENS:
        assert token not in source

    # Hook scripts the wizard writes (in-file templates + repo first-party scripts).
    for name in _HOOK_SCRIPTS:
        template = _HOOK_SCRIPT_TEMPLATES.get(name)
        repo_src = _repo_hook_script(name)
        text = template or (repo_src.read_text() if repo_src else "")
        assert text, f"hook {name} has neither a template nor a repo script"
        assert "API_KEY" not in text, f"hook {name} references an external key"


class _StubLocalEmbedder:
    """Stand-in for FastEmbedProvider: no ONNX download in tests."""

    vectors: list[list[float]] = [[0.1, 0.2, 0.3]]

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def embed(self, texts):
        return [list(v) for v in self.vectors for _ in texts]


def test_smoke_test_probes_local_embedder(monkeypatch, capsys):
    """Smoke test embeds one string with the LOCAL embedder; no API key involved."""
    monkeypatch.setattr("engram.embeddings.provider.FastEmbedProvider", _StubLocalEmbedder)

    _smoke_test({})

    out = capsys.readouterr().out
    assert "engram package importable" in out
    assert "Local embedder reachable" in out
    assert "dim=3" in out
    assert "anthropic" not in out.lower()
    assert "API key" not in out


def test_smoke_test_warns_when_local_embedder_cannot_embed(monkeypatch, capsys):
    """A broken model cache surfaces as a warning, not a silent pass."""

    class _Broken(_StubLocalEmbedder):
        vectors = []

    monkeypatch.setattr("engram.embeddings.provider.FastEmbedProvider", _Broken)

    _smoke_test({})

    out = capsys.readouterr().out
    assert "cannot embed" in out
    assert "FASTEMBED_CACHE_PATH" in out


def test_capture_response_hook_marks_truncation():
    """The hook template no longer cuts responses silently (2026-09-04)."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "engram" / "setup.py"
    text = source.read_text()
    assert 'RESPONSE="${RESPONSE:0:2000}"\n' not in text
    assert "[engram: truncated at 2000 of ${#RESPONSE} chars]" in text
