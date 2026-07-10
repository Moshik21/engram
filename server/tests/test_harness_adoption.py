"""Tests for harness adoption helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.harness_adoption import (
    MANAGED_PRIMING_MARKER,
    client_needs_priming_instructions,
    client_supports_axi_hooks,
    priming_instruction_relpath,
    priming_instruction_text,
    runtime_needs_bootstrap,
    should_auto_bootstrap,
    should_auto_install_axi,
    write_priming_instruction,
)


def test_client_supports_axi_hooks_for_codex_and_claude() -> None:
    assert client_supports_axi_hooks("codex")
    assert client_supports_axi_hooks("claude-code")
    assert client_supports_axi_hooks("claude")
    assert not client_supports_axi_hooks("cursor")


def test_client_needs_priming_instructions() -> None:
    assert client_needs_priming_instructions("cursor")
    assert client_needs_priming_instructions("windsurf")
    assert client_needs_priming_instructions("grok-build")
    assert client_needs_priming_instructions("grok")
    assert not client_needs_priming_instructions("codex")


def test_should_auto_install_axi_defaults() -> None:
    assert should_auto_install_axi("codex")
    assert should_auto_install_axi("codex", no_axi=True) is False
    assert should_auto_install_axi("cursor", explicit_axi=True)


def test_should_auto_bootstrap_requires_project_path() -> None:
    assert should_auto_bootstrap(project_path="/tmp/proj", no_bootstrap=False)
    assert not should_auto_bootstrap(project_path=None, no_bootstrap=False)
    assert not should_auto_bootstrap(project_path="/tmp/proj", no_bootstrap=True)


def test_priming_instruction_paths() -> None:
    assert priming_instruction_relpath("cursor") == ".cursor/rules/engram-memory.mdc"
    assert priming_instruction_relpath("windsurf") == ".windsurf/rules/engram-memory.md"
    assert priming_instruction_relpath("grok") == ".grok/rules/engram-memory.md"


def test_priming_instruction_text_contains_protocol() -> None:
    text = priming_instruction_text()
    assert MANAGED_PRIMING_MARKER in text
    assert text.index("Why this matters") < text.index("Before every substantive answer")
    assert "Skipping Engram today steals from future sessions" in text
    assert "I checked project memory" in text
    assert "claim_authority" in text
    assert "get_context" in text
    assert "recall" in text
    assert "bootstrap_project" in text
    assert "observe" in text


def test_write_priming_instruction_creates_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    payload = write_priming_instruction(client="cursor", project_path=str(project))
    path = Path(payload["path"])
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "claim_authority" in content
    assert payload["relative_path"] == ".cursor/rules/engram-memory.mdc"


@pytest.mark.parametrize(
    ("runtime", "expected"),
    [
        (
            {
                "agentAdoption": {"status": "fresh_runtime", "doNotTreatEmptyAsFailure": True},
                "artifactBootstrap": {"artifactCount": 0, "lastObservedAt": None},
            },
            True,
        ),
        (
            {
                "agentAdoption": {"status": "ready"},
                "artifactBootstrap": {"artifactCount": 3, "lastObservedAt": "2026-01-01T00:00:00Z"},
            },
            False,
        ),
        (
            {
                "agentAdoption": {"status": "needs_project_bootstrap"},
                "artifactBootstrap": {"artifactCount": 1, "staleArtifactCount": 2},
            },
            True,
        ),
        ({"status": "degraded"}, False),
    ],
)
def test_runtime_needs_bootstrap(runtime: dict, expected: bool) -> None:
    assert runtime_needs_bootstrap(runtime) is expected
