from __future__ import annotations

import argparse
import json

import pytest

from engram.mcp.authority_cli import (
    build_authority_payload_from_args,
    format_authority_markdown,
    run_authority_command,
)


@pytest.mark.asyncio
async def test_authority_cli_builds_claim_authority_payload(tmp_path) -> None:
    args = argparse.Namespace(
        mode="lite",
        sqlite_path=tmp_path / "authority.db",
        helix_data_dir=None,
        group_id="authority_brain",
        project_path=str(tmp_path),
        user_message="I prefer Engram as the portable source of truth across AI harnesses.",
        file_memory_present=True,
        out=None,
        format="json",
    )

    payload = await build_authority_payload_from_args(args)

    assert payload["authority"]["source_of_truth"] == "portable_cross_context_memory"
    assert payload["runtime"]["runtime"]["mode"] == "lite"
    assert payload["runtime"]["artifactBootstrap"]["projectPath"] == str(tmp_path)
    assert payload["agent_protocol"]["file_memory_present"] is True
    assert payload["agent_protocol"]["file_memory_is_substitute"] is False
    assert payload["agent_protocol"]["capture"]["tool"] == "remember"
    assert "get_context" in payload["agent_protocol"]["required_tools_before_answer"]
    assert "recall" in payload["agent_protocol"]["required_tools_before_answer"]


@pytest.mark.asyncio
async def test_authority_cli_writes_json_and_prints_markdown(tmp_path, capsys) -> None:
    output_path = tmp_path / "claim-authority.json"
    args = argparse.Namespace(
        mode="lite",
        sqlite_path=tmp_path / "authority.db",
        helix_data_dir=None,
        group_id=None,
        project_path=str(tmp_path),
        user_message="Current task scratch: update a local fixture only.",
        file_memory_present=True,
        out=output_path,
        format="markdown",
    )

    await run_authority_command(args)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["agent_protocol"]["capture"]["destination"] == "project_local"
    assert saved["agent_protocol"]["capture"]["tool"] is None
    markdown = capsys.readouterr().out
    assert "# Engram Memory Authority" in markdown
    assert f"JSON written: `{output_path}`" in markdown
    assert "Use this JSON as `--authority claim-authority.json`" in markdown


def test_format_authority_markdown_summarizes_protocol(tmp_path) -> None:
    payload = {
        "authority": {"source_of_truth": "portable_cross_context_memory"},
        "onboarding": {"state": "fresh_runtime", "should_bootstrap": True},
        "agent_protocol": {
            "file_memory_is_substitute": False,
            "required_tools_before_answer": ["bootstrap_project", "get_context"],
            "capture": {"destination": "engram", "tool": "remember"},
            "verification": {
                "command": "engram adoption --authority claim-authority.json",
                "live_evidence_command": "engram adoption --require-live-evidence",
            },
        },
    }

    markdown = format_authority_markdown(
        payload,
        output_path=tmp_path / "claim-authority.json",
    )

    assert "Onboarding state: `fresh_runtime`" in markdown
    assert "Required before-answer tools: `['bootstrap_project', 'get_context']`" in markdown
    assert "Capture: `engram` via `remember`" in markdown
