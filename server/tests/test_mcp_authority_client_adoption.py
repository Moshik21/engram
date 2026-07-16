from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from engram.retrieval.memory_authority import validate_agent_protocol_calls


@pytest.mark.asyncio
async def test_stdio_mcp_client_can_follow_claim_authority_protocol(tmp_path: Path) -> None:
    await asyncio.wait_for(_run_stdio_authority_protocol(tmp_path), timeout=30)


async def _run_stdio_authority_protocol(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "README.md").write_text(
        "# Engram adoption fixture\n\nThis project validates MCP memory authority.\n",
        encoding="utf-8",
    )

    user_message = (
        "I am building Engram as the cross-context AI memory brain and prefer it "
        "as the portable source of truth across AI harnesses."
    )
    transcript: list[dict[str, str]] = []
    server_dir = Path(__file__).resolve().parents[1]
    env = _stdio_test_env(tmp_path)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "engram", "mcp", "--transport", "stdio"],
        cwd=server_dir,
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            authority = await _call_json_tool(
                session,
                "claim_authority",
                {
                    "project_path": str(project_dir),
                    "user_message": user_message,
                    "file_memory_present": True,
                },
            )
            protocol = authority["agent_protocol"]

            assert authority["onboarding"]["state"] in {
                "fresh_runtime",
                "needs_project_bootstrap",
            }
            assert authority["onboarding"]["should_bootstrap"] is True
            assert protocol["required_tools_before_answer"] == [
                "bootstrap_project",
                "get_context",
                "recall",
            ]
            assert protocol["capture"]["tool"] == "remember"
            assert protocol["verification"]["transcript_schema"]["format"] == "jsonl"
            assert protocol["verification"]["command"].startswith("engram adoption")
            assert protocol["verification"]["capture_required"] is True

            for action in protocol["before_answer"]:
                tool_name = action["tool"]
                await _call_json_tool(
                    session,
                    tool_name,
                    action.get("args") or {},
                )
                transcript.append({"phase": "before_answer", "tool": tool_name})

            await _call_json_tool(
                session,
                protocol["capture"]["tool"],
                {"content": user_message, "source": "mcp_adoption_test"},
            )
            transcript.append({"phase": "capture", "tool": protocol["capture"]["tool"]})

    validation = validate_agent_protocol_calls(protocol, transcript)

    assert validation["status"] == "passed"
    assert validation["required_tools_before_answer"]["observed"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert validation["capture"]["observed_tools"] == ["remember"]
    assert validation["file_memory"]["substituted_for_engram"] is False


def _stdio_test_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "VOYAGE_API_KEY"):
        env.pop(key, None)
    env.update(
        {
            "ENGRAM_MODE": "lite",
            "ENGRAM_TRANSPORT": "stdio",
            # The bootstrap flow under test requires the artifact substrate;
            # dev/CI env profiles must not decide this test's behavior.
            "ENGRAM_ACTIVATION__ARTIFACT_BOOTSTRAP_ENABLED": "true",
            "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "off",
            "ENGRAM_SQLITE__PATH": str(tmp_path / "engram-mcp-adoption.db"),
            "ENGRAM_EMBEDDING__PROVIDER": "noop",
            "ENGRAM_DEFAULT_GROUP_ID": "mcp_adoption_test",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


async def _call_json_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await session.call_tool(name, arguments)
    assert result.content
    text = getattr(result.content[0], "text", "")
    assert text
    return json.loads(text)
