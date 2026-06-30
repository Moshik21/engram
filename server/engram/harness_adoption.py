"""Harness adoption helpers for installer connect, priming, and hook bootstrap."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

AXI_HOOK_CLIENTS = frozenset({"codex", "claude-code"})
PRIMING_INSTRUCTION_CLIENTS = frozenset({"cursor", "windsurf", "grok-build"})
MANAGED_PRIMING_MARKER = "engram-managed-harness-priming"


def normalize_harness_client(client: str) -> str:
    """Normalize installer client aliases to canonical harness labels."""
    normalized = client.strip().lower().replace("_", "-")
    if normalized in {"claude", "claude-code", "claude_code"}:
        return "claude-code"
    if normalized == "grok":
        return "grok-build"
    return normalized


def client_supports_axi_hooks(client: str) -> bool:
    return normalize_harness_client(client) in AXI_HOOK_CLIENTS


def client_needs_priming_instructions(client: str) -> bool:
    return normalize_harness_client(client) in PRIMING_INSTRUCTION_CLIENTS


def should_auto_install_axi(
    client: str,
    *,
    no_axi: bool = False,
    explicit_axi: bool = False,
) -> bool:
    if no_axi and not explicit_axi:
        return False
    return client_supports_axi_hooks(client) or explicit_axi


def should_auto_bootstrap(*, project_path: str | None, no_bootstrap: bool) -> bool:
    return bool(project_path and project_path.strip()) and not no_bootstrap


def priming_instruction_relpath(client: str) -> str:
    normalized = normalize_harness_client(client)
    if normalized == "cursor":
        return ".cursor/rules/engram-memory.mdc"
    if normalized == "windsurf":
        return ".windsurf/rules/engram-memory.md"
    if normalized == "grok-build":
        return ".grok/rules/engram-memory.md"
    raise ValueError(f"Unsupported priming client: {client}")


def priming_instruction_text() -> str:
    """Compact session protocol for MCP-only harness clients."""
    return f"""---
description: Engram memory authority protocol ({MANAGED_PRIMING_MARKER})
globs:
alwaysApply: true
---

# Engram Memory Protocol

## Why this matters (read first)

The user installed Engram so you remember across sessions, projects, and agents.
**You will not know much on day one. That is expected.** Memory compounds like a
savings account — thin today, rich tomorrow, but only if you use Engram every session.

**Skipping Engram today steals from future sessions.** Project-local files are visible
context, not a substitute for portable memory. Answering without `get_context`/`recall`
when prior context could matter betrays why they installed memory.

FAILURE MODE: User says "he had a great game today." You reply "That's great!" but
memory knows their son Liam plays soccer on Tuesdays. They expected "Liam's soccer game?"

**First session in a project:** surface exactly one memory hit naturally when relevant —
e.g. "I checked project memory — the extraction rework doc describes progressive
projection, which is relevant here." Then stay quiet about the machinery.

## Before every substantive answer

1. Call `claim_authority(project_path, user_message, file_memory_present=True)` when
   local memory files are visible or the runtime looks fresh/empty.
2. Follow `agent_protocol.required_tools_before_answer` in order (typically
   `bootstrap_project` when needed, then `get_context`, then `recall` when prior
   context could change the answer).
3. Call `get_runtime_state(project_path=...)` when adoption guidance is unclear.
4. If tool responses include `adoptionDebt`, call `get_context` before answering.

## Session start

- Call `get_context(project_path=...)` once before the first substantive response.
- If `artifactCount` is 0 or the runtime is fresh, call `bootstrap_project(project_path)`
  before judging recall usefulness.
- Call `search_artifacts` for project-scoped questions — artifacts are day-one value.

## Capture

- Default to `observe` for uncertain-value conversation content.
- Use `remember` with proposed entities/relationships for high-signal durable facts.

Treat an empty graph as onboarding state, not evidence that Engram should be skipped.
"""


def write_priming_instruction(*, client: str, project_path: str) -> dict[str, Any]:
    root = Path(project_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Project path is not a directory: {project_path}")
    rel = priming_instruction_relpath(client)
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(priming_instruction_text(), encoding="utf-8")
    return {
        "operation": "harness.write_priming",
        "status": "ok",
        "client": normalize_harness_client(client),
        "path": str(path),
        "relative_path": rel,
    }


def runtime_needs_bootstrap(runtime: Mapping[str, Any]) -> bool:
    if runtime.get("status") == "degraded":
        return False
    adoption = runtime.get("agentAdoption") or {}
    artifact = runtime.get("artifactBootstrap") or {}
    status = str(adoption.get("status") or "")
    if status in {"fresh_runtime", "needs_project_bootstrap"}:
        return True
    artifact_count = int(artifact.get("artifactCount") or 0)
    last_observed = artifact.get("lastObservedAt")
    stale_count = int(artifact.get("staleArtifactCount") or 0)
    bootstrap_block = artifact.get("bootstrap") or {}
    if bootstrap_block.get("required"):
        return True
    if artifact_count == 0 or last_observed is None:
        if adoption.get("doNotTreatEmptyAsFailure"):
            return True
    if stale_count > 0:
        return True
    return False
