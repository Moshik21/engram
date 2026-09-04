"""Engram's live processes must not need an external model SDK.

Standing rule (2026-09-04): Engram never calls Anthropic, Voyage, Gemini, Cohere
or any external model in operation; the resident harness agent is the only
intelligence. Benchmark code is exempt and keeps its clients behind the
`benchmark` extra. These tests pin the process boundary, not any one module:
`engram serve` and `engram mcp` import cleanly with every such SDK absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SERVER_DIR = Path(__file__).resolve().parents[1]
_EXTERNAL_SDKS = ("anthropic", "voyageai", "google.genai", "cohere")


def test_serve_and_mcp_import_with_every_external_model_sdk_absent() -> None:
    """Simulate the SDKs being uninstalled (sys.modules[name] = None makes import fail)."""
    blockers = "; ".join(f"sys.modules[{name!r}] = None" for name in _EXTERNAL_SDKS)
    script = (
        "import sys; "
        f"{blockers}; "
        "import engram.main, engram.mcp.server; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_SERVER_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


def test_external_model_sdks_are_not_base_dependencies() -> None:
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads((_SERVER_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    def names(requirements: list[str]) -> set[str]:
        return {req.split(">")[0].split("=")[0].split("[")[0].strip() for req in requirements}

    base = names(pyproject["project"]["dependencies"])
    extras = pyproject["project"]["optional-dependencies"]
    assert base.isdisjoint({"anthropic", "voyageai", "google-genai", "cohere"}), base
    assert names(extras["full"]).isdisjoint({"anthropic", "voyageai"}), extras["full"]
    assert "anthropic" in names(extras["benchmark"])
    # Embedding clients are gone entirely: local FastEmbed is the only provider,
    # and the benchmark's Voyage lane speaks raw httpx.
    assert "gemini" not in extras
    every_extra = set().union(*(names(reqs) for reqs in extras.values()))
    assert every_extra.isdisjoint({"voyageai", "google-genai"}), every_extra
