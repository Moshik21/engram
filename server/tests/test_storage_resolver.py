"""Tests for storage mode resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.storage import resolver
from engram.storage.resolver import EngineMode, resolve_mode

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_resolve_explicit_helix_native_requires_extension(monkeypatch):
    monkeypatch.setenv("ENGRAM_HELIX__TRANSPORT", "native")
    monkeypatch.setattr(resolver, "_check_native", lambda: False)

    with pytest.raises(RuntimeError, match="helix_native PyO3 extension"):
        await resolve_mode("helix")


@pytest.mark.asyncio
async def test_resolve_explicit_helix_native_returns_helix(monkeypatch):
    monkeypatch.setenv("ENGRAM_HELIX__TRANSPORT", "native")
    monkeypatch.setattr(resolver, "_check_native", lambda: True)

    mode = await resolve_mode("helix")

    assert mode == EngineMode.HELIX


@pytest.mark.asyncio
async def test_resolve_auto_uses_native_when_available(monkeypatch):
    monkeypatch.delenv("ENGRAM_MODE", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__TRANSPORT", raising=False)
    monkeypatch.setattr(resolver, "_check_native", lambda: True)

    mode = await resolve_mode()

    assert mode == EngineMode.HELIX
    assert resolver.os.environ["ENGRAM_HELIX__TRANSPORT"] == "native"


@pytest.mark.asyncio
async def test_resolve_explicit_helix_http_requires_service(monkeypatch):
    async def helix_unavailable() -> bool:
        return False

    monkeypatch.setenv("ENGRAM_HELIX__TRANSPORT", "http")
    monkeypatch.setattr(resolver, "_check_helix", helix_unavailable)

    with pytest.raises(RuntimeError, match="Helix mode requested but HelixDB is not reachable"):
        await resolve_mode("helix")


@pytest.mark.asyncio
async def test_resolve_auto_falls_back_to_lite_when_no_services(monkeypatch):
    async def unavailable() -> bool:
        return False

    monkeypatch.delenv("ENGRAM_MODE", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__TRANSPORT", raising=False)
    monkeypatch.setattr(resolver, "_check_native", lambda: False)
    monkeypatch.setattr(resolver, "_check_helix", unavailable)
    monkeypatch.setattr(resolver, "_check_falkordb", unavailable)
    monkeypatch.setattr(resolver, "_check_redis", unavailable)

    mode = await resolve_mode()

    assert mode == EngineMode.LITE


def test_makefile_native_shortcuts_force_native_transport() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert "up-native: build-native" in makefile
    assert "mcp-native: build-native" in makefile
    assert "ENGRAM_HELIX__TRANSPORT=native" in makefile
