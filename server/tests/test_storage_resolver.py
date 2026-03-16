"""Tests for mode resolution — HelixDB is the sole backend."""

from __future__ import annotations

import pytest

from engram.storage.resolver import EngineMode, resolve_mode


@pytest.mark.asyncio
async def test_resolve_mode_returns_helix():
    mode = await resolve_mode("helix")
    assert mode == EngineMode.HELIX


@pytest.mark.asyncio
async def test_resolve_mode_default_returns_helix():
    mode = await resolve_mode()
    assert mode == EngineMode.HELIX
