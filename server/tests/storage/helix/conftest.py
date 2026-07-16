"""Test fixtures for HelixDB storage tests.

Product path is **native PyO3 + data-dir**. HTTP :6969 remains a secondary
compatibility path. Tests are marked ``requires_helix`` and skip when neither
backend is available.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio  # noqa: F401  -- needed for async fixture decorator

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.storage.helix.availability import (
    helix_available as probe_helix,
)
from engram.storage.helix.availability import (
    helix_http_available,
    helix_native_available,
)


def helix_available() -> bool:
    """True when native PyO3 (preferred) or HTTP Helix is usable."""
    return bool(probe_helix(prefer_native=True).get("available"))


def _native_data_dir() -> Path:
    """Disposable native data dir for integration tests."""
    env = os.environ.get("ENGRAM_TEST_HELIX_DATA_DIR")
    if env:
        path = Path(env).expanduser().resolve()
    else:
        path = Path(os.environ.get("TMPDIR", "/tmp")) / "engram-helix-pytest-native"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_helix_config() -> HelixDBConfig:
    """Prefer native data-dir; fall back to HTTP localhost:6969."""
    if helix_native_available():
        return HelixDBConfig(
            transport="native",
            data_dir=str(_native_data_dir()),
            verbose=False,
        )
    if helix_http_available():
        return HelixDBConfig(
            host="localhost",
            port=6969,
            transport="http",
            verbose=False,
        )
    # Unreachable when pytestmark skipif works; keep for direct imports.
    return HelixDBConfig(host="localhost", port=6969, verbose=False)


_probe = probe_helix(prefer_native=True)
_skip_reason = f"Helix not available (native={_probe.get('native')}, http={_probe.get('http')})"

pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not helix_available(), reason=_skip_reason),
]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def helix_config() -> HelixDBConfig:
    """HelixDB config: native data-dir when available, else HTTP."""
    return _build_helix_config()


@pytest.fixture
def test_group_id() -> str:
    """Unique group ID per test to guarantee isolation."""
    return f"test_{uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def helix_graph_store(helix_config: HelixDBConfig, test_group_id: str):
    """Initialized HelixGraphStore with per-test cleanup."""
    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(helix_config)
    try:
        await store.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")

    yield store

    # Cleanup: remove all data for this test's group
    try:
        await store.delete_group(test_group_id)
    except Exception:
        pass
    await store.close()


@pytest_asyncio.fixture
async def helix_search_index(helix_config: HelixDBConfig, test_group_id: str):
    """Initialized HelixSearchIndex with a noop embedding provider."""
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    provider = NoopProvider()
    embed_config = EmbeddingConfig()
    index = HelixSearchIndex(
        helix_config=helix_config,
        provider=provider,
        embed_config=embed_config,
        storage_dim=0,
        embed_provider="noop",
        embed_model="noop",
    )
    try:
        await index.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")

    yield index

    # Cleanup
    try:
        await index.delete_group(test_group_id)
    except Exception:
        pass
    await index.close()
