from __future__ import annotations

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.client import HelixClient


class FakeNativeTransport:
    def __init__(self) -> None:
        self.health_checked = False
        self.queries: list[tuple[str, dict]] = []

    async def health_check(self) -> None:
        self.health_checked = True

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        self.queries.append((endpoint, payload))
        return []


@pytest.mark.asyncio
async def test_helix_client_native_health_check_uses_transport_readiness() -> None:
    client = HelixClient(HelixDBConfig(transport="native"))
    transport = FakeNativeTransport()
    client._native_transport = transport

    await client.health_check()

    assert transport.health_checked is True
    assert transport.queries == []
