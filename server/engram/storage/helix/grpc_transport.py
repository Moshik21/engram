"""gRPC transport for HelixDB -- binary protobuf framing over HTTP/2.

Replaces JSON-over-HTTP with protobuf-over-gRPC for lower serialization
overhead and native multiplexing. Falls back gracefully if gRPC is
unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import grpc

    from engram.storage.helix.proto import helix_pb2, helix_pb2_grpc

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


class GrpcTransport:
    """Async gRPC transport for HelixDB.

    Mirrors the HelixClient HTTP interface so it can be used as a
    drop-in transport replacement.
    """

    def __init__(self, config: Any) -> None:
        if not HAS_GRPC:
            raise ImportError(
                "grpcio is required for gRPC transport. "
                "Install with: pip install engram[grpc]"
            )
        self._config = config
        self._channel: Any = None
        self._stub: Any = None

    async def initialize(self) -> None:
        """Create the gRPC channel and stub."""
        host = self._config.host
        if host == "localhost":
            host = "127.0.0.1"
        target = f"{host}:{self._config.grpc_port}"

        self._channel = grpc.aio.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self._stub = helix_pb2_grpc.HelixDBStub(self._channel)

        # Health check
        try:
            req = helix_pb2.QueryRequest(
                name="find_entities_by_group",
                body=json.dumps({"gid": "__health_check__"}).encode(),
            )
            await self._stub.Query(req, timeout=5.0)
            logger.info("GrpcTransport connected (%s)", target)
        except grpc.RpcError:
            raise RuntimeError(
                f"HelixDB gRPC not reachable at {target}. "
                f"Ensure HelixDB is running with gRPC enabled."
            )

    async def close(self) -> None:
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a single query via gRPC unary call."""
        if self._stub is None:
            raise RuntimeError("GrpcTransport not initialized")

        try:
            req = helix_pb2.QueryRequest(
                name=endpoint,
                body=json.dumps(payload).encode(),
            )
            resp = await self._stub.Query(req)

            if resp.status != 200:
                if resp.error and ("NoValue" in resp.error or "NotFound" in resp.error):
                    return []
                if resp.status == 404:
                    return []
                logger.warning(
                    "gRPC %s returned status %d: %s",
                    endpoint,
                    resp.status,
                    resp.error[:200],
                )
                return []

            if not resp.body:
                return []

            raw = json.loads(resp.body)
            if raw is None:
                return []
            if isinstance(raw, dict):
                raw = [raw]

            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(raw)

        except grpc.RpcError as e:
            code = e.code() if hasattr(e, "code") else None
            if code == grpc.StatusCode.UNAVAILABLE:
                logger.error("HelixDB gRPC connection lost for %s", endpoint)
            else:
                logger.warning("gRPC error for %s: %s", endpoint, e)
            return []

    async def query_many(
        self,
        endpoint: str,
        payloads: list[dict],
        max_concurrent: int = 8,
    ) -> list[list[dict]]:
        """Execute multiple payloads against the same endpoint."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(await asyncio.gather(*[_one(p) for p in payloads]))

    async def query_concurrent(
        self,
        queries: list[tuple[str, dict]],
        max_concurrent: int = 8,
    ) -> list[list[dict]]:
        """Execute multiple queries. Uses batch RPC for efficiency."""
        if len(queries) > 1:
            return await self.batch(queries)

        sem = asyncio.Semaphore(max_concurrent)

        async def _one(endpoint: str, payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(await asyncio.gather(*[_one(ep, pl) for ep, pl in queries]))

    async def batch(
        self,
        queries: list[tuple[str, dict]],
    ) -> list[list[dict]]:
        """Execute multiple queries via gRPC Batch RPC."""
        if self._stub is None:
            raise RuntimeError("GrpcTransport not initialized")

        if not queries:
            return []

        try:
            batch_req = helix_pb2.BatchRequest(
                queries=[
                    helix_pb2.QueryRequest(
                        name=endpoint,
                        body=json.dumps(body).encode(),
                    )
                    for endpoint, body in queries
                ]
            )
            resp = await self._stub.Batch(batch_req)

            from engram.storage.helix import unwrap_helix_results

            results: list[list[dict]] = []
            for entry in resp.results:
                if entry.status == 200:
                    if not entry.body:
                        results.append([])
                    else:
                        raw = json.loads(entry.body)
                        if isinstance(raw, dict):
                            raw = [raw]
                        results.append(unwrap_helix_results(raw) if raw else [])
                else:
                    if entry.error:
                        logger.warning("Batch sub-query failed: %s", entry.error)
                    results.append([])

            return results

        except grpc.RpcError as e:
            logger.error("gRPC batch failed: %s", e)
            # Fall back to individual queries
            return await self._fallback_concurrent(queries)

    async def _fallback_concurrent(
        self,
        queries: list[tuple[str, dict]],
    ) -> list[list[dict]]:
        """Individual concurrent queries -- fallback when batch fails."""
        sem = asyncio.Semaphore(8)

        async def _one(endpoint: str, payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(await asyncio.gather(*[_one(ep, pl) for ep, pl in queries]))

    @property
    def is_connected(self) -> bool:
        return self._stub is not None
