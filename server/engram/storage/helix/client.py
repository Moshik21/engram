"""Async HelixDB client with connection pooling, batching, and concurrency.

Replaces the ``helix-py`` SDK's synchronous ``urllib`` transport with
``httpx.AsyncClient`` for persistent HTTP/1.1 keep-alive connections.

Three query modes:
- ``query()``          — single async query (persistent connection, no thread pool)
- ``query_many()``     — batch N payloads to the same endpoint concurrently
- ``query_concurrent()`` — fire multiple (endpoint, payload) pairs concurrently
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from engram.config import HelixDBConfig

logger = logging.getLogger(__name__)


class HelixClient:
    """Async HelixDB HTTP client with connection pooling.

    One instance should be shared across all Helix stores for maximum
    connection reuse.
    """

    def __init__(self, config: HelixDBConfig) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._base_url: str = ""
        self._batch_available: bool = True  # optimistic; falls back on 404
        self._grpc_transport: object | None = None
        self._native_transport: object | None = None

    async def initialize(self) -> None:
        """Create the httpx client with persistent connection pool."""
        if self._config.api_endpoint:
            self._base_url = self._config.api_endpoint.rstrip("/")
        else:
            # Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues
            host = self._config.host
            if host == "localhost":
                host = "127.0.0.1"
            self._base_url = f"http://{host}:{self._config.port}"

        # Try native in-process transport FIRST (highest priority — zero network overhead)
        if self._config.transport in ("native", "auto"):
            try:
                from engram.storage.helix.native_transport import NativeTransport

                self._native_transport = NativeTransport(self._config)
                await self._native_transport.initialize()
                logger.info("HelixClient using native in-process transport")
                return  # Skip HTTP/gRPC setup entirely
            except (ImportError, RuntimeError) as e:
                if self._config.transport == "native":
                    raise  # User explicitly requested native, don't fall back
                logger.debug(
                    "Native transport unavailable (%s), trying other transports", e
                )
                self._native_transport = None

        # HTTP client for non-native transports
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=30.0,
            ),
            timeout=httpx.Timeout(60.0, connect=5.0),
            headers={
                "Content-Type": "application/json",
                **({"x-api-key": self._config.api_key} if self._config.api_key else {}),
            },
        )

        # Verify HTTP connectivity
        try:
            await self._client.post(
                "/find_entities_by_group",
                content=json.dumps({"gid": "__health_check__"}),
            )
            logger.info(
                "HelixClient connected (%s, pool=%d)",
                self._base_url,
                self._config.max_workers * 2,
            )
        except httpx.ConnectError:
            raise RuntimeError(
                f"HelixDB not reachable at {self._base_url}. "
                f"Ensure HelixDB is running (helix push dev or Docker)."
            )

        # If gRPC transport is configured, set up the gRPC transport
        # and delegate query methods to it
        if self._config.transport == "grpc":
            try:
                from engram.storage.helix.grpc_transport import GrpcTransport

                self._grpc_transport = GrpcTransport(self._config)
                await self._grpc_transport.initialize()
                logger.info("HelixClient using gRPC transport")
            except (ImportError, RuntimeError) as e:
                logger.warning("gRPC transport unavailable (%s), falling back to HTTP", e)
                self._grpc_transport = None

    async def close(self) -> None:
        """Close the httpx client and release connections."""
        if self._native_transport:
            await self._native_transport.close()
            self._native_transport = None
        if self._grpc_transport:
            await self._grpc_transport.close()
            self._grpc_transport = None
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a single query. Returns unwrapped results.

        Uses persistent HTTP/1.1 keep-alive connection — no TCP handshake
        overhead after the first call.
        """
        if self._native_transport:
            return await self._native_transport.query(endpoint, payload)
        if self._grpc_transport:
            return await self._grpc_transport.query(endpoint, payload)

        if self._client is None:
            raise RuntimeError("HelixClient not initialized")

        try:
            resp = await self._client.post(
                f"/{endpoint}",
                content=json.dumps(payload),
            )

            if resp.status_code == 400:
                # Empty result or bad query — not fatal
                return []

            if resp.status_code >= 500:
                body = resp.text
                if "NoValue" in body or "NotFound" in body:
                    return []
                logger.warning("Helix %s returned %d: %s", endpoint, resp.status_code, body[:200])
                return []

            if resp.status_code != 200:
                return []

            raw = resp.json()
            if raw is None:
                return []

            # Helix returns either a dict or a list depending on query
            # unwrap_helix_results expects a list of dicts
            if isinstance(raw, dict):
                raw = [raw]

            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(raw)

        except httpx.ConnectError:
            logger.error("HelixDB connection lost for %s", endpoint)
            return []
        except Exception as exc:
            exc_name = type(exc).__name__
            if "NoValue" in exc_name or "NotFound" in exc_name:
                return []
            raise

    async def query_many(
        self,
        endpoint: str,
        payloads: list[dict],
        max_concurrent: int = 8,
    ) -> list[list[dict]]:
        """Execute multiple payloads against the same endpoint concurrently.

        Uses ``asyncio.Semaphore`` to limit concurrency. Returns results
        in the same order as payloads.

        Use for bulk operations like corpus loading (200 create_entity calls).
        """
        if self._native_transport:
            return await self._native_transport.query_many(endpoint, payloads, max_concurrent)
        if self._grpc_transport:
            return await self._grpc_transport.query_many(endpoint, payloads, max_concurrent)

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
        """Execute multiple (endpoint, payload) pairs concurrently.

        Automatically uses the single-round-trip ``/batch`` endpoint when
        available, falling back to individual concurrent requests otherwise.
        """
        if self._native_transport:
            return await self._native_transport.query_concurrent(queries, max_concurrent)
        if self._grpc_transport:
            return await self._grpc_transport.query_concurrent(queries, max_concurrent)

        if self._batch_available and len(queries) > 1:
            return await self.batch(queries)

        sem = asyncio.Semaphore(max_concurrent)

        async def _one(endpoint: str, payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(
            await asyncio.gather(*[_one(ep, pl) for ep, pl in queries])
        )

    async def _fallback_concurrent(
        self,
        queries: list[tuple[str, dict]],
    ) -> list[list[dict]]:
        """Individual concurrent queries — used when /batch is unavailable."""
        sem = asyncio.Semaphore(8)

        async def _one(endpoint: str, payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(
            await asyncio.gather(*[_one(ep, pl) for ep, pl in queries])
        )

    async def batch(
        self,
        queries: list[tuple[str, dict]],
    ) -> list[list[dict]]:
        """Execute multiple queries in a single HTTP round-trip via POST /batch.

        Each query is dispatched concurrently on the HelixDB worker pool,
        so read queries run in parallel across threads. Results are returned
        in the same order as the input queries.

        Falls back to individual concurrent requests if the batch endpoint is
        unavailable (e.g., older HelixDB version).
        """
        if self._native_transport:
            return await self._native_transport.batch(queries)
        if self._grpc_transport:
            return await self._grpc_transport.batch(queries)

        if self._client is None:
            raise RuntimeError("HelixClient not initialized")

        if not queries:
            return []

        payload = {
            "queries": [
                {"name": endpoint, "body": body}
                for endpoint, body in queries
            ]
        }

        try:
            resp = await self._client.post(
                "/batch",
                content=json.dumps(payload),
            )

            if resp.status_code == 404:
                # Batch endpoint not available — disable and fall back
                logger.info("Batch endpoint not available, falling back to concurrent queries")
                self._batch_available = False
                return await self._fallback_concurrent(queries)

            if resp.status_code != 200:
                logger.warning("Batch returned %d, falling back", resp.status_code)
                return await self._fallback_concurrent(queries)

            raw = resp.json()
            batch_results = raw.get("results", [])

            from engram.storage.helix import unwrap_helix_results

            results: list[list[dict]] = []
            for entry in batch_results:
                if entry.get("status") == 200:
                    body = entry.get("body")
                    if body is None:
                        results.append([])
                    elif isinstance(body, dict):
                        results.append(unwrap_helix_results([body]))
                    elif isinstance(body, list):
                        results.append(unwrap_helix_results(body))
                    else:
                        results.append([])
                else:
                    error = entry.get("error", "unknown error")
                    logger.warning("Batch sub-query failed: %s", error)
                    results.append([])

            return results

        except httpx.ConnectError:
            logger.error("HelixDB connection lost for batch")
            return [[] for _ in queries]
        except Exception:
            logger.warning("Batch request failed, falling back", exc_info=True)
            return await self._fallback_concurrent(queries)

    @property
    def is_connected(self) -> bool:
        if self._native_transport:
            return self._native_transport.is_connected
        if self._grpc_transport:
            return self._grpc_transport.is_connected
        return self._client is not None
