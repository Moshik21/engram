"""Native in-process transport for HelixDB via PyO3 extension.

Zero network overhead — HelixGraphEngine runs in-process with LMDB,
HNSW vectors, and BM25 all embedded. Matches SQLite's ~97ms latency.
"""

from __future__ import annotations

import asyncio
import atexit
import gc
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

try:
    import helix_native

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

_ENGINE_CACHE: dict[str, Any] = {}
_ENGINE_CACHE_LOCK = Lock()
_ATEXIT_REGISTERED = False

# Per-query failure counters (module-level: multiple transports share one
# process). Silent-inert hardening: failures must be observable, never
# metered as empty successes.
_QUERY_FAILURES: dict[str, dict[str, int]] = {}
_QUERY_FAILURES_LOCK = Lock()


def _count_failure(endpoint: str, kind: str) -> None:
    with _QUERY_FAILURES_LOCK:
        stats = _QUERY_FAILURES.setdefault(
            endpoint, {"errors": 0, "timeouts": 0, "dim_mismatch": 0, "batch_item_errors": 0}
        )
        stats[kind] = stats.get(kind, 0) + 1


def get_query_failure_stats() -> dict[str, dict[str, int]]:
    """Snapshot of per-query failure counters (errors/timeouts/dim_mismatch)."""
    with _QUERY_FAILURES_LOCK:
        return {k: dict(v) for k, v in _QUERY_FAILURES.items()}


class NativeQueryError(RuntimeError):
    """A native Helix query failed or timed out.

    Silent-inert hardening: the transport previously swallowed EVERY failure
    into `[]`, which upstream code metered as an empty success — the root of
    eight confirmed production bugs (phantom writes, stale counts, invisible
    listings). Failures now raise; callers that can legitimately tolerate one
    must catch THIS type explicitly and mark the degradation.
    """

    def __init__(
        self,
        endpoint: str,
        cause: str,
        *,
        timeout: bool = False,
        elapsed_ms: float | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.cause = cause
        self.timeout = timeout
        self.elapsed_ms = elapsed_ms
        kind = "timed out" if timeout else "failed"
        super().__init__(f"native query {endpoint!r} {kind}: {cause}")


class NativeTransport:
    """In-process HelixDB transport via PyO3.

    Mirrors the HelixClient HTTP interface for drop-in replacement.
    """

    def __init__(self, config: Any) -> None:
        if not HAS_NATIVE:
            raise ImportError(
                "helix_native is required for native transport. "
                "Install with: pip install engram[native]  "
                "Or build from source: make build-native"
            )
        self._config = config
        self._engine: Any = None
        self._executor: ThreadPoolExecutor | None = None

    # BM25 field-level indexing: only these fields are tokenized for BM25.
    # Labels not listed here index all fields (backward compatible).
    BM25_FIELD_FILTERS = {
        "Episode": ["content"],
        "Entity": ["name", "summary"],
        "EpisodeChunk": ["content"],
        "EpisodeCue": ["cue_text"],
    }

    async def initialize(self) -> None:
        """Create the in-process HelixGraphEngine.

        Redirects stdout to stderr during init because the Rust engine
        prints status messages that would corrupt MCP stdio transport.
        """
        global _ATEXIT_REGISTERED
        loop = asyncio.get_event_loop()
        data_dir = getattr(self._config, "data_dir", None) or None
        cache_key = _native_cache_key(data_dir)
        num_workers = getattr(self._config, "max_workers", 4)

        with _ENGINE_CACHE_LOCK:
            cached_engine = _ENGINE_CACHE.get(cache_key)
            if not _ATEXIT_REGISTERED:
                atexit.register(_close_cached_engines)
                _ATEXIT_REGISTERED = True
        if cached_engine is not None:
            self._engine = cached_engine
            self._executor = ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="helix-native",
            )
            logger.info(
                "NativeTransport reused cached engine (workers=%d)",
                num_workers,
            )
            return

        def _create_engine():
            import os
            import sys

            # Redirect stdout → stderr so Rust prints don't corrupt MCP stdio
            original_stdout_fd = os.dup(1)
            os.dup2(sys.stderr.fileno(), 1)
            try:
                return helix_native.HelixEngine(
                    data_dir=data_dir if data_dir else None,
                    num_workers=num_workers,
                    bm25_field_filters=self.BM25_FIELD_FILTERS,
                )
            finally:
                os.dup2(original_stdout_fd, 1)
                os.close(original_stdout_fd)

        engine = await loop.run_in_executor(None, _create_engine)
        with _ENGINE_CACHE_LOCK:
            self._engine = _ENGINE_CACHE.setdefault(cache_key, engine)
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="helix-native",
        )
        logger.info(
            "NativeTransport initialized (workers=%d, routes=%d)",
            num_workers,
            len(self._engine.list_routes()),
        )

    async def close(self) -> None:
        """Release this transport's query pool.

        The PyO3 engine keeps the LMDB environment process-owned. Closing it and
        then opening the same data dir again in one Python process currently
        returns ``Env already open``, so engines are cached until process exit.
        """
        executor = self._executor
        self._engine = None
        self._executor = None
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        gc.collect()

    async def compact(self, dest_dir: str) -> int:
        """Write a compacting copy of the LMDB env to ``dest_dir/data.mdb``.

        LMDB never returns freed pages to the OS, so a churned brain keeps
        paying RAM residency for pages it no longer stores. Graph, HNSW and
        BM25 share this one env, so a single copy reclaims all three. Callers
        must already hold exclusive access (shell down + brain flock).
        """
        if self._engine is None or self._executor is None:
            raise RuntimeError("NativeTransport not initialized")
        compact = getattr(self._engine, "compact", None)
        if compact is None:
            raise ImportError(
                "the installed helix_native has no compact(); "
                "rebuild the extension with 'make build-native'"
            )
        loop = asyncio.get_event_loop()
        return int(await loop.run_in_executor(self._executor, compact, dest_dir))

    async def health_check(self) -> None:
        """Verify the in-process engine is ready without scanning graph data."""
        if self._engine is None or self._executor is None:
            raise RuntimeError("NativeTransport not initialized")
        await asyncio.sleep(0)

    def _query_timeout_seconds(self) -> float:
        raw = getattr(self._config, "query_timeout_seconds", 20.0)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 20.0

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a single query via in-process engine.

        Raises NativeQueryError on failure or timeout. Legitimate
        empty-result signals from the engine (NoValue/NotFound/empty HNSW)
        return []; nothing else does.
        """
        if self._engine is None:
            raise RuntimeError("NativeTransport not initialized")

        loop = asyncio.get_event_loop()
        body_json = json.dumps(payload)
        timeout_seconds = self._query_timeout_seconds()
        started = time.monotonic()

        try:
            future = loop.run_in_executor(
                self._executor,
                self._engine.query,
                endpoint,
                body_json,
            )
            if timeout_seconds > 0:
                # NOTE: cancellation cannot stop the Rust worker thread — the
                # scan keeps running; the counter makes that cost observable.
                result_json = await asyncio.wait_for(future, timeout=timeout_seconds)
            else:
                result_json = await future

            if not result_json:
                return []

            # Engine-level failures arrive as an error JSON *string*, not an
            # exception — treating them as results silently dropped whole
            # writes (the update_episode error=None incident).
            if result_json.startswith('{"error"'):
                _count_failure(endpoint, "errors")
                raise NativeQueryError(
                    endpoint,
                    result_json[:300],
                    elapsed_ms=(time.monotonic() - started) * 1000,
                )

            raw = json.loads(result_json)
            if raw is None:
                return []
            if isinstance(raw, dict):
                raw = [raw]

            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(raw)

        except NativeQueryError:
            raise
        except TimeoutError:
            _count_failure(endpoint, "timeouts")
            elapsed_ms = (time.monotonic() - started) * 1000
            logger.warning(
                "Native query %s timed out after %.0fms (worker thread continues)",
                endpoint,
                elapsed_ms,
            )
            raise NativeQueryError(
                endpoint,
                f"timeout after {timeout_seconds:.1f}s",
                timeout=True,
                elapsed_ms=elapsed_ms,
            ) from None
        except Exception as exc:
            exc_str = str(exc)
            if (
                "NoValue" in exc_str
                or "NotFound" in exc_str
                or "no entry point found for hnsw index" in exc_str.lower()
            ):
                # silent-ok: engine signals a legitimately empty result set
                # via these exception strings — this IS the empty contract.
                return []
            if (
                "invalid vector dimensions" in exc_str.lower()
                or "mis-match in vector dimensions" in exc_str.lower()
            ):
                # silent-ok: mixed-dimension brains (legacy 1024-d vectors
                # alongside 768-d) tolerate per-vector mismatches on SEARCH;
                # counted so drift is observable.
                _count_failure(endpoint, "dim_mismatch")
                logger.debug(
                    "Native query %s skipped due to vector dimension mismatch",
                    endpoint,
                )
                return []
            _count_failure(endpoint, "errors")
            logger.error("Native query %s failed: %s", endpoint, exc)
            raise NativeQueryError(
                endpoint,
                exc_str[:300],
                elapsed_ms=(time.monotonic() - started) * 1000,
            ) from exc

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
        """Execute multiple queries. Uses batch for efficiency."""
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
        """Execute multiple queries via in-process batch."""
        if self._engine is None:
            raise RuntimeError("NativeTransport not initialized")

        if not queries:
            return []

        loop = asyncio.get_event_loop()
        batch_input = [(ep, json.dumps(body)) for ep, body in queries]

        try:
            result_jsons = await loop.run_in_executor(
                self._executor,
                self._engine.batch,
                batch_input,
            )

            from engram.storage.helix import unwrap_helix_results

            results: list[list[dict]] = []
            for (item_endpoint, _), result_json in zip(queries, result_jsons):
                if result_json and result_json.startswith('{"error"'):
                    # silent-ok: batch partial tolerance — one bad item must
                    # not nuke a search fan-out; counted + warned so it is
                    # never invisible, and the single-query path (used by the
                    # whole-batch fallback below) raises on the same failure.
                    _count_failure(item_endpoint, "batch_item_errors")
                    logger.warning(
                        "Native batch item %s failed: %s",
                        item_endpoint,
                        result_json[:200],
                    )
                    results.append([])
                    continue
                if not result_json:
                    results.append([])
                    continue
                raw = json.loads(result_json)
                if isinstance(raw, dict):
                    raw = [raw]
                results.append(unwrap_helix_results(raw) if raw else [])

            return results

        except Exception:
            # silent-ok: whole-batch failure degrades to individual queries,
            # each of which raises NativeQueryError on real failures.
            logger.warning("Native batch failed, falling back", exc_info=True)
            return await self._fallback_concurrent(queries)

    async def _fallback_concurrent(
        self,
        queries: list[tuple[str, dict]],
    ) -> list[list[dict]]:
        """Individual concurrent queries — fallback."""
        sem = asyncio.Semaphore(8)

        async def _one(endpoint: str, payload: dict) -> list[dict]:
            async with sem:
                return await self.query(endpoint, payload)

        return list(await asyncio.gather(*[_one(ep, pl) for ep, pl in queries]))

    @property
    def is_connected(self) -> bool:
        return self._engine is not None


def _native_cache_key(data_dir: str | None) -> str:
    if not data_dir:
        return "__helix_native_default__"
    return str(Path(data_dir).expanduser().resolve())


def _close_cached_engines() -> None:
    with _ENGINE_CACHE_LOCK:
        engines = list(_ENGINE_CACHE.values())
        _ENGINE_CACHE.clear()
    for engine in engines:
        try:
            engine.close()
        except Exception:  # silent-ok: atexit teardown; process is exiting
            logger.debug("Failed to close cached native Helix engine", exc_info=True)
