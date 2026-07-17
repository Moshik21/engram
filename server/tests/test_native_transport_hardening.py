"""Silent-inert hardening: NativeTransport failure contract.

Failures raise NativeQueryError (counted per endpoint); legitimate empty
signals stay []; engine error-JSON strings raise instead of masquerading as
empty results (the update_episode write-drop incident)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from engram.storage.helix import native_transport as nt


def _transport_with_engine(engine, *, timeout: float = 5.0):
    transport = nt.NativeTransport.__new__(nt.NativeTransport)
    transport._config = SimpleNamespace(query_timeout_seconds=timeout)
    transport._engine = engine
    from concurrent.futures import ThreadPoolExecutor

    transport._executor = ThreadPoolExecutor(max_workers=2)
    return transport


class TestQueryFailureContract:
    @pytest.mark.asyncio
    async def test_engine_exception_raises_and_counts(self):
        class _Engine:
            def query(self, endpoint, body):
                raise RuntimeError("lmdb page fault")

        transport = _transport_with_engine(_Engine())
        before = nt.get_query_failure_stats().get("q_test", {}).get("errors", 0)
        with pytest.raises(nt.NativeQueryError, match="lmdb page fault"):
            await transport.query("q_test", {})
        after = nt.get_query_failure_stats()["q_test"]["errors"]
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_error_json_string_raises(self):
        """Engine-level failures arrive as '{"error"...}' strings, not
        exceptions — they must never parse into an empty success."""

        class _Engine:
            def query(self, endpoint, body):
                return '{"error": "invalid parameter: null String"}'

        transport = _transport_with_engine(_Engine())
        with pytest.raises(nt.NativeQueryError, match="null String"):
            await transport.query("update_episode_full", {})

    @pytest.mark.asyncio
    async def test_legitimate_empty_signals_stay_empty(self):
        for signal in ("NoValue", "NotFound", "no entry point found for hnsw index"):

            class _Engine:
                def __init__(self, msg):
                    self._msg = msg

                def query(self, endpoint, body):
                    raise RuntimeError(self._msg)

            transport = _transport_with_engine(_Engine(signal))
            assert await transport.query("q_empty", {}) == []

    @pytest.mark.asyncio
    async def test_dimension_mismatch_tolerated_but_counted(self):
        class _Engine:
            def query(self, endpoint, body):
                raise RuntimeError("Invalid vector dimensions for search")

        transport = _transport_with_engine(_Engine())
        before = nt.get_query_failure_stats().get("q_dim", {}).get("dim_mismatch", 0)
        assert await transport.query("q_dim", {}) == []
        assert nt.get_query_failure_stats()["q_dim"]["dim_mismatch"] == before + 1

    @pytest.mark.asyncio
    async def test_timeout_raises_with_counter(self):
        class _Engine:
            def query(self, endpoint, body):
                time.sleep(2.0)
                return "[]"

        transport = _transport_with_engine(_Engine(), timeout=0.1)
        before = nt.get_query_failure_stats().get("q_slow", {}).get("timeouts", 0)
        with pytest.raises(nt.NativeQueryError, match="timed out"):
            await transport.query("q_slow", {})
        assert nt.get_query_failure_stats()["q_slow"]["timeouts"] == before + 1

    @pytest.mark.asyncio
    async def test_successful_query_unchanged(self):
        class _Engine:
            def query(self, endpoint, body):
                return '[{"name": "Python"}]'

        transport = _transport_with_engine(_Engine())
        rows = await transport.query("q_ok", {})
        assert rows and rows[0]["name"] == "Python"

    @pytest.mark.asyncio
    async def test_batch_item_error_counted_not_fatal(self):
        class _Engine:
            def batch(self, batch_input):
                return ['[{"ok": 1}]', '{"error": "bad item"}']

        transport = _transport_with_engine(_Engine())
        before = nt.get_query_failure_stats().get("q_b2", {}).get("batch_item_errors", 0)
        results = await transport.batch([("q_b1", {}), ("q_b2", {})])
        assert results[0] and results[0][0]["ok"] == 1
        assert results[1] == []
        assert nt.get_query_failure_stats()["q_b2"]["batch_item_errors"] == before + 1


class TestLookupTimeoutMarker:
    @pytest.mark.asyncio
    async def test_type_only_listing_timeout_is_explicit(self):
        from engram.retrieval.lookup import build_api_entity_search_surface

        class _Manager:
            async def search_entities(self, **kwargs):
                raise nt.NativeQueryError("find_entities_by_type", "timeout", timeout=True)

        payload = await build_api_entity_search_surface(
            _Manager(), group_id="g", entity_type="Decision"
        )
        assert payload["items"] == []
        assert payload["status"] == "timeout"

    @pytest.mark.asyncio
    async def test_non_query_errors_still_propagate(self):
        from engram.retrieval.lookup import build_api_entity_search_surface

        class _Manager:
            async def search_entities(self, **kwargs):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            await build_api_entity_search_surface(_Manager(), group_id="g")
