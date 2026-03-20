"""Tests for recall middleware: unified recall on tool calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import ActivationConfig
from engram.mcp.server import _should_recall

# ─── _should_recall gate tests ─────────────────────────────────────


class TestShouldRecall:
    def test_returns_false_when_cfg_is_none(self):
        assert _should_recall("recall", None) is False

    def test_returns_false_for_non_recall_tool(self):
        cfg = ActivationConfig(auto_recall_on_tool_call=True)
        assert _should_recall("forget", cfg) is False
        assert _should_recall("bootstrap_project", cfg) is False
        assert _should_recall("get_runtime_state", cfg) is False

    def test_observe_respects_on_observe_flag(self):
        cfg_on = ActivationConfig(auto_recall_on_observe=True)
        cfg_off = ActivationConfig(auto_recall_on_observe=False)
        assert _should_recall("observe", cfg_on) is True
        assert _should_recall("observe", cfg_off) is False

    def test_remember_respects_on_remember_flag(self):
        cfg_on = ActivationConfig(auto_recall_on_remember=True)
        cfg_off = ActivationConfig(auto_recall_on_remember=False)
        assert _should_recall("remember", cfg_on) is True
        assert _should_recall("remember", cfg_off) is False

    def test_read_tools_respect_on_tool_call_flag(self):
        cfg_on = ActivationConfig(auto_recall_on_tool_call=True)
        cfg_off = ActivationConfig(auto_recall_on_tool_call=False)
        for tool in ("recall", "search_entities", "search_facts",
                      "get_context", "route_question", "search_artifacts"):
            assert _should_recall(tool, cfg_on) is True
            assert _should_recall(tool, cfg_off) is False


# ─── Middleware unit tests ──────────────────────────────────────────


class TestRecallMiddleware:
    """Unit tests for _recall_middleware()."""

    @pytest.mark.asyncio
    async def test_noop_when_flag_disabled(self):
        cfg = ActivationConfig(auto_recall_on_tool_call=False)
        response: dict = {"data": 1}
        with patch("engram.mcp.server._activation_cfg", cfg):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("test", response, tool_name="recall")
        assert "recalled_context" not in response

    @pytest.mark.asyncio
    async def test_noop_when_cfg_is_none(self):
        response: dict = {"data": 1}
        with patch("engram.mcp.server._activation_cfg", None):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("test", response, tool_name="recall")
        assert "recalled_context" not in response

    @pytest.mark.asyncio
    async def test_adds_recalled_context(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        recalled = {"source": "recall_lite", "entities": [{"name": "Alice"}]}
        manager = AsyncMock()
        manager._triggered_intentions = []
        response: dict = {"data": 1}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=recalled),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("Alice's project", response,
                                    tool_name="recall")
        assert response["recalled_context"] == recalled

    @pytest.mark.asyncio
    async def test_adds_session_context_first_call(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=True,
            notification_surfacing_enabled=False,
        )
        prime_result = {"context": "User briefing"}
        manager = AsyncMock()
        manager._triggered_intentions = []
        response: dict = {"data": 1}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=prime_result),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("hello", response, tool_name="recall")
        assert response["session_context"] == prime_result

    @pytest.mark.asyncio
    async def test_adds_triggered_intentions_and_clears(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        intention = MagicMock()
        intention.trigger_text = "meeting"
        intention.action_text = "remind about standup"
        intention.similarity = 0.95
        intention.matched_via = "embedding"
        intention.context = None
        intention.see_also = None
        manager = AsyncMock()
        manager._triggered_intentions = [intention]
        response: dict = {"data": 1}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("meeting notes", response,
                                    tool_name="recall")
        assert len(response["triggered_intentions"]) == 1
        assert response["triggered_intentions"][0]["trigger"] == "meeting"
        assert manager._triggered_intentions == []

    @pytest.mark.asyncio
    async def test_adds_memory_notifications(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=True,
        )
        notification = MagicMock()
        notification.notification_type = "new_connection"
        notification.title = "Found link"
        notification.body = "Alice -> Bob"
        notification.priority = 0.8
        ns = MagicMock()
        ns.get_for_mcp.return_value = [notification]
        manager = AsyncMock()
        manager._triggered_intentions = []
        response: dict = {"data": 1}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
            patch("engram.main._app_state",
                  {"notification_store": ns}),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("test", response, tool_name="recall")
        assert len(response["memory_notifications"]) == 1
        assert response["memory_notifications"][0]["title"] == "Found link"

    @pytest.mark.asyncio
    async def test_auto_observe_stores_long_content(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        manager = AsyncMock()
        manager._triggered_intentions = []
        manager.store_episode = AsyncMock(return_value="ep-123")
        response: dict = {}
        long_content = "What is the deployment strategy for the React dashboard project?"
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware(long_content, response,
                                    tool_name="route_question",
                                    auto_observe=True)
        manager.store_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_observe_skips_short_content(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        manager = AsyncMock()
        manager._triggered_intentions = []
        manager.store_episode = AsyncMock()
        response: dict = {}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("short", response,
                                    tool_name="route_question",
                                    auto_observe=True)
        manager.store_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_ingest_for_write_tools(self):
        """Write tools (observe/remember) do their own _ingest_live_turn."""
        cfg = ActivationConfig(
            auto_recall_on_observe=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        manager = AsyncMock()
        manager._triggered_intentions = []
        ingest_mock = AsyncMock()
        response: dict = {}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn", ingest_mock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("content", response,
                                    tool_name="observe")
        ingest_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_ingest_for_read_tools(self):
        cfg = ActivationConfig(
            auto_recall_on_tool_call=True,
            auto_recall_session_prime=False,
            notification_surfacing_enabled=False,
        )
        manager = AsyncMock()
        manager._triggered_intentions = []
        ingest_mock = AsyncMock()
        response: dict = {}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.mcp.server._get_manager", return_value=manager),
            patch("engram.mcp.server._session_prime",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._auto_recall_lite",
                  new_callable=AsyncMock, return_value=None),
            patch("engram.mcp.server._ingest_live_turn", ingest_mock),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("query", response, tool_name="recall")
        ingest_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_notification_fallback(self):
        """get_context surfaces notifications even when recall is disabled."""
        cfg = ActivationConfig(
            auto_recall_on_tool_call=False,
            notification_surfacing_enabled=True,
        )
        notification = MagicMock()
        notification.notification_type = "info"
        notification.title = "Test"
        notification.body = "body"
        notification.priority = 0.5
        ns = MagicMock()
        ns.get_for_mcp.return_value = [notification]
        response: dict = {}
        with (
            patch("engram.mcp.server._activation_cfg", cfg),
            patch("engram.main._app_state",
                  {"notification_store": ns}),
        ):
            from engram.mcp.server import _recall_middleware

            await _recall_middleware("", response, tool_name="get_context")
        assert len(response["memory_notifications"]) == 1


# ─── Config tests ──────────────────────────────────────────────────


class TestRecallConfig:
    def test_auto_recall_on_tool_call_default_false(self):
        cfg = ActivationConfig()
        assert cfg.auto_recall_on_tool_call is False

    def test_wave1_enables_tool_call(self):
        cfg = ActivationConfig(recall_profile="wave1")
        assert cfg.auto_recall_on_tool_call is True

    def test_wave2_enables_medium(self):
        cfg = ActivationConfig(recall_profile="wave2")
        assert cfg.auto_recall_level == "medium"

    def test_wave1_stays_lite(self):
        cfg = ActivationConfig(recall_profile="wave1")
        assert cfg.auto_recall_level == "lite"

    def test_off_does_not_enable(self):
        cfg = ActivationConfig(recall_profile="off")
        assert cfg.auto_recall_on_tool_call is False

    def test_rework_enables_tool_call(self):
        cfg = ActivationConfig(integration_profile="rework")
        assert cfg.auto_recall_on_tool_call is True


# ─── Tool integration tests ────────────────────────────────────────


class TestToolMiddlewareIntegration:
    @pytest.mark.asyncio
    async def test_recall_includes_recalled_context(self):
        recalled = {"source": "recall_lite", "entities": []}
        with (
            patch("engram.mcp.server._recall_middleware",
                  new_callable=AsyncMock,
                  side_effect=lambda content, resp, **kw: resp.update(
                      {"recalled_context": recalled})) as mw,
            patch("engram.mcp.server._get_manager") as gm,
            patch("engram.mcp.server._get_session") as gs,
            patch("engram.mcp.server._activation_cfg", ActivationConfig()),
            patch("engram.mcp.server._ingest_live_turn",
                  new_callable=AsyncMock),
            patch("engram.mcp.server._auto_recall_full",
                  new_callable=AsyncMock, return_value=None),
        ):
            manager = AsyncMock()
            manager.recall = AsyncMock(return_value=[])
            manager._triggered_intentions = []
            manager._last_near_misses = []
            manager._surprise_cache = None
            gm.return_value = manager
            session = MagicMock()
            session.auto_recall_primed = True
            gs.return_value = session

            from engram.mcp.server import recall

            json.loads(await recall(query="Engram project"))
            mw.assert_called_once()
            _, kwargs = mw.call_args
            assert kwargs["tool_name"] == "recall"

    @pytest.mark.asyncio
    async def test_search_entities_calls_middleware(self):
        with (
            patch("engram.mcp.server._recall_middleware",
                  new_callable=AsyncMock) as mw,
            patch("engram.mcp.server._get_manager") as gm,
        ):
            manager = AsyncMock()
            manager.search_entities = AsyncMock(return_value=[])
            gm.return_value = manager

            from engram.mcp.server import search_entities

            await search_entities(name="Alice")
            mw.assert_called_once()
            _, kwargs = mw.call_args
            assert kwargs["tool_name"] == "search_entities"

    @pytest.mark.asyncio
    async def test_route_question_auto_observes(self):
        with (
            patch("engram.mcp.server._recall_middleware",
                  new_callable=AsyncMock) as mw,
            patch("engram.mcp.server._get_manager") as gm,
            patch("engram.mcp.server._get_conv_context", return_value=None),
        ):
            manager = AsyncMock()
            manager.route_question = AsyncMock(
                return_value={"route": "remember"})
            manager._cfg = MagicMock()
            gm.return_value = manager

            from engram.mcp.server import route_question

            await route_question(question="How do I deploy?")
            _, kwargs = mw.call_args
            assert kwargs.get("auto_observe") is True
            assert kwargs["tool_name"] == "route_question"

    @pytest.mark.asyncio
    async def test_forget_does_not_call_middleware(self):
        with (
            patch("engram.mcp.server._recall_middleware",
                  new_callable=AsyncMock) as mw,
            patch("engram.mcp.server._get_manager") as gm,
        ):
            manager = AsyncMock()
            manager.forget_entity = AsyncMock(
                return_value={"status": "forgotten"})
            gm.return_value = manager

            from engram.mcp.server import forget

            result = json.loads(await forget(entity_name="OldEntity"))
            mw.assert_not_called()
            assert "recalled_context" not in result
