"""Tests for AutoRecall (Wave 1): piggyback recall on observe/remember."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.mcp.server import (
    RecallCooldown,
    SessionState,
    _auto_recall,
    _extract_recall_query,
    _session_prime,
)

# ─── TestExtractRecallQuery ─────────────────────────────────────────


class TestExtractRecallQuery:
    def test_extracts_proper_nouns(self):
        result = _extract_recall_query("I'm working with John Smith on the React project")
        assert "John" in result
        assert "Smith" in result
        assert "React" in result

    def test_falls_back_to_first_sentence(self):
        text = "working on a migration to the new framework. more details later."
        result = _extract_recall_query(text)
        assert result == "working on a migration to the new framework"

    def test_returns_empty_for_short_content(self):
        assert _extract_recall_query("hi there") == ""
        assert _extract_recall_query("short") == ""

    def test_truncates_to_200_chars(self):
        long_text = "The " + " ".join(["BigWord"] * 50)
        result = _extract_recall_query(long_text)
        assert len(result) <= 200

    def test_empty_string(self):
        assert _extract_recall_query("") == ""


# ─── TestRecallCooldown ─────────────────────────────────────────────


class TestRecallCooldown:
    def test_allows_first_query(self):
        cd = RecallCooldown(max_per_minute=3, cooldown_seconds=60.0)
        assert not cd.is_throttled("test query here", time.time())

    def test_rate_limits_after_max_per_minute(self):
        cd = RecallCooldown(max_per_minute=2, cooldown_seconds=60.0)
        now = time.time()
        cd.record("query one here", now)
        cd.record("query two here", now)
        assert cd.is_throttled("query three here", now)

    def test_deduplicates_by_token_overlap(self):
        cd = RecallCooldown(max_per_minute=10, cooldown_seconds=60.0)
        now = time.time()
        cd.record("working with React and Next.js", now)
        # High overlap — should be throttled
        assert cd.is_throttled("working with React and Next.js migration", now + 1)

    def test_allows_after_cooldown_expires(self):
        cd = RecallCooldown(max_per_minute=2, cooldown_seconds=5.0)
        past = time.time() - 120.0  # >60s ago so rate limit window also expires
        cd.record("query one here", past)
        cd.record("query two here", past)
        # Rate limit entries are old (>60s), should allow now
        assert not cd.is_throttled("query three here", time.time())

    def test_allows_different_topics(self):
        cd = RecallCooldown(max_per_minute=10, cooldown_seconds=60.0)
        now = time.time()
        cd.record("working with React", now)
        # Completely different topic — low overlap
        assert not cd.is_throttled("database migration PostgreSQL", now + 1)

    def test_handles_short_queries(self):
        cd = RecallCooldown(max_per_minute=3, cooldown_seconds=60.0)
        # Short tokens (<=2 chars) are filtered out in tokenize
        assert not cd.is_throttled("a b c d e", time.time())


# ─── TestAutoRecall ─────────────────────────────────────────────────


@pytest.mark.asyncio
class TestAutoRecall:
    def _make_cfg(self, **overrides) -> ActivationConfig:
        defaults = {
            "auto_recall_enabled": True,
            "auto_recall_limit": 3,
            "auto_recall_min_score": 0.3,
            "auto_recall_cooldown_seconds": 60.0,
            "auto_recall_max_per_minute": 10,
        }
        defaults.update(overrides)
        return ActivationConfig(**defaults)

    def _make_result(
        self,
        name: str,
        score: float,
        result_type: str = "entity",
        entity_id: str | None = None,
    ) -> dict:
        return {
            "entity": {
                "id": entity_id or f"ent_{name.lower().replace(' ', '_')}",
                "name": name,
                "type": "Technology",
                "summary": "A test entity for " + name,
            },
            "score": score,
            "result_type": result_type,
            "relationships": [
                {"predicate": "USES", "source_id": "s1", "target_id": "t1"},
                {"predicate": "EXPERT_IN", "source_id": "s2", "target_id": "t2"},
            ],
        }

    async def test_returns_none_when_disabled(self):
        cfg = self._make_cfg(auto_recall_enabled=False)
        manager = AsyncMock()
        result = await _auto_recall("Some content about React", manager, cfg)
        assert result is None
        manager.recall.assert_not_called()

    async def test_returns_none_for_short_content(self):
        cfg = self._make_cfg()
        manager = AsyncMock()
        # _auto_recall returns None early for empty query (short content)
        # but it still needs _extract_recall_query to return "" first
        result = await _auto_recall("hi", manager, cfg)
        assert result is None

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_returns_recalled_context(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.recall.return_value = [
            self._make_result("React", 0.8),
            self._make_result("Next.js", 0.6),
        ]
        result = await _auto_recall("Working with React and Next.js framework", manager, cfg)
        assert result is not None
        assert result["source"] == "auto_recall"
        assert len(result["entities"]) == 2
        assert result["entities"][0]["name"] == "React"
        assert result["entities"][0]["type"] == "Technology"

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_filters_below_min_score(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg(auto_recall_min_score=0.5)
        manager = AsyncMock()
        manager.recall.return_value = [
            self._make_result("React", 0.8),
            self._make_result("LowScore", 0.2),
        ]
        result = await _auto_recall("Working with React and LowScore tools", manager, cfg)
        assert result is not None
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "React"

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_filters_episode_results(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.recall.return_value = [
            self._make_result("React", 0.8, result_type="episode"),
        ]
        result = await _auto_recall("Working with React framework here", manager, cfg)
        assert result is None

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_surfaces_cue_episode_results(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.recall.return_value = [
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_1",
                    "cue_text": "mentions: React",
                    "supporting_spans": ["Working with React framework here"],
                    "projection_state": "cue_only",
                },
                "score": 0.8,
            },
        ]
        result = await _auto_recall("Working with React framework here", manager, cfg)
        assert result is not None
        assert result["cue_episodes"][0]["episode_id"] == "ep_1"
        assert result["cue_episodes"][0]["projection_state"] == "cue_only"

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_cooldown_throttled(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = True
        cfg = self._make_cfg()
        manager = AsyncMock()
        result = await _auto_recall("Working with React framework here", manager, cfg)
        assert result is None
        manager.recall.assert_not_called()

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session")
    async def test_skips_when_explicit_recall_recent(self, mock_session, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        mock_session.last_recall_time = time.time() - 10  # 10s ago, within 30s window
        cfg = self._make_cfg()
        manager = AsyncMock()
        result = await _auto_recall("Working with React framework here", manager, cfg)
        assert result is None
        manager.recall.assert_not_called()

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_compact_format_truncates_summary(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg()
        manager = AsyncMock()
        long_summary = "A" * 200
        manager.recall.return_value = [
            {
                "entity": {"name": "Test", "type": "Concept", "summary": long_summary},
                "score": 0.9,
                "result_type": "entity",
                "relationships": [],
            }
        ]
        result = await _auto_recall("Working with Test concept here today", manager, cfg)
        assert result is not None
        assert len(result["entities"][0]["summary"]) <= 100

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_top_facts_limited_to_3(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.recall.return_value = [
            {
                "entity": {"name": "Test", "type": "Concept", "summary": "A test"},
                "score": 0.9,
                "result_type": "entity",
                "relationships": [
                    {"predicate": "REL1", "source_id": "s", "target_id": "t"},
                    {"predicate": "REL2", "source_id": "s", "target_id": "t"},
                    {"predicate": "REL3", "source_id": "s", "target_id": "t"},
                    {"predicate": "REL4", "source_id": "s", "target_id": "t"},
                    {"predicate": "REL5", "source_id": "s", "target_id": "t"},
                ],
            }
        ]
        result = await _auto_recall("Working with Test concept in the project", manager, cfg)
        assert result is not None
        assert len(result["entities"][0]["top_facts"]) == 3

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_returns_packets_when_enabled(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg(recall_packets_enabled=True)
        manager = AsyncMock()
        manager.resolve_entity_name = AsyncMock(side_effect=lambda entity_id, group_id: entity_id)
        manager.recall.return_value = [
            self._make_result("React", 0.8),
        ]

        result = await _auto_recall("Working with React framework here", manager, cfg)

        assert result is not None
        assert result["packets"]
        assert result["packets"][0]["packet_type"] == "fact_packet"

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_need_analyzer_skips_low_value_turns(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg(recall_need_analyzer_enabled=True)
        manager = AsyncMock()
        manager._conv_context = None
        result = await _auto_recall("thanks", manager, cfg)
        assert result is None
        manager.recall.assert_not_called()

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_need_analyzer_recalls_project_follow_up(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg(recall_need_analyzer_enabled=True)
        manager = AsyncMock()
        manager._conv_context = None
        manager.recall.return_value = [self._make_result("React", 0.8)]

        result = await _auto_recall("How's the React migration going?", manager, cfg)

        assert result is not None
        call_kwargs = manager.recall.call_args.kwargs
        assert call_kwargs["query"]
        assert call_kwargs["record_access"] is True

    @patch("engram.mcp.server._recall_cooldown")
    @patch("engram.mcp.server._session", new=SessionState(last_recall_time=0.0))
    async def test_usage_feedback_marks_auto_recall_as_surfaced(self, mock_cooldown):
        mock_cooldown.is_throttled.return_value = False
        cfg = self._make_cfg(
            recall_need_analyzer_enabled=True,
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
        )
        manager = AsyncMock()
        manager._conv_context = None
        manager.recall.return_value = [self._make_result("React", 0.8)]

        result = await _auto_recall("How's the React migration going?", manager, cfg)

        assert result is not None
        call_kwargs = manager.recall.call_args.kwargs
        assert call_kwargs["record_access"] is False
        assert call_kwargs["interaction_type"] == "surfaced"
        assert call_kwargs["interaction_source"] == "auto_recall"


# ─── TestSessionPrime ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestSessionPrime:
    def _make_cfg(self, **overrides) -> ActivationConfig:
        defaults = {"auto_recall_session_prime": True, "auto_recall_session_prime_max_tokens": 500}
        defaults.update(overrides)
        return ActivationConfig(**defaults)

    @patch("engram.mcp.server._session", new=SessionState(auto_recall_primed=False))
    async def test_primes_on_first_call(self):
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.get_context.return_value = {"context": "User works on React", "entity_count": 3}
        result = await _session_prime("Working on React", manager, cfg)
        assert result is not None
        assert result["context"] == "User works on React"
        manager.get_context.assert_called_once()

    @patch("engram.mcp.server._session", new=SessionState(auto_recall_primed=True))
    async def test_returns_none_on_second_call(self):
        cfg = self._make_cfg()
        manager = AsyncMock()
        result = await _session_prime("Working on React", manager, cfg)
        assert result is None
        manager.get_context.assert_not_called()

    @patch("engram.mcp.server._session", new=SessionState(auto_recall_primed=False))
    async def test_returns_none_when_disabled(self):
        cfg = self._make_cfg(auto_recall_session_prime=False)
        manager = AsyncMock()
        result = await _session_prime("Working on React", manager, cfg)
        assert result is None

    @patch("engram.mcp.server._session", new=SessionState(auto_recall_primed=False))
    async def test_extracts_topic_from_content(self):
        cfg = self._make_cfg()
        manager = AsyncMock()
        manager.get_context.return_value = {"context": "briefing"}
        await _session_prime("Working with React and Next.js", manager, cfg)
        call_kwargs = manager.get_context.call_args[1]
        # Should extract a topic hint from content
        assert call_kwargs["topic_hint"] is not None
        assert call_kwargs["format"] == "structured"


# ─── TestObserveWithAutoRecall ──────────────────────────────────────


@pytest.mark.asyncio
class TestObserveWithAutoRecall:
    """Integration-style tests for observe() with auto-recall."""

    async def test_observe_returns_recalled_context_when_enabled(self):
        """observe() includes recalled_context in response when auto-recall fires."""
        from engram.mcp import server

        cfg = ActivationConfig(
            auto_recall_enabled=True,
            auto_recall_on_observe=True,
            auto_recall_session_prime=False,
        )
        mock_manager = AsyncMock()
        mock_manager.store_episode.return_value = "ep-123"
        mock_manager.recall.return_value = [
            {
                "entity": {"name": "React", "type": "Technology", "summary": "UI lib"},
                "score": 0.8,
                "result_type": "entity",
                "relationships": [],
            }
        ]
        session = SessionState(last_recall_time=0.0, auto_recall_primed=True)
        cooldown = RecallCooldown(max_per_minute=10, cooldown_seconds=60.0)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_recall_cooldown", cooldown),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.observe("Working on a React migration to Next.js project")
            result = json.loads(raw)

        assert result["status"] == "stored"
        assert "recalled_context" in result
        assert result["recalled_context"]["entities"][0]["name"] == "React"

    async def test_observe_returns_session_context_on_first_call(self):
        """observe() includes session_context on first call (priming)."""
        from engram.mcp import server

        cfg = ActivationConfig(
            auto_recall_enabled=True,
            auto_recall_on_observe=True,
            auto_recall_session_prime=True,
        )
        mock_manager = AsyncMock()
        mock_manager.store_episode.return_value = "ep-123"
        mock_manager.get_context.return_value = {
            "context": "User is a developer",
            "entity_count": 5,
        }
        mock_manager.recall.return_value = []
        session = SessionState(auto_recall_primed=False, last_recall_time=0.0)
        cooldown = RecallCooldown(max_per_minute=10, cooldown_seconds=60.0)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_recall_cooldown", cooldown),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.observe("Working on a React migration to Next.js project")
            result = json.loads(raw)

        assert result["status"] == "stored"
        assert "session_context" in result
        assert result["session_context"]["context"] == "User is a developer"

    async def test_observe_clean_when_disabled(self):
        """observe() returns clean JSON when auto-recall is disabled (backward compat)."""
        from engram.mcp import server

        cfg = ActivationConfig(auto_recall_enabled=False, auto_recall_on_observe=False)
        mock_manager = AsyncMock()
        mock_manager.store_episode.return_value = "ep-123"
        session = SessionState()

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.observe("some content here for testing")
            result = json.loads(raw)

        assert result["status"] == "stored"
        assert "recalled_context" not in result
        assert "session_context" not in result


# ─── TestRememberWithAutoRecall ─────────────────────────────────────


@pytest.mark.asyncio
class TestRememberWithAutoRecall:
    """Integration-style tests for remember() with auto-recall."""

    async def test_remember_returns_recalled_context_when_enabled(self):
        from engram.mcp import server

        cfg = ActivationConfig(
            auto_recall_enabled=True,
            auto_recall_on_remember=True,
            auto_recall_session_prime=False,
        )
        mock_manager = AsyncMock()
        mock_manager.ingest_episode.return_value = "ep-456"
        mock_manager.recall.return_value = [
            {
                "entity": {"name": "Python", "type": "Technology", "summary": "Language"},
                "score": 0.7,
                "result_type": "entity",
                "relationships": [],
            }
        ]
        session = SessionState(last_recall_time=0.0, auto_recall_primed=True)
        cooldown = RecallCooldown(max_per_minute=10, cooldown_seconds=60.0)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_recall_cooldown", cooldown),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.remember("User prefers Python for backend development work")
            result = json.loads(raw)

        assert result["status"] == "stored"
        assert "recalled_context" in result
        assert result["recalled_context"]["entities"][0]["name"] == "Python"

    async def test_remember_clean_when_disabled(self):
        from engram.mcp import server

        cfg = ActivationConfig(auto_recall_enabled=False, auto_recall_on_remember=False)
        mock_manager = AsyncMock()
        mock_manager.ingest_episode.return_value = "ep-456"
        session = SessionState()

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.remember("User prefers Python for backend development work")
            result = json.loads(raw)

        assert result["status"] == "stored"
        assert "recalled_context" not in result


# ─── TestRecallSetsLastTime ─────────────────────────────────────────


@pytest.mark.asyncio
class TestRecallSetsLastTime:
    async def test_explicit_recall_sets_last_recall_time(self):
        from engram.mcp import server

        mock_manager = AsyncMock()
        mock_manager.recall.return_value = []
        mock_manager._surprise_cache = None
        mock_manager._last_near_misses = None
        session = SessionState(last_recall_time=0.0, auto_recall_primed=False)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_group_id", "default"),
        ):
            before = time.time()
            await server.recall("test query")
            after = time.time()

        assert session.last_recall_time >= before
        assert session.last_recall_time <= after

    async def test_explicit_recall_sets_primed(self):
        from engram.mcp import server

        mock_manager = AsyncMock()
        mock_manager.recall.return_value = []
        mock_manager._surprise_cache = None
        mock_manager._last_near_misses = None
        session = SessionState(last_recall_time=0.0, auto_recall_primed=False)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_group_id", "default"),
        ):
            await server.recall("test query")

        assert session.auto_recall_primed is True

    async def test_explicit_recall_returns_packets(self):
        from engram.mcp import server

        mock_manager = AsyncMock()
        mock_manager.recall.return_value = [
            {
                "entity": {
                    "id": "ent_1",
                    "name": "React",
                    "type": "Technology",
                    "summary": "UI lib",
                },
                "score": 0.8,
                "score_breakdown": {"semantic": 0.7, "activation": 0.1, "edge_proximity": 0.0},
                "relationships": [],
            }
        ]
        mock_manager.resolve_entity_name = AsyncMock(return_value="React")
        mock_manager._activation.get_activation.return_value = type(
            "ActivationState",
            (),
            {"access_count": 1},
        )()
        mock_manager._surprise_cache = None
        mock_manager._last_near_misses = None
        session = SessionState(last_recall_time=0.0, auto_recall_primed=False)
        cfg = ActivationConfig(recall_packets_enabled=True)

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_group_id", "default"),
        ):
            raw = await server.recall("React")
            payload = json.loads(raw)

        assert payload["packets"]
        assert payload["packets"][0]["packet_type"] == "fact_packet"

    async def test_explicit_recall_marks_used_interaction(self):
        from engram.mcp import server

        mock_manager = AsyncMock()
        mock_manager.recall.return_value = []
        mock_manager._surprise_cache = None
        mock_manager._last_near_misses = None
        session = SessionState(last_recall_time=0.0, auto_recall_primed=False)
        cfg = ActivationConfig(
            recall_telemetry_enabled=True,
            recall_usage_feedback_enabled=True,
        )

        with (
            patch.object(server, "_manager", mock_manager),
            patch.object(server, "_session", session),
            patch.object(server, "_activation_cfg", cfg),
            patch.object(server, "_group_id", "default"),
        ):
            await server.recall("React")

        call_kwargs = mock_manager.recall.await_args.kwargs
        assert call_kwargs["interaction_type"] == "used"
        assert call_kwargs["interaction_source"] == "mcp_recall"
