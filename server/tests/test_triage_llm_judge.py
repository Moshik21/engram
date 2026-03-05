"""Tests for the LLM triage judge feature."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.triage import TriagePhase, _llm_judge_score


@dataclass
class FakeEpisode:
    id: str
    content: str = ""
    status: str = "queued"
    source: str = "test"


def _make_graph_store(episodes: list[FakeEpisode] | None = None):
    store = AsyncMock()
    store.get_episodes_paginated = AsyncMock(return_value=(episodes or [], None))
    store.update_episode = AsyncMock()
    return store


def _make_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "triage_enabled": True,
        "triage_extract_ratio": 0.35,
        "triage_min_score": 0.2,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _mock_llm_response(extract: bool, score: float, reason: str, tags: list[str] | None = None):
    """Create a mock Anthropic response for triage judge."""
    content_block = MagicMock()
    content_block.text = json.dumps({
        "extract": extract,
        "score": score,
        "reason": reason,
        "tags": tags or [],
    })
    response = MagicMock()
    response.content = [content_block]
    return response


class TestLLMJudgeScore:
    """Tests for the _llm_judge_score function."""

    @patch("anthropic.Anthropic")
    def test_happy_path(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_llm_response(
            True, 0.85, "rich factual content", ["technical", "factual"],
        )

        result = _llm_judge_score("Alice works at Google", "claude-haiku-4-5-20251001")

        assert result["extract"] is True
        assert result["score"] == 0.85
        assert result["reason"] == "rich factual content"
        assert "technical" in result["tags"]

    @patch("anthropic.Anthropic")
    def test_low_score(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_llm_response(
            False, 0.1, "greeting only",
        )

        result = _llm_judge_score("hi", "claude-haiku-4-5-20251001")

        assert result["extract"] is False
        assert result["score"] == 0.1

    @patch("anthropic.Anthropic")
    def test_api_error_returns_fallback(self, mock_anthropic_cls):
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        result = _llm_judge_score("test content", "claude-haiku-4-5-20251001")

        assert result["extract"] is True
        assert result["score"] == 0.5
        assert result["reason"] == "llm_error_fallback"

    @patch("anthropic.Anthropic")
    def test_personal_content_high_score(self, mock_anthropic_cls):
        """Personal/emotional content should score high (fixes triage bias)."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_llm_response(
            True, 0.9, "personal health event", ["personal", "emotional"],
        )

        result = _llm_judge_score(
            "My mom was diagnosed with cancer last month",
            "claude-haiku-4-5-20251001",
        )

        assert result["score"] >= 0.8
        assert "personal" in result["tags"]

    @patch("anthropic.Anthropic")
    def test_uses_cached_system_prompt(self, mock_anthropic_cls):
        """Verify the system kwarg is a list with cache_control."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _mock_llm_response(
            True, 0.5, "test",
        )

        _llm_judge_score("test", "claude-haiku-4-5-20251001")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
class TestTriagePhaseWithLLMJudge:
    """Tests for TriagePhase when LLM judge is enabled."""

    @patch("engram.consolidation.phases.triage._llm_judge_score")
    async def test_llm_judge_enabled_uses_llm_score(self, mock_judge):
        mock_judge.return_value = {
            "extract": True, "score": 0.9, "reason": "test reason", "tags": ["factual"],
        }

        episodes = [FakeEpisode(id="ep_1", content="Alice works at Google")]
        graph = _make_graph_store(episodes)
        manager = MagicMock()
        manager.project_episode = AsyncMock()

        phase = TriagePhase(graph_manager=manager)
        cfg = _make_cfg(
            triage_llm_judge_enabled=True,
            triage_multi_signal_enabled=False,
            triage_extract_ratio=1.0,
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 1
        assert records[0].score == 0.9
        assert records[0].llm_reason == "test reason"
        assert records[0].llm_tags == ["factual"]
        mock_judge.assert_called_once()

    @patch("engram.consolidation.phases.triage._llm_judge_score")
    async def test_llm_judge_disabled_uses_heuristics(self, mock_judge):
        episodes = [FakeEpisode(id="ep_1", content="Alice works at Google")]
        graph = _make_graph_store(episodes)
        manager = MagicMock()
        manager.project_episode = AsyncMock()

        phase = TriagePhase(graph_manager=manager)
        cfg = _make_cfg(triage_llm_judge_enabled=False, triage_extract_ratio=1.0)

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # LLM judge should NOT be called
        mock_judge.assert_not_called()
        assert result.items_processed == 1
        # No LLM metadata
        assert records[0].llm_reason is None

    @patch("engram.consolidation.phases.triage._llm_judge_score")
    async def test_llm_judge_error_fallback(self, mock_judge):
        """Judge error returns fallback score, phase continues."""
        mock_judge.return_value = {
            "extract": True, "score": 0.5, "reason": "llm_error_fallback", "tags": [],
        }

        episodes = [FakeEpisode(id="ep_1", content="test content")]
        graph = _make_graph_store(episodes)
        manager = MagicMock()
        manager.project_episode = AsyncMock()

        phase = TriagePhase(graph_manager=manager)
        cfg = _make_cfg(
            triage_llm_judge_enabled=True,
            triage_multi_signal_enabled=False,
            triage_extract_ratio=1.0,
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 1
        assert records[0].score == 0.5
        assert records[0].llm_reason == "llm_error_fallback"
