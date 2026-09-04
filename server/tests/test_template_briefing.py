"""Tests for template-based briefing in GraphManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager


def _make_manager() -> GraphManager:
    """Create a minimal GraphManager for briefing tests."""
    graph = MagicMock()
    activation = MagicMock()
    search = MagicMock()
    extractor = MagicMock()
    cfg = ActivationConfig()
    manager = GraphManager(
        graph,
        activation,
        search,
        extractor,
        cfg=cfg,
    )
    return manager


class TestTemplateBriefing:
    """Tests for _template_briefing."""

    def test_full_template(self):
        manager = _make_manager()
        context = (
            "## Identity Core\n\n"
            "- Konnor is a software engineer\n"
            "- Works at Anthropic\n\n"
            "## Project Context\n\n"
            "- Building Engram memory system\n"
            "- Uses Python and TypeScript\n\n"
            "## Recent Activity\n\n"
            "- Implemented extraction factory\n"
            "- Added template briefing\n"
        )
        result = manager._template_briefing(context, "default", "Engram")
        assert "Konnor" in result
        assert "Engram" in result
        assert "extraction factory" in result or "template briefing" in result

    def test_missing_identity_tier(self):
        manager = _make_manager()
        context = "## Project Context\n\n- Building Engram\n\n## Recent Activity\n\n- Fixed bugs\n"
        result = manager._template_briefing(context, "default", "Engram")
        assert "Engram" in result
        assert "Identity" not in result  # Should not mention missing tier

    def test_missing_all_tiers(self):
        manager = _make_manager()
        context = "## Active Memory Context\n\nNo memories loaded."
        result = manager._template_briefing(context, "default", None)
        # Falls back to structured context
        assert "No memories loaded" in result

    def test_empty_string(self):
        manager = _make_manager()
        result = manager._template_briefing("", "default", None)
        assert result == ""

    def test_caching(self):
        manager = _make_manager()
        context = "## Identity Core\n\n- User is a developer\n"
        result1 = manager._template_briefing(context, "default", "test")
        result2 = manager._template_briefing(context, "default", "test")
        assert result1 == result2

    def test_topic_hint_in_project_sentence(self):
        manager = _make_manager()
        context = "## Project Context\n\n- Using Python for backend\n"
        result = manager._template_briefing(context, "default", "Engram")
        assert "Engram" in result

    def test_only_recent_activity(self):
        manager = _make_manager()
        context = "## Recent Activity\n\n- Deployed v2.0\n- Fixed memory leak\n"
        result = manager._template_briefing(context, "default", None)
        assert "Deployed v2.0" in result or "memory leak" in result


