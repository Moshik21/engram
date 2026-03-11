"""Tests for the cross-encoder refinement tier (Tier 1)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.consolidation.scorers.cross_encoder import (
    _entity_description,
    cross_encoder_score,
    refine_infer_verdict,
    refine_merge_verdict,
)


def _entity(name, entity_type="Person", summary=None):
    return SimpleNamespace(
        id=f"ent_{name.lower()}",
        name=name,
        entity_type=entity_type,
        summary=summary,
    )


class TestEntityDescription:
    def test_name_only(self):
        e = _entity("Alice")
        assert _entity_description(e) == "Alice (Person)"

    def test_with_summary(self):
        e = _entity("Python", "Technology", "A programming language")
        desc = _entity_description(e)
        assert "Python" in desc
        assert "Technology" in desc
        assert "A programming language" in desc


class TestCrossEncoderScore:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_fastembed(self):
        with patch(
            "engram.consolidation.scorers.cross_encoder._get_cross_encoder",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await cross_encoder_score("hello", "world")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_float_with_mock_model(self):
        mock_model = MagicMock()
        # ms-marco returns raw logits; 2.0 → sigmoid ≈ 0.88
        mock_model.rerank.return_value = [2.0]

        with patch(
            "engram.consolidation.scorers.cross_encoder._get_cross_encoder",
            new_callable=AsyncMock,
            return_value=mock_model,
        ):
            result = await cross_encoder_score("query", "doc")
            assert result is not None
            expected = 1.0 / (1.0 + math.exp(-2.0))
            assert abs(result - expected) < 0.01

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        mock_model = MagicMock()
        mock_model.rerank.side_effect = RuntimeError("boom")

        with patch(
            "engram.consolidation.scorers.cross_encoder._get_cross_encoder",
            new_callable=AsyncMock,
            return_value=mock_model,
        ):
            result = await cross_encoder_score("a", "b")
            assert result is None


class TestRefineMergeVerdict:
    @pytest.mark.asyncio
    async def test_cross_encoder_promotes_to_merge(self):
        ea = _entity("React", "Technology", "A JavaScript UI library")
        eb = _entity("ReactJS", "Technology", "React JavaScript library")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=0.95,
        ):
            verdict, score = await refine_merge_verdict(ea, eb, 0.70)
            # 0.6*0.95 + 0.4*0.70 = 0.57 + 0.28 = 0.85 > 0.82
            assert verdict == "merge"
            assert score >= 0.82

    @pytest.mark.asyncio
    async def test_cross_encoder_keeps_separate(self):
        ea = _entity("Python", "Technology")
        eb = _entity("Python (snake)", "Animal")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=0.3,
        ):
            verdict, score = await refine_merge_verdict(ea, eb, 0.60)
            # 0.6*0.3 + 0.4*0.6 = 0.18 + 0.24 = 0.42 < 0.82
            assert verdict == "keep_separate"

    @pytest.mark.asyncio
    async def test_fallback_when_unavailable(self):
        ea = _entity("A")
        eb = _entity("B")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=None,
        ):
            verdict, score = await refine_merge_verdict(ea, eb, 0.70)
            assert verdict == "keep_separate"
            assert score == 0.70


class TestRefineInferVerdict:
    @pytest.mark.asyncio
    async def test_cross_encoder_approves(self):
        ea = _entity("Alice", "Person")
        eb = _entity("Acme Corp", "Organization")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=0.9,
        ):
            verdict, score = await refine_infer_verdict(
                ea,
                eb,
                "WORKS_AT",
                0.50,
            )
            # 0.5*0.9 + 0.5*0.5 = 0.45 + 0.25 = 0.70 > 0.65
            assert verdict == "approved"

    @pytest.mark.asyncio
    async def test_cross_encoder_rejects(self):
        ea = _entity("Alice", "Person")
        eb = _entity("Sushi", "Food")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=0.1,
        ):
            verdict, score = await refine_infer_verdict(
                ea,
                eb,
                "MENTIONED_WITH",
                0.45,
            )
            # 0.5*0.1 + 0.5*0.45 = 0.05 + 0.225 = 0.275 < 0.40
            assert verdict == "rejected"

    @pytest.mark.asyncio
    async def test_still_uncertain(self):
        ea = _entity("Alice", "Person")
        eb = _entity("Bob", "Person")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=0.5,
        ):
            verdict, score = await refine_infer_verdict(
                ea,
                eb,
                "MENTIONED_WITH",
                0.50,
            )
            # 0.5*0.5 + 0.5*0.5 = 0.50 — uncertain
            assert verdict == "uncertain"

    @pytest.mark.asyncio
    async def test_fallback_when_unavailable(self):
        ea = _entity("A")
        eb = _entity("B")

        with patch(
            "engram.consolidation.scorers.cross_encoder.cross_encoder_score",
            new_callable=AsyncMock,
            return_value=None,
        ):
            verdict, score = await refine_infer_verdict(
                ea,
                eb,
                "MENTIONED_WITH",
                0.50,
            )
            assert verdict == "uncertain"
            assert score == 0.50
