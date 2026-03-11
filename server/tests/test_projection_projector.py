"""Tests for typed projection bundle construction."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.extraction.extractor import ExtractionResult
from engram.extraction.models import ProjectedSpan, ProjectionPlan
from engram.extraction.projector import EpisodeProjector


@pytest.mark.asyncio
async def test_episode_projector_preserves_span_provenance():
    text = "Alex moved to Phoenix and now works on Engram."
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            entities=[
                {"name": "Alex", "entity_type": "Person"},
                {"name": "Phoenix", "entity_type": "Location"},
            ],
            relationships=[
                {"source": "Alex", "target": "Phoenix", "predicate": "LIVES_IN"},
            ],
        )
    )
    plan = ProjectionPlan(
        episode_id="ep_test",
        strategy="focused_span",
        spans=[
            ProjectedSpan(
                span_id="span_0",
                start_char=0,
                end_char=len(text),
                text=text,
                score=0.92,
                reasons=["cue_mentions"],
            )
        ],
        selected_text=text,
        selected_chars=len(text),
        total_chars=9001,
        was_truncated=True,
        warnings=["targeted_projection"],
    )

    bundle = await EpisodeProjector(extractor).project(plan)

    assert bundle.extractor_status == "ok"
    assert set(bundle.warnings) == {"targeted_projection", "planned_subset_only"}
    assert bundle.entities[0].source_span_ids == ["span_0"]
    assert bundle.claims[0].source_span_ids == ["span_0"]
