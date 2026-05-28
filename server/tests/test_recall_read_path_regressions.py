from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.scorer import ScoredResult


@pytest.mark.asyncio
async def test_recall_top_entity_evidence_excludes_invalidated_relationships() -> None:
    expired = Relationship(
        id="rel_old_job",
        source_id="person_alex",
        target_id="company_oldco",
        predicate="WORKS_AT",
        valid_to=datetime(2025, 1, 1, tzinfo=timezone.utc),
        group_id="default",
    )
    current = Relationship(
        id="rel_current_job",
        source_id="person_alex",
        target_id="company_newco",
        predicate="WORKS_AT",
        group_id="default",
    )

    async def get_relationships(
        _entity_id: str,
        *,
        group_id: str,
        active_only: bool = True,
    ) -> list[Relationship]:
        assert group_id == "default"
        rels = [expired, current]
        return [rel for rel in rels if rel.valid_to is None] if active_only else rels

    graph = SimpleNamespace(
        get_entity=AsyncMock(
            return_value=Entity(
                id="person_alex",
                name="Alex",
                entity_type="Person",
                summary="Alex's current employment.",
                group_id="default",
            ),
        ),
        get_relationships=AsyncMock(side_effect=get_relationships),
    )
    materializer = RecallPrimaryResultMaterializer(
        graph_store=graph,
        result_builder=RecallResultBuilder(ActivationConfig()),
        cue_feedback_recorder=SimpleNamespace(record_cue_feedback=AsyncMock()),
        entity_access_recorder=SimpleNamespace(record_entity_access=AsyncMock()),
        interaction_recorder=SimpleNamespace(record_entity_interaction=Mock()),
        working_memory_updater=SimpleNamespace(add_result=Mock()),
    )

    result = await materializer.materialize(
        [
            ScoredResult(
                node_id="person_alex",
                score=0.95,
                semantic_similarity=0.95,
                activation=0.0,
                spreading=0.0,
                edge_proximity=0.0,
            )
        ],
        group_id="default",
        query="Where does Alex work now?",
        record_access=False,
        interaction_type="used",
        interaction_source="test",
        now=0.0,
        working_memory=None,
    )

    assert result.results[0]["relationships"] == [
        {
            "id": "rel_current_job",
            "predicate": "WORKS_AT",
            "source_id": "person_alex",
            "target_id": "company_newco",
            "weight": 1.0,
            "polarity": "positive",
        }
    ]
    graph.get_relationships.assert_awaited_once()
