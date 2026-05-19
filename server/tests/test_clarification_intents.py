from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.adjudication_service import EvidenceAdjudicationService


@pytest.mark.asyncio
async def test_create_clarification_intents():
    mock_graph = MagicMock()
    mock_graph.create_entity = AsyncMock()

    cfg = ActivationConfig()
    cfg.active_adjudication_enabled = True

    service = EvidenceAdjudicationService(
        graph_store=mock_graph,
        search_index=MagicMock(),
        cfg=cfg,
        evidence_bridge=MagicMock(),
        apply_engine=MagicMock(),
        apply_bootstrap_part_of_edges=MagicMock(),
        index_entity_with_structure=MagicMock(),
        invalidate_briefing_cache=MagicMock(),
    )

    requests = [
        MagicMock(
            selected_text="Apple",
            request_reason="semantic_clash",
            request_id="adj1",
            episode_id="ep1",
            group_id="default",
            ambiguity_tags=["tag1"],
        ),
        MagicMock(
            selected_text="SF",
            request_reason="reference_error",
            request_id="adj2",
            episode_id="ep1",
            group_id="default",
            ambiguity_tags=["tag2"],
        ),
    ]

    await service.create_clarification_intents(requests)

    # Verify two intents were created
    assert mock_graph.create_entity.call_count == 2

    # Verify the second call was for SF
    args, kwargs = mock_graph.create_entity.call_args_list[1]
    intent = args[0]
    assert "SF" in intent.summary
    assert intent.entity_type == "ClarificationIntent"
