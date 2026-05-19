from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.immunity import ImmunityPhase


@pytest.mark.asyncio
async def test_immunity_dissolution_logic():
    cfg = ActivationConfig()
    cfg.immunity_enabled = True
    cfg.immunity_gravity_threshold = 0.5
    cfg.immunity_min_age_hours = 24
    phase = ImmunityPhase()

    mock_graph = MagicMock()
    from engram.utils.dates import utc_now
    now = utc_now()
    old_date = now - timedelta(days=2)

    # Mock some nodes with different gravity
    mock_graph.find_entities = AsyncMock(return_value=[
        {"id": "node_high", "name": "Important", "created_at": old_date},
        {"id": "node_low", "name": "Noise", "created_at": old_date}
    ])

    async def mock_get_neighbors(node_id):
        if node_id == "node_high":
            # node_id, weight
            return [("neighbor_1", 0.9), ("neighbor_2", 0.8)]
        return []

    mock_graph.get_active_neighbors_with_weights = AsyncMock(side_effect=mock_get_neighbors)
    mock_graph.delete_entity = AsyncMock()

    mock_search = MagicMock()
    mock_search.delete_entity = AsyncMock()

    result, audits = await phase.execute(
        group_id="default",
        graph_store=mock_graph,
        activation_store=MagicMock(),
        search_index=mock_search,
        cfg=cfg,
        cycle_id="cycle_1"
    )

    # Verify report contains the low gravity node
    dissolved_ids = [a.node_id for a in audits if a.decision == "pruned"]
    assert "node_low" in dissolved_ids
    assert "node_high" not in dissolved_ids
    assert mock_graph.delete_entity.call_count == 1
    assert mock_search.delete_entity.call_count == 1
