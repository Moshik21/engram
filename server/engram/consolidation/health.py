"""Graph health metrics for monitoring evidence pipeline quality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass
class GraphHealthMetrics:
    """Summary metrics for graph health monitoring."""

    entity_count: int = 0
    relationship_count: int = 0
    deferred_evidence_count: int = 0
    evidence_commit_rate: float = 0.0


async def compute_graph_health(
    graph_store: object,
    group_id: str = "default",
) -> GraphHealthMetrics:
    """Compute graph health metrics for a group.

    Args:
        graph_store: Store implementing GraphStore protocol
        group_id: The group to compute metrics for
    """
    store = cast(Any, graph_store)
    entity_count = await store.get_entity_count(group_id)

    db = getattr(store, "_db", None)
    if db is None:
        return GraphHealthMetrics(entity_count=entity_count)

    # Count relationships
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM relationships WHERE group_id = ?",
            (group_id,),
        )
        row = await cursor.fetchone()
        relationship_count = row[0] if row else 0
    except Exception:
        relationship_count = 0

    # Count unresolved evidence
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM episode_evidence "
            "WHERE group_id = ? AND status IN ('pending', 'deferred', 'approved')",
            (group_id,),
        )
        row = await cursor.fetchone()
        deferred_count = row[0] if row else 0
    except Exception:
        deferred_count = 0

    # Compute commit rate
    try:
        cursor = await db.execute(
            "SELECT "
            "  COUNT(CASE WHEN status = 'committed' THEN 1 END), "
            "  COUNT(*) "
            "FROM episode_evidence WHERE group_id = ?",
            (group_id,),
        )
        row = await cursor.fetchone()
        if row and row[1] > 0:
            commit_rate = row[0] / row[1]
        else:
            commit_rate = 0.0
    except Exception:
        commit_rate = 0.0

    return GraphHealthMetrics(
        entity_count=entity_count,
        relationship_count=relationship_count,
        deferred_evidence_count=deferred_count,
        evidence_commit_rate=round(commit_rate, 4),
    )
