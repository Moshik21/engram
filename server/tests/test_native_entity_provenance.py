from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pytest

from engram.config import HelixDBConfig
from engram.models import Entity
from engram.storage.helix.graph import HelixGraphStore


@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_entity_provenance_round_trips(tmp_path) -> None:
    """Native PyO3 Helix should preserve projected evidence lineage."""
    store = HelixGraphStore(
        HelixDBConfig(
            transport="native",
            data_dir=str(tmp_path / "native-entity-provenance"),
        )
    )
    await store.initialize()
    try:
        span_start = datetime(2026, 5, 14, 9, 0, tzinfo=timezone.utc)
        span_end = datetime(2026, 5, 14, 9, 15, tzinfo=timezone.utc)
        await store.create_entity(
            Entity(
                id="ent_native_provenance",
                name="Native Provenance",
                entity_type="Concept",
                group_id="native_brain",
                summary="Entity used to prove PyO3 preserves evidence lineage.",
                source_episode_ids=["ep_native_a", "ep_native_b"],
                evidence_count=2,
                evidence_span_start=span_start,
                evidence_span_end=span_end,
            )
        )

        loaded = await store.get_entity("ent_native_provenance", "native_brain")

        assert loaded is not None
        assert loaded.source_episode_ids == ["ep_native_a", "ep_native_b"]
        assert loaded.evidence_count == 2
        assert loaded.evidence_span_start == span_start
        assert loaded.evidence_span_end == span_end

        updated_start = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
        updated_end = datetime(2026, 5, 14, 10, 45, tzinfo=timezone.utc)
        await store.update_entity(
            "ent_native_provenance",
            {
                "source_episode_ids": ["ep_native_c"],
                "evidence_count": 3,
                "evidence_span_start": updated_start,
                "evidence_span_end": updated_end,
            },
            "native_brain",
        )

        updated = await store.get_entity("ent_native_provenance", "native_brain")

        assert updated is not None
        assert updated.source_episode_ids == ["ep_native_c"]
        assert updated.evidence_count == 3
        assert updated.evidence_span_start == updated_start
        assert updated.evidence_span_end == updated_end
    finally:
        await store.close()
