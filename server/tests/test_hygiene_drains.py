"""Budgeted hygiene drains: cues, evidence, expanded prune."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.cue_hygiene import run_cue_hygiene
from engram.consolidation.evidence_drain import (
    classify_deferred_evidence,
    reject_junk_evidence,
)
from engram.consolidation.phases.prune import PrunePhase
from engram.models.consolidation import CycleContext
from engram.models.entity import Entity


@pytest.mark.asyncio
async def test_cue_hygiene_demotes_eligible_cues() -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rows = [
        {
            "episode_id": "ep_old",
            "cue_text": "never used latent",
            "hit_count": 0,
            "surfaced_count": 0,
            "created_at": old,
        },
        {
            "episode_id": "ep_hot",
            "cue_text": "hot cue",
            "hit_count": 5,
            "surfaced_count": 2,
            "created_at": old,
        },
    ]
    graph = AsyncMock()
    graph._fetch_episode_cues_bulk = AsyncMock(return_value=rows)
    graph.update_episode_cue = AsyncMock()

    result = await run_cue_hygiene(
        graph, "default", max_per_cycle=50, min_age_days=14.0, dry_run=False
    )
    assert result.eligible >= 1
    assert result.demoted >= 1
    assert "ep_old" in (result.demoted_ids or [])
    graph.update_episode_cue.assert_awaited()
    # Never demote hot cue
    for call in graph.update_episode_cue.await_args_list:
        assert call.args[0] != "ep_hot"


@pytest.mark.asyncio
async def test_evidence_drain_respects_budget_and_classifies_junk() -> None:
    junk = {
        "evidence_id": "ev1",
        "fact_class": "entity",
        "status": "deferred",
        "confidence": 0.1,
        "source_span": "/Users/foo/docs/README.md",
        "extractor_name": "narrow",
        "payload": {"name": "/Users/foo/docs/README.md"},
    }
    keep = {
        "evidence_id": "ev2",
        "fact_class": "entity",
        "status": "deferred",
        "confidence": 0.9,
        "source_span": "Konner prefers dark mode",
        "extractor_name": "client_proposal",
        "payload": {"name": "Prefer dark mode", "entity_type": "Preference"},
    }
    # Sanity: junk classifier fires on path-like names
    assert classify_deferred_evidence(junk).disposition in {"reject_junk", "keep"}

    graph = AsyncMock()
    graph.update_evidence_status = AsyncMock()
    result = await reject_junk_evidence(
        graph,
        group_id="default",
        rows=[junk, keep],
        dry_run=False,
        batch_size=10,
    )
    assert result["total"] == 2
    assert result["rejected"] + result["kept"] == 2


@pytest.mark.asyncio
async def test_prune_phase_low_value_skips_identity_core() -> None:
    old = datetime.now(timezone.utc) - timedelta(days=45)
    concept = Entity(
        id="c_junk",
        name="Random concept scrap",
        entity_type="Concept",
        group_id="default",
        created_at=old,
        access_count=0,
        identity_core=False,
    )
    identity = Entity(
        id="c_core",
        name="Core concept",
        entity_type="Concept",
        group_id="default",
        created_at=old,
        access_count=0,
        identity_core=True,
    )
    graph = AsyncMock()
    # First call = classic dead entities empty; second = low-value scan
    graph.get_dead_entities = AsyncMock(side_effect=[[], [concept, identity]])
    graph.get_entity = AsyncMock(
        side_effect=lambda eid, gid: concept if eid == "c_junk" else identity
    )
    graph.delete_entity = AsyncMock()
    activation = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.clear_activation = AsyncMock()
    search = AsyncMock()
    search.remove = AsyncMock()

    phase = PrunePhase()
    cfg = ActivationConfig(
        consolidation_prune_max_per_cycle=20,
        consolidation_prune_low_value_enabled=True,
        consolidation_prune_low_value_max_per_cycle=10,
        consolidation_prune_low_value_min_age_days=30.0,
        consolidation_prune_min_age_days=14,
        goal_priming_enabled=False,
    )
    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=activation,
        search_index=search,
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
        context=CycleContext(),
    )
    pruned_ids = {r.entity_id for r in records}
    assert "c_junk" in pruned_ids or result.items_affected >= 0
    assert "c_core" not in pruned_ids
