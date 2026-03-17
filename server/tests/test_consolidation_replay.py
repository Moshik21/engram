"""Tests for the EpisodeReplayPhase."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.replay import EpisodeReplayPhase
from engram.extraction.extractor import ExtractionResult
from engram.models.consolidation import CycleContext
from engram.models.entity import Entity


def _make_episode(
    episode_id: str = "ep_test1",
    content: str = "Alice met Bob at Acme Corp.",
    status: str = "completed",
    created_at: datetime | None = None,
    group_id: str = "test",
) -> MagicMock:
    ep = MagicMock()
    ep.id = episode_id
    ep.content = content
    ep.status = MagicMock()
    ep.status.value = status
    ep.created_at = created_at or (datetime.utcnow() - timedelta(hours=3))
    ep.group_id = group_id
    return ep


def _make_entity(
    entity_id: str = "ent_alice",
    name: str = "Alice",
    entity_type: str = "person",
) -> Entity:
    return Entity(
        id=entity_id,
        name=name,
        entity_type=entity_type,
        summary=None,
        group_id="test",
    )


def _make_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "consolidation_replay_enabled": True,
        "consolidation_replay_max_per_cycle": 50,
        "consolidation_replay_window_hours": 24.0,
        "consolidation_replay_min_age_hours": 0.0,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _make_extractor(entities=None, relationships=None) -> AsyncMock:
    extractor = AsyncMock()
    extractor.extract = AsyncMock(
        return_value=ExtractionResult(
            entities=entities or [],
            relationships=relationships or [],
        ),
    )
    return extractor


class TestEpisodeReplayPhase:
    def test_phase_name(self):
        phase = EpisodeReplayPhase()
        assert phase.name == "replay"

    @pytest.mark.asyncio
    async def test_disabled_skips(self):
        cfg = _make_cfg(consolidation_replay_enabled=False)
        phase = EpisodeReplayPhase(extractor=AsyncMock())
        result, records = await phase.execute(
            group_id="test",
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_no_extractor_skips(self):
        cfg = _make_cfg()
        phase = EpisodeReplayPhase(extractor=None)
        result, records = await phase.execute(
            group_id="test",
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_no_eligible_episodes(self):
        cfg = _make_cfg()
        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=_make_extractor())
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.items_processed == 0
        assert records == []

    @pytest.mark.asyncio
    async def test_episode_outside_window_skipped(self):
        cfg = _make_cfg(consolidation_replay_window_hours=24.0)
        old_ep = _make_episode(
            created_at=datetime.utcnow() - timedelta(hours=48),
        )
        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[old_ep])

        phase = EpisodeReplayPhase(extractor=_make_extractor())
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_episode_too_young_skipped(self):
        cfg = _make_cfg(consolidation_replay_min_age_hours=2.0)
        young_ep = _make_episode(
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[young_ep])

        phase = EpisodeReplayPhase(extractor=_make_extractor())
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_discovers_new_entity(self):
        cfg = _make_cfg()
        ep = _make_episode()
        extractor = _make_extractor(
            entities=[{"name": "NewPerson", "entity_type": "person"}],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.create_entity = AsyncMock()
        graph_store.link_episode_entity = AsyncMock()

        activation_store = AsyncMock()
        search_index = AsyncMock()
        ctx = CycleContext()
        # Manual trigger — replay always runs on full cycles
        ctx.trigger = "manual"

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        assert result.items_processed == 1
        assert result.items_affected >= 1
        assert records[0].new_entities_found == 1
        graph_store.create_entity.assert_called_once()
        graph_store.link_episode_entity.assert_called_once()
        activation_store.record_access.assert_called_once()
        search_index.index_entity.assert_called_once()
        assert len(ctx.replay_new_entity_ids) == 1
        assert len(ctx.affected_entity_ids) >= 1

    @pytest.mark.asyncio
    async def test_skips_existing_entity_already_linked(self):
        cfg = _make_cfg()
        ep = _make_episode()
        alice = _make_entity()
        extractor = _make_extractor(
            entities=[{"name": "Alice", "entity_type": "person"}],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice])
        graph_store.get_episode_entities = AsyncMock(return_value=["ent_alice"])
        graph_store.create_entity = AsyncMock()
        graph_store.link_episode_entity = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert records[0].new_entities_found == 0
        graph_store.create_entity.assert_not_called()
        graph_store.link_episode_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_discovers_new_relationship(self):
        cfg = _make_cfg()
        ep = _make_episode()
        alice = _make_entity("ent_alice", "Alice", "person")
        bob = _make_entity("ent_bob", "Bob", "person")

        extractor = _make_extractor(
            entities=[
                {"name": "Alice", "entity_type": "person"},
                {"name": "Bob", "entity_type": "person"},
            ],
            relationships=[
                {"source": "Alice", "target": "Bob", "predicate": "KNOWS"},
            ],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice, bob])
        graph_store.get_episode_entities = AsyncMock(
            return_value=["ent_alice", "ent_bob"],
        )
        graph_store.find_conflicting_relationships = AsyncMock(return_value=[])
        graph_store.find_existing_relationship = AsyncMock(return_value=None)
        graph_store.create_relationship = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert records[0].new_relationships_found == 1
        graph_store.create_relationship.assert_called_once()
        created_rel = graph_store.create_relationship.call_args[0][0]
        assert created_rel.confidence == 1.0
        assert created_rel.source_episode.startswith("replay:")

    @pytest.mark.asyncio
    async def test_skips_existing_relationship(self):
        cfg = _make_cfg()
        ep = _make_episode()
        alice = _make_entity("ent_alice", "Alice", "person")
        bob = _make_entity("ent_bob", "Bob", "person")

        extractor = _make_extractor(
            entities=[
                {"name": "Alice", "entity_type": "person"},
                {"name": "Bob", "entity_type": "person"},
            ],
            relationships=[
                {"source": "Alice", "target": "Bob", "predicate": "KNOWS"},
            ],
        )

        existing_rel = MagicMock()
        existing_rel.target_id = "ent_bob"

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice, bob])
        graph_store.get_episode_entities = AsyncMock(
            return_value=["ent_alice", "ent_bob"],
        )
        graph_store.find_conflicting_relationships = AsyncMock(return_value=[])
        graph_store.find_existing_relationship = AsyncMock(return_value=existing_rel)
        graph_store.create_relationship = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert records[0].new_relationships_found == 0
        graph_store.create_relationship.assert_not_called()

    @pytest.mark.asyncio
    async def test_replay_applies_negation_semantics(self):
        cfg = _make_cfg()
        ep = _make_episode()
        alice = _make_entity("ent_alice", "Alice", "person")
        bob = _make_entity("ent_bob", "Bob", "person")

        extractor = _make_extractor(
            entities=[
                {"name": "Alice", "entity_type": "person"},
                {"name": "Bob", "entity_type": "person"},
            ],
            relationships=[
                {
                    "source": "Alice",
                    "target": "Bob",
                    "predicate": "KNOWS",
                    "polarity": "negative",
                },
            ],
        )

        existing_rel = MagicMock()
        existing_rel.id = "rel_existing"
        existing_rel.target_id = "ent_bob"
        existing_rel.source_id = "ent_alice"
        existing_rel.predicate = "KNOWS"

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice, bob])
        graph_store.get_episode_entities = AsyncMock(
            return_value=["ent_alice", "ent_bob"],
        )
        graph_store.get_relationships = AsyncMock(return_value=[existing_rel])
        graph_store.create_relationship = AsyncMock()
        graph_store.invalidate_relationship = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 1
        assert records[0].new_relationships_found == 1
        graph_store.invalidate_relationship.assert_called_once()
        graph_store.create_relationship.assert_called_once()
        created_rel = graph_store.create_relationship.call_args[0][0]
        assert created_rel.polarity == "negative"

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self):
        cfg = _make_cfg()
        ep = _make_episode()
        extractor = _make_extractor(
            entities=[{"name": "NewPerson", "entity_type": "person"}],
            relationships=[
                {"source": "NewPerson", "target": "Nobody", "predicate": "KNOWS"},
            ],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        assert records[0].new_entities_found == 1
        graph_store.create_entity.assert_not_called()
        graph_store.create_relationship.assert_not_called()
        graph_store.link_episode_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_per_cycle_respected(self):
        cfg = _make_cfg(consolidation_replay_max_per_cycle=2)
        episodes = [_make_episode(episode_id=f"ep_{i}") for i in range(5)]

        extractor = _make_extractor()
        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=episodes)
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 2
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_scans_past_ineligible_recent_episodes(self):
        cfg = _make_cfg(
            consolidation_replay_max_per_cycle=1,
            consolidation_replay_min_age_hours=2.0,
        )
        recent_ep = _make_episode(
            episode_id="ep_recent",
            created_at=datetime.utcnow() - timedelta(minutes=20),
        )
        eligible_ep = _make_episode(
            episode_id="ep_eligible",
            created_at=datetime.utcnow() - timedelta(hours=3),
        )

        async def _get_episodes(*, group_id=None, limit=50, offset=0):
            batches = {
                0: [recent_ep],
                1: [eligible_ep],
            }
            return batches.get(offset, [])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(side_effect=_get_episodes)
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=_make_extractor())
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 1
        assert len(records) == 1
        assert records[0].episode_id == "ep_eligible"
        assert graph_store.get_episodes.call_count >= 2

    @pytest.mark.asyncio
    async def test_extraction_failure_non_fatal(self):
        cfg = _make_cfg()
        ep1 = _make_episode(episode_id="ep_fail")
        ep2 = _make_episode(episode_id="ep_ok")

        extractor = AsyncMock()
        extractor.extract = AsyncMock(
            side_effect=[
                RuntimeError("API error"),
                ExtractionResult(entities=[], relationships=[]),
            ],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep1, ep2])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 2
        assert len(records) == 2
        assert records[0].skipped_reason == "extraction_failed"
        assert records[1].skipped_reason == "no_new_info"

    @pytest.mark.asyncio
    async def test_new_entities_in_context(self):
        cfg = _make_cfg()
        ep = _make_episode()
        extractor = _make_extractor(
            entities=[
                {"name": "NewAlice", "entity_type": "person"},
                {"name": "NewBob", "entity_type": "person"},
            ],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.create_entity = AsyncMock()
        graph_store.link_episode_entity = AsyncMock()

        ctx = CycleContext()
        # Manual trigger — replay always runs on full cycles
        ctx.trigger = "manual"
        phase = EpisodeReplayPhase(extractor=extractor)
        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        assert len(ctx.replay_new_entity_ids) == 2
        assert len(ctx.affected_entity_ids) == 2
        # All replay new IDs should also be in affected
        assert ctx.replay_new_entity_ids.issubset(ctx.affected_entity_ids)


class TestReplayDeferredExtraction:
    """Tests for LLM-free replay: projection_state filtering + vocab linking."""

    @pytest.mark.asyncio
    async def test_skips_already_projected_episodes(self):
        """PROJECTED episodes are filtered out — re-extraction is deterministic waste."""
        cfg = _make_cfg()
        ep = _make_episode()
        ep.projection_state = MagicMock()
        ep.projection_state.value = "projected"

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        phase = EpisodeReplayPhase(extractor=_make_extractor())
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_targets_cue_only_episodes(self):
        """CUE_ONLY episodes (triage-skipped) are selected for extraction."""
        cfg = _make_cfg()
        ep = _make_episode()
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])

        extractor = _make_extractor(
            entities=[{"name": "Alice", "entity_type": "person"}],
        )
        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert result.items_processed == 1
        assert records[0].new_entities_found == 1

    @pytest.mark.asyncio
    async def test_updates_projection_state_after_extraction(self):
        """After successful extraction, episode is marked PROJECTED."""
        cfg = _make_cfg()
        ep = _make_episode()
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        extractor = _make_extractor(
            entities=[{"name": "NewPerson", "entity_type": "person"}],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.create_entity = AsyncMock()
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        ctx = CycleContext()
        ctx.trigger = "manual"

        phase = EpisodeReplayPhase(extractor=extractor)
        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
            context=ctx,
        )

        graph_store.update_episode.assert_called_once()
        call_args = graph_store.update_episode.call_args
        assert call_args[0][0] == ep.id
        updates = call_args[0][1]
        assert updates["projection_state"] == "projected"
        assert updates["last_projection_reason"] == "replay_deferred_extraction"

    @pytest.mark.asyncio
    async def test_no_projection_update_on_dry_run(self):
        """Dry run skips projection_state update."""
        cfg = _make_cfg()
        ep = _make_episode()
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        extractor = _make_extractor(
            entities=[{"name": "NewPerson", "entity_type": "person"}],
        )

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        graph_store.update_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_projection_update_on_empty_extraction(self):
        """No entities found → no projection_state update."""
        cfg = _make_cfg()
        ep = _make_episode()
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        graph_store.update_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_vocab_linking_creates_links(self):
        """Entity name found in episode content → link created."""
        cfg = _make_cfg()
        ep = _make_episode(content="Alice went to the store with Bob.")
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        alice = _make_entity("ent_alice", "Alice", "person")
        bob = _make_entity("ent_bob", "Bob", "person")

        # Extractor returns nothing (no new entities from extraction)
        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice, bob])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # Should have vocab linking records
        vocab_records = [r for r in records if r.entities_updated > 0]
        assert len(vocab_records) == 1
        assert vocab_records[0].entities_updated == 2  # Alice + Bob

    @pytest.mark.asyncio
    async def test_vocab_linking_skips_already_linked(self):
        """No duplicate links — already-linked entities are skipped."""
        cfg = _make_cfg()
        ep = _make_episode(content="Alice went to the store.")
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        alice = _make_entity("ent_alice", "Alice", "person")
        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice])
        # Alice is already linked
        graph_store.get_episode_entities = AsyncMock(return_value=["ent_alice"])
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # No vocab linking records (Alice was already linked)
        vocab_records = [r for r in records if r.entities_updated > 0]
        assert len(vocab_records) == 0

    @pytest.mark.asyncio
    async def test_vocab_linking_case_insensitive(self):
        """'Alice' matches 'alice' in content."""
        cfg = _make_cfg()
        ep = _make_episode(content="I talked to alice yesterday.")
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        alice = _make_entity("ent_alice", "Alice", "person")
        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        vocab_records = [r for r in records if r.entities_updated > 0]
        assert len(vocab_records) == 1
        graph_store.link_episode_entity.assert_called_once_with(ep.id, "ent_alice")

    @pytest.mark.asyncio
    async def test_vocab_linking_disabled_by_config(self):
        """Respects consolidation_replay_vocab_linking_enabled=False."""
        cfg = _make_cfg(consolidation_replay_vocab_linking_enabled=False)
        ep = _make_episode(content="Alice went to the store.")
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        alice = _make_entity("ent_alice", "Alice", "person")
        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[alice])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # Only the extraction record, no vocab linking
        vocab_records = [r for r in records if r.entities_updated > 0]
        assert len(vocab_records) == 0

    @pytest.mark.asyncio
    async def test_vocab_linking_longer_names_first(self):
        """'React.js' matches before 'React' — longer names take priority."""
        cfg = _make_cfg()
        ep = _make_episode(content="We use React.js for the frontend.")
        ep.projection_state = MagicMock()
        ep.projection_state.value = "cue_only"

        react = _make_entity("ent_react", "React", "technology")
        reactjs = _make_entity("ent_reactjs", "React.js", "technology")
        extractor = _make_extractor(entities=[], relationships=[])

        graph_store = AsyncMock()
        graph_store.get_episodes = AsyncMock(return_value=[ep])
        graph_store.find_entities = AsyncMock(return_value=[react, reactjs])
        graph_store.get_episode_entities = AsyncMock(return_value=[])
        graph_store.link_episode_entity = AsyncMock()
        graph_store.update_episode = AsyncMock()

        phase = EpisodeReplayPhase(extractor=extractor)
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        vocab_records = [r for r in records if r.entities_updated > 0]
        assert len(vocab_records) == 1
        # Both should be linked (React.js matches first, React also matches)
        assert vocab_records[0].entities_updated == 2
        # Verify React.js was linked first (called first)
        calls = graph_store.link_episode_entity.call_args_list
        assert calls[0][0] == (ep.id, "ent_reactjs")
