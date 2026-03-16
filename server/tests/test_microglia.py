"""Tests for Microglia phase: complement-mediated graph immune surveillance."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.microglia import (
    MicrogliaPhase,
    _dedup_segments,
    _extract_cycle_number,
)
from engram.models.consolidation import CycleContext
from engram.models.entity import Entity
from engram.models.relationship import Relationship

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(
    eid: str,
    etype: str = "Person",
    name: str = "Test",
    summary: str | None = None,
    identity_core: bool = False,
    mat_tier: str | None = None,
) -> Entity:
    e = Entity(
        id=eid,
        name=name,
        entity_type=etype,
        group_id="default",
        summary=summary,
        identity_core=identity_core,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    if mat_tier is not None:
        object.__setattr__(e, "mat_tier", mat_tier)
    return e


def _rel(
    src: str,
    tgt: str,
    predicate: str = "RELATES_TO",
    weight: float = 1.0,
    source_episode: str | None = None,
) -> Relationship:
    return Relationship(
        id=f"rel_{src}_{tgt}",
        source_id=src,
        target_id=tgt,
        predicate=predicate,
        weight=weight,
        source_episode=source_episode,
        group_id="default",
    )


def _cfg(**overrides) -> ActivationConfig:
    defaults = dict(
        microglia_enabled=True,
        microglia_tag_threshold=0.5,
        microglia_confirm_threshold=0.4,
        microglia_min_cycles_to_demote=2,
        microglia_max_demotions_per_cycle=20,
        microglia_scan_edges_per_cycle=500,
        microglia_scan_entities_per_cycle=200,
    )
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _make_stores():
    graph = AsyncMock()
    activation = AsyncMock()
    search = AsyncMock()
    graph.get_identity_core_entities = AsyncMock(return_value=[])
    graph.sample_edges = AsyncMock(return_value=[])
    graph.find_entities = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(return_value=None)
    graph.update_entity = AsyncMock()
    graph.update_relationship_weight = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    graph._consolidation_store = None
    search.get_entity_embeddings = AsyncMock(return_value={})
    return graph, activation, search


# ---------------------------------------------------------------------------
# 1. Skipped when disabled
# ---------------------------------------------------------------------------


class TestMicrogliaSkipped:
    @pytest.mark.asyncio
    async def test_microglia_skipped_when_disabled(self):
        phase = MicrogliaPhase()
        cfg = ActivationConfig(microglia_enabled=False)
        graph, activation, search = _make_stores()

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        assert result.status == "skipped"
        assert result.duration_ms == 0.0
        assert records == []


# ---------------------------------------------------------------------------
# 2-4. C1q Domain Detector
# ---------------------------------------------------------------------------


class TestC1qDomainScorer:
    def setup_method(self):
        self.phase = MicrogliaPhase()
        self.cfg = _cfg()

    def test_c1q_domain_scores_incompatible_types(self):
        """Person-Software with generic predicate gets high score."""
        score = self.phase._score_c1q_domain(
            source_type="Person",
            target_type="Software",
            predicate="RELATES_TO",
            weight=0.2,
            has_episode_evidence=False,
            cfg=self.cfg,
        )
        # Base 0.3 + generic 0.25 + low weight 0.15 + no evidence 0.2 = 0.9
        assert score == pytest.approx(0.9, abs=0.01)

    def test_c1q_domain_allows_legitimate_predicates(self):
        """DEVELOPS/USES etc. get score 0 even with incompatible types."""
        for pred in ("DEVELOPS", "USES", "CREATED", "MAINTAINS", "WORKS_ON"):
            score = self.phase._score_c1q_domain(
                source_type="Person",
                target_type="Software",
                predicate=pred,
                weight=0.5,
                has_episode_evidence=True,
                cfg=self.cfg,
            )
            assert score == 0.0, f"Expected 0 for {pred}"

    def test_c1q_domain_compatible_types_score_zero(self):
        """Person-Person is not in the incompatible set."""
        score = self.phase._score_c1q_domain(
            source_type="Person",
            target_type="Person",
            predicate="RELATES_TO",
            weight=0.1,
            has_episode_evidence=False,
            cfg=self.cfg,
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# 5-7. C3 Summary Detector
# ---------------------------------------------------------------------------


class TestC3SummaryScorer:
    def setup_method(self):
        self.phase = MicrogliaPhase()

    def test_c3_summary_detects_meta_contamination(self):
        """Summary with 'knowledge graph' patterns gets flagged."""
        summary = "A node in the knowledge graph representing a developer"
        score, cleaned = self.phase._score_c3_summary(summary)
        assert score > 0.0

    def test_c3_summary_clean_summary_score_zero(self):
        """Normal summary passes without issue."""
        summary = "Alex is a software developer based in Portland"
        score, cleaned = self.phase._score_c3_summary(summary)
        assert score == 0.0
        assert cleaned is None

    def test_c3_summary_repairs_partial_contamination(self):
        """Mixed summary: meta segments removed, clean segments kept."""
        summary = "Alex is a software developer; A node in the knowledge graph; He enjoys hiking"
        score, cleaned = self.phase._score_c3_summary(summary)
        assert score > 0.0
        # Cleaned should contain real content but not meta
        assert cleaned is not None
        assert "software developer" in cleaned
        assert "knowledge graph" not in cleaned


# ---------------------------------------------------------------------------
# 8. Scan Edges Tags Contaminated (integration)
# ---------------------------------------------------------------------------


class TestScanEdgesIntegration:
    @pytest.mark.asyncio
    async def test_scan_edges_tags_contaminated(self):
        phase = MicrogliaPhase()
        cfg = _cfg()
        graph, activation, search = _make_stores()

        person = _entity("e1", "Person", "Alice")
        software = _entity("e2", "Software", "MyApp")

        # Return an edge with incompatible types + generic predicate
        edge = _rel("e1", "e2", "RELATES_TO", weight=0.2, source_episode=None)
        graph.sample_edges = AsyncMock(return_value=[edge])
        graph.get_entity = AsyncMock(
            side_effect=lambda eid, gid: person if eid == "e1" else software
        )

        # Set up consolidation store for tag creation
        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.create_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        assert result.status == "success"
        assert result.items_affected >= 1
        # Verify tag was created
        consol_store.create_complement_tag.assert_called_once()
        call_args = consol_store.create_complement_tag.call_args
        target_type = (
            call_args.kwargs.get("target_type")
            if call_args.kwargs
            else call_args.args[1]["target_type"]
        )
        assert target_type == "edge"

        # Check records
        tagged_records = [r for r in records if r.action == "tagged"]
        assert len(tagged_records) >= 1
        assert tagged_records[0].tag_type == "c1q_domain"


# ---------------------------------------------------------------------------
# 9. Scan Summaries Repairs Contaminated (integration)
# ---------------------------------------------------------------------------


class TestScanSummariesIntegration:
    @pytest.mark.asyncio
    async def test_scan_summaries_repairs_contaminated(self):
        phase = MicrogliaPhase()
        cfg = _cfg()
        graph, activation, search = _make_stores()

        contaminated = _entity(
            "e1",
            "Person",
            "Alice",
            summary="Alice is a developer; A node in the knowledge graph",
        )
        graph.find_entities = AsyncMock(return_value=[contaminated])
        graph.sample_edges = AsyncMock(return_value=[])

        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        graph._consolidation_store = consol_store

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        assert result.status == "success"
        # Entity should have been updated with cleaned summary
        graph.update_entity.assert_called_once()
        update_args = graph.update_entity.call_args
        updated_summary = update_args[0][1]["summary"]
        assert "knowledge graph" not in updated_summary
        assert "developer" in updated_summary

        repaired = [r for r in records if r.action == "repaired"]
        assert len(repaired) == 1


# ---------------------------------------------------------------------------
# 10. Demote reduces weight (Tag -> Confirm -> Demote lifecycle)
# ---------------------------------------------------------------------------


class TestDemoteLifecycle:
    @pytest.mark.asyncio
    async def test_demote_reduces_weight(self):
        phase = MicrogliaPhase()
        cfg = _cfg()
        graph, activation, search = _make_stores()

        # Set up confirmed tag ready for demotion
        confirmed_tag = {
            "id": "tag_1",
            "target_type": "edge",
            "target_id": "e1:e2:RELATES_TO",
            "tag_type": "c1q_domain",
            "score": 0.8,
        }

        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[confirmed_tag])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.clear_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        # Edge currently has weight 1.0
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("e2", 1.0, "RELATES_TO")]
        )
        graph.sample_edges = AsyncMock(return_value=[])
        graph.find_entities = AsyncMock(return_value=[])

        context = CycleContext()

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
            context=context,
        )

        assert result.status == "success"
        # Weight should be reduced by 90% (delta = -0.9)
        graph.update_relationship_weight.assert_called_once()
        call_args = graph.update_relationship_weight.call_args
        delta = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("delta")
        assert delta == pytest.approx(-0.9, abs=0.01)

        # Context should track the demotion
        assert "e1:e2:RELATES_TO" in context.microglia_demoted_edge_ids

        demoted = [r for r in records if r.action == "demoted"]
        assert len(demoted) == 1


# ---------------------------------------------------------------------------
# 11. Identity-core protection
# ---------------------------------------------------------------------------


class TestIdentityCoreProtection:
    @pytest.mark.asyncio
    async def test_identity_core_protection(self):
        phase = MicrogliaPhase()
        cfg = _cfg()
        graph, activation, search = _make_stores()

        core_entity = _entity("e1", "Person", "Alex", identity_core=True)
        graph.get_identity_core_entities = AsyncMock(return_value=[core_entity])

        # Tag targeting an identity-core entity
        active_tag = {
            "id": "tag_1",
            "target_type": "entity",
            "target_id": "e1",
            "tag_type": "c3_summary",
            "score": 0.7,
        }

        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[active_tag])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.clear_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store
        graph.sample_edges = AsyncMock(return_value=[])
        graph.find_entities = AsyncMock(return_value=[])

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        # Tag should have been cleared due to identity-core protection
        consol_store.clear_complement_tag.assert_called_once_with("tag_1")
        cleared = [r for r in records if r.action == "cleared"]
        assert len(cleared) == 1
        assert "identity_core" in cleared[0].detail


# ---------------------------------------------------------------------------
# 12. Max demotions cap
# ---------------------------------------------------------------------------


class TestMaxDemotionsCap:
    @pytest.mark.asyncio
    async def test_max_demotions_cap(self):
        phase = MicrogliaPhase()
        cfg = _cfg(microglia_max_demotions_per_cycle=2)
        graph, activation, search = _make_stores()

        # Create 5 confirmed tags
        confirmed_tags = [
            {
                "id": f"tag_{i}",
                "target_type": "edge",
                "target_id": f"e{i}:e{i + 10}:RELATES_TO",
                "tag_type": "c1q_domain",
                "score": 0.8,
            }
            for i in range(5)
        ]

        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=confirmed_tags)
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.clear_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        # Return neighbors that match the target_id format e{i}:e{i+10}:RELATES_TO
        async def mock_neighbors(entity_id, group_id=None):
            # For any source entity e{i}, return the matching target e{i+10}
            for tag in confirmed_tags:
                parts = tag["target_id"].split(":", 2)
                if parts[0] == entity_id:
                    return [(parts[1], 1.0, parts[2])]
            return []

        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=mock_neighbors)
        graph.sample_edges = AsyncMock(return_value=[])
        graph.find_entities = AsyncMock(return_value=[])

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        # Only 2 demotions should happen despite 5 confirmed tags
        demoted = [r for r in records if r.action == "demoted"]
        assert len(demoted) == 2
        assert graph.update_relationship_weight.call_count == 2


# ---------------------------------------------------------------------------
# 13-14. Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_dedup_segments(self):
        segments = [
            "Alice is a developer",
            "Alice is a software developer",  # near-duplicate
            "She lives in Portland",
        ]
        result = _dedup_segments(segments, threshold=0.6)
        # The near-duplicate should be removed
        assert len(result) == 2
        assert "Alice is a developer" in result
        assert "She lives in Portland" in result

    def test_dedup_segments_empty(self):
        assert _dedup_segments([]) == []

    def test_dedup_segments_no_duplicates(self):
        segments = ["cats are furry", "the sky is blue"]
        result = _dedup_segments(segments)
        assert len(result) == 2

    def test_extract_cycle_number(self):
        num = _extract_cycle_number("cyc_abc123")
        assert isinstance(num, int)
        assert 0 <= num < 1_000_000
        # Deterministic
        assert _extract_cycle_number("cyc_abc123") == num

    def test_extract_cycle_number_invalid(self):
        # Non-hex should fall back to time-based int
        num = _extract_cycle_number("not_a_cycle")
        assert isinstance(num, int)

    def test_extract_cycle_number_none(self):
        num = _extract_cycle_number(None)  # type: ignore[arg-type]
        assert isinstance(num, int)


# ---------------------------------------------------------------------------
# 15-18. Dream LTD Sweep Tests
# ---------------------------------------------------------------------------


class TestLtdSweep:
    @pytest.mark.asyncio
    async def test_ltd_sweep_decays_low_activation_edges(self):
        """Basic LTD sweep functionality: low-activation entities get edges decayed."""
        from engram.consolidation.phases.dream import DreamSpreadingPhase

        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(
            consolidation_dream_ltd_sweep_enabled=True,
            consolidation_dream_ltd_sweep_size=50,
            consolidation_dream_ltd_sweep_decay=0.002,
        )
        graph = AsyncMock()
        activation = AsyncMock()

        # Entity with very low activation
        state = SimpleNamespace(access_history=[])
        activation.get_top_activated = AsyncMock(return_value=[("low_ent", state)])

        graph.get_identity_core_entities = AsyncMock(return_value=[])
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("neighbor_1", 0.5, "KNOWS", "Person")]
        )
        graph.update_relationship_weight = AsyncMock()

        now = 1_000_000.0

        with patch("engram.consolidation.phases.dream.compute_activation", return_value=0.01):
            decayed = await phase._apply_ltd_low_activation_sweep(
                activation_store=activation,
                graph_store=graph,
                group_id="default",
                cfg=cfg,
                now=now,
            )

        assert decayed >= 1
        graph.update_relationship_weight.assert_called()

    @pytest.mark.asyncio
    async def test_ltd_sweep_skips_identity_core(self):
        """Identity-core entities are protected from LTD sweep."""
        from engram.consolidation.phases.dream import DreamSpreadingPhase

        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(
            consolidation_dream_ltd_sweep_enabled=True,
            consolidation_dream_ltd_sweep_size=50,
            consolidation_dream_ltd_sweep_decay=0.002,
        )
        graph = AsyncMock()
        activation = AsyncMock()

        state = SimpleNamespace(access_history=[])
        activation.get_top_activated = AsyncMock(return_value=[("core_ent", state)])

        core = _entity("core_ent", "Person", "Alex", identity_core=True)
        graph.get_identity_core_entities = AsyncMock(return_value=[core])
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("neighbor_1", 0.5, "KNOWS", "Person")]
        )
        graph.update_relationship_weight = AsyncMock()

        now = 1_000_000.0

        with patch("engram.consolidation.phases.dream.compute_activation", return_value=0.01):
            decayed = await phase._apply_ltd_low_activation_sweep(
                activation_store=activation,
                graph_store=graph,
                group_id="default",
                cfg=cfg,
                now=now,
            )

        assert decayed == 0
        graph.update_relationship_weight.assert_not_called()

    @pytest.mark.asyncio
    async def test_ltd_sweep_skips_dream_associated(self):
        """DREAM_ASSOCIATED edges are not decayed by LTD sweep."""
        from engram.consolidation.phases.dream import DreamSpreadingPhase

        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(
            consolidation_dream_ltd_sweep_enabled=True,
            consolidation_dream_ltd_sweep_size=50,
            consolidation_dream_ltd_sweep_decay=0.002,
        )
        graph = AsyncMock()
        activation = AsyncMock()

        state = SimpleNamespace(access_history=[])
        activation.get_top_activated = AsyncMock(return_value=[("low_ent", state)])

        graph.get_identity_core_entities = AsyncMock(return_value=[])
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("neighbor_1", 0.5, "DREAM_ASSOCIATED", "Concept")]
        )
        graph.update_relationship_weight = AsyncMock()

        now = 1_000_000.0

        with patch("engram.consolidation.phases.dream.compute_activation", return_value=0.01):
            decayed = await phase._apply_ltd_low_activation_sweep(
                activation_store=activation,
                graph_store=graph,
                group_id="default",
                cfg=cfg,
                now=now,
            )

        assert decayed == 0
        graph.update_relationship_weight.assert_not_called()

    @pytest.mark.asyncio
    async def test_ltd_sweep_disabled_by_default(self):
        """LTD sweep doesn't run when config disabled."""
        from engram.consolidation.phases.dream import DreamSpreadingPhase

        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(consolidation_dream_ltd_sweep_enabled=False)
        graph = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()

        graph.get_identity_core_entities = AsyncMock(return_value=[])
        graph.get_active_neighbors_with_weights = AsyncMock(return_value=[])
        activation.get_top_activated = AsyncMock(return_value=[])

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        # Dream phase skips entirely when no seeds and sweep disabled
        assert result.status in ("success", "skipped")


# ---------------------------------------------------------------------------
# 19. Orphan Edge Fix: get_active_neighbors excludes soft-deleted entities
# ---------------------------------------------------------------------------


class TestOrphanEdgeFix:
    @pytest.mark.asyncio
    async def test_get_active_neighbors_excludes_deleted_entities(self):
        from engram.config import HelixDBConfig
        from engram.storage.helix.graph import HelixGraphStore

        """Verify get_active_neighbors_with_weights filters out soft-deleted entities."""
        store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await store.initialize()

        # Create two entities and a relationship
        e1 = Entity(
            id="e1",
            name="Alice",
            entity_type="Person",
            group_id="default",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        e2 = Entity(
            id="e2",
            name="Bob",
            entity_type="Person",
            group_id="default",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await store.create_entity(e1)
        await store.create_entity(e2)

        rel = Relationship(
            id="r1",
            source_id="e1",
            target_id="e2",
            predicate="KNOWS",
            weight=1.0,
            group_id="default",
        )
        await store.create_relationship(rel)

        # Before deletion: neighbor visible
        neighbors = await store.get_active_neighbors_with_weights(
            "e1",
            group_id="default",
        )
        assert any(nid == "e2" for nid, *_ in neighbors)

        # Soft-delete e2
        await store.delete_entity("e2", soft=True, group_id="default")

        # After deletion: neighbor excluded
        neighbors = await store.get_active_neighbors_with_weights(
            "e1",
            group_id="default",
        )
        assert not any(nid == "e2" for nid, *_ in neighbors)

        await store.close()
