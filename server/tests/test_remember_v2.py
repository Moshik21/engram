"""Tests for remember tool v2 (evidence extraction integration)."""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from tests.conftest import MockExtractor


class TestRememberV2Config:
    """Test evidence extraction config fields."""

    def test_default_enabled(self):
        cfg = ActivationConfig()
        assert cfg.evidence_extraction_enabled is True

    def test_enable_evidence_extraction(self):
        cfg = ActivationConfig(evidence_extraction_enabled=True)
        assert cfg.evidence_extraction_enabled is True

    def test_default_thresholds(self):
        cfg = ActivationConfig()
        assert cfg.evidence_commit_entity_threshold == 0.70
        assert cfg.evidence_commit_relationship_threshold == 0.75
        assert cfg.evidence_commit_attribute_threshold == 0.65
        assert cfg.evidence_commit_temporal_threshold == 0.60

    def test_custom_thresholds(self):
        cfg = ActivationConfig(
            evidence_commit_entity_threshold=0.80,
            evidence_commit_relationship_threshold=0.85,
        )
        assert cfg.evidence_commit_entity_threshold == 0.80
        assert cfg.evidence_commit_relationship_threshold == 0.85

    def test_adaptive_thresholds_default_on(self):
        cfg = ActivationConfig()
        assert cfg.evidence_adaptive_thresholds is True

    def test_store_deferred_default_on(self):
        cfg = ActivationConfig()
        assert cfg.evidence_store_deferred is True

    def test_client_proposals_default_off(self):
        cfg = ActivationConfig()
        assert cfg.evidence_client_proposals_enabled is False

    def test_forced_commit_cycles(self):
        cfg = ActivationConfig(evidence_forced_commit_cycles=10)
        assert cfg.evidence_forced_commit_cycles == 10

    def test_edge_adjudication_defaults(self):
        cfg = ActivationConfig()
        assert cfg.edge_adjudication_enabled is True
        assert cfg.edge_adjudication_client_enabled is True
        assert cfg.edge_adjudication_server_enabled is False


class TestGraphManagerEvidencePath:
    """Test that GraphManager uses evidence path when enabled."""

    @pytest.mark.asyncio
    async def test_evidence_pipeline_initialized(self):
        """When evidence_extraction_enabled, pipeline objects are created."""
        from unittest.mock import MagicMock

        cfg = ActivationConfig(evidence_extraction_enabled=True)
        graph_store = MagicMock()
        activation_store = MagicMock()
        search_index = MagicMock()
        extractor = MagicMock()

        from engram.graph_manager import GraphManager

        manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=extractor,
            cfg=cfg,
        )
        assert manager._evidence_pipeline is not None
        assert manager._commit_policy is not None
        assert manager._evidence_bridge is not None

    @pytest.mark.asyncio
    async def test_no_evidence_pipeline_when_disabled(self):
        """When evidence_extraction_enabled=False, pipeline objects are None."""
        from unittest.mock import MagicMock

        cfg = ActivationConfig(evidence_extraction_enabled=False)
        graph_store = MagicMock()
        activation_store = MagicMock()
        search_index = MagicMock()
        extractor = MagicMock()

        from engram.graph_manager import GraphManager

        manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=extractor,
            cfg=cfg,
        )
        assert manager._evidence_pipeline is None
        assert manager._commit_policy is None
        assert manager._evidence_bridge is None

    def test_client_proposals_short_circuit_pipeline_when_enabled(self):
        """Validated client proposals should bypass deterministic extraction."""
        from unittest.mock import MagicMock

        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_client_proposals_enabled=True,
        )
        graph_store = MagicMock()
        activation_store = MagicMock()
        search_index = MagicMock()
        extractor = MagicMock()

        from engram.graph_manager import GraphManager

        manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=extractor,
            cfg=cfg,
        )
        manager._evidence_pipeline = MagicMock()
        manager._evidence_pipeline.extract.side_effect = AssertionError(
            "deterministic pipeline should not run when client proposals are present",
        )

        bundle = manager._build_evidence_bundle(
            text="Alice works at Google",
            episode_id="ep1",
            group_id="default",
            proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
            proposed_relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            model_tier="opus",
        )

        assert len(bundle.candidates) == 2
        assert all(c.source_type == "client_proposal" for c in bundle.candidates)
        assert bundle.extractor_stats["client_proposals"]["count"] == 2

    @pytest.mark.asyncio
    async def test_ingest_episode_materializes_client_proposals(
        self,
        graph_store,
        activation_store,
        search_index,
    ):
        """Hot-path remember still commits graph state through the shared materializer."""
        from engram.graph_manager import GraphManager

        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_client_proposals_enabled=True,
            evidence_store_deferred=True,
        )
        manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=MockExtractor(),
            cfg=cfg,
        )

        episode_id = await manager.ingest_episode(
            content="Alice works at Google.",
            source="test",
            proposed_entities=[
                {"name": "Alice", "entity_type": "Person"},
                {"name": "Google", "entity_type": "Organization"},
            ],
            proposed_relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            model_tier="opus",
        )

        evidence = await graph_store.get_episode_evidence(episode_id, group_id="default")
        relationship_row = next(row for row in evidence if row["fact_class"] == "relationship")
        alice_row = next(row for row in evidence if row["payload"].get("name") == "Alice")
        rels = await graph_store.get_relationships(
            alice_row["committed_id"],
            direction="outgoing",
            group_id="default",
        )

        assert len(evidence) == 3
        assert {row["status"] for row in evidence} == {"committed"}
        assert all(row["committed_id"] for row in evidence)
        assert any(rel.id == relationship_row["committed_id"] for rel in rels)
