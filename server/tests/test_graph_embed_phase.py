"""Tests for GraphEmbedPhase consolidation phase."""

from __future__ import annotations

import importlib.util
from unittest.mock import AsyncMock

import pytest

import engram.consolidation.phases.graph_embed as graph_embed_module
from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
from engram.consolidation.phases.graph_embed import GraphEmbedPhase, _get_trainer
from engram.models.consolidation import CycleContext, GraphEmbedRecord
from engram.storage.helix.search import HelixSearchIndex


class TestGraphEmbedPhase:
    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self):
        """Phase should skip when no methods are enabled."""
        phase = GraphEmbedPhase()
        cfg = ActivationConfig(
            consolidation_profile="off",
            graph_embedding_node2vec_enabled=False,
            graph_embedding_transe_enabled=False,
            graph_embedding_gnn_enabled=False,
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=None,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="test_cycle",
        )

        assert result.phase == "graph_embed"
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_node2vec_training(self):
        """Phase should train node2vec when enabled and enough entities."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        class MockSearchIndex:
            pass

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=MockSearchIndex(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
        )

        assert result.status == "success"
        assert result.items_processed > 0
        assert len(records) == 1
        assert isinstance(records[0], GraphEmbedRecord)
        assert records[0].method == "node2vec"
        assert records[0].entities_trained == n
        assert records[0].dimensions == 16

    @pytest.mark.asyncio
    async def test_dry_run_no_storage(self):
        """In dry_run mode, embeddings should be trained but not stored."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 15
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        # Add an affected entity so phase doesn't skip
        context = CycleContext()
        context.affected_entity_ids.add("e0")

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        assert result.status == "success"
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_context_passes_affected_ids(self):
        """Context.affected_entity_ids should trigger full retrain."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()
        context = CycleContext()
        context.affected_entity_ids.add("e0")

        n = 15
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        assert result.status == "success"
        assert records[0].full_retrain is True


class TestGetTrainer:
    def test_node2vec(self):
        cfg = ActivationConfig()
        t = _get_trainer("node2vec", cfg)
        assert t is not None
        assert t.method_name() == "node2vec"

    def test_transe(self):
        cfg = ActivationConfig()
        t = _get_trainer("transe", cfg)
        assert t is not None
        assert t.method_name() == "transe"

    def test_gnn(self):
        cfg = ActivationConfig()
        t = _get_trainer("gnn", cfg)
        assert t is not None
        assert t.method_name() == "gnn"

    def test_unknown(self):
        cfg = ActivationConfig()
        t = _get_trainer("unknown", cfg)
        assert t is None


class TestGraphEmbedPhaseWarnings:
    @pytest.mark.asyncio
    async def test_db_none_warns(self, caplog):
        """Should log warning when db=None and dry_run=False."""
        import logging

        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 15
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        class MockSearchIndex:
            pass

        with caplog.at_level(logging.WARNING):
            result, records = await phase.execute(
                group_id="default",
                graph_store=MockGraph(),
                activation_store=None,
                search_index=MockSearchIndex(),
                cfg=cfg,
                cycle_id="test_cycle",
                dry_run=False,
            )

        assert result.status == "success"
        assert len(records) == 1
        assert any("db=None" in msg for msg in caplog.messages)


class TestGraphEmbedRecord:
    def test_dataclass(self):
        """GraphEmbedRecord should have all required fields."""
        record = GraphEmbedRecord(
            cycle_id="cyc_1",
            group_id="default",
            method="node2vec",
            entities_trained=100,
            dimensions=64,
            training_duration_ms=150.5,
            full_retrain=True,
        )
        assert record.method == "node2vec"
        assert record.entities_trained == 100
        assert record.id.startswith("gemb_")


class TestGraphEmbedIncremental:
    """Tests for incremental training behavior."""

    @pytest.mark.asyncio
    async def test_skip_when_no_changes(self):
        """GraphEmbedPhase should skip when no entities are affected."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

        # Empty context — no affected entities, tiered trigger enables skip
        context = CycleContext(trigger="tiered:cold")

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        assert result.status == "skipped"
        assert result.items_processed == 0
        assert records == []

    @pytest.mark.asyncio
    async def test_skip_when_below_threshold(self):
        """GraphEmbedPhase should skip full retrain when change ratio is below threshold."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
            # 1 affected out of 100 = 1% < 5% threshold
            graph_embedding_retrain_threshold=0.05,
        )
        phase = GraphEmbedPhase()

        n = 100
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        context = CycleContext(trigger="tiered:cold")
        context.affected_entity_ids.add("e0")

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        # Should still succeed (incremental, not full retrain)
        assert result.status == "success"
        # Node2Vec should have run; verify it was NOT a full retrain
        assert len(records) >= 1
        node2vec_rec = [r for r in records if r.method == "node2vec"]
        assert len(node2vec_rec) == 1
        assert node2vec_rec[0].full_retrain is False
        assert node2vec_rec[0].entities_trained == 2

    @pytest.mark.asyncio
    async def test_full_retrain_above_threshold(self):
        """GraphEmbedPhase should do full retrain when change ratio exceeds threshold."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=1,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
            graph_embedding_retrain_threshold=0.05,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        context = CycleContext(trigger="tiered:cold")
        # 5 out of 20 = 25% > 5% threshold
        for i in range(5):
            context.affected_entity_ids.add(f"e{i}")

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        assert result.status == "success"
        node2vec_rec = [r for r in records if r.method == "node2vec"]
        assert len(node2vec_rec) == 1
        assert node2vec_rec[0].full_retrain is True

    @pytest.mark.asyncio
    async def test_stagger_skips_transe(self):
        """TransE should be skipped on non-stagger cycles when not full retrain."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=False,
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=16,
            graph_embedding_transe_epochs=10,
            graph_embedding_gnn_enabled=False,
            graph_embedding_retrain_threshold=0.50,  # high threshold so 1/20 won't trigger
            graph_embedding_stagger_transe=3,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        context = CycleContext(trigger="tiered:cold")
        context.affected_entity_ids.add("e0")

        # Try multiple cycle_ids; at most 1 out of 3 should run TransE
        ran_count = 0
        skipped_count = 0
        for i in range(3):
            result, records = await phase.execute(
                group_id="default",
                graph_store=MockGraph(),
                activation_store=None,
                search_index=type("SI", (), {})(),
                cfg=cfg,
                cycle_id=f"cycle_{i}",
                dry_run=True,
                context=context,
            )
            transe_recs = [r for r in records if r.method == "transe"]
            if transe_recs:
                ran_count += 1
            else:
                skipped_count += 1

        # At least one should have been skipped due to staggering
        assert skipped_count >= 1, f"Expected staggering to skip at least 1 cycle, ran={ran_count}"

    @pytest.mark.asyncio
    async def test_partial_cycle_transe_reports_full_retrain(self):
        """TransE still retrains globally on partial cycles and should report that honestly."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=False,
            graph_embedding_transe_enabled=True,
            graph_embedding_transe_min_triples=20,
            graph_embedding_transe_dimensions=16,
            graph_embedding_transe_epochs=10,
            graph_embedding_gnn_enabled=False,
            graph_embedding_retrain_threshold=0.50,
            graph_embedding_stagger_transe=1,
        )
        phase = GraphEmbedPhase()

        n = 30
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        def make_rel(src, tgt, pred):
            return type(
                "R",
                (),
                {
                    "source_id": src,
                    "target_id": tgt,
                    "predicate": pred,
                },
            )()

        relationships = {}
        for i in range(n):
            rels = []
            if i < n - 1:
                rels.append(make_rel(f"e{i}", f"e{i + 1}", "NEXT"))
            if i % 2 == 0 and i + 2 < n:
                rels.append(make_rel(f"e{i}", f"e{i + 2}", "SKIP"))
            relationships[f"e{i}"] = rels

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

            async def get_relationships(self, eid, direction, group_id):
                return relationships.get(eid, [])

        context = CycleContext(trigger="tiered:cold")
        context.affected_entity_ids.add("e0")

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=type("SI", (), {})(),
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
            context=context,
        )

        assert result.status == "success"
        transe_rec = [r for r in records if r.method == "transe"]
        assert len(transe_rec) == 1
        assert transe_rec[0].full_retrain is True


def _make_helix_search_mock(**extras):
    """Create a mock search index that looks like HelixSearchIndex.

    Avoids MagicMock auto-creating ``_vectors``/``db`` attributes which
    confuse the db-handle detection in GraphEmbedPhase.
    """

    class _HelixSI:
        pass

    si = _HelixSI()
    for k, v in extras.items():
        setattr(si, k, v)
    return si


class TestHelixDBSync:
    """Tests for syncing graph embeddings to HelixDB."""

    @pytest.mark.asyncio
    async def test_sync_called_when_search_index_supports_it(self):
        """GraphEmbedPhase should call sync_graph_embeddings on the search index."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        sync_mock = AsyncMock(return_value=n)
        clear_mock = AsyncMock(return_value=0)
        mock_search = _make_helix_search_mock(
            sync_graph_embeddings=sync_mock,
            clear_graph_embed_vectors=clear_mock,
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=False,
        )

        assert result.status == "success"
        assert len(records) == 1
        sync_mock.assert_called_once()
        embeddings_arg = sync_mock.call_args[0][0]
        assert len(embeddings_arg) == n

    @pytest.mark.asyncio
    async def test_clear_called_on_full_retrain(self):
        """Full retrain should clear stale vectors before syncing."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        sync_mock = AsyncMock(return_value=n)
        clear_mock = AsyncMock(return_value=5)
        mock_search = _make_helix_search_mock(
            sync_graph_embeddings=sync_mock,
            clear_graph_embed_vectors=clear_mock,
        )

        # No context = full cycle = full retrain
        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=False,
        )

        assert result.status == "success"
        clear_mock.assert_called_once_with(group_id="default", method="node2vec")
        sync_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_sync_in_dry_run(self):
        """Dry run should not call sync_graph_embeddings."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        sync_mock = AsyncMock(return_value=0)
        mock_search = _make_helix_search_mock(
            sync_graph_embeddings=sync_mock,
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=True,
        )

        assert result.status == "success"
        sync_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_failure_does_not_crash_phase(self):
        """HelixDB sync failure should be logged but not crash the phase."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        mock_search = _make_helix_search_mock(
            sync_graph_embeddings=AsyncMock(side_effect=RuntimeError("HelixDB down")),
            clear_graph_embed_vectors=AsyncMock(return_value=0),
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="test_cycle",
            dry_run=False,
        )

        # Phase should succeed even though sync failed
        assert result.status == "success"
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_no_sync_for_plain_search_index(self, caplog):
        """Search index without sync_graph_embeddings logs a warning."""
        import logging

        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=10,
            graph_embedding_node2vec_dimensions=16,
            graph_embedding_node2vec_num_walks=2,
            graph_embedding_node2vec_walk_length=5,
            graph_embedding_node2vec_epochs=1,
        )
        phase = GraphEmbedPhase()

        n = 20
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        # Plain object — no sync_graph_embeddings, no db
        plain_search = type("PlainSI", (), {})()

        with caplog.at_level(logging.WARNING):
            result, records = await phase.execute(
                group_id="default",
                graph_store=MockGraph(),
                activation_store=None,
                search_index=plain_search,
                cfg=cfg,
                cycle_id="test_cycle",
                dry_run=False,
            )

        assert result.status == "success"
        assert any("no storage backend" in msg for msg in caplog.messages)

    @pytest.mark.skipif(
        importlib.util.find_spec("helix_native") is None,
        reason="helix_native PyO3 extension is not installed",
    )
    @pytest.mark.asyncio
    async def test_native_full_retrain_replaces_stale_graph_vectors(
        self, tmp_path, monkeypatch
    ):
        """Full retrain should clear stale native vectors before syncing new ones."""

        class FakeEmbeddingProvider:
            def dimension(self):
                return 64

            async def embed_query(self, text):
                return [0.0] * 64

            async def embed(self, texts):
                return [[0.0] * 64 for _ in texts]

        new_vectors = {
            "ent_graph_a": [0.01 * (i + 1) for i in range(64)],
            "ent_graph_b": [0.02 * (i + 1) for i in range(64)],
        }

        class FakeTrainer:
            async def train(self, graph_store, group_id, existing_embeddings=None):
                return new_vectors

        class MockGraph:
            pass

        monkeypatch.setattr(
            graph_embed_module,
            "_get_trainer",
            lambda method, cfg: FakeTrainer() if method == "node2vec" else None,
        )

        index = HelixSearchIndex(
            helix_config=HelixDBConfig(
                transport="native",
                data_dir=str(tmp_path / "native-graph-embed-phase-data"),
            ),
            provider=FakeEmbeddingProvider(),
            embed_config=EmbeddingConfig(),
            storage_dim=64,
            embed_provider="fake",
            embed_model="fake-64",
        )
        await index.initialize()
        try:
            stale_vectors = {
                "ent_graph_a": [0.9 for _ in range(64)],
                "ent_graph_old": [0.8 for _ in range(64)],
            }
            seeded = await index.sync_graph_embeddings(
                stale_vectors,
                method="node2vec",
                group_id="native_brain",
                model_version="old",
            )

            cfg = ActivationConfig(
                graph_embedding_node2vec_enabled=True,
                graph_embedding_node2vec_dimensions=64,
                graph_embedding_transe_enabled=False,
                graph_embedding_gnn_enabled=False,
            )
            phase = GraphEmbedPhase()
            result, records = await phase.execute(
                group_id="native_brain",
                graph_store=MockGraph(),
                activation_store=None,
                search_index=index,
                cfg=cfg,
                cycle_id="native_phase_full_retrain",
                dry_run=False,
            )

            loaded = await index.get_graph_embeddings(
                ["ent_graph_a", "ent_graph_b", "ent_graph_old"],
                method="node2vec",
                group_id="native_brain",
            )

            assert seeded == 2
            assert result.status == "success"
            assert len(records) == 1
            assert records[0].full_retrain is True
            assert records[0].entities_trained == 2
            assert set(loaded) == set(new_vectors)
            assert loaded["ent_graph_a"] == pytest.approx(new_vectors["ent_graph_a"])
            assert loaded["ent_graph_b"] == pytest.approx(new_vectors["ent_graph_b"])
        finally:
            await index.close()
