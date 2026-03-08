"""Tests for the entity merge consolidation phase."""

import time
import uuid

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.merge import EntityMergePhase
from engram.models.activation import ActivationState
from engram.models.consolidation import IdentifierReviewRecord, MergeRecord
from engram.models.entity import Entity
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


def _entity(name, entity_type="Person", group_id="test", access_count=0, **kwargs):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
        access_count=access_count,
        **kwargs,
    )


class TestEntityMergePhase:
    """Tests for EntityMergePhase."""

    @pytest.mark.asyncio
    async def test_similar_names_merged(self, store, activation, search):
        e1 = _entity("John Smith")
        e2 = _entity("john smith")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_affected == 1
        assert len(records) == 1
        assert records[0].similarity >= 0.85

    @pytest.mark.asyncio
    async def test_below_threshold_not_merged(self, store, activation, search):
        e1 = _entity("Alice Johnson")
        e2 = _entity("Bob Williams")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.88)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_numeric_identifiers_do_not_merge(self, store, activation, search):
        e1 = _entity("1712061", entity_type="Thing")
        e2 = _entity("1712018", entity_type="Thing")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_numeric",
            dry_run=False,
        )

        assert result.items_affected == 0
        assert not [record for record in records if isinstance(record, MergeRecord)]
        reviews = [record for record in records if isinstance(record, IdentifierReviewRecord)]
        assert len(reviews) == 1
        assert reviews[0].decision_reason == "identifier_mismatch"

    @pytest.mark.asyncio
    async def test_labeled_identifier_alias_merges(self, store, activation, search):
        e1 = _entity("1712061", entity_type="Thing")
        e2 = _entity("SKU 1712061", entity_type="Thing")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_identifier_alias",
            dry_run=False,
        )

        assert result.items_affected == 1
        assert len(records) == 1
        assert records[0].decision_source == "identifier_policy"
        assert records[0].decision_reason == "identifier_exact_match"
        assert records[0].decision_confidence == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_exact_identifier_aliases_merge_across_types(self, store, activation, search):
        e1 = _entity("1712061", entity_type="Technology", access_count=10)
        e2 = _entity("SKU 1712061", entity_type="Identifier", access_count=1)
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.88)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_cross_type_identifier",
            dry_run=False,
        )

        merges = [record for record in records if isinstance(record, MergeRecord)]
        assert result.items_affected == 1
        assert len(merges) == 1
        survivor = await store.get_entity(e1.id, "test")
        assert survivor is not None
        assert survivor.entity_type == "Identifier"

    @pytest.mark.asyncio
    async def test_same_type_enforcement(self, store, activation, search):
        e1 = _entity("Python", entity_type="Technology")
        e2 = _entity("python", entity_type="Animal")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_require_same_type=True,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # Different types — should NOT merge
        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_dry_run_no_modifications(self, store, activation, search):
        e1 = _entity("John Smith")
        e2 = _entity("john smith")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        assert result.items_affected == 1  # Reported as would-merge
        # Both entities should still exist
        assert await store.get_entity(e1.id, "test") is not None
        assert await store.get_entity(e2.id, "test") is not None

    @pytest.mark.asyncio
    async def test_access_count_survivor(self, store, activation, search):
        """Survivor should be the entity with highest access_count."""
        e1 = _entity("John Smith", access_count=5)
        e2 = _entity("john smith", access_count=10)
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert len(records) == 1
        assert records[0].keep_id == e2.id  # Higher access_count survives
        assert records[0].remove_id == e1.id

    @pytest.mark.asyncio
    async def test_prefix_subblocking_applied(self, store, activation, search):
        """Oversized type block uses prefix sub-blocking and finds correct duplicates."""
        # Create >500 entities with varied prefixes to trigger sub-blocking
        prefixes = ["AA", "BB", "CC", "DD", "EE"]
        count = 0
        for pfx in prefixes:
            for i in range(102):
                await store.create_entity(_entity(f"{pfx}_{i:04d}"))
                count += 1

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_block_size=500,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_sub",
            dry_run=True,
        )

        # Full O(n²) would be count*(count-1)/2; sub-blocking should reduce this
        full_pairs = count * (count - 1) // 2
        assert result.items_processed < full_pairs

    @pytest.mark.asyncio
    async def test_prefix_subblocking_finds_matches(self, store, activation, search):
        """Entities sharing a name prefix are still compared and merged."""
        # Create >500 entities to trigger sub-blocking, plus a near-duplicate pair
        for i in range(500):
            await store.create_entity(_entity(f"Unique_{i:04d}"))
        await store.create_entity(_entity("Alice Smith"))
        await store.create_entity(_entity("alice smith"))

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_block_size=500,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_sub2",
            dry_run=True,
        )

        # "Alice Smith" and "alice smith" share prefix "al" → should merge
        assert result.items_affected >= 1
        merges = [record for record in records if isinstance(record, MergeRecord)]
        names = {(r.keep_name.lower(), r.remove_name.lower()) for r in merges}
        assert any("alice smith" in n for pair in names for n in pair)

    @pytest.mark.asyncio
    async def test_cross_prefix_not_compared(self, store, activation, search):
        """Entities with different prefixes in oversized block are never compared."""
        # Build a block of 510 entities: 255 starting with "aa", 255 with "zz"
        for i in range(255):
            await store.create_entity(_entity(f"AA_{i:04d}"))
        for i in range(255):
            await store.create_entity(_entity(f"ZZ_{i:04d}"))

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_block_size=500,
        )
        phase = EntityMergePhase()
        result, _ = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_cross",
            dry_run=True,
        )

        # Full O(n²) would be 510*509/2 = 129,795 pairs
        # Sub-blocking: aa-block C(255,2) + zz-block C(255,2) = 32,385 + 32,385 = 64,770
        full_pairs = 510 * 509 // 2
        assert result.items_processed < full_pairs

    @pytest.mark.asyncio
    async def test_max_merges_limit(self, store, activation, search):
        """Should respect consolidation_merge_max_per_cycle."""
        # Create 5 pairs of duplicates
        for i in range(5):
            await store.create_entity(_entity(f"Entity {i}"))
            await store.create_entity(_entity(f"entity {i}"))

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_max_per_cycle=2,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_affected <= 2

    @pytest.mark.asyncio
    async def test_group_isolation(self, store, activation, search):
        """Entities from different groups should not affect each other."""
        e1 = _entity("John Smith", group_id="group_a")
        e2 = _entity("john smith", group_id="group_b")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="group_a",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        # Only one entity in group_a, nothing to merge
        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_empty_graph(self, store, activation, search):
        cfg = ActivationConfig()
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.items_processed == 0
        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_merge_sums_consolidated_strength(self, store, activation, search):
        """Survivor should accumulate loser's consolidated_strength."""
        now = time.time()
        e1 = _entity("John Smith", access_count=5)
        e2 = _entity("john smith", access_count=10)
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Set up activation states with consolidated_strength
        s1 = ActivationState(
            node_id=e1.id,
            access_history=[now - 60],
            access_count=5,
            consolidated_strength=2.5,
        )
        s2 = ActivationState(
            node_id=e2.id,
            access_history=[now - 120],
            access_count=10,
            consolidated_strength=3.7,
        )
        await activation.set_activation(e1.id, s1)
        await activation.set_activation(e2.id, s2)

        cfg = ActivationConfig(consolidation_merge_threshold=0.85)
        phase = EntityMergePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_cs",
            dry_run=False,
        )

        assert len(records) == 1
        survivor_id = records[0].keep_id
        survivor_state = await activation.get_activation(survivor_id)
        # Survivor (e2, higher access_count) should have combined cs
        assert survivor_state.consolidated_strength == pytest.approx(2.5 + 3.7)


class TestMergeANNEmbeddings:
    """Tests for embedding-based ANN candidate pre-filtering."""

    @pytest.mark.asyncio
    async def test_ann_finds_duplicates_via_embeddings(self, store, activation, search):
        """When embeddings are available, ANN path should find duplicates."""
        e1 = _entity("John Smith")
        e2 = _entity("john smith")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Mock search index that returns embeddings — similar entities get similar vectors
        class MockSearchWithEmbeddings:
            def __init__(self, inner):
                self._inner = inner

            async def get_entity_embeddings(self, entity_ids, group_id=None):
                result = {}
                for eid in entity_ids:
                    if eid == e1.id:
                        result[eid] = [1.0, 0.0, 0.0] * 10  # 30-dim
                    elif eid == e2.id:
                        result[eid] = [0.99, 0.01, 0.0] * 10  # Very similar
                return result

            async def remove(self, entity_id):
                await self._inner.remove(entity_id)

        mock_search = MockSearchWithEmbeddings(search)
        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_use_embeddings=True,
            consolidation_merge_embedding_threshold=0.85,
            consolidation_merge_embedding_min_coverage=0.5,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="cyc_ann",
            dry_run=False,
        )

        assert result.items_affected == 1
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_ann_skips_dissimilar_entities(self, store, activation, search):
        """ANN should not return pairs with low embedding similarity."""
        e1 = _entity("Alice Johnson")
        e2 = _entity("Bob Williams")
        await store.create_entity(e1)
        await store.create_entity(e2)

        class MockSearchWithEmbeddings:
            def __init__(self, inner):
                self._inner = inner

            async def get_entity_embeddings(self, entity_ids, group_id=None):
                result = {}
                for eid in entity_ids:
                    if eid == e1.id:
                        result[eid] = [1.0, 0.0, 0.0] * 10  # Orthogonal
                    elif eid == e2.id:
                        result[eid] = [0.0, 1.0, 0.0] * 10
                return result

            async def remove(self, entity_id):
                await self._inner.remove(entity_id)

        mock_search = MockSearchWithEmbeddings(search)
        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_use_embeddings=True,
            consolidation_merge_embedding_threshold=0.85,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="cyc_ann2",
            dry_run=False,
        )

        # Embeddings are orthogonal — no candidates, no merges
        assert result.items_affected == 0
        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_ann_respects_same_type_requirement(self, store, activation, search):
        """ANN candidates must not merge across types when same-type is required."""
        e1 = _entity("React", entity_type="Technology")
        e2 = _entity("React", entity_type="Project")
        await store.create_entity(e1)
        await store.create_entity(e2)

        class MockSearchWithEmbeddings:
            def __init__(self, inner):
                self._inner = inner

            async def get_entity_embeddings(self, entity_ids, group_id=None):
                return {
                    e1.id: [1.0, 0.0, 0.0] * 10,
                    e2.id: [1.0, 0.0, 0.0] * 10,
                }

            async def remove(self, entity_id):
                await self._inner.remove(entity_id)

        mock_search = MockSearchWithEmbeddings(search)
        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_use_embeddings=True,
            consolidation_merge_embedding_threshold=0.85,
            consolidation_merge_require_same_type=True,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="cyc_ann_same_type",
            dry_run=False,
        )

        assert result.items_affected == 0
        assert records == []

    @pytest.mark.asyncio
    async def test_ann_falls_back_on_low_coverage(self, store, activation, search):
        """Should fall back to O(N²) when embedding coverage is too low."""
        e1 = _entity("John Smith")
        e2 = _entity("john smith")
        e3 = _entity("Alice Jones")
        await store.create_entity(e1)
        await store.create_entity(e2)
        await store.create_entity(e3)

        class MockSearchLowCoverage:
            def __init__(self, inner):
                self._inner = inner

            async def get_entity_embeddings(self, entity_ids, group_id=None):
                # Only 1 out of 3 entities has embeddings (33% < 50%)
                return {e1.id: [1.0, 0.0, 0.0] * 10}

            async def remove(self, entity_id):
                await self._inner.remove(entity_id)

        mock_search = MockSearchLowCoverage(search)
        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_use_embeddings=True,
            consolidation_merge_embedding_min_coverage=0.5,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=mock_search,
            cfg=cfg,
            cycle_id="cyc_fallback",
            dry_run=False,
        )

        # Should fall back to O(N²) and still find the duplicate
        assert result.items_affected == 1

    @pytest.mark.asyncio
    async def test_ann_disabled_uses_fallback(self, store, activation, search):
        """When use_embeddings=False, should use O(N²) path."""
        e1 = _entity("John Smith")
        e2 = _entity("john smith")
        await store.create_entity(e1)
        await store.create_entity(e2)

        cfg = ActivationConfig(
            consolidation_merge_threshold=0.85,
            consolidation_merge_use_embeddings=False,
        )
        phase = EntityMergePhase()
        result, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_disabled",
            dry_run=False,
        )

        assert result.items_affected == 1
        assert result.items_processed > 0  # O(N²) pairs checked
