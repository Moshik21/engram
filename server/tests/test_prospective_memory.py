"""Tests for prospective memory (Wave 4): intentions and trigger matching."""

from __future__ import annotations

import socket
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from engram.models.prospective import Intention, IntentionMatch
from engram.retrieval.prospective import _cosine_similarity, _is_active, check_triggers


def _helix_available() -> bool:
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not _helix_available(), reason="HelixDB not available"),
]


# ─── Model tests ──────────────────────────────────────────────────────


class TestIntentionModel:
    def test_intention_defaults(self):
        i = Intention(
            id="int_abc123",
            trigger_text="Python upgrades",
            action_text="User has a migration plan",
        )
        assert i.trigger_type == "semantic"
        assert i.threshold == 0.7
        assert i.max_fires == 5
        assert i.fire_count == 0
        assert i.enabled is True
        assert i.group_id == "default"
        assert i.entity_name is None
        assert i.expires_at is None

    def test_intention_match_dataclass(self):
        m = IntentionMatch(
            intention_id="int_abc",
            trigger_text="Python",
            action_text="migration plan",
            similarity=0.85,
            matched_via="semantic",
        )
        assert m.intention_id == "int_abc"
        assert m.similarity == 0.85
        assert m.matched_via == "semantic"


# ─── Trigger matching tests ──────────────────────────────────────────


class TestIsActive:
    def test_active_intention(self):
        i = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            enabled=True,
            fire_count=0,
            max_fires=5,
        )
        assert _is_active(i) is True

    def test_disabled_intention(self):
        i = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            enabled=False,
        )
        assert _is_active(i) is False

    def test_exhausted_intention(self):
        i = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            fire_count=5,
            max_fires=5,
        )
        assert _is_active(i) is False

    def test_expired_intention(self):
        i = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert _is_active(i) is False

    def test_future_expiry_is_active(self):
        i = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        assert _is_active(i) is True


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0


@pytest.mark.asyncio
class TestTriggerMatching:
    async def test_entity_mention_trigger_fires(self):
        intention = Intention(
            id="int_1",
            trigger_text="Python mentioned",
            action_text="check migration",
            trigger_type="entity_mention",
            entity_name="Python",
        )
        matches = await check_triggers(
            content="We discussed Python today",
            entity_names=["Python", "Django"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 1
        assert matches[0].matched_via == "entity_mention"
        assert matches[0].similarity == 1.0

    async def test_entity_mention_case_insensitive(self):
        intention = Intention(
            id="int_1",
            trigger_text="python mentioned",
            action_text="check it",
            trigger_type="entity_mention",
            entity_name="python",
        )
        matches = await check_triggers(
            content="",
            entity_names=["Python", "Django"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 1

    async def test_entity_mention_no_match(self):
        intention = Intention(
            id="int_1",
            trigger_text="Rust mentioned",
            action_text="check it",
            trigger_type="entity_mention",
            entity_name="Rust",
        )
        matches = await check_triggers(
            content="",
            entity_names=["Python", "Django"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 0

    async def test_semantic_trigger_fires_above_threshold(self):
        async def mock_embed(text: str) -> list[float]:
            if "upgrade" in text.lower() or "python" in text.lower():
                return [0.9, 0.1, 0.0]
            return [0.1, 0.9, 0.0]

        intention = Intention(
            id="int_1",
            trigger_text="Python upgrades",
            action_text="migration plan exists",
            threshold=0.5,
        )
        matches = await check_triggers(
            content="We are upgrading Python",
            entity_names=[],
            intentions=[intention],
            embed_fn=mock_embed,
        )
        assert len(matches) == 1
        assert matches[0].matched_via == "semantic"
        assert matches[0].similarity > 0.5

    async def test_semantic_trigger_skips_below_threshold(self):
        async def mock_embed(text: str) -> list[float]:
            if "python" in text.lower():
                return [0.9, 0.1, 0.0]
            return [0.0, 0.1, 0.9]

        intention = Intention(
            id="int_1",
            trigger_text="Python upgrades",
            action_text="migration plan",
            threshold=0.95,
        )
        matches = await check_triggers(
            content="We went grocery shopping",
            entity_names=[],
            intentions=[intention],
            embed_fn=mock_embed,
        )
        assert len(matches) == 0

    async def test_disabled_intention_skipped(self):
        intention = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            trigger_type="entity_mention",
            entity_name="Python",
            enabled=False,
        )
        matches = await check_triggers(
            content="",
            entity_names=["Python"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 0

    async def test_max_fires_respected(self):
        intention = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            trigger_type="entity_mention",
            entity_name="Python",
            fire_count=5,
            max_fires=5,
        )
        matches = await check_triggers(
            content="",
            entity_names=["Python"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 0

    async def test_expired_intention_skipped(self):
        intention = Intention(
            id="int_1",
            trigger_text="test",
            action_text="action",
            trigger_type="entity_mention",
            entity_name="Python",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        matches = await check_triggers(
            content="",
            entity_names=["Python"],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 0

    async def test_empty_intentions_returns_empty(self):
        matches = await check_triggers(
            content="anything",
            entity_names=["A"],
            intentions=[],
            embed_fn=None,
        )
        assert matches == []

    async def test_results_sorted_by_similarity(self):
        async def mock_embed(text: str) -> list[float]:
            if "high" in text.lower():
                return [0.95, 0.05, 0.0]
            if "medium" in text.lower():
                return [0.7, 0.3, 0.0]
            return [0.85, 0.1, 0.05]

        intentions = [
            Intention(
                id="int_1",
                trigger_text="medium match",
                action_text="action1",
                threshold=0.5,
            ),
            Intention(
                id="int_2",
                trigger_text="high match",
                action_text="action2",
                threshold=0.5,
            ),
        ]
        matches = await check_triggers(
            content="high relevance content",
            entity_names=[],
            intentions=intentions,
            embed_fn=mock_embed,
        )
        if len(matches) >= 2:
            assert matches[0].similarity >= matches[1].similarity

    async def test_no_embed_fn_skips_semantic(self):
        """Semantic triggers should be skipped when embed_fn is None."""
        intention = Intention(
            id="int_1",
            trigger_text="Python upgrades",
            action_text="plan exists",
            threshold=0.5,
        )
        matches = await check_triggers(
            content="We are upgrading Python",
            entity_names=[],
            intentions=[intention],
            embed_fn=None,
        )
        assert len(matches) == 0


# ─── SQLite storage tests ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestSQLiteIntentionStorage:
    @pytest.fixture
    async def graph_store(self, tmp_path):
        from engram.config import HelixDBConfig
        from engram.storage.helix.graph import HelixGraphStore

        store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def gid(self):
        return f"test_{uuid4().hex[:8]}"

    async def test_create_and_get_intention(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_1",
            trigger_text="Python upgrades",
            action_text="Migration plan exists",
            group_id=gid,
        )
        result_id = await graph_store.create_intention(intention)
        assert result_id == f"int_{gid}_1"

        fetched = await graph_store.get_intention(f"int_{gid}_1", gid)
        assert fetched is not None
        assert fetched.trigger_text == "Python upgrades"
        assert fetched.action_text == "Migration plan exists"
        assert fetched.enabled is True
        assert fetched.fire_count == 0

    async def test_list_intentions_enabled_only(self, graph_store, gid):
        i1 = Intention(
            id=f"int_{gid}_a",
            trigger_text="a",
            action_text="a",
            group_id=gid,
            enabled=True,
        )
        i2 = Intention(
            id=f"int_{gid}_b",
            trigger_text="b",
            action_text="b",
            group_id=gid,
            enabled=False,
        )
        await graph_store.create_intention(i1)
        await graph_store.create_intention(i2)

        enabled = await graph_store.list_intentions(gid, enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].id == f"int_{gid}_a"

        all_items = await graph_store.list_intentions(gid, enabled_only=False)
        assert len(all_items) == 2

    async def test_increment_fire_count(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_fire",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.increment_intention_fire_count(f"int_{gid}_fire", gid)
        await graph_store.increment_intention_fire_count(f"int_{gid}_fire", gid)

        fetched = await graph_store.get_intention(f"int_{gid}_fire", gid)
        assert fetched.fire_count == 2

    async def test_delete_intention_soft(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_del",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.delete_intention(f"int_{gid}_del", gid, soft=True)

        fetched = await graph_store.get_intention(f"int_{gid}_del", gid)
        assert fetched is not None
        assert fetched.enabled is False

    async def test_delete_intention_hard(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_del2",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.delete_intention(f"int_{gid}_del2", gid, soft=False)

        fetched = await graph_store.get_intention(f"int_{gid}_del2", gid)
        assert fetched is None

    async def test_list_intentions_filters_expired(self, graph_store, gid):
        expired = Intention(
            id=f"int_{gid}_exp",
            trigger_text="old",
            action_text="old",
            group_id=gid,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        active = Intention(
            id=f"int_{gid}_act",
            trigger_text="new",
            action_text="new",
            group_id=gid,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        await graph_store.create_intention(expired)
        await graph_store.create_intention(active)

        enabled = await graph_store.list_intentions(gid, enabled_only=True)
        ids = [i.id for i in enabled]
        assert f"int_{gid}_exp" not in ids
        assert f"int_{gid}_act" in ids

    async def test_update_intention(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_upd",
            trigger_text="t",
            action_text="a",
            group_id=gid,
            threshold=0.7,
        )
        await graph_store.create_intention(intention)
        await graph_store.update_intention(
            f"int_{gid}_upd",
            {"threshold": 0.9, "enabled": False},
            gid,
        )
        fetched = await graph_store.get_intention(f"int_{gid}_upd", gid)
        assert fetched.threshold == 0.9
        assert fetched.enabled is False

    async def test_get_intention_wrong_group(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_grp",
            trigger_text="t",
            action_text="a",
            group_id=f"{gid}_a",
        )
        await graph_store.create_intention(intention)
        fetched = await graph_store.get_intention(f"int_{gid}_grp", f"{gid}_b")
        assert fetched is None


# ─── HelixDB storage tests ──────────────────────────────────────────────


@pytest.mark.requires_helix
@pytest.mark.asyncio
class TestHelixIntentionStorage:
    @pytest.fixture
    async def graph_store(self):
        from engram.config import HelixDBConfig
        from engram.storage.helix.graph import HelixGraphStore

        config = HelixDBConfig(host="localhost", port=6969)
        store = HelixGraphStore(config)
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def gid(self):
        return f"test_{uuid4().hex[:8]}"

    async def test_create_and_get_intention(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_1",
            trigger_text="Python upgrades",
            action_text="Migration plan exists",
            group_id=gid,
        )
        result_id = await graph_store.create_intention(intention)
        assert result_id == f"int_{gid}_1"

        fetched = await graph_store.get_intention(f"int_{gid}_1", gid)
        assert fetched is not None
        assert fetched.trigger_text == "Python upgrades"
        assert fetched.action_text == "Migration plan exists"
        assert fetched.enabled is True
        assert fetched.fire_count == 0

    async def test_list_intentions_enabled_only(self, graph_store, gid):
        i1 = Intention(
            id=f"int_{gid}_a",
            trigger_text="a",
            action_text="a",
            group_id=gid,
            enabled=True,
        )
        i2 = Intention(
            id=f"int_{gid}_b",
            trigger_text="b",
            action_text="b",
            group_id=gid,
            enabled=False,
        )
        await graph_store.create_intention(i1)
        await graph_store.create_intention(i2)

        enabled = await graph_store.list_intentions(gid, enabled_only=True)
        assert len(enabled) == 1
        assert enabled[0].id == f"int_{gid}_a"

        all_items = await graph_store.list_intentions(gid, enabled_only=False)
        assert len(all_items) == 2

    async def test_increment_fire_count(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_fire",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.increment_intention_fire_count(f"int_{gid}_fire", gid)
        await graph_store.increment_intention_fire_count(f"int_{gid}_fire", gid)

        fetched = await graph_store.get_intention(f"int_{gid}_fire", gid)
        assert fetched.fire_count == 2

    async def test_delete_intention_soft(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_del",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.delete_intention(f"int_{gid}_del", gid, soft=True)

        fetched = await graph_store.get_intention(f"int_{gid}_del", gid)
        assert fetched is not None
        assert fetched.enabled is False

    async def test_delete_intention_hard(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_del2",
            trigger_text="t",
            action_text="a",
            group_id=gid,
        )
        await graph_store.create_intention(intention)
        await graph_store.delete_intention(f"int_{gid}_del2", gid, soft=False)

        fetched = await graph_store.get_intention(f"int_{gid}_del2", gid)
        assert fetched is None

    async def test_list_intentions_filters_expired(self, graph_store, gid):
        expired = Intention(
            id=f"int_{gid}_exp",
            trigger_text="old",
            action_text="old",
            group_id=gid,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        active = Intention(
            id=f"int_{gid}_act",
            trigger_text="new",
            action_text="new",
            group_id=gid,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )
        await graph_store.create_intention(expired)
        await graph_store.create_intention(active)

        enabled = await graph_store.list_intentions(gid, enabled_only=True)
        ids = [i.id for i in enabled]
        assert f"int_{gid}_exp" not in ids
        assert f"int_{gid}_act" in ids

    async def test_update_intention(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_upd",
            trigger_text="t",
            action_text="a",
            group_id=gid,
            threshold=0.7,
        )
        await graph_store.create_intention(intention)
        await graph_store.update_intention(
            f"int_{gid}_upd",
            {"threshold": 0.9, "enabled": False},
            gid,
        )
        fetched = await graph_store.get_intention(f"int_{gid}_upd", gid)
        assert fetched.threshold == 0.9
        assert fetched.enabled is False

    async def test_get_intention_wrong_group(self, graph_store, gid):
        intention = Intention(
            id=f"int_{gid}_grp",
            trigger_text="t",
            action_text="a",
            group_id=f"{gid}_a",
        )
        await graph_store.create_intention(intention)
        fetched = await graph_store.get_intention(f"int_{gid}_grp", f"{gid}_b")
        assert fetched is None


# ─── Config tests ──────────────────────────────────────────────────────


class TestProspectiveConfig:
    def test_prospective_disabled_by_default(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig()
        assert cfg.prospective_memory_enabled is False

    def test_config_defaults_match_plan(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig()
        assert cfg.prospective_similarity_threshold == 0.7
        assert cfg.prospective_max_fires == 5
        assert cfg.prospective_ttl_days == 90
        assert cfg.prospective_max_per_episode == 3


class TestRecallProfile:
    def test_recall_profile_off(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig(recall_profile="off")
        assert cfg.auto_recall_enabled is False
        assert cfg.recall_usage_feedback_enabled is False
        assert cfg.conv_context_enabled is False
        assert cfg.surprise_detection_enabled is False
        assert cfg.prospective_memory_enabled is False

    def test_recall_profile_wave1(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig(recall_profile="wave1")
        assert cfg.auto_recall_enabled is True
        assert cfg.auto_recall_on_observe is True
        assert cfg.auto_recall_on_remember is True
        assert cfg.auto_recall_session_prime is True
        assert cfg.recall_need_analyzer_enabled is True
        assert cfg.recall_need_structural_enabled is True
        assert cfg.recall_need_graph_probe_enabled is False
        assert cfg.recall_usage_feedback_enabled is True
        # Wave 2+ should be off
        assert cfg.conv_context_enabled is False
        assert cfg.surprise_detection_enabled is False
        assert cfg.prospective_memory_enabled is False

    def test_recall_profile_wave2_cumulative(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig(recall_profile="wave2")
        # Wave 1
        assert cfg.auto_recall_enabled is True
        assert cfg.auto_recall_on_observe is True
        assert cfg.recall_usage_feedback_enabled is True
        # Wave 2
        assert cfg.conv_context_enabled is True
        assert cfg.conv_fingerprint_enabled is True
        assert cfg.conv_multi_query_enabled is True
        assert cfg.conv_session_entity_seeds_enabled is True
        assert cfg.conv_near_miss_enabled is True
        assert cfg.recall_need_structural_enabled is True
        assert cfg.recall_need_graph_probe_enabled is True
        assert cfg.recall_need_shift_enabled is False
        # Wave 3+ should be off
        assert cfg.conv_topic_shift_enabled is False
        assert cfg.surprise_detection_enabled is False
        assert cfg.prospective_memory_enabled is False

    def test_recall_profile_wave3_cumulative(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig(recall_profile="wave3")
        # Wave 1
        assert cfg.auto_recall_enabled is True
        assert cfg.recall_usage_feedback_enabled is True
        # Wave 2
        assert cfg.conv_context_enabled is True
        assert cfg.conv_near_miss_enabled is True
        # Wave 3
        assert cfg.conv_topic_shift_enabled is True
        assert cfg.surprise_detection_enabled is True
        assert cfg.retrieval_priming_enabled is True
        assert cfg.gc_mmr_enabled is True
        assert cfg.recall_need_graph_probe_enabled is True
        assert cfg.recall_need_shift_enabled is True
        assert cfg.recall_need_impoverishment_enabled is True
        assert cfg.recall_need_shift_shadow_only is False
        assert cfg.recall_need_impoverishment_shadow_only is False
        # Wave 4 should be off
        assert cfg.prospective_memory_enabled is False

    def test_recall_profile_all_enables_every_recall_wave(self):
        from engram.config import ActivationConfig

        cfg = ActivationConfig(recall_profile="all")
        # Wave 1
        assert cfg.auto_recall_enabled is True
        assert cfg.auto_recall_on_observe is True
        assert cfg.auto_recall_on_remember is True
        assert cfg.auto_recall_session_prime is True
        assert cfg.recall_usage_feedback_enabled is True
        # Wave 2
        assert cfg.conv_context_enabled is True
        assert cfg.conv_fingerprint_enabled is True
        assert cfg.conv_multi_query_enabled is True
        assert cfg.conv_session_entity_seeds_enabled is True
        assert cfg.conv_near_miss_enabled is True
        # Wave 3
        assert cfg.conv_topic_shift_enabled is True
        assert cfg.surprise_detection_enabled is True
        assert cfg.retrieval_priming_enabled is True
        assert cfg.gc_mmr_enabled is True
        assert cfg.recall_need_structural_enabled is True
        assert cfg.recall_need_graph_probe_enabled is True
        assert cfg.recall_need_shift_enabled is True
        assert cfg.recall_need_impoverishment_enabled is True
        # Wave 4
        assert cfg.prospective_memory_enabled is True
        # Cue/projection rollout remains a separate integration profile.
        assert cfg.cue_layer_enabled is False
        assert cfg.cue_recall_enabled is False

    def test_recall_profile_all_same_as_wave4(self):
        from engram.config import ActivationConfig

        cfg_all = ActivationConfig(recall_profile="all")
        cfg_w4 = ActivationConfig(recall_profile="wave4")
        flags = [
            "auto_recall_enabled",
            "auto_recall_on_observe",
            "auto_recall_on_remember",
            "auto_recall_session_prime",
            "recall_usage_feedback_enabled",
            "recall_need_analyzer_enabled",
            "recall_need_structural_enabled",
            "recall_need_graph_probe_enabled",
            "recall_need_shift_enabled",
            "recall_need_impoverishment_enabled",
            "recall_need_shift_shadow_only",
            "recall_need_impoverishment_shadow_only",
            "conv_context_enabled",
            "conv_fingerprint_enabled",
            "conv_multi_query_enabled",
            "conv_session_entity_seeds_enabled",
            "conv_near_miss_enabled",
            "conv_topic_shift_enabled",
            "surprise_detection_enabled",
            "retrieval_priming_enabled",
            "gc_mmr_enabled",
            "prospective_memory_enabled",
        ]
        for flag in flags:
            assert getattr(cfg_all, flag) == getattr(cfg_w4, flag), f"{flag} mismatch"


# ─── GraphManager integration tests ──────────────────────────────────


@pytest.mark.asyncio
class TestGraphManagerIntegration:
    async def test_create_intention_generates_id(self, tmp_path):
        from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.extraction.extractor import EntityExtractor
        from engram.graph_manager import GraphManager
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex
        from engram.storage.memory.activation import MemoryActivationStore
        graph = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph.initialize()
        activation = MemoryActivationStore(cfg=ActivationConfig())
        search = HelixSearchIndex(
     helix_config=HelixDBConfig(host="localhost", port=6969),
     provider=NoopProvider(),
     embed_config=EmbeddingConfig(),
     storage_dim=0,
     embed_provider="noop",
     embed_model="noop",
 )
        await search.initialize()
        cfg = ActivationConfig(prospective_memory_enabled=True, prospective_graph_embedded=False)
        extractor = EntityExtractor()

        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)
        gid = f"test_{uuid4().hex[:8]}"
        intention_id = await gm.create_intention(
            trigger_text="Python upgrades",
            action_text="User has a migration plan",
            group_id=gid,
        )
        assert intention_id.startswith("int_")
        assert len(intention_id) > 4

        intentions = await gm.list_intentions(gid)
        assert len(intentions) == 1
        assert intentions[0].trigger_text == "Python upgrades"

        await graph.close()

    async def test_create_intention_entity_mention(self, tmp_path):
        from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.extraction.extractor import EntityExtractor
        from engram.graph_manager import GraphManager
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex
        from engram.storage.memory.activation import MemoryActivationStore
        graph = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph.initialize()
        activation = MemoryActivationStore(cfg=ActivationConfig())
        search = HelixSearchIndex(
     helix_config=HelixDBConfig(host="localhost", port=6969),
     provider=NoopProvider(),
     embed_config=EmbeddingConfig(),
     storage_dim=0,
     embed_provider="noop",
     embed_model="noop",
 )
        await search.initialize()
        cfg = ActivationConfig(prospective_memory_enabled=True, prospective_graph_embedded=False)
        extractor = EntityExtractor()

        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)

        # entity_mention requires entity_name
        with pytest.raises(ValueError, match="entity_name required"):
            await gm.create_intention(
                trigger_text="test",
                action_text="action",
                trigger_type="entity_mention",
            )

        # valid entity_mention
        intent_id = await gm.create_intention(
            trigger_text="Python mentioned",
            action_text="Check migration",
            trigger_type="entity_mention",
            entity_name="Python",
        )
        assert intent_id.startswith("int_")

        await graph.close()

    async def test_delete_intention(self, tmp_path):
        from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
        from engram.embeddings.provider import NoopProvider
        from engram.extraction.extractor import EntityExtractor
        from engram.graph_manager import GraphManager
        from engram.storage.helix.graph import HelixGraphStore
        from engram.storage.helix.search import HelixSearchIndex
        from engram.storage.memory.activation import MemoryActivationStore
        graph = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
        await graph.initialize()
        activation = MemoryActivationStore(cfg=ActivationConfig())
        search = HelixSearchIndex(
     helix_config=HelixDBConfig(host="localhost", port=6969),
     provider=NoopProvider(),
     embed_config=EmbeddingConfig(),
     storage_dim=0,
     embed_provider="noop",
     embed_model="noop",
 )
        await search.initialize()
        cfg = ActivationConfig(prospective_memory_enabled=True, prospective_graph_embedded=False)
        extractor = EntityExtractor()

        gm = GraphManager(graph, activation, search, extractor, cfg=cfg)
        gid = f"test_{uuid4().hex[:8]}"
        intent_id = await gm.create_intention(
            trigger_text="test",
            action_text="action",
            group_id=gid,
        )
        await gm.delete_intention(intent_id, group_id=gid)

        intentions = await gm.list_intentions(gid, enabled_only=True)
        assert len(intentions) == 0

        await graph.close()
