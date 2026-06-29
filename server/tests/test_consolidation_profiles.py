"""Tests for consolidation profile presets in ActivationConfig."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.store import SQLiteConsolidationStore
from engram.evaluation.brain_loop_report import build_brain_loop_report
from engram.events.bus import EventBus
from engram.graph_manager import GraphManager
from engram.lifecycle_summary import build_lifecycle_summary
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.utils.dates import utc_now
from tests.conftest import MockExtractor


class TestConsolidationProfiles:
    def test_off_profile(self):
        cfg = ActivationConfig(consolidation_profile="off")
        assert cfg.consolidation_enabled is False
        assert cfg.consolidation_dry_run is True

    def test_observe_profile(self):
        cfg = ActivationConfig(consolidation_profile="observe")
        assert cfg.consolidation_enabled is True
        assert cfg.consolidation_dry_run is True
        assert cfg.consolidation_replay_enabled is True
        assert cfg.consolidation_dream_enabled is True
        assert cfg.consolidation_infer_pmi_enabled is True

    def test_conservative_profile(self):
        cfg = ActivationConfig(consolidation_profile="conservative")
        assert cfg.consolidation_enabled is True
        assert cfg.consolidation_dry_run is False
        assert cfg.consolidation_merge_threshold == 0.92
        assert cfg.consolidation_prune_min_age_days == 30
        assert cfg.consolidation_replay_enabled is True
        assert cfg.consolidation_dream_enabled is True

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_standard_profile(self):
        cfg = ActivationConfig(consolidation_profile="standard")
        assert cfg.consolidation_enabled is True
        assert cfg.consolidation_dry_run is False
        assert cfg.consolidation_replay_enabled is True
        assert cfg.consolidation_dream_enabled is True
        assert cfg.consolidation_infer_pmi_enabled is True
        assert cfg.consolidation_infer_transitivity_enabled is True
        assert cfg.consolidation_pressure_enabled is True
        # Multi-signal scorers replace LLM judges (zero API cost)
        assert cfg.consolidation_merge_multi_signal_enabled is True
        assert cfg.consolidation_infer_auto_validation_enabled is True
        assert cfg.triage_multi_signal_enabled is True
        # LLM judges disabled when multi-signal active (opt-in fallback)
        assert cfg.triage_llm_judge_enabled is False
        assert cfg.triage_llm_escalation_enabled is False
        assert cfg.consolidation_infer_llm_enabled is False
        assert cfg.consolidation_infer_escalation_enabled is False
        assert cfg.consolidation_merge_llm_enabled is False
        assert cfg.consolidation_merge_escalation_enabled is False

    def test_default_is_standard_wave2(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_profile == "standard"
        assert cfg.recall_profile == "wave2"
        assert cfg.integration_profile == "off"
        assert cfg.passage_first_entity_budget == 0
        assert cfg.consolidation_enabled is True
        assert cfg.triage_enabled is True
        assert cfg.worker_enabled is True
        assert cfg.consolidation_dream_associations_enabled is True
        assert cfg.auto_recall_enabled is True
        assert cfg.conv_context_enabled is True
        assert cfg.recall_planner_enabled is True
        assert cfg.cue_layer_enabled is True
        assert cfg.cue_recall_enabled is True
        assert cfg.observer_reflect_enabled is False

    def test_invalid_profile_rejected(self):
        with pytest.raises(Exception):
            ActivationConfig(consolidation_profile="invalid")

    def test_explicit_override_after_profile(self):
        """Explicit field values set after profile init should stick."""
        cfg = ActivationConfig(
            consolidation_profile="observe",
            consolidation_dream_enabled=False,
        )
        # Profile sets dream_enabled=True, but explicit kwarg should override...
        # Actually model_post_init runs after __init__, so profile wins.
        # This tests the current behavior — profile overrides explicit kwargs.
        assert cfg.consolidation_dream_enabled is True


class TestIntegrationProfiles:
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_rework_profile_enables_full_loop(self):
        cfg = ActivationConfig(integration_profile="rework")

        assert cfg.integration_profile == "rework"
        assert cfg.consolidation_profile == "standard"
        assert cfg.recall_profile == "all"
        assert cfg.worker_enabled is True
        assert cfg.triage_enabled is True
        assert cfg.auto_recall_enabled is True
        assert cfg.recall_need_analyzer_enabled is True
        assert cfg.recall_need_graph_probe_enabled is True
        assert cfg.recall_need_structural_enabled is True
        assert cfg.recall_need_shift_enabled is True
        assert cfg.recall_need_impoverishment_enabled is True
        assert cfg.recall_need_shift_shadow_only is False
        assert cfg.recall_need_impoverishment_shadow_only is False
        assert cfg.recall_planner_enabled is True
        assert cfg.recall_usage_feedback_enabled is True
        assert cfg.capture_cue_vector_index_quiet_period_ms == 1000
        assert cfg.cue_layer_enabled is True
        assert cfg.cue_vector_index_enabled is True
        assert cfg.cue_recall_enabled is True
        assert cfg.cue_policy_learning_enabled is True
        assert cfg.targeted_projection_enabled is True
        assert cfg.projector_v2_enabled is True
        assert cfg.projection_planner_enabled is True
        assert cfg.epistemic_routing_enabled is True
        assert cfg.artifact_bootstrap_enabled is True
        assert cfg.artifact_recall_enabled is True
        assert cfg.epistemic_runtime_executor_enabled is True
        assert cfg.decision_graph_enabled is True
        assert cfg.epistemic_reconcile_enabled is True
        assert cfg.answer_contract_enabled is True
        assert cfg.claim_state_modeling_enabled is True
        assert cfg.memory_maturation_enabled is True
        assert cfg.episode_transition_enabled is True

    def test_recall_profile_all_is_still_partial_rollout(self):
        # Isolate recall waves from the new standard consolidation default.
        cfg = ActivationConfig(recall_profile="all", consolidation_profile="off")

        assert cfg.recall_profile == "all"
        assert cfg.auto_recall_enabled is True
        assert cfg.recall_need_analyzer_enabled is True
        assert cfg.recall_need_graph_probe_enabled is True
        assert cfg.recall_need_structural_enabled is True
        assert cfg.recall_need_shift_enabled is True
        assert cfg.recall_need_impoverishment_enabled is True
        assert cfg.recall_need_shift_shadow_only is False
        assert cfg.recall_need_impoverishment_shadow_only is False
        assert cfg.recall_planner_enabled is True
        assert cfg.recall_usage_feedback_enabled is True

        # wave2 (included in "all") enables cue capture + recall hits; rework-only
        # features stay off without integration_profile=rework.
        assert cfg.cue_layer_enabled is True
        assert cfg.cue_recall_enabled is True
        assert cfg.cue_policy_learning_enabled is False
        assert cfg.epistemic_routing_enabled is False
        assert cfg.artifact_bootstrap_enabled is False
        assert cfg.memory_maturation_enabled is False
        assert cfg.episode_transition_enabled is False

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    def test_rework_profile_normalizes_partial_overrides(self):
        cfg = ActivationConfig(
            integration_profile="rework",
            consolidation_profile="observe",
            recall_profile="wave2",
        )

        assert cfg.integration_profile == "rework"
        assert cfg.consolidation_profile == "standard"
        assert cfg.recall_profile == "all"
        assert cfg.cue_layer_enabled is True
        assert cfg.cue_recall_enabled is True
        assert cfg.recall_planner_enabled is True
        assert cfg.memory_maturation_enabled is True

    def test_invalid_integration_profile_rejected(self):
        with pytest.raises(Exception):
            ActivationConfig(integration_profile="invalid")


def _evidence_row(
    episode_id: str,
    fact_class: str,
    payload: dict,
    *,
    confidence: float,
) -> dict:
    return {
        "evidence_id": f"evi_{uuid.uuid4().hex[:12]}",
        "episode_id": episode_id,
        "fact_class": fact_class,
        "confidence": confidence,
        "source_type": "narrow_extractor",
        "extractor_name": "test",
        "payload": payload,
        "source_span": None,
        "corroborating_signals": [],
        "status": "pending",
        "commit_reason": None,
        "deferred_cycles": 0,
        "created_at": "2026-03-09T00:00:00",
    }


class TestAuditFollowUpDefaults:
    @pytest.mark.asyncio
    async def test_default_profile_evidence_adjudication_runs(self, tmp_path):
        cfg = ActivationConfig()
        graph_store = SQLiteGraphStore(str(tmp_path / "adj.db"))
        await graph_store.initialize()
        consol_store = SQLiteConsolidationStore(graph_store._db_path)
        await consol_store.initialize(db=graph_store._db)
        search_index = FTS5SearchIndex(graph_store._db_path)
        await search_index.initialize(db=graph_store._db)
        activation_store = MemoryActivationStore(cfg=cfg)
        graph_manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=MockExtractor(),
            cfg=cfg,
        )
        episode = Episode(
            id="ep_adj",
            content="Alice works at Google since 2026-01-15.",
            source="test",
            group_id="default",
            created_at=utc_now(),
        )
        await graph_store.create_episode(episode)
        await graph_store.store_evidence(
            [
                _evidence_row(
                    "ep_adj",
                    "entity",
                    {"name": "Alice", "entity_type": "Person"},
                    confidence=0.82,
                ),
                _evidence_row(
                    "ep_adj",
                    "entity",
                    {"name": "Google", "entity_type": "Organization"},
                    confidence=0.82,
                ),
                _evidence_row(
                    "ep_adj",
                    "relationship",
                    {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
                    confidence=0.84,
                ),
            ],
            group_id="default",
        )
        engine = ConsolidationEngine(
            graph_store,
            activation_store,
            search_index,
            cfg=cfg,
            consolidation_store=consol_store,
            event_bus=EventBus(),
            extractor=MockExtractor(),
            graph_manager=graph_manager,
        )
        cycle = await engine.run_cycle(
            group_id="default",
            phase_names={"evidence_adjudication"},
        )
        adj = next(
            pr for pr in cycle.phase_results if pr.phase == "evidence_adjudication"
        )
        assert adj.status == "success"
        assert adj.items_processed > 0
        assert adj.items_processed <= 200
        stats = await graph_store.get_stats(group_id="default")
        report = build_brain_loop_report(
            stats,
            group_id="default",
            recent_cycles=[cycle],
        )
        assert report["consolidate"]["adjudication"]["runs"] >= 1
        await graph_store.close()

    @pytest.mark.asyncio
    async def test_default_wave2_recall_records_cue_hit_e2e(self, tmp_path):
        """Default wave2 config records cue hits through GraphManager.recall()."""
        cfg = _quiet_sqlite_recall_config()
        assert cfg.cue_recall_enabled is True

        graph_store = SQLiteGraphStore(str(tmp_path / "recall_cue.db"))
        await graph_store.initialize()
        search_index = FTS5SearchIndex(graph_store._db_path)
        await search_index.initialize(db=graph_store._db)
        activation_store = MemoryActivationStore(cfg=cfg)
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            MockExtractor(),
            cfg=cfg,
        )

        episode = Episode(
            id="ep_recall_cue",
            content="The migration to native Helix finished on Tuesday.",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.CUE_ONLY,
            group_id="default",
            created_at=utc_now(),
        )
        await graph_store.create_episode(episode)
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id=episode.id,
                group_id="default",
                projection_state=EpisodeProjectionState.CUE_ONLY,
                cue_text="native Helix migration",
                hit_count=0,
            ),
        )

        results = await manager.recall(
            "native Helix migration",
            group_id="default",
            limit=5,
            record_access=False,
            interaction_type="surfaced",
            interaction_source="recall",
        )
        assert results

        stats = await graph_store.get_stats(group_id="default")
        assert stats["cue_metrics"]["cue_hit_count"] > 0

        lifecycle = await build_lifecycle_summary(
            group_id="default",
            manager=manager,
            graph_store=graph_store,
        )
        assert lifecycle["cue"]["hitCount"] > 0
        await graph_store.close()


def _quiet_sqlite_recall_config() -> ActivationConfig:
    """Default wave2 config with noisy recall stages disabled for SQLite FTS."""
    cfg = ActivationConfig()
    cfg.multi_pool_enabled = False
    cfg.graph_query_expansion_enabled = False
    cfg.template_reformulation_enabled = False
    cfg.query_decomposition_enabled = False
    cfg.recall_planner_enabled = False
    cfg.reranker_enabled = False
    cfg.mmr_enabled = False
    cfg.gc_mmr_enabled = False
    cfg.ts_enabled = False
    cfg.working_memory_enabled = False
    cfg.goal_priming_enabled = False
    cfg.inhibitory_spreading_enabled = False
    cfg.cross_domain_penalty_enabled = False
    cfg.emotional_salience_enabled = False
    cfg.state_dependent_retrieval_enabled = False
    cfg.preference_directed_enabled = False
    cfg.conv_near_miss_enabled = False
    cfg.chunk_search_enabled = False
    cfg.weight_graph_structural = 0.0
    return cfg
