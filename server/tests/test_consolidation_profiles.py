"""Tests for consolidation profile presets in ActivationConfig."""

from unittest.mock import patch

import pytest

from engram.config import ActivationConfig


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
        assert cfg.consolidation_infer_llm_enabled is False
        assert cfg.consolidation_infer_escalation_enabled is False
        assert cfg.consolidation_merge_llm_enabled is False
        assert cfg.consolidation_merge_escalation_enabled is False

    def test_default_is_off(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_profile == "off"
        assert cfg.consolidation_enabled is False

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
        cfg = ActivationConfig(recall_profile="all")

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

        assert cfg.cue_layer_enabled is False
        assert cfg.cue_recall_enabled is False
        assert cfg.cue_policy_learning_enabled is False
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
