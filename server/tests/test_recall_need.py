"""Tests for memory-need analysis and graph grounding."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.recall import MemoryNeed
from engram.models.relationship import Relationship
from engram.retrieval.context import ConversationContext
from engram.retrieval.control import RecallNeedController
from engram.retrieval.graph_probe import GraphProbe, ProbeResult
from engram.retrieval.need import analyze_memory_need
from engram.retrieval.signals import SignalReport, SignalResult, extract_signals


def _cfg(**overrides) -> ActivationConfig:
    defaults = {
        "recall_need_graph_probe_enabled": False,
        "recall_need_structural_enabled": False,
        "recall_need_shift_enabled": False,
        "recall_need_impoverishment_enabled": False,
        "recall_need_shift_shadow_only": True,
        "recall_need_impoverishment_shadow_only": True,
        "recall_need_graph_override_enabled": False,
        "recall_need_adaptive_thresholds_enabled": False,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _make_graph_probe() -> GraphProbe:
    now = time.time()
    user = Entity(
        id="ent_user",
        name="Alex",
        entity_type="Person",
        summary="User identity",
        group_id="default",
        identity_core=True,
    )
    son = Entity(
        id="ent_ben",
        name="Ben",
        entity_type="Person",
        summary="Alex's son",
        group_id="default",
    )
    will = Entity(
        id="ent_will",
        name="Will",
        entity_type="Person",
        summary="Friend from soccer",
        group_id="default",
    )
    migration = Entity(
        id="ent_auth",
        name="Auth Migration",
        entity_type="Project",
        summary="Authentication rollout",
        group_id="default",
    )

    graph_store = AsyncMock()
    graph_store.get_stats.return_value = {"entities": 4, "relationships": 4}

    async def find_entity_candidates(name, group_id, limit=30):
        key = name.lower()
        if key in {"will", "ben"}:
            return [will if key == "will" else son]
        if key in {"auth migration", "auth", "migration"}:
            return [migration]
        return []

    async def get_relationships(
        entity_id, direction="both", predicate=None, active_only=True, group_id="default"
    ):
        if entity_id == "ent_user":
            return [
                Relationship(
                    id="rel_parent",
                    source_id="ent_user",
                    target_id="ent_ben",
                    predicate="PARENT_OF",
                    group_id=group_id,
                )
            ]
        if entity_id == "ent_will":
            return [
                Relationship(
                    id="rel_friend_1",
                    source_id="ent_will",
                    target_id="ent_auth",
                    predicate="KNOWS",
                    group_id=group_id,
                ),
                Relationship(
                    id="rel_friend_2",
                    source_id="ent_will",
                    target_id="ent_user",
                    predicate="KNOWS",
                    group_id=group_id,
                ),
            ]
        if entity_id == "ent_auth":
            return [
                Relationship(
                    id="rel_auth_1",
                    source_id="ent_auth",
                    target_id="ent_user",
                    predicate="MENTIONED_WITH",
                    group_id=group_id,
                )
            ]
        return []

    async def get_entity(entity_id, group_id):
        lookup = {
            "ent_user": user,
            "ent_ben": son,
            "ent_will": will,
            "ent_auth": migration,
        }
        return lookup.get(entity_id)

    graph_store.find_entity_candidates.side_effect = find_entity_candidates
    graph_store.get_identity_core_entities.return_value = [user]
    graph_store.get_relationships.side_effect = get_relationships
    graph_store.get_entity.side_effect = get_entity

    activation_store = AsyncMock()
    activation_store.batch_get.return_value = {
        "ent_ben": ActivationState(
            node_id="ent_ben",
            access_count=4,
            last_accessed=now - 120.0,
        ),
        "ent_will": ActivationState(
            node_id="ent_will",
            access_count=5,
            last_accessed=now - 60.0,
        ),
        "ent_auth": ActivationState(
            node_id="ent_auth",
            access_count=3,
            last_accessed=now - 300.0,
        ),
    }
    return GraphProbe(graph_store, activation_store)


class TestRecallNeedTelemetry:
    def test_signal_scores_populated_when_signals_exist(self):
        report = SignalReport(
            pragmatic_score=0.35,
            dominant_family="pragmatic",
            signals=[SignalResult(name="P1_possessive_relational", score=0.40)],
        )
        scores = report.to_scores_dict()
        assert scores["pragmatic"] == 0.35
        assert scores["P1_possessive_relational"] == 0.40

    def test_new_fields_default_values(self):
        need = MemoryNeed(need_type="none", should_recall=False, confidence=0.8)
        assert need.signal_scores is None
        assert need.trigger_family is None
        assert need.trigger_kind is None
        assert need.detected_entities is None
        assert need.detected_referents is None
        assert need.resonance_score == 0.0
        assert need.decision_path is None
        assert need.thresholds is None
        assert need.analyzer_latency_ms == 0.0
        assert need.probe_triggered is False
        assert need.probe_latency_ms == 0.0
        assert need.graph_override_used is False

    def test_to_payload_includes_signal_metadata(self):
        need = MemoryNeed(
            need_type="fact_lookup",
            should_recall=True,
            confidence=0.75,
            signal_scores={"pragmatic": 0.35, "P1_possessive_relational": 0.40},
            trigger_family="pragmatic",
            trigger_kind="possessive_relational",
            detected_entities=["ent_ben"],
            detected_referents=["son"],
            resonance_score=0.6,
            decision_path="graph_lift",
            thresholds={"linguistic": 0.3, "borderline": 0.15, "resonance": 0.45},
            analyzer_latency_ms=2.4,
            probe_triggered=True,
            probe_latency_ms=1.3,
        )
        payload = need.to_payload(source="mcp", mode="auto_recall", turn_preview="my son")
        assert payload["signalScores"]["pragmatic"] == 0.35
        assert payload["triggerFamily"] == "pragmatic"
        assert payload["triggerKind"] == "possessive_relational"
        assert payload["detectedEntities"] == ["ent_ben"]
        assert payload["detectedReferents"] == ["son"]
        assert payload["resonanceScore"] == 0.6
        assert payload["decisionPath"] == "graph_lift"
        assert payload["thresholds"]["linguistic"] == 0.3
        assert payload["analyzerLatencyMs"] == 2.4
        assert payload["probeTriggered"] is True
        assert payload["probeLatencyMs"] == 1.3

    def test_to_payload_omits_empty_signal_metadata(self):
        need = MemoryNeed(need_type="none", should_recall=False, confidence=0.8)
        payload = need.to_payload(source="mcp", mode="auto_recall", turn_preview="hi")
        assert "signalScores" not in payload
        assert "triggerFamily" not in payload
        assert "triggerKind" not in payload
        assert "detectedEntities" not in payload
        assert "detectedReferents" not in payload
        assert "resonanceScore" not in payload


class TestSignalExtraction:
    def test_extract_signals_returns_possessive_relational_report(self):
        report = extract_signals("my son had a great game", "my son had a great game")
        assert report.pragmatic_score > 0.0
        assert report.linguistic_score == report.pragmatic_score
        assert report.dominant_family == "pragmatic"
        assert report.dominant_trigger_kind == "possessive_relational"
        assert "son" in report.all_referents

    def test_extract_signals_returns_bare_name_report(self):
        report = extract_signals("Emma scored two goals", "emma scored two goals")
        assert report.dominant_trigger_kind == "bare_name"
        assert report.best_query_hint == "Emma"

    def test_extract_signals_skips_technical_phrase(self):
        report = extract_signals("my PR needs review", "my pr needs review")
        assert report.pragmatic_score == 0.0

    def test_extract_signals_structural_disabled_by_default(self):
        report = extract_signals(
            "Actually, it's not PostgreSQL, it's MySQL",
            "actually, it's not postgresql, it's mysql",
        )
        assert report.structural_score == 0.0


@pytest.mark.asyncio
class TestGraphProbe:
    async def test_probe_resolves_relational_reference(self):
        probe = _make_graph_probe()

        result = await probe.probe(
            "my son had a great game",
            "my son had a great game",
            referents=["son"],
            group_id="default",
        )

        assert result.resonance_score > 0.45
        assert "ent_ben" in result.detected_entities
        assert result.entity_scores["ent_ben"] > 0.5


@pytest.mark.asyncio
class TestRecallNeedAnalyzer:
    async def test_acknowledgement_skips_recall(self):
        need = await analyze_memory_need("thanks", mode="auto_recall", cfg=_cfg())
        assert need.need_type == "none"
        assert need.should_recall is False

    async def test_keyword_paths_still_win(self):
        need = await analyze_memory_need(
            "Did we decide on that yet?",
            recent_turns=["We were debating Redis for the cache layer."],
            session_entity_names=["Redis"],
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "open_loop"
        assert need.should_recall is True
        assert need.trigger_family == "keyword"
        assert need.trigger_kind == "open_loop"

    async def test_project_follow_up_triggers_project_state(self):
        need = await analyze_memory_need(
            "How's the auth migration going?",
            recent_turns=["We were fixing auth edge cases yesterday."],
            session_entity_names=["Auth Migration"],
            mode="chat",
            cfg=_cfg(),
        )
        assert need.need_type == "project_state"
        assert need.should_recall is True
        assert need.query_hint is not None

    async def test_phase1_pragmatic_signal_triggers_recall(self):
        need = await analyze_memory_need(
            "my son had a great game",
            mode="auto_recall",
            cfg=_cfg(),
        )
        assert need.need_type == "fact_lookup"
        assert need.should_recall is True
        assert need.trigger_family == "pragmatic"
        assert need.trigger_kind == "possessive_relational"
        assert need.detected_referents == ["son"]

    async def test_graph_probe_lifts_borderline_linguistic_hit(self):
        probe = _make_graph_probe()

        need = await analyze_memory_need(
            "Will scored",
            mode="auto_recall",
            graph_probe=probe,
            group_id="default",
            cfg=_cfg(recall_need_graph_probe_enabled=True),
        )

        assert need.should_recall is True
        assert need.detected_entities == ["ent_will"]
        assert need.resonance_score >= 0.45

    async def test_zero_signal_turn_does_not_probe_or_recall(self):
        probe = AsyncMock()
        probe.probe = AsyncMock(
            return_value=ProbeResult(resonance_score=0.9, detected_entities=["ent_auth"])
        )

        need = await analyze_memory_need(
            "deploy to staging after build",
            mode="auto_recall",
            graph_probe=probe,
            cfg=_cfg(recall_need_graph_probe_enabled=True),
        )

        assert need.should_recall is False
        probe.probe.assert_not_awaited()

    async def test_structural_callback_maps_to_open_loop(self):
        need = await analyze_memory_need(
            "remember when we talked about Sarah?",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "open_loop"
        assert need.should_recall is True
        assert need.trigger_family == "structural"
        assert need.trigger_kind == "callback"

    async def test_structural_memory_gap_maps_to_open_loop_when_decision_language_present(self):
        need = await analyze_memory_need(
            "I can't remember if we decided on Postgres",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "open_loop"
        assert need.trigger_kind == "memory_gap"

    async def test_structural_correction_maps_to_temporal_update(self):
        need = await analyze_memory_need(
            "Actually, it's not PostgreSQL, it's MySQL",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "temporal_update"
        assert need.trigger_kind == "correction"

    async def test_structural_life_update_maps_to_temporal_update(self):
        need = await analyze_memory_need(
            "we moved to Austin",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "temporal_update"
        assert need.trigger_kind == "life_update"

    async def test_structural_life_update_maps_to_project_state_for_project_turn(self):
        need = await analyze_memory_need(
            "we finally shipped the auth migration",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "project_state"
        assert need.trigger_kind in {"project_state", "life_update", "milestone"}

    async def test_structural_identity_claim_maps_to_identity(self):
        need = await analyze_memory_need(
            "I'm more of a backend person",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "identity"
        assert need.trigger_kind == "identity_claim"

    async def test_structural_status_check_maps_to_project_state(self):
        need = await analyze_memory_need(
            "Did that land?",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "project_state"
        assert need.trigger_kind == "status_check"

    async def test_structural_delegation_maps_to_project_state(self):
        need = await analyze_memory_need(
            "Sarah sent me the final deck",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "project_state"
        assert need.trigger_kind == "delegation"

    async def test_structural_emotional_anchor_maps_to_broad_context(self):
        need = await analyze_memory_need(
            "Really worried about Ben lately",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.need_type == "broad_context"
        assert need.trigger_kind == "emotional_anchor"

    async def test_mixed_pragmatic_and_structural_prefers_higher_scoring_family(self):
        need = await analyze_memory_need(
            "Emma moved to Austin",
            mode="chat",
            cfg=_cfg(recall_need_structural_enabled=True),
        )
        assert need.should_recall is True
        assert need.trigger_family == "structural"
        assert need.trigger_kind == "life_update"
        assert need.signal_scores is not None
        assert need.signal_scores["pragmatic"] > 0.0
        assert need.signal_scores["structural"] > 0.0

    async def test_shift_and_impoverishment_shadow_only_do_not_change_decision(self):
        ctx = ConversationContext()
        ctx.add_turn("Working on the auth migration")
        ctx.add_turn("Need to fix the Redis cache layer")

        need = await analyze_memory_need(
            "by the way, really big news",
            mode="chat",
            conv_context=ctx,
            cfg=_cfg(
                recall_need_shift_enabled=True,
                recall_need_impoverishment_enabled=True,
                recall_need_shift_shadow_only=True,
                recall_need_impoverishment_shadow_only=True,
            ),
        )

        assert need.should_recall is False
        assert need.signal_scores is not None
        assert need.signal_scores["shift"] > 0.0
        assert need.signal_scores["impoverishment"] > 0.0

    async def test_live_impoverishment_can_trigger_recall_after_shadow_rollout(self):
        ctx = ConversationContext()
        ctx.add_turn("Working on the auth migration")
        ctx.add_turn("Need to fix the Redis cache layer")

        need = await analyze_memory_need(
            "by the way, really big news",
            mode="chat",
            conv_context=ctx,
            cfg=_cfg(
                recall_need_shift_enabled=True,
                recall_need_impoverishment_enabled=True,
                recall_need_shift_shadow_only=False,
                recall_need_impoverishment_shadow_only=False,
            ),
        )

        assert need.should_recall is True
        assert need.trigger_family in {"shift", "impoverishment"}

    async def test_greeting_and_generic_command_stay_below_threshold(self):
        ctx = ConversationContext()
        ctx.add_turn("Working on the auth migration")
        ctx.add_turn("Need to fix the Redis cache layer")

        greeting = await analyze_memory_need(
            "hey",
            mode="chat",
            conv_context=ctx,
            cfg=_cfg(
                recall_need_shift_enabled=True,
                recall_need_impoverishment_enabled=True,
                recall_need_shift_shadow_only=False,
                recall_need_impoverishment_shadow_only=False,
            ),
        )
        command = await analyze_memory_need(
            "Can you write a for loop?",
            mode="chat",
            conv_context=ctx,
            cfg=_cfg(
                recall_need_shift_enabled=True,
                recall_need_impoverishment_enabled=True,
                recall_need_shift_shadow_only=False,
                recall_need_impoverishment_shadow_only=False,
            ),
        )

        assert greeting.should_recall is False
        assert command.should_recall is False

    async def test_dampening_blocks_technical_temporal_over_triggering(self):
        need = await analyze_memory_need(
            "please deploy after the build completes",
            mode="chat",
            cfg=_cfg(
                recall_need_impoverishment_enabled=True,
                recall_need_impoverishment_shadow_only=False,
            ),
        )
        report = extract_signals(
            "please deploy after the build completes",
            "please deploy after the build completes",
            cfg=_cfg(
                recall_need_impoverishment_enabled=True,
                recall_need_impoverishment_shadow_only=False,
            ),
        )

        assert report.dampening_factor < 1.0
        assert need.should_recall is False

    async def test_graph_override_stays_off_by_default(self):
        low_signal = SignalReport(
            pragmatic_score=0.05,
            linguistic_score=0.05,
            dominant_family="pragmatic",
            dominant_trigger_kind="bare_name",
        )
        probe = AsyncMock()
        probe.probe = AsyncMock(
            return_value=ProbeResult(
                resonance_score=0.82,
                detected_entities=["ent_ben"],
                entity_scores={"ent_ben": 0.82},
                anchored_entity_ids=["ent_ben"],
            )
        )

        with patch("engram.retrieval.need.extract_signals", return_value=low_signal):
            need = await analyze_memory_need(
                "Ben",
                mode="chat",
                graph_probe=probe,
                cfg=_cfg(recall_need_graph_probe_enabled=True),
            )

        assert need.should_recall is False
        probe.probe.assert_not_awaited()

    async def test_graph_override_requires_anchored_high_confidence_graph_match(self):
        low_signal = SignalReport(
            pragmatic_score=0.05,
            linguistic_score=0.05,
            dominant_family="pragmatic",
            dominant_trigger_kind="bare_name",
        )
        probe = AsyncMock()
        probe.probe = AsyncMock(
            return_value=ProbeResult(
                resonance_score=0.82,
                detected_entities=["ent_ben"],
                entity_scores={"ent_ben": 0.82},
                anchored_entity_ids=["ent_ben"],
            )
        )

        with patch("engram.retrieval.need.extract_signals", return_value=low_signal):
            need = await analyze_memory_need(
                "Ben",
                mode="chat",
                graph_probe=probe,
                cfg=_cfg(
                    recall_need_graph_probe_enabled=True,
                    recall_need_graph_override_enabled=True,
                ),
            )

        assert need.should_recall is True
        assert need.need_type == "broad_context"
        assert need.trigger_family == "graph"
        assert need.trigger_kind == "graph_override"
        assert need.decision_path == "graph_override"
        assert need.graph_override_used is True
        probe.probe.assert_awaited_once()


class TestRecallNeedController:
    def test_snapshot_reports_runtime_metrics(self):
        controller = RecallNeedController(
            ActivationConfig(
                recall_need_adaptive_thresholds_enabled=False,
            )
        )
        controller.record_analysis(
            "default",
            MemoryNeed(
                need_type="project_state",
                should_recall=True,
                confidence=0.8,
                trigger_family="structural",
                analyzer_latency_ms=3.2,
                probe_triggered=True,
                probe_latency_ms=1.4,
                decision_path="graph_lift",
            ),
        )
        controller.record_interaction("default", "surfaced")
        controller.record_interaction("default", "used")
        controller.record_interaction("default", "dismissed")

        snapshot = controller.snapshot("default")

        assert snapshot["trigger_count"] == 1
        assert snapshot["used_count"] == 1
        assert snapshot["dismissed_count"] == 1
        assert snapshot["surfaced_count"] == 1
        assert snapshot["graph_lift_rate"] == pytest.approx(1.0)
        assert snapshot["probe_trigger_rate"] == pytest.approx(1.0)
        assert snapshot["family_contributions"]["structural"] == 1

    def test_adaptive_thresholds_stay_within_bounds(self):
        controller = RecallNeedController(
            ActivationConfig(
                recall_need_adaptive_thresholds_enabled=True,
                recall_need_adaptive_min_samples=5,
                recall_need_threshold_window=10,
            )
        )
        for _ in range(5):
            controller.record_analysis(
                "default",
                MemoryNeed(
                    need_type="fact_lookup",
                    should_recall=True,
                    confidence=0.7,
                    analyzer_latency_ms=2.0,
                ),
            )

        raised = controller.get_thresholds("default")
        assert 0.30 < raised.linguistic_score <= 0.45
        assert 0.15 < raised.borderline_score <= 0.25
        assert 0.45 < raised.resonance_score <= 0.60

        for _ in range(5):
            controller.record_interaction("default", "used")

        lowered = controller.get_thresholds("default")
        assert 0.25 <= lowered.linguistic_score <= 0.45
        assert 0.10 <= lowered.borderline_score <= 0.25
        assert 0.40 <= lowered.resonance_score <= 0.60
