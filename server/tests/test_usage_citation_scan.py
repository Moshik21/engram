"""M1.2 + M1.4 (RF goal): tier-tagged feedback + echo-guarded citation scan.

Covers the SurfacedUsageBuffer echo guard / dedup / short-circuit, the
Capture fast-path scan hook, confirmed/corrected tier tagging through the
interaction applier, and confirmed-tier events on cue-hit promotion.
"""

from __future__ import annotations

from typing import Any

import pytest

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.policy import ProjectionPolicy
from engram.ingestion.capture_surface import store_observation
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.control import RecallNeedController
from engram.retrieval.feedback import (
    RecallCueFeedbackRecorder,
    RecallEntityAccessRecorder,
    RecallInteractionRecorder,
    RecallMemoryInteractionApplier,
    SurfacedUsageBuffer,
    record_observed_usage_events,
)


class SpyActivationStore:
    """Records record_access calls (entity_id, ts, group_id, tier)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, str | None, str]] = []

    async def record_access(
        self,
        entity_id: str,
        timestamp: float,
        group_id: str | None = None,
        tier: str = "surfaced",
    ) -> None:
        self.calls.append((entity_id, timestamp, group_id, tier))

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return None

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        return None

    def tiers_for(self, entity_id: str) -> list[str]:
        return [tier for eid, _, _, tier in self.calls if eid == entity_id]


def _cfg(**overrides: Any) -> ActivationConfig:
    return ActivationConfig(recall_usage_feedback_enabled=True, **overrides)


MELANIE_SNIPPET = "Melanie enjoys pottery classes on weekends and paints landscapes"
PYTHON_SNIPPET = "Python is the preferred language for the data pipeline services"


def _buffer_with(*entries: tuple[str, str, str], ts: float = 1000.0) -> SurfacedUsageBuffer:
    buffer = SurfacedUsageBuffer()
    for entity_id, name, snippet in entries:
        buffer.note_surfaced("g1", entity_id=entity_id, name=name, snippet=snippet, ts=ts)
    return buffer


class TestEchoGuard:
    def test_novel_wording_reliance_fires(self):
        buffer = _buffer_with(("e1", "Melanie", MELANIE_SNIPPET))
        fired = buffer.scan_novel_mentions(
            "g1",
            "Sounds like Melanie would love that new pottery studio downtown",
            now=1100.0,
        )
        assert [entry.entity_id for entry in fired] == ["e1"]

    def test_verbatim_payload_echo_never_fires(self):
        buffer = _buffer_with(
            ("e1", "Melanie", MELANIE_SNIPPET),
            ("e2", "Python", PYTHON_SNIPPET),
        )
        echoed = f"{MELANIE_SNIPPET}. {PYTHON_SNIPPET}."
        assert buffer.scan_novel_mentions("g1", echoed, now=1100.0) == []

    def test_incidental_common_word_without_context_does_not_fire(self):
        buffer = _buffer_with(("e2", "Python", PYTHON_SNIPPET))
        fired = buffer.scan_novel_mentions(
            "g1",
            "I wrote a quick python one-liner to rename those photo files",
            now=1100.0,
        )
        assert fired == []

    def test_single_token_name_with_snippet_context_fires(self):
        buffer = _buffer_with(("e2", "Python", PYTHON_SNIPPET))
        fired = buffer.scan_novel_mentions(
            "g1",
            "We should keep python since the whole pipeline already depends on it",
            now=1100.0,
        )
        assert [entry.entity_id for entry in fired] == ["e2"]

    def test_unmentioned_entity_does_not_fire(self):
        buffer = _buffer_with(("e1", "Melanie", MELANIE_SNIPPET))
        fired = buffer.scan_novel_mentions(
            "g1",
            "Let's schedule the deploy for Thursday afternoon instead",
            now=1100.0,
        )
        assert fired == []

    def test_short_name_rejected(self):
        buffer = _buffer_with(("e3", "Go", "Go is used for the ingest worker binaries"))
        fired = buffer.scan_novel_mentions(
            "g1",
            "We could go with the worker rewrite next sprint",
            now=1100.0,
        )
        assert fired == []


class TestDedupAndShortCircuit:
    def test_dedup_within_rolling_window(self):
        buffer = _buffer_with(("e1", "Melanie", MELANIE_SNIPPET))
        content = "Melanie mentioned her pottery kiln again"
        assert buffer.scan_novel_mentions("g1", content, now=1100.0)
        # Same entity within the 30-minute window: suppressed.
        buffer.note_surfaced(
            "g1", entity_id="e1", name="Melanie", snippet=MELANIE_SNIPPET, ts=1200.0
        )
        assert buffer.scan_novel_mentions("g1", content, now=1300.0) == []
        # After the window expires it may fire again.
        buffer.note_surfaced(
            "g1", entity_id="e1", name="Melanie", snippet=MELANIE_SNIPPET, ts=1100.0 + 1900.0
        )
        assert buffer.scan_novel_mentions("g1", content, now=1100.0 + 1901.0)

    def test_ring_buffer_bounded_at_cap(self):
        buffer = SurfacedUsageBuffer(cap=4)
        for i in range(10):
            buffer.note_surfaced(
                "g1",
                entity_id=f"e{i}",
                name=f"Entity{i}",
                snippet=f"Entity{i} does something distinctive number {i}",
                ts=1000.0 + i,
            )
        assert len(buffer._entries["g1"]) == 4

    def test_empty_buffer_short_circuits(self):
        buffer = SurfacedUsageBuffer()
        assert buffer.is_empty("g1")
        assert buffer.scan_novel_mentions("g1", "Melanie pottery", now=1100.0) == []

    async def test_record_helper_flag_off_records_nothing(self):
        store = SpyActivationStore()
        buffer = _buffer_with(("e1", "Melanie", MELANIE_SNIPPET))
        cfg = ActivationConfig(recall_usage_feedback_enabled=False)
        fired = await record_observed_usage_events(
            activation_store=store,
            cfg=cfg,
            group_id="g1",
            content="Melanie loved that pottery wheel",
            now=1100.0,
            usage_buffer=buffer,
        )
        assert fired == []
        assert store.calls == []

    async def test_record_helper_fires_used_tier(self):
        store = SpyActivationStore()
        buffer = _buffer_with(("e1", "Melanie", MELANIE_SNIPPET))
        fired = await record_observed_usage_events(
            activation_store=store,
            cfg=_cfg(),
            group_id="g1",
            content="Melanie loved that pottery wheel",
            now=1100.0,
            usage_buffer=buffer,
        )
        assert fired == ["e1"]
        assert store.calls == [("e1", 1100.0, "g1", "used")]


class _FakeCaptureManager:
    """Minimal manager facade for the store_observation fast path."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg
        self._activation = SpyActivationStore()
        self.stored: list[str] = []

    async def store_episode(self, **kwargs: Any) -> str:
        self.stored.append(kwargs["content"])
        return "ep1"


class TestCaptureFastPathScan:
    async def test_plain_observe_without_prior_recall_records_zero_used_events(self):
        from engram.retrieval import feedback as feedback_module

        feedback_module.get_usage_buffer().reset()
        manager = _FakeCaptureManager(_cfg())
        episode_id = await store_observation(
            manager,
            content="Melanie loved that pottery wheel",
            group_id="g-fresh",
        )
        assert episode_id == "ep1"
        assert manager._activation.calls == []

    async def test_observe_after_surfacing_records_used_event(self):
        from engram.retrieval import feedback as feedback_module

        buffer = feedback_module.get_usage_buffer()
        buffer.reset()
        try:
            buffer.note_surfaced(
                "g-live",
                entity_id="e1",
                name="Melanie",
                snippet=MELANIE_SNIPPET,
                ts=1000.0,
            )
            manager = _FakeCaptureManager(_cfg())
            await store_observation(
                manager,
                content="Melanie loved that pottery wheel",
                group_id="g-live",
            )
            assert [(c[0], c[3]) for c in manager._activation.calls] == [("e1", "used")]
        finally:
            buffer.reset()

    async def test_scan_failure_never_fails_the_store(self):
        from engram.retrieval import feedback as feedback_module

        buffer = feedback_module.get_usage_buffer()
        buffer.reset()
        try:
            buffer.note_surfaced(
                "g-err",
                entity_id="e1",
                name="Melanie",
                snippet=MELANIE_SNIPPET,
                ts=1000.0,
            )
            manager = _FakeCaptureManager(_cfg())

            async def _boom(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("store down")

            manager._activation.record_access = _boom  # type: ignore[method-assign]
            episode_id = await store_observation(
                manager,
                content="Melanie loved that pottery wheel",
                group_id="g-err",
            )
            assert episode_id == "ep1"
        finally:
            buffer.reset()


class _FakeGraphForApplier:
    def __init__(self, entity: Entity) -> None:
        self._entity = entity

    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None:
        return self._entity if entity_id == self._entity.id else None


class TestFeedbackTierTagging:
    def _applier(
        self, cfg: ActivationConfig, store: SpyActivationStore, entity: Entity
    ) -> RecallMemoryInteractionApplier:
        controller = RecallNeedController(cfg)
        return RecallMemoryInteractionApplier(
            cfg=cfg,
            graph_store=_FakeGraphForApplier(entity),
            activation_store=store,
            cue_feedback_recorder=RecallCueFeedbackRecorder(
                cfg=cfg,
                graph_store=_FakeGraphForApplier(entity),
                projection_policy=ProjectionPolicy(cfg),
                recall_need_controller=controller,
                event_bus=None,
            ),
            entity_access_recorder=RecallEntityAccessRecorder(
                cfg=cfg,
                activation_store=store,
                event_bus=None,
                labile_tracker=None,
                usage_buffer=SurfacedUsageBuffer(),
            ),
            interaction_recorder=RecallInteractionRecorder(
                cfg=cfg,
                event_bus=None,
                recall_need_controller=controller,
            ),
            recall_need_controller=controller,
        )

    @pytest.mark.parametrize(
        ("interaction_type", "expected_tier"),
        [("confirmed", "confirmed"), ("corrected", "corrected"), ("used", "used")],
    )
    async def test_feedback_records_matching_tier(self, interaction_type, expected_tier):
        entity = Entity(id="e1", name="React", entity_type="Technology", group_id="g1")
        store = SpyActivationStore()
        applier = self._applier(_cfg(), store, entity)
        await applier.apply(["e1"], interaction_type=interaction_type, group_id="g1")
        assert store.tiers_for("e1") == [expected_tier]

    async def test_dismissed_records_no_access(self):
        entity = Entity(id="e1", name="React", entity_type="Technology", group_id="g1")
        store = SpyActivationStore()
        applier = self._applier(_cfg(), store, entity)
        await applier.apply(["e1"], interaction_type="dismissed", group_id="g1")
        assert store.calls == []


EPISODE_ID = "ep_promote"


class _FakeGraphForPromotion:
    def __init__(self, linked_entities: list[str]) -> None:
        self.episode = Episode(
            id=EPISODE_ID,
            content="The migration to native Helix finished on Tuesday.",
            group_id="g1",
            projection_state=EpisodeProjectionState.CUE_ONLY,
        )
        self.cue = EpisodeCue(
            episode_id=EPISODE_ID,
            group_id="g1",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text="native Helix migration",
            hit_count=0,
        )
        self._linked_entities = linked_entities

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        return self.cue if episode_id == EPISODE_ID else None

    async def get_episode_entities(self, episode_id: str, *, group_id: str) -> list[str]:
        return list(self._linked_entities)

    async def update_episode_cue(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        if "hit_count" in updates:
            self.cue.hit_count = int(updates["hit_count"])

    async def update_episode(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        new_state = updates.get("projection_state")
        if new_state is not None:
            self.episode.projection_state = EpisodeProjectionState(new_state)


class TestCuePromotionConfirmedTier:
    def _recorder(
        self,
        graph: _FakeGraphForPromotion,
        cfg: ActivationConfig,
        store: SpyActivationStore | None,
    ) -> RecallCueFeedbackRecorder:
        return RecallCueFeedbackRecorder(
            cfg=cfg,
            graph_store=graph,
            projection_policy=ProjectionPolicy(cfg),
            recall_need_controller=RecallNeedController(cfg),
            event_bus=EventBus(),
            activation_store=store,
        )

    async def test_promotion_records_confirmed_tier_for_linked_entities(self):
        cfg = _cfg(cue_recall_hit_threshold=1)
        graph = _FakeGraphForPromotion(["e1", "e2"])
        store = SpyActivationStore()
        recorder = self._recorder(graph, cfg, store)

        await recorder.record_cue_feedback(graph.episode, 0.9, "helix migration")

        assert graph.episode.projection_state == EpisodeProjectionState.SCHEDULED
        assert sorted((c[0], c[3]) for c in store.calls) == [
            ("e1", "confirmed"),
            ("e2", "confirmed"),
        ]

    async def test_promotion_without_linked_entities_records_nothing(self):
        cfg = _cfg(cue_recall_hit_threshold=1)
        graph = _FakeGraphForPromotion([])
        store = SpyActivationStore()
        recorder = self._recorder(graph, cfg, store)

        await recorder.record_cue_feedback(graph.episode, 0.9, "helix migration")

        assert graph.episode.projection_state == EpisodeProjectionState.SCHEDULED
        assert store.calls == []

    async def test_promotion_usage_gated_off_by_default_flag(self):
        cfg = ActivationConfig(
            recall_usage_feedback_enabled=False,
            cue_recall_hit_threshold=1,
        )
        graph = _FakeGraphForPromotion(["e1"])
        store = SpyActivationStore()
        recorder = self._recorder(graph, cfg, store)

        await recorder.record_cue_feedback(graph.episode, 0.9, "helix migration")

        assert graph.episode.projection_state == EpisodeProjectionState.SCHEDULED
        assert store.calls == []

    async def test_non_promoting_hit_records_no_usage(self):
        cfg = _cfg(cue_recall_hit_threshold=5)
        graph = _FakeGraphForPromotion(["e1"])
        store = SpyActivationStore()
        recorder = self._recorder(graph, cfg, store)

        await recorder.record_cue_feedback(graph.episode, 0.9, "helix migration")

        assert graph.episode.projection_state == EpisodeProjectionState.CUE_ONLY
        assert store.calls == []


class TestSurfacedTextMask:
    def test_episode_text_echo_does_not_fire(self):
        """Verifier bypass: an agent echoing surfaced EPISODE content verbatim
        must not fire used events for entities mentioned inside it — the
        mask-only text channel covers all surfaced text, not just entity
        snippets."""
        from engram.retrieval.feedback import SurfacedUsageBuffer

        buffer = SurfacedUsageBuffer()
        episode_text = (
            "Melanie spent the weekend at her pottery class working on a new "
            "glaze technique for the studio showcase next month"
        )
        buffer.note_surfaced(
            "g1",
            entity_id="ent_melanie",
            name="Melanie",
            # Production snippets are name + summary; single-token names need
            # context overlap with this vocabulary to fire (common-word guard).
            snippet="Melanie. Person. Enjoys pottery classes and the studio showcase.",
            ts=100.0,
        )
        buffer.note_surfaced_text("g1", episode_text, 100.0)

        # Verbatim echo of the episode text: mention is inside the mask.
        fired = buffer.scan_novel_mentions("g1", f"Sure — {episode_text}.", now=200.0)
        assert fired == []

        # Genuinely novel reliance still fires.
        fired = buffer.scan_novel_mentions(
            "g1",
            "Melanie should book the kiln early since her showcase entry needs two firings",
            now=300.0,
        )
        assert [e.entity_id for e in fired] == ["ent_melanie"]

    def test_response_feed_helper_masks_results_and_packets(self):
        from types import SimpleNamespace

        from engram.retrieval.feedback import (
            get_usage_buffer,
            note_surfaced_texts_from_response,
        )

        buffer = get_usage_buffer()
        buffer.reset()
        buffer.note_surfaced(
            "g2", entity_id="ent_a", name="Quarterly Report", snippet="Quarterly Report", ts=1.0
        )
        response = {
            "results": [{"text": "The quarterly report shows revenue grew nine percent"}],
            "packets": [{"title": "Finance", "summary": "quarterly report revenue analysis"}],
        }
        cfg = SimpleNamespace(recall_usage_feedback_enabled=True)
        note_surfaced_texts_from_response("g2", response, cfg, now=1.0)

        fired = buffer.scan_novel_mentions(
            "g2", "The quarterly report shows revenue grew nine percent", now=2.0
        )
        assert fired == []
        buffer.reset()
