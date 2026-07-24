"""M5.1 + M5.2 (RF goal): episode u via the cue substrate.

Covers the compute_u_values pure refactor, the tier-weighted cue usage fields
(model + sqlite columns + helix supporting_spans_json trailer), the same-pass
echo-guarded cue citation scan, the flag-gated episode-u composition
(final = rrf x (1 + beta_route*u_episode) x (1 + temporal_cue_boost)), the
composition stack bound, and the Step-5.05 undated-episode no-boost edge.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from engram.activation.engine import compute_u, compute_u_values
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.feedback import (
    SurfacedUsageBuffer,
    record_observed_usage_events,
)
from engram.retrieval.pipeline import _apply_episode_usage_tiebreaker, retrieve
from engram.retrieval.scorer import ScoredResult
from engram.storage.helix.graph import (
    _encode_cue_usage_mentions,
    _split_cue_usage_mentions,
)
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.dates import utc_now

# ---------------------------------------------------------------------------
# compute_u_values — pure refactor of compute_u (zero behavior change)
# ---------------------------------------------------------------------------


class TestComputeUValues:
    def test_compute_u_delegates_to_compute_u_values(self):
        cfg = ActivationConfig()
        now = 1_000_000.0
        for weights, last_ts in [
            ([0.3], now - 60.0),
            ([0.3, 1.0, 0.5], now - 86400.0),
            ([1.0] * 60, now - 30 * 86400.0),
        ]:
            state = ActivationState(node_id="e1")
            for i, w in enumerate(weights):
                state.record_usage_event(last_ts - i, w)
            state.usage_last_ts = last_ts
            assert compute_u(state, now, cfg) == compute_u_values(
                state.n_eff, state.usage_last_ts, now, cfg
            )

    def test_zero_n_eff_is_exactly_zero(self):
        cfg = ActivationConfig()
        assert compute_u_values(0.0, 500.0, 1000.0, cfg) == 0.0
        assert compute_u_values(-1.0, 500.0, 1000.0, cfg) == 0.0

    def test_formula_pinned(self):
        cfg = ActivationConfig()
        now = 1_000_000.0
        n_eff = 5.0
        last_ts = now - 14 * 86400.0  # exactly one half-life
        f = min(1.0, math.log1p(n_eff) / math.log1p(cfg.usage_n_cap))
        r_prime = cfg.usage_r_floor + (1.0 - cfg.usage_r_floor) * 0.5
        assert compute_u_values(n_eff, last_ts, now, cfg) == pytest.approx(f * r_prime)

    def test_bounded_zero_one(self):
        cfg = ActivationConfig()
        now = 1_000_000.0
        assert 0.0 < compute_u_values(1000.0, now, now, cfg) <= 1.0


# ---------------------------------------------------------------------------
# Cue model + sqlite persistence
# ---------------------------------------------------------------------------


class TestEpisodeCueModel:
    def test_usage_fields_default_empty(self):
        cue = EpisodeCue(episode_id="ep1")
        assert cue.usage_used_count == 0.0
        assert cue.usage_last_used_at is None
        # Legacy hygiene counter untouched.
        assert cue.used_count == 0
        assert isinstance(cue.used_count, int)


@pytest.mark.asyncio
class TestSqliteCueUsage:
    async def _store(self, tmp_path) -> SQLiteGraphStore:
        store = SQLiteGraphStore(str(tmp_path / "cue_usage.db"))
        await store.initialize()
        # episode_cues carries a FK to episodes: the cue's episode must exist.
        await store.create_episode(
            Episode(
                id="ep1",
                content="episode one content",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            )
        )
        return store

    async def test_round_trip_defaults(self, tmp_path):
        store = await self._store(tmp_path)
        await store.upsert_episode_cue(EpisodeCue(episode_id="ep1", cue_text="alpha"))
        cue = await store.get_episode_cue("ep1", "default")
        assert cue is not None
        assert cue.usage_used_count == 0.0
        assert cue.usage_last_used_at is None
        await store.close()

    async def test_round_trip_with_usage(self, tmp_path):
        store = await self._store(tmp_path)
        ts = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
        await store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep1",
                cue_text="alpha",
                used_count=2,
                usage_used_count=1.3,
                usage_last_used_at=ts,
            )
        )
        cue = await store.get_episode_cue("ep1", "default")
        assert cue.usage_used_count == pytest.approx(1.3)
        assert cue.usage_last_used_at == ts
        assert cue.used_count == 2  # hygiene counter independent
        await store.close()

    async def test_partial_update_episode_cue(self, tmp_path):
        store = await self._store(tmp_path)
        await store.upsert_episode_cue(EpisodeCue(episode_id="ep1", cue_text="alpha", hit_count=3))
        ts = datetime(2026, 7, 21, 13, 0, 0, tzinfo=timezone.utc)
        await store.update_episode_cue(
            "ep1",
            {"usage_used_count": 0.3, "usage_last_used_at": ts},
            group_id="default",
        )
        cue = await store.get_episode_cue("ep1", "default")
        assert cue.usage_used_count == pytest.approx(0.3)
        assert cue.usage_last_used_at == ts
        assert cue.hit_count == 3
        await store.close()


# ---------------------------------------------------------------------------
# Helix trailer encoding (fixed native schema — no schema.hx regen)
# ---------------------------------------------------------------------------


class TestHelixCueUsageEncoding:
    def test_no_usage_is_byte_identical_to_plain_mentions(self):
        mentions = [{"name": "Melanie", "type": "Person"}]
        assert _encode_cue_usage_mentions(mentions, 0.0, None) == json.dumps(mentions)

    def test_encode_split_round_trip(self):
        mentions = [{"name": "Melanie"}, {"name": "Pottery"}]
        raw = _encode_cue_usage_mentions(
            mentions, 0.6, datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
        )
        clean, used_count, last_used_at = _split_cue_usage_mentions(raw)
        assert clean == mentions
        assert used_count == pytest.approx(0.6)
        assert last_used_at == "2026-07-21T12:00:00+00:00"

    def test_dict_to_episode_cue_strips_trailer(self):
        from engram.storage.helix.graph import HelixGraphStore

        store = HelixGraphStore.__new__(HelixGraphStore)
        store._cue_id_cache = {}
        raw = _encode_cue_usage_mentions(
            [{"name": "Melanie"}], 1.3, datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
        )
        cue = store._dict_to_episode_cue(
            {
                "episode_id": "ep1",
                "group_id": "default",
                "cue_text": "mentions: Melanie",
                "supporting_spans_json": raw,
                "used_count": 2,
            }
        )
        assert cue.entity_mentions == [{"name": "Melanie"}]
        assert cue.usage_used_count == pytest.approx(1.3)
        assert cue.usage_last_used_at == datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)
        assert cue.used_count == 2

    def test_model_payload_round_trip(self):
        from engram.storage.helix.graph import HelixGraphStore, _cue_model_payload

        cue = EpisodeCue(
            episode_id="ep1",
            cue_text="mentions: Melanie",
            entity_mentions=[{"name": "Melanie"}],
            usage_used_count=0.3,
            usage_last_used_at=datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc),
        )
        payload = _cue_model_payload(cue, utc_now().isoformat())
        store = HelixGraphStore.__new__(HelixGraphStore)
        store._cue_id_cache = {}
        loaded = store._dict_to_episode_cue(payload)
        assert loaded.entity_mentions == [{"name": "Melanie"}]
        assert loaded.usage_used_count == pytest.approx(0.3)
        assert loaded.usage_last_used_at == cue.usage_last_used_at


# ---------------------------------------------------------------------------
# Same-pass echo-guarded cue citation scan
# ---------------------------------------------------------------------------

CUE_TEXT = "mentions: Melanie | spans: pottery classes moved to Tuesday evenings this month"
CUE_SPAN = "pottery classes moved to Tuesday evenings this month"


def _cue_buffer(ts: float = 1000.0) -> SurfacedUsageBuffer:
    buffer = SurfacedUsageBuffer()
    buffer.note_surfaced_cue(
        "g1",
        episode_id="ep1",
        cue_text=CUE_TEXT,
        supporting_spans=[CUE_SPAN],
        ts=ts,
    )
    return buffer


class TestCueCitationScan:
    def test_novel_phrase_reuse_fires(self):
        buffer = _cue_buffer()
        fired = buffer.scan_novel_cue_matches(
            "g1",
            "Let's plan around the Tuesday evenings slot she mentioned",
            now=1100.0,
        )
        assert [entry.episode_id for entry in fired] == ["ep1"]

    def test_verbatim_echo_never_fires(self):
        buffer = _cue_buffer()
        assert buffer.scan_novel_cue_matches("g1", f"You said: {CUE_SPAN}.", now=1100.0) == []

    def test_unrelated_content_does_not_fire(self):
        buffer = _cue_buffer()
        fired = buffer.scan_novel_cue_matches(
            "g1",
            "Deploy the ingest worker on Thursday afternoon instead",
            now=1100.0,
        )
        assert fired == []

    def test_dedup_window_blocks_double_fire(self):
        buffer = _cue_buffer()
        content = "Let's plan around the Tuesday evenings slot she mentioned"
        assert buffer.scan_novel_cue_matches("g1", content, now=1100.0)
        assert buffer.scan_novel_cue_matches("g1", content, now=1200.0) == []

    def test_is_empty_accounts_for_cues(self):
        buffer = _cue_buffer()
        assert not buffer.is_empty("g1")
        assert buffer.is_empty("g2")
        buffer.reset()
        assert buffer.is_empty("g1")


@pytest.mark.asyncio
class TestRecordObservedCueUsage:
    async def test_cue_fire_updates_cue_usage_fields(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=True)
        buffer = _cue_buffer()
        graph = AsyncMock()
        graph.get_episode_cue = AsyncMock(
            return_value=EpisodeCue(episode_id="ep1", cue_text=CUE_TEXT)
        )
        graph.update_episode_cue = AsyncMock()
        activation = AsyncMock()

        fired = await record_observed_usage_events(
            activation_store=activation,
            cfg=cfg,
            group_id="g1",
            content="Let's plan around the Tuesday evenings slot she mentioned",
            now=1100.0,
            usage_buffer=buffer,
            graph_store=graph,
        )

        assert fired == ["cue::ep1"]
        graph.update_episode_cue.assert_awaited_once()
        args, kwargs = graph.update_episode_cue.await_args
        assert args[0] == "ep1"
        updates = args[1]
        assert updates["usage_used_count"] == pytest.approx(cfg.usage_tier_weights["used"])
        assert updates["usage_last_used_at"] == datetime.fromtimestamp(1100.0, tz=timezone.utc)
        assert kwargs["group_id"] == "g1"

    async def test_no_graph_store_keeps_entity_only_contract(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=True)
        buffer = _cue_buffer()
        activation = AsyncMock()
        fired = await record_observed_usage_events(
            activation_store=activation,
            cfg=cfg,
            group_id="g1",
            content="Let's plan around the Tuesday evenings slot she mentioned",
            now=1100.0,
            usage_buffer=buffer,
        )
        assert fired == []

    async def test_flag_off_is_inert(self):
        cfg = ActivationConfig()
        assert cfg.recall_usage_feedback_enabled is False
        buffer = _cue_buffer()
        graph = AsyncMock()
        fired = await record_observed_usage_events(
            activation_store=AsyncMock(),
            cfg=cfg,
            group_id="g1",
            content="Let's plan around the Tuesday evenings slot she mentioned",
            now=1100.0,
            usage_buffer=buffer,
            graph_store=graph,
        )
        assert fired == []
        graph.get_episode_cue.assert_not_called()


# ---------------------------------------------------------------------------
# Surfaced-cue registration (P2: surfaced -> used was structurally impossible
# off the explicit-recall surfaces)
# ---------------------------------------------------------------------------


REG_CUE_TEXT = "mentions: Helix || spans: the native Helix migration finished on Tuesday"
REG_SPAN = "the native Helix migration finished on Tuesday"
REG_EPISODE_CONTENT = "We shipped it: the native Helix migration finished on Tuesday afternoon."
REG_NOVEL_TURN = "Then let's tell the team the native Helix migration is done and close the ticket"


class _FakeCueGraph:
    """Minimal graph store for the surfaced-cue registration path."""

    def __init__(self) -> None:
        self.episode = Episode(
            id="ep_reg",
            content=REG_EPISODE_CONTENT,
            group_id="g_reg",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            status=EpisodeStatus.QUEUED,
        )
        self.cue = EpisodeCue(
            episode_id="ep_reg",
            group_id="g_reg",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text=REG_CUE_TEXT,
            first_spans=[REG_SPAN],
        )

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        return self.cue if episode_id == self.cue.episode_id else None

    async def get_episode_entities(self, episode_id: str, *, group_id: str) -> list[str]:
        return []

    async def update_episode_cue(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        for field, value in updates.items():
            if hasattr(self.cue, field):
                setattr(self.cue, field, value)

    async def update_episode(self, episode_id: str, updates: dict, *, group_id: str) -> None:
        state = updates.get("projection_state")
        if state is not None:
            self.episode.projection_state = EpisodeProjectionState(state)


def _cue_recorder(graph: _FakeCueGraph, cfg: ActivationConfig, buffer: SurfacedUsageBuffer):
    from engram.extraction.policy import ProjectionPolicy
    from engram.retrieval.control import RecallNeedController
    from engram.retrieval.feedback import RecallCueFeedbackRecorder

    return RecallCueFeedbackRecorder(
        cfg=cfg,
        graph_store=graph,
        projection_policy=ProjectionPolicy(cfg),
        recall_need_controller=RecallNeedController(cfg),
        event_bus=None,
        usage_buffer=buffer,
    )


@pytest.mark.asyncio
class TestSurfacedCueRegistration:
    async def test_pipeline_surfacing_makes_a_use_event_recordable(self):
        """End to end: surfacing a cue through the shared recorder (the path
        every surface uses, not just explicit recall) then observing a turn
        that reuses the cue phrase in novel wording writes cue usage."""
        cfg = ActivationConfig(recall_usage_feedback_enabled=True, cue_recall_hit_threshold=20)
        graph = _FakeCueGraph()
        buffer = SurfacedUsageBuffer()

        await _cue_recorder(graph, cfg, buffer).record_cue_feedback(
            graph.episode,
            0.9,
            "helix migration",
            interaction_type="surfaced",
        )
        assert not buffer.is_empty("g_reg")

        fired = await record_observed_usage_events(
            activation_store=AsyncMock(),
            cfg=cfg,
            group_id="g_reg",
            content=REG_NOVEL_TURN,
            now=time.time(),
            usage_buffer=buffer,
            graph_store=graph,
        )

        assert fired == ["cue::ep_reg"]
        assert graph.cue.usage_used_count == pytest.approx(cfg.usage_tier_weights["used"])
        assert graph.cue.usage_last_used_at is not None

    async def test_verbatim_payload_echo_does_not_fire(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=True, cue_recall_hit_threshold=20)
        graph = _FakeCueGraph()
        buffer = SurfacedUsageBuffer()

        await _cue_recorder(graph, cfg, buffer).record_cue_feedback(
            graph.episode,
            0.9,
            "helix migration",
            interaction_type="surfaced",
        )
        fired = await record_observed_usage_events(
            activation_store=AsyncMock(),
            cfg=cfg,
            group_id="g_reg",
            content=f"You already told me: {REG_EPISODE_CONTENT}",
            now=time.time(),
            usage_buffer=buffer,
            graph_store=graph,
        )

        assert fired == []
        assert graph.cue.usage_used_count == 0.0

    async def test_near_miss_does_not_register(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=True, cue_recall_hit_threshold=20)
        graph = _FakeCueGraph()
        buffer = SurfacedUsageBuffer()

        await _cue_recorder(graph, cfg, buffer).record_cue_feedback(
            graph.episode,
            0.9,
            "helix migration",
            near_miss=True,
        )

        assert buffer.is_empty("g_reg")

    async def test_flag_off_registers_nothing(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=False, cue_recall_hit_threshold=20)
        graph = _FakeCueGraph()
        buffer = SurfacedUsageBuffer()

        await _cue_recorder(graph, cfg, buffer).record_cue_feedback(
            graph.episode,
            0.9,
            "helix migration",
            interaction_type="surfaced",
        )

        assert buffer.is_empty("g_reg")

    async def test_repeat_surfacing_keeps_one_ring_entry(self):
        cfg = ActivationConfig(recall_usage_feedback_enabled=True, cue_recall_hit_threshold=20)
        graph = _FakeCueGraph()
        buffer = SurfacedUsageBuffer()
        recorder = _cue_recorder(graph, cfg, buffer)

        for _ in range(3):
            await recorder.record_cue_feedback(
                graph.episode,
                0.9,
                "helix migration",
                interaction_type="surfaced",
            )

        assert len(buffer._cue_entries["g_reg"]) == 1


# ---------------------------------------------------------------------------
# Episode-u composition (flag ON only)
# ---------------------------------------------------------------------------


def _episode_result(node_id: str, score: float, result_type: str = "episode") -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
        exploration_bonus=0.0,
        result_type=result_type,
    )


def _used_cue(episode_id: str, now: float, used_count: float = 5.0) -> EpisodeCue:
    return EpisodeCue(
        episode_id=episode_id,
        cue_text="c",
        usage_used_count=used_count,
        usage_last_used_at=datetime.fromtimestamp(now - 60.0, tz=timezone.utc),
    )


@pytest.mark.asyncio
class TestEpisodeUsageComposition:
    async def test_used_cue_episode_wins_equal_rrf_tie(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        now = time.time()
        ep_a = _episode_result("ep_a", 0.5)
        ep_b = _episode_result("ep_b", 0.5)
        graph = AsyncMock()

        async def _get_cue(episode_id, group_id):
            return _used_cue("ep_a", now) if episode_id == "ep_a" else None

        graph.get_episode_cue = AsyncMock(side_effect=_get_cue)
        await _apply_episode_usage_tiebreaker(
            [ep_a, ep_b], [], graph_store=graph, group_id="default", now=now, cfg=cfg
        )
        assert ep_a.score > ep_b.score
        assert ep_b.score == 0.5
        u = compute_u_values(5.0, now - 60.0, now, cfg)
        assert ep_a.score == pytest.approx(0.5 * (1.0 + cfg.usage_beta_route * u))

    async def test_empty_cue_usage_is_noop(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        now = time.time()
        ep_a = _episode_result("ep_a", 0.5)
        graph = AsyncMock()
        graph.get_episode_cue = AsyncMock(return_value=EpisodeCue(episode_id="ep_a", cue_text="c"))
        await _apply_episode_usage_tiebreaker(
            [ep_a], [], graph_store=graph, group_id="default", now=now, cfg=cfg
        )
        assert ep_a.score == 0.5

    async def test_cue_read_failure_is_nonfatal_no_boost(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        now = time.time()
        ep_a = _episode_result("ep_a", 0.5)
        graph = AsyncMock()
        graph.get_episode_cue = AsyncMock(side_effect=RuntimeError("down"))
        await _apply_episode_usage_tiebreaker(
            [ep_a], [], graph_store=graph, group_id="default", now=now, cfg=cfg
        )
        assert ep_a.score == 0.5

    async def test_flag_off_pipeline_never_reads_cues_for_u(self):
        """The Step-1.4 call site is gated on usage_ranking_enabled (default off)."""
        cfg = ActivationConfig()
        assert cfg.usage_ranking_enabled is False


class TestCompositionStackBound:
    def test_beta_route_capped_at_030(self):
        with pytest.raises(Exception):
            ActivationConfig(usage_beta_route=0.31)

    def test_boosted_both_cannot_overtake_3_9x_rrf(self):
        """Usage stack <= 1.30, temporal envelope <= 3.0 => combined <= 3.9x."""
        cfg = ActivationConfig(usage_ranking_enabled=True, usage_beta_route=0.30)
        u_max = 1.0
        usage_factor = 1.0 + cfg.usage_beta_route * u_max
        assert usage_factor <= 1.30 + 1e-12
        temporal_envelope = 3.0
        max_stack = usage_factor * temporal_envelope
        assert max_stack <= 3.9 + 1e-12
        # A >= 3.9x-stronger rrf item can never be overtaken.
        weak, strong = 0.1, 0.1 * 3.9
        assert weak * max_stack <= strong + 1e-12
        # And is strictly safe with any margin above 3.9x.
        assert weak * max_stack < 0.1 * 4.0

    def test_actual_temporal_factor_within_envelope(self):
        # Step 5.05 factor = 1 + exp(-age/h) (or 1 + (1-exp)) is bounded by 2.0,
        # inside the conservative 3.0 envelope used for the bound.
        for age_days in [0.0, 1.0, 14.0, 365.0]:
            assert 1.0 + math.exp(-age_days / 14.0) <= 2.0
            assert 1.0 + (1.0 - math.exp(-age_days / 14.0)) <= 2.0


# ---------------------------------------------------------------------------
# M5.2 — Step 5.05 undated-episode no-boost edge
# ---------------------------------------------------------------------------


def _temporal_search_index():
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=[("e1", 0.9)])
    idx.search_episodes = AsyncMock(return_value=[("ep_dated", 0.5), ("ep_undated", 0.5)])
    idx.search_episode_cues = AsyncMock(return_value=[])
    idx.compute_similarity = AsyncMock(return_value={})
    idx._embeddings_enabled = False
    return idx


def _temporal_graph_store():
    from engram.models.entity import Entity

    store = AsyncMock()
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    store.update_episode = AsyncMock()
    store.update_episode_cue = AsyncMock()
    store.get_entity = AsyncMock(
        return_value=Entity(
            id="e1", name="Test", entity_type="Thing", summary="s", group_id="default"
        )
    )

    def _episode(episode_id: str) -> Episode:
        return Episode(
            id=episode_id,
            content=f"content of {episode_id}",
            source="test",
            status=EpisodeStatus.COMPLETED,
            projection_state=EpisodeProjectionState.CUE_ONLY,
            group_id="default",
            created_at=utc_now(),
            conversation_date=(utc_now() - timedelta(days=1) if episode_id == "ep_dated" else None),
        )

    store.get_episode_by_id = AsyncMock(side_effect=lambda eid, gid: _episode(eid))
    store.get_episode_cue = AsyncMock(return_value=None)
    store.get_episode_entities = AsyncMock(return_value=[])
    return store


@pytest.mark.asyncio
class TestUndatedEpisodeNoBoost:
    async def test_undated_episode_gets_no_temporal_boost(self):
        cfg = ActivationConfig(episode_retrieval_enabled=True)
        assert cfg.temporal_retrieval_enabled is True
        results = await retrieve(
            query="what is the latest update on the project",
            group_id="default",
            graph_store=_temporal_graph_store(),
            activation_store=AsyncMock(
                batch_get=AsyncMock(return_value={}),
                get_activation=AsyncMock(return_value=None),
                set_activation=AsyncMock(),
                record_access=AsyncMock(),
                get_top_activated=AsyncMock(return_value=[]),
            ),
            search_index=_temporal_search_index(),
            cfg=cfg,
        )
        by_id = {r.node_id: r for r in results if r.result_type == "episode"}
        assert "ep_dated" in by_id and "ep_undated" in by_id
        # Base episode score = original weight_semantic * sem * weight (passage_first => 1.0)
        base = cfg.weight_semantic * 0.5
        assert by_id["ep_undated"].score == pytest.approx(base)
        assert by_id["ep_dated"].score > by_id["ep_undated"].score


class TestEpisodeUsageSkipCache:
    """The per-process cue-usage marker (gate-rerun latency fix): after 3
    all-zero probes the tiebreaker skips its cue reads; the write-side scan
    re-arms it; finding usage makes it sticky-True."""

    def setup_method(self):
        from engram.retrieval import pipeline

        pipeline._EPISODE_USAGE_SEEN.clear()

    def teardown_method(self):
        from engram.retrieval import pipeline

        pipeline._EPISODE_USAGE_SEEN.clear()

    @pytest.mark.asyncio
    async def test_three_empty_probes_stop_cue_reads(self):
        from engram.retrieval import pipeline
        from engram.retrieval.scorer import ScoredResult

        reads = []

        class Store:
            async def get_episode_cue(self, episode_id, group_id):
                reads.append(episode_id)
                return None

        cfg = ActivationConfig(usage_ranking_enabled=True)

        def cands():
            return [
                ScoredResult(
                    node_id="ep1",
                    score=0.5,
                    semantic_similarity=0.5,
                    activation=0.0,
                    spreading=0.0,
                    edge_proximity=0.0,
                    exploration_bonus=0.0,
                )
            ]

        for expected_reads in (1, 2, 3, 3, 3):
            await pipeline._apply_episode_usage_tiebreaker(
                cands(), [], graph_store=Store(), group_id="g1", now=1000.0, cfg=cfg
            )
            assert len(reads) == expected_reads

        # Write-side invalidation re-arms the reads.
        pipeline.note_group_cue_usage_written("g1")
        await pipeline._apply_episode_usage_tiebreaker(
            cands(), [], graph_store=Store(), group_id="g1", now=1000.0, cfg=cfg
        )
        assert len(reads) == 4

    @pytest.mark.asyncio
    async def test_found_usage_is_sticky_and_reads_continue(self):
        from datetime import datetime, timezone

        from engram.retrieval import pipeline
        from engram.retrieval.scorer import ScoredResult

        reads = []

        class Cue:
            usage_used_count = 0.3
            usage_last_used_at = datetime.fromtimestamp(900.0, tz=timezone.utc)

        class Store:
            async def get_episode_cue(self, episode_id, group_id):
                reads.append(episode_id)
                return Cue()

        cfg = ActivationConfig(usage_ranking_enabled=True)
        for i in range(5):
            sr = ScoredResult(
                node_id="ep1",
                score=0.5,
                semantic_similarity=0.5,
                activation=0.0,
                spreading=0.0,
                edge_proximity=0.0,
                exploration_bonus=0.0,
            )
            await pipeline._apply_episode_usage_tiebreaker(
                [sr], [], graph_store=Store(), group_id="g2", now=1000.0, cfg=cfg
            )
            assert sr.score > 0.5  # tiebreaker applied every time
            assert len(reads) == i + 1
        assert pipeline._EPISODE_USAGE_SEEN["g2"] is True
