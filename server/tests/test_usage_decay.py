"""M4.1 usage-decay demotion (AGENT_EXPERIENCE D4 = demotion-first).

Chronic surfaced-never-used items (surfaced >= N_min, used == 0, age > floor)
get an offline demotion marker in the mop window. P5 boundary pinned here:
the marker is forgetting evidence, never ranking evidence — the scorer can
never read it, the presenter demotion lives behind an OFF-default eval-gated
flag, and no retrieval module references the marker at all.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import engram
from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.consolidation.usage_decay import (
    USAGE_DECAY_MARKER,
    USAGE_DECAY_SURFACED,
    decode_usage_decay_marker,
    demote_surfaced_never_used_results,
    is_chronic_non_use,
    prune_feed_eligible,
    run_usage_decay,
)
from engram.models.consolidation import PhaseResult
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.storage.memory.activation import MemoryActivationStore

GROUP = "g"


def _days_ago(days: float) -> datetime:
    return datetime.now(tz=UTC) - timedelta(days=days)


def _episode(episode_id: str, *, age_days: float = 30.0, **kwargs) -> Episode:
    return Episode(
        id=episode_id,
        content=f"content for {episode_id}",
        group_id=GROUP,
        created_at=_days_ago(age_days),
        **kwargs,
    )


def _cue(episode_id: str, **kwargs) -> EpisodeCue:
    return EpisodeCue(episode_id=episode_id, group_id=GROUP, **kwargs)


def _entity(entity_id: str, *, age_days: float = 30.0, **kwargs) -> Entity:
    kwargs.setdefault("name", entity_id)
    kwargs.setdefault("entity_type", "Concept")
    return Entity(id=entity_id, group_id=GROUP, created_at=_days_ago(age_days), **kwargs)


class FakeGraphStore:
    """Minimal store: episodes + cues + entities with real read-back."""

    def __init__(self) -> None:
        self.episodes: dict[str, Episode] = {}
        self.cues: dict[str, EpisodeCue] = {}
        self.entities: dict[str, Entity] = {}
        self.episode_updates: list[tuple[str, dict]] = []
        self.entity_updates: list[tuple[str, dict]] = []

    async def get_episodes(self, group_id=None, limit=50, offset=0):
        return list(self.episodes.values())[:limit]

    async def get_episode_cue(self, episode_id: str, group_id: str):
        return self.cues.get(episode_id)

    async def update_episode(self, episode_id: str, updates: dict, group_id: str = "default"):
        self.episode_updates.append((episode_id, dict(updates)))
        episode = self.episodes[episode_id]
        for key, value in updates.items():
            setattr(episode, key, value)

    async def get_entity(self, entity_id: str, group_id: str):
        return self.entities.get(entity_id)

    async def update_entity(self, entity_id: str, updates: dict, group_id: str):
        self.entity_updates.append((entity_id, dict(updates)))
        entity = self.entities.get(entity_id)
        if entity is None:
            return
        for key, value in updates.items():
            if key == "attributes" and isinstance(value, str):
                entity.attributes = json.loads(value)
            elif hasattr(entity, key):
                setattr(entity, key, value)


async def _surfaced_activation(entity_id: str, count: int) -> MemoryActivationStore:
    store = MemoryActivationStore()
    now = time.time()
    for i in range(count):
        await store.record_access(entity_id, now - 3600.0 * (i + 1), group_id=GROUP)
    return store


def _cfg(**overrides) -> ActivationConfig:
    return ActivationConfig(**overrides)


class TestChronicPredicate:
    def test_thresholds(self):
        kw = {"min_surfaced": 12, "min_age_days": 14.0}
        assert is_chronic_non_use(12, 0.0, 15.0, **kw)
        assert not is_chronic_non_use(11, 0.0, 15.0, **kw)  # under N_min
        assert not is_chronic_non_use(12, 0.1, 15.0, **kw)  # used
        assert not is_chronic_non_use(12, 0.0, 14.0, **kw)  # age not strictly over
        assert is_chronic_non_use(50, 0.0, 14.01, **kw)


class TestEpisodeDemotion:
    @pytest.mark.asyncio
    async def test_chronic_cue_episode_demotes_and_preserves_blob(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode(
            "ep1", encoding_context=json.dumps({"salience_class": "substantive"})
        )
        graph.cues["ep1"] = _cue("ep1", surfaced_count=10, hit_count=5)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 1
        blob = json.loads(graph.episodes["ep1"].encoding_context)
        assert blob["salience_class"] == "substantive"
        assert blob[USAGE_DECAY_MARKER]
        assert blob[USAGE_DECAY_SURFACED] == 15
        assert result.cursors_next["episodes"][1] == "ep1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cue_kwargs",
        [
            {"surfaced_count": 15, "usage_used_count": 0.3},  # citation-scan used
            {"surfaced_count": 15, "used_count": 1},  # legacy used
            {"surfaced_count": 11},  # under N_min
        ],
    )
    async def test_used_or_undersurfaced_episode_never_demotes(self, cue_kwargs):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1")
        graph.cues["ep1"] = _cue("ep1", **cue_kwargs)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 0
        assert graph.episode_updates == []

    @pytest.mark.asyncio
    async def test_young_episode_never_demotes(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1", age_days=10.0)
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 0

    @pytest.mark.asyncio
    async def test_non_episodic_tier_exempt(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1", memory_tier="semantic")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 0
        assert result.exempt == 1

    @pytest.mark.asyncio
    async def test_already_marked_is_idempotent_and_feeds_prune_after_window(self):
        graph = FakeGraphStore()
        old_marker = _days_ago(40).isoformat()
        graph.episodes["ep1"] = _episode(
            "ep1", age_days=60.0, encoding_context=json.dumps({USAGE_DECAY_MARKER: old_marker})
        )
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 0
        assert result.already_marked == 1
        assert result.prune_feed_ready == 1  # marker older than 30d window
        assert graph.episode_updates == []


class TestEntityDemotion:
    @pytest.mark.asyncio
    async def test_chronic_entity_demotes(self):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg())

        assert result.demoted_entities == 1
        attrs = graph.entities["e1"].attributes
        assert attrs[USAGE_DECAY_MARKER]
        assert attrs[USAGE_DECAY_SURFACED] == 12
        # Attributes travel as a JSON string (the only shape every backend's
        # update_entity accepts for the attributes column).
        assert isinstance(graph.entity_updates[0][1]["attributes"], str)
        assert result.cursors_next["entities"] == "e1"

    @pytest.mark.asyncio
    async def test_used_entity_never_demotes(self):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)
        await activation.record_access("e1", time.time(), group_id=GROUP, tier="used")

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg())

        assert result.demoted_entities == 0
        assert graph.entity_updates == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entity_kwargs",
        [
            {"identity_core": True},
            {"entity_type": "Decision"},  # durable-class
            {"mat_tier": "semantic"},
        ],
    )
    async def test_identity_durable_and_tier_exempt(self, entity_kwargs):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1", **entity_kwargs)
        activation = await _surfaced_activation("e1", 20)

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg())

        assert result.demoted_entities == 0
        assert result.exempt == 1
        assert graph.entity_updates == []

    @pytest.mark.asyncio
    async def test_young_entity_never_demotes(self):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1", age_days=10.0)
        activation = await _surfaced_activation("e1", 20)

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg())

        assert result.demoted_entities == 0

    @pytest.mark.asyncio
    async def test_other_group_entities_skipped(self):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1")
        activation = MemoryActivationStore()
        now = time.time()
        for i in range(20):
            await activation.record_access("e1", now - i - 1, group_id="other-group")

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg())

        assert result.scanned_entities == 0
        assert result.demoted_entities == 0


class TestBoundsAndCursor:
    @pytest.mark.asyncio
    async def test_budget_bounds_window_and_cursor_resumes(self):
        graph = FakeGraphStore()
        for eid in ("e1", "e2", "e3"):
            graph.entities[eid] = _entity(eid)
        activation = MemoryActivationStore()
        now = time.time()
        for eid in ("e1", "e2", "e3"):
            for i in range(12):
                await activation.record_access(eid, now - 3600.0 * (i + 1), group_id=GROUP)
        cfg = _cfg(usage_decay_max_per_window=2)

        first = await run_usage_decay(graph, activation, GROUP, cfg=cfg)
        assert first.demoted_entities == 2
        assert first.cursors_next["entities"] == "e2"

        second = await run_usage_decay(
            graph, activation, GROUP, cfg=cfg, cursors=first.cursors_next
        )
        assert second.demoted_entities == 1
        assert [eid for eid, _u in graph.entity_updates] == ["e1", "e2", "e3"]

    @pytest.mark.asyncio
    async def test_cursor_wraps_after_full_sweep(self):
        graph = FakeGraphStore()
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)

        result = await run_usage_decay(
            graph, activation, GROUP, cfg=_cfg(), cursors={"entities": "e9"}
        )

        assert result.demoted_entities == 1  # wrapped around to e1

    @pytest.mark.asyncio
    async def test_expired_deadline_scans_nothing(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)

        result = await run_usage_decay(
            graph, activation, GROUP, cfg=_cfg(), deadline_ts=time.monotonic() - 1.0
        )

        assert result.demoted_total == 0
        assert result.scanned_episodes == 0
        assert result.scanned_entities == 0

    @pytest.mark.asyncio
    async def test_dry_run_counts_but_never_writes(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)

        result = await run_usage_decay(graph, activation, GROUP, cfg=_cfg(), dry_run=True)

        assert result.demoted_episodes == 1
        assert result.demoted_entities == 1
        assert result.dry_run is True
        assert graph.episode_updates == []
        assert graph.entity_updates == []


class TestPruneFeed:
    def test_marker_feeds_prune_only_after_measurement_window(self):
        cfg = _cfg()
        now = time.time()
        aged = _days_ago(40).isoformat()
        fresh = _days_ago(5).isoformat()
        assert prune_feed_eligible(aged, now=now, cfg=cfg)
        assert not prune_feed_eligible(fresh, now=now, cfg=cfg)
        assert not prune_feed_eligible("", now=now, cfg=cfg)

    def test_identity_durable_and_tier_never_feed_prune(self):
        cfg = _cfg()
        now = time.time()
        aged = _days_ago(40).isoformat()
        assert not prune_feed_eligible(aged, now=now, cfg=cfg, identity_core=True)
        assert not prune_feed_eligible(aged, now=now, cfg=cfg, entity_type="Decision")
        assert not prune_feed_eligible(aged, now=now, cfg=cfg, memory_tier="semantic")


# ── P5 boundary: the ranker may not read the marker ─────────────────────────


class TestP5Boundary:
    def test_presenter_flag_defaults_off(self):
        assert ActivationConfig().usage_decay_presenter_demotion_enabled is False

    def test_flag_off_returns_same_object(self):
        results = [
            {"result_type": "entity", "entity": {"attributes": {USAGE_DECAY_MARKER: "x"}}},
            {"result_type": "entity", "entity": {"attributes": {}}},
        ]
        out = demote_surfaced_never_used_results(results, _cfg())
        assert out is results  # byte identity, not just equality

    def test_flag_on_demotes_marker_carriers_stably(self):
        cfg = _cfg(usage_decay_presenter_demotion_enabled=True)
        marked_entity = {
            "result_type": "entity",
            "entity": {"attributes": {USAGE_DECAY_MARKER: "2026-06-01T00:00:00+00:00"}},
        }
        marked_episode = {
            "result_type": "episode",
            "episode": {
                "encoding_context": json.dumps({USAGE_DECAY_MARKER: "2026-06-01T00:00:00+00:00"})
            },
        }
        clean_a = {"result_type": "entity", "entity": {"attributes": {}}}
        clean_b = {"result_type": "episode", "episode": {"encoding_context": None}}
        out = demote_surfaced_never_used_results(
            [marked_entity, clean_a, marked_episode, clean_b], cfg
        )
        assert out == [clean_a, clean_b, marked_entity, marked_episode]

    @pytest.mark.parametrize("usage_ranking_enabled", [False, True])
    @pytest.mark.parametrize("presenter_flag", [False, True])
    def test_scorer_never_reads_marker(self, usage_ranking_enabled, presenter_flag):
        """The demotion marker in entity attributes can NEVER change scoring —
        at any flag state, including the presenter flag ON (P5: the marker is
        forgetting evidence, never ranking evidence)."""
        from engram.models.activation import ActivationState
        from engram.retrieval.scorer import score_candidates

        now = time.time()
        cfg = _cfg(
            usage_ranking_enabled=usage_ranking_enabled,
            usage_decay_presenter_demotion_enabled=presenter_flag,
        )
        candidates = [("ent-a", 0.9), ("ent-b", 0.7)]
        states = {
            "ent-a": ActivationState(node_id="ent-a", access_history=[now - 60.0], access_count=1),
        }

        def _score(entity_attributes):
            return [
                (r.node_id, r.score)
                for r in score_candidates(
                    candidates=list(candidates),
                    spreading_bonuses={},
                    hop_distances={},
                    seed_node_ids=set(),
                    activation_states=states,
                    now=now,
                    cfg=cfg,
                    entity_attributes=entity_attributes,
                )
            ]

        baseline = _score({"ent-a": {}, "ent-b": {}})
        marked = _score(
            {
                "ent-a": {USAGE_DECAY_MARKER: "2020-01-01T00:00:00+00:00"},
                "ent-b": {USAGE_DECAY_MARKER: "2020-01-01T00:00:00+00:00"},
            }
        )
        assert marked == baseline

    def test_marker_key_referenced_only_in_usage_decay_module(self):
        """Static pin: the marker key lives in exactly one production module —
        any new reader must go through its flag-guarded helpers."""
        root = Path(engram.__file__).parent
        offenders = [
            str(path.relative_to(root))
            for path in sorted(root.rglob("*.py"))
            if USAGE_DECAY_MARKER in path.read_text(encoding="utf-8")
            and path.relative_to(root) != Path("consolidation/usage_decay.py")
        ]
        assert offenders == []

    def test_retrieval_package_never_touches_usage_decay(self):
        """Static pin: no live-rank path (engram/retrieval) references
        usage-decay at all. Wiring the presenter demotion requires flipping
        the OFF-default flag AND updating this pin with the guarded call
        site — after a battery + continuity pass (eval gate)."""
        root = Path(engram.__file__).parent / "retrieval"
        offenders = [
            str(path.name)
            for path in sorted(root.rglob("*.py"))
            if "usage_decay" in path.read_text(encoding="utf-8")
        ]
        assert offenders == []


# ── Mop wiring: report block + kill switch ──────────────────────────────────


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _debt(**overrides) -> HygieneDebtSnapshot:
    import inspect

    fields = dict(
        deferred_evidence=0,
        pending_evidence=0,
        cue_only_episodes=0,
        cue_count=0,
        near_miss_count=0,
        open_adjudication=0,
        orphan_candidates=0,
        low_value_entities=0,
    )
    fields.update(overrides)
    allowed = set(inspect.signature(HygieneDebtSnapshot).parameters)
    return HygieneDebtSnapshot(**{k: v for k, v in fields.items() if k in allowed})


def _mop_patches():
    return (
        patch(
            "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
            new=AsyncMock(return_value=_debt(deferred_evidence=600)),
        ),
        patch(
            "engram.consolidation.evidence_drain.load_deferred_evidence",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_junk_evidence",
            new=AsyncMock(return_value={"rejected": 0}),
        ),
        patch(
            "engram.consolidation.evidence_drain.reject_evidence_rows",
            new=AsyncMock(return_value={"rejected": 0}),
        ),
        patch(
            "engram.consolidation.cue_hygiene.run_cue_hygiene",
            new=AsyncMock(return_value=type("R", (), {"to_dict": lambda self: {}})()),
        ),
        patch("engram.consolidation.phases.prune.PrunePhase"),
    )


async def _run_mop(graph_store, activation_store, cfg) -> dict:
    from engram.hygiene_ops import execute_hygiene_mop

    patches = _mop_patches()
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as prune_cls:
        prune_cls.return_value.execute = AsyncMock(
            return_value=(PhaseResult(phase="prune", status="completed"), [])
        )
        return await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=object(),
            activation_cfg=cfg,
            group_id=GROUP,
            budget=100,
        )


class TestMopWiring:
    @pytest.mark.asyncio
    async def test_mop_report_carries_usage_decay_block(self, engram_home: Path):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)
        graph.entities["e1"] = _entity("e1")
        activation = await _surfaced_activation("e1", 12)

        report = await _run_mop(graph, activation, ActivationConfig())

        block = report["mop"]["usage_decay"]
        assert block["demoted"] == {"episodes": 1, "entities": 1}
        assert block["budget"] == 100
        # Cursors persisted for the next window.
        state = json.loads((engram_home / "hygiene-state.json").read_text())
        assert GROUP in state["usage_decay_cursors"]

    @pytest.mark.asyncio
    async def test_kill_switch_skips_pass(self, engram_home: Path):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)

        report = await _run_mop(
            graph, MemoryActivationStore(), ActivationConfig(usage_decay_enabled=False)
        )

        assert report["mop"]["usage_decay"] == {
            "skipped": True,
            "reason": "usage_decay_enabled=False",
        }
        assert graph.episode_updates == []


class TestMarkerCodec:
    def test_decode_tolerates_garbage(self):
        assert decode_usage_decay_marker(None) == ""
        assert decode_usage_decay_marker("") == ""
        assert decode_usage_decay_marker("not json") == ""
        assert decode_usage_decay_marker(json.dumps(["list"])) == ""
        assert decode_usage_decay_marker(json.dumps({USAGE_DECAY_MARKER: 7})) == ""

    @pytest.mark.asyncio
    async def test_opaque_encoding_context_is_unmarkable(self):
        graph = FakeGraphStore()
        graph.episodes["ep1"] = _episode("ep1", encoding_context="reflect-cluster-key")
        graph.cues["ep1"] = _cue("ep1", surfaced_count=20)

        result = await run_usage_decay(graph, MemoryActivationStore(), GROUP, cfg=_cfg())

        assert result.demoted_episodes == 0
        assert result.exempt == 1
        assert graph.episodes["ep1"].encoding_context == "reflect-cluster-key"
