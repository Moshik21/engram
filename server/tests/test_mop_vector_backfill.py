"""Durable vector-debt drain in the hygiene mop (episode + cue vectors).

Episode vectors are written only at projection and cue capture-indexing is
best-effort, so shell+quiet installs regrow vector debt forever. The mop
drains it under a per-window budget. Provider breakage must be LOUD but never
fail the mop (the M2.6 disaster was a broken provider staying invisible).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.models.consolidation import PhaseResult


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _episode(eid: str, content: str = "some episode content", created=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=eid, group_id="g", content=content, deleted_at=None, created_at=created
    )


def _cue_row(eid: str, text: str = "cue text", created=None) -> dict:
    return {"episode_id": eid, "group_id": "g", "cue_text": text, "created_at": created}


class FakeGraphStore:
    def __init__(self, episodes=(), cue_rows=()):
        self.episodes = list(episodes)
        self.cue_rows = list(cue_rows)

    async def get_episodes(self, group_id=None, limit=50, offset=0):
        return self.episodes[offset : offset + limit]

    async def _fetch_episode_cues_bulk(self, group_id):
        return list(self.cue_rows)


class FakeSearchIndex:
    """Vector-capable search index with a countable fake provider."""

    def __init__(
        self,
        *,
        episode_vecs=(),
        cue_vecs=(),
        broken_provider=False,
        raising_provider=False,
        failing_cue_ids=(),
    ):
        self._embeddings_enabled = True
        self.episode_vecs = set(episode_vecs)
        self.cue_vecs = set(cue_vecs)
        self.broken_provider = broken_provider
        self.raising_provider = raising_provider
        self.failing_cue_ids = set(failing_cue_ids)
        self.embed_calls = 0
        self.indexed_episode_ids: list[str] = []
        self.indexed_cue_ids: list[str] = []

    async def _embed_texts(self, texts):
        self.embed_calls += 1
        if self.raising_provider:
            raise RuntimeError("fastembed model missing")
        if self.broken_provider:
            return []
        return [[0.1, 0.2] for _ in texts]

    async def get_episode_embeddings(self, ids, group_id=None):
        return {i: [0.1, 0.2] for i in ids if i in self.episode_vecs}

    async def get_cue_embeddings(self, ids, group_id=None):
        return {i: [0.1, 0.2] for i in ids if i in self.cue_vecs}

    async def index_episode(self, episode):
        vecs = await self._embed_texts([episode.content])
        if not vecs:
            return  # mirrors production: embed failure swallowed, not raised
        self.episode_vecs.add(str(episode.id))
        self.indexed_episode_ids.append(str(episode.id))

    async def index_episode_cue(self, cue):
        if cue.episode_id in self.failing_cue_ids:
            raise RuntimeError("cue index write failed")
        vecs = await self._embed_texts([cue.cue_text])
        if not vecs:
            raise RuntimeError("cue embed failed")
        self.cue_vecs.add(cue.episode_id)
        self.indexed_cue_ids.append(cue.episode_id)


class CensusSearchIndex(FakeSearchIndex):
    """Census-only presence (no by-id probe), mimicking helix-native INCLUDING
    the measured undercount: the census sweep never reflects vectors added by
    the drain, so only the durable cursor can provide progression."""

    get_episode_embeddings = None  # not callable -> census presence path
    get_cue_embeddings = None

    async def _embed_text(self, text):
        vecs = await self._embed_texts([text])
        return vecs[0] if vecs else []

    async def _vector_search_episodes(self, vec, limit, group_id=None):
        return [], [], True

    async def _vector_search_cues(self, vec, limit, group_id=None):
        return [], [], True


def _debt(**overrides) -> HygieneDebtSnapshot:
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


async def _run_mop(graph_store, search_index, *, dry_run: bool = False) -> dict:
    from engram.hygiene_ops import execute_hygiene_mop

    patches = (
        patch(
            "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
            new=AsyncMock(return_value=_debt()),
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
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5] as prune_cls:
        prune_cls.return_value.execute = AsyncMock(
            return_value=(PhaseResult(phase="prune", status="completed"), [])
        )
        return await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=object(),
            search_index=search_index,
            activation_cfg=ActivationConfig(),
            group_id="g",
            budget=100,
            dry_run=dry_run,
        )


class TestMopVectorBackfill:
    @pytest.mark.asyncio
    async def test_backfills_missing_episode_and_cue_vectors(self, engram_home: Path):
        graph = FakeGraphStore(
            episodes=[_episode("ep1"), _episode("ep2"), _episode("ep3")],
            cue_rows=[_cue_row("ep1"), _cue_row("ep2")],
        )
        search = FakeSearchIndex(episode_vecs={"ep1"})  # ep1 already vectored

        report = await _run_mop(graph, search)

        vb = report["mop"]["vector_backfill"]
        assert vb["episodes"] == 2
        assert vb["cues"] == 2
        assert vb["failed"] == 0
        assert vb["missing_before"] == {"episodes": 2, "cues": 2}
        assert sorted(search.indexed_episode_ids) == ["ep2", "ep3"]
        assert sorted(search.indexed_cue_ids) == ["ep1", "ep2"]

    @pytest.mark.asyncio
    async def test_budget_respected(self, engram_home: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("engram.hygiene_ops._VECTOR_BACKFILL_EPISODES_MAX", 2)
        monkeypatch.setattr("engram.hygiene_ops._VECTOR_BACKFILL_CUES_MAX", 1)
        graph = FakeGraphStore(
            episodes=[_episode(f"ep{i}") for i in range(5)],
            cue_rows=[_cue_row(f"ep{i}") for i in range(3)],
        )
        search = FakeSearchIndex()

        report = await _run_mop(graph, search)

        vb = report["mop"]["vector_backfill"]
        assert vb["episodes"] == 2
        assert vb["cues"] == 1
        assert vb["missing_before"] == {"episodes": 5, "cues": 3}
        assert vb["budgets"] == {"episodes": 2, "cues": 1}
        assert len(search.indexed_episode_ids) == 2
        assert len(search.indexed_cue_ids) == 1

    @pytest.mark.asyncio
    async def test_dry_run_skips_with_label(self, engram_home: Path):
        graph = FakeGraphStore(episodes=[_episode("ep1")], cue_rows=[_cue_row("ep1")])
        search = FakeSearchIndex()

        report = await _run_mop(graph, search, dry_run=True)

        assert report["mop"]["vector_backfill"] == {"skipped": True, "reason": "dry_run"}
        assert search.embed_calls == 0
        assert search.indexed_episode_ids == []
        assert search.indexed_cue_ids == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_kwargs",
        [
            {"broken_provider": True},
            {"raising_provider": True},
        ],
    )
    async def test_provider_broken_is_loud_but_nonfatal(
        self, engram_home: Path, caplog: pytest.LogCaptureFixture, provider_kwargs: dict
    ):
        graph = FakeGraphStore(episodes=[_episode("ep1")], cue_rows=[_cue_row("ep1")])
        search = FakeSearchIndex(**provider_kwargs)

        with caplog.at_level(logging.WARNING, logger="engram.hygiene_ops"):
            report = await _run_mop(graph, search)

        vb = report["mop"]["vector_backfill"]
        assert vb == {"status": "provider_unavailable", "episodes": 0}
        assert any("provider unavailable" in r.message for r in caplog.records)
        # The mop itself completed — other sections are still present.
        assert "prune" in report["mop"]

    @pytest.mark.asyncio
    async def test_zero_missing_vectors_does_zero_embed_calls(self, engram_home: Path):
        """Regression pin: full coverage must not spend a single embed call."""
        graph = FakeGraphStore(
            episodes=[_episode("ep1"), _episode("ep2")],
            cue_rows=[_cue_row("ep1")],
        )
        search = FakeSearchIndex(episode_vecs={"ep1", "ep2"}, cue_vecs={"ep1"})

        report = await _run_mop(graph, search)

        vb = report["mop"]["vector_backfill"]
        assert vb["episodes"] == 0
        assert vb["cues"] == 0
        assert vb["failed"] == 0
        assert search.embed_calls == 0

    @pytest.mark.asyncio
    async def test_vector_less_search_index_skips_labeled(self, engram_home: Path):
        """Lite/FTS5 (no _embeddings_enabled attribute) gets a labeled skip."""
        report = await _run_mop(FakeGraphStore(), object())

        vb = report["mop"]["vector_backfill"]
        assert vb["skipped"] is True
        assert "no embedding support" in vb["reason"]

    @pytest.mark.asyncio
    async def test_embeddings_disabled_skips_labeled(self, engram_home: Path):
        search = FakeSearchIndex()
        search._embeddings_enabled = False

        report = await _run_mop(FakeGraphStore(episodes=[_episode("ep1")]), search)

        vb = report["mop"]["vector_backfill"]
        assert vb == {"skipped": True, "reason": "embeddings disabled"}
        assert search.embed_calls == 0

    @pytest.mark.asyncio
    async def test_partial_cue_failure_counts_failed(self, engram_home: Path):
        graph = FakeGraphStore(cue_rows=[_cue_row("ep1"), _cue_row("ep2")])
        search = FakeSearchIndex(failing_cue_ids={"ep1"})

        report = await _run_mop(graph, search)

        vb = report["mop"]["vector_backfill"]
        assert vb["cues"] == 1
        assert vb["failed"] == 1
        assert search.indexed_cue_ids == ["ep2"]

    @pytest.mark.asyncio
    async def test_census_mode_cursor_progresses_across_windows(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """THE live failure mode: the helix-native ANN census undercounts, so
        without a durable cursor every window would re-embed the same first
        budget of items and grow duplicate vectors. Two mop windows must drain
        two DIFFERENT items, oldest first."""
        import json

        monkeypatch.setattr("engram.hygiene_ops._VECTOR_BACKFILL_EPISODES_MAX", 1)
        monkeypatch.setattr("engram.hygiene_ops._VECTOR_BACKFILL_CUES_MAX", 1)
        graph = FakeGraphStore(
            episodes=[_episode("ep2", created=200.0), _episode("ep1", created=100.0)],
            cue_rows=[
                _cue_row("ep2", created="2026-07-02T00:00:00+00:00"),
                _cue_row("ep1", created="2026-07-01T00:00:00+00:00"),
            ],
        )
        search = CensusSearchIndex()

        r1 = await _run_mop(graph, search)
        r2 = await _run_mop(graph, search)

        assert r1["mop"]["vector_backfill"]["episodes"] == 1
        assert r2["mop"]["vector_backfill"]["episodes"] == 1
        assert search.indexed_episode_ids == ["ep1", "ep2"]  # no re-embedding
        assert search.indexed_cue_ids == ["ep1", "ep2"]
        state = json.loads((engram_home / "hygiene-state.json").read_text())
        assert state["vector_backfill_cursors"]["g"]["episodes"] == [200.0, "ep2"]

    @pytest.mark.asyncio
    async def test_provider_broken_census_mode_does_not_advance_cursor(self, engram_home: Path):
        import json

        graph = FakeGraphStore(episodes=[_episode("ep1", created=100.0)])
        search = CensusSearchIndex(broken_provider=True)

        report = await _run_mop(graph, search)

        assert report["mop"]["vector_backfill"] == {
            "status": "provider_unavailable",
            "episodes": 0,
        }
        state_path = engram_home / "hygiene-state.json"
        if state_path.exists():
            assert "vector_backfill_cursors" not in json.loads(state_path.read_text())


class TestBackfillFunctionEdgeCases:
    @pytest.mark.asyncio
    async def test_swallowed_episode_embed_failures_surface_via_stats(self):
        """index_episode implementations swallow embed failures internally;
        the backfill recovers the true count from the index's _embed_stats so a
        half-broken provider can never report a clean drain."""
        from engram.storage.index_completeness import backfill_missing_episode_vectors

        class SwallowingIndex(FakeSearchIndex):
            def __init__(self):
                super().__init__()
                self._embed_stats = {"episodes_failed": 0}
                self._fail_after = 1

            async def index_episode(self, episode):
                if len(self.indexed_episode_ids) >= self._fail_after:
                    self._embed_stats["episodes_failed"] += 1
                    return  # swallowed failure, production-style
                self.indexed_episode_ids.append(str(episode.id))

        graph = FakeGraphStore(episodes=[_episode("ep1"), _episode("ep2"), _episode("ep3")])
        search = SwallowingIndex()

        result = await backfill_missing_episode_vectors(graph, search, "g")

        assert result.attempted == 3
        assert result.indexed == 1
        assert result.failed == 2

    @pytest.mark.asyncio
    async def test_store_without_listing_apis_returns_empty(self):
        from engram.storage.index_completeness import (
            backfill_missing_cue_vectors,
            backfill_missing_episode_vectors,
        )

        ep = await backfill_missing_episode_vectors(object(), FakeSearchIndex(), "g")
        cue = await backfill_missing_cue_vectors(object(), FakeSearchIndex(), "g")

        assert (ep.attempted, ep.indexed, ep.failed) == (0, 0, 0)
        assert (cue.attempted, cue.indexed, cue.failed) == (0, 0, 0)

    @pytest.mark.asyncio
    async def test_dry_run_plans_without_indexing(self):
        from engram.storage.index_completeness import backfill_missing_episode_vectors

        graph = FakeGraphStore(episodes=[_episode("ep1"), _episode("ep2")])
        search = FakeSearchIndex()

        result = await backfill_missing_episode_vectors(graph, search, "g", dry_run=True)

        assert result.attempted == 2
        assert result.indexed == 0
        assert search.embed_calls == 0

    @pytest.mark.asyncio
    async def test_census_fallback_via_vector_search(self):
        """Without by-id probes the ANN census (one embed call) finds present
        vectors — the helix-native path, where only _vector_search_* exists."""
        from engram.storage.index_completeness import backfill_missing_episode_vectors

        class CensusIndex(FakeSearchIndex):
            get_episode_embeddings = None  # not callable -> census path
            get_cue_embeddings = None

            async def _embed_text(self, text):
                self.embed_calls += 1
                return [0.1, 0.2]

            async def _vector_search_episodes(self, vec, limit, group_id=None):
                return [(i, 0.9) for i in self.episode_vecs], [], True

        graph = FakeGraphStore(episodes=[_episode("ep1"), _episode("ep2")])
        search = CensusIndex(episode_vecs={"ep1"})

        result = await backfill_missing_episode_vectors(graph, search, "g")

        assert result.missing_before == 1
        assert result.indexed == 1
        assert search.indexed_episode_ids == ["ep2"]

    @pytest.mark.asyncio
    async def test_cursor_filters_and_advances_in_census_mode(self):
        from engram.storage.index_completeness import backfill_missing_episode_vectors

        graph = FakeGraphStore(
            episodes=[
                _episode("ep1", created=100.0),
                _episode("ep2", created=200.0),
                _episode("ep3", created=300.0),
            ]
        )
        search = CensusSearchIndex()

        result = await backfill_missing_episode_vectors(graph, search, "g", cursor=(100.0, "ep1"))

        assert result.indexed_ids == ["ep2", "ep3"]
        assert result.cursor_next == (300.0, "ep3")

    @pytest.mark.asyncio
    async def test_exact_probe_mode_ignores_cursor_and_sets_none(self):
        """With a real by-id probe, presence is exact: progression comes from
        the probe itself, so the cursor neither filters nor advances."""
        from engram.storage.index_completeness import backfill_missing_episode_vectors

        graph = FakeGraphStore(
            episodes=[_episode("ep1", created=100.0), _episode("ep2", created=200.0)]
        )
        search = FakeSearchIndex(episode_vecs={"ep2"})

        result = await backfill_missing_episode_vectors(graph, search, "g", cursor=(500.0, "zz"))

        assert result.indexed_ids == ["ep1"]
        assert result.cursor_next is None
