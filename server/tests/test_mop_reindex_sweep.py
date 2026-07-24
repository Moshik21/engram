"""M1.3 historical re-index sweep in the hygiene mop (agent-experience D5).

The 2026-07-21 emergency backfill wrote ONE coarse vector per episode from
1,200-char truncated text (no chunks). The mop sweep replaces those with real
``index_episode`` output (full + chunk vectors), de-indexes machinery-class
episodes (D2), and skips episodes already properly indexed by capture —
discriminated by chunk-vector presence (coarse backfill wrote 0 chunks; the
capture path writes >=1 for chunkable content).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.models.consolidation import PhaseResult
from engram.storage.index_completeness import (
    ByIdVectorProbeUnavailableError,
    reindex_sweep_episodes,
)

# Long enough to chunk (> fake CHUNK_THRESHOLD), classifies substantive.
SUBSTANTIVE_LONG = (
    "Konner decided the reranker threshold should be 0.4 after the ablation "
    "runs showed diminishing returns above it. The cross encoder stays on "
    "for identity queries because those regressed without it. "
) * 2
SUBSTANTIVE_SHORT = "Konner prefers dark roast coffee."


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _episode(eid: str, content: str = SUBSTANTIVE_LONG, created=None, source=None):
    return SimpleNamespace(
        id=eid,
        group_id="g",
        content=content,
        deleted_at=None,
        created_at=created,
        source=source,
        attachments=None,
    )


def _machinery_episode(eid: str, created=None):
    return _episode(eid, content="New session started", created=created, source="auto:session")


class FakeGraphStore:
    def __init__(self, episodes=()):
        self.episodes = list(episodes)

    async def get_episodes(self, group_id=None, limit=50, offset=0):
        return self.episodes[offset : offset + limit]


class FakeSearchIndex:
    """Row-level vector index fake: by-id probes, deletes, real-shaped
    index_episode (full vector row + chunk rows for chunkable content)."""

    CHUNK_THRESHOLD = 200

    def __init__(
        self,
        *,
        broken_provider=False,
        failing_delete_ids=(),
        fail_index_ids=(),
        swallow_index_ids=(),
    ):
        self._embeddings_enabled = True
        self._embed_stats = {"episodes_failed": 0, "episodes_indexed": 0}
        self.rows: dict[str, list[dict]] = {"episode": [], "chunk": [], "cue": []}
        self._next_row_id = 0
        self.broken_provider = broken_provider
        self.failing_delete_ids = set(failing_delete_ids)
        self.fail_index_ids = set(fail_index_ids)
        self.swallow_index_ids = set(swallow_index_ids)
        self.embed_calls = 0
        self.index_episode_calls: list[str] = []

    def seed_row(self, kind: str, episode_id: str, chunk_index: int | None = None) -> dict:
        self._next_row_id += 1
        row = {"id": f"r{self._next_row_id}", "episode_id": episode_id, "group_id": "g"}
        if chunk_index is not None:
            row["chunk_index"] = chunk_index
        self.rows[kind].append(row)
        return row

    def episode_ids_with_rows(self, kind: str) -> list[str]:
        return [r["episode_id"] for r in self.rows[kind]]

    async def _embed_texts(self, texts):
        self.embed_calls += 1
        if self.broken_provider:
            return []
        return [[0.1, 0.2] for _ in texts]

    async def find_vector_rows_by_episode_ids(self, kind, episode_ids, group_id):
        wanted = {str(i) for i in episode_ids}
        return [dict(r) for r in self.rows[kind] if r["episode_id"] in wanted]

    async def delete_vector_row(self, kind, helix_id):
        for row in self.rows[kind]:
            if row["id"] == helix_id:
                if row["episode_id"] in self.failing_delete_ids:
                    return False
                self.rows[kind].remove(row)
                return True
        return False

    async def get_episode_embeddings(self, ids, group_id=None):
        present = {r["episode_id"] for r in self.rows["episode"]}
        return {i: [0.1, 0.2] for i in ids if i in present}

    async def index_episode(self, episode):
        eid = str(episode.id)
        self.index_episode_calls.append(eid)
        if eid in self.fail_index_ids:
            raise RuntimeError("index write failed")
        if eid in self.swallow_index_ids:
            # Production-style: embed failure swallowed, only the stat moves.
            self._embed_stats["episodes_failed"] += 1
            return
        vecs = await self._embed_texts([episode.content])
        if not vecs:
            self._embed_stats["episodes_failed"] += 1
            return
        self.seed_row("episode", eid)
        if len(episode.content) > self.CHUNK_THRESHOLD:
            self.seed_row("chunk", eid, 0)
            self.seed_row("chunk", eid, 1)
        self._embed_stats["episodes_indexed"] += 1


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


async def _run_mop(
    graph_store, search_index, *, dry_run: bool = False, sweep_enabled: bool = True
) -> dict:
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
            activation_cfg=ActivationConfig(reindex_sweep_enabled=sweep_enabled),
            group_id="g",
            budget=100,
            dry_run=dry_run,
        )


class TestMopReindexSweep:
    @pytest.mark.asyncio
    async def test_reindexes_coarse_deindexes_machinery_skips_capture_indexed(
        self, engram_home: Path
    ):
        """The three-way decision on one window, by-id census before/after."""
        graph = FakeGraphStore(
            episodes=[
                _episode("ep_coarse", created=100.0),
                _machinery_episode("ep_mach", created=200.0),
                _episode("ep_cap", created=300.0),
            ]
        )
        search = FakeSearchIndex()
        search.seed_row("episode", "ep_coarse")  # coarse backfill vector, 0 chunks
        search.seed_row("episode", "ep_mach")  # machinery noise, vector-indexed
        cap_row = search.seed_row("episode", "ep_cap")  # capture-indexed: has chunks
        search.seed_row("chunk", "ep_cap", 0)
        search.seed_row("chunk", "ep_cap", 1)

        report = await _run_mop(graph, search)

        rs = report["mop"]["reindex_sweep"]
        assert rs["reindexed"] == 1
        assert rs["deindexed_machinery"] == 1
        assert rs["skipped_capture_indexed"] == 1
        assert rs["failed"] == 0
        assert rs["complete"] is True
        assert rs["cursor"] == [300.0, "ep_cap"]
        # Census: coarse row replaced exactly once (no duplicates), machinery
        # fully de-indexed, capture-indexed rows byte-untouched.
        assert search.episode_ids_with_rows("episode").count("ep_coarse") == 1
        assert search.episode_ids_with_rows("episode").count("ep_mach") == 0
        assert search.episode_ids_with_rows("chunk").count("ep_coarse") == 2
        assert cap_row in search.rows["episode"]
        assert search.index_episode_calls == ["ep_coarse"]

    @pytest.mark.asyncio
    async def test_completed_sweep_is_one_time(self, engram_home: Path):
        graph = FakeGraphStore(episodes=[_episode("ep1", created=100.0)])
        search = FakeSearchIndex()
        search.seed_row("episode", "ep1")

        r1 = await _run_mop(graph, search)
        r2 = await _run_mop(graph, search)

        assert r1["mop"]["reindex_sweep"]["complete"] is True
        assert r2["mop"]["reindex_sweep"] == {"skipped": True, "reason": "sweep complete"}
        assert search.index_episode_calls == ["ep1"]  # never re-swept
        state = json.loads((engram_home / "hygiene-state.json").read_text())
        assert state["reindex_sweep"]["g"]["complete"] is True

    @pytest.mark.asyncio
    async def test_budget_and_cursor_progress_across_windows(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("engram.hygiene_ops._REINDEX_SWEEP_EPISODES_MAX", 2)
        eps = [_episode(f"ep{i}", created=float(i * 100)) for i in range(1, 5)]
        graph = FakeGraphStore(episodes=eps)
        search = FakeSearchIndex()
        for ep in eps:
            search.seed_row("episode", ep.id)

        r1 = await _run_mop(graph, search)
        r2 = await _run_mop(graph, search)

        assert r1["mop"]["reindex_sweep"]["reindexed"] == 2
        assert r1["mop"]["reindex_sweep"]["complete"] is False
        assert r1["mop"]["reindex_sweep"]["cursor"] == [200.0, "ep2"]
        assert r2["mop"]["reindex_sweep"]["reindexed"] == 2
        assert r2["mop"]["reindex_sweep"]["complete"] is True
        # Oldest-first, each episode exactly once — no window re-processing.
        assert search.index_episode_calls == ["ep1", "ep2", "ep3", "ep4"]
        for eid in ("ep1", "ep2", "ep3", "ep4"):
            assert search.episode_ids_with_rows("episode").count(eid) == 1

    @pytest.mark.asyncio
    async def test_provider_unavailable_is_loud_but_nonfatal(
        self, engram_home: Path, caplog: pytest.LogCaptureFixture
    ):
        graph = FakeGraphStore(episodes=[_episode("ep1", created=100.0)])
        search = FakeSearchIndex(broken_provider=True)
        search.seed_row("episode", "ep1")

        with caplog.at_level(logging.WARNING, logger="engram.hygiene_ops"):
            report = await _run_mop(graph, search)

        assert report["mop"]["reindex_sweep"] == {"status": "provider_unavailable"}
        assert any(
            "reindex sweep" in r.message and "provider unavailable" in r.message
            for r in caplog.records
        )
        assert "prune" in report["mop"]  # mop itself completed
        # No coarse row was deleted and no cursor advanced — retry next window.
        assert search.episode_ids_with_rows("episode") == ["ep1"]
        state_path = engram_home / "hygiene-state.json"
        if state_path.exists():
            assert "reindex_sweep" not in json.loads(state_path.read_text())

    @pytest.mark.asyncio
    async def test_dry_run_skips_with_label(self, engram_home: Path):
        graph = FakeGraphStore(episodes=[_episode("ep1", created=100.0)])
        search = FakeSearchIndex()
        search.seed_row("episode", "ep1")

        report = await _run_mop(graph, search, dry_run=True)

        assert report["mop"]["reindex_sweep"] == {"skipped": True, "reason": "dry_run"}
        assert search.index_episode_calls == []

    @pytest.mark.asyncio
    async def test_embeddings_disabled_skips_labeled(self, engram_home: Path):
        search = FakeSearchIndex()
        search._embeddings_enabled = False

        report = await _run_mop(FakeGraphStore(episodes=[_episode("ep1")]), search)

        assert report["mop"]["reindex_sweep"] == {
            "skipped": True,
            "reason": "embeddings disabled",
        }

    @pytest.mark.asyncio
    async def test_vector_less_search_index_skips_labeled(self, engram_home: Path):
        report = await _run_mop(FakeGraphStore(), object())

        rs = report["mop"]["reindex_sweep"]
        assert rs["skipped"] is True
        assert "no embedding support" in rs["reason"]


class TestReindexSweepFunction:
    @pytest.mark.asyncio
    async def test_chunk_presence_discriminator_pins(self):
        """THE one-time interplay pin: identical content, but the episode with
        >=1 chunk row (capture-indexed post-punch-list) is skipped untouched,
        while the chunk-less coarse row is delete-then-reindexed."""
        graph = FakeGraphStore(
            episodes=[_episode("coarse", created=100.0), _episode("capture", created=200.0)]
        )
        search = FakeSearchIndex()
        coarse_row = search.seed_row("episode", "coarse")
        capture_row = search.seed_row("episode", "capture")
        search.seed_row("chunk", "capture", 0)  # >=1 chunk == capture path wrote it

        result = await reindex_sweep_episodes(graph, search, "g", machinery=lambda ep: False)

        assert result.reindexed == 1
        assert result.skipped_capture_indexed == 1
        assert search.index_episode_calls == ["coarse"]
        assert capture_row in search.rows["episode"]  # untouched
        assert coarse_row not in search.rows["episode"]  # replaced
        assert search.episode_ids_with_rows("episode").count("coarse") == 1
        assert result.complete is True
        assert result.cursor_next == (200.0, "capture")

    @pytest.mark.asyncio
    async def test_delete_failure_blocks_index_and_cursor(self):
        """A failed coarse-row delete must NOT be followed by index_episode
        (AddV appends — that would duplicate) and must block the cursor so
        the episode is retried next window."""
        graph = FakeGraphStore(
            episodes=[_episode("ep1", created=100.0), _episode("ep2", created=200.0)]
        )
        search = FakeSearchIndex(failing_delete_ids={"ep1"})
        search.seed_row("episode", "ep1")
        search.seed_row("episode", "ep2")

        result = await reindex_sweep_episodes(graph, search, "g", machinery=lambda ep: False)

        assert result.failed == 1
        assert result.reindexed == 1
        assert "ep1" not in search.index_episode_calls  # no duplicate risk
        assert search.episode_ids_with_rows("episode").count("ep1") == 1
        assert result.cursor_next is None  # prefix broken at ep1
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_swallowed_embed_failure_counts_failed_and_blocks_cursor(self):
        """index_episode swallows embed failures internally; the _embed_stats
        delta must surface them so a half-broken provider never reports a
        clean sweep."""
        graph = FakeGraphStore(
            episodes=[_episode("ep1", created=100.0), _episode("ep2", created=200.0)]
        )
        search = FakeSearchIndex(swallow_index_ids={"ep1"})
        search.seed_row("episode", "ep1")
        search.seed_row("episode", "ep2")

        result = await reindex_sweep_episodes(graph, search, "g", machinery=lambda ep: False)

        assert result.failed == 1
        assert result.reindexed == 1
        assert result.cursor_next is None
        assert result.complete is False

    @pytest.mark.asyncio
    async def test_machinery_only_window_deindexes_without_provider(self):
        """De-indexing needs no embeddings: a dead provider must not block a
        pure-machinery window (deletes proceed, nothing raises)."""
        graph = FakeGraphStore(episodes=[_machinery_episode("m1", created=100.0)])
        search = FakeSearchIndex(broken_provider=True)
        search.seed_row("episode", "m1")
        search.seed_row("chunk", "m1", 0)

        result = await reindex_sweep_episodes(graph, search, "g")

        assert result.deindexed_machinery == 1
        assert result.rows_deleted == 2
        assert search.episode_ids_with_rows("episode") == []
        assert search.episode_ids_with_rows("chunk") == []
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_machinery_without_rows_advances_cursor_uncounted(self):
        graph = FakeGraphStore(episodes=[_machinery_episode("m1", created=100.0)])
        search = FakeSearchIndex()

        result = await reindex_sweep_episodes(graph, search, "g")

        assert result.deindexed_machinery == 0
        assert result.scanned == 1
        assert result.cursor_next == (100.0, "m1")
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_by_id_probe_unavailable_skips_with_reason(self):
        class ProbelessIndex(FakeSearchIndex):
            async def find_vector_rows_by_episode_ids(self, kind, episode_ids, group_id):
                raise ByIdVectorProbeUnavailableError("find_episode_chunk_vectors_by_ids")

        graph = FakeGraphStore(episodes=[_episode("ep1", created=100.0)])
        search = ProbelessIndex()

        result = await reindex_sweep_episodes(graph, search, "g")

        assert result.skipped_reason is not None
        assert result.cursor_next is None
        assert result.complete is False
        assert search.index_episode_calls == []

    @pytest.mark.asyncio
    async def test_deadline_closes_window_with_cursor_kept(self):
        import time as _time

        graph = FakeGraphStore(
            episodes=[_episode("ep1", created=100.0), _episode("ep2", created=200.0)]
        )
        search = FakeSearchIndex()
        search.seed_row("episode", "ep1")
        search.seed_row("episode", "ep2")

        result = await reindex_sweep_episodes(
            graph, search, "g", machinery=lambda ep: False, deadline_ts=_time.monotonic() - 1.0
        )

        assert result.stopped == "deadline"
        assert result.scanned == 0
        assert result.complete is False
        assert result.cursor_next is None

    @pytest.mark.asyncio
    async def test_cursor_filters_already_swept_episodes(self):
        graph = FakeGraphStore(
            episodes=[_episode("ep1", created=100.0), _episode("ep2", created=200.0)]
        )
        search = FakeSearchIndex()
        search.seed_row("episode", "ep1")
        search.seed_row("episode", "ep2")

        result = await reindex_sweep_episodes(
            graph, search, "g", machinery=lambda ep: False, cursor=(100.0, "ep1")
        )

        assert search.index_episode_calls == ["ep2"]
        assert result.cursor_next == (200.0, "ep2")
        assert result.complete is True

    @pytest.mark.asyncio
    async def test_short_substantive_reindexes_without_duplicates(self):
        """Short content (below chunk threshold) is indistinguishable from a
        coarse row — it gets one harmless delete+re-embed, never a duplicate."""
        graph = FakeGraphStore(
            episodes=[_episode("shorty", content=SUBSTANTIVE_SHORT, created=100.0)]
        )
        search = FakeSearchIndex()
        search.seed_row("episode", "shorty")

        result = await reindex_sweep_episodes(graph, search, "g")

        assert result.reindexed == 1
        assert search.episode_ids_with_rows("episode").count("shorty") == 1
        assert search.episode_ids_with_rows("chunk") == []


class TestReindexSweepNewestFirst:
    async def test_newest_first_processes_recent_before_old(self):
        # Three coarse (chunkless) substantive episodes at increasing ages.
        graph = FakeGraphStore(
            [
                _episode("ep_old", created=100.0),
                _episode("ep_mid", created=200.0),
                _episode("ep_new", created=300.0),
            ]
        )
        search = FakeSearchIndex()
        for eid in ("ep_old", "ep_mid", "ep_new"):
            search.seed_row("episode", eid)  # coarse row, no chunks

        # Budget 1, newest_first: only the newest episode is re-indexed.
        result = await reindex_sweep_episodes(
            graph, search, "g", max_episodes=1, machinery=lambda ep: False, newest_first=True
        )
        assert search.index_episode_calls == ["ep_new"]
        # Cursor is the newest key; a next window with it continues older.
        await reindex_sweep_episodes(
            graph,
            search,
            "g",
            max_episodes=1,
            cursor=result.cursor_next,
            machinery=lambda ep: False,
            newest_first=True,
        )
        assert search.index_episode_calls == ["ep_new", "ep_mid"]

    async def test_oldest_first_default_unchanged(self):
        graph = FakeGraphStore(
            [_episode("ep_old", created=100.0), _episode("ep_new", created=300.0)]
        )
        search = FakeSearchIndex()
        for eid in ("ep_old", "ep_new"):
            search.seed_row("episode", eid)
        await reindex_sweep_episodes(
            graph, search, "g", max_episodes=1, machinery=lambda ep: False
        )
        assert search.index_episode_calls == ["ep_old"]


class TestMopReindexSweepGate:
    async def test_sweep_disabled_by_default(self):
        graph = FakeGraphStore([_episode("ep1")])
        search = FakeSearchIndex()
        search.seed_row("episode", "ep1")
        report = await _run_mop(graph, search, sweep_enabled=False)
        assert report["mop"]["reindex_sweep"] == {
            "skipped": True,
            "reason": "reindex_sweep_enabled=False",
        }
        assert search.index_episode_calls == []  # never touched the index
