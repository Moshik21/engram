"""M1.1 + M1.3: tiered usage_events store, snapshot v2, confirmed-event journal.

Sequencing invariant (RF goal): these are WRITERS + DURABILITY only — nothing
on the ranking path reads usage_events, and every existing caller keeps the
default "surfaced" tier, which appends to access_history exactly as before.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.activation import DEFAULT_USAGE_TIER_WEIGHTS, ActivationState
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from tests.conftest import MockExtractor


def _store(tmp_path: Path, cfg: ActivationConfig | None = None) -> MemoryActivationStore:
    return MemoryActivationStore(
        cfg=cfg or ActivationConfig(),
        journal_path=tmp_path / "activation-usage-journal.jsonl",
    )


class TestTieredRecordAccess:
    @pytest.mark.asyncio
    async def test_default_surfaced_is_hygiene_only(self, tmp_path: Path):
        store = _store(tmp_path)
        await store.record_access("e1", 1000.0, group_id="g")
        state = await store.get_activation("e1")
        assert state is not None
        assert state.access_history == [1000.0]
        assert state.access_count == 1
        assert state.usage_events == []
        assert state.usage_weight_sum == 0.0
        assert not store.journal_path.exists()

    @pytest.mark.asyncio
    async def test_nonzero_tiers_append_to_both_stores(self, tmp_path: Path):
        store = _store(tmp_path)
        await store.record_access("e1", 1000.0, group_id="g", tier="mentioned")
        await store.record_access("e1", 2000.0, group_id="g", tier="used")
        state = await store.get_activation("e1")
        assert state is not None
        # Hygiene store gets every tier (prune/mature inputs whole).
        assert state.access_history == [1000.0, 2000.0]
        # Ranking store gets only the tier-weighted events (F10: mentioned=0.1).
        assert state.usage_events == [(1000.0, 0.1), (2000.0, 0.3)]
        assert state.usage_weight_sum == pytest.approx(0.4)
        assert state.n_eff == pytest.approx(0.4)
        assert state.usage_last_ts == 2000.0
        # used/mentioned are NOT journaled (crash-loss acceptable per design §5).
        assert not store.journal_path.exists()

    @pytest.mark.asyncio
    async def test_confirmed_and_corrected_write_through_journal(self, tmp_path: Path):
        store = _store(tmp_path)
        await store.record_access("e1", 1000.0, group_id="g", tier="confirmed")
        await store.record_access("e2", 2000.0, group_id="g2", tier="corrected")
        state = await store.get_activation("e1")
        assert state is not None
        assert state.usage_events == [(1000.0, 1.0)]
        lines = [
            json.loads(line) for line in store.journal_path.read_text(encoding="utf-8").splitlines()
        ]
        assert lines == [
            {"ts": 1000.0, "weight": 1.0, "entity_id": "e1", "group_id": "g"},
            {"ts": 2000.0, "weight": 0.5, "entity_id": "e2", "group_id": "g2"},
        ]

    @pytest.mark.asyncio
    async def test_unknown_tier_raises(self, tmp_path: Path):
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="Unknown usage tier"):
            await store.record_access("e1", 1000.0, tier="promoted")

    @pytest.mark.asyncio
    async def test_tier_weights_config_overridable(self, tmp_path: Path):
        cfg = ActivationConfig(usage_tier_weights={**DEFAULT_USAGE_TIER_WEIGHTS, "used": 0.9})
        store = _store(tmp_path, cfg=cfg)
        await store.record_access("e1", 1000.0, tier="used")
        state = await store.get_activation("e1")
        assert state is not None
        assert state.usage_events == [(1000.0, 0.9)]

    def test_default_tier_weights_pinned(self):
        # F1/F10 resolved values — a silent change here is a design change.
        assert DEFAULT_USAGE_TIER_WEIGHTS == {
            "surfaced": 0.0,
            "mentioned": 0.1,
            "used": 0.3,
            "corrected": 0.5,
            "confirmed": 1.0,
        }
        assert ActivationConfig().usage_tier_weights == DEFAULT_USAGE_TIER_WEIGHTS


class TestSnapshotV2:
    @pytest.mark.asyncio
    async def test_v2_round_trip(self, tmp_path: Path):
        now = time.time()
        store = _store(tmp_path)
        await store.record_access("e1", now - 50, group_id="g", tier="used")
        await store.record_access("e1", now - 10, group_id="g", tier="mentioned")
        await store.record_access("e1", now - 5, group_id="g")  # surfaced
        snap = tmp_path / "snap.json"
        assert store.save_to_file(snap) == 1

        payload = json.loads(snap.read_text(encoding="utf-8"))
        assert payload["version"] == 2

        fresh = _store(tmp_path)
        assert fresh.load_from_file(snap) == 1
        state = await fresh.get_activation("e1")
        assert state is not None
        assert state.usage_events == [(now - 50, 0.3), (now - 10, 0.1)]
        assert state.usage_weight_sum == pytest.approx(0.4)
        assert state.usage_last_ts == now - 10
        assert len(state.access_history) == 3

    @pytest.mark.asyncio
    async def test_v1_snapshot_loads_as_empty_usage(self, tmp_path: Path):
        snap = tmp_path / "snap.json"
        snap.write_text(
            json.dumps(
                {
                    "saved_at": time.time(),
                    "states": {
                        "e1": {
                            "node_id": "e1",
                            "access_history": [1.0, 2.0],
                            "last_accessed": 2.0,
                            "access_count": 2,
                        }
                    },
                }
            )
        )
        store = _store(tmp_path)
        assert store.load_from_file(snap) == 1
        state = await store.get_activation("e1")
        assert state is not None
        assert state.access_history == [1.0, 2.0]
        assert state.usage_events == []
        assert state.usage_weight_sum == 0.0
        assert state.usage_last_ts == 0.0

    @pytest.mark.asyncio
    async def test_v2_save_then_load_of_v1_then_v2_agree(self, tmp_path: Path):
        """v1 -> load -> save emits v2 that round-trips identically."""
        snap_v1 = tmp_path / "snap-v1.json"
        snap_v1.write_text(
            json.dumps(
                {
                    "saved_at": time.time(),
                    "states": {"e1": {"node_id": "e1", "access_history": [5.0]}},
                }
            )
        )
        store = _store(tmp_path)
        store.load_from_file(snap_v1)
        snap_v2 = tmp_path / "snap-v2.json"
        store.save_to_file(snap_v2)
        assert json.loads(snap_v2.read_text(encoding="utf-8"))["version"] == 2

        fresh = _store(tmp_path)
        fresh.load_from_file(snap_v2)
        state = await fresh.get_activation("e1")
        assert state is not None
        assert state.access_history == [5.0]
        assert state.usage_events == []

    @pytest.mark.asyncio
    async def test_stale_snapshot_ignored_but_journal_still_replays(self, tmp_path: Path):
        """The 14-day age guard is unchanged; the journal is exempt from it."""
        # A non-owner appended a confirmed event, then everything sat for 30d.
        appender = _store(tmp_path)
        await appender.record_access("e_conf", 1234.0, group_id="g", tier="confirmed")

        snap = tmp_path / "snap.json"
        snap.write_text(
            json.dumps(
                {
                    "saved_at": time.time() - 30 * 86400,
                    "states": {"e_old": {"node_id": "e_old", "access_history": [1.0]}},
                }
            )
        )
        fresh = _store(tmp_path)
        assert fresh.load_from_file(snap, max_age_days=14.0) == 0
        assert (fresh._states.get("e_old")) is None  # age guard unchanged
        state = fresh._states.get("e_conf")
        assert state is not None  # user signal does not expire in 14 days
        assert state.usage_events == [(1234.0, 1.0)]
        assert state.access_history == [1234.0]


class TestConfirmedEventJournal:
    @pytest.mark.asyncio
    async def test_second_process_event_survives_owner_save_and_truncate(self, tmp_path: Path):
        """M1.3 DoD: an event written by a second process is still present
        after the first process saves+truncates."""
        journal = tmp_path / "activation-usage-journal.jsonl"
        owner = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        second = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)

        await owner.record_access("e_owner", 1000.0, group_id="g", tier="confirmed")
        # Second process (MCP stdio / brain): appends, never truncates.
        await second.record_access("e_second", 2000.0, group_id="g", tier="corrected")

        snap = tmp_path / "snap.json"
        # Owner protocol: fold entire journal -> snapshot superset -> compact.
        assert owner.save_to_file(snap) == 2
        assert not journal.exists()  # fully folded -> truncated

        # Owner RAM absorbed the second process's event during the fold.
        folded = await owner.get_activation("e_second")
        assert folded is not None
        assert folded.usage_events == [(2000.0, 0.5)]

        fresh = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        fresh.load_from_file(snap)
        second_state = await fresh.get_activation("e_second")
        assert second_state is not None
        assert second_state.usage_events == [(2000.0, 0.5)]
        assert second_state.access_history == [2000.0]
        # The owner's own journaled event was folded idempotently: no double count.
        owner_state = await fresh.get_activation("e_owner")
        assert owner_state is not None
        assert owner_state.usage_events == [(1000.0, 1.0)]
        assert owner_state.access_history == [1000.0]

    @pytest.mark.asyncio
    async def test_toctou_line_appended_between_fold_and_truncate_survives(
        self, tmp_path: Path, monkeypatch
    ):
        """M1.3 DoD: a line appended between replay-at-save and truncate
        survives into the fresh journal segment."""
        journal = tmp_path / "activation-usage-journal.jsonl"
        owner = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        late = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        await owner.record_access("e1", 1000.0, group_id="g", tier="confirmed")

        orig_fold = MemoryActivationStore._fold_journal

        def fold_then_race(self: MemoryActivationStore, journal_path: Path) -> int:
            folded = orig_fold(self, journal_path)
            # A concurrent process appends while the owner writes the snapshot.
            late._journal_append("e_late", 3000.0, 1.0, "g")
            return folded

        monkeypatch.setattr(MemoryActivationStore, "_fold_journal", fold_then_race)
        snap = tmp_path / "snap.json"
        owner.save_to_file(snap)
        monkeypatch.undo()

        # The late line was re-scanned into a fresh journal segment...
        assert journal.exists()
        lines = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
        assert lines == [{"ts": 3000.0, "weight": 1.0, "entity_id": "e_late", "group_id": "g"}]
        # ...and into the owner's RAM.
        late_state = await owner.get_activation("e_late")
        assert late_state is not None
        assert late_state.usage_events == [(3000.0, 1.0)]

        # A fresh process sees it too: stale-free snapshot + journal replay.
        fresh = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        fresh.load_from_file(snap)
        state = await fresh.get_activation("e_late")
        assert state is not None
        assert state.usage_events == [(3000.0, 1.0)]

    @pytest.mark.asyncio
    async def test_non_owner_load_replays_without_truncating(self, tmp_path: Path):
        journal = tmp_path / "activation-usage-journal.jsonl"
        appender = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        await appender.record_access("e1", 1000.0, group_id="g", tier="confirmed")
        before = journal.read_text(encoding="utf-8")

        non_owner = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        non_owner.load_from_file(tmp_path / "no-snapshot.json")
        state = await non_owner.get_activation("e1")
        assert state is not None
        assert state.usage_events == [(1000.0, 1.0)]
        # Load is append-only for non-owners: the journal is untouched.
        assert journal.read_text(encoding="utf-8") == before

    @pytest.mark.asyncio
    async def test_malformed_journal_line_skipped(self, tmp_path: Path):
        journal = tmp_path / "activation-usage-journal.jsonl"
        appender = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        await appender.record_access("e1", 1000.0, group_id="g", tier="confirmed")
        with open(journal, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        await appender.record_access("e2", 2000.0, group_id="g", tier="confirmed")

        fresh = MemoryActivationStore(cfg=ActivationConfig(), journal_path=journal)
        fresh.load_from_file(tmp_path / "no-snapshot.json")
        assert (await fresh.get_activation("e1")) is not None
        assert (await fresh.get_activation("e2")) is not None


class TestRecallPathInertness:
    """G5-style store-spy: a plain recall path records ZERO usage_events."""

    @pytest.mark.asyncio
    async def test_plain_recall_records_only_hygiene_history(self, tmp_path: Path):
        from engram.extraction.extractor import ExtractionResult

        cfg = ActivationConfig()
        # Under shipped defaults recall surfaces ZERO entity results
        # (passage_first_entity_budget=0, the M4.1 arm-B0 finding), so the P1
        # access recorder would never fire and the spy would prove nothing.
        # Open one entity slot so the surfaced-tier write path genuinely runs.
        cfg.passage_first_entity_budget = 1
        graph_store = SQLiteGraphStore(str(tmp_path / "spy.db"))
        await graph_store.initialize()
        search_index = FTS5SearchIndex(graph_store._db_path)
        await search_index.initialize(db=graph_store._db)
        activation_store = _store(tmp_path, cfg=cfg)
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            MockExtractor(
                ExtractionResult(
                    entities=[
                        {
                            "name": "Python",
                            "entity_type": "Technology",
                            "summary": "Programming language",
                        },
                    ],
                    relationships=[],
                )
            ),
            cfg=cfg,
        )
        states: dict[str, ActivationState] = activation_store._states
        try:
            await manager.ingest_episode(
                "FastAPI is built with Python.", group_id="default", source="test"
            )
            # Ingestion mentions legitimately record a mentioned-tier event
            # (M1.6/F10, w=0.1) — snapshot the usage store before recall so the
            # spy isolates the RECALL path.
            usage_before = {eid: list(state.usage_events) for eid, state in states.items()}
            history_before = sum(len(state.access_history) for state in states.values())
            await manager.recall("Python", group_id="default", limit=5)
        finally:
            await graph_store.close()

        assert states, "expected the ingest path to record accesses"
        assert history_before >= 1
        # The inertness proof: recall added ZERO ranking-eligible usage events.
        for eid, state in states.items():
            assert state.usage_events == usage_before.get(eid, []), (
                f"recall wrote a ranking-eligible usage event for {eid}"
            )
        # And no journal write: recall never emits confirmed/corrected tiers.
        assert not activation_store.journal_path.exists()

    @pytest.mark.asyncio
    async def test_recall_entity_access_recorder_surfaced_is_hygiene_only(self, tmp_path: Path):
        """Drive the exact recorder recall's P1 site uses (surfaced tier):
        access_history grows, usage_events does not."""
        from engram.models.entity import Entity
        from engram.retrieval.feedback import RecallEntityAccessRecorder

        cfg = ActivationConfig()
        activation_store = _store(tmp_path, cfg=cfg)
        recorder = RecallEntityAccessRecorder(
            cfg=cfg,
            activation_store=activation_store,
            event_bus=None,
            labile_tracker=None,
        )
        entity = Entity(
            id="ent_spy",
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id="default",
        )
        await recorder.record_entity_access(
            entity,
            group_id="default",
            query="Python",
            source="recall",
            timestamp=1000.0,
            tier="surfaced",
        )
        state = await activation_store.get_activation("ent_spy")
        assert state is not None
        assert state.access_history == [1000.0]  # hygiene recorded
        assert state.usage_events == []  # ranking store untouched
        assert not activation_store.journal_path.exists()
