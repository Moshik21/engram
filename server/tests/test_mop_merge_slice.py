"""M3.5: bounded exact-merge slice in the hygiene mop.

Consumer installs never run the merge phase; the slice gives the mop a
deterministic, budgeted dedup pass (flag hygiene_mop_merge_enabled, default off).
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.hygiene_debt import HygieneDebtSnapshot
from engram.consolidation.phases.merge import run_exact_merge_slice
from engram.models.consolidation import PhaseResult
from engram.models.entity import Entity
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


@pytest.fixture
def gid():
    return f"test_{uuid.uuid4().hex[:8]}"


def _entity(name, entity_type="Concept", group_id="test", identity_core=False):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
        identity_core=identity_core,
    )


class TestExactMergeSlice:
    @pytest.mark.asyncio
    async def test_exact_dup_pair_merges(self, store, activation, search, gid):
        a = _entity("John Smith", entity_type="Person", group_id=gid)
        b = _entity("john smith", entity_type="Person", group_id=gid)
        await store.create_entity(a)
        await store.create_entity(b)

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=25,
            activation_store=activation,
            search_index=search,
        )

        assert result["dry_run"] is False
        assert result["merged"] == 1
        assert result["errors"] == 0
        assert len(result["pairs"]) == 1
        remaining = await store.find_entities(group_id=gid, limit=10)
        assert len(remaining) == 1

    @pytest.mark.asyncio
    async def test_budget_respected(self, store, activation, search, gid):
        for name in ("Alpha One", "Beta Two", "Gamma Three"):
            await store.create_entity(_entity(name, group_id=gid))
            await store.create_entity(_entity(name.lower(), group_id=gid))

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=1,
            activation_store=activation,
            search_index=search,
        )

        assert result["total"] == 3
        assert result["merged"] == 1
        remaining = await store.find_entities(group_id=gid, limit=10)
        assert len(remaining) == 5

    @pytest.mark.asyncio
    async def test_artifact_pair_not_merged(self, store, activation, search, gid):
        a = _entity("README.md", entity_type="Artifact", group_id=gid)
        b = _entity("README.md", entity_type="Artifact", group_id=gid)
        await store.create_entity(a)
        await store.create_entity(b)

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=25,
            activation_store=activation,
            search_index=search,
        )

        assert result["total"] == 0
        assert result["merged"] == 0
        assert await store.get_entity(a.id, gid) is not None
        assert await store.get_entity(b.id, gid) is not None

    @pytest.mark.asyncio
    async def test_identity_core_protected_pair_not_merged(self, store, activation, search, gid):
        # decision_statement scrap names block identity_core merges outright.
        name = "decision_statement: use Postgres"
        a = _entity(name, entity_type="Decision", group_id=gid, identity_core=True)
        b = _entity(name, entity_type="Decision", group_id=gid)
        await store.create_entity(a)
        await store.create_entity(b)

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=25,
            activation_store=activation,
            search_index=search,
        )

        assert result["total"] == 1
        assert result["merged"] == 0
        assert await store.get_entity(a.id, gid) is not None
        assert await store.get_entity(b.id, gid) is not None

    @pytest.mark.asyncio
    async def test_containment_below_slice_floor_not_merged(self, store, activation, search, gid):
        """Substring containment caps at 0.9 — below the 0.95 slice floor."""
        await store.create_entity(_entity("Melanie", entity_type="Person", group_id=gid))
        await store.create_entity(_entity("Melanie Smith", entity_type="Person", group_id=gid))

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=25,
            activation_store=activation,
            search_index=search,
        )

        assert result["merged"] == 0
        remaining = await store.find_entities(group_id=gid, limit=10)
        assert len(remaining) == 2

    @pytest.mark.asyncio
    async def test_dry_run_honored(self, store, activation, search, gid):
        a = _entity("Kubernetes", group_id=gid)
        b = _entity("kubernetes", group_id=gid)
        await store.create_entity(a)
        await store.create_entity(b)

        result = await run_exact_merge_slice(
            store,
            gid,
            budget=25,
            activation_store=activation,
            search_index=search,
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["merged"] == 1
        assert await store.get_entity(a.id, gid) is not None
        assert await store.get_entity(b.id, gid) is not None


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _mop_patches():
    debt = HygieneDebtSnapshot(deferred_evidence=600)
    return (
        patch(
            "engram.consolidation.hygiene_debt.collect_hygiene_debt_from_store",
            new=AsyncMock(return_value=debt),
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
    )


class TestMopWiring:
    @pytest.mark.asyncio
    async def test_flag_off_slice_skipped(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop, mark_cue_scan_done

        with ExitStack() as stack:
            for p in _mop_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "engram.consolidation.phases.merge.run_exact_merge_slice",
                    new=AsyncMock(side_effect=AssertionError("flag off — must not run")),
                )
            )
            prune_cls = stack.enter_context(patch("engram.consolidation.phases.prune.PrunePhase"))
            prune_cls.return_value.execute = AsyncMock(
                return_value=(PhaseResult(phase="prune", status="completed"), [])
            )
            mark_cue_scan_done()
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=ActivationConfig(),
                group_id="g",
                budget=100,
            )
        assert report["mop"]["merge_slice"]["skipped"] is True

    @pytest.mark.asyncio
    async def test_flag_on_runs_budgeted_slice(self, engram_home: Path):
        from engram.hygiene_ops import execute_hygiene_mop, mark_cue_scan_done

        slice_result = {"dry_run": True, "total": 2, "merged": 2, "errors": 0, "pairs": []}
        slice_mock = AsyncMock(return_value=slice_result)
        cfg = ActivationConfig(hygiene_mop_merge_enabled=True, hygiene_mop_merge_budget=7)
        with ExitStack() as stack:
            for p in _mop_patches():
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "engram.consolidation.phases.merge.run_exact_merge_slice",
                    new=slice_mock,
                )
            )
            prune_cls = stack.enter_context(patch("engram.consolidation.phases.prune.PrunePhase"))
            prune_cls.return_value.execute = AsyncMock(
                return_value=(PhaseResult(phase="prune", status="completed"), [])
            )
            mark_cue_scan_done()
            report = await execute_hygiene_mop(
                graph_store=object(),
                activation_store=object(),
                search_index=object(),
                activation_cfg=cfg,
                group_id="g",
                budget=100,
                dry_run=True,
            )
        assert report["mop"]["merge_slice"] == slice_result
        assert slice_mock.await_args.kwargs["budget"] == 7
        assert slice_mock.await_args.kwargs["dry_run"] is True
