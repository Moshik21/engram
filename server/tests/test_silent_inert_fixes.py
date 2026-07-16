"""M3 silent-inert purge: loop store consistency, worker floor, executor wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engram.loop_adjustment import (
    LoopAdjustment,
    clamp_loop_adjustment,
    clear_active_adjustment,
    save_active_adjustment,
    save_active_adjustment_async,
    stamp_applied,
    status_payload,
)


def _adj(reason: str = "test", group_id: str = "default") -> LoopAdjustment:
    result = clamp_loop_adjustment(
        LoopAdjustment.from_mapping(
            {"reason": reason, "group_id": group_id, "ttl_hours": 12, "max_risk": "low"}
        )
    )
    assert not result.rejected
    return stamp_applied(result.adjustment)


@pytest.fixture()
def loop_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    adj = tmp_path / "loop-adjustment.json"
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(adj))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(audit))
    return {"adj": adj, "audit": audit}


class TestFileFirstStatus:
    def test_status_reports_file_source(self, loop_paths):
        save_active_adjustment(_adj(), path=loop_paths["adj"], audit_path=loop_paths["audit"])
        payload = status_payload("default", path=loop_paths["adj"])
        assert payload["active"] is True
        assert payload["store"] == "file"

    def test_cli_clear_after_api_apply_is_consistent(self, loop_paths):
        """CLI clear (file-only) must leave status inactive — the runtime
        honors the file, so status must not resurrect the graph copy."""

        class _Store:
            def __init__(self):
                self.saved = None
                self.cleared = []

            async def save_loop_adjustment(self, group_id, payload):
                self.saved = payload

            async def load_loop_adjustment(self, group_id):
                return self.saved

            async def clear_loop_adjustment(self, group_id):
                self.cleared.append(group_id)
                self.saved = None

        import asyncio

        store = _Store()
        adj = _adj()
        asyncio.run(
            save_active_adjustment_async(
                adj,
                path=loop_paths["adj"],
                audit_path=loop_paths["audit"],
                graph_store=store,
            )
        )
        # CLI-style clear: file only.
        cleared = clear_active_adjustment(
            "default",
            path=loop_paths["adj"],
            audit_path=loop_paths["audit"],
            cleared_by="test",
        )
        assert cleared is True
        payload = status_payload("default", path=loop_paths["adj"])
        assert payload["active"] is False


class TestSidecarMirrorOnlyWrite:
    @pytest.mark.asyncio
    async def test_file_write_false_skips_file_and_audit(self, loop_paths):
        class _Store:
            def __init__(self):
                self.saved = None

            async def save_loop_adjustment(self, group_id, payload):
                self.saved = payload

        store = _Store()
        adj = _adj()
        await save_active_adjustment_async(
            adj,
            path=loop_paths["adj"],
            audit_path=loop_paths["audit"],
            graph_store=store,
            file_write=False,
        )
        assert not loop_paths["adj"].exists()
        assert store.saved is not None
        events = [
            json.loads(line)["event"]
            for line in loop_paths["audit"].read_text().splitlines()
            if line.strip()
        ]
        assert "apply" not in events  # no duplicate file-apply audit
        assert "apply_graph" in events


class TestClearGroupMismatch:
    def test_mismatched_file_group_still_clears_graph_copy(self, loop_paths):
        save_active_adjustment(
            _adj(group_id="other"),
            path=loop_paths["adj"],
            audit_path=loop_paths["audit"],
        )

        class _Store:
            def __init__(self):
                self.cleared = []

            def clear_loop_adjustment_sync(self, group_id):
                self.cleared.append(group_id)

        store = _Store()
        cleared = clear_active_adjustment(
            "default",
            path=loop_paths["adj"],
            audit_path=loop_paths["audit"],
            cleared_by="test",
            graph_store=store,
        )
        # File belongs to 'other' → untouched; but 'default' graph copy cleared.
        assert loop_paths["adj"].exists()
        assert store.cleared == ["default"]
        assert cleared is True  # the file existed


class TestExpireAuditPath:
    def test_expire_event_honors_custom_audit_path(self, loop_paths, tmp_path: Path):
        from datetime import datetime, timedelta, timezone

        from engram.loop_adjustment import load_active_adjustment

        adj = _adj()
        adj.expires_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        loop_paths["adj"].write_text(json.dumps(adj.to_dict()))
        custom_audit = tmp_path / "custom-audit.jsonl"
        got = load_active_adjustment(
            "default",
            path=loop_paths["adj"],
            audit_path=custom_audit,
        )
        assert got is None
        events = [
            json.loads(line)["event"]
            for line in custom_audit.read_text().splitlines()
            if line.strip()
        ]
        assert "expire" in events


class TestWorkerFloorZero:
    def test_zero_floor_is_honored(self):
        from engram.config import ActivationConfig
        from engram.ingestion.worker_routing import EpisodeWorkerProjectionRouter

        cfg = ActivationConfig(worker_auto_capture_extract_score_floor=0.0)
        router = EpisodeWorkerProjectionRouter.__new__(EpisodeWorkerProjectionRouter)
        router._cfg = cfg
        # Bypass the steward overlay path for a pure unit check.
        router.effective_cfg = lambda group_id="default": cfg
        assert router.auto_capture_extract_score_floor() == 0.0


class TestExecutorClarificationWiring:
    @pytest.mark.asyncio
    async def test_injected_callable_is_used(self):
        from engram.config import ActivationConfig
        from engram.ingestion.projection_execution import EvidenceProjectionExecutor

        called = []

        async def fake_intents(requests):
            called.append(requests)

        executor = EvidenceProjectionExecutor(
            graph_store=AsyncMock(),
            cfg=ActivationConfig(),
            build_evidence_bundle=AsyncMock(),
            build_adjudication_requests=AsyncMock(),
            serialize_candidate_records=lambda *a, **k: [],
            serialize_evidence_records=lambda *a, **k: [],
            materialize_evidence=AsyncMock(),
            apply_committed_ids=AsyncMock(),
            update_episode_status=AsyncMock(),
            create_clarification_intents=fake_intents,
        )
        assert executor._create_clarification_intents is fake_intents
