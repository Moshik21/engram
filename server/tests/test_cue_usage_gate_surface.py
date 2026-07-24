"""P10: the cue-usefulness gate must read a counter that can actually move.

`cue_used_count` sums the LEGACY int `used_count`, which only increments for
`interaction_type == "used"`. `public_surface_policy.chat_tool_recall_policy`
emits `"used"` ONLY when `recall_usage_feedback_enabled` is False — live it is
True, so the chat path sends `"selected"` and the legacy counter is structurally
incapable of moving. The M5.1 signal that DOES move lands in `usage_used_count`.

Every test here builds a store/metrics payload where the legacy counter is ZERO
and only the M5.1 usage signal is present. Before the fix every gate reader
returns 0; after the fix they return the live signal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from engram.config import HelixDBConfig
from engram.evaluation.brain_loop_report import build_brain_loop_report
from engram.lifecycle_summary import build_lifecycle_summary
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.storage.helix.graph import HelixGraphStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.dates import utc_now

USAGE_TS = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)


# ─── lite / SQLite backend ─────────────────────────────────────────


async def _lite_store(tmp_path) -> SQLiteGraphStore:
    store = SQLiteGraphStore(str(tmp_path / "cue_usage_gate.db"))
    await store.initialize()
    for episode_id in ("ep_usage", "ep_legacy"):
        await store.create_episode(
            Episode(
                id=episode_id,
                content=f"content for {episode_id}",
                source="test",
                status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                group_id="default",
                created_at=utc_now(),
            )
        )
    return store


@pytest.mark.asyncio
async def test_lite_stats_surface_m51_usage_when_legacy_counter_is_dead(tmp_path):
    """Lite backend: a cue with ONLY usage_used_count must show up in stats."""
    store = await _lite_store(tmp_path)
    try:
        await store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_usage",
                cue_text="the cue the agent actually cited",
                surfaced_count=4,
                selected_count=2,
                used_count=0,  # dead field: chat path never emits "used"
                usage_used_count=0.3,  # live M5.1 citation-scan signal
                usage_last_used_at=USAGE_TS,
            )
        )
        stats = await store.get_stats(group_id="default")
        cue_metrics = stats["cue_metrics"]

        assert cue_metrics["cue_used_count"] == 0
        assert cue_metrics["cue_usage_used_count"] == pytest.approx(0.3)
        assert cue_metrics["cue_usage_used_episode_count"] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_lite_stats_keep_legacy_only_cues_counted(tmp_path):
    """Backward compat: pre-M5.1 cues carry only used_count and still count."""
    store = await _lite_store(tmp_path)
    try:
        await store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_legacy",
                cue_text="a cue stored before M5.1",
                surfaced_count=2,
                used_count=3,
            )
        )
        cue_metrics = (await store.get_stats(group_id="default"))["cue_metrics"]

        assert cue_metrics["cue_used_count"] == 3
        assert cue_metrics["cue_usage_used_count"] == 0.0
        assert cue_metrics["cue_usage_used_episode_count"] == 0
    finally:
        await store.close()


# ─── helix backend (native + HTTP share HelixGraphStore) ───────────


@pytest.mark.asyncio
async def test_helix_stats_surface_m51_usage_from_the_span_trailer(monkeypatch):
    """Helix backend: usage rides supporting_spans_json, not a native column."""
    store = HelixGraphStore(HelixDBConfig())

    usage_trailer = (
        '[{"_engram_cue_usage": {"used_count": 0.6, '
        '"last_used_at": "2026-07-24T12:00:00+00:00"}}]'
    )

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_entities_all":
            return []
        if endpoint == "find_episodes_all":
            return [
                {"episode_id": "ep_usage", "group_id": "default", "projection_state": "cued"},
                {"episode_id": "ep_legacy", "group_id": "default", "projection_state": "cued"},
            ]
        if endpoint == "find_cues_all":
            return [
                {
                    "episode_id": "ep_usage",
                    "group_id": "default",
                    "cue_text": "cited cue",
                    "projection_state": "cued",
                    "surfaced_count": 4,
                    "selected_count": 2,
                    "used_count": 0,
                    "supporting_spans_json": usage_trailer,
                },
                {
                    "episode_id": "ep_legacy",
                    "group_id": "default",
                    "cue_text": "legacy cue",
                    "projection_state": "cued",
                    "surfaced_count": 2,
                    "used_count": 3,
                    "supporting_spans_json": "[]",
                },
            ]
        return []

    monkeypatch.setattr(store, "_query", fake_query)

    cue_metrics = (await store.get_stats())["cue_metrics"]

    # Legacy counter only reflects the pre-M5.1 cue.
    assert cue_metrics["cue_used_count"] == 3
    assert cue_metrics["cue_usage_used_count"] == pytest.approx(0.6)
    assert cue_metrics["cue_usage_used_episode_count"] == 1


# ─── brain_loop_report: the gate metric ────────────────────────────


def _m51_only_stats() -> dict[str, Any]:
    """Stats shaped like a live install: legacy used_count pinned at zero."""
    return {
        "episodes": 3,
        "entities": 2,
        "relationships": 1,
        "cue_metrics": {
            "cue_count": 3,
            "episodes_without_cues": 0,
            "cue_coverage": 1.0,
            "cue_hit_count": 6,
            "cue_surfaced_count": 8,
            "cue_selected_count": 4,
            "cue_used_count": 0,
            "cue_usage_used_count": 0.6,
            "cue_usage_used_episode_count": 2,
            "cue_near_miss_count": 1,
        },
        "projection_metrics": {"state_counts": {"projected": 1}},
    }


def test_brain_loop_gate_metric_moves_on_the_m51_signal() -> None:
    """used_rate (the cue_usefulness gate metric) must read the live counter."""
    report = build_brain_loop_report(
        _m51_only_stats(),
        generated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    cue = report["cue"]

    assert cue["used_count"] == 0  # legacy field still reported, unchanged
    assert cue["usage_used_count"] == pytest.approx(0.6)
    assert cue["usage_used_episode_count"] == 2
    assert cue["effective_used_count"] == 2
    # 2 used observations / 8 surfaced. Reads 0.0 while the gate is wired to the
    # dead legacy counter.
    assert cue["used_rate"] == pytest.approx(0.25)

    signal = report["evaluation_signals"]["cue_usefulness"]
    assert signal["metric"] == pytest.approx(0.25)


def test_brain_loop_gate_metric_still_counts_legacy_only_installs() -> None:
    """Backward compat: an install with only the legacy counter is unaffected."""
    stats = _m51_only_stats()
    stats["cue_metrics"]["cue_used_count"] = 2
    stats["cue_metrics"]["cue_usage_used_count"] = 0.0
    stats["cue_metrics"]["cue_usage_used_episode_count"] = 0

    report = build_brain_loop_report(
        stats,
        generated_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    assert report["cue"]["effective_used_count"] == 2
    assert report["cue"]["used_rate"] == pytest.approx(0.25)


# ─── lifecycle summary: the operator-visible instrument ────────────


class _StubManager:
    def __init__(self, stats: dict[str, Any]) -> None:
        self._stats = stats

    async def get_graph_state(self, **_kwargs: Any) -> dict[str, Any]:
        return {"stats": self._stats, "top_activated": []}


class _StubEngine:
    is_running = False

    async def get_recent_cycles(self, _group_id: str, limit: int = 10) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_lifecycle_cue_used_count_moves_on_the_m51_signal() -> None:
    summary = await build_lifecycle_summary(
        group_id="default",
        manager=_StubManager(_m51_only_stats()),
        consolidation_engine=_StubEngine(),
    )
    cue = summary["cue"]

    assert cue["legacyUsedCount"] == 0
    assert cue["usageUsedCount"] == pytest.approx(0.6)
    assert cue["usageUsedEpisodeCount"] == 2
    # Reads 0 while usedCount is wired to the dead legacy counter.
    assert cue["usedCount"] == 2
