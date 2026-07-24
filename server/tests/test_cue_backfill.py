"""P11: the cue backfill sweep (`engram cues backfill`).

Episodes captured before the cue layer shipped have no cue row and nothing in
the system ever writes one -- cue creation is capture-time only. These tests
cover the drain that closes that debt, against a REAL lite/SQLite brain (not a
fake store) because the properties that matter are storage properties:

- dry run reports N and writes 0
- apply writes N
- apply AGAIN writes 0 (idempotency; the guard is the pre-write cue probe)
- a backfilled cue is byte-identical to what the capture path would have
  produced for the same content (same derivation function, no drift)
- capture's skip rules are honoured (empty content, discourse_class == "system")
- an already-PROJECTED episode does not get a cue claiming SCHEDULED, which
  would re-queue it for extraction

NEUTER CHECK: delete the `already_cued` early-continue in
`backfill_missing_episode_cues` and `test_apply_twice_is_idempotent` goes red.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.ingestion.cue_backfill import (
    backfill_missing_episode_cues,
    build_backfill_cue,
)
from engram.models.episode import Episode, EpisodeProjectionState
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.dates import utc_now

GROUP = "default"

# Derivation-only fields: created_at/updated_at are wall-clock at write time and
# differ between a capture-time cue and a backfilled one by construction.
_DERIVED_FIELDS = (
    "episode_id",
    "group_id",
    "cue_version",
    "discourse_class",
    "projection_state",
    "cue_score",
    "salience_score",
    "projection_priority",
    "route_reason",
    "cue_text",
    "entity_mentions",
    "temporal_markers",
    "quote_spans",
    "contradiction_keys",
    "first_spans",
    "policy_score",
)


def _derived(cue) -> dict:
    dumped = cue.model_dump()
    return {k: dumped[k] for k in _DERIVED_FIELDS}


def _cfg() -> ActivationConfig:
    return ActivationConfig(cue_layer_enabled=True)


def _episode(idx: int, content: str, *, age_days: int = 60, **kwargs) -> Episode:
    return Episode(
        id=f"ep-{idx:03d}",
        content=content,
        group_id=GROUP,
        source="conversation",
        created_at=utc_now() - timedelta(days=age_days),
        **kwargs,
    )


# Pre-cue-layer content: entity-dense, temporal, quoted -- the cue derivation
# has something to bite on for each.
_PRE_CUE_CONTENTS = [
    'Konner decided on 2026-05-14 that Engram should stay local-first. He said "no external keys".',
    "The HelixDB migration moved server/engram/storage/helix/schema.hx to native PyO3 last week.",
    "Sarah no longer works at Acme; she moved to Berlin in 2026-03 and now runs the SDK team.",
    "Benchmark run finished: recall improved 12% after the FTS5 reindex on engram/config.py.",
    "Actually the LongMemEval numbers were wrong -- the group_id was shared across sessions.",
]


@pytest_asyncio.fixture
async def lite_brain(tmp_path):
    store = SQLiteGraphStore(str(tmp_path / "cue_backfill.db"))
    await store.initialize()
    yield store
    await store.close()


async def _seed_pre_cue_episodes(store: SQLiteGraphStore) -> list[Episode]:
    episodes = [_episode(i, content) for i, content in enumerate(_PRE_CUE_CONTENTS)]
    for ep in episodes:
        await store.create_episode(ep)
    # Precondition: this is a pre-cue-layer brain -- zero cues exist.
    for ep in episodes:
        assert await store.get_episode_cue(ep.id, GROUP) is None
    return episodes


@pytest.mark.asyncio
async def test_dry_run_reports_and_writes_nothing(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)

    result = await backfill_missing_episode_cues(lite_brain, _cfg(), GROUP)

    assert result.dry_run is True
    assert result.would_write == len(episodes)
    assert result.written == 0
    assert result.cursor_next is None  # a dry run must not advance the cursor
    for ep in episodes:
        assert await lite_brain.get_episode_cue(ep.id, GROUP) is None


@pytest.mark.asyncio
async def test_apply_writes_every_missing_cue(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)

    result = await backfill_missing_episode_cues(lite_brain, _cfg(), GROUP, apply=True)

    assert result.written == len(episodes)
    assert result.failed == 0
    assert result.complete is True
    assert result.cursor_next is not None
    for ep in episodes:
        cue = await lite_brain.get_episode_cue(ep.id, GROUP)
        assert cue is not None
        assert cue.cue_text.strip()


@pytest.mark.asyncio
async def test_apply_twice_is_idempotent(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)
    cfg = _cfg()

    first = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)
    assert first.written == len(episodes)

    # Second run WITHOUT the cursor: the only thing that can stop a double-write
    # is the pre-write cue probe. This is the neuter target.
    second = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)

    assert second.written == 0, "second apply re-wrote cues — idempotency guard is dead"
    assert second.already_cued == len(episodes)


@pytest.mark.asyncio
async def test_backfilled_cue_matches_capture_path_byte_for_byte(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)
    cfg = _cfg()

    await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)

    for ep in episodes:
        stored = await lite_brain.get_episode_cue(ep.id, GROUP)
        # What CaptureService._store_episode_cue would have derived and upserted.
        expected = build_episode_cue(ep, cfg)
        assert expected is not None
        assert _derived(stored) == _derived(expected)


@pytest.mark.asyncio
async def test_skips_exactly_what_capture_skips(lite_brain):
    cfg = _cfg()
    empty = _episode(90, "   \n  ")
    system = _episode(
        91,
        "The extraction pipeline logged activation score and access_count "
        "for every episode worker run.",
    )
    real = _episode(92, _PRE_CUE_CONTENTS[0])
    for ep in (empty, system, real):
        await lite_brain.create_episode(ep)

    # Precondition: capture itself declines these two.
    assert build_episode_cue(empty, cfg) is None
    assert build_episode_cue(system, cfg) is None
    assert build_episode_cue(real, cfg) is not None

    result = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)

    assert result.written == 1
    assert result.skipped_empty_content == 1
    assert result.skipped_by_policy == 1
    assert await lite_brain.get_episode_cue(empty.id, GROUP) is None
    assert await lite_brain.get_episode_cue(system.id, GROUP) is None
    assert await lite_brain.get_episode_cue(real.id, GROUP) is not None


@pytest.mark.asyncio
async def test_projected_episode_does_not_get_a_scheduled_cue(lite_brain):
    """A cue claiming SCHEDULED on an already-extracted episode is a lie.

    The derived route for identity/contradiction content is SCHEDULED. If the
    backfill wrote that onto the 8k already-PROJECTED historical episodes, the
    cue row would contradict the episode and advertise extraction work that has
    already been done.
    """
    cfg = _cfg()
    ep = _episode(
        70,
        "My name is Konner and I no longer work at Acme.",
        projection_state=EpisodeProjectionState.PROJECTED,
    )
    await lite_brain.create_episode(ep)

    # Precondition: the capture-time derivation alone WOULD say SCHEDULED.
    naive = build_episode_cue(ep, cfg)
    assert naive.projection_state == EpisodeProjectionState.SCHEDULED

    reconciled, inherited = build_backfill_cue(ep, cfg)
    assert inherited is True
    assert reconciled.projection_state == EpisodeProjectionState.PROJECTED.value

    result = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)
    assert result.state_inherited == 1
    stored = await lite_brain.get_episode_cue(ep.id, GROUP)
    assert stored.projection_state == EpisodeProjectionState.PROJECTED


@pytest.mark.asyncio
async def test_limit_bounds_the_window_and_cursor_resumes(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)
    cfg = _cfg()

    first = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, limit=2, apply=True)
    assert first.written == 2
    assert first.complete is False
    assert first.cursor_next is not None

    second = await backfill_missing_episode_cues(
        lite_brain,
        cfg,
        GROUP,
        limit=2,
        apply=True,
        cursor=first.cursor_next,
    )
    assert second.written == 2
    assert second.probed == 2, "cursor did not skip the already-backfilled prefix"

    third = await backfill_missing_episode_cues(
        lite_brain,
        cfg,
        GROUP,
        limit=10,
        apply=True,
        cursor=second.cursor_next,
    )
    assert third.written == len(episodes) - 4
    assert third.complete is True
    for ep in episodes:
        assert await lite_brain.get_episode_cue(ep.id, GROUP) is not None


@pytest.mark.asyncio
async def test_write_failure_does_not_advance_the_cursor_past_it(lite_brain):
    episodes = await _seed_pre_cue_episodes(lite_brain)
    cfg = _cfg()
    real_upsert = lite_brain.upsert_episode_cue
    failed_id = episodes[1].id

    async def flaky(cue):
        if cue.episode_id == failed_id:
            raise RuntimeError("simulated cue write failure")
        await real_upsert(cue)

    lite_brain.upsert_episode_cue = flaky
    try:
        result = await backfill_missing_episode_cues(lite_brain, cfg, GROUP, apply=True)
    finally:
        lite_brain.upsert_episode_cue = real_upsert

    assert result.failed == 1
    assert result.complete is False
    # Cursor stops before the failure, so the next window retries it.
    assert result.cursor_next is not None
    assert result.cursor_next[1] == episodes[0].id

    retry = await backfill_missing_episode_cues(
        lite_brain,
        cfg,
        GROUP,
        apply=True,
        cursor=result.cursor_next,
    )
    assert failed_id in retry.written_ids
