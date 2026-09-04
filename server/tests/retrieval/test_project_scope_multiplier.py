"""Episode candidates from another project are demoted, graduated by config.

Measured 2026-09-03: for 'why was Thompson sampling removed' the vector lane's
15 candidates were all other projects' captures and the two answer episodes
sat at fused positions 8 and 11. Episodes carry no project field; the hook's
'[role|project]' header is the marker.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.retrieval.pipeline import _episode_project_multipliers, _project_of_content

pytestmark = pytest.mark.asyncio


def test_project_is_read_from_the_capture_header() -> None:
    assert _project_of_content("[assistant|Engram] Ranking. The instrument") == "Engram"
    assert _project_of_content("[user|shielded-bid] <task-notification>") == "shielded-bid"
    assert _project_of_content("[project-bootstrap|server|docs/x.md] # Engram") == "server"
    assert _project_of_content("RESULT 2026-07-17 (batch 2): Refactors") is None
    assert _project_of_content("") is None


class _Store:
    def __init__(self, contents: dict[str, str]) -> None:
        self.contents = contents
        self.reads = 0

    async def get_episode_by_id(self, episode_id: str, group_id: str):
        self.reads += 1
        text = self.contents.get(episode_id)
        return None if text is None else SimpleNamespace(id=episode_id, content=text)


async def test_other_project_is_demoted_same_and_unknown_are_not() -> None:
    store = _Store({
        "ep_mine": "[assistant|Engram] Thompson sampling removed: noise",
        "ep_other": "[assistant|MachineShopScheduler] Done. The machine is clean.",
        "ep_unknown": "RESULT 2026-07-17: refactors pushed",
    })
    mult = await _episode_project_multipliers(
        store, "default", ["ep_mine", "ep_other", "ep_unknown", "ep_missing"],
        "/Users/x/Engram",
        ActivationConfig(
            recall_other_project_multiplier=0.6,
            recall_short_episode_floor_chars=0,
            recall_machinery_episode_multiplier=1.0,
        ),
        {},
    )
    assert mult == {"ep_other": 0.6}


async def test_multiplier_one_or_no_project_path_reads_nothing() -> None:
    store = _Store({"ep_other": "[assistant|Other] x"})
    off = ActivationConfig(
        recall_other_project_multiplier=1.0,
        recall_short_episode_floor_chars=0,
        recall_machinery_episode_multiplier=1.0,
    )
    on = ActivationConfig(
        recall_other_project_multiplier=0.6,
        recall_short_episode_floor_chars=0,
        recall_machinery_episode_multiplier=1.0,
    )
    assert (
        await _episode_project_multipliers(store, "default", ["ep_other"], "/x/Engram", off, {})
        == {}
    )
    assert await _episode_project_multipliers(store, "default", ["ep_other"], None, on, {}) == {}
    assert store.reads == 0


async def test_short_episodes_are_demoted_in_proportion_to_length() -> None:
    """A twenty-character chat prompt cannot answer anything (2026-09-03)."""
    store = _Store({
        "ep_tiny": "[user|Engram] is it running again",
        "ep_mid": "[user|Engram] " + "x" * 150,
        "ep_long": "[assistant|Engram] " + "y" * 400,
    })
    cfg = ActivationConfig(
        recall_other_project_multiplier=1.0, recall_short_episode_floor_chars=300
    )
    mult = await _episode_project_multipliers(
        store, "default", ["ep_tiny", "ep_mid", "ep_long"], "/x/Engram", cfg, {}
    )
    assert mult["ep_tiny"] == pytest.approx(0.3)
    assert mult["ep_mid"] == pytest.approx(0.5)
    assert "ep_long" not in mult


async def test_machinery_captures_are_demoted() -> None:
    """<system-reminder>/<task-notification> wrappers are BM25-indexed but say nothing."""
    store = _Store({
        "ep_sys": "[user|Engram] <system-reminder> Background task done </system-reminder>",
        "ep_note": (
            "[user|Engram] <task-notification>\n<task-id>abc</task-id>\n</task-notification>"
        ),
        "ep_real": (
            "[assistant|Engram] "
            + "The consolidation phases mature and semanticize were deleted. " * 6
        ),
    })
    cfg = ActivationConfig(
        recall_other_project_multiplier=1.0,
        recall_short_episode_floor_chars=0,
        recall_machinery_episode_multiplier=0.3,
    )
    mult = await _episode_project_multipliers(
        store, "default", ["ep_sys", "ep_note", "ep_real"], "/x/Engram", cfg, {}
    )
    assert mult.get("ep_sys") == pytest.approx(0.3)
    assert mult.get("ep_note") == pytest.approx(0.3)
    assert "ep_real" not in mult
