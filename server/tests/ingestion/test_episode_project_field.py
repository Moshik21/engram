"""Episodes carry a real project field at capture time.

Before 2026-09-04 the hook's project name lived only in the "[role|project]"
content header; scoping parsed it back out of the text on every recall. The
field rides inside the existing encoding_context JSON blob, so no Helix
schema change is needed, and old rows still fall back to the header.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_surface import store_observation
from engram.ingestion.salience import (
    decode_context_field,
    decode_salience_class,
    encode_context_field,
    encode_salience_class,
)
from engram.retrieval.pipeline import _episode_project_multipliers

pytestmark = pytest.mark.asyncio


def test_project_round_trips_next_to_the_salience_class() -> None:
    blob = encode_context_field(encode_salience_class(None, "substantive"), "project", "Engram")
    assert decode_context_field(blob, "project") == "Engram"
    assert decode_salience_class(blob) == "substantive"
    # Empty project is byte-identical (the kill-switch contract of the blob).
    assert encode_context_field("{}", "project", None) == "{}"
    assert encode_context_field(None, "project", "") is None
    # A non-object blob (reflect cluster keys) is left alone and decodes to None.
    assert encode_context_field("cluster:7", "project", "Engram") == "cluster:7"
    assert decode_context_field("cluster:7", "project") is None
    assert decode_context_field(None, "project") is None


async def test_store_observation_forwards_the_project_to_the_manager() -> None:
    seen: list[dict] = []

    class _Manager:
        async def store_episode(self, **kwargs):
            seen.append(kwargs)
            return "ep_1"

    await store_observation(_Manager(), content="x", group_id="default", project="Engram")
    await store_observation(_Manager(), content="x", group_id="default")
    assert seen[0]["project"] == "Engram"
    assert "project" not in seen[1]


class _Store:
    def __init__(self, rows: dict[str, tuple[str, str | None]]) -> None:
        self.rows = rows

    async def get_episode_by_id(self, episode_id: str, group_id: str):
        row = self.rows.get(episode_id)
        if row is None:
            return None
        content, project = row
        return SimpleNamespace(id=episode_id, content=content, project=project, source=None)


async def test_scoping_prefers_the_field_and_falls_back_to_the_header() -> None:
    cfg = ActivationConfig(
        recall_other_project_multiplier=0.5,
        recall_short_episode_floor_chars=0,
        recall_machinery_episode_multiplier=1.0,
    )
    store = _Store(
        {
            "field-other": ("no header here at all " * 4, "shielded-bid"),
            "field-same": ("[user|shielded-bid] header lies, field wins " * 2, "Engram"),
            "header-other": ("[user|shielded-bid] old row, header only " * 2, None),
            "unknown": ("plain text with nothing " * 3, None),
        }
    )
    mult = await _episode_project_multipliers(
        store, "default", list(store.rows), "/Users/k/Engram", cfg, None
    )
    assert mult == {"field-other": 0.5, "header-other": 0.5}
