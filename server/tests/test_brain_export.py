"""The bulk export: full content, classification, edges deduped, identity core, sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engram.brain_export import classify_episode, export_brain

pytestmark = pytest.mark.asyncio


def test_classification_rules():
    assert (
        classify_episode("[project-bootstrap|server|docs/x.md]\n# T", "auto:bootstrap")
        == "bootstrap"
    )
    assert classify_episode("[project-bootstrap|server|x]", "auto:prompt") == "bootstrap"
    assert classify_episode("session started", "auto:session") == "session_marker"
    assert classify_episode("write-latency probe 1", "latency-probe") == "probe"
    assert (
        classify_episode(
            "<task-notification>\n<task-id>x</task-id>\n</task-notification>", "auto:prompt"
        )
        == "machinery"
    )
    assert (
        classify_episode(
            "[user|Engram] why was Thompson sampling removed from the ranker", "auto:prompt"
        )
        == "conversation"
    )


class _Store:
    def __init__(self):
        self.calls = []

    async def _query(self, endpoint, payload):
        self.calls.append(endpoint)
        if endpoint == "find_episodes_by_group":
            return [
                {
                    "id": 1,
                    "episode_id": "ep_b",
                    "content": "x" * 5000,
                    "source": "auto:prompt",
                    "created_at": "2026-06-02",
                },
                {
                    "id": 2,
                    "episode_id": "ep_a",
                    "content": "[project-bootstrap|s|f]\nq",
                    "source": "auto:bootstrap",
                    "created_at": "2026-06-01",
                },
            ]
        if endpoint == "find_entities_by_group":
            return [
                {"id": 10, "entity_id": "ent_a", "name": "A"},
                {"id": 11, "entity_id": "ent_b", "name": "B"},
            ]
        if endpoint == "get_outgoing_edges":
            return (
                [{"id": 99, "rel_id": "rel_ab", "predicate": "WORKS_ON"}]
                if payload["id"] == 10
                else [{"id": 99, "rel_id": "rel_ab", "predicate": "WORKS_ON"}]
            )
        if endpoint == "find_cues_by_group":
            return [{"episode_id": "ep_b", "hit_count": 3}]
        raise AssertionError(endpoint)

    async def get_identity_core_entities(self, group_id):
        return [SimpleNamespace(id="ent_a", name="A", entity_type="Person")]


async def test_export_writes_full_content_and_dedupes_edges(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "activation-snapshot.json").write_text("{}")
    report = await export_brain(_Store(), tmp_path / "out", engram_home=home)
    eps = [
        json.loads(line) for line in (tmp_path / "out" / "episodes.jsonl").read_text().splitlines()
    ]
    assert [e["episode_id"] for e in eps] == ["ep_a", "ep_b"]  # created_at order
    assert len(eps[1]["content"]) == 5000  # full content, not the REST 200-char cap
    assert report.by_classification == {"conversation": 1, "bootstrap": 1}
    assert report.relationships == 1  # the same edge seen from both endpoints, written once
    assert report.identity_core == 1 and report.cues == 1
    assert report.sidecars == ["activation-snapshot.json"]
    assert json.loads((tmp_path / "out" / "export-report.json").read_text())["episodes"] == 2
