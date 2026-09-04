"""Re-seed: verbatim content, original source/session/date, project from header, class filter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engram.brain_import import import_episodes, row_to_payload

pytestmark = pytest.mark.asyncio


def test_payload_keeps_content_and_recovers_project_from_the_header():
    row = {
        "content": "[assistant|Engram] Spreading passed its bar.",
        "source": "auto:response",
        "session_id": "s1",
        "created_at": "2026-06-01T10:00:00+00:00",
    }
    p = row_to_payload(row)
    assert p["content"] == row["content"] and p["role"] == "assistant" and p["project"] == "Engram"
    assert p["source"] == "auto:response" and p["session_id"] == "s1"
    assert p["conversation_date"] == "2026-06-01T10:00:00+00:00"
    assert row_to_payload({"content": "short"}) is None


async def test_import_posts_only_the_chosen_classes(tmp_path: Path):
    rows = [
        {
            "episode_id": "a",
            "classification": "conversation",
            "content": "[user|Engram] " + "x" * 40,
            "source": "auto:prompt",
        },
        {
            "episode_id": "b",
            "classification": "bootstrap",
            "content": "[project-bootstrap|s|f]\n" + "y" * 40,
            "source": "auto:bootstrap",
        },
        {
            "episode_id": "c",
            "classification": "machinery",
            "content": "<task-notification>" * 3,
            "source": "auto:prompt",
        },
    ]
    (tmp_path / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    seen = []

    async def post(payload):
        seen.append(payload)
        return {"status": "observed"}

    report = await import_episodes(tmp_path, post, rate_per_s=0)
    assert [p["content"][:6] for p in seen] == ["[user|"]
    assert report["chosen"] == 1 and report["posted"] == 1 and report["statuses"] == {"observed": 1}


async def test_import_survives_unicode_line_separators_in_content(tmp_path: Path):
    """The export writes ensure_ascii=False, so U+2028 lands raw in the file."""
    row = {
        "episode_id": "u",
        "classification": "conversation",
        "content": "[user|Engram] first\u2028second line of a real answer",
        "source": "auto:prompt",
    }
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    seen = []

    async def post(payload):
        seen.append(payload)
        return {"status": "observed"}

    report = await import_episodes(tmp_path, post, rate_per_s=0)
    assert report["posted"] == 1 and "\u2028" in seen[0]["content"]
