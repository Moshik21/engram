"""Tests for captain preference export/import."""

from __future__ import annotations

from datetime import datetime

from engram.identity.captain_export import (
    export_captain_markdown,
    parse_captain_markdown,
    write_captain_file,
)
from engram.models.entity import Entity


def _entity(
    *,
    name: str,
    entity_type: str,
    summary: str | None = None,
    identity_core: bool = True,
) -> Entity:
    return Entity(
        id=f"ent_{name.lower()}",
        name=name,
        entity_type=entity_type,
        summary=summary,
        identity_core=identity_core,
        group_id="default",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


def test_export_captain_markdown_includes_identity_core_entities() -> None:
    markdown = export_captain_markdown(
        [
            _entity(name="Konner", entity_type="Person"),
            _entity(
                name="Passive harness memory",
                entity_type="Preference",
                summary="Prefers passive harness memory over per-turn observe",
            ),
        ]
    )
    assert "engram-captain-version: 1" in markdown
    assert "## Identity" in markdown
    assert "- Konner" in markdown
    assert "Prefers passive harness memory" in markdown
    assert "identity_core" in markdown


def test_parse_captain_markdown_round_trip_items() -> None:
    markdown = export_captain_markdown([_entity(name="Liam", entity_type="Person")])
    parsed = parse_captain_markdown(markdown)
    assert "Liam" in parsed["remember_items"]
    assert parsed["version"] == 1


def test_write_captain_file_writes_markdown(tmp_path) -> None:
    target = tmp_path / "captain.md"
    payload = write_captain_file([_entity(name="Ada", entity_type="Person")], path=target)
    assert payload["status"] == "ok"
    assert target.is_file()
    assert "Ada" in target.read_text(encoding="utf-8")