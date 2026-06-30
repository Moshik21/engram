"""Export and import human-editable captain preferences."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engram.models.entity import Entity

CAPTAIN_VERSION = 1
DEFAULT_CAPTAIN_PATH = Path.home() / ".engram" / "captain.md"
_VERSION_MARKER = "<!-- engram-captain-version:"
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^-\s+(.+?)\s*$", re.MULTILINE)


def captain_file_exists(path: Path | None = None) -> bool:
    target = path or DEFAULT_CAPTAIN_PATH
    return target.is_file() and target.stat().st_size > 0


def captain_file_summary(path: Path | None = None) -> str | None:
    target = path or DEFAULT_CAPTAIN_PATH
    if not captain_file_exists(target):
        return None
    return f"Portable prefs available at {target}"


def export_captain_markdown(entities: Sequence[Entity]) -> str:
    """Render identity_core entities to captain markdown."""
    identity_lines: list[str] = []
    style_lines: list[str] = []
    protected_lines: list[str] = []

    for entity in entities:
        if not getattr(entity, "identity_core", False):
            continue
        label = f"{entity.name} ({entity.entity_type}, identity_core)"
        if entity.entity_type.lower() in {"person", "user"}:
            identity_lines.append(f"- {entity.name}")
        elif entity.summary:
            style_lines.append(f"- {entity.summary}")
        protected_lines.append(f"- {label}")

    lines = [
        "# Captain preferences",
        f"{_VERSION_MARKER} {CAPTAIN_VERSION} -->",
        "",
        "## Identity",
    ]
    lines.extend(identity_lines or ["- (none yet)"])
    lines.extend(["", "## Working style"])
    lines.extend(style_lines or ["- (add preferences here)"])
    lines.extend(["", "## Protected entities"])
    lines.extend(protected_lines or ["- (none yet)"])
    lines.append("")
    return "\n".join(lines)


def parse_captain_markdown(text: str) -> dict[str, Any]:
    """Parse captain markdown into structured import payloads."""
    sections: dict[str, list[str]] = {}
    current = "preamble"
    for line in text.splitlines():
        section_match = re.match(r"^##\s+(.+?)\s*$", line.strip())
        if section_match:
            current = section_match.group(1).strip().lower()
            sections.setdefault(current, [])
            continue
        bullet_match = re.match(r"^-\s+(.+?)\s*$", line.strip())
        if bullet_match and current != "preamble":
            value = bullet_match.group(1).strip()
            if value.startswith("(") and value.endswith(")"):
                continue
            sections.setdefault(current, []).append(value)

    remember_items: list[str] = []
    for key in ("identity", "working style"):
        remember_items.extend(sections.get(key, []))

    return {
        "version": CAPTAIN_VERSION,
        "sections": sections,
        "remember_items": remember_items,
        "protected_entities": sections.get("protected entities", []),
    }


def write_captain_file(
    entities: Sequence[Entity],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    target = path or DEFAULT_CAPTAIN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    markdown = export_captain_markdown(entities)
    target.write_text(markdown, encoding="utf-8")
    return {
        "operation": "captain.export",
        "status": "ok",
        "path": str(target),
        "entity_count": sum(1 for e in entities if getattr(e, "identity_core", False)),
        "bytes": len(markdown.encode("utf-8")),
    }


def read_captain_import_payload(
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    target = path or DEFAULT_CAPTAIN_PATH
    if not target.is_file():
        raise FileNotFoundError(f"Captain file not found: {target}")
    parsed = parse_captain_markdown(target.read_text(encoding="utf-8"))
    return {
        "operation": "captain.import",
        "status": "ok",
        "path": str(target),
        "remember_items": parsed["remember_items"],
        "item_count": len(parsed["remember_items"]),
    }
