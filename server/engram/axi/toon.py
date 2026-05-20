"""Small TOON-compatible renderer for Engram AXI payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def render_toon(payload: Mapping[str, Any]) -> str:
    """Render a compact, agent-readable representation of a mapping."""
    return "\n".join(_render_mapping(payload, indent=0)).rstrip() + "\n"


def _render_mapping(mapping: Mapping[str, Any], *, indent: int) -> list[str]:
    lines: list[str] = []
    for key, value in mapping.items():
        if value in (None, [], {}):
            continue
        lines.extend(_render_key_value(str(key), value, indent=indent))
    return lines


def _render_key_value(key: str, value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, Mapping):
        nested = _render_mapping(value, indent=indent + 2)
        if not nested:
            return []
        return [f"{prefix}{key}:"] + nested
    if _is_uniform_object_list(value):
        return _render_object_table(key, value, indent=indent)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        rows: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(f"{prefix}{key}:")
                rows.extend(_render_mapping(item, indent=indent + 2))
            else:
                rows.append(f"{prefix}{key}: {_scalar(item)}")
        return rows
    return [f"{prefix}{key}: {_scalar(value)}"]


def _is_uniform_object_list(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return False
    if not value or not all(isinstance(item, Mapping) for item in value):
        return False
    keys = list(value[0].keys())
    return bool(keys) and all(list(item.keys()) == keys for item in value)


def _render_object_table(key: str, rows: Sequence[Mapping[str, Any]], *, indent: int) -> list[str]:
    prefix = " " * indent
    fields = list(rows[0].keys())
    rendered = [f"{prefix}{key}[{len(rows)}]{{{','.join(fields)}}}:"]
    row_prefix = " " * (indent + 2)
    for row in rows:
        rendered.append(f"{row_prefix}{','.join(_scalar(row.get(field)) for field in fields)}")
    return rendered


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if _needs_quotes(text):
        return json.dumps(text, ensure_ascii=False)
    return text


def _needs_quotes(text: str) -> bool:
    if text != text.strip():
        return True
    special = {":", ",", "{", "}", "[", "]", "#", '"', "\n", "\r", "\t"}
    return any(char in text for char in special)
