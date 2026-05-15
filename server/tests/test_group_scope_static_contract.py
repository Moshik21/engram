from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "engram"
FORBIDDEN_DEFAULT_FALLBACKS = (
    re.compile(r"group_id\s+or\s+[\"']default[\"']"),
    re.compile(r"gid\s*=\s*group_id\s+or\s+[\"']default[\"']"),
)
GROUP_SCOPED_GRAPH_CALLS = {
    "get_entity": 2,
    "find_entities": 3,
    "get_relationships": 5,
    "find_conflicting_relationships": 3,
    "find_existing_relationship": 4,
    "get_episode_by_id": 2,
    "get_episode_entities": 2,
    "get_episode_cue": 2,
    "get_episodes_for_entity": 2,
    "get_adjacent_episodes": 2,
    "link_episode_entity": 3,
    "update_episode": 3,
    "update_episode_cue": 3,
    "record_access": 3,
    "get_stats": 1,
}


def _production_paths() -> list[Path]:
    return sorted(ROOT.rglob("*.py"))


def test_optional_group_id_is_not_silently_narrowed_to_default() -> None:
    offenders: list[str] = []
    for path in _production_paths():
        text = path.read_text()
        for pattern in FORBIDDEN_DEFAULT_FALLBACKS:
            for match in pattern.finditer(text):
                rel_path = path.relative_to(ROOT.parent)
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{rel_path}:{line_no}: {match.group(0)}")

    assert offenders == []


def test_group_scoped_graph_calls_do_not_omit_group_scope() -> None:
    offenders: list[str] = []
    for path in _production_paths():
        text = path.read_text()
        tree = ast.parse(text)
        lines = text.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method_name = node.func.attr
            required_positional_count = GROUP_SCOPED_GRAPH_CALLS.get(method_name)
            if required_positional_count is None:
                continue
            if any(keyword.arg == "group_id" for keyword in node.keywords):
                continue
            if len(node.args) >= required_positional_count:
                continue
            rel_path = path.relative_to(ROOT.parent)
            offenders.append(
                f"{rel_path}:{node.lineno}: {method_name}: {lines[node.lineno - 1].strip()}"
            )

    assert offenders == []
