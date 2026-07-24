"""Static contract: metric surfaces must not fabricate measurements.

THE RULE (docs/product/INSTRUMENT_AUDIT.md): a metric must be either accurate
or absent, never plausible-but-wrong. An absent metric prompts investigation.
A wrong metric ENDS investigation with a false answer, which is strictly worse
than no metric at all.

This test walks the dict literals built by Engram's metric surfaces. A key whose
NAME implies a measurement (``*_count``, ``*_rate``, ``avg_*``, ``totalMs`` ...)
bound to a zeroish LITERAL (``0``, ``0.0``, ``False``, ``[]``, ``{}``) is a
fabricated measurement: the payload contains real computed values next to it, so
a consumer cannot tell the fake field from the measured ones.

``None`` is deliberately NOT flagged. ``None`` is the honest encoding of
"absent" and is exactly what the rule asks for.

Two escape hatches, both explicit:

* ``# metric-ok: <reason>`` on or beside the line — a REVIEWED decision that
  the literal is correct here (e.g. a genuinely empty summary of empty input).
* ``KNOWN_FABRICATIONS`` — the ledger of confirmed lies that are documented in
  INSTRUMENT_AUDIT.md but not yet fixed. It is a RATCHET: entries may only be
  removed. ``test_ledger_has_no_stale_entries`` fails when a ledgered site is
  fixed, forcing the ledger (and the audit doc) to be updated with it.

``test_scanner_is_not_inert`` is the canary. Engram's dominant bug class is
"code that runs and whose result is discarded" — a test that silently stops
detecting anything is the same disease. The canary asserts the scanner still
fires on a synthetic fabrication, so tightening the matcher into uselessness
fails loudly instead of passing quietly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[1]

# Surfaces whose output is read as a MEASUREMENT and acted on: /api/stats,
# /api/storage, /api/episodes, /api/lifecycle/summary, /api/knowledge/runtime.
METRIC_SURFACES = (
    "engram/storage/helix/graph.py",
    "engram/storage/sqlite/graph.py",
    "engram/storage/diagnostics.py",
    "engram/retrieval/graph_state.py",
    "engram/retrieval/runtime_state.py",
    "engram/lifecycle_summary.py",
)

METRIC_OK_MARKER = "# metric-ok:"

_METRIC_SUFFIX = re.compile(r"(?:^|_)(count|counts|total|totals|rate|ratio|avg|mean|pct|ms)$", re.I)
_METRIC_CAMEL = re.compile(r"(Count|Counts|Total|Totals|Rate|Ratio|Avg|Mean|Pct|Ms)$")
_METRIC_PREFIX = re.compile(r"^(avg|mean|total|num|count|pct)_", re.I)

# Confirmed fabrications, catalogued in docs/product/INSTRUMENT_AUDIT.md.
# Keyed by (surface, metric key) — NOT line number, which drifts.
# THIS SET MAY ONLY SHRINK.
KNOWN_FABRICATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # AUDIT-2: projection yield is a literal, not a measurement. The
        # "0.0 rels/episode" on /api/stats is typed in, not counted.
        ("engram/storage/helix/graph.py", "relationship_count"),
        ("engram/storage/helix/graph.py", "avg_relationships_per_projected_episode"),
        # AUDIT-8: the fast count route returns exact entity/episode/relationship
        # counts but zero-fills the whole projection block beside them.
        ("engram/storage/helix/graph.py", "state_counts"),
        ("engram/storage/helix/graph.py", "attempted_episode_count"),
        ("engram/storage/helix/graph.py", "total_attempts"),
        ("engram/storage/helix/graph.py", "failure_count"),
        ("engram/storage/helix/graph.py", "dead_letter_count"),
        ("engram/storage/helix/graph.py", "failure_rate"),
        ("engram/storage/helix/graph.py", "avg_processing_duration_ms"),
        ("engram/storage/helix/graph.py", "avg_time_to_projection_ms"),
        ("engram/storage/helix/graph.py", "linked_entity_count"),
        ("engram/storage/helix/graph.py", "avg_linked_entities_per_projected_episode"),
        # AUDIT-3: /api/episodes reports every episode as having zero facts.
        ("engram/retrieval/graph_state.py", "factsCount"),
        # AUDIT-9: same fabrication on the /api/lifecycle/summary episode rows.
        ("engram/lifecycle_summary.py", "factsCount"),
        # AUDIT-10: temporal graph nodes report accessCount 0 for every node.
        ("engram/retrieval/graph_state.py", "accessCount"),
        # AUDIT-11: the fast runtime packet reports "no artifacts" when it means
        # "did not look"; its `status: not_inspected` guard has no reader.
        ("engram/retrieval/runtime_state.py", "artifactCount"),
        ("engram/retrieval/runtime_state.py", "freshArtifactCount"),
        ("engram/retrieval/runtime_state.py", "staleArtifactCount"),
    }
)


def _is_metric_key(name: str) -> bool:
    return bool(
        _METRIC_SUFFIX.search(name) or _METRIC_CAMEL.search(name) or _METRIC_PREFIX.match(name)
    )


def _is_fabricated_measurement(node: ast.expr) -> bool:
    """Zeroish literals that READ as a real measurement.

    ``None`` is exempt on purpose: absent is the honest encoding.
    """
    if isinstance(node, ast.Constant) and node.value in (0, 0.0, False):
        return True
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    return False


def _has_computed_value(node: ast.expr) -> bool:
    """True when the dict tree carries at least one non-literal value.

    An all-literal dict is an honest constant (an empty summary of empty input).
    A dict that MIXES computed values with a metric literal is the dangerous
    shape: the function had data and chose to fake that one field.
    """
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.IfExp)):
            return True
    return False


def _scan_dict(node: ast.Dict, hits: list[tuple[int, str, str]]) -> None:
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if _is_metric_key(key.value) and _is_fabricated_measurement(value):
                hits.append((key.lineno, key.value, ast.unparse(value)))
        if isinstance(value, ast.Dict):
            _scan_dict(value, hits)


def scan_source(source: str) -> list[tuple[int, str, str]]:
    """Return (lineno, metric_key, literal) for every fabricated measurement."""
    tree = ast.parse(source)
    nested = {
        id(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for value in node.values
        if isinstance(value, ast.Dict)
    }
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or id(node) in nested:
            continue
        if not _has_computed_value(node):
            continue
        _scan_dict(node, hits)
    return hits


def _marked_ok(lines: list[str], lineno: int) -> bool:
    window = lines[max(0, lineno - 2) : lineno + 1]
    return any(METRIC_OK_MARKER in line for line in window)


def _live_hits() -> dict[str, list[tuple[int, str, str]]]:
    found: dict[str, list[tuple[int, str, str]]] = {}
    for rel in METRIC_SURFACES:
        source = (SERVER_ROOT / rel).read_text(encoding="utf-8")
        lines = source.splitlines()
        kept = [hit for hit in scan_source(source) if not _marked_ok(lines, hit[0])]
        if kept:
            found[rel] = kept
    return found


def test_metric_surfaces_do_not_fabricate_measurements() -> None:
    offenders = [
        f"{rel}:{lineno}  {key!r} = {literal}"
        for rel, hits in _live_hits().items()
        for lineno, key, literal in hits
        if (rel, key) not in KNOWN_FABRICATIONS
    ]
    assert offenders == [], (
        "New fabricated measurements on a metric surface. A metric must be "
        "either accurate or ABSENT (None) — never a plausible-but-wrong "
        "literal. Compute it, drop the key, or justify with "
        f"'{METRIC_OK_MARKER} <reason>':\n  " + "\n  ".join(offenders)
    )


def test_scanner_is_not_inert() -> None:
    """Canary: the scanner must still fire on a synthetic fabrication.

    Without this, tightening the matcher would make the contract silently
    vacuous — the exact bug class this file exists to prevent.
    """
    fixture = (
        "def build_report(rows):\n"
        "    return {\n"
        '        "episodes": len(rows),\n'
        '        "relationship_count": 0,\n'
        '        "avg_latency_ms": 0.0,\n'
        '        "lastSeenAt": None,\n'
        "    }\n"
    )
    keys = {key for _, key, _ in scan_source(fixture)}
    assert "relationship_count" in keys, "scanner missed a hardcoded *_count"
    assert "avg_latency_ms" in keys, "scanner missed a hardcoded avg_* rate"
    assert "lastSeenAt" not in keys, "None must stay legal — absent is the honest encoding"

    honest = (
        "def empty_summary():\n"
        '    return {"count": 0, "avg": 0.0, "p95": 0.0}\n'
    )
    assert scan_source(honest) == [], "all-literal constants are honest, not fabrications"


def test_ledger_has_no_stale_entries() -> None:
    """The ratchet: a fixed fabrication must be struck from the ledger and the doc."""
    live = {(rel, key) for rel, hits in _live_hits().items() for _, key, _ in hits}
    stale = sorted(KNOWN_FABRICATIONS - live)
    assert stale == [], (
        "KNOWN_FABRICATIONS lists sites that no longer fabricate. Delete these "
        "entries and update docs/product/INSTRUMENT_AUDIT.md to record the "
        f"fix:\n  {stale}"
    )
