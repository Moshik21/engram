"""Static contract: no NEW silent-swallow patterns in the storage layer.

The silent-inert bug class (eight confirmed production instances: phantom
writes, poisoned empty vectors, stale counts, invisible listings, dropped
episode updates...) always had one anatomy: an ``except`` block in
``server/engram/storage/`` that converts a failure into an empty value the
caller meters as success.

This test walks every except-handler in the storage package. A handler whose
body only produces an empty/None/pass outcome must carry an explicit
``# silent-ok: <reason>`` marker (on the except line or inside the handler)
justifying why swallowing is correct there. Anything unmarked fails CI.

Adding a marker is a REVIEWED decision, not a formality: prefer raising
(writes must never swallow) or degrading with an explicit status marker the
caller can see (the diagnostics ``countsStatus`` pattern).
"""

from __future__ import annotations

import ast
from pathlib import Path

STORAGE_ROOT = Path(__file__).resolve().parents[1] / "engram" / "storage"

SILENT_OK_MARKER = "# silent-ok:"


def _is_empty_value(node: ast.expr | None) -> bool:
    if node is None:  # bare `return`
        return True
    if isinstance(node, ast.Constant) and node.value in (None, "", 0, False):
        return True
    if isinstance(node, (ast.List, ast.Dict, ast.Tuple, ast.Set)) and not getattr(
        node, "elts", getattr(node, "keys", [])
    ):
        return True
    return False


def _handler_is_silent(handler: ast.ExceptHandler) -> bool:
    """True when the handler's outcome is only pass/continue/empty-return.

    Logging calls do not make a swallow loud — the caller still sees success.
    """
    meaningful = False
    for stmt in handler.body:
        if isinstance(stmt, (ast.Pass, ast.Continue)):
            continue
        if isinstance(stmt, ast.Return):
            if _is_empty_value(stmt.value):
                continue
            meaningful = True
            break
        if isinstance(stmt, ast.Raise):
            meaningful = True
            break
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            # logger.debug/info/warning/error/exception calls are not outcomes.
            if isinstance(func, ast.Attribute) and func.attr in {
                "debug",
                "info",
                "warning",
                "error",
                "exception",
            }:
                continue
            meaningful = True
            break
        # Assignments, other statements: assume the handler does real recovery.
        meaningful = True
        break
    return not meaningful


def _has_marker(source_lines: list[str], handler: ast.ExceptHandler) -> bool:
    start = max(0, handler.lineno - 2)  # marker may sit just above `except`
    end = min(len(source_lines), (handler.body[-1].end_lineno or handler.lineno) + 1)
    return any(SILENT_OK_MARKER in line for line in source_lines[start:end])


def test_storage_except_handlers_never_swallow_silently() -> None:
    offenders: list[str] = []
    for path in sorted(STORAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _handler_is_silent(node):
                continue
            if _has_marker(lines, node):
                continue
            rel = path.relative_to(STORAGE_ROOT.parent.parent)
            offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], (
        "Unmarked silent-swallow except handlers in storage/ (either raise, "
        "degrade with an explicit status marker, or justify with "
        f"'{SILENT_OK_MARKER} <reason>'):\n  " + "\n  ".join(offenders)
    )
