"""M5.2 fresh-agent suite (gate B6 instrument).

Simulates a FRESH-context agent that has ONLY Engram tools, versus the
project-file-only control an agent gets when Engram is degraded (the
fallback lane's packets). Fully local and deterministic — no external LLM.

Scripted agent loop, per battery question:
1. ``get_context`` once per session (session start, amortized across
   questions; counted separately in the report).
2. ``recall`` with the verbatim question.
3. On miss, ONE reformulated recall. Deterministic reformulation rule:
   lowercase the question, drop stopwords/question words, keep the
   remaining key noun-phrase tokens in original order, joined by spaces
   (e.g. "what is the flip condition for usage ranking" ->
   "flip condition usage ranking").
4. "Answers" by extraction: containment scoring of the surfaced text
   against ``expected_tokens`` — the same scorer as M5.1 (imported).

Control arm: the same questions scored against the project files the
fallback lane surfaces when Engram is degraded (each file is one packet;
a group must land wholly inside one file). No tool calls by construction.

Metrics per question: hit, tool calls, chars surfaced (token-cost proxy).
Lift = engram hits - project-file hits.

``--judge ollama`` is a stub for future local-LLM grading; containment is
the only implemented grader.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from engram.evaluation.battery import (
    BATTERY_PATH,
    group_contained,
    load_battery,
    score_question,
    top3_result_texts,
)

# Default fallback-lane packets: the project files an agent sees when
# Engram is degraded. Paths relative to the repo root (parents[3]).
DEFAULT_PROJECT_FILES = ("CLAUDE.md", "docs/CURRENT_HANDOFF.md")

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Deterministic reformulation stopword list (documented in the module
# docstring). Question words + closed-class glue words only.
_REFORMULATE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "did",
        "do",
        "does",
        "for",
        "from",
        "his",
        "her",
        "how",
        "in",
        "is",
        "it",
        "its",
        "now",
        "of",
        "on",
        "or",
        "the",
        "their",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.'-]+")


def reformulate_question(question: str) -> str:
    """Deterministic reformulation: keep key noun-phrase tokens in order."""
    tokens = _TOKEN_RE.findall(question)
    kept = [t for t in tokens if t.casefold() not in _REFORMULATE_STOPWORDS]
    return " ".join(kept)


def _score_projectfile_arm(question: dict[str, Any], file_texts: list[str]) -> dict[str, Any]:
    groups = question.get("expected_tokens") or []
    hit_group = next(
        (g for g in groups if any(group_contained(text, g) for text in file_texts)),
        None,
    )
    return {
        "id": question.get("id"),
        "hit": hit_group is not None,
        "hit_group": hit_group,
    }


def load_project_file_texts(paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Load fallback-lane packets. Missing files are reported, not swallowed."""
    resolved = paths or [_REPO_ROOT / p for p in DEFAULT_PROJECT_FILES]
    packets: list[dict[str, Any]] = []
    for path in resolved:
        try:
            text = Path(path).read_text(encoding="utf-8")
            packets.append({"path": str(path), "text": text, "chars": len(text)})
        except OSError as exc:
            packets.append({"path": str(path), "text": "", "chars": 0, "error": str(exc)})
    return packets


def run_fresh_agent(
    questions: list[dict[str, Any]],
    *,
    recall_fn: Callable[[str], dict[str, Any]],
    context_fn: Callable[[], str],
    project_packets: list[dict[str, Any]],
    judge: str = "containment",
) -> dict[str, Any]:
    """Pure agent-loop core: injected tools, deterministic, lite-testable."""
    if judge != "containment":
        raise NotImplementedError(f"judge={judge!r} is a stub; only 'containment' is implemented")
    started = time.perf_counter()

    # Session start: one get_context for the whole fresh session.
    context_blob = context_fn()
    session_tool_calls = 1
    session_chars = len(context_blob)

    file_texts = [p["text"] for p in project_packets if p.get("text")]
    projectfile_chars = sum(p.get("chars", 0) for p in project_packets)

    per_question: list[dict[str, Any]] = []
    for question in questions:
        q = str(question["q"])
        tool_calls = 0
        chars = 0
        texts: list[str] = []
        reformulated: str | None = None
        errors: list[str] = []

        row: dict[str, Any] = {}
        for attempt_query in (q, reformulate_question(q)):
            if tool_calls and attempt_query == q:
                break  # reformulation identical to the question: no retry value
            payload = recall_fn(attempt_query)
            tool_calls += 1
            if payload.get("error"):
                errors.append(str(payload["error"]))
            new_texts = top3_result_texts(payload)
            texts.extend(new_texts)
            chars += sum(len(t) for t in new_texts)
            row = score_question(question, texts, context_blob)
            if row["hit"]:
                break
            reformulated = reformulate_question(q)

        pf = _score_projectfile_arm(question, file_texts)
        entry = {
            **row,
            "tool_calls": tool_calls,
            "chars_surfaced": chars,
            "reformulated_query": reformulated if tool_calls > 1 else None,
            "projectfile_hit": pf["hit"],
        }
        if errors:
            entry["recall_errors"] = errors
        per_question.append(entry)

    engram_hits = sum(1 for r in per_question if r["hit"])
    projectfile_hits = sum(1 for r in per_question if r["projectfile_hit"])
    total_chars = session_chars + sum(r["chars_surfaced"] for r in per_question)
    return {
        "mode": "fresh_agent",
        "judge": judge,
        "engram_score": engram_hits,
        "projectfile_score": projectfile_hits,
        "lift": engram_hits - projectfile_hits,
        "total": len(per_question),
        "session_tool_calls": session_tool_calls,
        "session_context_chars": session_chars,
        "tool_calls_total": session_tool_calls + sum(r["tool_calls"] for r in per_question),
        "chars_surfaced_total": total_chars,
        "projectfile_chars": projectfile_chars,
        "project_files": [{k: v for k, v in p.items() if k != "text"} for p in project_packets],
        "questions": per_question,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def run_fresh_agent_against_live(
    *,
    server_url: str = "http://127.0.0.1:8100",
    battery_path: Path | None = None,
    project_files: list[Path] | None = None,
    judge: str = "containment",
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Live wrapper: HTTP recall/context tools over a running server."""
    battery = load_battery(battery_path or BATTERY_PATH)
    base = server_url.rstrip("/")

    def _get(path: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{base}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def recall_fn(query: str) -> dict[str, Any]:
        try:
            return _get(f"/api/knowledge/recall?q={urllib.parse.quote(query)}&limit=3")
        except Exception as exc:
            return {"error": str(exc), "items": []}

    def context_fn() -> str:
        try:
            payload = _get("/api/knowledge/context?max_tokens=1500")
        except Exception as exc:
            payload = {"error": str(exc)}
        from engram.evaluation.battery import _flatten_text

        parts: list[str] = []
        _flatten_text(payload, parts)
        return "\n".join(parts)

    result = run_fresh_agent(
        battery["questions"],
        recall_fn=recall_fn,
        context_fn=context_fn,
        project_packets=load_project_file_texts(project_files),
        judge=judge,
    )
    result["server_url"] = base
    return result


def format_fresh_agent_report(result: dict[str, Any]) -> str:
    lines = [
        "# Fresh-agent suite (M5.2, gate B6): "
        f"engram {result['engram_score']}/{result['total']} vs "
        f"project-file {result['projectfile_score']}/{result['total']} "
        f"(lift {result['lift']:+d})",
        "",
        f"- judge: {result['judge']} (deterministic containment)",
        f"- tool calls: {result['tool_calls_total']} "
        f"(incl. 1 session get_context, {result['session_context_chars']} chars)",
        f"- chars surfaced (engram arm): {result['chars_surfaced_total']}",
        f"- project-file arm packet size: {result['projectfile_chars']} chars, 0 tool calls",
        f"- duration: {result['duration_ms']} ms",
        "",
        "| question | engram | projfile | calls | chars | reformulated |",
        "|---|---|---|---|---|---|",
    ]
    for row in result.get("questions") or []:
        lines.append(
            f"| {row.get('id')} | {'HIT' if row.get('hit') else 'miss'} "
            f"| {'HIT' if row.get('projectfile_hit') else 'miss'} "
            f"| {row.get('tool_calls')} | {row.get('chars_surfaced')} "
            f"| {row.get('reformulated_query') or ''} |"
        )
    errored = [r["id"] for r in result.get("questions") or [] if r.get("recall_errors")]
    if errored:
        lines.append("")
        lines.append(f"- recall errors on: {', '.join(str(i) for i in errored)}")
    return "\n".join(lines) + "\n"
