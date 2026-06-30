"""Execute scripted showcase beats and format terminal output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engram.showcase.beats import SHOWCASE_BEATS, ShowcaseBeat
from engram.showcase.resources import prepare_showcase_db
from engram.showcase.runtime import open_showcase_manager

SHOWCASE_GROUP_ID = "showcase"


@dataclass
class ShowcaseBeatResult:
    beat: ShowcaseBeat
    action: str
    recall_results: list[dict[str, Any]] = field(default_factory=list)
    context_payload: dict[str, Any] | None = None
    highlights: list[str] = field(default_factory=list)
    matched_tokens: list[str] = field(default_factory=list)
    passed: bool = False
    error: str | None = None


def _flatten_result_text(result: dict[str, Any]) -> str:
    chunks: list[str] = []
    entity = result.get("entity")
    if isinstance(entity, dict):
        for key in ("name", "summary", "entity_type"):
            value = entity.get(key)
            if isinstance(value, str):
                chunks.append(value)
    for key in ("content", "summary", "cue_text", "text"):
        value = result.get(key)
        if isinstance(value, str):
            chunks.append(value)
    episode = result.get("episode")
    if isinstance(episode, dict):
        content = episode.get("content")
        if isinstance(content, str):
            chunks.append(content)
    return " ".join(chunks).lower()


def _context_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("context", "briefing", "text", "summary"):
        value = payload.get(key)
        if isinstance(value, str):
            parts.append(value)
    sections = payload.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                for key in ("title", "content", "summary"):
                    value = section.get(key)
                    if isinstance(value, str):
                        parts.append(value)
    return " ".join(parts).lower()


def _highlights_from_results(results: list[dict[str, Any]], limit: int = 4) -> list[str]:
    highlights: list[str] = []
    for result in results[:limit]:
        entity = result.get("entity")
        if isinstance(entity, dict) and entity.get("name"):
            summary = entity.get("summary") or entity.get("entity_type") or ""
            highlights.append(f"{entity['name']}: {summary}".strip(": "))
            continue
        episode = result.get("episode")
        if isinstance(episode, dict):
            content = episode.get("content")
            if isinstance(content, str) and content.strip():
                highlights.append(content.strip()[:120])
                continue
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            highlights.append(content.strip()[:120])
    return highlights


async def execute_showcase_beat(
    manager,
    beat: ShowcaseBeat,
    *,
    group_id: str = SHOWCASE_GROUP_ID,
) -> ShowcaseBeatResult:
    outcome = ShowcaseBeatResult(beat=beat, action=beat.action)
    try:
        if beat.action == "recall":
            results = await manager.recall(
                beat.query,
                group_id=group_id,
                limit=8,
                record_access=False,
                interaction_source="showcase_recall",
            )
            outcome.recall_results = results
            blob = " ".join(_flatten_result_text(item) for item in results)
            outcome.highlights = _highlights_from_results(results)
            outcome.matched_tokens = [
                token for token in beat.expect_tokens if token.lower() in blob
            ]
            outcome.passed = bool(outcome.matched_tokens)
            return outcome

        context = await manager.get_context(
            group_id=group_id,
            topic_hint=beat.query,
            format="structured",
            operation_source="showcase_context",
        )
        outcome.context_payload = context
        blob = _context_text(context)
        outcome.highlights = []
        context_text = context.get("context") or context.get("briefing")
        if isinstance(context_text, str) and context_text.strip():
            for line in context_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") and "Liam" in stripped:
                    outcome.highlights.append(stripped[:200])
                    break
            if not outcome.highlights:
                first_line = next(
                    (line.strip() for line in context_text.splitlines() if line.strip()),
                    context_text.strip(),
                )
                outcome.highlights.append(first_line[:200])
        outcome.matched_tokens = [
            token for token in beat.expect_tokens if token.lower() in blob
        ]
        outcome.passed = bool(outcome.matched_tokens) or bool(outcome.highlights)
        return outcome
    except Exception as exc:  # pragma: no cover - surfaced in CLI/tests
        outcome.error = str(exc)
        outcome.passed = False
        return outcome


async def run_showcase_beats(
    *,
    db_path: Path | None = None,
    group_id: str = SHOWCASE_GROUP_ID,
    prepared_db_path: Path | None = None,
) -> tuple[list[ShowcaseBeatResult], Path]:
    resolved = prepared_db_path or prepare_showcase_db(db_path=db_path)
    manager, graph_store = await open_showcase_manager(resolved, group_id=group_id)
    try:
        results: list[ShowcaseBeatResult] = []
        for beat in SHOWCASE_BEATS:
            results.append(await execute_showcase_beat(manager, beat, group_id=group_id))
        return results, resolved
    finally:
        await graph_store.close()


def format_showcase_run(results: list[ShowcaseBeatResult]) -> str:
    lines: list[str] = []
    lines.append("Engram showcase (bundled lite demo.db)")
    lines.append("")
    for index, result in enumerate(results, start=1):
        beat = result.beat
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"Beat {index}/{len(results)}: {beat.title} [{status}]")
        lines.append(f"  User: {beat.user_message}")
        lines.append(f"  Action: {beat.action}({beat.query})")
        lines.append(f"  Story: {beat.narrative}")
        if result.highlights:
            lines.append("  Recall highlights:")
            for highlight in result.highlights:
                lines.append(f"    - {highlight}")
        if result.matched_tokens:
            lines.append(f"  Matched: {', '.join(result.matched_tokens)}")
        lines.append(f"  Suggested reply: {beat.answer_hint}")
        if result.error:
            lines.append(f"  Error: {result.error}")
        lines.append("")
    passed = sum(1 for item in results if item.passed)
    lines.append(f"Summary: {passed}/{len(results)} beats passed")
    return "\n".join(lines).rstrip() + "\n"


def showcase_open_instructions(db_path: Path, *, api_port: int = 8100) -> str:
    return (
        "Explore the seeded brain locally:\n"
        f"  ENGRAM_MODE=lite ENGRAM_SQLITE__PATH={db_path} engram serve --port {api_port}\n"
        f"  API docs: http://127.0.0.1:{api_port}/docs\n"
        "  Dashboard: cd dashboard && pnpm dev  # http://localhost:5173\n"
    )


def beat_results_to_json(results: list[ShowcaseBeatResult]) -> dict[str, Any]:
    beats: list[dict[str, Any]] = []
    for result in results:
        beats.append(
            {
                "id": result.beat.id,
                "title": result.beat.title,
                "user_message": result.beat.user_message,
                "action": result.beat.action,
                "query": result.beat.query,
                "narrative": result.beat.narrative,
                "answer_hint": result.beat.answer_hint,
                "expect_tokens": list(result.beat.expect_tokens),
                "passed": result.passed,
                "matched_tokens": result.matched_tokens,
                "highlights": result.highlights,
                "recall_results": result.recall_results,
                "context": result.context_payload,
                "steps": [
                    {"label": "Episode", "detail": result.beat.narrative},
                    {"label": "Cue", "detail": "Deterministic latent traces from seeded turns"},
                    {
                        "label": "Recall",
                        "detail": "; ".join(result.highlights) if result.highlights else "No hits",
                    },
                    {"label": "Answer", "detail": result.beat.answer_hint},
                ],
            }
        )
    return {
        "kind": "engram_showcase_export",
        "version": 1,
        "group_id": SHOWCASE_GROUP_ID,
        "beats": beats,
        "summary": {
            "passed": sum(1 for item in results if item.passed),
            "total": len(results),
        },
    }


def beat_results_to_markdown(results: list[ShowcaseBeatResult]) -> str:
    payload = beat_results_to_json(results)
    lines = ["# Engram Showcase Export", ""]
    for beat in payload["beats"]:
        lines.append(f"## {beat['title']}")
        lines.append(f"- User: {beat['user_message']}")
        lines.append(f"- Action: `{beat['action']}`")
        lines.append(f"- Passed: {beat['passed']}")
        if beat.get("highlights"):
            lines.append("- Recall highlights:")
            for highlight in beat["highlights"]:
                lines.append(f"  - {highlight}")
        lines.append(f"- Answer: {beat['answer_hint']}")
        lines.append("")
    lines.append(
        f"Summary: {payload['summary']['passed']}/{payload['summary']['total']} beats passed"
    )
    lines.append("")
    return "\n".join(lines)
