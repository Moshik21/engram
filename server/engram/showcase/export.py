"""Export showcase beat payloads for website rendering."""

from __future__ import annotations

import json
from pathlib import Path

from engram.showcase.runner import (
    beat_results_to_json,
    beat_results_to_markdown,
    run_showcase_beats,
)


async def export_showcase_payload(
    *,
    db_path: Path | None = None,
    out_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict:
    results = await run_showcase_beats(db_path=db_path)
    payload = beat_results_to_json(results)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(beat_results_to_markdown(results), encoding="utf-8")
    return payload


def export_showcase_markdown(results_payload: dict) -> str:
    beats = results_payload.get("beats", [])
    lines = ["# Engram Showcase Export", ""]
    for beat in beats:
        lines.append(f"## {beat.get('title', beat.get('id', 'beat'))}")
        lines.append(f"- User: {beat.get('user_message', '')}")
        lines.append(f"- Action: `{beat.get('action', '')}`")
        lines.append(f"- Passed: {beat.get('passed', False)}")
        highlights = beat.get("highlights") or []
        if highlights:
            lines.append("- Recall highlights:")
            for highlight in highlights:
                lines.append(f"  - {highlight}")
        lines.append(f"- Answer: {beat.get('answer_hint', '')}")
        lines.append("")
    summary = results_payload.get("summary") or {}
    lines.append(f"Summary: {summary.get('passed', 0)}/{summary.get('total', 0)} beats passed")
    lines.append("")
    return "\n".join(lines)