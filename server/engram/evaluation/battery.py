"""Agent-experience battery runner (M5.1, gate B1/B4 instrument).

Scores server/tests/rigs/agent_experience_battery.json by containment@3:
a question is a HIT when any expected_tokens group is fully present
(case-insensitive) in the text of the top-3 recall results. get_context
containment is reported as a supplementary signal, not the score.

Modes:
- against-live: HTTP against a running server (axi recall surface + context).
- seeded: lite fixture brain built in-process from planted episodes that
  mirror the battery questions, so CI can run without the dogfood brain.
  Questions carrying ``"live_only": true`` in the rig are skipped in seeded
  mode.

Gate B4: no top-3 result may match the machinery-class signatures. The
predicate is imported from Lane B's salience module when present; a minimal
vendored regex is used otherwise (kept intentionally narrow: protocol frames,
task notifications, command-output/exit-code dumps, tool-use ids).
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BATTERY_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "rigs" / "agent_experience_battery.json"
)

_VENDORED_MACHINERY_PATTERNS = [
    re.compile(r"<task-notification\b", re.IGNORECASE),
    re.compile(r"<system-reminder>", re.IGNORECASE),
    re.compile(r"<command-(?:name|output|message)>", re.IGNORECASE),
    re.compile(r"\btoolu_[A-Za-z0-9]{6,}"),
    re.compile(r"\bexit code[:\s]+\d+", re.IGNORECASE),
    re.compile(r"^Command output:", re.IGNORECASE | re.MULTILINE),
]


def is_machinery_text(text: str) -> bool:
    """True when text matches machinery-class signatures (gate B4).

    Prefers the salience classifier (M1.1) when importable; falls back to
    the vendored minimal regexes on older trees only.
    """
    try:
        from engram.ingestion.salience import is_machinery

        return bool(is_machinery(text))
    except ImportError:
        return any(p.search(text) for p in _VENDORED_MACHINERY_PATTERNS)


def machinery_predicate_source() -> str:
    try:
        from engram.ingestion.salience import is_machinery  # noqa: F401

        return "salience_classifier"
    except ImportError:
        return "vendored_regex"


def load_battery(path: Path | None = None) -> dict[str, Any]:
    with open(path or BATTERY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _flatten_text(obj: Any, parts: list[str]) -> None:
    if isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _flatten_text(value, parts)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _flatten_text(value, parts)


def top3_result_texts(payload: dict[str, Any]) -> list[str]:
    """Flatten the top-3 surfaced results (items, else packets) to text blobs."""
    rows = payload.get("items") or payload.get("results") or []
    if not rows:
        rows = payload.get("packets") or payload.get("cached_packets") or []
    texts: list[str] = []
    for row in list(rows)[:3]:
        parts: list[str] = []
        _flatten_text(row, parts)
        texts.append("\n".join(parts))
    return texts


def group_contained(text: str, group: list[str]) -> bool:
    lowered = text.casefold()
    return all(str(token).casefold() in lowered for token in group)


def score_question(
    question: dict[str, Any],
    result_texts: list[str],
    context_blob: str = "",
) -> dict[str, Any]:
    groups = question.get("expected_tokens") or []
    # Per-result containment: a group must land wholly inside ONE surfaced
    # result — joining texts would let multi-token groups hit on tokens split
    # across unrelated results (verify-pass finding).
    hit_group = next(
        (g for g in groups if any(group_contained(text, g) for text in result_texts)),
        None,
    )
    machinery = [i for i, t in enumerate(result_texts) if is_machinery_text(t)]
    return {
        "id": question.get("id"),
        "q": question.get("q"),
        "hit": hit_group is not None,
        "hit_group": hit_group,
        "context_hit": any(group_contained(context_blob, g) for g in groups),
        "result_count": len(result_texts),
        "machinery_top3_indexes": machinery,
    }


def _summarize(
    per_question: list[dict[str, Any]],
    *,
    mode: str,
    skipped: list[str] | None = None,
) -> dict[str, Any]:
    hits = sum(1 for r in per_question if r["hit"])
    machinery_total = sum(len(r["machinery_top3_indexes"]) for r in per_question)
    return {
        "mode": mode,
        "score": hits,
        "total": len(per_question),
        "machinery_in_top3": machinery_total,
        "machinery_clean": machinery_total == 0,
        "machinery_predicate": machinery_predicate_source(),
        "skipped_live_only": skipped or [],
        "questions": per_question,
    }


def run_battery_against_live(
    *,
    server_url: str = "http://127.0.0.1:8100",
    battery_path: Path | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run every battery question through the live recall + context surfaces."""
    battery = load_battery(battery_path)
    base = server_url.rstrip("/")

    def _get(path: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{base}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    started = time.perf_counter()
    per_question: list[dict[str, Any]] = []
    for question in battery["questions"]:
        q = urllib.parse.quote(str(question["q"]))
        try:
            recall_payload = _get(f"/api/knowledge/recall?q={q}&limit=3")
        except Exception as exc:
            recall_payload = {"error": str(exc), "items": []}
        try:
            context_payload = _get(f"/api/knowledge/context?max_tokens=1500&topic_hint={q}")
        except Exception as exc:
            context_payload = {"error": str(exc), "context": ""}
        context_parts: list[str] = []
        _flatten_text(context_payload, context_parts)
        row = score_question(question, top3_result_texts(recall_payload), "\n".join(context_parts))
        if recall_payload.get("error"):
            row["recall_error"] = recall_payload["error"]
        per_question.append(row)

    result = _summarize(per_question, mode="live")
    result["server_url"] = base
    result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def _seed_episode_for(question: dict[str, Any]) -> tuple[str, str, str]:
    """(entity_name, summary, content) planted for a battery question."""
    qid = str(question.get("id"))
    groups = question.get("expected_tokens") or []
    tokens = " ".join(t for g in groups for t in g)
    name = f"Battery fact {qid}"
    summary = f"Answer to '{question.get('q')}': {tokens}."
    content = f"{question.get('q')}? {summary}"
    return name, summary, content


async def run_battery_seeded(
    *,
    battery_path: Path | None = None,
    group_id: str = "battery_seeded",
) -> dict[str, Any]:
    """CI mode: lite fixture brain seeded from the battery's own questions."""
    from engram.config import EngramConfig
    from engram.extraction.factory import create_extractor
    from engram.graph_manager import GraphManager
    from engram.retrieval.recall_surface import build_api_recall_surface
    from engram.storage.bootstrap import open_local_stores
    from engram.storage.resolver import EngineMode

    battery = load_battery(battery_path)
    started = time.perf_counter()
    skipped = [str(q.get("id")) for q in battery["questions"] if q.get("live_only")]
    runnable = [q for q in battery["questions"] if not q.get("live_only")]

    with tempfile.TemporaryDirectory(prefix="engram-battery-") as tmp:
        config = EngramConfig(
            mode="lite",
            sqlite={"path": str(Path(tmp) / "battery.db")},
            embedding={"provider": "noop"},
            activation={
                "extraction_provider": "narrow",
                "worker_enabled": False,
                "cue_layer_enabled": True,
            },
            _env_file=None,
        )
        async with open_local_stores(config, mode=EngineMode.LITE) as stores:
            manager = GraphManager(
                graph_store=stores.graph_store,
                activation_store=stores.activation_store,
                search_index=stores.search_index,
                extractor=create_extractor(config),
                cfg=config.activation,
            )
            for question in runnable:
                name, summary, content = _seed_episode_for(question)
                await manager.ingest_episode(
                    content=content,
                    source="battery_seed",
                    group_id=group_id,
                    proposed_entities=[
                        {
                            "name": name,
                            "entity_type": "Fact",
                            "source_span": name,
                            "summary": summary,
                        }
                    ],
                    model_tier="sonnet",
                )
            per_question: list[dict[str, Any]] = []
            for question in runnable:
                payload = await build_api_recall_surface(
                    manager,
                    group_id=group_id,
                    query=str(question["q"]),
                    limit=3,
                    project_path=None,
                    operation_source="api_recall",
                )
                per_question.append(score_question(question, top3_result_texts(payload)))

    result = _summarize(per_question, mode="seeded", skipped=skipped)
    result["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


def format_battery_report(result: dict[str, Any], *, floor: int | None = None) -> str:
    lines = [
        f"# Agent-experience battery ({result.get('mode')}): "
        f"{result.get('score')}/{result.get('total')}",
        "",
        f"- machinery in top-3 (gate B4): {result.get('machinery_in_top3')} "
        f"({'clean' if result.get('machinery_clean') else 'VIOLATION'}; "
        f"predicate: {result.get('machinery_predicate')})",
        f"- duration: {result.get('duration_ms')} ms",
    ]
    if floor is not None:
        lines.append(f"- floor: {floor} ({'PASS' if result.get('score', 0) >= floor else 'FAIL'})")
    if result.get("skipped_live_only"):
        lines.append(f"- skipped (live-only): {', '.join(result['skipped_live_only'])}")
    lines.append("")
    for row in result.get("questions") or []:
        mark = "HIT " if row.get("hit") else "MISS"
        extra = ""
        if row.get("context_hit") and not row.get("hit"):
            extra = " (context-only hit)"
        if row.get("machinery_top3_indexes"):
            extra += f" [machinery@{row['machinery_top3_indexes']}]"
        if row.get("recall_error"):
            extra += f" [recall error: {row['recall_error']}]"
        lines.append(f"- {mark} {row.get('id')}: {row.get('q')}{extra}")
    return "\n".join(lines) + "\n"
