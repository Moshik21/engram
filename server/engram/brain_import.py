"""Re-seed an exported brain through the LIVE capture path (2026-09-04).

Reads ``episodes.jsonl`` from ``engram backup export`` and posts each chosen
row to the running shell's auto-observe route, so every row goes through the
same capture-time cue + vector indexing as a fresh capture. Content is posted
verbatim (the ``[role|project]`` header it already carries is kept), with the
original ``source``, ``session_id`` and ``created_at`` as ``conversation_date``.

Only the classes named in ``--classes`` are posted (default: conversation).
Bootstrap snapshots, machinery, session markers and probes stay behind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

DEFAULT_CLASSES = ("conversation",)


def row_to_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    content = (row.get("content") or "").strip()
    if len(content) < 10:
        return None
    header = content[:80]
    role = "assistant" if header.startswith("[assistant|") else "user"
    payload: dict[str, Any] = {
        "content": content,
        "role": role,
        "source": row.get("source") or "import",
    }
    if row.get("session_id"):
        payload["session_id"] = row["session_id"]
    if row.get("created_at"):
        payload["conversation_date"] = row["created_at"]
    project = row.get("project")
    if not project and header.startswith("["):
        parts = header[1:].split("]", 1)[0].split("|")
        if len(parts) >= 2 and parts[1].strip():
            project = parts[1].strip()
    if project:
        payload["project"] = project
    return payload


async def import_episodes(
    export_dir: Path,
    post: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    *,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    limit: int | None = None,
    rate_per_s: float = 5.0,
) -> dict[str, Any]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028/U+000B and the
    # like, which real conversation content contains (first live run died on it).
    with (export_dir / "episodes.jsonl").open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    chosen = [r for r in rows if r.get("classification") in classes]
    if limit is not None:
        chosen = chosen[:limit]
    posted = skipped = failed = 0
    statuses: dict[str, int] = {}
    interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
    for row in chosen:
        payload = row_to_payload(row)
        if payload is None:
            skipped += 1
            continue
        try:
            resp = await post(payload)
        except Exception:
            failed += 1
            continue
        status = str(resp.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        posted += 1
        if interval:
            await asyncio.sleep(interval)
    return {
        "rows_in_export": len(rows),
        "chosen": len(chosen),
        "posted": posted,
        "skipped_short": skipped,
        "failed": failed,
        "statuses": statuses,
        "classes": list(classes),
    }


async def _http_post_factory(url: str):
    import httpx

    client = httpx.AsyncClient(timeout=60.0)

    async def post(payload: dict[str, Any]) -> dict[str, Any]:
        r = await client.post(f"{url.rstrip('/')}/api/knowledge/auto-observe", json=payload)
        r.raise_for_status()
        return r.json()

    return post, client


def build_parser(sub: Any) -> None:
    p = sub.add_parser("import", help="Re-seed an export through the live shell's capture path")
    p.add_argument("--from", dest="export_dir", required=True, help="Export directory")
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument(
        "--classes", default=",".join(DEFAULT_CLASSES), help="Comma-separated classes to post"
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--rate", type=float, default=5.0, help="Posts per second")


async def run_import(args: argparse.Namespace) -> int:
    post, client = await _http_post_factory(args.url)
    started = time.monotonic()
    try:
        report = await import_episodes(
            Path(args.export_dir),
            post,
            classes=tuple(c.strip() for c in args.classes.split(",") if c.strip()),
            limit=args.limit,
            rate_per_s=args.rate,
        )
    finally:
        await client.aclose()
    report["seconds"] = round(time.monotonic() - started, 1)
    print(json.dumps(report, indent=1))
    return 0 if report["failed"] == 0 else 1
