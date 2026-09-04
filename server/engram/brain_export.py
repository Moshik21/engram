"""Bulk export of a native brain to JSONL (2026-09-04, the fresh-brain path).

Everything the store holds that cannot be regenerated leaves through here:
episodes with FULL content (the REST listing caps at 200 chars), entities,
relationships, cues with their feedback counters, the identity-core set, and
the operator sidecars. Vectors are not exported (regenerable by re-indexing).

Each episode row carries a ``classification`` so a re-seed can choose:
``conversation`` (irreplaceable), ``bootstrap`` (project-bootstrap snapshot,
regenerable), ``machinery`` (harness frames), ``session_marker``, ``probe``.

Exclusive access is required: the shell must be down (LMDB must never be
opened by two processes). ``--force-local`` skips that check for scratch dirs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BOOTSTRAP_RE = re.compile(r"^\s*\[project-bootstrap\|")
_PROBE_SOURCE_RE = re.compile(r"probe|soak|gate|latency|step1|projfield|freshtest", re.I)
_SESSION_SOURCE_RE = re.compile(r"^(auto:session|claude:precompact|auto:compact|session)", re.I)


@dataclass
class ExportReport:
    out_dir: str
    data_dir: str
    episodes: int = 0
    by_classification: dict[str, int] = field(default_factory=dict)
    entities: int = 0
    relationships: int = 0
    cues: int = 0
    identity_core: int = 0
    sidecars: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "data_dir": self.data_dir,
            "episodes": self.episodes,
            "by_classification": dict(self.by_classification),
            "entities": self.entities,
            "relationships": self.relationships,
            "cues": self.cues,
            "identity_core": self.identity_core,
            "sidecars": list(self.sidecars),
            "errors": list(self.errors),
        }


def classify_episode(content: str | None, source: str | None) -> str:
    """Irreplaceable vs regenerable, by the same rules the assessment used."""
    from engram.ingestion.salience import is_machinery

    text = content or ""
    src = source or ""
    if _BOOTSTRAP_RE.match(text) or src == "auto:bootstrap":
        return "bootstrap"
    if _SESSION_SOURCE_RE.match(src):
        return "session_marker"
    if _PROBE_SOURCE_RE.search(src):
        return "probe"
    if is_machinery(text, src):
        return "machinery"
    return "conversation"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    return len(rows)


async def export_brain(
    graph_store: Any,
    out_dir: Path,
    *,
    group_id: str = "default",
    data_dir: str = "",
    engram_home: Path | None = None,
) -> ExportReport:
    """Write episodes/entities/relationships/cues/identity_core JSONL under *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ExportReport(out_dir=str(out_dir), data_dir=data_dir)

    # Episodes: raw rows carry the full content (the REST listing truncates).
    rows = await graph_store._query("find_episodes_by_group", {"gid": group_id})
    episodes: list[dict[str, Any]] = []
    for r in rows or []:
        row = dict(r)
        row["classification"] = classify_episode(row.get("content"), row.get("source"))
        episodes.append(row)
        report.by_classification[row["classification"]] = (
            report.by_classification.get(row["classification"], 0) + 1
        )
    episodes.sort(key=lambda r: str(r.get("created_at") or ""))
    report.episodes = _write_jsonl(out_dir / "episodes.jsonl", episodes)

    # Entities (raw rows, all fields including attributes_json / identity_core).
    ent_rows = await graph_store._query("find_entities_by_group", {"gid": group_id})
    entities = [dict(r) for r in ent_rows or []]
    report.entities = _write_jsonl(out_dir / "entities.jsonl", entities)

    # Relationships: no group-wide listing exists natively; walk outgoing edges
    # per entity and dedupe by relationship id.
    seen: set[str] = set()
    rels: list[dict[str, Any]] = []
    for ent in entities:
        hid = ent.get("id")
        if hid is None:
            continue
        try:
            edges = await graph_store._query("get_outgoing_edges", {"id": hid})
        except Exception as exc:  # keep going; record the miss
            report.errors.append(f"edges for {ent.get('entity_id')}: {exc}")
            continue
        for e in edges or []:
            row = dict(e)
            key = str(
                row.get("rel_id") or row.get("id") or json.dumps(row, sort_keys=True, default=str)
            )
            if key in seen:
                continue
            seen.add(key)
            row["_source_entity_id"] = ent.get("entity_id")
            rels.append(row)
    report.relationships = _write_jsonl(out_dir / "relationships.jsonl", rels)

    # Cues with their feedback counters.
    cue_rows = await graph_store._query("find_cues_by_group", {"gid": group_id})
    report.cues = _write_jsonl(out_dir / "cues.jsonl", [dict(r) for r in cue_rows or []])

    # Identity core (the set no REST route exposes).
    try:
        core = await graph_store.get_identity_core_entities(group_id)
        core_rows = [
            {"entity_id": getattr(e, "id", None), "name": getattr(e, "name", None),
             "entity_type": getattr(e, "entity_type", None)}
            for e in core or []
        ]
    except Exception as exc:
        report.errors.append(f"identity_core: {exc}")
        core_rows = []
    report.identity_core = _write_jsonl(out_dir / "identity_core.jsonl", core_rows)

    # Operator sidecars, copied verbatim.
    home = engram_home or (Path.home() / ".engram")
    for name in ("activation-snapshot.json", "hygiene-state.json", "harness-metrics.json",
                 "brain-status.json", "bm25-breaker-state.json", "loop-adjustments.jsonl"):
        src = home / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            report.sidecars.append(name)

    (out_dir / "export-report.json").write_text(json.dumps(report.to_dict(), indent=1))
    return report


async def _run(args: argparse.Namespace) -> int:
    from engram.brain_cli import _shell_healthy, serve_process_alive
    from engram.config import EngramConfig, HelixDBConfig
    from engram.storage.helix.client import HelixClient
    from engram.storage.helix.graph import HelixGraphStore

    if not args.force_local and (_shell_healthy(args.port) or serve_process_alive()):
        print(
            "brain export: stop the shell first (engramctl stop); "
            "refusing to double-open the store"
        )
        return 2
    data_dir = args.data_dir
    if data_dir is None:
        from engram.storage.diagnostics import resolve_helix_native_data_dir

        data_dir = resolve_helix_native_data_dir(EngramConfig())
    cfg = HelixDBConfig(transport="native", data_dir=str(data_dir), max_workers=2)
    client = HelixClient(cfg)
    store = HelixGraphStore(cfg, client=client, owns_client=True)
    await store.initialize()
    try:
        report = await export_brain(
            store, Path(args.out), group_id=args.group_id, data_dir=str(data_dir),
            engram_home=Path(args.engram_home) if args.engram_home else None,
        )
    finally:
        await store.close()
    print(json.dumps(report.to_dict(), indent=1))
    return 0 if not report.errors else 1


def build_parser(sub: Any) -> None:
    p = sub.add_parser("export", help="Bulk-export the native brain to JSONL (shell must be down)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--data-dir", default=None, help="Native data dir (default: from config)")
    p.add_argument("--group-id", default="default")
    p.add_argument("--engram-home", default=None, help="Operator home to copy sidecars from")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument(
        "--force-local", action="store_true", help="Skip the shell-down check (scratch dirs only)"
    )


async def run_export(args: argparse.Namespace) -> int:
    return await _run(args)
