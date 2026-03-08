"""Report historical merge records that the current identifier policy would block."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

from engram.config import EngramConfig
from engram.entity_dedup_policy import dedup_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan consolidation merge history and report past merges that now look "
            "like blocked identifier/code mismatches."
        )
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the Engram SQLite sidecar database. Defaults to configured sqlite.path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of suspect merges to print.",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Optional tenant/group filter.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a plain-text table.",
    )
    return parser.parse_args()


def _default_db_path() -> str:
    return str(EngramConfig().get_sqlite_path())


def _format_timestamp(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _load_suspects(db_path: str, group_id: str | None, limit: int) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = """
            SELECT cycle_id, group_id, keep_name, remove_name, similarity,
                   decision_confidence, decision_source, decision_reason, timestamp
            FROM consolidation_merges
        """
        params: list[object] = []
        if group_id:
            sql += " WHERE group_id = ?"
            params.append(group_id)
        sql += " ORDER BY timestamp DESC"

        suspects: list[dict] = []
        for row in conn.execute(sql, params):
            decision = dedup_policy(row["keep_name"], row["remove_name"])
            if decision.allowed:
                continue
            suspects.append(
                {
                    "cycle_id": row["cycle_id"],
                    "group_id": row["group_id"],
                    "keep_name": row["keep_name"],
                    "remove_name": row["remove_name"],
                    "similarity": row["similarity"],
                    "decision_confidence": row["decision_confidence"],
                    "decision_source": row["decision_source"],
                    "decision_reason": row["decision_reason"],
                    "current_policy_reason": decision.reason,
                    "canonical_identifier_a": decision.left.canonical_code,
                    "canonical_identifier_b": decision.right.canonical_code,
                    "timestamp": row["timestamp"],
                }
            )
            if len(suspects) >= limit:
                break
        return suspects
    finally:
        conn.close()


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No suspect identifier merges found.")
        return

    print(
        "timestamp\tgroup_id\tcycle_id\tkeep_name\tremove_name\t"
        "current_policy_reason\tprevious_reason\tdecision_source"
    )
    for row in rows:
        print(
            "\t".join(
                [
                    _format_timestamp(row["timestamp"]),
                    row["group_id"] or "",
                    row["cycle_id"] or "",
                    row["keep_name"] or "",
                    row["remove_name"] or "",
                    row["current_policy_reason"] or "",
                    row["decision_reason"] or "",
                    row["decision_source"] or "",
                ]
            )
        )


def main() -> int:
    args = _parse_args()
    db_path = args.db or _default_db_path()
    rows = _load_suspects(db_path, args.group_id, args.limit)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
