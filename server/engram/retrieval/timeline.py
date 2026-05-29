"""Dated timeline read surface (temporal PR1).

Fans out to ``recall()`` (episodes / cue episodes) and ``search_facts()``
(current-valid relationships), resolves a single date per item, and returns a
chronologically-ordered, dated timeline so a consuming LLM can answer temporal
questions ("which came first", "how many days between X and Y", "what is the
current value of X") trivially.

This adds NO embedded LLM. A deterministic planner answers the easy temporal
sub-language with pure date arithmetic; anything harder is left to the agent,
which now gets the evidence pre-ordered and dated.

Honesty note: ``conversation_date`` is optional and frequently absent, and many
relationships have no ``valid_from``. Each row therefore carries an explicit
``date_basis`` ("conversation_date" | "valid_from" | "created_at") so the
consumer never mistakes an ingestion timestamp for a real event date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


class _TimelineManager(Protocol):
    async def recall(self, query: str, group_id: str = ..., limit: int = ...) -> list[dict]: ...

    async def search_facts(
        self,
        group_id: str = ...,
        query: str = ...,
        include_expired: bool = ...,
        limit: int = ...,
    ) -> list[dict]: ...


@dataclass
class TimelineRow:
    """One dated entry in the timeline."""

    date: datetime  # normalized to naive UTC for stable ordering
    date_iso: str  # YYYY-MM-DD
    date_basis: str  # conversation_date | valid_from | created_at
    kind: str  # episode | cue | fact
    label: str
    source_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date_iso,
            "date_basis": self.date_basis,
            "kind": self.kind,
            "label": self.label,
            "source_id": self.source_id,
        }


def _to_naive_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 string (or datetime) into a naive-UTC datetime.

    Mixed tz-aware / naive inputs would otherwise make ascending sort raise, so
    everything is normalized to naive UTC before comparison.
    """
    if value is None or value == "":
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            # Accept date-only / YYYY-MM by padding.
            for suffix in ("-01", "-01-01"):
                try:
                    dt = datetime.fromisoformat(text + suffix)
                    break
                except ValueError:
                    continue
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(tz=None).replace(tzinfo=None)
    return dt


def _truncate(text: str, limit: int = 200) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _episode_row(result: dict) -> TimelineRow | None:
    ep = result.get("episode") if isinstance(result.get("episode"), dict) else {}
    conv = _to_naive_utc(ep.get("conversation_date"))
    created = _to_naive_utc(ep.get("created_at"))
    if conv is not None:
        date, basis = conv, "conversation_date"
    elif created is not None:
        date, basis = created, "created_at"
    else:
        return None
    rtype = result.get("result_type")
    if rtype == "cue_episode":
        cue = result.get("cue") if isinstance(result.get("cue"), dict) else {}
        label = _truncate(cue.get("cue_text") or ep.get("content", ""))
        kind = "cue"
    else:
        label = _truncate(ep.get("content", ""))
        kind = "episode"
    return TimelineRow(
        date=date,
        date_iso=date.strftime("%Y-%m-%d"),
        date_basis=basis,
        kind=kind,
        label=label,
        source_id=ep.get("id"),
    )


def _fact_row(fact: dict) -> TimelineRow | None:
    valid_from = _to_naive_utc(fact.get("valid_from"))
    created = _to_naive_utc(fact.get("created_at"))
    if valid_from is not None:
        date, basis = valid_from, "valid_from"
    elif created is not None:
        date, basis = created, "created_at"
    else:
        return None
    subject = fact.get("subject") or "?"
    predicate = (fact.get("predicate") or "").replace("_", " ").lower()
    obj = fact.get("object") or "?"
    return TimelineRow(
        date=date,
        date_iso=date.strftime("%Y-%m-%d"),
        date_basis=basis,
        kind="fact",
        label=f"{subject} {predicate} {obj}".strip(),
        source_id=fact.get("source_episode"),
    )


def build_timeline_rows(
    recall_results: list[dict],
    facts: list[dict],
) -> list[TimelineRow]:
    """Pure assembly: map recall results + facts to dated rows, sorted ascending.

    Rows without any resolvable date are dropped (a timeline entry with no date
    is meaningless and would corrupt ordering).
    """
    rows: list[TimelineRow] = []
    for result in recall_results or []:
        if not isinstance(result, dict):
            continue
        if result.get("result_type") in ("episode", "cue_episode"):
            row = _episode_row(result)
            if row is not None:
                rows.append(row)
    for fact in facts or []:
        if isinstance(fact, dict):
            row = _fact_row(fact)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda r: r.date)
    return rows


def plan_temporal(rows: list[TimelineRow]) -> dict[str, Any]:
    """Deterministic answers to the easy temporal sub-language (no LLM).

    - first / last: the earliest / latest dated row
    - span_days: whole days between the first and last row
    - current_values: latest row per fact (subject, predicate)
    """
    if not rows:
        return {"first": None, "last": None, "span_days": None, "current_values": []}

    first, last = rows[0], rows[-1]
    span_days = (last.date - first.date).days

    latest_by_key: dict[tuple[str, str], TimelineRow] = {}
    for row in rows:
        if row.kind != "fact":
            continue
        # label is "subject predicate object"; key on subject+predicate so a
        # later assertion supersedes an earlier one (the "current" value).
        parts = row.label.split(" ", 2)
        key = (parts[0], parts[1] if len(parts) > 1 else "")
        latest_by_key[key] = row  # rows are ascending, so last write wins

    return {
        "first": first.to_dict(),
        "last": last.to_dict(),
        "span_days": span_days,
        "current_values": [r.to_dict() for r in latest_by_key.values()],
    }


def render_timeline_markdown(rows: list[TimelineRow]) -> str:
    if not rows:
        return "(no dated memory found for this query)"
    lines = []
    for r in rows:
        flag = "" if r.date_basis in ("conversation_date", "valid_from") else " ~ingested"
        lines.append(f"{r.date_iso} — {r.label}{flag}")
    return "\n".join(lines)


async def build_mcp_timeline_tool_surface(
    manager: _TimelineManager,
    *,
    group_id: str,
    query: str,
    limit: int = 20,
) -> dict[str, Any]:
    """MCP-surface wrapper for the timeline tool (keeps the route handler behind
    a shared build_* surface per the public-surface boundary contract)."""
    return await build_timeline_surface(manager, query, group_id=group_id, limit=limit)


async def build_timeline_surface(
    manager: _TimelineManager,
    query: str,
    group_id: str = "default",
    limit: int = 20,
) -> dict[str, Any]:
    """Assemble the dated timeline for a query.

    Returns a structured payload: ordered ``rows`` (each with ``date_basis``),
    a ``markdown`` rendering, and a deterministic ``planner`` block. Any row
    whose date is only ``created_at``/ingestion-time is flagged via
    ``date_basis`` so the consuming LLM does not over-trust it.
    """
    recall_results = await manager.recall(query, group_id=group_id, limit=limit)
    facts = await manager.search_facts(
        group_id=group_id,
        query=query,
        include_expired=False,
        limit=limit,
    )
    rows = build_timeline_rows(recall_results, facts)
    event_time_rows = sum(
        1 for r in rows if r.date_basis in ("conversation_date", "valid_from")
    )
    return {
        "operation": "timeline",
        "query": query,
        "row_count": len(rows),
        "event_time_row_count": event_time_rows,
        "rows": [r.to_dict() for r in rows],
        "markdown": render_timeline_markdown(rows),
        "planner": plan_temporal(rows),
    }
