"""Operational metrics: shell availability and brain-window cost.

The north-star gate historically only ran "when the shell was up", making it
blind to the biggest live product failure — availability (an overnight brain
window once kept the shell down 10h44m and the gate never noticed).
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Timestamped markers in ~/.engram/logs/engram.log. Uvicorn's own
# "INFO: Shutting down" has no timestamp; the MCP session manager line and
# engram.main startup line do.
_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+"
_SHUTDOWN_RE = re.compile(_TS + r" .*StreamableHTTP session manager shutting down")
_STARTED_RE = re.compile(_TS + r" .*engram\.main: Engram v[\w.]+ started")


def default_shell_log_path() -> Path:
    home = Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()
    return home / "logs" / "engram.log"


@dataclass
class AvailabilityReport:
    window_hours: float
    availability_pct: float | None
    downtime_seconds: float
    outage_count: int
    max_outage_seconds: float
    open_outage: bool  # shell currently down (last marker was a shutdown)
    outages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_ts(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def compute_shell_availability(
    log_text: str | None = None,
    *,
    log_path: Path | None = None,
    window_hours: float = 24.0,
    now: datetime | None = None,
) -> AvailabilityReport:
    """Reconstruct shell downtime windows from the engram log.

    Pairs each timestamped shutdown marker with the next startup marker;
    unpaired trailing shutdown counts as an open outage up to *now*.
    """
    if log_text is None:
        path = log_path or default_shell_log_path()
        try:
            # Tail-read: logs grow to hundreds of MB; 4 MB covers days.
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - 4 * 1024 * 1024))
                log_text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            return AvailabilityReport(
                window_hours=window_hours,
                availability_pct=None,
                downtime_seconds=0.0,
                outage_count=0,
                max_outage_seconds=0.0,
                open_outage=False,
            )

    events: list[tuple[datetime, str]] = []
    for match in _SHUTDOWN_RE.finditer(log_text):
        events.append((_parse_ts(match.group(1)), "down"))
    for match in _STARTED_RE.finditer(log_text):
        events.append((_parse_ts(match.group(1)), "up"))
    events.sort(key=lambda pair: pair[0])

    if now is None:
        now = datetime.now()
    window_start = now - timedelta(hours=window_hours)

    outages: list[dict[str, Any]] = []
    down_since: datetime | None = None
    for ts, kind in events:
        if kind == "down":
            if down_since is None:
                down_since = ts
        else:
            if down_since is not None:
                outages.append({"down_at": down_since, "up_at": ts})
                down_since = None
    open_outage = down_since is not None
    if down_since is not None:
        outages.append({"down_at": down_since, "up_at": None})

    downtime = 0.0
    max_outage = 0.0
    counted = 0
    trimmed: list[dict[str, Any]] = []
    for outage in outages:
        start = outage["down_at"]
        end = outage["up_at"] or now
        clipped_start = max(start, window_start)
        clipped_end = min(end, now)
        seconds = (clipped_end - clipped_start).total_seconds()
        if seconds <= 0:
            continue
        counted += 1
        downtime += seconds
        max_outage = max(max_outage, seconds)
        trimmed.append(
            {
                "down_at": start.isoformat(sep=" "),
                "up_at": outage["up_at"].isoformat(sep=" ") if outage["up_at"] else None,
                "seconds": round(seconds, 1),
            }
        )

    total = window_hours * 3600.0
    pct = None
    if events:
        pct = round(max(0.0, 100.0 * (1.0 - downtime / total)), 3)
    return AvailabilityReport(
        window_hours=window_hours,
        availability_pct=pct,
        downtime_seconds=round(downtime, 1),
        outage_count=counted,
        max_outage_seconds=round(max_outage, 1),
        open_outage=open_outage,
        outages=trimmed,
    )


def brain_status_anomalies(
    status: dict[str, Any] | None, *, now: datetime | None = None
) -> list[str]:
    """Human-readable anomalies from the last brain-status.json record."""
    if not status:
        return ["no brain runs recorded (brain-status.json missing)"]
    anomalies: list[str] = []
    if status.get("ok") is False:
        anomalies.append(f"last brain run failed: {status.get('error')}")
    if status.get("system_slept"):
        anomalies.append(
            "system slept during the last brain run "
            f"(wall {status.get('duration_s')}s vs monotonic "
            f"{status.get('duration_monotonic_s')}s)"
        )
    finished = status.get("finished_at")
    if finished:
        try:
            from datetime import timezone

            finished_dt = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
            ref = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            age_hours = (ref - finished_dt).total_seconds() / 3600.0
            if age_hours > 26.0:
                anomalies.append(
                    f"last brain run finished {age_hours:.0f}h ago — the "
                    "LaunchAgent (2h cadence) may not be firing"
                )
        except ValueError:
            pass
    duration = status.get("duration_monotonic_s") or status.get("duration_s")
    if isinstance(duration, (int, float)) and duration > 1800:
        anomalies.append(f"last brain run took {duration:.0f}s (deadline is 1800s)")
    return anomalies
