"""Harness-as-extractor scoreboard counters (process + durable file).

Product metric: client proposals are the primary meaning path. External LLM
extractors must not be the default.

Process counters support unit tests; a JSON file under the Engram home dir
lets operators read promote_rate / client_proposal_share from CLI after
remembers ran in the MCP/server process.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_METRIC_KEYS = (
    "remember_with_proposals",
    "remember_without_proposals",
    "client_proposal_commits",
    "client_proposal_defers",
    "client_proposal_rejects",
    "span_unverified_defers",
    "predicate_not_allowed_rejects",
    "identity_core_conflicts",
    "narrow_extractions",
    "external_extractor_skipped",
    "external_extractor_invoked",
)


@dataclass
class HarnessMetricsSnapshot:
    """Point-in-time harness extraction scoreboard."""

    remember_with_proposals: int = 0
    remember_without_proposals: int = 0
    client_proposal_commits: int = 0
    client_proposal_defers: int = 0
    client_proposal_rejects: int = 0
    span_unverified_defers: int = 0
    predicate_not_allowed_rejects: int = 0
    identity_core_conflicts: int = 0
    narrow_extractions: int = 0
    external_extractor_skipped: int = 0  # proposals present → no API extract
    external_extractor_invoked: int = 0  # observe/no-proposal path only

    def client_proposal_share(self) -> float | None:
        """Fraction of evidence commits that came from client proposals."""
        total = self.client_proposal_commits + self.narrow_extractions
        if total <= 0:
            return None
        return self.client_proposal_commits / total

    def promote_rate(self) -> float | None:
        """Fraction of remember calls that carried proposals."""
        total = self.remember_with_proposals + self.remember_without_proposals
        if total <= 0:
            return None
        return self.remember_with_proposals / total

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["client_proposal_share"] = self.client_proposal_share()
        data["promote_rate"] = self.promote_rate()
        return data


@dataclass
class _Counters:
    remember_with_proposals: int = 0
    remember_without_proposals: int = 0
    client_proposal_commits: int = 0
    client_proposal_defers: int = 0
    client_proposal_rejects: int = 0
    span_unverified_defers: int = 0
    predicate_not_allowed_rejects: int = 0
    identity_core_conflicts: int = 0
    narrow_extractions: int = 0
    external_extractor_skipped: int = 0
    external_extractor_invoked: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_COUNTERS = _Counters()


def harness_metrics_path() -> Path:
    """Durable scoreboard path (server + CLI share this file)."""
    override = os.environ.get("ENGRAM_HARNESS_METRICS_PATH")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("ENGRAM_HOME") or str(Path.home() / ".engram")
    return Path(home).expanduser() / "harness-metrics.json"


def _load_persistent() -> dict[str, int]:
    path = harness_metrics_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {k: 0 for k in _METRIC_KEYS}
    if not isinstance(raw, dict):
        return {k: 0 for k in _METRIC_KEYS}
    return {k: int(raw.get(k, 0) or 0) for k in _METRIC_KEYS}


def _persist_add(deltas: dict[str, int]) -> None:
    """Add deltas into the durable JSON scoreboard (best-effort)."""
    if not any(deltas.values()):
        return
    path = harness_metrics_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        current = _load_persistent()
        for key, delta in deltas.items():
            if key in current and delta:
                current[key] = max(0, current[key] + int(delta))
        path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    except OSError:
        return


def reset_harness_metrics(*, clear_persistent: bool = False) -> None:
    """Reset process counters (tests only). Optionally wipe durable file."""
    with _COUNTERS._lock:
        for name in _METRIC_KEYS:
            setattr(_COUNTERS, name, 0)
    if clear_persistent:
        path = harness_metrics_path()
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def record_remember_call(*, has_proposals: bool) -> None:
    with _COUNTERS._lock:
        if has_proposals:
            _COUNTERS.remember_with_proposals += 1
            _persist_add({"remember_with_proposals": 1})
        else:
            _COUNTERS.remember_without_proposals += 1
            _persist_add({"remember_without_proposals": 1})


def record_external_extractor_skipped() -> None:
    with _COUNTERS._lock:
        _COUNTERS.external_extractor_skipped += 1
        _persist_add({"external_extractor_skipped": 1})


def record_external_extractor_invoked() -> None:
    with _COUNTERS._lock:
        _COUNTERS.external_extractor_invoked += 1
        _persist_add({"external_extractor_invoked": 1})


def record_narrow_extraction() -> None:
    with _COUNTERS._lock:
        _COUNTERS.narrow_extractions += 1
        _persist_add({"narrow_extractions": 1})


def record_client_proposal_outcomes(
    *,
    commits: int = 0,
    defers: int = 0,
    rejects: int = 0,
    span_unverified_defers: int = 0,
    predicate_rejects: int = 0,
    identity_conflicts: int = 0,
) -> None:
    with _COUNTERS._lock:
        _COUNTERS.client_proposal_commits += commits
        _COUNTERS.client_proposal_defers += defers
        _COUNTERS.client_proposal_rejects += rejects
        _COUNTERS.span_unverified_defers += span_unverified_defers
        _COUNTERS.predicate_not_allowed_rejects += predicate_rejects
        _COUNTERS.identity_core_conflicts += identity_conflicts
        _persist_add(
            {
                "client_proposal_commits": commits,
                "client_proposal_defers": defers,
                "client_proposal_rejects": rejects,
                "span_unverified_defers": span_unverified_defers,
                "predicate_not_allowed_rejects": predicate_rejects,
                "identity_core_conflicts": identity_conflicts,
            }
        )


def get_harness_metrics(*, prefer_persistent: bool = True) -> HarnessMetricsSnapshot:
    """Return process snapshot, or durable file when process counters are empty."""
    with _COUNTERS._lock:
        process = HarnessMetricsSnapshot(
            remember_with_proposals=_COUNTERS.remember_with_proposals,
            remember_without_proposals=_COUNTERS.remember_without_proposals,
            client_proposal_commits=_COUNTERS.client_proposal_commits,
            client_proposal_defers=_COUNTERS.client_proposal_defers,
            client_proposal_rejects=_COUNTERS.client_proposal_rejects,
            span_unverified_defers=_COUNTERS.span_unverified_defers,
            predicate_not_allowed_rejects=_COUNTERS.predicate_not_allowed_rejects,
            identity_core_conflicts=_COUNTERS.identity_core_conflicts,
            narrow_extractions=_COUNTERS.narrow_extractions,
            external_extractor_skipped=_COUNTERS.external_extractor_skipped,
            external_extractor_invoked=_COUNTERS.external_extractor_invoked,
        )
    process_total = process.remember_with_proposals + process.remember_without_proposals
    if not prefer_persistent or process_total > 0 or process.client_proposal_commits > 0:
        return process
    persistent = _load_persistent()
    return HarnessMetricsSnapshot(**persistent)


def harness_scoreboard_payload() -> dict[str, Any]:
    """Operator-facing scoreboard JSON (client_proposal share + promote rate)."""
    snap = get_harness_metrics(prefer_persistent=True)
    return {
        "status": "ok",
        "scoreboard": "harness_extractor",
        "metrics": snap.to_dict(),
        "path": str(harness_metrics_path()),
        "north_star": "cold Decision hit rate (engram continuity --against-live)",
        "notes": [
            "client_proposal_share = proposal commits / (proposal commits + narrow extractions)",
            "promote_rate = remember_with_proposals / all remember calls",
            "external_extractor_skipped should rise with good harness habit",
            "metrics persist under ~/.engram/harness-metrics.json for cross-process reads",
        ],
    }
