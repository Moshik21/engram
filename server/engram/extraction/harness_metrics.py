"""Process-local harness-as-extractor scoreboard counters.

Product metric: client proposals are the primary meaning path. External LLM
extractors must not be the default. These counters are intentionally local
(in-memory) so unit tests and doctor can read them without a second DB.

Persist later via evaluation samples if needed; process counters are enough
for CI and operator spot-checks.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any


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


def reset_harness_metrics() -> None:
    """Reset process counters (tests only)."""
    with _COUNTERS._lock:
        for name in (
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
        ):
            setattr(_COUNTERS, name, 0)


def record_remember_call(*, has_proposals: bool) -> None:
    with _COUNTERS._lock:
        if has_proposals:
            _COUNTERS.remember_with_proposals += 1
        else:
            _COUNTERS.remember_without_proposals += 1


def record_external_extractor_skipped() -> None:
    with _COUNTERS._lock:
        _COUNTERS.external_extractor_skipped += 1


def record_external_extractor_invoked() -> None:
    with _COUNTERS._lock:
        _COUNTERS.external_extractor_invoked += 1


def record_narrow_extraction() -> None:
    with _COUNTERS._lock:
        _COUNTERS.narrow_extractions += 1


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


def get_harness_metrics() -> HarnessMetricsSnapshot:
    with _COUNTERS._lock:
        return HarnessMetricsSnapshot(
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


def harness_scoreboard_payload() -> dict[str, Any]:
    """Operator-facing scoreboard JSON (client_proposal share + promote rate)."""
    snap = get_harness_metrics()
    return {
        "status": "ok",
        "scoreboard": "harness_extractor",
        "metrics": snap.to_dict(),
        "north_star": "cold Decision hit rate (engram continuity --smoke)",
        "notes": [
            "client_proposal_share = proposal commits / (proposal commits + narrow extractions)",
            "promote_rate = remember_with_proposals / all remember calls",
            "external_extractor_skipped should rise with good harness habit",
        ],
    }
