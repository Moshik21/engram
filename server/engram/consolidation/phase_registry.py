"""Shared consolidation phase registry."""

from __future__ import annotations

from collections.abc import Iterable

CONSOLIDATION_PHASE_ORDER: tuple[str, ...] = (
    "triage",
    "merge",
    "calibrate",
    "infer",
    "evidence_adjudication",
    "edge_adjudication",
    "replay",
    "prune",
    "compact",
    "reflect",
    "reindex",
    "graph_embed",
    "microglia",
    "immunity",
    "dream",
)

CONSOLIDATION_PHASE_TIERS: dict[str, str] = {
    "triage": "hot",
    "merge": "warm",
    "calibrate": "warm",
    "infer": "warm",
    "evidence_adjudication": "warm",
    "edge_adjudication": "warm",
    "compact": "warm",
    "reindex": "warm",
    "microglia": "warm",
    "immunity": "cold",
    "replay": "cold",
    "prune": "cold",
    "reflect": "cold",
    "graph_embed": "cold",
    "dream": "cold",
}

_known = set(CONSOLIDATION_PHASE_ORDER)
_tiered = set(CONSOLIDATION_PHASE_TIERS)
if _known != _tiered:
    missing = ", ".join(sorted(_known - _tiered)) or "none"
    extra = ", ".join(sorted(_tiered - _known)) or "none"
    raise RuntimeError(f"Consolidation phase tier registry drift: missing={missing}; extra={extra}")


def validate_consolidation_phase_order(phase_names: Iterable[str]) -> tuple[str, ...]:
    """Validate runtime phase construction against the shared phase registry."""
    names = tuple(phase_names)
    if names == CONSOLIDATION_PHASE_ORDER:
        return names

    missing = ", ".join(name for name in CONSOLIDATION_PHASE_ORDER if name not in names)
    extra = ", ".join(name for name in names if name not in CONSOLIDATION_PHASE_ORDER)
    raise RuntimeError(
        "Consolidation phase order drift: "
        f"expected={list(CONSOLIDATION_PHASE_ORDER)}; actual={list(names)}; "
        f"missing={missing or 'none'}; extra={extra or 'none'}"
    )
