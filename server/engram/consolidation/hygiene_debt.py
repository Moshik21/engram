"""Hygiene debt scoreboard — real sludge, not just episode pressure.

Surfaces deferred evidence, cue_only backlog, near-misses, open adjudication,
and orphan/low-value pressure so auto-mop triggers when the brain is actually
dirty — not only when new episodes arrive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class HygieneDebtSnapshot:
    """Point-in-time hygiene debt for a graph group."""

    deferred_evidence: int = 0
    pending_evidence: int = 0
    cue_only_episodes: int = 0
    cue_count: int = 0
    near_miss_count: int = 0
    open_adjudication: int = 0
    orphan_candidates: int = 0
    low_value_entity_candidates: int = 0
    episodes: int = 0
    entities: int = 0

    @property
    def open_work(self) -> int:
        return (
            max(0, self.deferred_evidence)
            + max(0, self.pending_evidence)
            + max(0, self.open_adjudication)
        )

    @property
    def total_debt_units(self) -> int:
        return (
            self.open_work
            + max(0, self.cue_only_episodes)
            + max(0, self.near_miss_count)
            + max(0, self.orphan_candidates)
            + max(0, self.low_value_entity_candidates)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["open_work"] = self.open_work
        payload["total_debt_units"] = self.total_debt_units
        return payload


def hygiene_debt_from_stats(stats: Mapping[str, Any] | None) -> HygieneDebtSnapshot:
    """Build a snapshot from graph ``get_stats`` / open-work style payloads."""
    if not isinstance(stats, Mapping):
        return HygieneDebtSnapshot()

    cue = stats.get("cue_metrics") if isinstance(stats.get("cue_metrics"), Mapping) else {}
    proj = (
        stats.get("projection_metrics")
        if isinstance(stats.get("projection_metrics"), Mapping)
        else {}
    )
    adj = (
        stats.get("adjudication_metrics")
        if isinstance(stats.get("adjudication_metrics"), Mapping)
        else {}
    )
    state_counts = proj.get("state_counts") if isinstance(proj.get("state_counts"), Mapping) else {}
    evidence_status = (
        adj.get("evidence_status_counts")
        if isinstance(adj.get("evidence_status_counts"), Mapping)
        else {}
    )
    request_status = (
        adj.get("request_status_counts")
        if isinstance(adj.get("request_status_counts"), Mapping)
        else {}
    )

    deferred = int(
        evidence_status.get("deferred")
        or adj.get("deferred_evidence_count")
        or stats.get("deferred_evidence")
        or 0
    )
    pending = int(
        evidence_status.get("pending")
        or adj.get("pending_evidence_count")
        or stats.get("pending_evidence")
        or 0
    )
    open_adj = int(
        adj.get("open_request_count")
        or request_status.get("pending")
        or stats.get("open_adjudication")
        or 0
    )
    cue_only = int(state_counts.get("cue_only") or stats.get("cue_only_episodes") or 0)
    near_miss = int(cue.get("cue_near_miss_count") or stats.get("near_miss_count") or 0)

    return HygieneDebtSnapshot(
        deferred_evidence=deferred,
        pending_evidence=pending,
        cue_only_episodes=cue_only,
        cue_count=int(cue.get("cue_count") or stats.get("cues") or 0),
        near_miss_count=near_miss,
        open_adjudication=open_adj,
        orphan_candidates=int(stats.get("orphan_candidates") or 0),
        low_value_entity_candidates=int(stats.get("low_value_entity_candidates") or 0),
        episodes=int(stats.get("episodes") or 0),
        entities=int(stats.get("entities") or 0),
    )


def debt_pressure_contribution(
    debt: HygieneDebtSnapshot,
    *,
    weight_deferred: float = 0.02,
    weight_cue_only: float = 0.01,
    weight_near_miss: float = 0.5,
    weight_open_adj: float = 0.05,
    weight_orphan: float = 0.1,
    weight_low_value: float = 0.05,
) -> float:
    """Convert hygiene debt into consolidation pressure units.

    Tuned so ~10k deferred evidence alone exceeds a typical threshold of 100
    (10_000 * 0.02 = 200) while a clean brain contributes ~0.
    """
    return (
        weight_deferred * max(0, debt.deferred_evidence)
        + weight_cue_only * max(0, debt.cue_only_episodes)
        + weight_near_miss * max(0, debt.near_miss_count)
        + weight_open_adj * max(0, debt.open_adjudication)
        + weight_orphan * max(0, debt.orphan_candidates)
        + weight_low_value * max(0, debt.low_value_entity_candidates)
    )


def debt_should_trigger_mop(
    debt: HygieneDebtSnapshot,
    *,
    deferred_threshold: int = 500,
    cue_only_threshold: int = 200,
    open_adj_threshold: int = 50,
    pressure_threshold: float = 100.0,
    cfg_weights: Mapping[str, float] | None = None,
) -> bool:
    """Return True when debt alone justifies a mop/warm consolidation cycle."""
    if debt.deferred_evidence >= deferred_threshold:
        return True
    if debt.cue_only_episodes >= cue_only_threshold:
        return True
    if debt.open_adjudication >= open_adj_threshold:
        return True
    weights = dict(cfg_weights or {})
    pressure = debt_pressure_contribution(
        debt,
        weight_deferred=float(weights.get("deferred", 0.02)),
        weight_cue_only=float(weights.get("cue_only", 0.01)),
        weight_near_miss=float(weights.get("near_miss", 0.5)),
        weight_open_adj=float(weights.get("open_adj", 0.05)),
        weight_orphan=float(weights.get("orphan", 0.1)),
        weight_low_value=float(weights.get("low_value", 0.05)),
    )
    return pressure >= pressure_threshold


@dataclass
class CueHygieneCandidate:
    episode_id: str
    hit_count: int = 0
    surfaced_count: int = 0
    cue_text: str = ""
    age_days: float = 0.0


def select_cue_hygiene_candidates(
    cues: list[CueHygieneCandidate] | list[Mapping[str, Any]],
    *,
    max_age_days: float = 14.0,
    max_hit_count: int = 0,
    max_surfaced_count: int = 0,
    limit: int = 200,
) -> list[CueHygieneCandidate]:
    """Select never-used / dead latent cues eligible for demote/drop."""
    selected: list[CueHygieneCandidate] = []
    for raw in cues:
        if isinstance(raw, CueHygieneCandidate):
            cue = raw
        else:
            cue = CueHygieneCandidate(
                episode_id=str(raw.get("episode_id") or raw.get("id") or ""),
                hit_count=int(raw.get("hit_count") or 0),
                surfaced_count=int(raw.get("surfaced_count") or 0),
                cue_text=str(raw.get("cue_text") or ""),
                age_days=float(raw.get("age_days") or 0.0),
            )
        if not cue.episode_id or not cue.cue_text.strip():
            continue
        if cue.hit_count > max_hit_count:
            continue
        if cue.surfaced_count > max_surfaced_count:
            continue
        if cue.age_days < max_age_days:
            continue
        selected.append(cue)
        if len(selected) >= max(1, limit):
            break
    return selected


@dataclass
class LowValueEntityCandidate:
    entity_id: str
    entity_type: str
    access_count: int = 0
    age_days: float = 0.0
    identity_core: bool = False
    relationship_count: int = 0


_LOW_VALUE_TYPES = frozenset({"Concept", "Artifact", "Technology", "Identifier"})


def select_low_value_prune_candidates(
    entities: list[LowValueEntityCandidate] | list[Mapping[str, Any]],
    *,
    min_age_days: float = 30.0,
    max_access_count: int = 1,
    max_relationships: int = 0,
    limit: int = 50,
    types: frozenset[str] | None = None,
) -> list[LowValueEntityCandidate]:
    """Select age/access-qualified low-value entities for expanded prune.

    Never returns identity_core. Requires low access and few/no relationships.
    """
    allowed = types or _LOW_VALUE_TYPES
    selected: list[LowValueEntityCandidate] = []
    for raw in entities:
        if isinstance(raw, LowValueEntityCandidate):
            ent = raw
        else:
            ent = LowValueEntityCandidate(
                entity_id=str(raw.get("id") or raw.get("entity_id") or ""),
                entity_type=str(raw.get("entity_type") or raw.get("type") or ""),
                access_count=int(raw.get("access_count") or 0),
                age_days=float(raw.get("age_days") or 0.0),
                identity_core=bool(raw.get("identity_core") or False),
                relationship_count=int(raw.get("relationship_count") or 0),
            )
        if not ent.entity_id or ent.identity_core:
            continue
        if ent.entity_type not in allowed:
            continue
        if ent.access_count > max_access_count:
            continue
        if ent.relationship_count > max_relationships:
            continue
        if ent.age_days < min_age_days:
            continue
        selected.append(ent)
        if len(selected) >= max(1, limit):
            break
    return selected


async def collect_hygiene_debt_from_store(
    graph_store: Any,
    group_id: str,
    *,
    orphan_limit: int = 50,
) -> HygieneDebtSnapshot:
    """Best-effort live debt collection from a graph store."""
    stats: dict[str, Any] = {}
    get_stats = getattr(graph_store, "get_stats", None)
    if callable(get_stats):
        try:
            raw = await get_stats(group_id)
            if isinstance(raw, Mapping):
                stats = dict(raw)
        except Exception:
            stats = {}

    open_metrics: dict[str, Any] = {}
    get_open = getattr(graph_store, "get_open_work_metrics", None)
    if callable(get_open):
        try:
            open_metrics = dict(await get_open(group_id) or {})
        except Exception:
            open_metrics = {}

    if open_metrics:
        stats.setdefault(
            "adjudication_metrics",
            {
                "deferred_evidence_count": open_metrics.get("deferred_evidence_count", 0),
                "pending_evidence_count": open_metrics.get("pending_evidence_count", 0),
                "open_request_count": open_metrics.get("open_request_count", 0),
                "open_work_count": open_metrics.get("open_work_count", 0),
            },
        )

    # Optional orphan probe via dead-entity API (cheap limit).
    get_dead = getattr(graph_store, "get_dead_entities", None)
    if callable(get_dead):
        try:
            dead = await get_dead(
                group_id=group_id,
                min_age_days=14,
                limit=orphan_limit,
                max_access_count=2,
            )
            stats["orphan_candidates"] = len(dead or [])
        except Exception:
            pass

    return hygiene_debt_from_stats(stats)
