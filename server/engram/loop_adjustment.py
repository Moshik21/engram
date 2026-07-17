"""Loop Steward control plane: TTL LoopAdjustment store and effective overlays.

Harness (subconscious) writes short-lived allowlisted adjustments; the shell
reads them at cycle/mop start without mutating process-boot ActivationConfig.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phase_registry import CONSOLIDATION_PHASE_ORDER

logger = logging.getLogger(__name__)

KNOWN_PHASES: frozenset[str] = frozenset(CONSOLIDATION_PHASE_ORDER)
KNOWN_REGIMES: frozenset[str] = frozenset(
    {
        "healthy",
        "intake_heavy",
        "debt_heavy",
        "latency_degraded",
        "offline",
    }
)

_MIN_TTL_HOURS = 1
_MAX_TTL_HOURS = 48
_DEFAULT_TTL_HOURS = 12


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            text = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def default_adjustment_path() -> Path:
    override = os.environ.get("ENGRAM_LOOP_ADJUSTMENT_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".engram" / "loop-adjustment.json"


def default_audit_path() -> Path:
    override = os.environ.get("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".engram" / "loop-adjustments.jsonl"


@dataclass
class LoopAdjustment:
    """Short-lived allowlisted control set-point for one group."""

    version: int = 1
    group_id: str = "default"
    regime: str = "healthy"
    reason: str = ""
    ttl_hours: int = _DEFAULT_TTL_HOURS
    created_by: str = "harness"
    max_risk: str = "low"
    budgets: dict[str, int] = field(default_factory=dict)
    phase_boost: list[str] = field(default_factory=list)
    phase_defer: list[str] = field(default_factory=list)
    intake: dict[str, Any] = field(default_factory=dict)
    actions_allowed: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)
    applied_at: str | None = None
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LoopAdjustment:
        budgets_raw = data.get("budgets") or {}
        budgets: dict[str, int] = {}
        if isinstance(budgets_raw, Mapping):
            for key, value in budgets_raw.items():
                try:
                    budgets[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        boost = data.get("phase_boost") or []
        defer = data.get("phase_defer") or []
        intake = data.get("intake") if isinstance(data.get("intake"), Mapping) else {}
        expected = data.get("expected") if isinstance(data.get("expected"), Mapping) else {}
        actions = data.get("actions_allowed") or []
        try:
            ttl = int(data.get("ttl_hours", _DEFAULT_TTL_HOURS))
        except (TypeError, ValueError):
            ttl = _DEFAULT_TTL_HOURS
        try:
            version = int(data.get("version", 1))
        except (TypeError, ValueError):
            version = 1
        return cls(
            version=version,
            group_id=str(data.get("group_id") or "default"),
            regime=str(data.get("regime") or "healthy"),
            reason=str(data.get("reason") or ""),
            ttl_hours=ttl,
            created_by=str(data.get("created_by") or "harness"),
            max_risk=str(data.get("max_risk") or "low"),
            budgets=budgets,
            phase_boost=[str(p) for p in boost] if isinstance(boost, list) else [],
            phase_defer=[str(p) for p in defer] if isinstance(defer, list) else [],
            intake=dict(intake),
            actions_allowed=[str(a) for a in actions] if isinstance(actions, list) else [],
            expected=dict(expected),
            applied_at=str(data["applied_at"]) if data.get("applied_at") else None,
            expires_at=str(data["expires_at"]) if data.get("expires_at") else None,
        )


@dataclass(frozen=True)
class ClampResult:
    adjustment: LoopAdjustment
    warnings: tuple[str, ...] = ()
    rejected: bool = False
    reject_reason: str | None = None


def clamp_loop_adjustment(
    adj: LoopAdjustment,
    *,
    hard_caps: Mapping[str, int] | None = None,
) -> ClampResult:
    """Validate and clamp an adjustment. Rejects hard errors; clamps soft ones."""
    warnings: list[str] = []
    if not str(adj.reason or "").strip():
        return ClampResult(
            adjustment=adj,
            rejected=True,
            reject_reason="reason_required",
        )
    if adj.max_risk and adj.max_risk != "low":
        return ClampResult(
            adjustment=adj,
            rejected=True,
            reject_reason="max_risk_must_be_low",
        )

    regime = adj.regime if adj.regime in KNOWN_REGIMES else "healthy"
    if regime != adj.regime:
        warnings.append(f"unknown_regime:{adj.regime}")

    ttl = int(adj.ttl_hours)
    if ttl < _MIN_TTL_HOURS:
        warnings.append(f"ttl_clamped_min:{ttl}")
        ttl = _MIN_TTL_HOURS
    if ttl > _MAX_TTL_HOURS:
        warnings.append(f"ttl_clamped_max:{ttl}")
        ttl = _MAX_TTL_HOURS

    caps = {
        "evidence_drain": 5000,
        "already_exists": 10000,
        "stale_reject": 10000,
        "cue_hygiene": 5000,
        "adjudication_limit": 5000,
    }
    if hard_caps:
        for key, value in hard_caps.items():
            try:
                caps[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    clamped_budgets: dict[str, int] = {}
    for key, value in (adj.budgets or {}).items():
        try:
            iv = int(value)
        except (TypeError, ValueError):
            warnings.append(f"budget_ignored:{key}")
            continue
        if iv < 0:
            warnings.append(f"budget_clamped_min:{key}")
            iv = 0
        cap = caps.get(key)
        if cap is not None and iv > cap:
            warnings.append(f"budget_clamped_max:{key}:{iv}->{cap}")
            iv = cap
        clamped_budgets[key] = iv

    def _filter_phases(names: list[str], label: str) -> list[str]:
        out: list[str] = []
        for name in names:
            if name not in KNOWN_PHASES:
                warnings.append(f"unknown_phase_{label}:{name}")
                continue
            if name not in out:
                out.append(name)
        return out

    boost = _filter_phases(list(adj.phase_boost or []), "boost")
    defer = _filter_phases(list(adj.phase_defer or []), "defer")

    intake = dict(adj.intake or {})
    if "auto_extract_min_score" in intake:
        try:
            score = float(intake["auto_extract_min_score"])
            score = min(1.0, max(0.0, score))
            intake["auto_extract_min_score"] = score
        except (TypeError, ValueError):
            warnings.append("intake_score_ignored")
            intake.pop("auto_extract_min_score", None)
    if intake.get("pattern_junk_reject") is False:
        # v1: steward cannot disable pattern junk reject. The knob has no
        # runtime consumer by design: the hot-path junk gate in
        # extraction/commit_policy is unconditionally on, so True is simply
        # the truthful (and only) state.
        warnings.append("pattern_junk_reject_forced_on")
        intake["pattern_junk_reject"] = True

    clamped = LoopAdjustment(
        version=1,
        group_id=adj.group_id if adj.group_id else "default",
        regime=regime,
        reason=str(adj.reason).strip(),
        ttl_hours=ttl,
        created_by=adj.created_by or "harness",
        max_risk="low",
        budgets=clamped_budgets,
        phase_boost=boost,
        phase_defer=defer,
        intake=intake,
        actions_allowed=list(adj.actions_allowed or []),
        expected=dict(adj.expected or {}),
        applied_at=adj.applied_at,
        expires_at=adj.expires_at,
    )
    return ClampResult(adjustment=clamped, warnings=tuple(warnings))


def is_expired(adj: LoopAdjustment, *, now: datetime | None = None) -> bool:
    clock = now or _utc_now()
    expires = _parse_dt(adj.expires_at)
    if expires is not None:
        return clock >= expires
    applied = _parse_dt(adj.applied_at)
    if applied is None:
        return False
    return clock >= applied + timedelta(hours=int(adj.ttl_hours or _DEFAULT_TTL_HOURS))


def remaining_ttl_seconds(adj: LoopAdjustment, *, now: datetime | None = None) -> float:
    clock = now or _utc_now()
    expires = _parse_dt(adj.expires_at)
    if expires is None:
        applied = _parse_dt(adj.applied_at)
        if applied is None:
            return float(int(adj.ttl_hours) * 3600)
        expires = applied + timedelta(hours=int(adj.ttl_hours or _DEFAULT_TTL_HOURS))
    return max(0.0, (expires - clock).total_seconds())


def stamp_applied(adj: LoopAdjustment, *, now: datetime | None = None) -> LoopAdjustment:
    clock = now or _utc_now()
    expires = clock + timedelta(hours=int(adj.ttl_hours or _DEFAULT_TTL_HOURS))
    adj.applied_at = clock.isoformat()
    adj.expires_at = expires.isoformat()
    return adj


def _load_from_file(
    group_id: str,
    *,
    path: Path | None,
    now: datetime | None,
    clear_if_expired: bool,
    audit_path: Path | None = None,
) -> LoopAdjustment | None:
    store_path = path or default_adjustment_path()
    if not store_path.is_file():
        return None
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read loop adjustment from %s", store_path, exc_info=True)
        return None
    if not isinstance(raw, Mapping):
        return None
    adj = LoopAdjustment.from_mapping(raw)
    if adj.group_id != group_id:
        return None
    if is_expired(adj, now=now):
        if clear_if_expired:
            try:
                store_path.unlink(missing_ok=True)
            except OSError:
                pass
            append_audit_event(
                {
                    "event": "expire",
                    "group_id": group_id,
                    "reason": adj.reason,
                    "created_by": adj.created_by,
                    "store": "file",
                },
                path=audit_path,
            )
        return None
    adj._loaded_from = "file"
    return adj


def load_active_adjustment(
    group_id: str = "default",
    *,
    path: Path | None = None,
    now: datetime | None = None,
    clear_if_expired: bool = True,
    graph_store: Any | None = None,
    audit_path: Path | None = None,
) -> LoopAdjustment | None:
    """Load active adjustment: prefer graph/consol store, dual-read file fallback.

    Helix native uses the consolidation sidecar (``save_loop_adjustment``) when
    provided; file under ``~/.engram`` remains the portable dual-write path.
    """
    # Prefer graph-backed payload when available (sync duck-type for file-like;
    # async stores use load_active_adjustment_async).
    if graph_store is not None:
        loader = getattr(graph_store, "get_loop_adjustment_sync", None)
        if callable(loader):
            try:
                raw = loader(group_id)
                if isinstance(raw, Mapping):
                    adj = LoopAdjustment.from_mapping(raw)
                    if adj.group_id == group_id and not is_expired(adj, now=now):
                        adj._loaded_from = "graph"
                        return adj
                    if clear_if_expired and is_expired(adj, now=now):
                        clearer = getattr(graph_store, "clear_loop_adjustment_sync", None)
                        if callable(clearer):
                            clearer(group_id)
            except Exception:
                logger.debug("graph sync loop load failed", exc_info=True)

    return _load_from_file(
        group_id,
        path=path,
        now=now,
        clear_if_expired=clear_if_expired,
        audit_path=audit_path,
    )


async def load_active_adjustment_async(
    group_id: str = "default",
    *,
    path: Path | None = None,
    now: datetime | None = None,
    clear_if_expired: bool = True,
    graph_store: Any | None = None,
    audit_path: Path | None = None,
) -> LoopAdjustment | None:
    """Async dual-read: Helix/consol graph store first, then file."""
    if graph_store is not None:
        try:
            adj = await graph_load_loop_adjustment(graph_store, group_id)
            if adj is not None and adj.group_id == group_id:
                if is_expired(adj, now=now):
                    if clear_if_expired:
                        await graph_clear_loop_adjustment(graph_store, group_id)
                        append_audit_event(
                            {
                                "event": "expire",
                                "group_id": group_id,
                                "reason": adj.reason,
                                "store": "graph",
                            }
                        )
                    adj = None
                else:
                    return adj
        except Exception:
            logger.debug("graph async loop load failed", exc_info=True)
    return _load_from_file(
        group_id,
        path=path,
        now=now,
        clear_if_expired=clear_if_expired,
        audit_path=audit_path,
    )


def save_active_adjustment(
    adj: LoopAdjustment,
    *,
    path: Path | None = None,
    audit_path: Path | None = None,
    graph_store: Any | None = None,
) -> LoopAdjustment:
    """Replace the single active adjustment (must already be clamped + stamped).

    Always writes the file dual-path; also writes graph/consol store when a
    sync saver is present. Prefer ``save_active_adjustment_async`` for Helix.
    """
    store_path = path or default_adjustment_path()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = adj.to_dict()
    store_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if graph_store is not None:
        saver = getattr(graph_store, "save_loop_adjustment_sync", None)
        if callable(saver):
            try:
                saver(adj.group_id, payload)
            except Exception:
                logger.debug("graph sync loop save failed", exc_info=True)
    append_audit_event(
        {
            "event": "apply",
            "group_id": adj.group_id,
            "regime": adj.regime,
            "reason": adj.reason,
            "created_by": adj.created_by,
            "ttl_hours": adj.ttl_hours,
            "expires_at": adj.expires_at,
            "budgets": adj.budgets,
            "phase_boost": adj.phase_boost,
            "phase_defer": adj.phase_defer,
            "intake": adj.intake,
            "store": "file+graph" if graph_store is not None else "file",
        },
        path=audit_path,
    )
    return adj


async def save_active_adjustment_async(
    adj: LoopAdjustment,
    *,
    path: Path | None = None,
    audit_path: Path | None = None,
    graph_store: Any | None = None,
    file_write: bool = True,
) -> LoopAdjustment:
    """Dual-write file + Helix/consol graph store.

    file_write=False mirrors to the graph sidecar only — used when the file
    was already written (e.g. run_steward_once) so the apply audit event is
    not duplicated.
    """
    if file_write:
        save_active_adjustment(adj, path=path, audit_path=audit_path, graph_store=None)
    if graph_store is not None:
        try:
            await graph_save_loop_adjustment(graph_store, adj)
            append_audit_event(
                {
                    "event": "apply_graph",
                    "group_id": adj.group_id,
                    "store": "graph",
                },
                path=audit_path,
            )
        except Exception:
            logger.debug("graph async loop save failed", exc_info=True)
    return adj


def clear_active_adjustment(
    group_id: str = "default",
    *,
    path: Path | None = None,
    audit_path: Path | None = None,
    cleared_by: str = "harness",
    graph_store: Any | None = None,
) -> bool:
    store_path = path or default_adjustment_path()
    existed = store_path.is_file()
    if existed:
        keep_file = False
        try:
            raw = json.loads(store_path.read_text(encoding="utf-8"))
            prev_group = str((raw or {}).get("group_id") or "") or "default"
            if prev_group != group_id and isinstance(raw, Mapping):
                # Different group — leave the file, but still clear OUR
                # group's graph sidecar copy below (an early return here
                # left stale graph copies reported as active).
                keep_file = True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        if not keep_file:
            try:
                store_path.unlink(missing_ok=True)
            except OSError:
                return False
    if graph_store is not None:
        clearer = getattr(graph_store, "clear_loop_adjustment_sync", None)
        if callable(clearer):
            try:
                clearer(group_id)
            except Exception:
                logger.debug("graph sync loop clear failed", exc_info=True)
    append_audit_event(
        {
            "event": "clear",
            "group_id": group_id,
            "created_by": cleared_by,
            "existed": existed,
        },
        path=audit_path,
    )
    return existed


async def clear_active_adjustment_async(
    group_id: str = "default",
    *,
    path: Path | None = None,
    audit_path: Path | None = None,
    cleared_by: str = "harness",
    graph_store: Any | None = None,
) -> bool:
    existed = clear_active_adjustment(
        group_id,
        path=path,
        audit_path=audit_path,
        cleared_by=cleared_by,
        graph_store=None,
    )
    if graph_store is not None:
        try:
            if await graph_clear_loop_adjustment(graph_store, group_id):
                existed = True
        except Exception:
            logger.debug("graph async loop clear failed", exc_info=True)
    return existed


def append_audit_event(
    event: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> None:
    audit = path or default_audit_path()
    try:
        audit.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", _utc_now().isoformat())
        with audit.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        logger.debug("Failed to append loop adjustment audit", exc_info=True)


def effective_activation_config(
    base: ActivationConfig,
    adj: LoopAdjustment | None,
) -> ActivationConfig:
    """Return base cfg or a model_copy overlay from the active adjustment."""
    if adj is None:
        return base
    updates: dict[str, Any] = {}
    budgets = adj.budgets or {}
    if "evidence_drain" in budgets:
        updates["consolidation_evidence_drain_max_per_cycle"] = int(budgets["evidence_drain"])
    if "already_exists" in budgets:
        updates["consolidation_evidence_already_exists_max_per_cycle"] = int(
            budgets["already_exists"]
        )
    if "stale_reject" in budgets:
        updates["consolidation_evidence_stale_max_per_cycle"] = int(budgets["stale_reject"])
    if "cue_hygiene" in budgets:
        updates["consolidation_cue_hygiene_max_per_cycle"] = int(budgets["cue_hygiene"])
    if "adjudication_limit" in budgets:
        updates["consolidation_evidence_adjudication_limit"] = int(budgets["adjudication_limit"])
    intake = adj.intake or {}
    if "auto_extract_min_score" in intake:
        try:
            updates["worker_auto_capture_extract_score_floor"] = float(
                intake["auto_extract_min_score"]
            )
        except (TypeError, ValueError):
            pass
    if not updates:
        return base
    return base.model_copy(update=updates)


def effective_phase_names(
    due: set[str] | frozenset[str] | None,
    adj: LoopAdjustment | None,
    *,
    full_order: tuple[str, ...] | None = None,
) -> set[str] | None:
    """Apply phase_defer / phase_boost to a due-phase set.

    ``due is None`` means full cycle (all phases). Defer still applies and
    returns an explicit set when bias is present; without bias returns None
    (engine runs all phases).

    Never returns empty when the pre-bias set was non-empty.
    """
    order = full_order or CONSOLIDATION_PHASE_ORDER
    known = set(order)

    if adj is None or not (adj.phase_defer or adj.phase_boost):
        if due is None:
            return None
        return set(due) & known

    if due is None:
        selected: set[str] = set(known)
        original: set[str] = set(known)
    else:
        selected = set(due) & known
        original = set(selected)

    for name in adj.phase_defer or []:
        if name in known:
            selected.discard(name)

    for name in adj.phase_boost or []:
        if name in known:
            selected.add(name)

    if original and not selected:
        return original

    return selected


def hard_caps_from_config(cfg: ActivationConfig) -> dict[str, int]:
    """Derive clamp caps from activation config."""
    return {
        "evidence_drain": int(
            getattr(cfg, "consolidation_evidence_drain_max_budget", 5000) or 5000
        ),
        "already_exists": int(
            getattr(cfg, "consolidation_evidence_already_exists_max_per_cycle", 500) or 500
        )
        * 20,  # allow steward to raise above default cycle cap up to 20x default floor
        "stale_reject": 10000,
        "cue_hygiene": 5000,
        "adjudication_limit": 5000,
    }


def mop_knob_budgets(
    cli_budget: int,
    adj: LoopAdjustment | None,
) -> dict[str, int]:
    """Resolve per-drain mop budgets.

    CLI ``--budget`` is the floor for every knob. Active steward ``budgets`` keys
    raise only the knobs they set (divergent per-knob overlays).
    """
    floor = max(1, int(cli_budget))
    keys = ("evidence_drain", "already_exists", "stale_reject", "cue_hygiene")
    out: dict[str, int] = {k: floor for k in keys}
    if adj is None:
        return out
    for key in keys:
        if key not in (adj.budgets or {}):
            continue
        try:
            out[key] = max(floor, int(adj.budgets[key]))
        except (TypeError, ValueError):
            continue
    return out


def status_payload(
    group_id: str = "default",
    *,
    path: Path | None = None,
    now: datetime | None = None,
    graph_store: Any | None = None,
) -> dict[str, Any]:
    adj = load_active_adjustment(
        group_id, path=path, now=now, clear_if_expired=True, graph_store=graph_store
    )
    if adj is None:
        return {
            "group_id": group_id,
            "active": False,
            "adjustment": None,
            "remaining_ttl_seconds": 0,
            "store": "none",
        }
    return {
        "group_id": group_id,
        "active": True,
        "adjustment": adj.to_dict(),
        "remaining_ttl_seconds": round(remaining_ttl_seconds(adj, now=now), 1),
        "regime": adj.regime,
        "reason": adj.reason,
        "expires_at": adj.expires_at,
        "store": getattr(adj, "_loaded_from", None) or "file",
    }


def classify_regime_from_debt(
    debt: Mapping[str, Any] | None,
    *,
    server_reachable: bool | None = None,
    continuity_ok: bool | None = None,
    latency_degraded: bool | None = None,
) -> str:
    """Deterministic regime from a hygiene scoreboard / debt snapshot."""
    if server_reachable is False:
        return "offline"
    if latency_degraded is True or continuity_ok is False:
        return "latency_degraded"

    if not isinstance(debt, Mapping):
        return "healthy"

    deferred = int(debt.get("deferred_evidence") or debt.get("deferred") or 0)
    cue_only = int(debt.get("cue_only_episodes") or debt.get("cue_only") or 0)
    open_work = int(debt.get("open_work") or 0)
    should_mop = bool(debt.get("should_trigger_mop") or debt.get("should_mop"))
    pressure = debt.get("pressure") if isinstance(debt.get("pressure"), Mapping) else {}
    if isinstance(pressure, Mapping) and pressure.get("should_trigger_mop"):
        should_mop = True

    # Intake flood signal: large cue_only relative to deferred
    if cue_only >= 500 and cue_only >= max(deferred, 1) * 0.4 and deferred < 2000:
        return "intake_heavy"

    if should_mop or deferred >= 500 or open_work >= 800:
        return "debt_heavy"

    if deferred >= 100 or cue_only >= 200:
        return "debt_heavy"

    return "healthy"


def propose_from_report(
    debt: Mapping[str, Any] | None = None,
    *,
    group_id: str = "default",
    created_by: str = "harness:propose-from-report",
    server_reachable: bool | None = None,
    continuity_ok: bool | None = None,
    latency_degraded: bool | None = None,
    ttl_hours: int = 12,
    hard_caps: Mapping[str, int] | None = None,
) -> ClampResult:
    """Build a clamped LoopAdjustment from debt scoreboard — does not write active state."""
    regime = classify_regime_from_debt(
        debt,
        server_reachable=server_reachable,
        continuity_ok=continuity_ok,
        latency_degraded=latency_degraded,
    )
    deferred = 0
    cue_only = 0
    if isinstance(debt, Mapping):
        deferred = int(debt.get("deferred_evidence") or debt.get("deferred") or 0)
        cue_only = int(debt.get("cue_only_episodes") or debt.get("cue_only") or 0)

    if regime == "healthy":
        raw = LoopAdjustment(
            group_id=group_id,
            regime="healthy",
            reason="scoreboard healthy; no adjustment recommended",
            ttl_hours=ttl_hours,
            created_by=created_by,
            max_risk="low",
            budgets={},
            phase_boost=[],
            phase_defer=[],
            intake={"pattern_junk_reject": True},
            expected={"continuity_must_pass": True},
        )
        return clamp_loop_adjustment(raw, hard_caps=hard_caps)

    if regime == "offline":
        raw = LoopAdjustment(
            group_id=group_id,
            regime="offline",
            reason="runtime unreachable; queue debt-heavy recovery for next healthy start",
            ttl_hours=min(ttl_hours, 24),
            created_by=created_by,
            max_risk="low",
            budgets={
                "evidence_drain": min(2000, max(500, deferred // 4 or 500)),
                "already_exists": 500,
                "stale_reject": 500,
                "cue_hygiene": 500,
                "adjudication_limit": 400,
            },
            phase_boost=["evidence_adjudication", "prune"],
            phase_defer=["graph_embed", "dream"],
            intake={"pattern_junk_reject": True, "auto_extract_min_score": 0.85},
            expected={"continuity_must_pass": False},
        )
        return clamp_loop_adjustment(raw, hard_caps=hard_caps)

    if regime == "latency_degraded":
        raw = LoopAdjustment(
            group_id=group_id,
            regime="latency_degraded",
            reason="continuity miss or latency degraded; protect durable path",
            ttl_hours=ttl_hours,
            created_by=created_by,
            max_risk="low",
            budgets={
                "evidence_drain": min(1500, max(500, deferred // 5 or 500)),
                "already_exists": 500,
                "stale_reject": 400,
                "cue_hygiene": 300,
                "adjudication_limit": 300,
            },
            phase_boost=["evidence_adjudication"],
            phase_defer=["graph_embed", "dream"],
            intake={"pattern_junk_reject": True, "auto_extract_min_score": 0.88},
            expected={"continuity_must_pass": True},
        )
        return clamp_loop_adjustment(raw, hard_caps=hard_caps)

    if regime == "intake_heavy":
        raw = LoopAdjustment(
            group_id=group_id,
            regime="intake_heavy",
            reason=f"cue_only={cue_only} rising; raise auto-capture floor",
            ttl_hours=ttl_hours,
            created_by=created_by,
            max_risk="low",
            budgets={
                "evidence_drain": 800,
                "already_exists": 400,
                "stale_reject": 400,
                "cue_hygiene": min(2000, max(500, cue_only // 3 or 500)),
                "adjudication_limit": 300,
            },
            phase_boost=["compact", "prune"],
            phase_defer=["dream", "graph_embed"],
            intake={"pattern_junk_reject": True, "auto_extract_min_score": 0.9},
            expected={"continuity_must_pass": True},
        )
        return clamp_loop_adjustment(raw, hard_caps=hard_caps)

    # debt_heavy default
    raw = LoopAdjustment(
        group_id=group_id,
        regime="debt_heavy",
        reason=f"deferred≈{deferred}; should_mop recovery window",
        ttl_hours=ttl_hours,
        created_by=created_by,
        max_risk="low",
        budgets={
            "evidence_drain": min(5000, max(1000, deferred // 4 or 1000)),
            "already_exists": 500,
            "stale_reject": 500,
            "cue_hygiene": 500,
            "adjudication_limit": 400,
        },
        phase_boost=["evidence_adjudication", "prune"],
        phase_defer=["graph_embed", "dream"],
        intake={"pattern_junk_reject": True, "auto_extract_min_score": 0.85},
        expected={"continuity_must_pass": True},
    )
    return clamp_loop_adjustment(raw, hard_caps=hard_caps)


def run_steward_once(
    debt: Mapping[str, Any] | None,
    *,
    group_id: str = "default",
    created_by: str = "harness:steward-once",
    dry_run: bool = False,
    do_mop: bool = False,
    mop_budget: int = 200,
    server_reachable: bool | None = None,
    continuity_ok: bool | None = None,
    latency_degraded: bool | None = None,
    hard_caps: Mapping[str, int] | None = None,
    path: Path | None = None,
    audit_path: Path | None = None,
    mop_fn: Any | None = None,
) -> dict[str, Any]:
    """One-shot sense→propose→apply (if needed)→optional mop. Pure orchestration.

    ``debt`` is a hygiene scoreboard dict (or full report with nested ``debt`` /
    ``pressure``). Healthy regime never writes an active adjustment.
    """
    debt_map: dict[str, Any] | None
    if isinstance(debt, Mapping) and isinstance(debt.get("debt"), Mapping):
        debt_map = dict(debt["debt"])
        if isinstance(debt.get("pressure"), Mapping):
            debt_map["pressure"] = debt["pressure"]
            debt_map["should_trigger_mop"] = debt["pressure"].get("should_trigger_mop")
    elif isinstance(debt, Mapping):
        debt_map = dict(debt)
    else:
        debt_map = None

    debt_before = dict(debt_map) if debt_map else None
    result = propose_from_report(
        debt_map,
        group_id=group_id,
        created_by=created_by,
        server_reachable=server_reachable,
        continuity_ok=continuity_ok,
        latency_degraded=latency_degraded,
        hard_caps=hard_caps,
    )
    if result.rejected:
        return {
            "status": "error",
            "error": result.reject_reason,
            "regime": None,
            "applied": False,
            "wrote_active": False,
            "dry_run": dry_run,
            "debt_before": debt_before,
            "debt_after": None,
            "mop": None,
            "warnings": list(result.warnings),
        }

    regime = result.adjustment.regime
    payload: dict[str, Any] = {
        "status": "ok",
        "regime": regime,
        "applied": False,
        "wrote_active": False,
        "dry_run": dry_run,
        "reason": result.adjustment.reason,
        "adjustment": result.adjustment.to_dict(),
        "warnings": list(result.warnings),
        "debt_before": debt_before,
        "debt_after": None,
        "mop": None,
        "healthy_noop": regime == "healthy",
    }

    if regime == "healthy":
        payload["status_payload"] = status_payload(group_id, path=path)
        return payload

    if dry_run:
        payload["would_apply"] = True
        payload["status_payload"] = status_payload(group_id, path=path)
        return payload

    stamped = stamp_applied(result.adjustment)
    save_active_adjustment(stamped, path=path, audit_path=audit_path)
    payload["applied"] = True
    payload["wrote_active"] = True
    payload["adjustment"] = stamped.to_dict()
    payload["status_payload"] = status_payload(group_id, path=path)

    if do_mop and mop_fn is not None:
        try:
            mop_result = mop_fn(budget=mop_budget, dry_run=False)
            payload["mop"] = mop_result
            if isinstance(mop_result, Mapping) and isinstance(
                mop_result.get("debt_after"), Mapping
            ):
                payload["debt_after"] = mop_result["debt_after"]
            elif isinstance(mop_result, Mapping) and isinstance(mop_result.get("debt"), Mapping):
                payload["debt_after"] = mop_result["debt"]
        except Exception as exc:
            payload["mop"] = {"error": str(exc)}
    elif do_mop:
        payload["mop"] = {
            "skipped": True,
            "reason": "mop_fn_not_provided",
            "hint": "CLI/MCP supply mop runner; unit tests inject mop_fn",
        }

    return payload


# ── Graph dual-store helpers (async graph methods, sync file fallback) ──


async def graph_save_loop_adjustment(graph_store: Any, adj: LoopAdjustment) -> bool:
    saver = getattr(graph_store, "save_loop_adjustment", None)
    if not callable(saver):
        return False
    await saver(adj.group_id, adj.to_dict())
    return True


async def graph_load_loop_adjustment(graph_store: Any, group_id: str) -> LoopAdjustment | None:
    loader = getattr(graph_store, "get_loop_adjustment", None)
    if not callable(loader):
        return None
    raw = await loader(group_id)
    if not isinstance(raw, Mapping):
        return None
    return LoopAdjustment.from_mapping(raw)


async def graph_clear_loop_adjustment(graph_store: Any, group_id: str) -> bool:
    clearer = getattr(graph_store, "clear_loop_adjustment", None)
    if not callable(clearer):
        return False
    return bool(await clearer(group_id))
