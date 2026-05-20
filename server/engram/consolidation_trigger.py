"""Consolidation trigger helpers for public control surfaces."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from engram.config import NerveCenterConfig
from engram.storage.bootstrap import (
    borrowed_sqlite_db,
    create_borrowed_sqlite_consolidation_store,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiConsolidationTriggerSurface:
    """REST trigger payload plus whether a background cycle should be scheduled."""

    status_code: int
    payload: dict
    should_run: bool


@dataclass(frozen=True)
class ApiConsolidationDetailSurface:
    """REST consolidation detail payload plus HTTP status."""

    status_code: int
    payload: dict


def build_api_consolidation_trigger_surface(
    engine: Any,
    *,
    group_id: str,
    dry_run: bool,
) -> ApiConsolidationTriggerSurface:
    """Return the REST trigger response and run decision."""
    if engine.is_running:
        return ApiConsolidationTriggerSurface(
            status_code=409,
            payload={"detail": "A consolidation cycle is already running"},
            should_run=False,
        )
    return ApiConsolidationTriggerSurface(
        status_code=200,
        payload={
            "status": "triggered",
            "group_id": group_id,
            "dry_run": dry_run,
        },
        should_run=True,
    )


def build_api_consolidation_trigger_response_surface(
    engine: Any,
    *,
    group_id: str,
    dry_run: bool,
    background_tasks: Any,
    logger: logging.Logger | None = None,
) -> ApiConsolidationTriggerSurface:
    """Return the REST trigger response and schedule the background cycle."""
    result = build_api_consolidation_trigger_surface(
        engine,
        group_id=group_id,
        dry_run=dry_run,
    )
    if result.should_run:
        background_tasks.add_task(
            run_api_consolidation_cycle,
            engine,
            group_id=group_id,
            dry_run=dry_run,
            logger=logger,
        )
    return result


async def run_api_consolidation_cycle(
    engine: Any,
    *,
    group_id: str,
    dry_run: bool,
    logger: logging.Logger | None = None,
) -> None:
    """Run a REST-triggered consolidation cycle for a background task."""
    try:
        await engine.run_cycle(group_id=group_id, trigger="manual", dry_run=dry_run)
    except Exception:
        if logger is not None:
            logger.exception("Background consolidation cycle failed")


async def run_shutdown_consolidation(
    engine: Any | None,
    *,
    config: Any | None,
    logger: logging.Logger | None = None,
) -> None:
    """Cancel or run the final REST shutdown consolidation cycle."""
    if engine is None:
        return
    if engine.is_running:
        engine.cancel()
        return
    if config is None or not config.activation.consolidation_enabled:
        return
    try:
        run = engine.run_cycle(
            group_id=config.default_group_id,
            trigger="shutdown",
            dry_run=False,
        )
        timeout = getattr(config.activation, "consolidation_shutdown_timeout_seconds", 5.0)
        if timeout and timeout > 0:
            await asyncio.wait_for(run, timeout=timeout)
        else:
            await run
    except TimeoutError:
        engine.cancel()
        if logger is not None:
            logger.warning("Shutdown consolidation timed out; cancelled final cycle")
    except Exception:
        if logger is not None:
            logger.warning("Shutdown consolidation failed", exc_info=True)


async def build_api_consolidation_status_surface(
    engine: Any,
    *,
    group_id: str,
    scheduler: Any | None = None,
    pressure: Any | None = None,
    activation_cfg: Any | None = None,
) -> dict:
    """Return the REST consolidation status payload."""
    from engram.consolidation.presenter import serialize_cycle_summary

    result: dict = {
        "is_running": engine.is_running,
        "scheduler_active": scheduler.is_active if scheduler else False,
    }

    if pressure and activation_cfg:
        snapshot = pressure.get_snapshot(group_id)
        if snapshot:
            result["pressure"] = {
                "value": round(pressure.get_pressure(group_id, activation_cfg), 2),
                "threshold": activation_cfg.consolidation_pressure_threshold,
                "episodes_since_last": snapshot.episodes_since_last,
                "entities_created": snapshot.entities_created,
                "last_cycle_time": snapshot.last_cycle_time,
            }

    latest_cycle = await engine.get_latest_cycle(group_id)
    if latest_cycle is not None:
        result["latest_cycle"] = serialize_cycle_summary(latest_cycle)
    return result


async def build_api_consolidation_status_response_surface(
    engine: Any,
    *,
    group_id: str,
    scheduler: Any | None = None,
    pressure: Any | None = None,
    config: Any | None = None,
) -> dict:
    """Return the REST consolidation status payload from route dependencies."""
    activation_cfg = config.activation if pressure is not None and config is not None else None
    return await build_api_consolidation_status_surface(
        engine,
        group_id=group_id,
        scheduler=scheduler,
        pressure=pressure,
        activation_cfg=activation_cfg,
    )


async def build_api_consolidation_history_surface(
    engine: Any,
    *,
    group_id: str,
    limit: int,
) -> dict:
    """Return the REST consolidation history payload."""
    from engram.consolidation.presenter import serialize_cycle_summary

    cycles = await engine.get_recent_cycles(group_id, limit=limit)
    return {"cycles": [serialize_cycle_summary(cycle) for cycle in cycles]}


async def build_api_consolidation_cycle_detail_surface(
    engine: Any,
    *,
    group_id: str,
    cycle_id: str,
) -> ApiConsolidationDetailSurface:
    """Return the REST consolidation cycle detail payload."""
    from engram.consolidation.presenter import serialize_cycle_detail

    if not engine.audit_store_available:
        return ApiConsolidationDetailSurface(
            status_code=404,
            payload={"detail": "Consolidation store not available"},
        )

    detail = await engine.get_cycle_detail(cycle_id, group_id)
    if detail is None:
        return ApiConsolidationDetailSurface(
            status_code=404,
            payload={"detail": "Cycle not found"},
        )

    return ApiConsolidationDetailSurface(
        status_code=200,
        payload=serialize_cycle_detail(detail),
    )


async def build_mcp_consolidation_status_surface(
    consolidation_store: Any | None,
    *,
    group_id: str,
) -> dict:
    """Return the MCP consolidation status payload from the active audit store."""
    from engram.consolidation.audit_reader import ConsolidationAuditReader
    from engram.consolidation.presenter import serialize_cycle_summary

    result = {
        "is_running": False,
        "message": "Use trigger_consolidation to run a cycle. "
        "In MCP mode, cycles run synchronously.",
    }
    latest_cycle = await ConsolidationAuditReader(consolidation_store).latest_cycle(group_id)
    if latest_cycle is not None:
        result["latest_cycle"] = serialize_cycle_summary(latest_cycle)
    return result


async def build_mcp_consolidation_trigger_surface(
    manager: Any,
    *,
    group_id: str,
    dry_run: bool,
    consolidation_store: Any | None,
) -> dict:
    """Run an MCP-triggered consolidation cycle and return its public payload."""
    from engram.consolidation.presenter import serialize_cycle_summary

    trigger_result = await manager.trigger_consolidation_cycle(
        group_id=group_id,
        trigger="mcp",
        dry_run=dry_run,
        consolidation_store=consolidation_store,
    )
    result = serialize_cycle_summary(trigger_result.cycle)
    result["cycle_id"] = result.pop("id")
    result["graph_stats"] = trigger_result.graph_stats
    return result


async def resolve_mcp_consolidation_trigger_store(
    manager: Any,
    active_store: Any | None,
) -> Any | None:
    """Return the active MCP audit store, with a lite shared-DB fallback."""
    if active_store is not None:
        return active_store

    db = manager.get_consolidation_shared_db()
    return await create_borrowed_sqlite_consolidation_store(db)


@dataclass(frozen=True)
class ConsolidationTriggerResult:
    """Result of a public consolidation trigger."""

    cycle: Any
    graph_stats: dict


def _build_consolidation_engine(*args, **kwargs):
    from engram.consolidation.engine import ConsolidationEngine

    return ConsolidationEngine(*args, **kwargs)


class ConsolidationTriggerService:
    """Own ad hoc consolidation cycle construction and execution."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: Any,
        extractor: Any,
        nerve_center_cfg: Any | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._extractor = extractor
        self._nerve_center_cfg = (
            nerve_center_cfg
            or getattr(cfg, "nerve_center", None)
            or NerveCenterConfig()
        )

    async def trigger_consolidation_cycle(
        self,
        *,
        group_id: str,
        trigger: str,
        dry_run: bool,
        consolidation_store: Any | None = None,
    ) -> ConsolidationTriggerResult:
        """Run a public ad hoc consolidation cycle."""
        graph_stats = await self._graph.get_stats(group_id)

        # Level Gating logic
        total_entities = graph_stats.get("total_entities", 0)
        current_level = (total_entities // 50) + 1

        # If trigger is not manual and level is too low for autonomous consolidation,
        # we might need to block or warn
        if trigger.startswith("tiered") or trigger == "pressure" or trigger == "scheduled":
            nerve_cfg = self._nerve_center_cfg
            if nerve_cfg.level_gating_enabled:
                required_level = nerve_cfg.level_unlock_autonomous_consolidation
                if current_level < required_level:
                    logger.warning(
                        "Autonomous consolidation blocked: Level %d too low (Requires Level %d)",
                        current_level,
                        required_level,
                    )
                    # Allow short maintenance cycles to keep capture responsive;
                    # block pressure and cold autonomous work until the graph matures.
                    if "cold" in trigger or trigger == "pressure":
                        raise PermissionError(
                            f"Autonomous consolidation requires Level {required_level}"
                        )

        engine = _build_consolidation_engine(
            self._graph,
            self._activation,
            self._search,
            cfg=self._cfg,
            consolidation_store=consolidation_store,
            extractor=self._extractor,
        )
        cycle = await engine.run_cycle(
            group_id=group_id,
            trigger=trigger,
            dry_run=dry_run,
        )
        return ConsolidationTriggerResult(cycle=cycle, graph_stats=graph_stats)

    def shared_sqlite_db(self) -> Any | None:
        """Return the shared sqlite handle used by lite graph stores, if any."""
        return borrowed_sqlite_db(self._graph)
