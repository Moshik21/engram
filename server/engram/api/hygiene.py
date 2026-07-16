"""Read-only hygiene debt scoreboard over the shell's already-open graph.

Lets the cold brain (and steward CLI) sense debt WITHOUT opening the native
store from a second process: the brain preflights a mop window here and skips
the shell pause entirely when there is no actionable work.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_config, get_graph_store
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hygiene", tags=["hygiene"])


@router.get("/debt")
async def hygiene_debt(request: Request) -> JSONResponse:
    """Debt scoreboard + mop trigger, same shape as `engram hygiene report`."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    graph_store = get_graph_store()
    config = get_config()

    from engram.consolidation.hygiene_debt import (
        collect_hygiene_debt_from_store,
        debt_pressure_contribution,
        debt_should_trigger_mop,
    )

    try:
        debt = await collect_hygiene_debt_from_store(graph_store, group_id)
    except Exception:
        logger.exception("hygiene debt collection failed")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "debt collection failed"},
        )
    debt_pressure = debt_pressure_contribution(debt)
    threshold = float(config.activation.consolidation_pressure_threshold)
    should_mop = debt_should_trigger_mop(debt, pressure_threshold=threshold)
    return JSONResponse(
        content={
            "group_id": group_id,
            "debt": debt.to_dict(),
            "pressure": {
                "hygiene_debt": round(debt_pressure, 2),
                "threshold": threshold,
                "should_trigger_mop": should_mop,
            },
        }
    )
