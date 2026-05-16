"""Admin endpoints for loading benchmark corpus into the running server."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_manager
from engram.benchmark_loader import build_api_benchmark_load_surface
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/load-benchmark")
async def load_benchmark(
    request: Request,
    seed: int = Query(42, description="Corpus RNG seed"),
    structure_aware: bool = Query(
        False,
        description="Enrich entity text with predicates before embedding",
    ),
) -> JSONResponse:
    """Load the benchmark corpus into the running server's live stores.

    Loads 1,000 entities, 2,500+ relationships, and 6,126 access events
    directly into the graph store, activation store, and search index.
    The dashboard immediately reflects the loaded data.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_benchmark_load_surface(
        manager,
        group_id=group_id,
        seed=seed,
        structure_aware=structure_aware,
    )
    return JSONResponse(content=payload)
