"""Admin endpoints for loading benchmark corpus into the running server."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_manager
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/load-benchmark")
async def load_benchmark(
    request: Request,
    seed: int = Query(42, description="Corpus RNG seed"),
    structure_aware: bool = Query(
        False, description="Enrich entity text with predicates before embedding",
    ),
) -> JSONResponse:
    """Load the benchmark corpus into the running server's live stores.

    Loads 1,000 entities, 2,500+ relationships, and 6,126 access events
    directly into the graph store, activation store, and search index.
    The dashboard immediately reflects the loaded data.
    """
    from engram.benchmark.corpus import CorpusGenerator

    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    graph_store = manager._graph
    activation_store = manager._activation
    search_index = manager._search

    # Generate corpus
    corpus_gen = CorpusGenerator(seed=seed)
    corpus = corpus_gen.generate()

    # Override group_id to match the current tenant
    for entity in corpus.entities:
        entity.group_id = group_id
    for rel in corpus.relationships:
        rel.group_id = group_id

    # Load into live stores (entities, relationships, access events)
    elapsed = await corpus_gen.load(
        corpus, graph_store, activation_store, search_index,
        structure_aware=structure_aware,
    )

    return JSONResponse(content={
        "loaded": True,
        "seed": seed,
        "group_id": group_id,
        "entities": len(corpus.entities),
        "relationships": len(corpus.relationships),
        "access_events": len(corpus.access_events),
        "queries": len(corpus.ground_truth),
        "elapsed_seconds": round(elapsed, 2),
        "structure_aware": structure_aware,
    })
