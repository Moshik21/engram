"""A/B benchmark framework for Engram retrieval methods."""

from engram.benchmark.corpus import CorpusGenerator, CorpusScale, CorpusSpec, GroundTruthQuery
from engram.benchmark.methods import (
    ALL_METHODS,
    METHOD_COMMUNITY,
    METHOD_CONTEXT_GATED,
    METHOD_FULL_ENGRAM,
    METHOD_FULL_STACK,
    METHOD_MULTI_POOL,
    METHOD_NO_SPREADING,
    METHOD_PURE_SEARCH,
    METHOD_SEARCH_RECENCY,
    RetrievalMethod,
    run_retrieval,
)
from engram.benchmark.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "ALL_METHODS",
    "CorpusGenerator",
    "CorpusScale",
    "CorpusSpec",
    "GroundTruthQuery",
    "METHOD_COMMUNITY",
    "METHOD_CONTEXT_GATED",
    "METHOD_FULL_ENGRAM",
    "METHOD_FULL_STACK",
    "METHOD_MULTI_POOL",
    "METHOD_NO_SPREADING",
    "METHOD_PURE_SEARCH",
    "METHOD_SEARCH_RECENCY",
    "RetrievalMethod",
    "bootstrap_ci",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_retrieval",
]
