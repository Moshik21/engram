"""Benchmark corpus loading service for public admin routes."""

from __future__ import annotations

from typing import Any


class BenchmarkLoadService:
    """Load generated benchmark corpora into the active runtime stores."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index

    async def load_benchmark(
        self,
        *,
        group_id: str,
        seed: int,
        structure_aware: bool = False,
    ) -> dict:
        from engram.benchmark.corpus import CorpusGenerator

        corpus_gen = CorpusGenerator(seed=seed)
        corpus = corpus_gen.generate()

        for entity in corpus.entities:
            entity.group_id = group_id
        for relationship in corpus.relationships:
            relationship.group_id = group_id

        elapsed = await corpus_gen.load(
            corpus,
            self._graph,
            self._activation,
            self._search,
            structure_aware=structure_aware,
        )

        return {
            "loaded": True,
            "seed": seed,
            "group_id": group_id,
            "entities": len(corpus.entities),
            "relationships": len(corpus.relationships),
            "access_events": len(corpus.access_events),
            "queries": len(corpus.ground_truth),
            "elapsed_seconds": round(elapsed, 2),
            "structure_aware": structure_aware,
        }
