"""Graph embedding consolidation phase: trains structural embeddings."""

from __future__ import annotations

import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.embeddings.graph.storage import GraphEmbeddingStore
from engram.models.consolidation import CycleContext, GraphEmbedRecord, PhaseResult

logger = logging.getLogger(__name__)

_INCREMENTAL_SCOPE_HOPS = 1
_INCREMENTAL_SCOPE_LIMIT = 2000


class GraphEmbedPhase(ConsolidationPhase):
    """Train structural graph embeddings during consolidation.

    Checks which methods are enabled (node2vec, transe, gnn), trains each,
    and stores results in the graph_embeddings table.

    Runs between reindex and dream phases — after entity modifications are
    finalized but before dream spreading uses the embeddings.
    """

    @property
    def name(self) -> str:
        return "graph_embed"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        methods = {"find_entities", "get_active_neighbors_with_weights"}
        if cfg.graph_embedding_transe_enabled:
            methods.add("get_relationships")
        return methods

    async def execute(
        self,
        group_id: str,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[Any]]:
        t0 = time.perf_counter()

        # Check if any method is enabled
        methods_enabled = (
            cfg.graph_embedding_node2vec_enabled
            or cfg.graph_embedding_transe_enabled
            or cfg.graph_embedding_gnn_enabled
        )
        if not methods_enabled:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=0.0,
            ), []

        records: list[Any] = []
        total_trained = 0

        # Determine if full retrain is needed based on change threshold
        affected = context.affected_entity_ids if context else set()
        is_full_cycle = context is None or not context.trigger.startswith("tiered")
        if is_full_cycle:
            full_retrain = True
        else:
            total_entities = len(await graph_store.find_entities(group_id=group_id, limit=100000))
            change_ratio = len(affected) / max(total_entities, 1)
            full_retrain = change_ratio >= cfg.graph_embedding_retrain_threshold

        if not full_retrain and len(affected) == 0:
            # Nothing changed at all — skip entirely (tiered scheduling)
            elapsed = (time.perf_counter() - t0) * 1000
            return PhaseResult(
                phase=self.name,
                status="skipped",
                items_processed=0,
                items_affected=0,
                duration_ms=round(elapsed, 1),
            ), []

        # Stagger methods across cycles using cycle_id hash
        cycle_hash = hash(cycle_id) & 0xFFFFFFFF

        # Get db handle for storage
        db = None
        if hasattr(search_index, "_vectors") and hasattr(search_index._vectors, "db"):
            db = search_index._vectors.db
        elif hasattr(search_index, "db"):
            db = search_index.db

        if db is None and not dry_run:
            logger.warning(
                "GraphEmbedPhase: db=None and dry_run=False — "
                "trained embeddings will not be persisted"
            )

        store = GraphEmbeddingStore()
        if db is not None:
            await store.initialize(db)

        # Node2Vec
        if cfg.graph_embedding_node2vec_enabled:
            record = await self._train_method(
                "node2vec",
                graph_store,
                search_index,
                group_id,
                cfg,
                cycle_id,
                dry_run,
                full_retrain,
                store,
                db,
                affected,
            )
            if record:
                records.append(record)
                total_trained += record.entities_trained

        # TransE
        if cfg.graph_embedding_transe_enabled:
            if not full_retrain and (cycle_hash % cfg.graph_embedding_stagger_transe != 0):
                logger.info(
                    "GraphEmbedPhase: skipping TransE (stagger cycle %d mod %d)",
                    cycle_hash,
                    cfg.graph_embedding_stagger_transe,
                )
            else:
                record = await self._train_method(
                    "transe",
                    graph_store,
                    search_index,
                    group_id,
                    cfg,
                    cycle_id,
                    dry_run,
                    full_retrain,
                    store,
                    db,
                    affected,
                )
                if record:
                    records.append(record)
                    total_trained += record.entities_trained

        # GNN
        if cfg.graph_embedding_gnn_enabled:
            if not full_retrain and (cycle_hash % cfg.graph_embedding_stagger_gnn != 0):
                logger.info(
                    "GraphEmbedPhase: skipping GNN (stagger cycle %d mod %d)",
                    cycle_hash,
                    cfg.graph_embedding_stagger_gnn,
                )
            else:
                record = await self._train_method(
                    "gnn",
                    graph_store,
                    search_index,
                    group_id,
                    cfg,
                    cycle_id,
                    dry_run,
                    full_retrain,
                    store,
                    db,
                    affected,
                )
                if record:
                    records.append(record)
                    total_trained += record.entities_trained

        elapsed = (time.perf_counter() - t0) * 1000
        return PhaseResult(
            phase=self.name,
            status="success",
            items_processed=total_trained,
            items_affected=total_trained,
            duration_ms=round(elapsed, 1),
        ), records

    async def _train_method(
        self,
        method: str,
        graph_store,
        search_index,
        group_id: str,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool,
        full_retrain: bool,
        store: GraphEmbeddingStore,
        db,
        affected_ids: set[str],
    ) -> GraphEmbedRecord | None:
        """Train a single embedding method and store results."""
        t0 = time.perf_counter()

        try:
            trainer = _get_trainer(method, cfg)
            if trainer is None:
                return None

            # Get existing embeddings for warm-start (if not full retrain)
            existing = None
            if not full_retrain and db is not None:
                existing = await store.get_all_embeddings(db, method, group_id)

            method_full_retrain = full_retrain
            model_version = f"{method}_full_v2"

            # Train
            if method == "node2vec" and not full_retrain:
                if not dry_run and not existing:
                    logger.info(
                        "GraphEmbedPhase: %s has no baseline embeddings; "
                        "falling back to full retrain",
                        method,
                    )
                    method_full_retrain = True
                    model_version = f"{method}_full_v2"
                    embeddings = await trainer.train(
                        graph_store,
                        group_id,
                        existing_embeddings=existing,
                    )
                else:
                    dirty_scope = await _build_dirty_entity_scope(
                        graph_store,
                        group_id,
                        affected_ids,
                    )
                    embeddings = await trainer.train_incremental(
                        graph_store,
                        group_id,
                        dirty_scope,
                        existing_embeddings=existing,
                    )
                    method_full_retrain = False
                    model_version = f"{method}_delta_v2"
            elif method == "gnn":
                if not full_retrain:
                    logger.info(
                        "GraphEmbedPhase: %s incremental path unavailable, retraining full graph",
                        method,
                    )
                    method_full_retrain = True
                embeddings = await trainer.train(
                    graph_store,
                    group_id,
                    existing_embeddings=existing,
                    search_index=search_index,
                )
            else:
                if not full_retrain:
                    logger.info(
                        "GraphEmbedPhase: %s incremental path unavailable, retraining full graph",
                        method,
                    )
                    method_full_retrain = True
                embeddings = await trainer.train(
                    graph_store,
                    group_id,
                    existing_embeddings=existing,
                )

            if not embeddings:
                return None

            # Determine dimensions from first embedding
            first_vec = next(iter(embeddings.values()))
            dimensions = len(first_vec)

            # Store (unless dry_run)
            if not dry_run and db is None:
                logger.warning(
                    "GraphEmbedPhase: %s trained %d embeddings but db=None — discarding",
                    method,
                    len(embeddings),
                )
            if not dry_run and db is not None:
                await store.upsert_batch(
                    db,
                    embeddings,
                    method,
                    group_id,
                    model_version=model_version,
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            return GraphEmbedRecord(
                cycle_id=cycle_id,
                group_id=group_id,
                method=method,
                entities_trained=len(embeddings),
                dimensions=dimensions,
                training_duration_ms=round(elapsed_ms, 1),
                full_retrain=method_full_retrain,
            )

        except Exception as exc:
            logger.error("Graph embedding %s failed: %s", method, exc, exc_info=True)
            return None


def _get_trainer(method: str, cfg: ActivationConfig):
    """Get the trainer instance for a method."""
    if method == "node2vec":
        from engram.embeddings.graph.node2vec import Node2VecTrainer

        return Node2VecTrainer(cfg)
    elif method == "transe":
        from engram.embeddings.graph.transe import TransETrainer

        return TransETrainer(cfg)
    elif method == "gnn":
        from engram.embeddings.graph.gnn import GNNTrainer

        return GNNTrainer(cfg)
    return None


async def _build_dirty_entity_scope(
    graph_store,
    group_id: str,
    affected_ids: set[str],
) -> set[str]:
    """Expand affected entities into a bounded dirty subgraph scope."""
    if not affected_ids:
        return set()

    scope = set(affected_ids)
    frontier = set(affected_ids)
    for _ in range(_INCREMENTAL_SCOPE_HOPS):
        next_frontier: set[str] = set()
        for entity_id in frontier:
            if len(scope) >= _INCREMENTAL_SCOPE_LIMIT:
                return set(sorted(scope)[:_INCREMENTAL_SCOPE_LIMIT])
            try:
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    entity_id,
                    group_id=group_id,
                )
            except Exception:
                continue
            for neighbor_id, _weight, _predicate, *_rest in neighbors:
                if neighbor_id not in scope:
                    next_frontier.add(neighbor_id)
        scope.update(next_frontier)
        frontier = next_frontier
    if len(scope) > _INCREMENTAL_SCOPE_LIMIT:
        return set(sorted(scope)[:_INCREMENTAL_SCOPE_LIMIT])
    return scope
