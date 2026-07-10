"""Reflect phase: cluster related episodes and synthesize durable observations.

WRITE-SIDE OBSERVER. Offline, importance-gated phase that clusters raw episodes
by shared graph entities and synthesizes ONE durable, fact-dense "observation"
episode per cluster (``memory_tier="observation"``, ``source="observer:reflector"``).
The observation is a standard Episode embedded into the existing EpisodeVec/FTS5
store, so it rides the EXISTING episode retrieval path and introduces NO new
retrieval surface.

SHIP-DARK GUARD: ``execute()`` returns ``(PhaseResult(status="skipped"), [])`` as
its first statement when ``observer_reflect_enabled`` is False — byte-for-byte
mirroring ``SchemaFormationPhase``. With the flag OFF the phase makes zero reads,
zero writes, zero ``index_episode`` calls, and emits zero records, so the system
is byte-identical to today. The synthesizer modules are imported lazily inside
the enabled branch, so an OFF runtime never constructs an LLM client.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, ObservationRecord, PhaseResult
from engram.models.episode import (
    Episode,
    EpisodeProjectionState,
    EpisodeStatus,
)
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)

_OBSERVER_SOURCE = "observer:reflector"
_CLUSTER_KEY_PREFIX = "reflect_cluster:"


def cluster_key(episode_ids: tuple[str, ...]) -> str:
    """Stable key identifying a cluster by its sorted member episode ids.

    Used for idempotency: a cluster whose key already has an observation child is
    not re-synthesized on a later cold-tier re-run.
    """
    joined = ",".join(sorted(episode_ids))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{_CLUSTER_KEY_PREFIX}{digest}"


class ObserverReflectPhase(ConsolidationPhase):
    """Synthesize cross-episode observations from entity-clustered episodes."""

    def __init__(
        self,
        *,
        extractor: object | None = None,
        llm_client: object | None = None,
    ) -> None:
        self._extractor = extractor
        self._llm_client = llm_client

    @property
    def name(self) -> str:
        return "reflect"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if not cfg.observer_reflect_enabled:
            return set()
        return {
            "get_episodes",
            "get_episode_entities",
            "get_entity",
            "get_relationships",
            "create_episode",
            "link_episode_entity",
        }

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

        # --- SHIP-DARK GUARD (mirror of SchemaFormationPhase) --------------- #
        if not cfg.observer_reflect_enabled:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                duration_ms=_elapsed_ms(t0),
            ), []

        # Lazy imports keep the OFF runtime from touching synthesis/LLM code.
        from engram.consolidation.observer.clustering import cluster_episodes_by_entity
        from engram.consolidation.observer.importance import cluster_importance
        from engram.consolidation.observer.synthesizer import select_synthesizer

        episodes: list[Episode] = await graph_store.get_episodes(
            group_id=group_id,
            limit=cfg.observer_reflect_max_episodes_scan,
        )
        # Only synthesize from real source episodes — never from prior
        # observations (idempotency / no recursive synthesis) and only COMPLETED.
        candidates = [
            ep
            for ep in episodes
            if ep.source != _OBSERVER_SOURCE and _status_value(ep.status) == "completed"
        ]
        if not candidates:
            return PhaseResult(
                phase=self.name,
                status="skipped",
                items_processed=0,
                duration_ms=_elapsed_ms(t0),
            ), []

        episode_entities: dict[str, list[str]] = {}
        for ep in candidates:
            try:
                ent_ids = await graph_store.get_episode_entities(ep.id, group_id=group_id)
            except Exception:
                ent_ids = []
            episode_entities[ep.id] = list(ent_ids or [])

        clusters = cluster_episodes_by_entity(
            candidates,
            episode_entities,
            min_cluster_size=cfg.observer_reflect_min_cluster,
        )

        # Idempotency: keys of clusters that already have an observation child.
        existing_keys = _existing_cluster_keys(episodes)

        synthesizer = select_synthesizer(
            llm_enabled=cfg.observer_reflect_llm_enabled,
            extractor=self._extractor,
            llm_client=self._llm_client,
        )

        episode_by_id = {ep.id: ep for ep in candidates}
        records: list[ObservationRecord] = []
        created = 0

        for cluster in clusters:
            if created >= cfg.observer_reflect_max_observations_per_cycle:
                break

            base_key = cluster_key(cluster.episode_ids)

            cluster_eps = [episode_by_id[eid] for eid in cluster.episode_ids]
            importance = cluster_importance([ep.content for ep in cluster_eps])

            if importance < cfg.observer_reflect_min_importance:
                records.append(
                    ObservationRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        observation_episode_id="",
                        cluster_episode_ids=list(cluster.episode_ids),
                        cluster_size=len(cluster.episode_ids),
                        importance=round(importance, 4),
                        synthesizer=synthesizer.name,
                        action="skipped_low_importance",
                    )
                )
                continue

            # Resolve cluster entities + their relationships for the synthesizer.
            entities = []
            for ent_id in cluster.entity_ids:
                try:
                    ent = await graph_store.get_entity(ent_id, group_id)
                except Exception:
                    ent = None
                if ent is not None and ent.deleted_at is None:
                    entities.append(ent)

            cluster_entity_ids = {e.id for e in entities}
            relationships = []
            seen_rel_ids: set[str] = set()
            for ent in entities:
                try:
                    rels = await graph_store.get_relationships(
                        ent.id,
                        direction="both",
                        group_id=group_id,
                    )
                except Exception:
                    rels = []
                for rel in rels:
                    if rel.id in seen_rel_ids:
                        continue
                    if rel.source_id in cluster_entity_ids and rel.target_id in cluster_entity_ids:
                        seen_rel_ids.add(rel.id)
                        relationships.append(rel)

            # FOCUSED synthesis returns one dense observation per subject entity.
            # Each is persisted as its own observation episode with a per-content
            # idempotency key so a later cold-tier re-run does not duplicate it.
            newest = _max_conversation_date(cluster_eps)
            for content in synthesizer.synthesize(cluster_eps, entities, relationships):
                if created >= cfg.observer_reflect_max_observations_per_cycle:
                    break
                obs_key = f"{base_key}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]}"
                if obs_key in existing_keys:
                    continue  # already synthesized — do not duplicate
                existing_keys.add(obs_key)
                obs_id = f"ep_{_obs_uuid()}"
                observation = Episode(
                    id=obs_id,
                    content=content,
                    source=_OBSERVER_SOURCE,
                    status=EpisodeStatus.COMPLETED,
                    group_id=group_id,
                    conversation_date=newest,
                    memory_tier="observation",
                    projection_state=EpisodeProjectionState.PROJECTED,
                    encoding_context=obs_key,
                )

                if not dry_run:
                    await graph_store.create_episode(observation)
                    # REQUIRED for vector retrievability — create_episode does NOT
                    # embed. Treat an embed failure as non-fatal (FTS5/BM25 still
                    # index it via the same call on most backends).
                    try:
                        await search_index.index_episode(observation)
                    except Exception:
                        logger.warning(
                            "reflect: index_episode failed for %s; observation rides FTS5 only",
                            obs_id,
                            exc_info=True,
                        )
                    # Entity-link so entity-seeded traversal can reach the observation.
                    for ent in entities:
                        try:
                            await graph_store.link_episode_entity(obs_id, ent.id, group_id=group_id)
                        except Exception:
                            logger.debug("reflect: link_episode_entity failed", exc_info=True)
                    if context is not None:
                        context.observation_episode_ids.add(obs_id)

                records.append(
                    ObservationRecord(
                        cycle_id=cycle_id,
                        group_id=group_id,
                        observation_episode_id=obs_id,
                        cluster_episode_ids=list(cluster.episode_ids),
                        cluster_size=len(cluster.episode_ids),
                        importance=round(importance, 4),
                        synthesizer=synthesizer.name,
                        action="created",
                    )
                )
                created += 1

        return PhaseResult(
            phase=self.name,
            items_processed=len(candidates),
            items_affected=created,
            duration_ms=_elapsed_ms(t0),
        ), records


def _existing_cluster_keys(episodes: list[Episode]) -> set[str]:
    keys: set[str] = set()
    for ep in episodes:
        if ep.source == _OBSERVER_SOURCE and ep.encoding_context:
            if ep.encoding_context.startswith(_CLUSTER_KEY_PREFIX):
                keys.add(ep.encoding_context)
    return keys


def _status_value(status: Any) -> str:
    return status.value if hasattr(status, "value") else str(status)


def _max_conversation_date(episodes: list[Episode]):
    dates = [ep.conversation_date for ep in episodes if ep.conversation_date is not None]
    if not dates:
        return utc_now()
    return max(dates)


def _obs_uuid() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
