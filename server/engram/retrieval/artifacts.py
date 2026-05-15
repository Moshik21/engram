"""Artifact search/read services for project-backed epistemic memory."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import datetime

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.epistemic import ArtifactHit, EvidenceClaim
from engram.storage.protocols import GraphStore, SearchIndex
from engram.utils.dates import utc_now

ProjectBootstrapper = Callable[..., Awaitable[dict]]


class ArtifactSearchService:
    """Search bootstrapped project artifacts and format artifact hits."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
        bootstrap_project: ProjectBootstrapper,
    ) -> None:
        self._graph = graph_store
        self._search = search_index
        self._cfg = cfg
        self._bootstrap_project = bootstrap_project

    async def list_project_artifacts(
        self,
        *,
        group_id: str,
        project_path: str | None = None,
        limit: int = 200,
    ) -> list[Entity]:
        artifacts = await self._graph.find_entities(
            entity_type="Artifact",
            group_id=group_id,
            limit=limit,
        )
        if project_path is None:
            return artifacts
        return [
            entity
            for entity in artifacts
            if (entity.attributes or {}).get("project_path") == project_path
        ]

    async def search_artifacts(
        self,
        *,
        query: str,
        group_id: str = "default",
        project_path: str | None = None,
        limit: int = 5,
    ) -> list[ArtifactHit]:
        """Search bootstrapped project artifacts by semantic or lexical match."""
        if project_path and self._cfg.artifact_bootstrap_enabled:
            await self._bootstrap_project(project_path, group_id=group_id)

        hits: list[ArtifactHit] = []
        seen_ids: set[str] = set()
        scored_ids = await self._search.search(
            query=query,
            entity_types=["Artifact"],
            group_id=group_id,
            limit=max(limit * 4, 10),
        )
        if not scored_ids:
            fallback = await self._graph.find_entities(
                name=query,
                entity_type="Artifact",
                group_id=group_id,
                limit=max(limit * 2, 10),
            )
            scored_ids = [(entity.id, 0.4) for entity in fallback]
        if project_path or not scored_ids:
            artifacts = await self.list_project_artifacts(
                group_id=group_id,
                project_path=project_path,
                limit=max(limit * 4, 50),
            )
            existing_ids = {entity_id for entity_id, _score in scored_ids}
            lexical_scores = [
                (entity.id, score)
                for entity in artifacts
                if entity.id not in existing_ids
                and (score := self.artifact_lexical_score(query, entity)) > 0
            ]
            scored_ids.extend(lexical_scores)
            scored_ids.sort(key=lambda item: item[1], reverse=True)

        for entity_id, score in scored_ids:
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            entity = await self._graph.get_entity(entity_id, group_id)
            if entity is None:
                continue
            attrs = entity.attributes or {}
            if project_path and attrs.get("project_path") != project_path:
                continue
            claims = [
                self.claim_from_attr(claim_data)
                for claim_data in attrs.get("claims", [])[:4]
                if isinstance(claim_data, dict)
            ]
            hits.append(
                ArtifactHit(
                    artifact_id=entity.id,
                    path=str(attrs.get("rel_path") or entity.name),
                    artifact_class=str(attrs.get("artifact_class") or "artifact"),
                    snippet=str(attrs.get("snippet") or entity.summary or ""),
                    last_observed_at=attrs.get("last_observed_at"),
                    score=score,
                    stale=self.artifact_is_stale(
                        entity,
                        int(attrs.get("stale_after") or self._cfg.artifact_bootstrap_stale_seconds),
                    ),
                    supporting_claims=claims,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    @staticmethod
    def artifact_is_stale(entity: Entity, stale_seconds: int) -> bool:
        attrs = entity.attributes or {}
        last_observed = attrs.get("last_observed_at")
        if not last_observed:
            return True
        try:
            observed_dt = datetime.fromisoformat(last_observed)
        except (TypeError, ValueError):
            return True
        return (utc_now() - observed_dt).total_seconds() >= stale_seconds

    @staticmethod
    def artifact_lexical_score(query: str, entity: Entity) -> float:
        terms = [term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 1]
        if not terms:
            return 0.0

        attrs = entity.attributes or {}
        claims = attrs.get("claims") if isinstance(attrs.get("claims"), list) else []
        claim_text = " ".join(
            str(claim.get(key, ""))
            for claim in claims
            if isinstance(claim, dict)
            for key in ("subject", "predicate", "object")
        )
        haystack = " ".join(
            str(part or "")
            for part in (
                entity.name,
                entity.summary,
                attrs.get("rel_path"),
                attrs.get("artifact_class"),
                attrs.get("snippet"),
                claim_text,
            )
        ).lower()
        if not haystack:
            return 0.0

        matched = sum(1 for term in terms if term in haystack)
        if matched == 0:
            return 0.0
        return min(1.0, matched / len(terms))

    @staticmethod
    def claim_from_attr(claim_data: dict) -> EvidenceClaim:
        return EvidenceClaim(
            subject=str(claim_data.get("subject", "")),
            predicate=str(claim_data.get("predicate", "")),
            object=str(claim_data.get("object", "")),
            source_type=str(
                claim_data.get("source_type") or claim_data.get("sourceType") or "artifact"
            ),
            authority_type=str(
                claim_data.get("authority_type") or claim_data.get("authorityType") or "canonical"
            ),
            externalization_state=str(
                claim_data.get("externalization_state")
                or claim_data.get("externalizationState")
                or "documented"
            ),
            claim_state=str(
                claim_data.get("claim_state") or claim_data.get("claimState") or "mentioned"
            ),
            timestamp=claim_data.get("timestamp"),
            confidence=float(claim_data.get("confidence", 0.0) or 0.0),
            provenance=dict(claim_data.get("provenance") or {}),
        )
