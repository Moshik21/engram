"""Project bootstrap ingestion for artifact-backed project context."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.models.episode import EpisodeProjectionState, EpisodeStatus
from engram.models.epistemic import EvidenceClaim
from engram.retrieval.epistemic import artifact_class_for_path, extract_artifact_claims
from engram.storage.protocols import ActivationStore, GraphStore
from engram.utils.dates import utc_now, utc_now_iso

EventPublisher = Callable[[str, str, dict], None]
StoreEpisode = Callable[..., Awaitable[str]]
SyncProjectionState = Callable[..., Awaitable[None]]
EnsureRelationship = Callable[..., Awaitable[None]]
IndexEntity = Callable[[Entity, str], Awaitable[None]]
MaterializeArtifactDecisions = Callable[..., Awaitable[None]]


async def build_project_bootstrap_surface(
    manager: Any,
    *,
    group_id: str,
    project_path: str,
    include_patterns: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
    """Run project bootstrap through the shared manager compatibility facade."""
    result = await manager.bootstrap_project(
        project_path=project_path,
        group_id=group_id,
        include_patterns=include_patterns,
        session_id=session_id,
    )
    if result.get("status") not in {"skipped"}:
        runtime_service = getattr(manager, "_runtime_state_service", None)
        invalidate = getattr(runtime_service, "invalidate_cache", None)
        if callable(invalidate):
            invalidate(group_id=group_id, project_path=project_path)
    return result


def project_bootstrap_http_status(result: dict) -> int:
    """Map bootstrap result status to the REST endpoint status code."""
    return 200 if result.get("status") != "skipped" else 400


class ProjectBootstrapService:
    """Bootstrap project files into artifact entities and cue-only episodes."""

    EXPLICIT_SOURCE_MAX_CHARS = 6000

    BOOTSTRAP_FILES: list[tuple[str, int]] = [
        ("README.md", 2000),
        ("MEMORY.md", 4000),
        ("package.json", 3000),
        ("pyproject.toml", 3000),
        ("Makefile", 3000),
        (".env.example", 2000),
        ("docker-compose.yml", 3000),
        ("CLAUDE.md", 2500),
        ("docs/**/*.md", 4000),
        ("docs/design/**/*.md", 4000),
        ("docs/vision/**/*.md", 4000),
        ("notes/**/*.md", 4000),
        ("memory/**/*.md", 4000),
        ("memory/**/*.json", 6000),
        ("memories/**/*.md", 4000),
        ("memories/**/*.json", 6000),
        ("exports/**/*.md", 4000),
        ("exports/**/*.json", 6000),
        ("skills/**/SKILL.md", 3500),
    ]

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cfg: ActivationConfig,
        publish_event: EventPublisher,
        store_episode: StoreEpisode,
        sync_projection_state: SyncProjectionState,
        ensure_relationship: EnsureRelationship,
        index_entity: IndexEntity,
        materialize_artifact_decisions: MaterializeArtifactDecisions,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._publish = publish_event
        self._store_episode = store_episode
        self._sync_projection_state = sync_projection_state
        self._ensure_relationship = ensure_relationship
        self._index_entity = index_entity
        self._materialize_artifact_decisions = materialize_artifact_decisions

    async def bootstrap_project(
        self,
        project_path: str,
        group_id: str = "default",
        include_patterns: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Create or refresh Project/Artifact graph context for a local project path."""
        project_dir = Path(project_path).expanduser()
        try:
            project_dir = project_dir.resolve()
        except OSError:
            pass
        project_path = str(project_dir)
        project_name = project_dir.name
        if not project_name or str(project_dir) in (str(Path.home()), "/"):
            return {"status": "skipped", "reason": "invalid_path"}

        now_iso = utc_now_iso()
        existing = await self._graph.find_entities(
            name=project_name,
            entity_type="Project",
            group_id=group_id,
            limit=1,
        )

        if existing:
            entity = existing[0]
            entity_id = entity.id
            attrs = entity.attributes or {}
            last_bootstrapped = attrs.get("last_bootstrapped")

            if last_bootstrapped:
                try:
                    last_dt = datetime.fromisoformat(last_bootstrapped)
                    age_seconds = (utc_now() - last_dt).total_seconds()
                    if (
                        not include_patterns
                        and age_seconds < self._cfg.artifact_bootstrap_stale_seconds
                    ):
                        return {
                            "status": "already_bootstrapped",
                            "project_entity_id": entity_id,
                        }
                except (ValueError, TypeError):
                    pass

            files_observed = await self.observe_project_files(
                project_dir,
                project_name,
                group_id,
                session_id,
                include_patterns=include_patterns,
            )

            merged_attrs = {**attrs, "last_bootstrapped": now_iso}
            await self._graph.update_entity(
                entity_id,
                {"attributes": json.dumps(merged_attrs)},
                group_id=group_id,
            )
            await self._activation.record_access(
                entity_id,
                time.time(),
                group_id=group_id,
            )

            self._publish(
                group_id,
                "project.refreshed",
                {
                    "project_name": project_name,
                    "project_entity_id": entity_id,
                    "files_observed": files_observed,
                },
            )

            return {
                "status": "refreshed",
                "project_entity_id": entity_id,
                "files_observed": files_observed,
            }

        entity_id = f"ent_{uuid.uuid4().hex[:12]}"
        entity = Entity(
            id=entity_id,
            name=project_name,
            entity_type="Project",
            summary=f"Software project at {project_path}",
            attributes={
                "project_path": str(project_dir),
                "last_bootstrapped": now_iso,
            },
            group_id=group_id,
        )
        await self._graph.create_entity(entity)
        await self._index_entity(entity, group_id)
        await self._activation.record_access(
            entity_id,
            time.time(),
            group_id=group_id,
        )

        files_observed = await self.observe_project_files(
            project_dir,
            project_name,
            group_id,
            session_id,
            include_patterns=include_patterns,
        )

        self._publish(
            group_id,
            "project.bootstrapped",
            {
                "project_name": project_name,
                "project_entity_id": entity_id,
                "files_observed": files_observed,
            },
        )

        return {
            "status": "bootstrapped",
            "project_entity_id": entity_id,
            "files_observed": files_observed,
        }

    async def observe_project_files(
        self,
        project_dir: Path,
        project_name: str,
        group_id: str,
        session_id: str | None,
        include_patterns: list[str] | None = None,
    ) -> list[str]:
        """Read, index, and optionally store bootstrapped project artifacts."""
        files_observed: list[str] = []
        project_path = str(project_dir)
        project_entity_id = await self.resolve_project_entity_id(project_name, group_id)
        now_iso = utc_now_iso()
        seen_rel_paths: set[str] = set()

        for filepath, rel_path, max_chars in self.iter_bootstrap_files(
            project_dir,
            include_patterns=include_patterns,
        ):
            if rel_path in seen_rel_paths:
                continue
            seen_rel_paths.add(rel_path)
            try:
                raw_content = filepath.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            truncated = raw_content[:max_chars]
            artifact_class = artifact_class_for_path(rel_path)
            content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            claims = extract_artifact_claims(
                truncated,
                rel_path=rel_path,
                artifact_class=artifact_class,
                project_name=project_name,
                timestamp=now_iso,
            )
            artifact_entity, changed = await self.upsert_artifact_entity(
                project_name=project_name,
                project_path=project_path,
                source_path=str(filepath),
                rel_path=rel_path,
                artifact_class=artifact_class,
                content=truncated,
                content_hash=content_hash,
                claims=claims,
                group_id=group_id,
                now_iso=now_iso,
            )
            if project_entity_id is not None and changed:
                await self._ensure_relationship(
                    artifact_entity.id,
                    project_entity_id,
                    "PART_OF",
                    group_id=group_id,
                )
            if changed:
                episode_id = await self._store_bootstrap_episode(
                    project_name=project_name,
                    rel_path=rel_path,
                    content=truncated,
                    group_id=group_id,
                    session_id=session_id,
                )
                await self._graph.update_entity(
                    artifact_entity.id,
                    {
                        "attributes": json.dumps(
                            self.merge_attributes(
                                artifact_entity.attributes,
                                {"last_episode_id": episode_id},
                            )
                        )
                    },
                    group_id=group_id,
                )
            if self._cfg.decision_graph_enabled and claims and changed:
                await self._materialize_artifact_decisions(
                    artifact_entity,
                    claims,
                    group_id=group_id,
                )
            files_observed.append(rel_path)
        return files_observed

    def iter_bootstrap_files(
        self,
        project_dir: Path,
        *,
        include_patterns: list[str] | None = None,
    ) -> list[tuple[Path, str, int]]:
        """Expand bootstrap patterns into concrete files."""
        matches: list[tuple[Path, str, int]] = []
        patterns = list(self.BOOTSTRAP_FILES)
        patterns.extend(
            (pattern, self.EXPLICIT_SOURCE_MAX_CHARS)
            for pattern in self.normalize_include_patterns(include_patterns)
        )
        for pattern, max_chars in patterns:
            for filepath in sorted(project_dir.glob(pattern)):
                if not filepath.is_file():
                    continue
                try:
                    rel_path = filepath.relative_to(project_dir).as_posix()
                except ValueError:
                    continue
                matches.append((filepath, rel_path, max_chars))
        return matches

    @staticmethod
    def normalize_include_patterns(include_patterns: list[str] | None) -> list[str]:
        """Return safe project-relative user-approved bootstrap globs."""
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_pattern in include_patterns or []:
            pattern = raw_pattern.strip()
            while pattern.startswith("./"):
                pattern = pattern[2:]
            if not pattern:
                continue
            parsed = PurePosixPath(pattern)
            if parsed.is_absolute() or ".." in parsed.parts:
                continue
            if pattern not in seen:
                normalized.append(pattern)
                seen.add(pattern)
        return normalized

    async def resolve_project_entity_id(
        self,
        project_name: str,
        group_id: str,
    ) -> str | None:
        existing = await self._graph.find_entities(
            name=project_name,
            entity_type="Project",
            group_id=group_id,
            limit=1,
        )
        if existing:
            return existing[0].id
        return None

    async def upsert_artifact_entity(
        self,
        *,
        project_name: str,
        project_path: str,
        rel_path: str,
        artifact_class: str,
        content: str,
        content_hash: str,
        claims: list[EvidenceClaim],
        group_id: str,
        now_iso: str,
        source_path: str | None = None,
    ) -> tuple[Entity, bool]:
        """Create or update a bootstrapped Artifact entity."""
        artifact_key = f"{group_id}:{project_path}:{rel_path}"
        artifact_id = f"art_{hashlib.sha256(artifact_key.encode()).hexdigest()[:12]}"
        attributes = {
            "project_path": project_path,
            "source_path": source_path or str(Path(project_path) / rel_path),
            "rel_path": rel_path,
            "artifact_class": artifact_class,
            "content_hash": content_hash,
            "last_observed_at": now_iso,
            "stale_after": self._cfg.artifact_bootstrap_stale_seconds,
            "snippet": self.artifact_snippet(content),
            "claims": [self.claim_to_attr(claim) for claim in claims],
        }
        entity = await self._graph.get_entity(artifact_id, group_id)
        if not isinstance(entity, Entity):
            entity = None
        if entity is None:
            entity = Entity(
                id=artifact_id,
                name=rel_path,
                entity_type="Artifact",
                summary=self.artifact_summary(project_name, rel_path, content, claims),
                attributes=attributes,
                group_id=group_id,
            )
            await self._graph.create_entity(entity)
            await self._index_entity(entity, group_id)
            await self._activation.record_access(entity.id, time.time(), group_id=group_id)
            return entity, True

        current_hash = (entity.attributes or {}).get("content_hash")
        changed = current_hash != content_hash
        merged_attrs = self.merge_attributes(entity.attributes, attributes)
        updates: dict[str, object] = {"attributes": json.dumps(merged_attrs)}
        if changed:
            updates["summary"] = self.artifact_summary(project_name, rel_path, content, claims)
        await self._graph.update_entity(entity.id, updates, group_id=group_id)
        entity.attributes = merged_attrs
        if changed:
            entity.summary = str(updates["summary"])
            await self._index_entity(entity, group_id)
        return entity, changed

    async def _store_bootstrap_episode(
        self,
        *,
        project_name: str,
        rel_path: str,
        content: str,
        group_id: str,
        session_id: str | None,
    ) -> str:
        tagged = f"[project-bootstrap|{project_name}|{rel_path}]\n{content}"
        episode_id = await self._store_episode(
            content=tagged,
            group_id=group_id,
            source="auto:bootstrap",
            session_id=session_id,
        )
        await self._sync_projection_state(
            episode_id,
            EpisodeProjectionState.CUE_ONLY,
            group_id=group_id,
            reason="project_bootstrap_artifact",
            episode_updates={
                "status": EpisodeStatus.COMPLETED.value,
            },
            cue_reason="project_bootstrap_artifact",
        )
        return episode_id

    @staticmethod
    def claim_to_attr(claim: EvidenceClaim) -> dict:
        return {
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "source_type": claim.source_type,
            "authority_type": claim.authority_type,
            "externalization_state": claim.externalization_state,
            "timestamp": claim.timestamp,
            "confidence": claim.confidence,
            "provenance": claim.provenance,
        }

    @staticmethod
    def merge_attributes(existing: dict | None, updates: dict) -> dict:
        merged = dict(existing or {})
        merged.update(updates)
        return merged

    @staticmethod
    def artifact_snippet(content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:240]
        return content[:240]

    @staticmethod
    def artifact_summary(
        project_name: str,
        rel_path: str,
        content: str,
        claims: list[EvidenceClaim],
    ) -> str:
        summary_parts = [f"{project_name} artifact {rel_path}"]
        if claims:
            summary_parts.append(
                "; ".join(f"{claim.predicate}={claim.object}" for claim in claims[:3])
            )
        else:
            summary_parts.append(ProjectBootstrapService.artifact_snippet(content))
        return " — ".join(part for part in summary_parts if part)[:500]
