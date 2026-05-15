"""Tests for artifact search/read service behavior."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.artifacts import ArtifactSearchService
from engram.utils.dates import utc_now


class FakeArtifactGraph:
    def __init__(self, artifact: Entity) -> None:
        self.artifact = artifact

    async def find_entities(self, **kwargs):
        if kwargs.get("entity_type") == "Artifact":
            return [self.artifact]
        return []

    async def get_entity(self, entity_id: str, _group_id: str):
        if entity_id == self.artifact.id:
            return self.artifact
        return None


class EmptySearchIndex:
    async def search(self, **_kwargs):
        return []


async def test_artifact_search_service_uses_lexical_claim_fallback_and_bootstrap():
    project_path = "/tmp/engram-artifact-service"
    artifact = Entity(
        id="art_readme",
        name="README.md",
        entity_type="Artifact",
        summary="Project readme",
        attributes={
            "project_path": project_path,
            "rel_path": "README.md",
            "artifact_class": "readme",
            "snippet": "Launch path is OpenClaw",
            "claims": [
                {
                    "subject": "Engram",
                    "predicate": "public_launch_path",
                    "object": "OpenClaw",
                    "confidence": 0.9,
                }
            ],
            "last_observed_at": utc_now().isoformat(),
            "stale_after": 3600,
        },
        group_id="test",
    )
    bootstrapped: list[tuple[str, str]] = []

    async def bootstrap_project(path: str, *, group_id: str):
        bootstrapped.append((path, group_id))
        return {"status": "already_bootstrapped"}

    service = ArtifactSearchService(
        graph_store=FakeArtifactGraph(artifact),
        search_index=EmptySearchIndex(),
        cfg=ActivationConfig(artifact_bootstrap_enabled=True),
        bootstrap_project=bootstrap_project,
    )

    hits = await service.search_artifacts(
        query="OpenClaw launch",
        group_id="test",
        project_path=project_path,
    )

    assert bootstrapped == [(project_path, "test")]
    assert len(hits) == 1
    assert hits[0].path == "README.md"
    assert hits[0].artifact_class == "readme"
    assert hits[0].stale is False
    assert hits[0].supporting_claims[0].object == "OpenClaw"
