"""Tests for artifact_class filtering on the file-search surface (M2.4)."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.models.entity import Entity
from engram.retrieval.artifacts import ArtifactSearchService
from engram.utils.dates import utc_now


class FakeArtifactGraph:
    def __init__(self, artifacts: list[Entity]) -> None:
        self.artifacts = artifacts

    async def find_entities(self, **kwargs):
        if kwargs.get("entity_type") == "Artifact":
            return list(self.artifacts)
        return []

    async def get_entity(self, entity_id: str, _group_id: str):
        for artifact in self.artifacts:
            if artifact.id == entity_id:
                return artifact
        return None


class EmptySearchIndex:
    async def search(self, **_kwargs):
        return []


def _artifact(entity_id: str, name: str, attributes: dict) -> Entity:
    return Entity(
        id=entity_id,
        name=name,
        entity_type="Artifact",
        summary=f"Summary for {name}",
        attributes=attributes,
        group_id="test",
    )


def _service(artifacts: list[Entity]) -> ArtifactSearchService:
    async def bootstrap_project(path: str, *, group_id: str):
        return {"status": "already_bootstrapped"}

    return ArtifactSearchService(
        graph_store=FakeArtifactGraph(artifacts),
        search_index=EmptySearchIndex(),
        cfg=ActivationConfig(artifact_bootstrap_enabled=False),
        bootstrap_project=bootstrap_project,
    )


async def test_search_artifacts_excludes_conversation_records():
    file_artifact = _artifact(
        "art_readme",
        "README.md",
        {
            "rel_path": "README.md",
            "artifact_class": "readme",
            "snippet": "Engram launch plan",
            "last_observed_at": utc_now().isoformat(),
            "stale_after": 3600,
        },
    )
    conversation = _artifact(
        "art_conv_1",
        "conversation:ep_1",
        {
            "artifact_class": "conversation_record",
            "source_episode": "ep_1",
            "snippet": "Engram launch plan discussed in conversation",
            "last_observed_at": utc_now().isoformat(),
            "stale_after": 3600,
        },
    )
    service = _service([conversation, file_artifact])

    hits = await service.search_artifacts(query="Engram launch", group_id="test")

    assert [hit.artifact_id for hit in hits] == ["art_readme"]


async def test_search_artifacts_treats_missing_class_as_file():
    legacy = _artifact(
        "art_legacy",
        "docs/plan.md",
        {
            "rel_path": "docs/plan.md",
            "snippet": "Engram launch plan",
            "last_observed_at": utc_now().isoformat(),
            "stale_after": 3600,
        },
    )
    service = _service([legacy])

    hits = await service.search_artifacts(query="Engram launch", group_id="test")

    assert [hit.artifact_id for hit in hits] == ["art_legacy"]
