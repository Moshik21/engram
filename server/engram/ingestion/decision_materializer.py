"""Decision graph materialization for captured conversations and artifacts."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Awaitable, Callable

from engram.config import DEFAULT_DECISION_VOCABULARY, ActivationConfig
from engram.models.entity import Entity
from engram.models.epistemic import EvidenceClaim
from engram.models.relationship import Relationship
from engram.retrieval.epistemic import (
    extract_decision_claims,
    infer_claim_state,
    should_materialize_conversation_decision,
)
from engram.storage.protocols import GraphStore
from engram.utils.dates import utc_now_iso

EntityIndexer = Callable[[Entity, str], Awaitable[None]]

_SUBJECT_TERMS_PREFIX = "subject_terms:"
_DECISION_NAME_MAX_CHARS = 96


class DecisionMaterializer:
    """Materialize conversation and artifact decisions into graph entities."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        cfg: ActivationConfig,
        index_entity: EntityIndexer,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._index_entity = index_entity

    async def materialize_artifact_decisions(
        self,
        artifact_entity: Entity,
        claims: list[EvidenceClaim],
        *,
        group_id: str,
    ) -> None:
        attrs = artifact_entity.attributes or {}
        artifact_class = str(attrs.get("artifact_class") or "design_doc")
        link_predicate = "DOCUMENTED_IN"
        if artifact_class == "config":
            link_predicate = "IMPLEMENTED_BY"
        elif artifact_class in {"readme", "skill"}:
            link_predicate = "ANNOUNCED_AS"
        for claim in claims:
            if not self.is_decision_claim(claim, self._vocabulary()):
                continue
            decision = await self.upsert_decision_entity(claim, group_id=group_id)
            await self.ensure_relationship(
                decision.id,
                artifact_entity.id,
                link_predicate,
                group_id=group_id,
            )

    async def materialize_conversation_decisions(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> None:
        subject = self.infer_decision_subject(content, self._vocabulary())
        if subject is None:
            return
        claims: list[EvidenceClaim] = []
        for chunk in re.split(r"[\n.!?]+", content):
            if not should_materialize_conversation_decision(chunk):
                continue
            claims.extend(
                extract_decision_claims(
                    chunk,
                    subject=subject,
                    source_type="memory",
                    authority_type="historical",
                    externalization_state="discussed",
                    provenance={"episode_id": episode_id},
                )
            )
        filtered_claims: list[EvidenceClaim] = []
        for claim in claims:
            if not self.is_decision_claim(claim, self._vocabulary()):
                continue
            claim.claim_state = infer_claim_state(claim)
            if claim.claim_state != "decided":
                continue
            filtered_claims.append(claim)
        if not filtered_claims:
            return

        artifact = await self.upsert_conversation_artifact(
            content,
            episode_id=episode_id,
            group_id=group_id,
        )
        for claim in filtered_claims:
            decision = await self.upsert_decision_entity(claim, group_id=group_id)
            await self.ensure_relationship(
                decision.id,
                artifact.id,
                "DECIDED_IN",
                group_id=group_id,
                source_episode=episode_id,
            )

    async def upsert_conversation_artifact(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> Entity:
        artifact_id = f"art_conv_{episode_id.split('_')[-1]}"
        existing = await self._graph.get_entity(artifact_id, group_id)
        if existing is not None:
            return existing
        artifact = Entity(
            id=artifact_id,
            name=f"conversation:{episode_id}",
            entity_type="Artifact",
            summary=f"Conversation record for decision provenance: {content[:180]}",
            attributes={
                "artifact_class": "conversation_record",
                "source_episode": episode_id,
                "snippet": content[:240],
                "last_observed_at": utc_now_iso(),
                "stale_after": self._cfg.artifact_bootstrap_stale_seconds,
            },
            group_id=group_id,
        )
        await self._graph.create_entity(artifact)
        await self._index_entity(artifact, group_id)
        return artifact

    async def upsert_decision_entity(
        self,
        claim: EvidenceClaim,
        *,
        group_id: str,
    ) -> Entity:
        # Dedup key is the (subject, predicate, object) attribute triple, not the
        # name: names are human-readable display strings, and legacy rows still
        # carry the old '{subject}:{predicate}:{object}' shape.
        existing = [
            candidate
            for candidate in await self._graph.find_entities(
                entity_type="Decision",
                group_id=group_id,
                limit=200,
            )
            if (candidate.attributes or {}).get("subject") == claim.subject
            and (candidate.attributes or {}).get("canonical_predicate") == claim.predicate
        ]
        for candidate in existing:
            attrs = candidate.attributes or {}
            if attrs.get("decision_object") == claim.object:
                merged_attrs = self.merge_attributes(
                    attrs,
                    {
                        "last_seen_at": utc_now_iso(),
                        "authority_type": claim.authority_type,
                        "externalization_state": claim.externalization_state,
                        "source_type": claim.source_type,
                    },
                )
                await self._graph.update_entity(
                    candidate.id,
                    {"attributes": json.dumps(merged_attrs)},
                    group_id=group_id,
                )
                candidate.attributes = merged_attrs
                return candidate

        readable = self.compose_decision_text(claim)
        decision = Entity(
            id=f"dec_{uuid.uuid4().hex[:12]}",
            name=self._truncate_decision_name(readable),
            entity_type="Decision",
            summary=readable[:500],
            attributes={
                "subject": claim.subject,
                "canonical_predicate": claim.predicate,
                "decision_object": claim.object,
                "authority_type": claim.authority_type,
                "externalization_state": claim.externalization_state,
                "source_type": claim.source_type,
                "last_seen_at": utc_now_iso(),
            },
            group_id=group_id,
        )
        await self._graph.create_entity(decision)
        await self._index_entity(decision, group_id)
        for candidate in existing:
            attrs = candidate.attributes or {}
            if attrs.get("decision_object") != claim.object:
                await self.ensure_relationship(
                    candidate.id,
                    decision.id,
                    "SUPERSEDED_BY",
                    group_id=group_id,
                )
        return decision

    async def ensure_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        group_id: str,
        source_episode: str | None = None,
    ) -> None:
        existing = await self._graph.find_existing_relationship(
            source_id,
            target_id,
            predicate,
            group_id,
        )
        if existing is not None:
            return
        await self._graph.create_relationship(
            Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=source_id,
                target_id=target_id,
                predicate=predicate,
                group_id=group_id,
                source_episode=source_episode,
            )
        )

    def _vocabulary(self) -> dict[str, list[str]]:
        return getattr(self._cfg, "decision_vocabulary", None) or DEFAULT_DECISION_VOCABULARY

    @staticmethod
    def compose_decision_text(claim: EvidenceClaim) -> str:
        """Human-readable decision text — never '{subject}:{predicate}:{object}'."""
        subject = " ".join(claim.subject.split())
        detail = " ".join(str(claim.object).split())
        if claim.predicate != "decision_statement":
            predicate = claim.predicate
            if predicate.startswith("config:"):
                predicate = predicate.split(":", 1)[1]
            readable_predicate = predicate.replace("__", " ").replace("_", " ").strip()
            detail = f"{readable_predicate} = {detail}" if detail else readable_predicate
        return f"{subject}: {detail}" if detail else subject

    @staticmethod
    def _truncate_decision_name(text: str) -> str:
        if len(text) <= _DECISION_NAME_MAX_CHARS:
            return text
        cut = text[:_DECISION_NAME_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
        return f"{cut or text[:_DECISION_NAME_MAX_CHARS]}…"

    @staticmethod
    def is_decision_claim(
        claim: EvidenceClaim,
        vocabulary: dict[str, list[str]] | None = None,
    ) -> bool:
        vocab = DEFAULT_DECISION_VOCABULARY if vocabulary is None else vocabulary
        if claim.predicate in set(vocab.get("predicates", [])):
            return True
        return claim.predicate.startswith(tuple(vocab.get("predicate_prefixes", [])))

    @staticmethod
    def infer_decision_subject(
        content: str,
        vocabulary: dict[str, list[str]] | None = None,
    ) -> str | None:
        vocab = DEFAULT_DECISION_VOCABULARY if vocabulary is None else vocabulary
        lowered = content.lower()
        for key, terms in vocab.items():
            if not key.startswith(_SUBJECT_TERMS_PREFIX):
                continue
            if any(term in lowered for term in terms):
                return key.split(":", 1)[1]
        return None

    @staticmethod
    def merge_attributes(existing: dict | None, updates: dict) -> dict:
        merged = dict(existing or {})
        merged.update(updates)
        return merged
