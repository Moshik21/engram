"""Adapter wrapping NarrowExtractionPipeline behind the EntityExtractor interface."""

from __future__ import annotations

import logging

from engram.config import ActivationConfig
from engram.extraction.commit_policy import AdaptiveCommitPolicy, CommitThresholds
from engram.extraction.evidence_bridge import EvidenceBridge
from engram.extraction.extractor import ExtractionResult, ExtractionStatus
from engram.extraction.narrow.pipeline import NarrowExtractionPipeline

logger = logging.getLogger(__name__)


class NarrowExtractorAdapter:
    """Zero-dependency extraction using deterministic narrow extractors.

    Wraps NarrowExtractionPipeline + AdaptiveCommitPolicy + EvidenceBridge
    behind the same ``async extract(text) -> ExtractionResult`` interface
    that EntityExtractor uses.
    """

    def __init__(self, cfg: ActivationConfig | None = None) -> None:
        self._cfg = cfg or ActivationConfig()
        self._pipeline = NarrowExtractionPipeline(self._cfg)
        self._commit_policy = AdaptiveCommitPolicy(
            CommitThresholds(
                entity=self._cfg.evidence_commit_entity_threshold,
                relationship=self._cfg.evidence_commit_relationship_threshold,
                attribute=self._cfg.evidence_commit_attribute_threshold,
                temporal=self._cfg.evidence_commit_temporal_threshold,
            ),
            adaptive=self._cfg.evidence_adaptive_thresholds,
        )
        self._bridge = EvidenceBridge()

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships using narrow pipeline."""
        if not text or not text.strip():
            return ExtractionResult(
                entities=[], relationships=[], status=ExtractionStatus.EMPTY,
            )

        try:
            # 1. Run narrow extractors
            bundle = self._pipeline.extract(
                text=text, episode_id="adapter", group_id="default",
            )

            # 2. Evaluate commit decisions
            decisions = self._commit_policy.evaluate(bundle)

            # 3. Filter to committed candidates
            committed = []
            for candidate, decision in zip(bundle.candidates, decisions):
                if decision.action == "commit":
                    committed.append((candidate, decision))

            # 4. Bridge to EntityCandidate/ClaimCandidate
            entity_candidates, claim_candidates = self._bridge.bridge(committed)

            # 5. Convert to ExtractionResult dict format
            entities = []
            for ec in entity_candidates:
                entities.append({
                    "name": ec.name,
                    "entity_type": ec.entity_type,
                    "summary": ec.summary or "",
                    **({"attributes": ec.attributes} if ec.attributes else {}),
                })

            relationships = []
            for cc in claim_candidates:
                relationships.append({
                    "source": cc.subject_text,
                    "target": cc.object_text or "",
                    "predicate": cc.predicate,
                })

            logger.info(
                "Narrow extraction: %d entities, %d relationships from %d chars (%.1fms)",
                len(entities), len(relationships), len(text), bundle.total_ms,
            )

            status = (
                ExtractionStatus.OK if entities or relationships
                else ExtractionStatus.EMPTY
            )
            return ExtractionResult(
                entities=entities, relationships=relationships, status=status,
            )
        except Exception as e:
            logger.error("Narrow extraction failed: %s", e)
            return ExtractionResult(
                entities=[], relationships=[],
                status=ExtractionStatus.API_ERROR, error=str(e),
            )
