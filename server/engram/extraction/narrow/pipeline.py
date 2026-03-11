"""Pipeline orchestrator for narrow extractors."""

from __future__ import annotations

import time
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceBundle, EvidenceCandidate
from engram.extraction.narrow.attribute_extractor import (
    AttributeEvidenceExtractor,
)
from engram.extraction.narrow.entity_extractor import (
    IdentityEntityExtractor,
)
from engram.extraction.narrow.relationship_extractor import (
    RelationshipPatternExtractor,
)
from engram.extraction.narrow.temporal_extractor import (
    TemporalEvidenceExtractor,
)
from engram.models.episode_cue import EpisodeCue


class NarrowExtractionPipeline:
    """Runs all narrow extractors and produces an EvidenceBundle."""

    def __init__(self, cfg: ActivationConfig | None = None) -> None:
        self._cfg = cfg or ActivationConfig()
        self._extractors: list[Any] = [
            IdentityEntityExtractor(),
            RelationshipPatternExtractor(),
            TemporalEvidenceExtractor(),
            AttributeEvidenceExtractor(),
        ]

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str = "default",
        cue: EpisodeCue | None = None,
    ) -> EvidenceBundle:
        """Run all extractors, cross-corroborate, deduplicate, return bundle."""
        all_candidates: list[EvidenceCandidate] = []
        stats: dict[str, dict] = {}
        total_start = time.monotonic()

        for extractor in self._extractors:
            ext_start = time.monotonic()
            candidates = extractor.extract(
                text=text,
                episode_id=episode_id,
                group_id=group_id,
                cue=cue,
                cfg=self._cfg,
            )
            ext_ms = (time.monotonic() - ext_start) * 1000
            stats[extractor.name] = {
                "count": len(candidates),
                "duration_ms": round(ext_ms, 2),
            }
            all_candidates.extend(candidates)

        # Cross-extractor corroboration
        all_candidates = self._cross_corroborate(all_candidates)

        # Deduplication
        all_candidates = self._deduplicate(all_candidates)

        total_ms = (time.monotonic() - total_start) * 1000
        return EvidenceBundle(
            episode_id=episode_id,
            group_id=group_id,
            candidates=all_candidates,
            extractor_stats=stats,
            total_ms=round(total_ms, 2),
        )

    def _cross_corroborate(
        self, candidates: list[EvidenceCandidate],
    ) -> list[EvidenceCandidate]:
        """Boost confidence when entity is mentioned by multiple extractors."""
        # Build entity name -> extractors map
        entity_names: dict[str, set[str]] = {}
        for c in candidates:
            if c.fact_class == "entity":
                name = c.payload.get("name", "").lower()
                if name:
                    entity_names.setdefault(name, set()).add(
                        c.extractor_name,
                    )
            elif c.fact_class == "relationship":
                for field in ("subject", "object"):
                    name = c.payload.get(field, "").lower()
                    if name and name != "user":
                        entity_names.setdefault(name, set()).add(
                            c.extractor_name,
                        )

        # Boost entity candidates mentioned by relationship extractor too
        multi_source = {
            name for name, exts in entity_names.items() if len(exts) > 1
        }
        for c in candidates:
            if c.fact_class == "entity":
                name = c.payload.get("name", "").lower()
                if (
                    name in multi_source
                    and "cross_extractor_corroboration"
                    not in c.corroborating_signals
                ):
                    c.confidence = min(1.0, c.confidence + 0.10)
                    c.corroborating_signals.append(
                        "cross_extractor_corroboration",
                    )

        return candidates

    def _deduplicate(
        self, candidates: list[EvidenceCandidate],
    ) -> list[EvidenceCandidate]:
        """Merge duplicate entity candidates, keeping highest confidence."""
        entity_map: dict[str, EvidenceCandidate] = {}
        non_entity: list[EvidenceCandidate] = []

        for c in candidates:
            if c.fact_class == "entity":
                key = (
                    c.payload.get("name", "").lower()
                    + ":"
                    + c.payload.get("entity_type", "").lower()
                )
                existing = entity_map.get(key)
                if existing is None:
                    entity_map[key] = c
                elif c.confidence > existing.confidence:
                    # Merge signals from both
                    c.corroborating_signals = list(
                        set(c.corroborating_signals)
                        | set(existing.corroborating_signals)
                    )
                    entity_map[key] = c
                else:
                    existing.corroborating_signals = list(
                        set(existing.corroborating_signals)
                        | set(c.corroborating_signals)
                    )
            else:
                non_entity.append(c)

        return list(entity_map.values()) + non_entity
