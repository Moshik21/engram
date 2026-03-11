"""Base protocol for narrow extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from engram.config import ActivationConfig
    from engram.extraction.evidence import EvidenceCandidate
    from engram.models.episode_cue import EpisodeCue


@runtime_checkable
class NarrowExtractor(Protocol):
    """Protocol for deterministic narrow extractors."""

    name: str

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]: ...
