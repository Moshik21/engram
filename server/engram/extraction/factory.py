"""Factory for the internal extraction rung.

2026-09-04: Engram never calls an external model. The resident agent holding
the MCP is the extractor (``proposed_entities`` / ``proposed_relationships``);
the only internal rung is the deterministic narrow adapter, which writes cues
for content the agent did not structure. The anthropic/ollama rungs and the
``auto`` ladder were removed; ``ActivationConfig`` maps their leftovers to
narrow with a warning.
"""

from __future__ import annotations

import logging

from engram.config import ActivationConfig, EngramConfig

logger = logging.getLogger(__name__)


def create_extractor(config: EngramConfig):
    """Return the narrow adapter -- the only internal extractor that exists."""
    return _make_narrow(config.activation)


def _make_narrow(cfg: ActivationConfig):
    """Create the deterministic narrow extraction adapter."""
    from engram.extraction.narrow_adapter import NarrowExtractorAdapter

    logger.info("Extraction provider: Narrow (deterministic; the resident agent proposes)")
    return NarrowExtractorAdapter(cfg)
