"""Neuroplasticity implementation for Project Synapse."""

from __future__ import annotations

import logging
from typing import Any

from engram.config import ActivationConfig

logger = logging.getLogger(__name__)


class NeuroplasticityEngine:
    """Meta-learning loop for self-tuning activation parameters.

    Adjusts ACT-R decay and sensitivity parameters based on Retrieval ROI.
    """

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg
        # Step size for parameter adjustments
        self._step_size = 0.01

    def handle_positive_feedback(
        self,
        entity_id: str,
        activation_state: Any = None,
    ) -> None:
        """Adjust parameters when a memory is successfully retrieved and used."""
        if not self._cfg.neuroplasticity_enabled:
            return

        # Heuristic: If we get positive feedback, our decay might be too aggressive,
        # or our spreading activation is working well.
        # We don't directly mutate the global config (which might be shared/frozen),
        # but we emit tuning signals that a higher-level manager could use.
        logger.info(
            "Neuroplasticity: Positive feedback for %s. "
            "Recommendation: decrease decay_exponent by %f",
            entity_id,
            self._step_size,
        )

    def handle_negative_feedback(
        self,
        entity_id: str,
        activation_state: Any = None,
    ) -> None:
        """Adjust parameters when a memory is retrieved but ignored or rejected."""
        if not self._cfg.neuroplasticity_enabled:
            return

        # Heuristic: If we get negative feedback, our decay might be too slow
        # (keeping irrelevant stuff alive), or spreading is too noisy.
        logger.info(
            "Neuroplasticity: Negative feedback for %s. "
            "Recommendation: increase decay_exponent by %f",
            entity_id,
            self._step_size,
        )
