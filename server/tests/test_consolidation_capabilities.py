from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.consolidation.capabilities import ConsolidationCapabilityValidator
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import PhaseResult


class CapabilityPhase(ConsolidationPhase):
    def __init__(
        self,
        *,
        graph_methods: set[str] | None = None,
        activation_methods: set[str] | None = None,
        search_methods: set[str] | None = None,
        enabled_by_config: bool = False,
    ) -> None:
        self._graph_methods = graph_methods or set()
        self._activation_methods = activation_methods or set()
        self._search_methods = search_methods or set()
        self._enabled_by_config = enabled_by_config

    @property
    def name(self) -> str:
        return "capability"

    def required_graph_store_methods(self, cfg: ActivationConfig) -> set[str]:
        if self._enabled_by_config and not cfg.consolidation_replay_enabled:
            return set()
        return self._graph_methods

    def required_activation_store_methods(self, cfg: ActivationConfig) -> set[str]:
        return self._activation_methods

    def required_search_index_methods(self, cfg: ActivationConfig) -> set[str]:
        return self._search_methods

    async def execute(self, *args, **kwargs):
        return PhaseResult(phase=self.name), []


def test_validator_accepts_available_capabilities():
    validator = ConsolidationCapabilityValidator(
        graph_store=SimpleNamespace(fetch_entities=object()),
        activation_store=SimpleNamespace(track_merge=object()),
        search_index=SimpleNamespace(remove_entity=object()),
    )

    validator.validate(
        (
            CapabilityPhase(
                graph_methods={"fetch_entities"},
                activation_methods={"track_merge"},
                search_methods={"remove_entity"},
            ),
        ),
        cfg=ActivationConfig(),
    )


def test_validator_reports_missing_capability_with_target_name():
    validator = ConsolidationCapabilityValidator(
        graph_store=SimpleNamespace(fetch_entities=object()),
        activation_store=SimpleNamespace(),
        search_index=SimpleNamespace(),
    )

    with pytest.raises(
        RuntimeError,
        match="Phase 'capability' requires graph_store methods: missing_graph_method",
    ):
        validator.validate(
            (CapabilityPhase(graph_methods={"fetch_entities", "missing_graph_method"}),),
            cfg=ActivationConfig(),
        )


def test_validator_uses_current_config_for_phase_requirements():
    validator = ConsolidationCapabilityValidator(
        graph_store=SimpleNamespace(),
        activation_store=SimpleNamespace(),
        search_index=SimpleNamespace(),
    )
    phase = CapabilityPhase(
        graph_methods={"replay_graph_changes"},
        enabled_by_config=True,
    )

    validator.validate(
        (phase,),
        cfg=ActivationConfig(consolidation_replay_enabled=False),
    )

    with pytest.raises(RuntimeError, match="replay_graph_changes"):
        validator.validate(
            (phase,),
            cfg=ActivationConfig(consolidation_replay_enabled=True),
        )
