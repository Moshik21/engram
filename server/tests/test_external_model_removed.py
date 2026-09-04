"""Anti-resurrection contract: Engram calls no external model in operation.

2026-09-04: the dead-by-config Anthropic branches were deleted -- HyDE, the
triage LLM judge and its key-gated escalation, infer Tier-3 validation and
Sonnet escalation, the merge LLM pass, the edge-adjudication server
adjudicator, and the Anthropic ``EntityExtractor`` body. Each was held off
only by a config default or an ``ANTHROPIC_API_KEY`` read at call time (the
silent-inert shape): a key left on disk could re-arm an external call with
no flag involved.

Modelled on test_ts_kill.py: the code is gone, the knobs are gone from the
schema, and a stale ``.env`` line is dropped with a warning instead of
crashing the shell (``ActivationConfig`` is ``extra='forbid'``).
"""

from __future__ import annotations

import importlib
import inspect
import logging

import pytest

from engram.config import ActivationConfig

RETIRED_KNOBS = (
    "hyde_enabled",
    "hyde_model",
    "triage_llm_judge_enabled",
    "triage_llm_judge_model",
    "triage_llm_escalation_enabled",
    "triage_llm_escalation_low",
    "triage_llm_escalation_high",
    "triage_llm_escalation_max_per_cycle",
    "consolidation_infer_llm_enabled",
    "consolidation_infer_llm_model",
    "consolidation_infer_escalation_enabled",
    "consolidation_infer_escalation_model",
    "consolidation_infer_escalation_max_per_cycle",
    "consolidation_merge_llm_enabled",
    "consolidation_merge_llm_model",
    "consolidation_merge_escalation_enabled",
    "consolidation_merge_escalation_model",
    "consolidation_merge_ann_llm_max",
    "edge_adjudication_server_enabled",
    "edge_adjudication_server_model",
    "edge_adjudication_server_max_per_cycle",
    "edge_adjudication_server_daily_budget",
    "edge_adjudication_server_min_age_minutes",
)

# Modules that used to hold a lazy ``import anthropic`` or a key read.
SOURCE_MODULES = (
    "engram.config",
    "engram.consolidation.phases.triage",
    "engram.consolidation.phases.infer",
    "engram.consolidation.phases.merge",
    "engram.consolidation.phases.edge_adjudication",
    "engram.extraction.extractor",
    "engram.extraction.prompts",
    "engram.retrieval.pipeline",
)


def test_hyde_module_is_gone():
    with pytest.raises(ImportError):
        importlib.import_module("engram.retrieval.hyde")


@pytest.mark.parametrize("module_name", SOURCE_MODULES)
def test_no_external_model_call_site(module_name):
    source = inspect.getsource(importlib.import_module(module_name))
    assert "import anthropic" not in source
    assert "ANTHROPIC_API_KEY" not in source
    assert "messages.create" not in source


def test_llm_branches_removed():
    from engram.consolidation.phases import triage
    from engram.consolidation.phases.edge_adjudication import EdgeAdjudicationPhase
    from engram.consolidation.phases.infer import EdgeInferencePhase
    from engram.consolidation.phases.merge import EntityMergePhase
    from engram.extraction import prompts
    from engram.extraction.extractor import EntityExtractor

    assert not hasattr(triage, "_llm_judge_score")
    assert not hasattr(EdgeInferencePhase, "_run_llm_validation_pass")
    assert not hasattr(EdgeInferencePhase, "_run_escalation_pass")
    assert not hasattr(EntityMergePhase, "_run_llm_merge_pass")
    assert not hasattr(EntityMergePhase, "_escalate_merge")
    assert not hasattr(EdgeAdjudicationPhase, "_call_server_adjudicator")
    assert not hasattr(EdgeAdjudicationPhase, "_resolve_with_server")
    assert not hasattr(EntityExtractor, "_get_client")
    assert not hasattr(prompts, "TRIAGE_JUDGE_SYSTEM_CACHED")
    assert not hasattr(prompts, "EXTRACTION_SYSTEM_CACHED")


def test_multi_signal_paths_keep_their_live_knobs():
    # Census section 4: decide the gate, never the parameters under it. These
    # are read by the live multi-signal / TTL paths and must survive the sweep.
    fields = ActivationConfig.model_fields
    assert "consolidation_infer_llm_confidence_threshold" in fields
    assert "consolidation_infer_llm_max_per_cycle" in fields
    assert "consolidation_merge_soft_threshold" in fields
    assert "edge_adjudication_request_ttl_hours" in fields


@pytest.mark.parametrize("knob", RETIRED_KNOBS)
def test_retired_knob_absent_from_schema(knob):
    assert knob not in ActivationConfig.model_fields


@pytest.mark.parametrize("knob", RETIRED_KNOBS)
def test_stale_env_line_is_dropped_with_warning(knob, caplog):
    """A .env still carrying the knob must warn, not crash the shell."""
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = ActivationConfig(**{knob: "1"})
    assert not hasattr(cfg, knob)
    assert any(
        knob.upper() in record.getMessage() and "retired" in record.getMessage()
        for record in caplog.records
    )
