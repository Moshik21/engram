"""Tests for MCP prompt guidance."""

from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT


def test_system_prompt_requires_lookup_before_answer():
    assert "before you answer" in ENGRAM_SYSTEM_PROMPT
    assert "Do not answer first and only then call `observe` or `remember`" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_covers_natural_followups():
    for phrase in (
        "my son did great today",
        "talked to Sarah about it",
        "still dealing with that bug",
        "which son plays soccer?",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_covers_epistemic_routing_tools():
    for phrase in (
        "route_question",
        "answerContract",
        "search_artifacts",
        "get_runtime_state",
        "what did we decide about launching Engram publicly?",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_covers_answer_contract_guidance():
    for phrase in (
        "carry the same `project_path`",
        "If `answerContract.operator` is `compare`",
        "If `answerContract.operator` is `recommend` or `plan`",
        "`evidencePlan.requiredNextSources`",
        "Do not substitute `search_facts` for required artifact inspection",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_context_loader_prompt_remains_explicit():
    assert ENGRAM_CONTEXT_LOADER_PROMPT.startswith("Before responding, call get_context")
