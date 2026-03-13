"""Tests for MCP prompt guidance."""

from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT


def test_system_prompt_requires_lookup_before_answer():
    assert "BEFORE answering" in ENGRAM_SYSTEM_PROMPT
    assert "observe(user_message)" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_mandatory_observe_every_turn():
    assert "on EVERY turn" in ENGRAM_SYSTEM_PROMPT
    assert "protocol violation" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_failure_mode_example():
    # Single vivid failure example replaces the 12 trigger examples
    assert "Liam plays soccer" in ENGRAM_SYSTEM_PROMPT
    assert "FAILURE MODE" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_covers_epistemic_routing_tools():
    for phrase in (
        "route_question",
        "answerContract",
        "search_artifacts",
        "get_runtime_state",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_covers_answer_contract_guidance():
    for phrase in (
        "Carry the same `project_path`",
        "`compare`",
        "`recommend` / `plan`",
        "`evidencePlan.requiredNextSources`",
        "Do not substitute `search_facts` for required artifact inspection",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_recalled_context_mandatory():
    assert "MUST" in ENGRAM_SYSTEM_PROMPT
    assert "recalled_context" in ENGRAM_SYSTEM_PROMPT


def test_context_loader_prompt_remains_explicit():
    assert ENGRAM_CONTEXT_LOADER_PROMPT.startswith("Before responding, call get_context")
