"""Tests for MCP prompt guidance."""

from engram.mcp import server as mcp_server
from engram.mcp.prompts import ENGRAM_CONTEXT_LOADER_PROMPT, ENGRAM_SYSTEM_PROMPT


def test_system_prompt_requires_lookup_before_answer():
    assert "BEFORE answering" in ENGRAM_SYSTEM_PROMPT
    assert "observe(user_message)" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_names_brain_loop_contract():
    assert "Capture -> Cue -> Project -> Recall -> Consolidate" in ENGRAM_SYSTEM_PROMPT
    assert "cueable latent memory" in ENGRAM_SYSTEM_PROMPT
    assert "durable graph" in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_observe_guidance():
    assert "new information worth" in ENGRAM_SYSTEM_PROMPT
    assert "returns recalled" in ENGRAM_SYSTEM_PROMPT


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


def test_system_prompt_recalled_context_guidance():
    assert "recalled_context" in ENGRAM_SYSTEM_PROMPT
    assert "freshness" in ENGRAM_SYSTEM_PROMPT


def test_context_loader_prompt_remains_explicit():
    assert ENGRAM_CONTEXT_LOADER_PROMPT.startswith("Before responding, call get_context")


def test_mcp_system_prompt_function_returns_brain_loop_guidance():
    assert mcp_server.engram_system() == ENGRAM_SYSTEM_PROMPT
    assert "Capture -> Cue -> Project -> Recall -> Consolidate" in mcp_server.engram_system()


def test_mcp_context_loader_prompt_function_adds_topic_hint():
    prompt = mcp_server.engram_context_loader(topic="Engram native mode")

    assert prompt.startswith(ENGRAM_CONTEXT_LOADER_PROMPT)
    assert 'topic_hint="Engram native mode"' in prompt
