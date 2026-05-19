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


def test_system_prompt_claims_cross_context_authority():
    for phrase in (
        "Engram is the source of truth for portable, cross-context memory",
        "Do not skip Engram just because another file-based",
        "Engram owns: cross-project user facts",
        "Project-local files own: repo-specific coding conventions",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_treats_empty_runtime_as_onboarding_state():
    for phrase in (
        "`artifactCount` is 0",
        "`lastObservedAt` is null",
        "call `bootstrap_project(project_path)` once",
        "call `claim_authority(project_path, user_message",
        "`agent_protocol`",
        "follow `required_tools_before_answer` in order",
        "use its `capture` decision",
        "A fresh graph is an onboarding state",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


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
        "claim_authority",
        "answerContract",
        "search_artifacts",
        "get_runtime_state",
        "beforeAnswer",
    ):
        assert phrase in ENGRAM_SYSTEM_PROMPT


def test_system_prompt_bootstrap_covers_user_approved_sources():
    assert "project docs, notes, and memory exports" in ENGRAM_SYSTEM_PROMPT
    assert "explicit user-approved source globs" in ENGRAM_SYSTEM_PROMPT


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
