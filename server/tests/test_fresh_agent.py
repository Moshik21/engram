"""M5.2 fresh-agent harness logic — lite CI mode (no server, injected tools)."""

from __future__ import annotations

from typing import Any

import pytest

from engram.evaluation.fresh_agent import (
    format_fresh_agent_report,
    load_project_file_texts,
    reformulate_question,
    run_fresh_agent,
)

Q_FLIP = {
    "id": "flip-condition",
    "q": "what is the flip condition for usage ranking",
    "expected_tokens": [["organic", "yield"]],
}
Q_TS = {
    "id": "ts-kill",
    "q": "why was Thompson sampling removed",
    "expected_tokens": [["Thompson", "noise"]],
}


def _packets(text: str = "") -> list[dict[str, Any]]:
    return [{"path": "fake.md", "text": text, "chars": len(text)}]


def test_reformulation_drops_stopwords_keeps_order():
    assert (
        reformulate_question("what is the flip condition for usage ranking")
        == "flip condition usage ranking"
    )
    assert reformulate_question("why was Thompson sampling removed") == (
        "Thompson sampling removed"
    )


def test_hit_on_first_recall_makes_one_call():
    calls: list[str] = []

    def recall_fn(q: str) -> dict[str, Any]:
        calls.append(q)
        return {"items": [{"text": "flip when organic yield > 0"}]}

    result = run_fresh_agent(
        [Q_FLIP],
        recall_fn=recall_fn,
        context_fn=lambda: "",
        project_packets=_packets(),
    )
    row = result["questions"][0]
    assert row["hit"] and row["tool_calls"] == 1
    assert row["reformulated_query"] is None
    assert calls == ["what is the flip condition for usage ranking"]
    assert result["engram_score"] == 1
    assert result["lift"] == 1


def test_miss_triggers_one_deterministic_reformulation():
    calls: list[str] = []

    def recall_fn(q: str) -> dict[str, Any]:
        calls.append(q)
        if q == "Thompson sampling removed":
            return {"items": [{"text": "Thompson sampling was pure noise"}]}
        return {"items": [{"text": "unrelated"}]}

    result = run_fresh_agent(
        [Q_TS],
        recall_fn=recall_fn,
        context_fn=lambda: "",
        project_packets=_packets(),
    )
    row = result["questions"][0]
    assert row["hit"] and row["tool_calls"] == 2
    assert row["reformulated_query"] == "Thompson sampling removed"
    assert calls == ["why was Thompson sampling removed", "Thompson sampling removed"]


def test_identical_reformulation_is_not_retried():
    question = {"id": "x", "q": "flip condition", "expected_tokens": [["nope"]]}
    calls: list[str] = []

    def recall_fn(q: str) -> dict[str, Any]:
        calls.append(q)
        return {"items": []}

    result = run_fresh_agent(
        [question],
        recall_fn=recall_fn,
        context_fn=lambda: "",
        project_packets=_packets(),
    )
    assert result["questions"][0]["tool_calls"] == 1
    assert calls == ["flip condition"]


def test_projectfile_arm_scores_against_file_text_with_zero_calls():
    result = run_fresh_agent(
        [Q_FLIP],
        recall_fn=lambda q: {"items": []},
        context_fn=lambda: "",
        project_packets=_packets("the flip waits on organic yield"),
    )
    row = result["questions"][0]
    assert row["projectfile_hit"] and not row["hit"]
    assert result["projectfile_score"] == 1
    assert result["lift"] == -1


def test_token_cost_and_session_context_counted():
    result = run_fresh_agent(
        [Q_FLIP],
        recall_fn=lambda q: {"items": [{"text": "organic yield"}]},
        context_fn=lambda: "ctx" * 10,
        project_packets=_packets(),
    )
    assert result["session_context_chars"] == 30
    assert result["tool_calls_total"] == 2  # get_context + 1 recall
    assert result["chars_surfaced_total"] == 30 + result["questions"][0]["chars_surfaced"]


def test_judge_stub_raises():
    with pytest.raises(NotImplementedError):
        run_fresh_agent(
            [Q_FLIP],
            recall_fn=lambda q: {"items": []},
            context_fn=lambda: "",
            project_packets=_packets(),
            judge="ollama",
        )


def test_recall_errors_surface_in_report():
    result = run_fresh_agent(
        [Q_TS],
        recall_fn=lambda q: {"error": "boom", "items": []},
        context_fn=lambda: "",
        project_packets=_packets(),
    )
    assert result["questions"][0]["recall_errors"]
    assert "recall errors on: ts-kill" in format_fresh_agent_report(result)


def test_missing_project_file_reported_not_swallowed(tmp_path):
    packets = load_project_file_texts([tmp_path / "absent.md"])
    assert packets[0]["error"] and packets[0]["chars"] == 0
