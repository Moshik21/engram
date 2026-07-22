"""M5.1 battery runner tests: seeded CI mode + machinery-exclusion (gate B4)."""

from __future__ import annotations

import pytest

from engram.evaluation.battery import (
    BATTERY_PATH,
    format_battery_report,
    group_contained,
    is_machinery_text,
    load_battery,
    run_battery_seeded,
    score_question,
    top3_result_texts,
)


def test_battery_rig_loads_and_has_ten_questions() -> None:
    battery = load_battery()
    assert BATTERY_PATH.exists()
    questions = battery["questions"]
    assert len(questions) == 10
    for q in questions:
        assert q["id"] and q["q"]
        assert q["expected_tokens"], q["id"]


def test_group_containment_is_case_insensitive() -> None:
    assert group_contained("The ORGANIC yield arrived", ["organic", "yield"])
    assert not group_contained("organic only", ["organic", "yield"])


def test_top3_boundary_is_enforced() -> None:
    payload = {"items": [{"name": f"r{i}"} for i in range(5)]}
    texts = top3_result_texts(payload)
    assert texts == ["r0", "r1", "r2"]


def test_score_question_hit_and_miss() -> None:
    question = {"id": "x", "q": "q", "expected_tokens": [["alpha", "beta"], ["gamma"]]}
    # Per-result containment: a group split ACROSS results must NOT hit ...
    split = score_question(question, ["something Alpha", "and BETA too"])
    assert not split["hit"]
    # ... while a group wholly inside one result does.
    hit = score_question(question, ["something Alpha and BETA too", "unrelated"])
    assert hit["hit"] and hit["hit_group"] == ["alpha", "beta"]
    miss = score_question(question, ["nothing relevant"])
    assert not miss["hit"]


def test_machinery_predicate_flags_hook_noise_not_content() -> None:
    # Gate B4 signatures (defects 3: task-notification/command-output episodes).
    assert is_machinery_text("<task-notification>agent done</task-notification>")
    # A stripped tool-output dump (opaque ids, no natural language) is
    # machinery; NOTE prose merely mentioning "exit code 1" is deliberately
    # NOT flagged by the salience classifier (false-positive guard).
    assert is_machinery_text("bhdxqbaan\ntoolu_016V5AUszMcGhdYupaRXC2NY completed exit code 0")
    assert is_machinery_text("tool call toolu_01AbCdEf123 returned")
    # Genuine content must not be flagged.
    assert not is_machinery_text("The FastEmbed outage root cause was an interrupted download")
    assert not is_machinery_text("Konner decided Engram must run fully local")


def test_score_question_reports_machinery_in_top3() -> None:
    question = {"id": "x", "q": "q", "expected_tokens": [["alpha"]]}
    row = score_question(question, ["alpha", "<task-notification>done</task-notification>"])
    assert row["machinery_top3_indexes"] == [1]


@pytest.mark.asyncio
async def test_seeded_battery_ci_mode_scores_full_and_machinery_clean() -> None:
    result = await run_battery_seeded()
    total = result["total"]
    assert total + len(result["skipped_live_only"]) == 10
    # CI corpus is planted to be answerable: every runnable question must hit.
    misses = [r["id"] for r in result["questions"] if not r["hit"]]
    assert result["score"] == total, f"seeded misses: {misses}"
    # Gate B4: no machinery-class text in any top-3.
    assert result["machinery_clean"], result
    report = format_battery_report(result, floor=total)
    assert f"{total}/{total}" in report
    assert "PASS" in report
